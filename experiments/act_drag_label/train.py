"""Behavior cloning training loop for the ACT policy.

Loads HDF5 expert demonstrations, trains with:
- Masked L1 loss on dx, dy (continuous pixel deltas)
- Masked BCE loss on click (binary)
- Masked CE loss on key (28-class classification)
- Unmasked BCE loss on pad prediction
- KL divergence from CVAE encoder (annealed)
Saves checkpoints + training history.
"""

import argparse
import collections
import glob
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.act_drag_label.config import ACTION, CHUNK, TRAIN
from experiments.act_drag_label.model import ACT, count_parameters


def _auto_cache_size(num_workers: int = 0) -> int:
    """Pick LRU cache size based on system RAM. ~5MB per cached episode."""
    try:
        total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        total_gb = total_bytes / (1024 ** 3)
    except (ValueError, OSError):
        total_gb = 8  # conservative fallback

    if total_gb > 24:
        limit = 999_999  # effectively unlimited
    elif total_gb > 12:
        limit = 2000
    else:
        limit = 100  # 8GB M1 range

    # Each DataLoader worker gets its own cache via fork
    effective_processes = max(1, num_workers + 1)
    return max(10, limit // effective_processes)


class ChunkDataset(Dataset):
    """Loads HDF5 episodes and samples (obs, proprio, action_chunk, ...) tuples.

    Every timestep t in every episode is a valid start index.
    Action chunks are zero-padded when near the episode end.
    """

    def __init__(self, episode_files: list[str], chunk_size: int,
                 cache_max: int | None = None) -> None:
        self.chunk_size = chunk_size
        self._cache_max = cache_max if cache_max is not None else _auto_cache_size()
        self._cache: collections.OrderedDict[str, dict[str, np.ndarray]] = (
            collections.OrderedDict()
        )
        self.index: list[tuple[str, int]] = []

        for path in episode_files:
            with h5py.File(path, "r") as f:
                n_steps = len(f["actions_dx"])
            # Every timestep is a valid start
            for t in range(n_steps):
                self.index.append((path, t))

    def _load_episode(self, path: str) -> dict[str, np.ndarray]:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]

        with h5py.File(path, "r") as f:
            data = {
                "observations": f["observations"][:],
                "actions_dx": f["actions_dx"][:],
                "actions_dy": f["actions_dy"][:],
                "actions_click": f["actions_click"][:],
                "actions_key": f["actions_key"][:],
            }

        self._cache[path] = data
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)  # evict oldest

        return data

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        torch.Tensor,  # obs (3, 224, 224)
        torch.Tensor,  # proprio (31,)
        torch.Tensor,  # actions_cvae (chunk, 31) for CVAE encoder
        torch.Tensor,  # dx_chunk (chunk,)
        torch.Tensor,  # dy_chunk (chunk,)
        torch.Tensor,  # click_chunk (chunk,)
        torch.Tensor,  # key_chunk (chunk,)
        torch.Tensor,  # pad_mask (chunk,) — 1 where padded, 0 where real
    ]:
        path, t = self.index[idx]
        ep = self._load_episode(path)
        T = len(ep["actions_dx"])
        C = self.chunk_size

        # --- Observation at time t ---
        obs = torch.from_numpy(ep["observations"][t]).permute(2, 0, 1).float() / 255.0

        # --- Proprioception at time t ---
        # Placeholder mouse position (0.5, 0.5), plus click and key from actions
        click_t = float(ep["actions_click"][t])
        key_t = int(ep["actions_key"][t])
        key_onehot = np.zeros(ACTION.num_key_classes, dtype=np.float32)
        key_onehot[key_t] = 1.0
        proprio = np.zeros(31, dtype=np.float32)
        proprio[0] = 0.5  # mouse_x_norm placeholder
        proprio[1] = 0.5  # mouse_y_norm placeholder
        proprio[2] = click_t
        proprio[3:] = key_onehot
        proprio = torch.from_numpy(proprio)

        # --- Action chunk [t:t+C], zero-padded if near end ---
        end = min(t + C, T)
        real_len = end - t

        dx_chunk = np.zeros(C, dtype=np.float32)
        dy_chunk = np.zeros(C, dtype=np.float32)
        click_chunk = np.zeros(C, dtype=np.float32)
        key_chunk = np.zeros(C, dtype=np.int64)
        pad_mask = np.ones(C, dtype=np.float32)  # 1=padded, 0=real

        dx_chunk[:real_len] = ep["actions_dx"][t:end]
        dy_chunk[:real_len] = ep["actions_dy"][t:end]
        click_chunk[:real_len] = ep["actions_click"][t:end].astype(np.float32)
        key_chunk[:real_len] = ep["actions_key"][t:end].astype(np.int64)
        pad_mask[:real_len] = 0.0

        dx_chunk = torch.from_numpy(dx_chunk)
        dy_chunk = torch.from_numpy(dy_chunk)
        click_chunk = torch.from_numpy(click_chunk)
        key_chunk = torch.from_numpy(key_chunk)
        pad_mask = torch.from_numpy(pad_mask)

        # --- Actions for CVAE encoder: (chunk, action_dim) ---
        # action_dim = 2 (dx,dy) + 1 (click) + 28 (key onehot) = 31
        actions_cvae = torch.zeros(C, 2 + 1 + ACTION.num_key_classes)
        actions_cvae[:real_len, 0] = dx_chunk[:real_len]
        actions_cvae[:real_len, 1] = dy_chunk[:real_len]
        actions_cvae[:real_len, 2] = click_chunk[:real_len]
        for i in range(real_len):
            k = key_chunk[i].item()
            actions_cvae[i, 3 + k] = 1.0

        return obs, proprio, actions_cvae, dx_chunk, dy_chunk, click_chunk, key_chunk, pad_mask


def train(
    backbone: str = "resnet18",
    chunk_size: int = CHUNK.default_chunk_size,
    batch_size: int = TRAIN.batch_size,
    max_episodes: int | None = None,
    num_workers: int | None = None,
    cache_episodes: int | None = None,
    data_dir: str | None = None,
    checkpoint_dir: str | None = None,
    device: str = "cpu",
) -> None:
    base = os.path.dirname(__file__)
    if data_dir is None:
        data_dir = os.path.join(base, "data")
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(
            base, "checkpoints", f"{backbone}_chunk{chunk_size}"
        )
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Find episodes
    episode_files = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
    if not episode_files:
        print(f"No episodes found in {data_dir}. Run generate_data.py first.")
        return
    if max_episodes is not None and max_episodes < len(episode_files):
        episode_files = episode_files[:max_episodes]
    print(f"Using {len(episode_files)} episodes")

    # Train/val split by episode
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(episode_files))
    val_count = max(1, int(len(episode_files) * TRAIN.val_fraction))
    val_files = [episode_files[i] for i in indices[:val_count]]
    train_files = [episode_files[i] for i in indices[val_count:]]

    print(f"Train: {len(train_files)} episodes, Val: {len(val_files)} episodes")

    # Default num_workers: 4 for CUDA (overlap data loading with GPU), 0 for MPS/CPU
    if num_workers is None:
        num_workers = 4 if device.startswith("cuda") else 0

    cache_max = cache_episodes if cache_episodes is not None else _auto_cache_size(num_workers)
    print(f"DataLoader workers: {num_workers}, Episode cache: {cache_max}")

    train_dataset = ChunkDataset(train_files, chunk_size, cache_max=cache_max)
    val_dataset = ChunkDataset(val_files, chunk_size, cache_max=cache_max)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=num_workers > 0,
    )

    # Model
    model = ACT(backbone_name=backbone, chunk_size=chunk_size).to(device)
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Optimizer with two param groups
    backbone_params = []
    non_backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen ViT params
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            non_backbone_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": non_backbone_params, "lr": TRAIN.lr},
            {"params": backbone_params, "lr": TRAIN.backbone_lr},
        ],
        weight_decay=TRAIN.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAIN.epochs
    )

    # AMP — GradScaler only works on CUDA, not MPS
    use_amp = TRAIN.use_amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)
    amp_dtype = torch.float16 if use_amp else torch.float32

    print(f"AMP: {'enabled' if use_amp else 'disabled (GradScaler requires CUDA)'}")
    print(f"Device: {device}, Backbone: {backbone}, Chunk size: {chunk_size}")

    # Training loop
    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
    }
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    kl_anneal_epochs = int(TRAIN.epochs * TRAIN.kl_anneal_fraction)

    for epoch in range(TRAIN.epochs):
        t0 = time.perf_counter()

        # KL weight annealing: 0 -> kl_weight_max over first 20% of epochs
        if kl_anneal_epochs > 0:
            kl_weight = min(1.0, epoch / kl_anneal_epochs) * TRAIN.kl_weight_max
        else:
            kl_weight = TRAIN.kl_weight_max

        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_total = 0

        for batch in train_loader:
            (obs, proprio, actions_cvae, dx_gt, dy_gt, click_gt, key_gt, pad_gt) = batch
            obs = obs.to(device)
            proprio = proprio.to(device)
            actions_cvae = actions_cvae.to(device)
            dx_gt = dx_gt.to(device)
            dy_gt = dy_gt.to(device)
            click_gt = click_gt.to(device)
            key_gt = key_gt.to(device)
            pad_gt = pad_gt.to(device)

            with torch.amp.autocast(
                device_type=device.split(":")[0], dtype=amp_dtype, enabled=use_amp
            ):
                out = model(obs, proprio, actions_cvae)

                # Mask: 1 where real, 0 where padded
                mask = 1.0 - pad_gt  # (B, chunk)
                mask_sum = mask.sum().clamp(min=1)

                # Masked L1 losses for dx, dy
                loss_dx = (
                    F.l1_loss(out["dx"], dx_gt, reduction="none") * mask
                ).sum() / mask_sum
                loss_dy = (
                    F.l1_loss(out["dy"], dy_gt, reduction="none") * mask
                ).sum() / mask_sum

                # Masked BCE for click
                loss_click = (
                    F.binary_cross_entropy_with_logits(
                        out["click"], click_gt, reduction="none"
                    )
                    * mask
                ).sum() / mask_sum

                # Masked CE for key — reshape for cross_entropy
                B, C_chunk = key_gt.shape
                key_logits_flat = out["key_logits"].reshape(B * C_chunk, -1)
                key_gt_flat = key_gt.reshape(B * C_chunk)
                mask_flat = mask.reshape(B * C_chunk)
                loss_key_per = F.cross_entropy(
                    key_logits_flat, key_gt_flat, reduction="none"
                )
                loss_key = (loss_key_per * mask_flat).sum() / mask_flat.sum().clamp(
                    min=1
                )

                # Unmasked BCE for pad prediction
                loss_pad = F.binary_cross_entropy_with_logits(
                    out["pad_logits"], pad_gt
                )

                # KL divergence
                mu = out["mu"]
                logvar = out["logvar"]
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

                total_loss = (
                    loss_dx
                    + loss_dy
                    + TRAIN.loss_weight_click * loss_click
                    + TRAIN.loss_weight_key * loss_key
                    + TRAIN.loss_weight_pad * loss_pad
                    + kl_weight * kl
                )

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = obs.size(0)
            train_loss_sum += total_loss.item() * bs
            train_total += bs

        scheduler.step()
        train_loss = train_loss_sum / max(train_total, 1)

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                (obs, proprio, actions_cvae, dx_gt, dy_gt, click_gt, key_gt, pad_gt) = (
                    batch
                )
                obs = obs.to(device)
                proprio = proprio.to(device)
                actions_cvae = actions_cvae.to(device)
                dx_gt = dx_gt.to(device)
                dy_gt = dy_gt.to(device)
                click_gt = click_gt.to(device)
                key_gt = key_gt.to(device)
                pad_gt = pad_gt.to(device)

                out = model(obs, proprio, actions_cvae)

                mask = 1.0 - pad_gt
                mask_sum = mask.sum().clamp(min=1)

                loss_dx = (
                    F.l1_loss(out["dx"], dx_gt, reduction="none") * mask
                ).sum() / mask_sum
                loss_dy = (
                    F.l1_loss(out["dy"], dy_gt, reduction="none") * mask
                ).sum() / mask_sum

                loss_click = (
                    F.binary_cross_entropy_with_logits(
                        out["click"], click_gt, reduction="none"
                    )
                    * mask
                ).sum() / mask_sum

                B, C_chunk = key_gt.shape
                key_logits_flat = out["key_logits"].reshape(B * C_chunk, -1)
                key_gt_flat = key_gt.reshape(B * C_chunk)
                mask_flat = mask.reshape(B * C_chunk)
                loss_key_per = F.cross_entropy(
                    key_logits_flat, key_gt_flat, reduction="none"
                )
                loss_key = (loss_key_per * mask_flat).sum() / mask_flat.sum().clamp(
                    min=1
                )

                loss_pad = F.binary_cross_entropy_with_logits(
                    out["pad_logits"], pad_gt
                )

                mu = out["mu"]
                logvar = out["logvar"]
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

                total_loss = (
                    loss_dx
                    + loss_dy
                    + TRAIN.loss_weight_click * loss_click
                    + TRAIN.loss_weight_key * loss_key
                    + TRAIN.loss_weight_pad * loss_pad
                    + kl_weight * kl
                )

                bs = obs.size(0)
                val_loss_sum += total_loss.item() * bs
                val_total += bs

        val_loss = val_loss_sum / max(val_total, 1)
        elapsed = time.perf_counter() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best.pt"))
        else:
            epochs_without_improvement += 1

        if True:
            remaining_epochs = TRAIN.epochs - epoch - 1
            eta_min = remaining_epochs * elapsed / 60
            print(
                f"  Epoch {epoch+1:3d}/{TRAIN.epochs} | "
                f"train={train_loss:.4f} val={val_loss:.4f} | "
                f"kl_w={kl_weight:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.6f} | "
                f"{elapsed:.1f}s/ep | ETA ~{eta_min:.0f}min"
            )

        # Early stopping
        if epochs_without_improvement >= TRAIN.early_stop_patience:
            print(
                f"  Early stopping at epoch {epoch+1} "
                f"(no improvement for {TRAIN.early_stop_patience} epochs)"
            )
            break

    # Save final
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "final.pt"))
    torch.save(history, os.path.join(checkpoint_dir, "history.pt"))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ACT via behavior cloning")
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "dinov2-vits14", "siglip2-base"],
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK.default_chunk_size)
    parser.add_argument("--batch-size", type=int, default=TRAIN.batch_size)
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Cap number of episodes to load (default: all)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers (default: 4 for CUDA, 0 for MPS/CPU)")
    parser.add_argument("--cache-episodes", type=int, default=None,
                        help="LRU cache size per worker (default: auto based on RAM)")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train(
        backbone=args.backbone,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        max_episodes=args.max_episodes,
        num_workers=args.num_workers,
        cache_episodes=args.cache_episodes,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
