"""Behavior cloning training loop for the ACT policy (Experiment 3).

Adapted from Experiment 2 for the multi-binary held-state action space.
Key differences from Exp 2:
  - Loads HDF5 episodes from data/{task_name}/{shard}/episode_*.hdf5
  - Proprio: [cursor_x, cursor_y, mouse_left, keys_held_0..42] = 46 dims
  - CVAE action vector: [dx, dy, mouse_left, keys_held_0..42] = 46 dims
  - Loss: masked BCE for mouse_left, sum-of-BCE for 43 independent keys

Saves checkpoints + training history.
"""

import argparse
import glob
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.miniwob_pygame.config import ACTION, CHUNK, TRAIN
from experiments.miniwob_pygame.model import ACT, count_parameters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_cache_size() -> int:
    """Pick a reasonable HDF5 chunk cache size based on available RAM."""
    try:
        import psutil
        avail = psutil.virtual_memory().available
        return min(avail // 4, 4 * 1024**3)  # 25% of free RAM, cap 4 GB
    except ImportError:
        return 512 * 1024**2  # 512 MB fallback


def discover_episodes(
    data_dir: str,
    tasks: list[str] | None = None,
) -> list[str]:
    """Find all episode HDF5 files under data_dir/{task_name}/**/*.hdf5.

    Args:
        data_dir: Root data directory.
        tasks: Optional list of task names to include. If None, all tasks found.

    Returns:
        Sorted list of absolute paths to episode files.
    """
    if tasks is not None:
        # Only look in specified task directories
        task_dirs = [os.path.join(data_dir, t) for t in tasks]
    else:
        # Auto-discover: every subdirectory of data_dir that contains episodes
        task_dirs = []
        for entry in sorted(os.listdir(data_dir)):
            candidate = os.path.join(data_dir, entry)
            if os.path.isdir(candidate):
                task_dirs.append(candidate)

    paths: list[str] = []
    for td in task_dirs:
        if not os.path.isdir(td):
            continue
        found = glob.glob(os.path.join(td, "**", "episode_*.hdf5"), recursive=True)
        paths.extend(found)

    paths.sort()
    return paths


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChunkDataset(Dataset):
    """HDF5-backed dataset that samples (obs, proprio, action_chunk, ...) tuples.

    Every timestep t in every episode is a valid start index.
    Action chunks are zero-padded when near the episode end.
    """

    def __init__(
        self,
        episode_paths: list[str],
        chunk_size: int,
    ) -> None:
        self.episode_paths = episode_paths
        self.chunk_size = chunk_size

        # Build flat index: (episode_idx, timestep)
        self.index: list[tuple[int, int]] = []
        self.episode_lengths: list[int] = []

        for ep_idx, path in enumerate(episode_paths):
            with h5py.File(path, "r") as f:
                T = f["observations"].shape[0]
            self.episode_lengths.append(T)
            for t in range(T):
                self.index.append((ep_idx, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        torch.Tensor,  # obs (3, 224, 224)
        torch.Tensor,  # proprio (46,)
        torch.Tensor,  # actions_cvae (chunk, 46) for CVAE encoder
        torch.Tensor,  # dx_chunk (chunk,)
        torch.Tensor,  # dy_chunk (chunk,)
        torch.Tensor,  # mouse_chunk (chunk,)
        torch.Tensor,  # keys_chunk (chunk, 43)
        torch.Tensor,  # pad_mask (chunk,) — 1 where padded, 0 where real
    ]:
        ep_idx, t = self.index[idx]
        C = self.chunk_size
        ep_len = self.episode_lengths[ep_idx]

        with h5py.File(self.episode_paths[ep_idx], "r") as f:
            # --- Observation at time t ---
            obs_np = f["observations"][t]  # (224, 224, 3) uint8
            obs = torch.from_numpy(obs_np).permute(2, 0, 1).float() / 255.0

            # --- Proprioception at time t ---
            cursor_xy = f["cursor_positions"][t]  # (2,) float32, already normalized
            mouse_t = float(f["actions_mouse_left"][t])
            keys_t = f["actions_keys_held"][t].astype(np.float32)  # (43,)

            proprio = np.zeros(2 + 1 + ACTION.num_keys, dtype=np.float32)
            proprio[0] = cursor_xy[0]
            proprio[1] = cursor_xy[1]
            proprio[2] = mouse_t
            proprio[3:] = keys_t
            proprio = torch.from_numpy(proprio)

            # --- Action chunk [t:t+C], zero-padded if near end ---
            end_t = min(t + C, ep_len)
            real_len = end_t - t

            dx_chunk = np.zeros(C, dtype=np.float32)
            dy_chunk = np.zeros(C, dtype=np.float32)
            mouse_chunk = np.zeros(C, dtype=np.float32)
            keys_chunk = np.zeros((C, ACTION.num_keys), dtype=np.float32)
            pad_mask = np.ones(C, dtype=np.float32)  # 1=padded, 0=real

            dx_chunk[:real_len] = f["actions_dx"][t:end_t]
            dy_chunk[:real_len] = f["actions_dy"][t:end_t]
            mouse_chunk[:real_len] = f["actions_mouse_left"][t:end_t].astype(np.float32)
            keys_chunk[:real_len] = f["actions_keys_held"][t:end_t].astype(np.float32)
            pad_mask[:real_len] = 0.0

        dx_chunk = torch.from_numpy(dx_chunk)
        dy_chunk = torch.from_numpy(dy_chunk)
        mouse_chunk = torch.from_numpy(mouse_chunk)
        keys_chunk = torch.from_numpy(keys_chunk)
        pad_mask = torch.from_numpy(pad_mask)

        # --- Actions for CVAE encoder: (chunk, action_dim=46) ---
        # action_dim = 2 (dx,dy) + 1 (mouse_left) + 43 (keys_held)
        actions_cvae = torch.zeros(C, 2 + 1 + ACTION.num_keys)
        actions_cvae[:real_len, 0] = dx_chunk[:real_len]
        actions_cvae[:real_len, 1] = dy_chunk[:real_len]
        actions_cvae[:real_len, 2] = mouse_chunk[:real_len]
        actions_cvae[:real_len, 3:] = keys_chunk[:real_len]

        return obs, proprio, actions_cvae, dx_chunk, dy_chunk, mouse_chunk, keys_chunk, pad_mask


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    backbone: str = "resnet18",
    chunk_size: int = CHUNK.default_chunk_size,
    batch_size: int = TRAIN.batch_size,
    max_episodes: int | None = None,
    num_workers: int | None = None,
    data_dir: str | None = None,
    checkpoint_dir: str | None = None,
    device: str = "cpu",
    tasks: list[str] | None = None,
    max_epochs: int | None = None,
    hf_upload_repo: str | None = None,
) -> None:
    base = os.path.dirname(__file__)

    if data_dir is None:
        data_dir = os.path.join(base, "data")

    # Discover episodes
    episode_paths = discover_episodes(data_dir, tasks=tasks)
    if not episode_paths:
        raise ValueError(f"No episodes found in {data_dir} (tasks={tasks})")

    print(f"Discovered {len(episode_paths)} episodes in {data_dir}")

    if max_episodes is not None and max_episodes < len(episode_paths):
        episode_paths = episode_paths[:max_episodes]
        print(f"Capped to {max_episodes} episodes")

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(
            base, "checkpoints", f"{backbone}_chunk{chunk_size}"
        )
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Train/val split by episode
    rng = np.random.default_rng(42)
    n_eps = len(episode_paths)
    indices = rng.permutation(n_eps)
    val_count = max(1, int(n_eps * TRAIN.val_fraction))
    val_indices = set(indices[:val_count].tolist())
    train_indices = set(indices[val_count:].tolist())

    train_paths = [episode_paths[i] for i in sorted(train_indices)]
    val_paths = [episode_paths[i] for i in sorted(val_indices)]

    print(f"Train: {len(train_paths)} episodes, Val: {len(val_paths)} episodes")

    # Default num_workers
    if num_workers is None:
        num_workers = 4 if device.startswith("cuda") else 0
    print(f"DataLoader workers: {num_workers}")

    train_dataset = ChunkDataset(train_paths, chunk_size)
    val_dataset = ChunkDataset(val_paths, chunk_size)
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
            continue
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

    epochs = max_epochs if max_epochs is not None else TRAIN.epochs

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
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
    kl_anneal_epochs = int(epochs * TRAIN.kl_anneal_fraction)

    for epoch in range(epochs):
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
            (obs, proprio, actions_cvae, dx_gt, dy_gt, mouse_gt, keys_gt, pad_gt) = batch
            obs = obs.to(device)
            proprio = proprio.to(device)
            actions_cvae = actions_cvae.to(device)
            dx_gt = dx_gt.to(device)
            dy_gt = dy_gt.to(device)
            mouse_gt = mouse_gt.to(device)
            keys_gt = keys_gt.to(device)
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

                # Masked BCE for mouse_left
                loss_mouse = (
                    F.binary_cross_entropy_with_logits(
                        out["mouse_left"], mouse_gt, reduction="none"
                    )
                    * mask
                ).sum() / mask_sum

                # Masked sum-of-BCE for 43 independent key sigmoids
                # keys_gt: (B, chunk, 43), out["keys_held"]: (B, chunk, 43)
                loss_keys_per = F.binary_cross_entropy_with_logits(
                    out["keys_held"], keys_gt, reduction="none"
                )  # (B, chunk, 43)
                # Sum over keys, then mask and average over (steps x batch)
                loss_keys_summed = loss_keys_per.sum(dim=-1)  # (B, chunk)
                loss_keys = (loss_keys_summed * mask).sum() / mask_sum

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
                    + TRAIN.loss_weight_mouse * loss_mouse
                    + TRAIN.loss_weight_keys * loss_keys
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
        model.train(False)
        val_loss_sum = 0.0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                (obs, proprio, actions_cvae, dx_gt, dy_gt, mouse_gt, keys_gt, pad_gt) = batch
                obs = obs.to(device)
                proprio = proprio.to(device)
                actions_cvae = actions_cvae.to(device)
                dx_gt = dx_gt.to(device)
                dy_gt = dy_gt.to(device)
                mouse_gt = mouse_gt.to(device)
                keys_gt = keys_gt.to(device)
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

                loss_mouse = (
                    F.binary_cross_entropy_with_logits(
                        out["mouse_left"], mouse_gt, reduction="none"
                    )
                    * mask
                ).sum() / mask_sum

                loss_keys_per = F.binary_cross_entropy_with_logits(
                    out["keys_held"], keys_gt, reduction="none"
                )
                loss_keys_summed = loss_keys_per.sum(dim=-1)
                loss_keys = (loss_keys_summed * mask).sum() / mask_sum

                loss_pad = F.binary_cross_entropy_with_logits(
                    out["pad_logits"], pad_gt
                )

                mu = out["mu"]
                logvar = out["logvar"]
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

                total_loss = (
                    loss_dx
                    + loss_dy
                    + TRAIN.loss_weight_mouse * loss_mouse
                    + TRAIN.loss_weight_keys * loss_keys
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

        remaining_epochs = epochs - epoch - 1
        eta_min = remaining_epochs * elapsed / 60
        print(
            f"  Epoch {epoch+1:3d}/{epochs} | "
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

    # Upload checkpoints to HF Hub if repo specified
    if hf_upload_repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(hf_upload_repo, repo_type="model", exist_ok=True)
        print(f"Uploading checkpoints to {hf_upload_repo} ...")
        api.upload_folder(
            repo_id=hf_upload_repo,
            repo_type="model",
            folder_path=checkpoint_dir,
            path_in_repo=f"{backbone}_chunk{chunk_size}",
        )
        print("Upload complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ACT via behavior cloning (Exp 3)")
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
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Root data directory containing task subdirectories")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Task names to include (default: all tasks in data_dir)")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override number of epochs (useful for smoke tests)")
    parser.add_argument("--hf-upload-repo", type=str, default=None,
                        help="HF model repo to upload checkpoints to after training")
    args = parser.parse_args()

    train(
        backbone=args.backbone,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        max_episodes=args.max_episodes,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        tasks=args.tasks,
        max_epochs=args.max_epochs,
        hf_upload_repo=args.hf_upload_repo,
    )
