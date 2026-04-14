"""Behavior cloning training loop for the BaselineCNN.

Single-step, no-chunking baseline. Same data as ACT but simpler:
- L1 loss on dx, dy
- BCE loss on click
- CE loss on key (28-class)
No CVAE, no chunking, no proprioception, no temporal ensemble.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.act_drag_label.config import ACTION, TRAIN
from experiments.act_drag_label.baseline_cnn import BaselineCNN, count_parameters
from experiments.act_drag_label.train import build_episode_offsets


# ---------------------------------------------------------------------------
# Dataset — one sample per timestep, no chunking
# ---------------------------------------------------------------------------

class SingleStepDataset(Dataset):
    """Arrow-backed dataset returning (obs, dx, dy, click, key) per timestep."""

    def __init__(
        self,
        ds,
        episode_offsets: dict[int, tuple[int, int]],
        episode_ids: set[int],
        action_arrays: dict[str, np.ndarray],
    ) -> None:
        self.ds = ds
        self.action_dx = action_arrays["action_dx"]
        self.action_dy = action_arrays["action_dy"]
        self.action_click = action_arrays["action_click"]
        self.action_key = action_arrays["action_key"]

        # Build flat index: every timestep in the selected episodes
        self.row_indices: list[int] = []
        for eid in sorted(episode_ids):
            if eid not in episode_offsets:
                continue
            start, length = episode_offsets[eid]
            self.row_indices.extend(range(start, start + length))

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, idx: int) -> tuple[
        torch.Tensor,  # obs (3, 224, 224)
        torch.Tensor,  # dx scalar
        torch.Tensor,  # dy scalar
        torch.Tensor,  # click scalar
        torch.Tensor,  # key int
    ]:
        row_idx = self.row_indices[idx]

        # Image from Arrow (single PNG decode)
        row = self.ds[row_idx]
        obs = torch.from_numpy(np.array(row["image"])).permute(2, 0, 1).float() / 255.0

        # Actions from pre-extracted numpy
        dx = torch.tensor(self.action_dx[row_idx], dtype=torch.float32)
        dy = torch.tensor(self.action_dy[row_idx], dtype=torch.float32)
        click = torch.tensor(float(self.action_click[row_idx]), dtype=torch.float32)
        key = torch.tensor(int(self.action_key[row_idx]), dtype=torch.int64)

        return obs, dx, dy, click, key


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    batch_size: int = TRAIN.batch_size,
    max_episodes: int | None = None,
    num_workers: int | None = None,
    data_dir: str | None = None,
    checkpoint_dir: str | None = None,
    device: str = "cpu",
    epochs: int = TRAIN.epochs,
    hf_data_repo: str | None = None,
    hf_upload_repo: str | None = None,
) -> None:
    from datasets import load_dataset, load_from_disk

    base = os.path.dirname(__file__)

    # Load dataset
    if hf_data_repo:
        print(f"Loading dataset from {hf_data_repo} ...")
        ds = load_dataset(hf_data_repo, split="train")
        print("Dataset loaded.")
    elif data_dir:
        print(f"Loading dataset from {data_dir} ...")
        ds = load_from_disk(data_dir)
    else:
        default_dir = os.path.join(base, "data")
        print(f"Loading dataset from {default_dir} ...")
        ds = load_from_disk(default_dir)

    print(f"Dataset: {ds}", flush=True)

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(base, "checkpoints", "baseline")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Pre-extract scalar columns
    t_extract = time.perf_counter()
    ep_ids_np = np.array(ds["episode_id"], dtype=np.int32)
    action_arrays = {
        "action_dx": np.array(ds["action_dx"], dtype=np.float32),
        "action_dy": np.array(ds["action_dy"], dtype=np.float32),
        "action_click": np.array(ds["action_click"], dtype=np.int8),
        "action_key": np.array(ds["action_key"], dtype=np.int8),
    }
    print(
        f"Scalar columns extracted in {time.perf_counter() - t_extract:.2f}s",
        flush=True,
    )

    episode_offsets = build_episode_offsets(ep_ids_np)
    all_episode_ids = sorted(episode_offsets.keys())

    if max_episodes is not None and max_episodes < len(all_episode_ids):
        all_episode_ids = all_episode_ids[:max_episodes]

    print(f"Using {len(all_episode_ids)} episodes")

    # Train/val split (same seed as ACT for fair comparison)
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(all_episode_ids))
    val_count = max(1, int(len(all_episode_ids) * TRAIN.val_fraction))
    val_ids = set(all_episode_ids[i] for i in indices[:val_count])
    train_ids = set(all_episode_ids[i] for i in indices[val_count:])

    print(f"Train: {len(train_ids)} episodes, Val: {len(val_ids)} episodes")

    if num_workers is None:
        num_workers = 4 if device.startswith("cuda") else 0

    train_dataset = SingleStepDataset(ds, episode_offsets, train_ids, action_arrays)
    val_dataset = SingleStepDataset(ds, episode_offsets, val_ids, action_arrays)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}", flush=True)

    use_cuda = device.startswith("cuda")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        pin_memory=use_cuda,
    )

    # Model
    model = BaselineCNN().to(device)
    if use_cuda:
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model)
        print("Model compiled with torch.compile (TF32 enabled)", flush=True)
    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN.lr, weight_decay=TRAIN.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = TRAIN.use_amp and use_cuda
    scaler = torch.amp.GradScaler(enabled=use_amp)
    amp_dtype = torch.float16 if use_amp else torch.float32

    print(f"Device: {device}, Batch size: {batch_size}, Epochs: {epochs}", flush=True)

    history: dict[str, list] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        t0 = time.perf_counter()

        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_total = 0
        n_batches = len(train_loader)

        for batch_idx, (obs, dx_gt, dy_gt, click_gt, key_gt) in enumerate(train_loader):
            obs = obs.to(device)
            dx_gt = dx_gt.to(device)
            dy_gt = dy_gt.to(device)
            click_gt = click_gt.to(device)
            key_gt = key_gt.to(device)

            with torch.amp.autocast(
                device_type=device.split(":")[0], dtype=amp_dtype, enabled=use_amp
            ):
                dx_pred, dy_pred, click_logit, key_logits = model(obs)

                loss_dx = F.l1_loss(dx_pred.squeeze(-1), dx_gt)
                loss_dy = F.l1_loss(dy_pred.squeeze(-1), dy_gt)
                loss_click = F.binary_cross_entropy_with_logits(
                    click_logit.squeeze(-1), click_gt
                )
                loss_key = F.cross_entropy(key_logits, key_gt)

                total_loss = (
                    loss_dx + loss_dy
                    + TRAIN.loss_weight_click * loss_click
                    + TRAIN.loss_weight_key * loss_key
                )

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = obs.size(0)
            train_loss_sum += total_loss.item() * bs
            train_total += bs

            if epoch == 0 and (batch_idx == 0 or (batch_idx + 1) % 100 == 0):
                print(
                    f"    batch {batch_idx+1}/{n_batches} | "
                    f"loss={total_loss.item():.4f} | "
                    f"{time.perf_counter() - t0:.1f}s elapsed",
                    flush=True,
                )

        scheduler.step()
        train_loss = train_loss_sum / max(train_total, 1)

        # --- Validate ---
        model.train(False)
        val_loss_sum = 0.0
        val_total = 0

        with torch.no_grad():
            for obs, dx_gt, dy_gt, click_gt, key_gt in val_loader:
                obs = obs.to(device)
                dx_gt = dx_gt.to(device)
                dy_gt = dy_gt.to(device)
                click_gt = click_gt.to(device)
                key_gt = key_gt.to(device)

                with torch.amp.autocast(
                    device_type=device.split(":")[0], dtype=amp_dtype, enabled=use_amp
                ):
                    dx_pred, dy_pred, click_logit, key_logits = model(obs)
                    loss_dx = F.l1_loss(dx_pred.squeeze(-1), dx_gt)
                    loss_dy = F.l1_loss(dy_pred.squeeze(-1), dy_gt)
                    loss_click = F.binary_cross_entropy_with_logits(
                        click_logit.squeeze(-1), click_gt
                    )
                    loss_key = F.cross_entropy(key_logits, key_gt)
                    total_loss = (
                        loss_dx + loss_dy
                        + TRAIN.loss_weight_click * loss_click
                        + TRAIN.loss_weight_key * loss_key
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
            best_pt_path = os.path.join(checkpoint_dir, "best.pt")
            state = {k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()}
            torch.save(state, best_pt_path)
            if hf_upload_repo:
                import logging
                import threading
                from huggingface_hub import HfApi
                _val = val_loss
                def _upload():
                    try:
                        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
                        from huggingface_hub.utils import disable_progress_bars
                        disable_progress_bars()
                        api = HfApi()
                        api.create_repo(hf_upload_repo, repo_type="model", exist_ok=True)
                        api.upload_file(
                            path_or_fileobj=best_pt_path,
                            path_in_repo="baseline/best.pt",
                            repo_id=hf_upload_repo,
                            repo_type="model",
                        )
                        print(f"    Uploaded baseline best.pt (val={_val:.4f})", flush=True)
                    except Exception as e:
                        print(f"    Upload failed: {e}", flush=True)
                threading.Thread(target=_upload, daemon=True).start()
        else:
            epochs_without_improvement += 1

        remaining = epochs - epoch - 1
        eta_min = remaining * elapsed / 60
        print(
            f"  Epoch {epoch+1:3d}/{epochs} | "
            f"train={train_loss:.4f} val={val_loss:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.6f} | "
            f"{elapsed:.1f}s/ep | ETA ~{eta_min:.0f}min",
            flush=True,
        )

        if epochs_without_improvement >= TRAIN.early_stop_patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Save final
    state = {k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()}
    torch.save(state, os.path.join(checkpoint_dir, "final.pt"))
    torch.save(history, os.path.join(checkpoint_dir, "history.pt"))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")

    if hf_upload_repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(hf_upload_repo, repo_type="model", exist_ok=True)
        print(f"Uploading checkpoints to {hf_upload_repo} ...")
        api.upload_folder(
            repo_id=hf_upload_repo,
            repo_type="model",
            folder_path=checkpoint_dir,
            path_in_repo="baseline",
        )
        print("Upload complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BaselineCNN via behavior cloning")
    parser.add_argument("--batch-size", type=int, default=TRAIN.batch_size)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=TRAIN.epochs)
    parser.add_argument("--hf-data-repo", type=str, default=None)
    parser.add_argument("--hf-upload-repo", type=str, default=None)
    args = parser.parse_args()

    train(
        batch_size=args.batch_size,
        max_episodes=args.max_episodes,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        epochs=args.epochs,
        hf_data_repo=args.hf_data_repo,
        hf_upload_repo=args.hf_upload_repo,
    )
