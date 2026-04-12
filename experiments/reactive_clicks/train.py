"""Behavior cloning training loop for the TinyCNN policy.

Loads HDF5 expert demonstrations, trains with:
- L1 loss on dx, dy (continuous pixel deltas)
- Cross-entropy loss on btn (3-class classification)
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.reactive_clicks.config import TRAIN
from experiments.reactive_clicks.model import TinyCNN, count_parameters


class EpisodeDataset(Dataset):
    """Loads all HDF5 episodes into memory as a flat list of (obs, dx, dy, btn) frames."""

    def __init__(self, episode_files: list[str]) -> None:
        self.observations: list[np.ndarray] = []
        self.dx: list[float] = []
        self.dy: list[float] = []
        self.btn: list[int] = []

        for path in episode_files:
            with h5py.File(path, "r") as f:
                obs = f["observations"][:]
                dx = f["actions_dx"][:]
                dy = f["actions_dy"][:]
                btn = f["actions_btn"][:]
                for i in range(len(dx)):
                    self.observations.append(obs[i])
                    self.dx.append(float(dx[i]))
                    self.dy.append(float(dy[i]))
                    self.btn.append(int(btn[i]))

    def __len__(self) -> int:
        return len(self.dx)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float, float, int]:
        obs = torch.from_numpy(self.observations[idx]).permute(2, 0, 1).float() / 255.0
        return obs, self.dx[idx], self.dy[idx], self.btn[idx]


def train(
    data_dir: str | None = None,
    checkpoint_dir: str | None = None,
    device: str = "cpu",
) -> None:
    base = os.path.dirname(__file__)
    if data_dir is None:
        data_dir = os.path.join(base, "data")
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(base, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Find episodes
    episode_files = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
    if not episode_files:
        print(f"No episodes found in {data_dir}. Run generate_data.py first.")
        return
    print(f"Found {len(episode_files)} episodes")

    # Train/val split by episode
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(episode_files))
    val_count = max(1, int(len(episode_files) * TRAIN.val_fraction))
    val_files = [episode_files[i] for i in indices[:val_count]]
    train_files = [episode_files[i] for i in indices[val_count:]]

    print(f"Train: {len(train_files)} episodes, Val: {len(val_files)} episodes")

    train_dataset = EpisodeDataset(train_files)
    val_dataset = EpisodeDataset(val_files)
    print(f"Train frames: {len(train_dataset)}, Val frames: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=TRAIN.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=TRAIN.batch_size, shuffle=False, num_workers=0
    )

    # Model
    model = TinyCNN().to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Loss
    btn_weights = torch.tensor(TRAIN.btn_class_weights, dtype=torch.float32, device=device)
    ce_btn = nn.CrossEntropyLoss(weight=btn_weights)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TRAIN.lr, weight_decay=TRAIN.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAIN.epochs
    )

    # Training loop
    history: dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_mae_px": [], "val_mae_px": [],
        "train_btn_acc": [], "val_btn_acc": [],
    }
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(TRAIN.epochs):
        t0 = time.perf_counter()

        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_mae_sum = 0.0
        train_btn_correct = 0
        train_total = 0

        for obs, dx_gt, dy_gt, btn_gt in train_loader:
            obs = obs.to(device)
            dx_gt = dx_gt.to(device, dtype=torch.float32).unsqueeze(1)
            dy_gt = dy_gt.to(device, dtype=torch.float32).unsqueeze(1)
            btn_gt = btn_gt.to(device, dtype=torch.long)

            dx_pred, dy_pred, btn_logits = model(obs)

            loss_dx = F.l1_loss(dx_pred, dx_gt)
            loss_dy = F.l1_loss(dy_pred, dy_gt)
            loss_btn = ce_btn(btn_logits, btn_gt)
            loss = loss_dx + loss_dy + loss_btn

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = obs.size(0)
            train_loss_sum += loss.item() * bs
            train_mae_sum += (loss_dx.item() + loss_dy.item()) / 2 * bs
            train_btn_correct += (btn_logits.argmax(1) == btn_gt).sum().item()
            train_total += bs

        scheduler.step()

        train_loss = train_loss_sum / train_total
        train_mae = train_mae_sum / train_total
        train_btn_acc = train_btn_correct / train_total

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        val_btn_correct = 0
        val_total = 0

        with torch.no_grad():
            for obs, dx_gt, dy_gt, btn_gt in val_loader:
                obs = obs.to(device)
                dx_gt = dx_gt.to(device, dtype=torch.float32).unsqueeze(1)
                dy_gt = dy_gt.to(device, dtype=torch.float32).unsqueeze(1)
                btn_gt = btn_gt.to(device, dtype=torch.long)

                dx_pred, dy_pred, btn_logits = model(obs)

                loss_dx = F.l1_loss(dx_pred, dx_gt)
                loss_dy = F.l1_loss(dy_pred, dy_gt)
                loss_btn = ce_btn(btn_logits, btn_gt)
                loss = loss_dx + loss_dy + loss_btn

                bs = obs.size(0)
                val_loss_sum += loss.item() * bs
                val_mae_sum += (loss_dx.item() + loss_dy.item()) / 2 * bs
                val_btn_correct += (btn_logits.argmax(1) == btn_gt).sum().item()
                val_total += bs

        val_loss = val_loss_sum / val_total
        val_mae = val_mae_sum / val_total
        val_btn_acc = val_btn_correct / val_total

        elapsed = time.perf_counter() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mae_px"].append(train_mae)
        history["val_mae_px"].append(val_mae)
        history["train_btn_acc"].append(train_btn_acc)
        history["val_btn_acc"].append(val_btn_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best.pt"))
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{TRAIN.epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"mae={val_mae:.2f}px | btn_acc={val_btn_acc:.3f} | "
                f"lr={scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s"
            )

        # Early stopping
        if epochs_without_improvement >= TRAIN.early_stop_patience:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {TRAIN.early_stop_patience} epochs)")
            break

    # Save final
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "final.pt"))
    torch.save(history, os.path.join(checkpoint_dir, "history.pt"))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TinyCNN via behavior cloning")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train(data_dir=args.data_dir, checkpoint_dir=args.checkpoint_dir, device=args.device)
