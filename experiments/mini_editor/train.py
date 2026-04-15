"""Behavior cloning training loop for the ACT policy (Experiment 5: Mini Text Editor).

Loads expert demonstrations from HF datasets, trains with:
- Gaussian-smoothed soft CE loss on dx, dy (discrete 49-bin exponential grid)
- Expected-value L1 loss on dx, dy (pixel accuracy)
- Masked BCE loss on mouse_left (binary)
- Masked focal BCE loss on keys_held (53 independent channels)
- Unmasked BCE loss on pad prediction

First V+L+A training: vision (locate words), language (parse edit instruction),
action (multi-step motor sequences at 30 Hz).

Saves checkpoints + training history with per-head loss tracking.
"""

import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.mini_editor.config import (
    ACTION,
    BIN_CENTERS,
    CHUNK,
    DATA,
    ENV,
    FOCAL,
    MODEL,
    NUM_BINS,
    NUM_KEYS,
    PROPRIO_DIM,
    SOFT_BIN_TARGETS,
    TRAIN,
)
from experiments.mini_editor.model import ACT, count_parameters
from experiments.mini_editor.text_encoder import build_text_encoder, tokenize_instruction


# ---------------------------------------------------------------------------
# Bin quantisation helper
# ---------------------------------------------------------------------------


def _delta_to_bin(raw_delta: np.ndarray) -> np.ndarray:
    """Convert raw pixel deltas to closest bin indices.

    BIN_CENTERS is a sorted (49,) float32 array of exponential bin centers.
    Returns int64 indices in [0, NUM_BINS-1].
    """
    idx = np.searchsorted(BIN_CENTERS, raw_delta, side="left")
    idx = np.clip(idx, 0, NUM_BINS - 1)
    left = np.clip(idx - 1, 0, NUM_BINS - 1)
    closer_left = np.abs(raw_delta - BIN_CENTERS[left]) < np.abs(
        raw_delta - BIN_CENTERS[idx]
    )
    return np.where(closer_left, left, idx).astype(np.int64)


# ---------------------------------------------------------------------------
# Episode offset table
# ---------------------------------------------------------------------------


def build_episode_offsets(
    ep_ids: np.ndarray,
) -> dict[int, tuple[int, int]]:
    """Return {episode_id: (start_row, length)} from a sorted episode_id array.

    Assumes the array is sorted by (episode_id, timestep) — which is the
    natural output order from generate_data.py / Dataset.from_generator().
    """
    changes = np.where(np.diff(ep_ids) != 0)[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(ep_ids)]])
    offsets: dict[int, tuple[int, int]] = {}
    for s, e in zip(starts, ends):
        offsets[int(ep_ids[s])] = (int(s), int(e - s))
    return offsets


# ---------------------------------------------------------------------------
# Focal BCE loss
# ---------------------------------------------------------------------------


class FocalBCELoss(nn.Module):
    """Focal binary cross-entropy loss for class-imbalanced binary targets.

    Reduces the contribution of well-classified examples so the model
    focuses on hard, misclassified ones (important for the sparse 53-key
    action space where most keys are released most of the time).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal BCE loss.

        Args:
            logits: Raw logits, any shape.
            targets: Binary targets (0 or 1), same shape as logits.

        Returns:
            Scalar mean focal loss.
        """
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # p_t = p for positive, (1-p) for negative
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma

        # alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        loss = alpha_t * focal_weight * ce
        return loss.mean()


# ---------------------------------------------------------------------------
# Image pre-decode memmap cache
# ---------------------------------------------------------------------------


def predecode_images(ds) -> np.ndarray:
    """Decode all images from Arrow/parquet into a disk-backed memmap.

    Returns a numpy memmap array of shape (N, obs_h, obs_w, 3) uint8
    at the dataset resolution (ENV.obs_h x ENV.obs_w = 384 x 512).
    Reuses an existing cache if the file size matches.

    Decodes in sequential batches to limit peak RAM usage — each batch
    decodes BATCH_SIZE images, writes to memmap, and flushes.
    """
    n_samples = len(ds)
    obs_h = ENV.obs_h  # 384
    obs_w = ENV.obs_w  # 512
    img_shape = (n_samples, obs_h, obs_w, 3)
    memmap_path = os.path.join("/tmp", "exp5_images_cache.dat")
    t_decode = time.perf_counter()

    if os.path.exists(memmap_path):
        expected_bytes = int(np.prod(img_shape))
        actual_bytes = os.path.getsize(memmap_path)
        if actual_bytes == expected_bytes:
            images = np.memmap(
                memmap_path, dtype=np.uint8, mode="r", shape=img_shape
            )
            print(
                f"Image cache loaded from {memmap_path} "
                f"({actual_bytes / 1024**3:.1f}GB)",
                flush=True,
            )
            return images
        else:
            os.remove(memmap_path)

    images = np.memmap(memmap_path, dtype=np.uint8, mode="w+", shape=img_shape)

    # Decode in batches to limit RAM: decode BATCH_SIZE images at a time,
    # write to memmap, flush, and move on. This avoids holding all decoded
    # images in RAM simultaneously.
    BATCH_SIZE = 5000
    done = 0
    for batch_start in range(0, n_samples, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_samples)
        # Use HF datasets batch access for efficiency
        batch = ds[batch_start:batch_end]
        for j, img in enumerate(batch["image"]):
            idx = batch_start + j
            img_np = np.array(img)
            if img_np.shape[:2] != (obs_h, obs_w):
                from PIL import Image

                img_pil = img if hasattr(img, "resize") else Image.fromarray(img_np)
                img_pil = img_pil.resize((obs_w, obs_h), Image.BILINEAR)
                img_np = np.array(img_pil)
            images[idx] = img_np
        images.flush()
        done = batch_end
        elapsed = time.perf_counter() - t_decode
        print(
            f"  Pre-decoded {done}/{n_samples} images "
            f"({elapsed:.1f}s, {done / elapsed:.0f} img/s)",
            flush=True,
        )

    images_gb = images.nbytes / 1024**3
    elapsed = time.perf_counter() - t_decode
    print(
        f"Images pre-decoded to {memmap_path} in "
        f"{elapsed:.1f}s "
        f"({images_gb:.1f}GB, "
        f"{n_samples / elapsed:.0f} img/s)",
        flush=True,
    )
    return images


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ChunkDataset(Dataset):
    """Dataset that samples (obs, proprio, input_ids, attn_mask, action_chunk) tuples.

    Every timestep t in every episode is a valid start index.
    Action chunks are zero-padded when near the episode end.

    All columns (including images) are pre-extracted to numpy arrays
    so __getitem__ is pure array indexing with no PNG decode.
    Instructions are pre-tokenized at init time.
    """

    def __init__(
        self,
        images: np.ndarray,
        episode_offsets: dict[int, tuple[int, int]],
        episode_ids: set[int],
        chunk_size: int,
        action_arrays: dict[str, np.ndarray],
        token_ids: np.ndarray,
        attention_masks: np.ndarray,
    ) -> None:
        self.images = images
        self.chunk_size = chunk_size
        self.episode_offsets = {
            eid: episode_offsets[eid]
            for eid in episode_ids
            if eid in episode_offsets
        }
        self.action_dx = action_arrays["action_dx"]
        self.action_dy = action_arrays["action_dy"]
        self.action_mouse_left = action_arrays["action_mouse_left"]
        self.action_keys_held = action_arrays["action_keys_held"]
        self.proprio = action_arrays["proprio"]
        self.token_ids = token_ids
        self.attention_masks = attention_masks

        # Model input dimensions (resized from dataset resolution)
        self.model_h = MODEL.obs_h  # 288
        self.model_w = MODEL.obs_w  # 384

        # Build flat index grouped by episode (sequential within each episode
        # for memmap-friendly access patterns)
        self.index: list[tuple[int, int]] = []
        self.episode_ranges: list[tuple[int, int]] = []
        for eid in sorted(self.episode_offsets):
            _, length = self.episode_offsets[eid]
            start = len(self.index)
            for t in range(length):
                self.index.append((eid, t))
            self.episode_ranges.append((start, length))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        torch.Tensor,  # obs (3, model_h, model_w)
        torch.Tensor,  # proprio (56,)
        torch.Tensor,  # input_ids (L,) int64
        torch.Tensor,  # attention_mask (L,) int64
        torch.Tensor,  # dx_chunk (chunk,) int64 bin indices
        torch.Tensor,  # dy_chunk (chunk,) int64 bin indices
        torch.Tensor,  # mouse_chunk (chunk,) float32
        torch.Tensor,  # keys_chunk (chunk, 53) float32
        torch.Tensor,  # pad_mask (chunk,) — 1 where padded, 0 where real
    ]:
        eid, t = self.index[idx]
        start_row, ep_len = self.episode_offsets[eid]
        C = self.chunk_size
        row_idx = start_row + t

        # --- Observation at time t (pre-decoded uint8 array) ---
        # Dataset resolution: (ENV.obs_h, ENV.obs_w, 3) = (384, 512, 3)
        # Resize to model input: (MODEL.obs_h, MODEL.obs_w) = (288, 384)
        img_np = self.images[row_idx]  # (384, 512, 3)
        if img_np.shape[0] != self.model_h or img_np.shape[1] != self.model_w:
            from PIL import Image

            img_pil = Image.fromarray(img_np)
            img_pil = img_pil.resize(
                (self.model_w, self.model_h), Image.BILINEAR
            )
            img_np = np.array(img_pil)
        obs = (
            torch.from_numpy(img_np.copy()).permute(2, 0, 1).float() / 255.0
        )

        # --- Proprioception at time t ---
        proprio = torch.from_numpy(self.proprio[row_idx].copy())

        # --- Instruction tokens (pre-tokenized) ---
        input_ids = torch.from_numpy(self.token_ids[row_idx].copy())
        attention_mask = torch.from_numpy(self.attention_masks[row_idx].copy())

        # --- Action chunk [t:t+C], zero-padded if near end ---
        end_t = min(t + C, ep_len)
        real_len = end_t - t
        sl = slice(row_idx, row_idx + real_len)

        center_bin = ACTION.num_bins_per_side  # 24 — index of the zero-movement bin
        dx_chunk = np.full(C, center_bin, dtype=np.int64)
        dy_chunk = np.full(C, center_bin, dtype=np.int64)
        mouse_chunk = np.zeros(C, dtype=np.float32)
        keys_chunk = np.zeros((C, NUM_KEYS), dtype=np.float32)
        pad_mask = np.ones(C, dtype=np.float32)  # 1=padded, 0=real

        dx_chunk[:real_len] = _delta_to_bin(self.action_dx[sl])
        dy_chunk[:real_len] = _delta_to_bin(self.action_dy[sl])
        mouse_chunk[:real_len] = self.action_mouse_left[sl].astype(np.float32)
        keys_chunk[:real_len] = self.action_keys_held[sl].astype(np.float32)
        pad_mask[:real_len] = 0.0

        dx_chunk = torch.from_numpy(dx_chunk)
        dy_chunk = torch.from_numpy(dy_chunk)
        mouse_chunk = torch.from_numpy(mouse_chunk)
        keys_chunk = torch.from_numpy(keys_chunk)
        pad_mask = torch.from_numpy(pad_mask)

        return (
            obs,
            proprio,
            input_ids,
            attention_mask,
            dx_chunk,
            dy_chunk,
            mouse_chunk,
            keys_chunk,
            pad_mask,
        )


class EpisodeSequentialSampler(Sampler[int]):
    """Shuffle episode order each epoch, but iterate timesteps within each
    episode sequentially. This produces near-sequential memmap reads
    (kernel readahead works) while still randomizing training order.
    """

    def __init__(self, dataset: ChunkDataset, seed: int = 0) -> None:
        self.episode_ranges = dataset.episode_ranges
        self.total = len(dataset)
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        ep_order = rng.permutation(len(self.episode_ranges))
        for ep_idx in ep_order:
            start, length = self.episode_ranges[ep_idx]
            yield from range(start, start + length)

    def __len__(self) -> int:
        return self.total

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


# ---------------------------------------------------------------------------
# Loss computation (shared by train and val)
# ---------------------------------------------------------------------------

# Loss weights (Exp 5 doesn't define these in config — use same as Exp 3)
LOSS_WEIGHT_EV = 1.0
LOSS_WEIGHT_MOUSE = 5.0
LOSS_WEIGHT_KEYS = 5.0
LOSS_WEIGHT_PAD = 1.0


def compute_losses(
    out: dict[str, torch.Tensor],
    dx_gt: torch.Tensor,  # (B, chunk) int64 bin indices
    dy_gt: torch.Tensor,  # (B, chunk) int64 bin indices
    mouse_gt: torch.Tensor,  # (B, chunk) float32
    keys_gt: torch.Tensor,  # (B, chunk, 53) float32
    pad_gt: torch.Tensor,  # (B, chunk) float32
    focal_loss_fn: FocalBCELoss,
    _soft_targets: torch.Tensor | None = None,
    _bin_centers: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute all loss components from model output and ground truth.

    Returns (total_loss, head_losses) where head_losses has keys:
    dx, dy, mouse, keys, pad, ev (raw, before weighting).
    dx/dy use Gaussian-smoothed soft CE + expected-value L1 loss.
    Keys use focal BCE instead of regular BCE.
    """
    mask = 1.0 - pad_gt
    mask_sum = mask.sum().clamp(min=1)
    B, C_chunk = dx_gt.shape
    mask_flat = mask.reshape(B * C_chunk)
    mask_flat_sum = mask_flat.sum().clamp(min=1)

    # Lazy-init soft targets and bin centers on correct device
    if _soft_targets is None:
        _soft_targets = torch.from_numpy(SOFT_BIN_TARGETS).to(dx_gt.device)
    if _bin_centers is None:
        _bin_centers = torch.from_numpy(BIN_CENTERS).to(dx_gt.device)

    # dx/dy: Gaussian-smoothed soft cross-entropy
    dx_logits_flat = out["dx_logits"].reshape(B * C_chunk, -1)  # (N, 49)
    dy_logits_flat = out["dy_logits"].reshape(B * C_chunk, -1)
    dx_gt_flat = dx_gt.reshape(B * C_chunk)
    dy_gt_flat = dy_gt.reshape(B * C_chunk)

    dx_soft = _soft_targets[dx_gt_flat]  # (N, 49) soft target distributions
    dy_soft = _soft_targets[dy_gt_flat]
    dx_log_probs = F.log_softmax(dx_logits_flat, dim=-1)
    dy_log_probs = F.log_softmax(dy_logits_flat, dim=-1)
    loss_dx = (
        -(dx_soft * dx_log_probs).sum(dim=-1) * mask_flat
    ).sum() / mask_flat_sum
    loss_dy = (
        -(dy_soft * dy_log_probs).sum(dim=-1) * mask_flat
    ).sum() / mask_flat_sum

    # dx/dy: expected-value L1 loss (pixel accuracy)
    dx_probs = F.softmax(dx_logits_flat, dim=-1)
    dy_probs = F.softmax(dy_logits_flat, dim=-1)
    dx_ev = (dx_probs * _bin_centers).sum(dim=-1)  # predicted px
    dy_ev = (dy_probs * _bin_centers).sum(dim=-1)
    dx_gt_px = _bin_centers[dx_gt_flat]  # ground truth px
    dy_gt_px = _bin_centers[dy_gt_flat]
    loss_ev = (
        (F.l1_loss(dx_ev, dx_gt_px, reduction="none") * mask_flat).sum()
        + (F.l1_loss(dy_ev, dy_gt_px, reduction="none") * mask_flat).sum()
    ) / mask_flat_sum

    # Masked BCE for mouse_left
    loss_mouse = (
        F.binary_cross_entropy_with_logits(
            out["mouse_left"], mouse_gt, reduction="none"
        )
        * mask
    ).sum() / mask_sum

    # Masked focal BCE for 53 independent key sigmoids
    keys_logits = out["keys_held"]  # (B, chunk, 53)
    # Compute focal BCE per element, then mask
    p = torch.sigmoid(keys_logits)
    ce_per = F.binary_cross_entropy_with_logits(
        keys_logits, keys_gt, reduction="none"
    )  # (B, chunk, 53)
    p_t = p * keys_gt + (1.0 - p) * (1.0 - keys_gt)
    focal_weight = (1.0 - p_t) ** focal_loss_fn.gamma
    alpha_t = (
        focal_loss_fn.alpha * keys_gt
        + (1.0 - focal_loss_fn.alpha) * (1.0 - keys_gt)
    )
    focal_per = alpha_t * focal_weight * ce_per  # (B, chunk, 53)
    focal_summed = focal_per.sum(dim=-1)  # (B, chunk)
    loss_keys = (focal_summed * mask).sum() / mask_sum

    # Unmasked BCE for pad prediction
    loss_pad = F.binary_cross_entropy_with_logits(out["pad_logits"], pad_gt)

    total_loss = (
        loss_dx
        + loss_dy
        + LOSS_WEIGHT_EV * loss_ev
        + LOSS_WEIGHT_MOUSE * loss_mouse
        + LOSS_WEIGHT_KEYS * loss_keys
        + LOSS_WEIGHT_PAD * loss_pad
    )

    head_losses = {
        "dx": loss_dx,
        "dy": loss_dy,
        "ev": loss_ev,
        "mouse": loss_mouse,
        "keys": loss_keys,
        "pad": loss_pad,
    }
    return total_loss, head_losses


# ---------------------------------------------------------------------------
# Epoch 1 diagnostics
# ---------------------------------------------------------------------------


def print_epoch1_diagnostics(
    timings: dict[str, float],
    n_batches: int,
    batch_size: int,
    train_total: int,
    use_cuda: bool,
    grad_norms: list[float],
    model: torch.nn.Module | None = None,
) -> None:
    """Print comprehensive training diagnostics after epoch 1."""
    t_wall = timings["wall"]
    samples_per_sec = train_total / t_wall

    print(f"\n{'='*70}")
    print(f"  EPOCH 1 TRAINING DIAGNOSTICS")
    print(f"{'='*70}")
    print(
        f"  Wall time:         {t_wall:.1f}s "
        f"({n_batches} batches x {batch_size} samples)"
    )
    print(f"  Throughput:        {samples_per_sec:.0f} samples/sec")
    print()

    phases = ["data", "xfer", "fwd", "loss", "bwd", "opt"]
    labels = {
        "data": "DataLoader wait",
        "xfer": "CPU->GPU transfer",
        "fwd": "Forward pass",
        "loss": "Loss computation",
        "bwd": "Backward pass",
        "opt": "Optimizer step",
    }

    print(f"  --- Time breakdown (training only, excludes validation) ---")
    for p in phases:
        t = timings[p]
        suffix = "  <- data loading bottleneck?" if p == "data" else ""
        print(f"  {labels[p]:20s} {t:6.1f}s ({t/t_wall*100:4.1f}%){suffix}")

    t_gpu = timings["fwd"] + timings["loss"] + timings["bwd"]
    t_accounted = sum(timings[p] for p in phases)
    t_overhead = t_wall - t_accounted
    print(
        f"  {'Other overhead':20s} {t_overhead:6.1f}s "
        f"({t_overhead/t_wall*100:4.1f}%)"
    )
    print(
        f"  {'GPU compute total':20s} {t_gpu:6.1f}s "
        f"({t_gpu/t_wall*100:4.1f}%)  <- GPU utilization"
    )
    print()

    print(f"  --- Per-batch averages ---")
    for p in phases:
        print(f"  {labels[p]:20s} {timings[p]/n_batches*1000:6.1f}ms/batch")
    print()

    if use_cuda:
        peak = torch.cuda.max_memory_allocated() / 1024**3
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  --- VRAM ---")
        print(
            f"  Peak allocated:  {peak:.1f}GB / {total_vram:.1f}GB "
            f"({peak/total_vram*100:.0f}%)"
        )
        print(f"  Headroom:        {total_vram-peak:.1f}GB")
        vram_per_sample = (peak - 0.9) / batch_size
        max_batch = int((total_vram * 0.9 - 0.9) / vram_per_sample)
        print(f"  Est. VRAM/sample: {vram_per_sample*1024:.1f}MB")
        print(f"  Est. max batch:   ~{max_batch} (at 90% VRAM)")
        print()

    if grad_norms:
        gn = np.array(grad_norms)
        print(f"  --- Gradient norms (sampled every 10 batches) ---")
        print(
            f"  Mean: {gn.mean():.2f}  Std: {gn.std():.2f}  "
            f"Min: {gn.min():.2f}  Max: {gn.max():.2f}"
        )
        if gn.max() > 100:
            print(
                f"  WARNING: Large gradient norms detected "
                f"-- consider gradient clipping"
            )

    print(f"  --- RAM ---")
    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None
    if psutil is not None:
        ram = psutil.virtual_memory()
        print(
            f"  Used: {ram.used/1024**3:.1f}GB / "
            f"{ram.total/1024**3:.1f}GB ({ram.percent}%)"
        )
    else:
        print("  psutil not installed; RAM diagnostics skipped")
    disk = shutil.disk_usage("/")
    print(f"  --- Disk ---")
    print(
        f"  Used: {disk.used/1024**3:.1f}GB / {disk.total/1024**3:.1f}GB "
        f"({disk.free/1024**3:.1f}GB free)"
    )

    # Per-head gradient norms
    if model is not None and hasattr(model, "_diag_head_grad_norms"):
        print()
        print(f"  --- Per-component gradient norms (sampled every 10 batches) ---")
        for group_name, norms in model._diag_head_grad_norms.items():
            if norms:
                gn = np.array(norms)
                print(
                    f"  {group_name:18s}  "
                    f"mean={gn.mean():.4f}  std={gn.std():.4f}  "
                    f"min={gn.min():.4f}  max={gn.max():.4f}"
                )
        # Check text encoder gradient flow
        te_norms = model._diag_head_grad_norms.get("text_encoder", [])
        if te_norms and np.mean(te_norms) < 1e-6:
            print(
                f"  WARNING: Text encoder gradients near zero — "
                f"end-to-end gradient flow may be broken!"
            )
        del model._diag_head_grad_norms

    # Key press accuracy
    if model is not None and hasattr(model, "_diag_key_stats"):
        ks = model._diag_key_stats
        tp, fp, fn, tn = ks["tp"], ks["fp"], ks["fn"], ks["tn"]
        total_pos = tp + fn
        total_neg = tn + fp
        print()
        print(f"  --- Key press accuracy (epoch 1) ---")
        print(
            f"  Positive samples:  {total_pos:,} "
            f"({total_pos/(total_pos+total_neg)*100:.2f}% of all key-frames)"
        )
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
        if recall < 0.01:
            print(
                f"  WARNING: Model predicting near-zero key presses "
                f"(recall={recall:.4f}). Focal BCE may need tuning."
            )
        del model._diag_key_stats

    print(f"{'='*70}\n", flush=True)


# ---------------------------------------------------------------------------
# Train / val epoch helpers
# ---------------------------------------------------------------------------


def _run_train_epoch(
    model,
    train_loader,
    optimizer,
    scaler,
    focal_loss_fn,
    device,
    use_cuda,
    use_amp,
    amp_dtype,
    batch_size,
    diag,
    t0,
) -> tuple[float, dict[str, float], int]:
    """Run one training epoch. Returns (avg_loss, head_sums, total_samples)."""
    model.train()
    loss_sum = 0.0
    head_sums = {
        "dx": 0.0,
        "dy": 0.0,
        "ev": 0.0,
        "mouse": 0.0,
        "keys": 0.0,
        "pad": 0.0,
    }
    total = 0
    n_batches = len(train_loader)

    if diag:
        timings = {
            "data": 0.0,
            "xfer": 0.0,
            "fwd": 0.0,
            "loss": 0.0,
            "bwd": 0.0,
            "opt": 0.0,
        }
        grad_norms: list[float] = []

    t_data_start = time.perf_counter()

    for batch_idx, batch in enumerate(train_loader):
        if diag:
            timings["data"] += time.perf_counter() - t_data_start

        (
            obs,
            proprio,
            input_ids,
            attention_mask,
            dx_gt,
            dy_gt,
            mouse_gt,
            keys_gt,
            pad_gt,
        ) = batch

        # --- Transfer to device ---
        if diag:
            t_mark = time.perf_counter()
        obs = obs.to(device, non_blocking=True)
        proprio = proprio.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        dx_gt = dx_gt.to(device, non_blocking=True)
        dy_gt = dy_gt.to(device, non_blocking=True)
        mouse_gt = mouse_gt.to(device, non_blocking=True)
        keys_gt = keys_gt.to(device, non_blocking=True)
        pad_gt = pad_gt.to(device, non_blocking=True)
        if diag:
            if use_cuda:
                torch.cuda.synchronize()
            timings["xfer"] += time.perf_counter() - t_mark

        # --- Forward + loss ---
        with torch.amp.autocast(
            device_type=device.split(":")[0], dtype=amp_dtype, enabled=use_amp
        ):
            if diag:
                if use_cuda:
                    torch.cuda.synchronize()
                t_mark = time.perf_counter()
            out = model(obs, proprio, input_ids, attention_mask)
            if diag:
                if use_cuda:
                    torch.cuda.synchronize()
                timings["fwd"] += time.perf_counter() - t_mark
                t_mark = time.perf_counter()

            total_loss, head_losses = compute_losses(
                out,
                dx_gt,
                dy_gt,
                mouse_gt,
                keys_gt,
                pad_gt,
                focal_loss_fn,
            )

        if diag:
            if use_cuda:
                torch.cuda.synchronize()
            timings["loss"] += time.perf_counter() - t_mark
            t_mark = time.perf_counter()

        # --- Backward ---
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=TRAIN.grad_clip_norm
        )

        if diag:
            if use_cuda:
                torch.cuda.synchronize()
            timings["bwd"] += time.perf_counter() - t_mark

            if batch_idx % 10 == 0:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.float().norm(2).item() ** 2
                grad_norms.append(total_norm**0.5)

                # Per-head gradient norms (sampled alongside total)
                if not hasattr(model, "_diag_head_grad_norms"):
                    model._diag_head_grad_norms = {
                        "backbone": [], "text_encoder": [],
                        "encoder": [], "decoder": [],
                        "head_dx": [], "head_dy": [],
                        "head_mouse": [], "head_keys": [], "head_pad": [],
                    }
                _mod = model._orig_mod if hasattr(model, "_orig_mod") else model
                for group_name, prefix in [
                    ("backbone", "backbone."), ("text_encoder", "text_encoder."),
                    ("encoder", "encoder."), ("decoder", "decoder."),
                    ("head_dx", "head_dx."), ("head_dy", "head_dy."),
                    ("head_mouse", "head_mouse."), ("head_keys", "head_keys."),
                    ("head_pad", "head_pad."),
                ]:
                    gn2 = 0.0
                    for n, p in _mod.named_parameters():
                        if n.startswith(prefix) and p.grad is not None:
                            gn2 += p.grad.data.float().norm(2).item() ** 2
                    model._diag_head_grad_norms[group_name].append(gn2**0.5)

            # Key press accuracy tracking (every batch during epoch 1)
            if not hasattr(model, "_diag_key_stats"):
                model._diag_key_stats = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            with torch.no_grad():
                key_preds = (torch.sigmoid(out["keys_held"]) > 0.5).float()
                real_mask = (1.0 - pad_gt).unsqueeze(-1)  # (B, chunk, 1)
                key_pred_masked = key_preds * real_mask
                key_gt_masked = keys_gt * real_mask
                model._diag_key_stats["tp"] += int(
                    ((key_pred_masked == 1) & (key_gt_masked == 1)).sum().item()
                )
                model._diag_key_stats["fp"] += int(
                    ((key_pred_masked == 1) & (key_gt_masked == 0)).sum().item()
                )
                model._diag_key_stats["fn"] += int(
                    ((key_pred_masked == 0) & (key_gt_masked == 1)).sum().item()
                )
                model._diag_key_stats["tn"] += int(
                    ((key_pred_masked == 0) & (key_gt_masked == 0)).sum().item()
                )

            t_mark = time.perf_counter()

        # --- Optimizer step ---
        scaler.step(optimizer)
        scaler.update()

        if diag:
            if use_cuda:
                torch.cuda.synchronize()
            timings["opt"] += time.perf_counter() - t_mark

        # --- Accumulate ---
        bs = obs.size(0)
        loss_sum += total_loss.item() * bs
        for h, v in head_losses.items():
            head_sums[h] += v.item() * bs
        total += bs

        # --- Periodic progress (epoch 0) ---
        if diag and (batch_idx == 0 or (batch_idx + 1) % 50 == 0):
            elapsed = time.perf_counter() - t0
            vram_str = ""
            if use_cuda:
                alloc = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                peak = torch.cuda.max_memory_allocated() / 1024**3
                vram_str = (
                    f" | VRAM: {alloc:.1f}/{reserved:.1f}/{peak:.1f}GB"
                    f" (alloc/reserved/peak)"
                )
            print(
                f"    batch {batch_idx+1}/{n_batches} | "
                f"loss={total_loss.item():.4f} | "
                f"{elapsed:.1f}s elapsed{vram_str}",
                flush=True,
            )

        if diag:
            t_data_start = time.perf_counter()

    if diag:
        timings["wall"] = time.perf_counter() - t0
        print_epoch1_diagnostics(
            timings, n_batches, batch_size, total, use_cuda, grad_norms, model
        )

    return loss_sum / max(total, 1), head_sums, total


def _run_val_epoch(
    model,
    val_loader,
    focal_loss_fn,
    device,
    use_cuda,
    use_amp,
    amp_dtype,
) -> tuple[float, dict[str, float], int]:
    """Run one validation epoch. Returns (avg_loss, head_sums, total_samples)."""
    model.train(False)
    loss_sum = 0.0
    head_sums = {
        "dx": 0.0,
        "dy": 0.0,
        "ev": 0.0,
        "mouse": 0.0,
        "keys": 0.0,
        "pad": 0.0,
    }
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            (
                obs,
                proprio,
                input_ids,
                attention_mask,
                dx_gt,
                dy_gt,
                mouse_gt,
                keys_gt,
                pad_gt,
            ) = batch
            obs = obs.to(device, non_blocking=True)
            proprio = proprio.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            dx_gt = dx_gt.to(device, non_blocking=True)
            dy_gt = dy_gt.to(device, non_blocking=True)
            mouse_gt = mouse_gt.to(device, non_blocking=True)
            keys_gt = keys_gt.to(device, non_blocking=True)
            pad_gt = pad_gt.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.split(":")[0],
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                out = model(obs, proprio, input_ids, attention_mask)
                total_loss, head_losses = compute_losses(
                    out,
                    dx_gt,
                    dy_gt,
                    mouse_gt,
                    keys_gt,
                    pad_gt,
                    focal_loss_fn,
                )

            bs = obs.size(0)
            loss_sum += total_loss.item() * bs
            for h, v in head_losses.items():
                head_sums[h] += v.item() * bs
            total += bs

    return loss_sum / max(total, 1), head_sums, total


def _upload_checkpoint_async(hf_upload_repo, path, repo_prefix, val_loss):
    """Upload best.pt to HF Hub in a background thread."""
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
                path_or_fileobj=path,
                path_in_repo=f"{repo_prefix}/best.pt",
                repo_id=hf_upload_repo,
                repo_type="model",
            )
            print(f"    Uploaded best.pt (val={_val:.4f})", flush=True)
        except Exception as e:
            print(f"    Upload failed: {e}", flush=True)

    threading.Thread(target=_upload, daemon=True).start()


# ---------------------------------------------------------------------------
# Data loading and pre-processing
# ---------------------------------------------------------------------------


def pretokenize_instructions(
    ds,
    tokenizer,
    token_id_map: dict[int, int],
    max_length: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-tokenize all instruction strings and cache as numpy arrays.

    Returns:
        (token_ids, attention_masks) each of shape (N, max_length) int64.
    """
    n = len(ds)
    token_ids = np.zeros((n, max_length), dtype=np.int64)
    attention_masks = np.zeros((n, max_length), dtype=np.int64)

    t0 = time.perf_counter()
    instructions = ds["instruction"]  # list[str]

    for i, text in enumerate(instructions):
        ids, mask = tokenize_instruction(text, tokenizer, token_id_map, max_length)
        token_ids[i] = ids[0].numpy()
        attention_masks[i] = mask[0].numpy()

    elapsed = time.perf_counter() - t0
    print(
        f"Pre-tokenized {n} instructions in {elapsed:.1f}s "
        f"({n / elapsed:.0f} instr/s, "
        f"{token_ids.nbytes / 1024**2:.1f}MB)",
        flush=True,
    )
    return token_ids, attention_masks


def load_dataset_splits(
    hf_data_repo: str | None,
    data_dir: str | None,
    base: str,
    checkpoint_dir: str,
    chunk_size: int,
    batch_size: int,
    max_episodes: int | None,
    num_workers: int,
    use_cuda: bool,
    tokenizer,
    token_id_map: dict[int, int],
    ds_preloaded=None,
) -> tuple[DataLoader, EpisodeSequentialSampler, DataLoader]:
    """Load dataset, pre-decode images, build DataLoaders.

    Returns (train_loader, train_sampler, val_loader).
    If ds_preloaded is provided, use it instead of loading from disk/hub.
    """
    if ds_preloaded is not None:
        ds = ds_preloaded
        print(f"Using preloaded dataset: {len(ds)} rows")
    elif hf_data_repo:
        from datasets import load_dataset as _load_dataset

        print(f"Loading dataset from HF Hub: {hf_data_repo} ...")
        ds = _load_dataset(hf_data_repo, split="train")
    else:
        if data_dir is None:
            data_dir = os.path.join(base, "data")
        from datasets import load_from_disk

        print(f"Loading dataset from {data_dir} ...")
        ds = load_from_disk(data_dir)
    print(f"Dataset: {ds}", flush=True)

    # Pre-extract scalar columns to numpy
    t_extract = time.perf_counter()
    ep_ids_np = np.array(ds["episode_id"], dtype=np.int32)
    action_arrays = {
        "action_dx": np.array(ds["action_dx"], dtype=np.float32),
        "action_dy": np.array(ds["action_dy"], dtype=np.float32),
        "action_mouse_left": np.array(ds["action_mouse_left"], dtype=np.int8),
        "action_keys_held": np.array(ds["action_keys_held"], dtype=np.int8),
        "proprio": np.array(ds["proprio"], dtype=np.float32),
    }
    extract_mb = (
        ep_ids_np.nbytes + sum(a.nbytes for a in action_arrays.values())
    ) / 1024 / 1024
    print(
        f"Scalar columns extracted to numpy in "
        f"{time.perf_counter() - t_extract:.2f}s ({extract_mb:.1f}MB)",
        flush=True,
    )

    # Pre-tokenize instructions
    token_ids, attention_masks = pretokenize_instructions(
        ds, tokenizer, token_id_map
    )

    # Pre-decode images to memmap
    images = predecode_images(ds)

    # Build episode offset table and split
    episode_offsets = build_episode_offsets(ep_ids_np)
    all_episode_ids = sorted(episode_offsets.keys())
    if max_episodes is not None and max_episodes < len(all_episode_ids):
        all_episode_ids = all_episode_ids[:max_episodes]

    print(f"Using {len(all_episode_ids)} episodes")

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(all_episode_ids))
    val_count = max(1, int(len(all_episode_ids) * TRAIN.val_fraction))
    val_ids = set(all_episode_ids[i] for i in indices[:val_count])
    train_ids = set(all_episode_ids[i] for i in indices[val_count:])

    print(f"Train: {len(train_ids)} episodes, Val: {len(val_ids)} episodes")
    print(f"DataLoader workers: {num_workers}")

    train_dataset = ChunkDataset(
        images,
        episode_offsets,
        train_ids,
        chunk_size,
        action_arrays,
        token_ids,
        attention_masks,
    )
    val_dataset = ChunkDataset(
        images,
        episode_offsets,
        val_ids,
        chunk_size,
        action_arrays,
        token_ids,
        attention_masks,
    )
    print(
        f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}",
        flush=True,
    )

    train_sampler = EpisodeSequentialSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=use_cuda,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=use_cuda,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return train_loader, train_sampler, val_loader


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(
    chunk_size: int = CHUNK.default_chunk_size,
    batch_size: int = TRAIN.batch_size,
    lr: float = TRAIN.lr,
    max_episodes: int | None = None,
    num_workers: int | None = None,
    data_dir: str | None = None,
    checkpoint_dir: str | None = None,
    device: str = "cpu",
    max_epochs: int | None = None,
    hf_upload_repo: str | None = None,
    hf_data_repo: str | None = None,
) -> None:
    from datetime import datetime, timezone

    base = os.path.dirname(__file__)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    repo_prefix = f"mini_editor_chunk{chunk_size}/{run_id}"

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(
            base, "checkpoints", f"chunk{chunk_size}"
        )
    os.makedirs(checkpoint_dir, exist_ok=True)

    use_cuda = device.startswith("cuda")
    if num_workers is None:
        num_workers = 0

    # --- Build text encoder ---
    # Collect corpus sentences from the dataset to ensure vocab coverage
    print("Building text encoder ...")
    t_te = time.perf_counter()

    # We need to load the dataset briefly to extract unique initial_text values
    from datasets import load_from_disk

    if data_dir is None:
        _data_dir = os.path.join(base, "data")
    else:
        _data_dir = data_dir

    # Load dataset: from HF Hub (parquet) or local (Arrow)
    if hf_data_repo:
        from datasets import load_dataset as _load_dataset

        print(f"Loading data from HF Hub: {hf_data_repo} ...")
        ds_temp = _load_dataset(hf_data_repo, split="train")
        print(f"Loaded {len(ds_temp)} rows from Hub.")
    else:
        ds_temp = load_from_disk(_data_dir)
    corpus_sentences = list(set(ds_temp["initial_text"]))
    print(f"Corpus: {len(corpus_sentences)} unique initial_text passages")

    text_encoder, tokenizer, token_id_map = build_text_encoder(corpus_sentences)
    del corpus_sentences
    print(
        f"Text encoder built in {time.perf_counter() - t_te:.1f}s",
        flush=True,
    )

    # --- Data ---
    # Pass the already-loaded dataset to avoid loading it a second time
    train_loader, train_sampler, val_loader = load_dataset_splits(
        None,  # don't re-download
        data_dir,
        base,
        checkpoint_dir,
        chunk_size,
        batch_size,
        max_episodes,
        num_workers,
        use_cuda,
        tokenizer,
        token_id_map,
        ds_preloaded=ds_temp,
    )
    del ds_temp

    # --- Model ---
    model = ACT(
        chunk_size=chunk_size,
        text_encoder=text_encoder,
    ).to(device)
    if use_cuda:
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model)
        print("Model compiled with torch.compile (TF32 enabled)", flush=True)
    print(
        f"Model parameters: {count_parameters(model, False):,} total, "
        f"{count_parameters(model, True):,} trainable"
    )

    # --- Focal BCE loss ---
    focal_loss_fn = FocalBCELoss(gamma=FOCAL.gamma, alpha=FOCAL.alpha)

    # --- Optimizer ---
    backbone_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and n.startswith("backbone.")
    ]
    non_backbone_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith("backbone.")
    ]

    backbone_lr = lr * 0.1  # Lower LR for pretrained backbone
    optimizer = torch.optim.AdamW(
        [
            {"params": non_backbone_params, "lr": lr},
            {"params": backbone_params, "lr": backbone_lr},
        ],
        weight_decay=TRAIN.weight_decay,
    )

    epochs = max_epochs if max_epochs is not None else TRAIN.epochs

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        total_iters=TRAIN.warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - TRAIN.warmup_epochs),
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[TRAIN.warmup_epochs],
    )

    # --- AMP ---
    use_amp = TRAIN.use_amp and use_cuda
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        scaler = torch.amp.GradScaler(enabled=False)
    elif use_amp:
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler(enabled=True)
    else:
        amp_dtype = torch.float32
        scaler = torch.amp.GradScaler(enabled=False)

    # --- Print config ---
    print(
        f"AMP: {amp_dtype if use_amp else 'disabled'} "
        f"(GradScaler: {scaler.is_enabled()})"
    )
    print(f"Device: {device}, Chunk size: {chunk_size}")
    print(f"LR: {lr} (backbone: {backbone_lr})")
    if use_cuda:
        total_vram = (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
        )
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {total_vram:.1f}GB")
    disk = shutil.disk_usage("/")
    print(
        f"Disk: {disk.used/1024**3:.1f}GB used / {disk.total/1024**3:.1f}GB "
        f"total ({disk.free/1024**3:.1f}GB free)",
        flush=True,
    )

    # --- Training loop ---
    head_names = ["dx", "dy", "ev", "mouse", "keys", "pad"]
    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        **{f"train_{h}": [] for h in head_names},
        **{f"val_{h}": [] for h in head_names},
    }
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        t0 = time.perf_counter()
        diag = epoch == 0
        train_sampler.set_epoch(epoch)

        # Train
        train_loss, head_sums_train, train_total = _run_train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            focal_loss_fn,
            device,
            use_cuda,
            use_amp,
            amp_dtype,
            batch_size,
            diag,
            t0,
        )
        scheduler.step()

        # Validate
        val_loss, head_sums_val, val_total = _run_val_epoch(
            model,
            val_loader,
            focal_loss_fn,
            device,
            use_cuda,
            use_amp,
            amp_dtype,
        )

        elapsed = time.perf_counter() - t0

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for h in head_names:
            history[f"train_{h}"].append(
                head_sums_train[h] / max(train_total, 1)
            )
            history[f"val_{h}"].append(
                head_sums_val[h] / max(val_total, 1)
            )

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_pt_path = os.path.join(checkpoint_dir, "best.pt")
            state = {
                k.removeprefix("_orig_mod."): v
                for k, v in model.state_dict().items()
            }
            torch.save(state, best_pt_path)
            if hf_upload_repo:
                _upload_checkpoint_async(
                    hf_upload_repo, best_pt_path, repo_prefix, val_loss
                )
        else:
            epochs_without_improvement += 1

        # Epoch log
        vh = {h: head_sums_val[h] / max(val_total, 1) for h in head_names}
        head_str = " ".join(f"{h}={vh[h]:.3f}" for h in head_names)
        eta_min = (epochs - epoch - 1) * elapsed / 60
        print(
            f"  Epoch {epoch+1:3d}/{epochs} | "
            f"train={train_loss:.4f} val={val_loss:.4f} | "
            f"val_heads: {head_str} | "
            f"lr={scheduler.get_last_lr()[0]:.6f} | "
            f"{elapsed:.1f}s/ep | ETA ~{eta_min:.0f}min",
            flush=True,
        )

        # Early stopping
        if epochs_without_improvement >= TRAIN.early_stop_patience:
            print(
                f"  Early stopping at epoch {epoch+1} "
                f"(no improvement for {TRAIN.early_stop_patience} epochs)"
            )
            break

    # Save final
    state = {
        k.removeprefix("_orig_mod."): v
        for k, v in model.state_dict().items()
    }
    torch.save(state, os.path.join(checkpoint_dir, "final.pt"))
    torch.save(history, os.path.join(checkpoint_dir, "history.pt"))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")

    if hf_upload_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(hf_upload_repo, repo_type="model", exist_ok=True)
        print(f"Uploading checkpoints to {hf_upload_repo}/{repo_prefix} ...")
        api.upload_folder(
            repo_id=hf_upload_repo,
            repo_type="model",
            folder_path=checkpoint_dir,
            path_in_repo=repo_prefix,
        )
        print("Upload complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ACT via behavior cloning (Exp 5: Mini Text Editor)"
    )
    parser.add_argument("--epochs", type=int, default=TRAIN.epochs)
    parser.add_argument("--batch-size", type=int, default=TRAIN.batch_size)
    parser.add_argument("--lr", type=float, default=TRAIN.lr)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK.default_chunk_size
    )
    parser.add_argument("--data-dir", type=str, default=DATA.output_dir)
    parser.add_argument(
        "--hf-data-repo",
        type=str,
        default=None,
        help="Load from HF Hub instead of local",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Limit episodes for debugging",
    )
    parser.add_argument(
        "--hf-upload-repo",
        type=str,
        default=None,
        help="Upload checkpoint to HF Hub",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default: 0 since images are pre-decoded)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    train(
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        lr=args.lr,
        max_episodes=args.num_episodes,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        max_epochs=args.epochs,
        hf_upload_repo=args.hf_upload_repo,
        hf_data_repo=args.hf_data_repo,
    )
