"""Behavior cloning training loop for the ACT policy.

Loads expert demonstrations from HF datasets (parquet), trains with:
- Masked L1 loss on dx, dy (continuous pixel deltas)
- Masked BCE loss on click (binary)
- Masked CE loss on key (28-class classification)
- Unmasked BCE loss on pad prediction
- KL divergence from CVAE encoder (annealed)
Saves checkpoints + training history.
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

from experiments.act_drag_label.config import ACTION, CHUNK, ENV, TRAIN
from experiments.act_drag_label.model import ACT, count_parameters


# ---------------------------------------------------------------------------
# Episode offset table — shared by ChunkDataset and external callers
# ---------------------------------------------------------------------------

def build_episode_offsets(
    ep_ids: np.ndarray,
) -> dict[int, tuple[int, int]]:
    """Return {episode_id: (start_row, length)} from a sorted episode_id array.

    Assumes the array is sorted by (episode_id, timestep) — which is the
    natural output order from generate_data.py / Dataset.from_generator().
    """
    # Find indices where episode_id changes
    changes = np.where(np.diff(ep_ids) != 0)[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(ep_ids)]])
    offsets: dict[int, tuple[int, int]] = {}
    for s, e in zip(starts, ends):
        offsets[int(ep_ids[s])] = (int(s), int(e - s))
    return offsets


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChunkDataset(Dataset):
    """Dataset that samples (obs, proprio, action_chunk, ...) tuples.

    Every timestep t in every episode is a valid start index.
    Action chunks are zero-padded when near the episode end.

    All columns (including images) are pre-extracted to numpy arrays
    so __getitem__ is pure array indexing with no PNG decode.
    """

    def __init__(
        self,
        images: np.ndarray,
        episode_offsets: dict[int, tuple[int, int]],
        episode_ids: set[int],
        chunk_size: int,
        action_arrays: dict[str, np.ndarray],
    ) -> None:
        self.images = images
        self.chunk_size = chunk_size
        self.episode_offsets = {
            eid: episode_offsets[eid] for eid in episode_ids
            if eid in episode_offsets
        }
        self.cursor_x = action_arrays["cursor_x"]
        self.cursor_y = action_arrays["cursor_y"]
        self.state_click = action_arrays["state_click"]
        self.state_key = action_arrays["state_key"]
        self.action_dx = action_arrays["action_dx"]
        self.action_dy = action_arrays["action_dy"]
        self.action_click = action_arrays["action_click"]
        self.action_key = action_arrays["action_key"]

        # Build flat index: every timestep in the selected episodes
        self.index: list[tuple[int, int]] = []
        for eid in sorted(self.episode_offsets):
            _, length = self.episode_offsets[eid]
            for t in range(length):
                self.index.append((eid, t))

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
        eid, t = self.index[idx]
        start_row, ep_len = self.episode_offsets[eid]
        C = self.chunk_size
        row_idx = start_row + t

        # --- Observation at time t (pre-decoded uint8 array) ---
        obs = torch.from_numpy(self.images[row_idx].copy()).permute(2, 0, 1).float() / 255.0

        # --- Proprioception at time t (env state before action, matches eval) ---
        cx_norm = float(self.cursor_x[row_idx]) / ENV.window_size
        cy_norm = float(self.cursor_y[row_idx]) / ENV.window_size
        click_t = float(self.state_click[row_idx])
        key_t = int(self.state_key[row_idx])
        key_onehot = np.zeros(ACTION.num_key_classes, dtype=np.float32)
        key_onehot[key_t] = 1.0
        proprio = np.zeros(31, dtype=np.float32)
        proprio[0] = cx_norm
        proprio[1] = cy_norm
        proprio[2] = click_t
        proprio[3:] = key_onehot
        proprio = torch.from_numpy(proprio)

        # --- Action chunk [t:t+C], zero-padded if near end (numpy slices) ---
        end_t = min(t + C, ep_len)
        real_len = end_t - t
        sl = slice(row_idx, row_idx + real_len)

        dx_chunk = np.zeros(C, dtype=np.float32)
        dy_chunk = np.zeros(C, dtype=np.float32)
        click_chunk = np.zeros(C, dtype=np.float32)
        key_chunk = np.zeros(C, dtype=np.int64)
        pad_mask = np.ones(C, dtype=np.float32)  # 1=padded, 0=real

        dx_chunk[:real_len] = self.action_dx[sl]
        dy_chunk[:real_len] = self.action_dy[sl]
        click_chunk[:real_len] = self.action_click[sl].astype(np.float32)
        key_chunk[:real_len] = self.action_key[sl].astype(np.int64)
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
        actions_cvae[torch.arange(real_len), 3 + key_chunk[:real_len]] = 1.0

        return obs, proprio, actions_cvae, dx_chunk, dy_chunk, click_chunk, key_chunk, pad_mask


def train(
    backbone: str = "resnet18",
    chunk_size: int = CHUNK.default_chunk_size,
    batch_size: int = TRAIN.batch_size,
    max_episodes: int | None = None,
    num_workers: int | None = None,
    data_dir: str | None = None,
    checkpoint_dir: str | None = None,
    device: str = "cpu",
    hf_data_repo: str | None = None,
    hf_upload_repo: str | None = None,
) -> None:
    from datasets import load_dataset, load_from_disk

    base = os.path.dirname(__file__)

    # Load dataset from HF Hub or local disk
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
        checkpoint_dir = os.path.join(
            base, "checkpoints", f"{backbone}_chunk{chunk_size}"
        )
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Pre-extract scalar columns to numpy (avoids per-sample Arrow access)
    t_extract = time.perf_counter()
    ep_ids_np = np.array(ds["episode_id"], dtype=np.int32)
    action_arrays = {
        "cursor_x": np.array(ds["cursor_x"], dtype=np.float32),
        "cursor_y": np.array(ds["cursor_y"], dtype=np.float32),
        "state_click": np.array(ds["state_click"], dtype=np.int8),
        "state_key": np.array(ds["state_key"], dtype=np.int8),
        "action_dx": np.array(ds["action_dx"], dtype=np.float32),
        "action_dy": np.array(ds["action_dy"], dtype=np.float32),
        "action_click": np.array(ds["action_click"], dtype=np.int8),
        "action_key": np.array(ds["action_key"], dtype=np.int8),
    }
    extract_mb = (ep_ids_np.nbytes + sum(a.nbytes for a in action_arrays.values())) / 1024 / 1024
    print(
        f"Scalar columns extracted to numpy in {time.perf_counter() - t_extract:.2f}s "
        f"({extract_mb:.1f}MB)",
        flush=True,
    )

    # Pre-decode all images to disk-backed memmap (eliminates PNG decode during training)
    n_samples = len(ds)
    obs_size = ENV.obs_size
    img_shape = (n_samples, obs_size, obs_size, 3)
    memmap_path = os.path.join(checkpoint_dir, "images_cache.dat")
    t_decode = time.perf_counter()

    if os.path.exists(memmap_path):
        # Check if existing cache matches expected size
        expected_bytes = int(np.prod(img_shape))
        actual_bytes = os.path.getsize(memmap_path)
        if actual_bytes == expected_bytes:
            images = np.memmap(memmap_path, dtype=np.uint8, mode="r", shape=img_shape)
            print(
                f"Image cache loaded from {memmap_path} "
                f"({actual_bytes / 1024**3:.1f}GB)",
                flush=True,
            )
        else:
            os.remove(memmap_path)
            images = None
    else:
        images = None

    if images is None:
        images = np.memmap(memmap_path, dtype=np.uint8, mode="w+", shape=img_shape)
        for i in range(n_samples):
            images[i] = np.array(ds[i]["image"])
            if (i + 1) % 50000 == 0 or i == 0:
                elapsed_decode = time.perf_counter() - t_decode
                print(
                    f"  Pre-decoded {i+1}/{n_samples} images "
                    f"({elapsed_decode:.1f}s elapsed)",
                    flush=True,
                )
        images.flush()
        images_gb = images.nbytes / 1024**3
        print(
            f"Images pre-decoded to {memmap_path} in {time.perf_counter() - t_decode:.1f}s "
            f"({images_gb:.1f}GB, {n_samples / (time.perf_counter() - t_decode):.0f} img/s)",
            flush=True,
        )

    # Build episode offset table
    t_init = time.perf_counter()
    episode_offsets = build_episode_offsets(ep_ids_np)
    print(f"Episode offsets built in {time.perf_counter() - t_init:.2f}s", flush=True)
    all_episode_ids = sorted(episode_offsets.keys())

    if max_episodes is not None and max_episodes < len(all_episode_ids):
        all_episode_ids = all_episode_ids[:max_episodes]

    print(f"Using {len(all_episode_ids)} episodes")

    # Train/val split by episode
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(all_episode_ids))
    val_count = max(1, int(len(all_episode_ids) * TRAIN.val_fraction))
    val_ids = set(all_episode_ids[i] for i in indices[:val_count])
    train_ids = set(all_episode_ids[i] for i in indices[val_count:])

    print(f"Train: {len(train_ids)} episodes, Val: {len(val_ids)} episodes")

    # Default num_workers: 8 for CUDA (overlap data loading with GPU), 0 for MPS/CPU
    if num_workers is None:
        num_workers = 8 if device.startswith("cuda") else 0

    print(f"DataLoader workers: {num_workers}")

    train_dataset = ChunkDataset(images, episode_offsets, train_ids, chunk_size, action_arrays)
    val_dataset = ChunkDataset(images, episode_offsets, val_ids, chunk_size, action_arrays)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}", flush=True)

    use_cuda = device.startswith("cuda")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        pin_memory=use_cuda, prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        pin_memory=use_cuda, prefetch_factor=4 if num_workers > 0 else None,
    )

    # Model
    model = ACT(backbone_name=backbone, chunk_size=chunk_size).to(device)
    if use_cuda:
        torch.set_float32_matmul_precision("high")  # enable TF32 tensor cores
        model = torch.compile(model)
        print("Model compiled with torch.compile (TF32 enabled)", flush=True)
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

    # AMP — prefer bf16 on Ampere+ (L4, A10G, A100), fall back to fp16 on older (T4)
    use_amp = TRAIN.use_amp and device.startswith("cuda")
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        scaler = torch.amp.GradScaler(enabled=False)
    elif use_amp:
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler(enabled=True)
    else:
        amp_dtype = torch.float32
        scaler = torch.amp.GradScaler(enabled=False)

    print(f"AMP: {amp_dtype if use_amp else 'disabled'} (GradScaler: {scaler.is_enabled()})")
    print(f"Device: {device}, Backbone: {backbone}, Chunk size: {chunk_size}")
    if use_cuda:
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {total_vram:.1f}GB")
    import shutil
    disk = shutil.disk_usage("/")
    print(f"Disk: {disk.used/1024**3:.1f}GB used / {disk.total/1024**3:.1f}GB total "
          f"({disk.free/1024**3:.1f}GB free)", flush=True)

    # Training loop
    head_names = ["dx", "dy", "click", "key", "pad", "kl"]
    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        **{f"train_{h}": [] for h in head_names},
        **{f"val_{h}": [] for h in head_names},
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
        head_sums_train = {"dx": 0.0, "dy": 0.0, "click": 0.0, "key": 0.0, "pad": 0.0, "kl": 0.0}
        train_total = 0
        n_batches = len(train_loader)

        # First-epoch diagnostics: per-phase timing accumulators
        diag = epoch == 0
        if diag:
            t_data_total = 0.0   # time waiting for DataLoader
            t_xfer_total = 0.0   # time in .to(device) transfers
            t_fwd_total = 0.0    # time in forward pass
            t_loss_total = 0.0   # time computing losses
            t_bwd_total = 0.0    # time in backward pass
            t_opt_total = 0.0    # time in optimizer step
            grad_norms = []      # gradient L2 norms (sampled)

        t_data_start = time.perf_counter()

        for batch_idx, batch in enumerate(train_loader):
            if diag:
                t_data_total += time.perf_counter() - t_data_start

            (obs, proprio, actions_cvae, dx_gt, dy_gt, click_gt, key_gt, pad_gt) = batch

            if diag:
                t_xfer_start = time.perf_counter()
            obs = obs.to(device, non_blocking=True)
            proprio = proprio.to(device, non_blocking=True)
            actions_cvae = actions_cvae.to(device, non_blocking=True)
            dx_gt = dx_gt.to(device, non_blocking=True)
            dy_gt = dy_gt.to(device, non_blocking=True)
            click_gt = click_gt.to(device, non_blocking=True)
            key_gt = key_gt.to(device, non_blocking=True)
            pad_gt = pad_gt.to(device, non_blocking=True)
            if diag:
                if use_cuda:
                    torch.cuda.synchronize()
                t_xfer_total += time.perf_counter() - t_xfer_start

            with torch.amp.autocast(
                device_type=device.split(":")[0], dtype=amp_dtype, enabled=use_amp
            ):
                if diag:
                    if use_cuda:
                        torch.cuda.synchronize()
                    t_fwd_start = time.perf_counter()
                out = model(obs, proprio, actions_cvae)
                if diag:
                    if use_cuda:
                        torch.cuda.synchronize()
                    t_fwd_total += time.perf_counter() - t_fwd_start

                    t_loss_start = time.perf_counter()

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

            if diag:
                if use_cuda:
                    torch.cuda.synchronize()
                t_loss_total += time.perf_counter() - t_loss_start

                t_bwd_start = time.perf_counter()

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()

            if diag:
                if use_cuda:
                    torch.cuda.synchronize()
                t_bwd_total += time.perf_counter() - t_bwd_start

                # Sample gradient norm every 10 batches
                if batch_idx % 10 == 0:
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.float().norm(2).item() ** 2
                    grad_norms.append(total_norm ** 0.5)

                t_opt_start = time.perf_counter()

            scaler.step(optimizer)
            scaler.update()

            if diag:
                if use_cuda:
                    torch.cuda.synchronize()
                t_opt_total += time.perf_counter() - t_opt_start

            bs = obs.size(0)
            train_loss_sum += total_loss.item() * bs
            head_sums_train["dx"] += loss_dx.item() * bs
            head_sums_train["dy"] += loss_dy.item() * bs
            head_sums_train["click"] += loss_click.item() * bs
            head_sums_train["key"] += loss_key.item() * bs
            head_sums_train["pad"] += loss_pad.item() * bs
            head_sums_train["kl"] += kl.item() * bs
            train_total += bs

            # First epoch: periodic progress with VRAM
            if diag and (batch_idx == 0 or (batch_idx + 1) % 50 == 0):
                elapsed_batch = time.perf_counter() - t0
                vram_str = ""
                if use_cuda:
                    alloc = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    peak = torch.cuda.max_memory_allocated() / 1024**3
                    vram_str = (f" | VRAM: {alloc:.1f}/{reserved:.1f}/{peak:.1f}GB"
                                f" (alloc/reserved/peak)")
                print(
                    f"    batch {batch_idx+1}/{n_batches} | "
                    f"loss={total_loss.item():.4f} | "
                    f"{elapsed_batch:.1f}s elapsed{vram_str}",
                    flush=True,
                )

            if diag:
                t_data_start = time.perf_counter()

        # End of epoch 1: print comprehensive diagnostics
        if diag:
            t_train_wall = time.perf_counter() - t0
            t_accounted = t_data_total + t_xfer_total + t_fwd_total + t_loss_total + t_bwd_total + t_opt_total
            t_overhead = t_train_wall - t_accounted
            samples_per_sec = train_total / t_train_wall

            print(f"\n{'='*70}")
            print(f"  EPOCH 1 TRAINING DIAGNOSTICS")
            print(f"{'='*70}")
            print(f"  Wall time:         {t_train_wall:.1f}s ({n_batches} batches × {batch_size} samples)")
            print(f"  Throughput:        {samples_per_sec:.0f} samples/sec")
            print(f"")
            print(f"  --- Time breakdown (training only, excludes validation) ---")
            print(f"  DataLoader wait:   {t_data_total:6.1f}s ({t_data_total/t_train_wall*100:4.1f}%)  ← data loading bottleneck?")
            print(f"  CPU→GPU transfer:  {t_xfer_total:6.1f}s ({t_xfer_total/t_train_wall*100:4.1f}%)")
            print(f"  Forward pass:      {t_fwd_total:6.1f}s ({t_fwd_total/t_train_wall*100:4.1f}%)")
            print(f"  Loss computation:  {t_loss_total:6.1f}s ({t_loss_total/t_train_wall*100:4.1f}%)")
            print(f"  Backward pass:     {t_bwd_total:6.1f}s ({t_bwd_total/t_train_wall*100:4.1f}%)")
            print(f"  Optimizer step:    {t_opt_total:6.1f}s ({t_opt_total/t_train_wall*100:4.1f}%)")
            print(f"  Other overhead:    {t_overhead:6.1f}s ({t_overhead/t_train_wall*100:4.1f}%)")
            print(f"  GPU compute total: {t_fwd_total+t_loss_total+t_bwd_total:6.1f}s ({(t_fwd_total+t_loss_total+t_bwd_total)/t_train_wall*100:4.1f}%)  ← GPU utilization")
            print(f"")
            print(f"  --- Per-batch averages ---")
            print(f"  DataLoader:  {t_data_total/n_batches*1000:6.1f}ms/batch")
            print(f"  Transfer:    {t_xfer_total/n_batches*1000:6.1f}ms/batch")
            print(f"  Forward:     {t_fwd_total/n_batches*1000:6.1f}ms/batch")
            print(f"  Loss:        {t_loss_total/n_batches*1000:6.1f}ms/batch")
            print(f"  Backward:    {t_bwd_total/n_batches*1000:6.1f}ms/batch")
            print(f"  Optimizer:   {t_opt_total/n_batches*1000:6.1f}ms/batch")
            print(f"")
            if use_cuda:
                peak = torch.cuda.max_memory_allocated() / 1024**3
                total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"  --- VRAM ---")
                print(f"  Peak allocated:  {peak:.1f}GB / {total_vram:.1f}GB ({peak/total_vram*100:.0f}%)")
                print(f"  Headroom:        {total_vram-peak:.1f}GB")
                # Estimate max batch size
                vram_per_sample = (peak - 0.9) / batch_size  # subtract ~0.9GB static
                max_batch = int((total_vram * 0.9 - 0.9) / vram_per_sample)  # 90% safety margin
                print(f"  Est. VRAM/sample: {vram_per_sample*1024:.1f}MB")
                print(f"  Est. max batch:   ~{max_batch} (at 90% VRAM)")
            print(f"")
            if grad_norms:
                gn = np.array(grad_norms)
                print(f"  --- Gradient norms (sampled every 10 batches) ---")
                print(f"  Mean: {gn.mean():.2f}  Std: {gn.std():.2f}  Min: {gn.min():.2f}  Max: {gn.max():.2f}")
                if gn.max() > 100:
                    print(f"  ⚠ Large gradient norms detected — consider gradient clipping")
            print(f"  --- RAM ---")
            import psutil
            ram = psutil.virtual_memory()
            print(f"  Used: {ram.used/1024**3:.1f}GB / {ram.total/1024**3:.1f}GB ({ram.percent}%)")
            disk = shutil.disk_usage("/")
            print(f"  --- Disk ---")
            print(f"  Used: {disk.used/1024**3:.1f}GB / {disk.total/1024**3:.1f}GB ({disk.free/1024**3:.1f}GB free)")
            print(f"{'='*70}\n", flush=True)

        scheduler.step()
        train_loss = train_loss_sum / max(train_total, 1)

        # --- Validate ---
        model.train(False)
        val_loss_sum = 0.0
        head_sums_val = {"dx": 0.0, "dy": 0.0, "click": 0.0, "key": 0.0, "pad": 0.0, "kl": 0.0}
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                (obs, proprio, actions_cvae, dx_gt, dy_gt, click_gt, key_gt, pad_gt) = (
                    batch
                )
                obs = obs.to(device, non_blocking=True)
                proprio = proprio.to(device, non_blocking=True)
                actions_cvae = actions_cvae.to(device, non_blocking=True)
                dx_gt = dx_gt.to(device, non_blocking=True)
                dy_gt = dy_gt.to(device, non_blocking=True)
                click_gt = click_gt.to(device, non_blocking=True)
                key_gt = key_gt.to(device, non_blocking=True)
                pad_gt = pad_gt.to(device, non_blocking=True)

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
                head_sums_val["dx"] += loss_dx.item() * bs
                head_sums_val["dy"] += loss_dy.item() * bs
                head_sums_val["click"] += loss_click.item() * bs
                head_sums_val["key"] += loss_key.item() * bs
                head_sums_val["pad"] += loss_pad.item() * bs
                head_sums_val["kl"] += kl.item() * bs
                val_total += bs

        val_loss = val_loss_sum / max(val_total, 1)
        elapsed = time.perf_counter() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for h in head_names:
            history[f"train_{h}"].append(head_sums_train[h] / max(train_total, 1))
            history[f"val_{h}"].append(head_sums_val[h] / max(val_total, 1))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_pt_path = os.path.join(checkpoint_dir, "best.pt")
            # Strip _orig_mod. prefix from torch.compile so checkpoints are portable
            state = {k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()}
            torch.save(state, best_pt_path)
            # Upload best checkpoint async so training continues immediately
            if hf_upload_repo:
                import logging
                import threading
                from huggingface_hub import HfApi
                _val = val_loss  # capture for closure
                def _upload():
                    try:
                        # Suppress HF upload progress bars
                        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
                        from huggingface_hub.utils import disable_progress_bars
                        disable_progress_bars()
                        api = HfApi()
                        api.create_repo(hf_upload_repo, repo_type="model", exist_ok=True)
                        api.upload_file(
                            path_or_fileobj=best_pt_path,
                            path_in_repo=f"{backbone}_chunk{chunk_size}/best.pt",
                            repo_id=hf_upload_repo,
                            repo_type="model",
                        )
                        print(f"    Uploaded best.pt (val={_val:.4f})", flush=True)
                    except Exception as e:
                        print(f"    Upload failed: {e}", flush=True)
                threading.Thread(target=_upload, daemon=True).start()
        else:
            epochs_without_improvement += 1

        if True:
            remaining_epochs = TRAIN.epochs - epoch - 1
            eta_min = remaining_epochs * elapsed / 60
            # Per-head val losses (raw, before weighting)
            vh = {h: head_sums_val[h] / max(val_total, 1) for h in head_names}
            head_str = " ".join(f"{h}={vh[h]:.3f}" for h in head_names)
            print(
                f"  Epoch {epoch+1:3d}/{TRAIN.epochs} | "
                f"train={train_loss:.4f} val={val_loss:.4f} | "
                f"val_heads: {head_str} | "
                f"kl_w={kl_weight:.4f} | "
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
    state = {k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()}
    torch.save(state, os.path.join(checkpoint_dir, "final.pt"))
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
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Local dataset directory (save_to_disk format)")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hf-data-repo", type=str, default=None,
                        help="HF dataset repo to load data from (e.g. PenTest-duck/cu-vla-data)")
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
        hf_data_repo=args.hf_data_repo,
        hf_upload_repo=args.hf_upload_repo,
    )
