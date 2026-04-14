# Experiment 2: Training Optimisation Learnings

Date: 2026-04-14
Context: ACT (Action Chunking with Transformers) model for drag-and-label task, 33M params, ResNet18 backbone, 10k episodes (~296k timesteps), trained on HF Jobs.

## Root Cause: Model Had No Cursor Position

The original model got **0/200 eval success rate**. Investigation revealed:

1. **Proprioception used hardcoded (0.5, 0.5)** for cursor position during training — the model never saw actual cursor coordinates.
2. **Cursor was invisible** in the image — 3px radius on 400x400, downsampled to ~1.7px on 224x224. ResNet18's 7x7 stride-2 first conv obliterated it.
3. **The model collapsed to predicting the global mean action** (near-zero deltas) because it couldn't determine cursor position from either signal.

**Diagnostic that confirmed it:** Probed the trained model with different proprio values and different scenes — output was near-constant (~dx=0, dy=+6) regardless of input. The model learned one fixed output.

**Fix:** Store actual cursor_x/cursor_y in dataset, use normalized (cx/W, cy/H) in proprio, enlarge cursor radius to 10px. Also fixed a train/eval mismatch where training used `action_click` (the action being taken) instead of `state_click` (env state before action) for proprio — a 16.8% disagreement rate.

## Gradient Explosion

### The problem

First training run with cursor fix showed gradient norms: mean=192, max=1285. The root cause was **multi-scale loss components**:

```
total = L1(dx) + L1(dy) + 5*BCE(click) + 5*CE(key) + BCE(pad) + kl*KL
```

The model output `tanh() * 50` for dx/dy, so the dx/dy heads pushed ~50x larger gradients through the shared backbone than click/key heads. This meant:
- dx/dy dominated all gradient updates
- click/key heads were starved of learning signal
- key loss was stuck at 0.52 vs 0.67 baseline (barely learning)

### What didn't work: Aggressive gradient clipping (max_norm=10)

Set clip to 10 based on the high norms. Result: **100% of batches were clipped**. Every gradient was capped to exactly norm=10. This effectively turned Adam into fixed-step-size SGD — all heads got the same capped gradient regardless of actual learning signal. The CVAE KL dropped to 0.55 (vs 6+ in healthy training), meaning the latent space stopped learning.

**Lesson:** Clipping treats the symptom, not the cause. If 100% of batches are clipped, the threshold is too low and you're actively harming training.

### What worked: Normalising dx/dy output to [-1, 1]

Removed `* max_delta_px` from the model output (now just `tanh()`), divided ground truth targets by 50 in the dataloader, and multiply back by 50 at inference. This reduced dx/dy gradient scale 50x, aligning all heads to similar gradient magnitudes.

Combined with:
- **LR warmup:** 5-epoch linear warmup (start_factor=1e-3) before cosine decay. Prevents large updates when initial predictions are random and all gradients are at their maximum.
- **Pre-LayerNorm:** `norm_first=True` on all transformer layers. Stabilises gradient flow through deep layers. One-line change per layer.
- **Generous clip threshold:** Raised from 10 to 100 as a safety net, not an active throttle.

### Results comparison at epoch 3

| Head | Run 1 (no clip) | Run 2 (clip=10) | Run 3 (normalised) |
|------|----------------|----------------|-------------------|
| dx (px) | 2.4 | 8.2* | 4.2 |
| click | 0.056 | 0.425* | **0.003** |
| key | 0.523 | 0.579* | **0.023** |
| grad mean | 192 | 10 (all clipped) | 68 |
| grad max | 1285 | 10 | 100 |
| KL | 2.27 | 0.55 | 5.08 |

*Run 2 only had 2 epochs before cancellation.

Run 1 had good dx/dy but through reckless oversized steps that starved other heads. Run 3 learns all heads proportionally — 23x better key, 18x better click than run 1.

### Key insight

The unconstrained run optimised dx/dy at the expense of everything else. The normalised run learns the entire task — movement, clicking, AND typing — simultaneously, because gradients from all heads contribute proportionally.

## Training Speed Optimisation

### Hardware comparison

Trained on HF Jobs. L40S was the optimal choice for this workload:

| GPU | $/hr | TFLOPS (bf16) | Epoch time | Total cost |
|-----|------|--------------|------------|------------|
| T4 | $0.60 | 65 | ~295s | ~$5.50 |
| L4 | $0.80 | 121 | ~345s | ~$5.15 |
| **L40S** | **$1.80** | **366** | **~183s** | **~$5.50** |
| A100 | $2.50 | 312 | ~200s* | ~$7.00 |

*estimated

L40S wins on wall time (1.9x faster than L4) at similar total cost. Key advantage: **62GB RAM** allows the entire 42GB image memmap to be cached in memory.

### Data loading was the bottleneck, not GPU compute

Initial diagnostics showed GPU was idle 35% of the time, waiting for data. The bottleneck evolved across runs:

| Approach | DataLoader/batch | Problem |
|----------|-----------------|---------|
| Arrow + PNG decode (T4, 4 workers) | ~190ms | PNG decode CPU-bound |
| Memmap 42GB + random shuffle (L4) | **752ms** | Page faults (42GB > 30GB RAM) |
| Memmap + sequential sampler (L4) | **415ms** | Per-episode random seeks |
| Memmap + sequential sampler (L40S) | **366ms** (ep1) → **~50ms** (ep2+) | Page cache warms by epoch 2 |

### Image pre-decode to memmap

Decode all 296k images from PNG to raw uint8 in a disk-backed `np.memmap` at training start. Eliminates ~20M redundant PNG decodes over a full run.

- **Parallel decode:** 8 threads via ThreadPoolExecutor (PIL releases GIL). ~234s one-time cost.
- **Stored at `/tmp/`** not checkpoint_dir — avoids uploading 42GB to HF Hub.
- **Cache reuse:** checks file size matches expected shape before reusing.

**Critical lesson:** In-memory numpy array OOM'd (42GB > 30GB RAM). Memmap writes to disk and lets the OS page in as needed. On machines with enough RAM (L40S 62GB), the OS page cache eventually holds the entire file.

### Sequential episode sampler

Random DataLoader shuffle over a 42GB memmap causes random I/O — each sample is a 150KB read at a random offset. On machines where the memmap doesn't fit in RAM, this causes page faults (~0.73ms/sample).

Fix: `EpisodeSequentialSampler` shuffles episode order each epoch but iterates timesteps within each episode sequentially. This enables OS readahead. Within-batch correlation is minimal (~34 episodes per batch of 1024).

### non_blocking GPU transfers

All `.to(device)` calls use `non_blocking=True`. Requires `pin_memory=True` on the DataLoader (already set). Allows CPU-to-GPU transfers to overlap with previous batch's GPU computation. Measured at ~2% of wall time — small but free.

### num_workers=0 with memmap

Forked DataLoader workers (num_workers>0) with a memmap caused:
1. **Log stream stalls** on HF Jobs — logs would vanish mid-training
2. **fork+memmap contention** — multiple processes accessing the same memory-mapped file

With pre-decoded memmap, data loading is just array indexing (~0.1ms/sample). Workers are unnecessary. Setting `num_workers=0` fixed the log issue and simplified the code.

### torch.compile

Enabled on CUDA with `torch.set_float32_matmul_precision("high")` for TF32 tensor cores. First-batch warmup takes ~56-60s (one-time cost). Steady-state speedup is significant but hard to measure in isolation.

The L40S showed a warning `"Not enough SMs to use max_autotune_gemm mode"` — the L40S has enough SMs but torch's heuristic was conservative. No functional impact.

### Batch size scaling

Increased from 256 to 1024 (4x) with linear LR scaling (1e-4 to 4e-4). On L4 (22GB VRAM), peak usage was 16.1GB — plenty of headroom. On L40S (44.4GB), only 36% utilised.

Bigger batches don't help much when the model is already well-utilised per sample — the per-sample GPU work is the same regardless of batch size. The main benefit is reducing per-batch overhead (DataLoader calls, optimizer steps), which was only ~3% of wall time. Diminishing returns beyond 1024.

### Epoch 1 diagnostics

Added comprehensive first-epoch instrumentation (disabled after epoch 1):
- Per-phase timing: DataLoader, GPU transfer, forward, loss, backward, optimizer — with percentages
- VRAM: allocated/reserved/peak, estimated max batch size
- Gradient norm statistics (sampled every 10 batches)
- RAM and disk usage

This was essential for identifying the DataLoader bottleneck and gradient explosion. The `torch.cuda.synchronize()` calls add ~100s overhead to epoch 1 but are disabled for subsequent epochs.

## Loss Function Observations

### Multi-phase task creates class imbalance

The drag-and-label task has distinct phases with very different action distributions:

| Phase | % of steps | dx/dy | click | key |
|-------|-----------|-------|-------|-----|
| Navigate | 32.9% | large | 0 | 0 |
| Drag | 31.7% | large | 1 | 0 |
| Pause | 21.9% | 0 | 0 | 0 |
| Type | 10.1% | 0 | 0 | >0 |
| Grab | 3.4% | 0 | 1 | 0 |

key=0 is 89.9% of samples. A model predicting "never type" gets key CE of 0.67 — barely above the trained model's early performance. The key head was being starved by dx/dy gradient dominance, not just class imbalance.

### Loss component scale matters

Before normalisation, dx/dy L1 contributed ~73% of total loss gradient. After normalisation, all heads contribute roughly equally (~0.1-0.5 each). This is why key went from 0.52 (barely learning) to 0.023 (solved) — it finally got proportional gradient signal.

### Per-head loss logging is essential

Added per-head loss tracking (dx, dy, click, key, pad, kl) to history.pt and epoch log lines. Without this, we would have seen "val loss is decreasing" without knowing that key was stuck. Always log per-head losses for multi-head models.

## HF Jobs Platform Notes

- **Disk:** 418.8GB total, ~56GB used after setup. Plenty for 42GB memmap.
- **RAM:** L4x1 has 30GB (too small for 42GB memmap in memory). L40S has 62GB (fits).
- **Log buffer:** Logs can vanish if forked processes flood stdout. Fix: `num_workers=0`.
- **Checkpoint versioning:** Upload to `{backbone}_chunk{chunk_size}/{YYYYMMDD-HHMM}/` to avoid overwriting previous runs.
- **GPU flavour naming:** API expects `l4x1` not `1x-l4`.
- **`total_memory` not `total_mem`** for `torch.cuda.get_device_properties()`.
