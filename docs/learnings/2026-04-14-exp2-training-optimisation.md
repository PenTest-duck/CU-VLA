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

## Open Questions & Decisions (2026-04-14)

Reviewed all open questions from the current training run (resnet18, chunk=10, L40S, 100 epochs). Decisions made at epoch 15 with val_loss=0.107 (best at epoch 14).

### Training progress at time of review

| Epoch | val_total | dx | dy | click | key | kl | LR |
|-------|-----------|------|------|-------|-------|-------|---------|
| 4 | 0.190 | 0.055 | 0.070 | 0.000 | 0.006 | 2.09 | 3.2e-4 |
| 5 | **0.572** | 0.077 | 0.112 | 0.004 | 0.057 | 3.73 | **4.0e-4** |
| 8 | 0.144 | 0.041 | 0.046 | 0.000 | 0.003 | 1.14 | 4.0e-4 |
| 10 | 0.128 | 0.041 | 0.037 | 0.000 | 0.001 | 0.95 | 4.0e-4 |
| 14 | **0.107** | 0.033 | 0.030 | 0.000 | 0.001 | 0.62 | 3.9e-4 |
| 15 | 0.146 | 0.032 | 0.038 | 0.000 | 0.006 | 0.53 | 3.9e-4 |

### Decisions

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 1 | Asymmetric key loss | **Monitor, don't implement** | Key loss at 0.000-0.006 after gradient normalisation — already solved. Flag if it regresses. |
| 2 | Eval results | **Wait for training to finish** | Training still running. Per-head losses can look good while closed-loop eval fails at phase transitions (compounding BC errors). |
| 3 | Loss weight rebalancing | **Reduce click/key from 5x to 2x next run** | Click (0.000) and key (0.001) are near-zero — 5x weights amplify noise on solved heads. dx/dy (0.033/0.030) are the remaining bottleneck and should get more relative gradient signal. |
| 4 | JPEG cache vs memmap | **Closed — not needed** | L40S has 62GB RAM, 42GB memmap fits comfortably. Only revisit if hardware downsizes. |
| 5 | max_delta_px clipping (50px/frame) | **Defer — 50px fine for 256×256** | At 30Hz, 50px/frame = 1500px/sec, crosses 256px screen in 0.17s. For future full-resolution scaling, the real issue is action representation (tanh with fixed scale can't serve both precision and range). Analysed log-space deltas vs discrete buckets — see below. |
| 6 | Port optimisations to exp 3 | **After exp 2 eval** | Finish exp 2 end-to-end first. Exp 3 has a different action space (multi-binary held state) so not all optimisations may apply directly. |
| 7 | Optimal LR (4e-4) | **Keep 4e-4** | Epoch 5 spike (0.19→0.57) was transient — model recovered to 0.107 by epoch 14. Cosine decay is handling the tail. Changing LR simultaneously with loss weights would confound comparison. |

### Epoch 5 LR spike analysis

Warmup completed at epoch 5 (LR hit full 4e-4). Val loss spiked 3x (0.190→0.572), all heads regressed. Gradient norms were hitting the clip ceiling (mean=68, max=100). However, the model fully recovered by epoch 8 (val=0.144) and continued improving to best-ever 0.107 at epoch 14. Conclusion: 4e-4 is at the aggressive edge but not destructive. The warmup→constant transition causes a transient shock that resolves within ~3 epochs.

### Action representation analysis (future reference)

For scaling beyond 256×256 to real screen resolutions, the current `tanh() * 50px` representation has a fundamental precision-range tradeoff. Two candidates analysed:

**Log-space deltas** (predict `sign(δ) × log(|δ|+1)`, invert with `sign × (exp(|pred|) - 1)`):
- Excellent precision near zero (where cursor accuracy matters most)
- Natural compression of large ballistic movements
- Matches Fitts's Law intuition (precision scales with log distance)
- Weakness: regression output = single point estimate, cannot represent multimodal distributions (e.g., two valid targets equidistant from cursor → averages to zero)

**Discrete buckets** (256 bins per dimension, cross-entropy loss):
- Universal VLA baseline (RT-2, OpenVLA, Octo all use this)
- Handles multimodality natively — softmax can express "either left or right"
- Very stable training (classification is well-behaved)
- Weakness: uniform bins have poor precision near zero (~4px bins over [-500,+500]). Fix: quantile-based binning (dense near zero, sparse at extremes)
- Parameter cost: 256 output neurons per dimension vs 1 for regression

**Advanced approaches from literature:**
- FAST (DCT + BPE compression): 5-13x compression, dominates for high-frequency manipulation
- OAT (learned tokeniser): coarse-to-fine structure, prefix-robust encoding, best overall but complex
- Hybrid coarse+fine heads: biologically inspired (ballistic + corrective saccades)

**Decision:** Deferred. Current 50px works for 256×256 experiments. Revisit when scaling to real screen resolutions. The multimodality argument favours discretisation long-term, but ACT's CVAE handles it for now.
