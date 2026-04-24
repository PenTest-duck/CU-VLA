# Spike B — L-click end-to-end

**Ran:** 2026-04-25 (eval on the 2026-04-24 training run)
**Checkpoint:** `PenTest-duck/cu-vla-exp6-phasea-ckpt:final.pt` (step 190, end of 5 epochs)
**Wandb run:** `phase-a-spike-b` (bjb7f2ig, the-duck-company/cu-vla-exp6)
**HF Jobs run:** `69eb1a1fe8e12c6f0a6757a1`

## Question

Does the minimum-viable Phase A architecture (SigLIP2-B-naflex + 16-query trunk + 6 factored heads + LoRA rank-8 on vision + frozen text tower) learn the L-click primitive end-to-end, given 3000 scripted-expert episodes and 5 epochs of behaviour cloning? Target per plan: **≥50% closed-loop success rate**.

## Training config

| Setting | Value |
|---|---|
| Episodes | 3000 (80/10/10 split → 2400 train / 300 val / 300 test) |
| Epochs | 5 |
| Steps/epoch | 38 (macro batch 64) |
| Total steps | 190 |
| Macro batch | 64 episodes |
| Micro batch | 4 episodes (L4-safe; 8 OOMs) |
| DataLoader workers | 4 |
| Hardware | HF Jobs L4 (24 GB) |
| Optimizer | AdamW, trunk lr 3e-4, LoRA lr 2e-4, betas (0.9, 0.95), wd 0.01 |
| LR schedule | Linear warmup 100 steps → cosine decay to 10% |
| Precision | bf16 autocast |
| Grad clip | 1.0 |

Model: **408.8M total** (vision 93M frozen + text 282M frozen + trunk 28M + heads 4M + encoders 1M + LoRA 0.6M), **33.6M trainable**.

## Offline eval (val split, per-head top-1 accuracy)

Ran: `uv run python -m experiments.action_primitives.evaluate --checkpoint final.pt --data-dir data/phase-a-lclick --n-rollouts 200 --device mps` on M1 MPS. Wall-clock: **35m 45s for offline** (300 val episodes × ~7.15 s/ep).

| Head | Top-1 accuracy | Chance baseline | × chance |
|---|---|---|---|
| dx | 0.9819 | 1/21 = 0.048 | 20.5× |
| dy | 0.9828 | 1/21 = 0.048 | 20.5× |
| click | 0.9908 | 1/5 = 0.200 | 5.0× |
| scroll | 1.0000 | 1/21 = 0.048 | trivial (always idle) |
| keys | 1.0000 | 1/3 per key = 0.333 | trivial (all 77 keys always idle) |
| done | 1.0000 | 0.500 | 2.0× |

The non-trivial heads (dx, dy, click) are essentially saturated. `scroll`, `keys`, `done` trivially achieve perfect accuracy because their label distribution collapses for L-click-only data (scroll always 0, keys always idle, done is deterministic given expert trajectory).

**Frame-level behaviour cloning is essentially solved.** Whatever failure modes show up closed-loop are not "the model mis-predicts the next action" — they're some form of distribution-shift or compounding-error.

## Closed-loop eval (200 rollouts)

Wall-clock: **24m 51s** (200 rollouts × ~7.46 s/ep on M1 MPS).

| Metric | Rate |
|---|---|
| **success_rate** | **0.7050** ← Spike B gate: ≥0.50 ✅ |
| click_within_0px | 0.0000 |
| click_within_3px | 0.1000 |
| click_within_5px | 0.2450 |
| click_within_10px | 0.4650 |

### Sanity-check: live pygame n=20 run

| Metric | n=20 (visual) | n=200 (headless) | Δ |
|---|---|---|---|
| success_rate | 0.70 | 0.705 | +0.005 (stable) |
| click_within_3px | 0.05 | 0.10 | +0.05 (small-N noise) |
| click_within_5px | 0.25 | 0.245 | -0.005 (stable) |
| click_within_10px | 0.60 | 0.465 | -0.135 (n=20 was lucky) |

n=200 is the authoritative number. The 10 px tolerance metric was noisy at n=20.

## Interpretation

### Verdict: **PASS** (0.705 vs 0.50 gate, stable across n=20 and n=200)

### The offline → closed-loop gap is the interesting finding

- **Offline per-head: ~99% across dx, dy, click.** Frame-level BC succeeded.
- **Closed-loop: 70.5% success.** A **~29 percentage-point gap** between "predicts each frame's action right" and "reaches the goal state by the end of the episode."

This is the classic **compounding-error / covariate-shift** pattern in imitation learning:
- Expert trajectories only visit the "on-the-way-to-target" state manifold.
- Model learns to imitate expert on that manifold and does so near-perfectly.
- At inference time, small per-frame errors accumulate: e.g., a slightly-too-small `dx` step at frame 3 leaves the cursor 10 px off where the expert was, which is a state slightly outside the training distribution, which causes a slightly worse next-frame prediction, which compounds.
- Some rollouts eventually leave the training manifold entirely (the "stuck far away" 30% cluster) and the model has no demonstrations for how to recover.

### Failure-mode analysis

From the n=20 visual pass:
- **Mode A — "stuck far away" (5/6 failures, dist ≥230 px):** hit max_frames=45 with cursor still ~250 px from target. The physical budget (45 × 18 px slow-tempo peak = 810 px > ~850 px diagonal) exists, so the failure is motor, not kinematic. Hypothesis: model outputs near-zero `dx`/`dy` (middle bin 10) on frames where it's uncertain → cursor stalls → more uncertainty.
- **Mode B — "close but didn't click" (1/6, dist ≈33 px):** rare.

The `done` head's 100% offline accuracy combined with the 30% "stuck-far-away" closed-loop failure is **fully consistent** with compounding error: when the cursor is stuck far from target, the model correctly predicts `done=0` (it *isn't* done and hasn't even tried the press sequence). Training never showed it an end-of-episode state where it *should* have been done but couldn't recover.

### Tolerance-curve shape

| tol | rate | gap vs binary success (0.705) |
|---|---|---|
| 0 px | 0.00 | — nobody nails center (expected, expert doesn't either) |
| 3 px | 0.10 | — 10% precision landings |
| 5 px | 0.245 | — 24.5% within a 10×10 square around center |
| 10 px | 0.465 | — 46.5% within a 20×20 square |
| success (binary) | 0.705 | — 24% of rollouts succeed AND are >10 px from center |

The 24% gap between `click_within_10px` (46.5%) and `success_rate` (70.5%) = rollouts that land in the bbox but >10 px from center. Buttons are 40–120 wide × 30–80 tall, so 20-40 px off-center is still inside. Model lands in the correct half of wider buttons but doesn't consistently hit center.

## Recommendation

**Proceed to Phase B as-designed, with two Phase B additions:**

1. **Add a covariate-shift diagnostic to Phase B training.** Log `frac(dx_bin == 10)` and `frac(dy_bin == 10)` per step during training. If bin-10 frequency is >~30% on frames where the expert moved >5 px, the model is collapsing to the zero-motion attractor — an early warning of the "stuck far away" Phase A failure mode.

2. **Consider at least a small DAgger-style correction phase in Phase B.** The closed-loop-vs-offline gap will only widen as primitive complexity grows. Cheapest version: after initial BC training, run the model in closed-loop, collect states where it fails, have the scripted expert label those states, and add them to training. Even one iteration would address the "no demonstrations for recovery" failure mode. If this is Phase B scope bloat, at least instrument Phase B so we can measure the same gap on multi-primitive runs.

**Don't** need to revisit model capacity, loss weighting, or trunk depth — offline accuracy shows the current architecture has the capacity to fit L-click perfectly. The bottleneck is training-distribution coverage, not model expressiveness.

## Follow-ups (not blocking)

- **Best.pt vs final.pt comparison.** The training run's `best.pt` (lowest val loss) and `final.pt` (step 190) are both uploaded to HF. Quick re-eval on `best.pt` would show whether training was over-fitting in the last ~30 steps. If best.pt gives ≥0.72 closed-loop, future runs should default to best.pt. If roughly equal, it doesn't matter.
- **M1 per-frame timing (Spike C, T24).** Closed-loop wall-clock was 7.46 s/rollout; rollouts average ~20 frames so that's ~370 ms/frame on M1 MPS — well below the Q25 assumed 7.6 Hz (130 ms/frame). Needs the dedicated `m1_eval_timing.py` measurement to confirm with warmup controlled and per-frame breakdown, but the ballpark suggests Phase B's Tier-B eval-cadence design may need adjusting.
- **Mode A root cause validation.** If anyone wants to spend 20 minutes, take any failed n=200 rollout (e.g., one of the stuck-far-away ones) and replay it with `--visual` to confirm the cursor is predicting near-zero motion rather than e.g., heading in the wrong direction. The bin-10 log would show this at scale but one visual trace would confirm the hypothesis.

## Next steps

- [x] Plug in n=200 numbers.
- [x] Verdict: pass at 0.705 (gate 0.50).
- [ ] Proceed to Spike C (M1 eval timing, T24) with `final.pt`.
- [ ] Phase A summary (T25) — update Spike B row.
- [ ] After T24: kick off Phase B implementation plan.
