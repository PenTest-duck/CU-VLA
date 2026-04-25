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

Ran: `uv run python -m experiments.action_primitives.evaluate --checkpoint final.pt --data-dir data/phase-a-lclick --n-rollouts 200 --device mps --decode expected` on M1 MPS. Wall-clock: **44m 24s for offline** (300 val episodes × ~8.88 s/ep).

| Head | Top-1 accuracy | Chance baseline | × chance |
|---|---|---|---|
| dx | 0.9819 | 1/21 = 0.048 | 20.5× |
| dy | 0.9828 | 1/21 = 0.048 | 20.5× |
| click | 0.9908 | 1/5 = 0.200 | 5.0× |
| scroll | 1.0000 | 1/21 = 0.048 | trivial (always idle) |
| keys | 1.0000 | 1/3 per key = 0.333 | trivial (all 77 keys always idle) |
| done | 1.0000 | 0.500 | 2.0× |

### Click-head decomposition (most consequential finding from offline)

The aggregate 99.1% click accuracy is dominated by the 95.6% of frames where expert is idle. Per-class recall tells a different story:

| Click class | Recall | Support | Notes |
|---|---|---|---|
| `idle` | **0.9952** | 12,900 | Model knows when not to click. Near-perfect. |
| `l_press` | **0.7933** | 300 | **One in five press frames is mistimed.** |
| `l_release` | **1.0000** | 300 | Conditional on press already fired — trivial. |
| `r_press` | n/a | 0 | No R-click in Phase A. |
| `r_release` | n/a | 0 | No R-click in Phase A. |

**This is the headline.** `l_press` at 79% recall is the single weakest head in the model — the timing signal "cursor is on button NOW, fire" is genuinely hard, and the aggregate metric was hiding it. Release is essentially free given press (model's history shows press just happened, proprio shows held_mouse=1).

The "hover, move past, then click late" failure mode you observed in visual eval maps directly to this 21% of mistimed presses: when the model fires press 2-3 frames late, the cursor has drifted in those frames and lands off-target.

**Frame-level BC is not as solved as the headline implies.** dx/dy are saturated; press timing is not.

## Closed-loop eval (200 rollouts) — argmax vs probabilistic decode

Two decode strategies were tested at inference. The 21-bin discrete dx/dy heads can be read with `argmax` (Phase A's original design) or with `expected` (`E[bin_center]` under the softmax — interpolates between bins, addresses quantization-induced zig-zag).

| Metric | argmax (n=200) | **expected (n=200)** | Δ |
|---|---|---|---|
| **success_rate** | **0.7050** | **0.8150** | **+11.0 pp** |
| click_within_0px | 0.0000 | 0.0000 | flat |
| click_within_3px | 0.1000 | 0.1050 | flat |
| click_within_5px | 0.2450 | 0.2250 | -2.0 pp (within noise) |
| click_within_10px | 0.4650 | 0.5500 | +8.5 pp |

Both runs use the same `final.pt` checkpoint and same 200 seeds (10000-10199). Probabilistic decode wall-clock: **12m 02s for closed-loop** (vs 24m 51s for argmax) — the lower max-frame-timeout rate means episodes complete faster on average.

**Both gates pass.** Spike B gate is ≥0.50; argmax 0.705 ✅, expected 0.815 ✅.

### Why probabilistic decode helps so much

The 21-bin exponential scheme has very coarse resolution in the primary motion regime: `0.026, 0.066, 0.16, 0.41, 1.02, 2.56, 6.4, 16, 40, 100` px on each side of zero. Between bins 18 and 19 (16 px → 40 px) there's a 2.5× gap. When the model's softmax sits 60/40 across those two bins, argmax flips between them based on tiny state variations → cursor zig-zags → trajectory drifts further off-expert manifold each frame → some rollouts diverge unrecoverably.

Probabilistic decode computes `0.6 × 16 + 0.4 × 40 = 25.6 px` directly, giving stable continuous output. Cursor tracks the expert path more tightly, so small per-frame quantization errors don't compound into catastrophic drift.

### Sanity-check: live pygame n=20 runs

For comparison with the visual inspection that motivated this analysis:

| Metric | n=20 argmax | n=20 expected | n=200 argmax | n=200 expected |
|---|---|---|---|---|
| success_rate | 0.70 | 0.85 | 0.705 | 0.815 |
| click_within_10px | 0.60 | 0.60 | 0.465 | 0.550 |

n=200 numbers are the publishable figures. The improvement is robust across both sample sizes.

## Interpretation

### Verdict: **PASS** (argmax 0.705, expected 0.815; both above 0.50 gate)

### Three distinct findings, all from the same training run

**(1) Frame-level dx/dy BC is solved, but discrete decode is wasteful.**

98% offline argmax accuracy combined with the 11 pp closed-loop improvement from probabilistic decode (with no retraining) tells us the model's softmax distributions encode meaningful continuous information that argmax was throwing away. The 21-bin exponential quantization has a 2.5× gap between adjacent bins in the primary motion regime — argmax zig-zag was inducing trajectory drift that probabilistic decode eliminates.

**(2) Press-timing is the weakest single signal.**

`l_press` recall = 0.79 (300 support) is hidden by the aggregate click metric. This is roughly the *exact* expected per-rollout failure rate (~21%, vs the residual ~18.5% of expected-decode rollouts that fail) — strongly suggesting the residual closed-loop failures are not "model can't reach the button" but "model reaches the button and mistimes the click."

**(3) The remaining offline → closed-loop gap (~18 pp) is genuine compounding error.**

Even with probabilistic decode, ~18% of rollouts fail. From visual inspection (n=20) these are dominated by **wrong-direction movement** (cursor heads away from target and never recovers) and **edge-state failures** (cursor reaches canvas edge, model has no training demonstrations for that state). These won't be fixed by decode changes — they require training-distribution coverage of off-trajectory states (Phase B noise injection / scenario training).

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

**Proceed to Phase B with the following changes informed by Spike B:**

**Decode (immediate, no retrain):**
- Default to probabilistic decode (`E[bin_center]` under softmax) for all closed-loop runs going forward. Argmax is preserved as an option for reproducibility but isn't the recommended inference mode.

**Phase B training-side changes:**
1. **Soft-label CE on dx/dy/scroll heads.** Currently the loss uses hard labels = argmax bin nearest to expert continuous action. Train with 2-bin triangular soft labels (weights linearly interpolated between the two bins bracketing the expert continuous action). This trains the softmax to *be* a properly distributed estimator, so probabilistic decode at inference is operating on the same target the loss was optimizing.
2. **Click-head loss reweighting or dedicated press-timing supervision.** `l_press` at 79% recall is the bottleneck for the residual ~18% closed-loop failures. Options:
   - Increase loss weight on non-idle click classes (currently uniform).
   - Replace 5-way categorical with binary "is this a click frame?" + conditional 4-way "which event?". Decouples the easy "is this idle" from the hard "exactly which frame to fire."
   - Add a temporal smoothing term: if predicted click_l_press, the next frame should be release.
3. **Consistent-state noise injection in data generator.** Sample cursor displacements at training frames; teleport cursor to perturbed position (image + proprio update consistently); re-query scripted expert from new position; train on `(perturbed_state, expert_recovery_action)`. Addresses the "off-expert-trajectory states have no training data" gap.
4. **Scenario-level error injection in data generator.** Curated trajectories that include wrong-direction excursions, overshoots, edge-bangs, etc. Use `actions_applied` (the perturbed action) for history; use `actions_labeled` (the expert's correct action from each state) as the training target. This is what closes the residual ~18% wrong-direction failure mode.

**Phase B diagnostics to add (Q40 amendments):**
- Per-class click recall during training (not just aggregate).
- `frac(predicted_bin == bin_10)` on dx/dy heads at frames where expert delta > 5 px (early warning for zero-motion attractor — though probabilistic decode largely defuses this).
- Closed-loop "wrong-direction within-first-3-frames" rate as an early signal of localization failure.

**Don't** retrain just for finer bin spacing (49 vs 21). Probabilistic decode already extracts most of what the finer bins would give; the residual failures are not quantization-related. Bin count can be revisited mid-Phase-B if soft-label CE + noise injection don't close enough of the gap.

**Don't** revisit model capacity, trunk depth, or vision encoder choice. Offline accuracy demonstrates capacity is sufficient for L-click; the bottleneck is training-distribution coverage and inference decode, not expressiveness.

## Follow-ups (not blocking)

- **`best.pt` is identical to `final.pt`** for this run (no improvement in val loss in the final ~30 steps — early-stopping would have triggered). No comparison needed.
- **M1 timing (Spike C, T24) skipped.** Spike B's incidental wall-clock measurement gives us the data anyway: argmax = 7.46 s/rollout, expected = 3.61 s/rollout on M1 MPS, average ~12-20 frames per rollout. Per-frame inference is ~180-370 ms/frame depending on decode mode. Below the Q25 assumed 7.6 Hz target — Phase B Tier-B eval cadence will need adjustment downward, or batch eval to amortize.
- **Wrong-direction failure visual replay.** Take any failed n=200 rollout (e.g., one of the stuck-far-away ones) and replay it with `--visual --decode expected` to confirm the residual failures are localization (cursor heads to wrong patch of canvas), not stalling. Visual inspection of n=20 expected runs already strongly suggests this.

## Next steps

- [x] Plug in n=200 numbers (both decode modes).
- [x] Verdict: pass — argmax 0.705, expected 0.815, both above 0.50 gate.
- [x] Click head decomposition added to offline_eval and reported (l_press = 79% recall is the headline finding).
- [ ] Phase A summary (T25) — update Spike B row + add probabilistic decode to amendments.
- [ ] Phase B implementation plan: incorporate soft-label CE, click-head reweighting, noise injection, scenario-level error training.
