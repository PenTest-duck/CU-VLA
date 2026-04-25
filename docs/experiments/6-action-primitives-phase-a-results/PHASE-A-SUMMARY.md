# Phase A — Summary & Phase B Handoff

Consolidates the four feasibility spikes (A, B, C, E) for Experiment 6. Drives the Phase B implementation plan.

**Branch:** `feat/exp6-phase-a` (not yet merged)
**Plan:** [docs/plans/2026-04-23-action-primitives-phase-a-implementation.md](../../plans/2026-04-23-action-primitives-phase-a-implementation.md)
**Design doc:** [docs/experiments/6-action-primitives.md](../6-action-primitives.md) — see amendments section at the bottom of that doc after this phase closes.

## Results at a glance

| Spike | Question | Result | Verdict | Phase B impact |
|---|---|---|---|---|
| **A** — typing legibility | Does SigLIP2@256 preserve char identity at 14pt? | Per-patch top-1 = 0.74 (46× chance) at 14pt | ✅ Pass | None — Q5/Q6 design stands. Keep `max_num_patches=256`, use visual feedback for typing. |
| **B** — L-click end-to-end | Does the minimum-viable pipeline learn L-click? | Closed-loop success 0.705 (argmax) / **0.815 (probabilistic decode)** at n=200; offline per-head 98-99% on dx/dy/click. **Click decomposition: l_press recall 0.79** (hidden by aggregate). | ✅ Pass (gate 0.50, both decodes) | Architecture stands. Phase B priorities: (a) **soft-label CE training** to match probabilistic decode at inference; (b) **click-head reweighting / press-timing supervision** (l_press 79% is the residual bottleneck); (c) **consistent-state noise injection + scenario-level error training** to close the residual ~18% wrong-direction/edge failures. |
| **C** — M1 closed-loop timing | Is eval cadence (~7.6 Hz) feasible on M1? | Skipped as a dedicated measurement; data extracted from Spike B wall-clock instead: ~180 ms/frame (probabilistic decode) to ~370 ms/frame (argmax) on M1 MPS = 2.7-5.5 Hz | ⚠️ Below Q25 target | Phase B Tier-B eval cadence needs adjustment (cut rollout count or eval less frequently). Probabilistic decode is materially faster (fewer max-frame timeouts). |
| **E** — pygame gen throughput | Does Q7's 200 eps/s hold? | 3.61 eps/s single-process (55× below); multi-proc at chunksize=1 doesn't help | ❌ Q7 target unachievable as written | Revise Q7 target. Phase B needs worker-writes-shards multiproc redesign. |

Full write-ups:
- [spike-a-typing-legibility.md](spike-a-typing-legibility.md) (3 probes: mean-pool, attention-pool, per-patch)
- [spike-b-lclick-end-to-end.md](spike-b-lclick-end-to-end.md)
- [spike-c-m1-eval-timing.md](spike-c-m1-eval-timing.md) (pending T24 run)
- [spike-e-generation-throughput.md](spike-e-generation-throughput.md)

## Design-doc amendments (Phase B)

Findings from this phase that invalidate or refine questions in the original design doc. Apply these as an `## Amendments` section on `docs/experiments/6-action-primitives.md` before writing the Phase B plan.

**Q7 — pygame generator throughput** (revised)
- Old: "pygame env generates at ≥200 eps/sec on consumer CPU."
- New: "≥20 eps/s single-process on M1; 200 eps/s is unachievable without a worker-writes-shards multiproc redesign (naive `Pool.imap` at default `chunksize=1` is IPC-bound and slower than serial)."
- Source: Spike E measurement (3.61 eps/s at N=1000, 1.79 eps/s at workers=4).

**Q8 — per-primitive window length** (revised)
- Old: `max_frames_lclick = 30`.
- New: `max_frames_lclick = 45`. 30 left a 5% label-noise tail (slow-tempo + long-distance episodes timing out before press). 45 drops this to ~0.25%.
- Source: Spike E side-finding; `experiments/action_primitives/config.py` updated in commit `794f9be`.

**Q15 — trainable parameter budget** (revised)
- Old: "~118M total params, ~30M trainable."
- New: "**408.8M total params** (SigLIP2-B vision 93M + text 282M both frozen, trunk 28M, heads 4M, encoders 1M, LoRA 0.6M), **33.6M trainable**. L4 fits comfortably at `micro_batch_episodes=4` with bf16 autocast."
- Source: empirical measurement during T13 model compose.
- Implication: text tower is 282M of frozen weight — essentially dead weight for Phase A (no language variation in L-click). Phase B consideration: is it worth caching text embeddings offline and dropping the text tower from the runtime model? Would shave 1.1 GB from the GPU footprint.

**Q2/Q8 — micro batch size** (new constraint)
- Plan said `micro_batch_episodes=8` (8 micros × 8 = 64 macro). **OOMs on L4.** At micro=8 → 360 images through SigLIP2 per forward → ~43 GB activations (L4 has 22 GB usable).
- Phase B default on L4: `micro_batch_episodes=4` (~14.5 GB total) with `num_workers=4` for prefetch. On L40S (48 GB): can try 8. On A100-80GB: can try 16.

**Q5/Q6 — typing legibility** (design stands, caveat documented)
- Old: "SigLIP2 naflex @ 256 patches preserves char identity at 14pt; use visual feedback for typing progress."
- New: **confirmed** via per-patch probe (74% top-1 at 14pt on 62-way char identity, 46× chance). Two caveats recorded for Phase B:
  - The string-presence probe (mean-pool and attention-pool) gave misleadingly low F1 (0.05-0.33 at 14pt) because of global pooling collapse, not feature degradation. If Phase B needs to re-measure legibility for a new font/size regime, use the per-patch probe, not presence.
  - Probe ceiling ~85-88% even at 32pt, consistent with patch-boundary ambiguity (glyphs straddling patch edges). If typing primitives underperform in Phase B training, first hypothesis is patch-aliasing — test by snapping rendered text to patch centers.
- Source: `spike-a-typing-legibility.md`.

**Q40 — epoch-1 diagnostics** (add new signals)
- Plan listed 4 signals to watch (loss decreasing, no NaN, grad norms stable, no head dominating).
- **Add: per-class click recall** (idle / l_press / l_release / r_press / r_release), not just aggregate click accuracy. Spike B revealed `l_press` recall = 0.79 hidden by aggregate 0.99 — exactly the regime where the headline metric misleads.
- **Add: closed-loop "wrong direction in first 3 frames" rate** as an early signal of localization failure. Spike B's visual inspection identified this as a ~15-18% residual failure mode that frame-level metrics don't surface.
- (Earlier draft: bin-10 frequency on dx/dy heads. **Demoted** — probabilistic decode at inference largely defuses the bin-10 attractor concern. Still cheap to log but no longer load-bearing.)

**Q3 — click-head architecture / loss** (new amendment from Spike B)
- Aggregate click top-1 accuracy of 99.1% hides per-class imbalance: idle 99.5% (12,900 support) vs l_press 79% (300 support). The ~21% press-timing miss rate matches the ~18% residual closed-loop failure rate after probabilistic decode — strongly suggesting press-timing is the dominant remaining bottleneck.
- Phase B options:
  - Loss reweighting: weight non-idle click classes more heavily (currently uniform).
  - Architecture: split into binary "is this a click frame?" head + conditional 4-way "which event?" head. Decouples easy/hard sub-decisions.
  - Temporal supervision: add a small auxiliary loss enforcing that `l_press` at frame t implies `l_release` at frame t+1 (ground truth always satisfies this).

**Inference decode** (new amendment, no Q-number — cross-cutting)
- Phase A trained with hard-label CE; inference defaulted to argmax over the 21-bin softmax. **Probabilistic decode (`E[bin_center]` under softmax) gives +11 pp closed-loop success with no retraining** — argmax 0.705 → expected 0.815 at n=200.
- Phase B should:
  - Default to probabilistic decode in `evaluate.py` (and any future closed-loop tooling).
  - Train with **2-bin triangular soft-label CE** (label = linear-interpolation weights between the two bins bracketing expert's continuous action). This makes training and inference operate on the same target — `Σ softmax · bin_centers` matches `expert_action` exactly when softmax matches the soft target.
- Bin count expansion (21 → 49) was considered and **deferred**. Probabilistic decode captures most of what finer bins would give; the residual failures are not quantization-related.

**Noise injection / synthetic error training** (new amendment from Spike B)
- ~18% residual closed-loop failures (with probabilistic decode) are dominated by wrong-direction / edge-state failures — pure distribution-shift symptoms, untouched by decode improvements.
- Phase B data generator should produce two new types of training data:
  1. **Consistent-state cursor perturbations.** At each training frame, sample a Gaussian displacement (e.g., σ=20 px), teleport cursor to perturbed position (image + proprio update consistently in the env), re-query scripted expert from new position, train on `(perturbed_state, expert_recovery_action)`.
  2. **Curated error scenarios.** Trajectories with deliberate excursions (wrong-direction segments, overshoots, edge-bangs). The `actions_applied` (perturbed) feed the env and history vector; `actions_labeled` (expert's correct action from each state) are the training targets. Model never learns to *take* the wrong actions, only to *recover* from them.
- DAgger (closed-loop rollout in training loop) was considered and **deferred** — synthetic perturbation gives most of the same value without the rollout-in-training overhead.

## New artefacts produced this phase

| File | Purpose |
|---|---|
| `experiments/action_primitives/` | Full Phase A code (env, expert, generator, dataset, model, training, eval, measurements, probes) |
| `tests/action_primitives/` | 71 passing unit tests; 1 slow integration (resume) deferred |
| `scripts/hf_job_train_exp6.py`, `scripts/launch_hf_job_exp6.py` | HF Jobs launcher for Phase A training |
| `data/phase-a-lclick/` (local, 158 MB, gitignored) | 3000-episode L-click dataset, 0.033% label noise |
| `PenTest-duck/cu-vla-exp6-phasea-lclick` (HF dataset) | Same data, shareable |
| `PenTest-duck/cu-vla-exp6-phasea-ckpt:final.pt` + `best.pt` | Trained Spike B checkpoint |

## HF Jobs debugging journey (5 iterations to first successful run)

Captured in `docs/research/hf-jobs-gotchas.md` for reference by exp7+. Short version: `pip install -e .` fails (no pip in UV env), branch must be specified explicitly (clones default=main otherwise), `--data-dir` and `--hf-data-repo` can't both be required, transformers renamed `attention_mask` → `pixel_attention_mask`, and `Pool.imap` at default chunksize is slower than serial. All now fixed and documented.

## Phase B handoff decisions (user to confirm)

**Decided during Phase A close-out (no further action needed):**

- ✅ **Inference decode:** probabilistic (`E[bin_center]` under softmax) is the new default. +11pp closed-loop, no retrain cost. Argmax remains as a CLI option for reproducing published Phase A numbers.
- ✅ **Bin count:** stay at 21 in Phase B first pass. Probabilistic decode addresses most of what finer bins would give; failures aren't quantization-related.
- ✅ **DAgger (closed-loop rollout in training):** deferred. Synthetic perturbation training gives most of the same value with less infrastructure overhead.
- ✅ **Noise injection design:** consistent-state cursor perturbation + curated error scenarios. Both image and proprio update consistently in the env; expert is re-queried from each perturbed state. Avoids my originally-flawed proprio-only framing.
- ✅ **Click-head improvements:** loss reweighting and/or architectural split (binary "click frame?" + conditional 4-way "which event?"). The 79% press-recall is the residual bottleneck after probabilistic decode.

**Still to confirm:**

- [ ] **Soft-label CE training implementation.** Switch dx/dy/scroll heads from hard-label CE to 2-bin triangular soft labels. Confirm: full switch in Phase B v0, or A/B test on a small sub-experiment to measure the lift first?
- [ ] **Loss weighting (Q2):** per-head `L_i^init` rebalancing before generating multi-primitive data, or continue with uniform weights? (Uniform was fine for L-click; with 6 primitives + per-class imbalances the keys head may over-dominate.)
- [ ] **Text tower removal:** cache instruction embeddings offline and drop the 282M text tower from the runtime model? Saves 1.1 GB GPU memory, 0% on quality if instructions are enumerable.
- [ ] **Dataset regen with baked-in history vectors and quantized bins:** Phase A's `__getitem__` does 10-30 ms of Python-loop work per episode. Phase B's 10×-larger dataset compounds this. Worth a one-time regen?
- [ ] **Multi-process generator:** implement worker-writes-shards design before Phase B data gen (24,500 eps at 2.5 eps/s serial = 2.7 h)?
- [ ] **Training compute budget:** Phase A: 5 epochs × macro-batch 64 on 2400 train eps → 0.815 (probabilistic decode). Phase B: 6 primitives × ~10× data × ~20 epochs → ~100× more compute. L4 projects ~15+ h; L40S halves it. Book L40S for Phase B?
- [ ] **Phase B success gate.** Phase A residual (probabilistic decode) is ~18%. With (soft-label CE + click-head fix + noise injection + scenario training), what's the Spike-B-equivalent gate? Suggested: ≥0.85 single-primitive (matches the argmax→expected gain), composed multi-primitive ≥0.50 (beats the naive `0.85^6 ≈ 0.38` lower bound).

## Next step

Draft the Phase B implementation plan at `docs/plans/2026-04-XX-action-primitives-phase-b-implementation.md`. Input:
1. The amendments above (copy into design doc first).
2. The handoff decisions once user answers them.
3. The Spike B + C final verdicts.
