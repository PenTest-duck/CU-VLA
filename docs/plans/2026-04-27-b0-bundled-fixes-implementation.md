# B0 Bundled Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bundle 8 targeted fixes onto `feat/exp6-phase-b0`, retrain to 2000 steps on AWS SageMaker, and re-run the 8-eval suite to clear the (now-relaxed) Phase B0 typed-disposition gate.

**Architecture:** All changes are additive/configurable on top of the existing B0 codebase. No backbone swap, no trunk redesign. Most fixes are guarded by a config flag so a flag-off run reproduces prior behavior. The single new architectural addition is a small auxiliary "target-button classifier" head attached to the trunk's pooled output - gradient-only, training-time-only, ~30K params.

**Tech Stack:** PyTorch, transformers (SigLIP2-naflex), peft (LoRA), HuggingFace `datasets`, parquet, AWS SageMaker.

**Branch:** `feat/exp6-phase-b0` (existing - no new worktree needed; user is already on this branch).

**Out of scope:** Action chunking (deferred to B1+); replacing SigLIP2 with a grounding-native backbone (deferred); larger 20k-episode dataset regen (only if gates miss after this bundle).

---

(Plan body continues below - see full content in subsequent edits)

## Background — what each fix targets

| Fix | Targets | Mechanism |
|---|---|---|
| F1: `focal_gamma` 3.0 → 2.0 + `class_weight=[1,5,5]` on click_left | Click-press recall bistability (val recall_press oscillates 0.40-0.97) | Restores Phase A's focal weight + adds explicit press/release upweighting to overcome 95% idle imbalance |
| F2: LoRA rank-4 on text tower's last 2 attn layers | 9 pp wrong-instruction degradation | Lets text encoder specialize for our short imperative instructions |
| F3: Bias scene generator toward 2+ button scenes | Insufficient incentive for trunk to read text on 1-button scenes | Shift n_buttons distribution from uniform(1,6) to weighted(1,6) with ~15% on 1-button |
| F4: Train to 2000 steps (was crashed at 512, planned 1250) | Val loss still falling at crash | Buys ~30% more compute on top of original schedule |
| A1: Click decode temperature + threshold + 3-frame smoothing at inference | Click-press recall bistability (eval-side) | Decouples eval from exact-frame press decision |
| A2: Image augmentation in training data loader | Adversarial/composite slice failures + visual robustness | Color jitter + small spatial jitter |
| A3: Auxiliary target-button classifier head | Wrong-instruction degradation root cause: no direct text→grounding signal | Adds explicit gradient signal: trunk must encode "which button is the target". Loss applied only on episode frames 0-2 to prevent history leakage |
| A4: Per-head loss weight rebalancing | Optimizer wastes capacity on near-zero heads (scroll, keys, done) | Drop weights on near-zero heads to 0.1 to free up motor/click gradient bandwidth |
| Gate: `phase_a_holdout` >=0.92 → >=0.80 | Recalibrate to Phase A spike B's actual ceiling (0.78) | Doc-only update; doesn't affect training |
| Already done: `sign_acc` idle-mask | Metric was deflated by idle frames | Already committed in metrics.py |

---

## File Structure

| File | Why touched | Type |
|---|---|---|
| `experiments/action_primitives/config.py` | F1, F4, A2 (toggles), A3 (toggle + dim), A4 (weights), F3 (scene distrib param) | Modify |
| `experiments/action_primitives/losses.py` | F1 (class weights), A3 (aux loss + first-3-frame mask), A4 (head weight wiring) | Modify |
| `experiments/action_primitives/heads.py` | A3 (new aux head class) | Modify |
| `experiments/action_primitives/model.py` | A3 (forward returns aux logits + episode_frame_idx plumbing) | Modify |
| `experiments/action_primitives/backbones.py` | F2 (text-tower LoRA option in `apply_lora`) | Modify |
| `experiments/action_primitives/dataset.py` | A2 (augmentation pipeline), A3 (`target_button_id` already in metadata, expose to batch) | Modify |
| `experiments/action_primitives/scene.py` | F3 (`n_buttons_distribution` parameter on `generate_scene`) | Modify |
| `experiments/action_primitives/generator.py` | F3 (forward distribution param) | Modify |
| `experiments/action_primitives/generate_data.py` | F3 (CLI flag for distribution) | Modify |
| `experiments/action_primitives/evaluate.py` | A1 (click decode policy options) | Modify |
| `experiments/action_primitives/train.py` | A3 (compute aux loss, log target_acc), A4 (apply head_weights from config), F4 (bump default epochs) | Modify |
| `experiments/action_primitives/augment.py` | A2 (new module) | Create |
| `tests/action_primitives/test_*.py` | Unit tests for each fix | Modify/Create |
| `docs/experiments/6-action-primitives.md` | Gate update note (in Amendments section) | Modify |
| `docs/plans/2026-04-25-action-primitives-phase-b0-design.md` | Gate update + bundled-fixes pointer | Modify |

---

## Phased Tasks

The plan is broken into 7 phases, each landing one or more atomic commits. Each commit is independently testable (TDD: write failing test, implement, watch test pass, commit).

### Phase 0: Pre-flight
- 0.1 Verify clean working tree on `feat/exp6-phase-b0`
- 0.2 Run baseline test suite to capture pre-change state
- 0.3 Commit prior session's metrics.py sign_acc fix if uncommitted

### Phase 1: Pure config knobs (F1 + F4 + A4)
- 1.1 LossConfig: gamma 3->2, click_class_weight=(1,5,5), head_weights with scroll/keys/done at 0.1; `_focal_ce_masked` accepts class_weight; `total_loss_b0` wires it. **Test:** weighted loss > unweighted on press-error sample.
- 1.2 Wire `LOSS.head_weights` into `train.py`'s `total_loss_b0` call; verify zero-weight heads zero out. **Test:** total_loss with weight=0 < total_loss with weight=1.
- 1.3 Add `phase_b0_epochs: int = 16` to TrainConfig; SageMaker launcher passes `--epochs 16` explicitly.

### Phase 2: Image aug + scene-distribution skew (A2 + F3)
- 2.1 Create `augment.py` with `color_jitter`, `spatial_jitter`, `augment_episode_images` (episode-consistent spatial). Add `AugConfig` to config. Wire into `PhaseB0EpisodeDataset.__getitem__` train-only. **Tests:** color_jitter changes pixels, spatial_jitter preserves canvas size, episode-consistency, train_augment flag respects split.
- 2.2 Add `n_buttons_distribution` parameter to `generate_scene` (default uniform). CLI flag `--n-buttons-distribution` on `generate_data.py` (default `0.15,0.20,0.20,0.20,0.15,0.10`). **Tests:** distribution skew measurably differs from uniform; default uniform unchanged.
- 2.3 Cleanup smoke-test data dir.

### Phase 3: Text-tower LoRA (F2)
- 3.1 Extend `apply_lora` with `text_rank` and `text_target_layers` kwargs. When >0, apply LoRA to top N text encoder layers' attn projections. Drop `@torch.no_grad()` from `encode_text` (callers must handle). Add `MODEL.text_lora_rank=4`, `MODEL.text_lora_target_layers=2`. Wire into `model.py`. **Tests:** text_rank>0 produces trainable text params, <1M; text_rank=0 keeps text fully frozen. Verify optimizer's param-group construction picks up text LoRA params.

### Phase 4: Auxiliary target-button head (A3)
- 4.1 Add `AuxTargetHead` class (2-layer MLP: d_model -> 256 -> max_buttons). Add `MODEL.aux_target_*` config (enabled, max_buttons=6, first_n_frames=3, hidden_dim=256). **Tests:** output shape (B, max_buttons), <1M params.
- 4.2 Wire into `ActionPrimitivesACT.forward`: pool over query tokens (mean), feed to aux head, return `aux_target_logits` on `ACTOutput`. Backward-compat when disabled. **Test:** forward returns aux_target_logits when enabled.
- 4.3 Add `aux_target_loss(logits, targets, n_buttons, episode_frame_idx, first_n)` to losses.py. Slot mask invalid button slots to -inf; episode-frame mask zeros out frames >= first_n. Integrate into `total_loss_b0` via optional kwargs. **Tests:** out-of-scope frames produce 0 loss; invalid slots masked correctly.
- 4.4 Plumb episode_frame_idx + target_button_id + n_buttons through dataset's __getitem__ output and train.py's micro-batch assembly. **Test:** dataset emits `episode_frame_idx` (T,) and metadata has `target_button_id`.
- 4.5 Log `val/diag/aux_target/{acc_all, acc_first_n, acc_multibtn}` in train.py's val diagnostics block. Critical signal: `acc_first_n` matches the loss scope; `acc_multibtn` isolates the gate-relevant subset.

### Phase 5: Click decode policy at inference (A1)
- 5.1 Add `_decode_b0_click_v2(head_logits, temperature, press_threshold)` and `temporal_smooth_click_history(history)` to evaluate.py. CLI flags `--click-temperature`, `--click-press-threshold`, `--click-smooth-window`. Rollout loop maintains a deque of last N click logits per button; smoothes before decode. **Tests:** temperature scales softmax; threshold fires press below argmax; 3-frame smoothing dampens single-frame spikes.

### Phase 6: Documentation
- 6.1 Update B0 design doc: gate value 0.92 -> 0.80 with note. Add Amendments entry to `docs/experiments/6-action-primitives.md` summarizing attempt-1 results + fix bundle.

### Phase 7: Integration smoke + data regen + training
- 7.1 Local CPU smoke: 5-episode dataset + 1-step train. Verify loss/aux_target appears, no nan/error.
- 7.2 M1 data regen: 10k episodes, new distribution. Upload to `PenTest-duck/cu-vla-exp6-b0-lclick-v2`.
- 7.3 SageMaker launch: `ml.g6e.xlarge` spot, `--epochs 16`, `--micro-batch-episodes 16`, `--num-workers 8`, `--early-stop-patience 5`. W&B run name `phase-b0-bundled-fixes`.
- 7.4 SageMaker eval suite: `--n-rollouts 500` (was 200 in attempt 1, per B4 strategic note); pass new click-decode flags. Upload JSONs to `PenTest-duck/cu-vla-exp6-b0-ckpt-v2`.
- 7.5 Write up `docs/experiments/6-action-primitives-phase-b0-attempt-2-results.md` with gate-by-gate pass/fail.

---

## Critical signals to watch during training

If any of these don't hold, kill and debug before burning more credits:

1. **`val/total` drops within first 200 steps** - sanity check; matches attempt 1's first 200-step trajectory.
2. **`loss/aux_target` is > 0 initially** - means aux head is wired correctly.
3. **`val/diag/aux_target/acc_first_n` rises above random (1/6 = 0.167) within first 200 steps** - means trunk is learning text-grounding.
4. **`val/diag/click_left/recall_press` more stable than attempt 1's 0.46-0.97 oscillation** - F1 (gamma + class weights) working.
5. **`val/diag/dx/sign_acc` shows meaningful number** post idle-mask fix - was 0.19 (metric-broken), should now be ~0.85+ if motor is healthy.
6. **No `nan` in any loss component** - integration health.

---

## Cost estimate

| Item | Cost |
|---|---|
| Local M1 data regen (10k episodes) | ~$0 (electricity) |
| SageMaker training (ml.g6e.xlarge spot, ~3-4h for 2000 steps) | $3-5 |
| SageMaker eval suite (8 evals × 500 rollouts, ~1-2h) | $1-2 |
| **Total** | **$4-7** |

vs. the user's $10k AWS grant budget, well within tolerance.

---

## Self-Review

- [x] All 8 fixes covered by tasks (F1, F2, F3, F4, A1, A2, A3, A4)
- [x] Gate change (0.92 -> 0.80) documented (Phase 6)
- [x] Type consistency: ACTOutput.aux_target_logits, AuxTargetHead, aux_target_loss
- [x] Test for each fix included
- [x] CPU smoke before SageMaker (Phase 7.1)
- [x] Critical signals defined for early kill of doomed runs

---

## Next-step routing

- **All gates pass** -> start B1 planning (additional primitives).
- **Gates miss but progress vs attempt 1 is substantial (>2x)** -> iterate on the failing gate's specific disposition (per the design doc's typed-disposition table).
- **Gates miss with no progress** -> step back: this becomes a signal that the architecture (frozen text tower, single-step decoding, etc.) needs the bigger reframings (B1, B2 from the strategic discussion).

---

## REVISIONS POST GPT-5.4 REVIEW (2026-04-27)

A critical review by GPT-5.4 identified three serious flaws in the original plan that would have caused attempt 2 to fail. Implementation reflects the revised bundle below.

### Critical findings from review

1. **A3 was wrong as designed**: `target_button_id` in the parquet is the RNG-shuffled button-creation order, not a semantic position slot. Predicting it teaches the head RNG-permutation priors, not grounding.
2. **F2 had hidden bugs**: `train.py` wraps `encode_text` in `torch.no_grad()` (lines 136-137, 199-200) AND passes `torch.ones(...)` as text mask (lines 150, 222) instead of the real attention mask. F2 (text LoRA) is a no-op without fixing both. **These are real existing bugs from attempt 1 that may have caused much of the 9 pp wrong-instruction degradation on their own.**
3. **A2 spatial jitter was mechanistically wrong**: cursor position in proprio isn't jittered, creating a screenshot↔cursor inconsistency that contradicts what the model sees.

### Revised bundle

| Fix | Status |
|---|---|
| F1: focal_gamma 3->2 + click_class_weight | KEEP (confirmed defensible) |
| F4: 2000 steps | KEEP |
| A4: head weight rebalance | KEEP |
| F2: text LoRA rank-4 top-2 layers | KEEP **+ fix text no_grad + text-mask bugs** (now coupled) |
| A3: aux head | **REDESIGNED** — predict target's grid cell (from `target_bbox_x/y` already in parquet, quantized to B0_POSITION_GRID = 3x2 = 6 cells); apply loss only on `episode_frame_idx == 0` (history is naturally zero there); use flattened query state (16 × 768), not pooled. No data regen needed. |
| A2: image augmentation | **DROPPED** — color jitter risks corrupting grounding (color is a supervised attribute); spatial jitter is wrong (cursor proprio inconsistency) |
| F3: scene distribution skew | **DROPPED** — too marginal (16.7% -> 15% 1-button); also requires data regen which user wants to avoid |
| A1: click decode policy at inference | **DROPPED** — eval-time only; user will drive eval next morning and can tune A1 then if needed |
| Gate: 0.92 -> 0.80 | KEEP (framed as soft stretch target, not principled recalibration) |

### Items also flagged but deferred

- Wrong-instruction probe contamination (always uses 2-attribute "color+shape" instruction): documented as a measurement caveat; doesn't gate training. User will revisit during eval.
- Probe suite only runs `zero/shuffled/wrong` on `multi_btn_generic`, not `multi_btn_composite`: documented; user can add when running eval.
- 1250->2000 is +60% not +30% (my original arithmetic was off): noted, no implementation impact.

### What this bundle gets us

The remaining 5 fixes (F1, F2 + text-path bug fixes, F4, A3-redesigned, A4) are mechanistically the most defensible per the review. F2's coupled text-path bug fix is the highest-impact single change — attempt 1 literally couldn't backprop through text. Even if A3 helps less than hoped, F2-with-bug-fixes alone should move the wrong-instruction gate substantially.
