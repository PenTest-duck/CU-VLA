# Experiment 6 — Phase B0 Design

**Date:** 2026-04-25
**Branch:** `feat/exp6-phase-b` (worktree: `feat/exp6-phase-b0` for B0 sub-phase)
**Status:** design v1 — derived from brainstorm session against Phase A close-out
**Predecessor:** [Phase A summary](../experiments/6-action-primitives-phase-a-results/PHASE-A-SUMMARY.md) and [design doc amendments](../experiments/6-action-primitives.md#amendments-phase-a-findings-2026-04-24)
**Successor:** Phase B1 design (TBD at B0 close-out)

## TL;DR

Phase B is staged into three sub-phases: **B0 (combined L-click hardening + V+L grounding)** → **B1 (multi-primitive validation, 1-3 primitives)** → **B2 (full v1)**. This document specifies B0.

B0 is **not** the original "L-click loss-side fixes only" sub-phase from the Phase A handoff. The user's pushback during brainstorming converted it into a richer design that simultaneously:

1. Applies the four loss/data-side fixes identified in Spike B (soft-label CE, click-head rework, scenario-error recovery, instruction-aware expert).
2. Introduces **distractor scenes** with attribute-grounded instructions, making the SigLIP2 text tower load-bearing for the first time.
3. Tests V+L grounding (the actual project goal) at the cheapest scope where it can be cleanly attributed — before multi-primitive interference can confound results.

Single regen, single retrain, single push. Four-criterion gate must pass to ship to B1.

**Cost & wall-clock:** ~$2.78 / ~2 hr end-to-end (a100-large training + M1 post-training eval).

## Phase B structure

| Sub-phase | Scope | Compute | Wall-clock | Gate |
|---|---|---|---|---|
| **B0** | L-click + distractors + grounded instructions + recovery trajectories. Single primitive, full V+L stack. | a100-large, ~$3 | ~2 weeks | 4-criterion (below) |
| **B1** | Hardened L-click + 1-3 additional primitives. Validates multi-task plumbing at intermediate scale. | a100-large, ~$10-12 | ~2-3 weeks | TBD at B0 close |
| **B2** | Full v1 — 6 primitives in single push. | a100-large or h200, ~$20-30 | ~3-4 weeks | Per design doc Q24 |

**Why staged:**
- B0's combined design tests V+L grounding at the cheapest scope, where it can be attributed.
- B1 catches multi-task interference at intermediate scale before committing to full-v1 data gen.
- B2 is the originally-planned full v1, contingent on B0 + B1 success.

## Architecture changes (B0)

Phase A's model is preserved except for two head/loss modifications. Total trainable params unchanged at ~33.6M.

### Click head — 5-way → two parallel 3-way heads {idle, press, release}, per button

**Why:** Spike B revealed `l_press` recall = 0.79 (vs aggregate click 0.99) is the dominant residual bottleneck (~21% press-timing miss matches the ~18% closed-loop residual after probabilistic decode).

**Design:**
- Replace single 5-way categorical head with **two parallel 3-way heads**, one per physical button (left, right). Each emits {idle, press, release}.
- Physically grounded — matches the project's held-state / event-level philosophy. Auto-extends to middle-click later by adding a third 3-way head.
- Keys head structure unchanged (77 × 3-way) — click is now structurally consistent with keys.
- Each 3-way head uses focal-CE with idle-biased smoothing (existing `losses.py`); focal **γ default = 3** (sweepable {2, 3, 5} if `l_press` recall is borderline post-training).
- Per-class loss weights NOT used (avoids per-primitive re-tuning across B1/B2).

**Multi-primitive transferability:** ★★★★★ — structural fix; auto-adapts as primitive mix changes (drag adds many click-active frames; scroll adds none). No per-primitive weight retuning. **Confidence 7/10** for B0; **8/10** for B1/B2.

**Trade-off accepted:** two independent heads can in principle output simultaneous L+R press on a single frame. Training data won't have this; model will learn it's effectively zero-probability without architectural enforcement. **Confidence 7/10** this is fine.

### dx/dy/scroll heads — hard-label CE → 2-bin triangular soft-label CE

**Why:** Phase A trained with hard-label argmax then discovered probabilistic decode (`E[bin_center]` under softmax) gives +11 pp closed-loop "for free" at inference. B0 closes the train/eval gap.

**Design:**
- Label is a 2-bin distribution: weights linearly interpolated between the two bins bracketing the expert's continuous action.
- E.g., expert action = 25.6 px, bins {16, 40} → label = {bin 18: 0.6, bin 19: 0.4}.
- Trains the softmax to BE a properly distributed estimator; probabilistic decode at inference operates on the same target the loss optimized.
- Falls back cleanly to hard-label argmax when expert action lands exactly on a bin center.
- Implementation: `losses.soft_label_ce(logits, expert_continuous, bin_centers)` constructs distribution on-the-fly. ~30 lines.

**Lit grounding:** standard pattern in discrete-distribution heads (DDPM noise prediction, OpenVLA-OFT bin smoothing). **Confidence 7/10** that 2-bin triangular suffices vs wider Gaussian smoothing — the precedent supports soft targets but doesn't directly prove sufficiency for our exponential-bin scheme + expected-value decode.

**Multi-primitive transferability:** ★★★★★ — applies universally to all primitives' continuous-action heads.

### Other heads (done, scroll, keys) — unchanged

### Text tower — kept

The Phase A "drop the tower" reasoning was based on instructions being trivial for L-click only. Combined B0 reverses this: instructions are now load-bearing for target disambiguation under distractors. Tower stays at 282M frozen params; LoRA stays vision-only. **Confidence 8/10** (the reasoning is sound; 8 not 9 because we haven't yet empirically confirmed the model uses the text tower — the wrong-instruction probe will).

## Data design (B0)

### Scene generator

Each scene contains:
- **1-6 buttons** (uniform discrete sample). 1-button scenes serve as the natural Phase-A holdout (no separate generator).
- **0-3 decorative non-clickable shapes** per scene. Rendered with non-button visual style (no border, semi-transparent, smaller). Forces the model to distinguish clickable targets from background visual elements.
- **Randomized solid pastel background color** per episode.

Per-button attributes sampled independently:

| Attribute | Values | Notes |
|---|---|---|
| Color | 8-10 discrete (red, blue, green, yellow, orange, purple, pink, cyan, white, black) | High color contrast across palette |
| Shape | 5 discrete (rect, circle, triangle, square, hexagon) | Pygame-renderable primitives |
| Size | 3 categorical (small/medium/large) | Concrete pixel ranges per category |
| Position | 3×3 absolute grid (top-left ... bottom-right) | Discretized canvas zones |

**Deferred to B1+:** pattern (texture fill), text labels, relative position ("left of X"), border/outline, themes, icons.

**Why these four and not others:**
- Color + shape + size are canonical V+L grounding attributes (VIMA, RT-2, OpenVLA). **Confidence 9/10** they belong in v1.
- Absolute position is cheap to render and a fundamentally distinct grounding skill.
- Text labels are conceptually adjacent to the type primitive — keep B0 click-focused.
- Relative position is a fundamentally harder grounding skill; deferring isolates B0 failures.

### Instruction generator

**Inverted flow:** scene generated first, then instruction sampled to uniquely identify ONE target (instruction-uniqueness invariant).

**Composite distribution:**
| Tier | Fraction | Example |
|---|---|---|
| 1 attribute | ~60% | "click the red button" |
| 2 attributes | ~30% | "click the small red one" |
| 3 attributes | ~10% | "click the large red square in the top-left" |

**Phrasing diversity:** 3-5 templates per attribute combination ("click the X", "select the X", "press the X", "tap the X", "find and click X"). Length distribution: short (3-5 words) majority, occasional 8+ words for full composite.

**Typos:** ~3-5% of instructions get character-level typos (random char swap/drop/insert).

**For 1-button scenes:** instruction can be generic ("click the button"). Uniqueness is trivially satisfied.

### Adversarial scenes (~25%)

Scenes deliberately designed s.t. single-attribute disambiguation fails. E.g., 2 red buttons forces "the small red"; 3 squares forces shape+color; clustered positions force position+attribute. Flagged with `is_adversarial=True` for eval-time slicing.

### Multimodality, negatives, other variability

- **Exactly 1 correct target per scene.** Instruction-uniqueness invariant guarantees this. Multimodality deferred to B1+.
- **No negative examples** (impossibility scenes). Deferred to B1+ where impossibility-detection is more meaningful (e.g., "type X" with no input field).
- **Cursor starting position:** random per episode (Phase A behavior).
- **Tempo profiles:** slow/medium/fast/superhuman bursts (Phase A's Q9 design).

### Expert — instruction-aware

`LClickExpert` extended to take `(scene, instruction)`:
1. Parse instruction → identify target button id from scene metadata.
2. Run Fitts-law trajectory + tempo profile to that target.
3. Re-queryable from any cursor state (used in recovery training).
4. Recovery is **always** toward the instruction-specified target — never toward the nearest distractor.

## Recovery trajectory design (B0)

Two complementary mechanisms cover different drift regimes:
1. **Start-chunk scenario errors** for explicit large-magnitude failure types.
2. **DART-style action noise** for continuous small-magnitude drift coverage.

This pairing was settled after extensive iteration — see "Design history" below for what was considered and rejected. Mid-trajectory chunk injection was specifically considered (per GPT-5.5 review) and superseded by DART, which provides more natural distribution coverage at lower complexity.

### Episode classes

**Standard episodes (~80-85%):** clean expert with DART action noise sprinkled across navigation frames (see DART section). `actions_applied` may differ from `actions_labeled` on noisy frames; loss_mask=1 throughout.

**Scenario-error episodes (~15-20%):** wrong segment at episode start, clean expert recovery thereafter.

### Mechanism 1 — Start-chunk scenario errors

**Wrong-segment generator:**
- **Length K:** uniform [5, 15] frames per scenario episode.
- **Type distribution:**
  - 50% **wrong-direction** — cursor heads ~90-180° away from instruction-specified target for K frames.
  - 30% **overshoot** — cursor passes target by 1.5-2× target size, then needs to reverse.
  - 20% **edge-bang** — cursor pinned to canvas edge for 3-5 frames within the K-frame window.
- **Position in episode:** wrong segment is the **first K frames**. Cursor starts at random position; takes wrong actions for K frames; then clean expert kicks in.

**Loss masking:**
- Frames 1..K (wrong segment): `loss_mask = 0`. Populates action-history vector and env state, contributes **zero gradient**.
- Frames K+1..N (recovery): `loss_mask = 1`. Standard training targets (clean expert action from each state).
- **Invariant:** model is never trained to emit a wrong action.

### Mechanism 2 — DART-style action noise (in clean episodes)

**Mechanism:** during data generation, with probability *p* per navigation frame, replace the expert's action with `expert_action + Gaussian(0, σ²)` on dx and dy. Env steps deterministically from the noisy action; cursor lands at a slightly off-trajectory position. Next frame's clean expert is re-queried from the actual env state for the **label**.

**Defaults:**
- *p* = 0.08 (per navigation frame in clean-phase episodes)
- σ = 20 px on dx and dy independently
- **Never applied** to click event frames (l_press, l_release) or to the 1-2 frames immediately preceding/following them (preserves click semantics).
- **Never applied** to scenario-error episodes' wrong segments (already off-trajectory by design).

**Loss masking:** loss_mask = 1 on noisy frames (label is the clean expert action from the resulting state — model is trained to recover, never to emit noise).

**Coherence:** physics-consistent throughout — `action_applied[t]` (noisy) genuinely produced state[t+1] under env physics. The action-history vector contains real applied actions; no teleport, no broken transitions.

### Drift regime coverage (combined)

| Drift magnitude | Frequency at inference | Covered by |
|---|---|---|
| Near-zero (clean expert path) | Most common | Clean expert episodes |
| Small drift (5-30 px) | Common | DART small perturbation |
| Medium cumulative drift (30-100 px) | Less common | DART cumulative over N frames (~σ·√(N·p)) |
| Large sudden drift (50-200 px) | Rare failure mode | Start-chunk scenarios |

### Per-episode and per-frame metadata

- Per-episode: `is_scenario_error` flag, `scenario_type` ∈ {wrong-dir, overshoot, edge-bang, none}, `k_wrong_frames`.
- Per-frame: `loss_mask`, `is_dart_noisy_frame` (for DART instrumentation / ablation).

### Design history (what was considered and rejected)

- **State teleport per-frame:** rejected on coherence grounds (history vector becomes physically impossible). Different from action noise — see notes.
- **Online perturbation at training time:** rejected on plumbing grounds (would require pygame in DataLoader workers + re-encoding through SigLIP2 per perturbed frame).
- **Mid-trajectory wrong-segment injection (5-15-frame chunks at random episode positions):** considered after GPT-5.5 review surfaced "wrong-at-start is too artificial." Superseded by DART because (a) Phase A's documented failure manifest contains zero mid-trajectory chunky failures (all "stuck far away" failures are near-start); (b) DART provides more natural distribution coverage with loss_mask=1 (more sample-efficient than chunks with loss_mask=0); (c) cleaner attribution (one fewer mechanism in B0).
- **Closed-loop DAgger:** deferred. Synthetic perturbation gives most of the same value without rollout-in-training overhead.

### Lit grounding (recovery design)

- DART (Laskey et al., 2017): action-noise + expert re-query is the canonical off-trajectory coverage technique. **Confidence 7/10** the technique applies cleanly to our setup.
- Diffusion Policy / Implicit BC: off-trajectory state coverage helps. **Confidence 7/10** the principle generalizes.
- "Burn-in / warmup" pattern in sequence learning: load context, predict from coherent point onwards. **Confidence 7/10** matches the start-chunk masking approach.

### Open question for B1 close-out

If B0 surfaces mid-trajectory chunky failures we haven't seen in Phase A, mid-trajectory injection re-enters the design palette for B1. For B0 it's deliberately omitted.

## Infrastructure (B0)

### Multiproc data generator

**Replaces** Phase A's `Pool.imap` (IPC-bound, slower than serial at chunksize=1).

**Design:** worker-writes-shards.
- N worker processes, each owns a contiguous episode-id range.
- Each worker initializes its own pygame instance ONCE at startup.
- Each worker writes its OWN parquet shard to disk independently — no IPC for episode data.
- Main process spawns workers, waits, reports progress via tqdm aggregating from atomic counter or pid files.

**Throughput target:** 25-30 eps/s on 8-core M1 (6-8× over Phase A's 3.6 eps/s). **Confidence 8/10** based on Spike E's analysis.

**Reusable** for B1/B2 data gen.

### Dataset spec

| Field | Value |
|---|---|
| Total episodes | 10,000 |
| Train/val/test split | 8K / 1K / 1K |
| Standard / scenario-error mix | 80-85% / 15-20% |
| Scene size distribution | uniform 1-6 buttons (1-button slice ≈ 15%) |
| Adversarial fraction | ~25% (within multi-button scenes) |
| Storage | ~500 MB local; HF Hub `PenTest-duck/cu-vla-exp6-b0-lclick` |
| Generation time | ~7 min on 8-core M1 with multiproc |

**Parquet schema additions** over Phase A:
- `loss_mask` (int per frame)
- `is_scenario_error` (bool per episode)
- `scenario_type` (str per episode)
- `k_wrong_frames` (int per episode)
- `target_button_id` (int per frame)
- `instruction` (str per frame; same value across frames within episode)
- `n_buttons` (int per episode)
- `composite_tier` (int per episode: 1/2/3 attributes)
- `is_adversarial` (bool per episode)

### Compute

| Setting | Value |
|---|---|
| Hardware | HF Jobs `a100-large` flavor: 1× A100 80GB, 12 vCPU, 142 GB RAM |
| Cost | $2.50/hr |
| `--micro-batch-episodes` | 16 (vs L4's 4) |
| `--num-workers` | 8 |
| Macro batch | 64 episodes (unchanged) |
| Epochs | 10 (vs Phase A's 5) — accommodates 10× larger dataset + harder task |
| Wall-clock estimate | ~1.1 hr |
| Total cost | **~$2.78** |
| Crash recovery | `best.pt` + step checkpoints to HF Hub `PenTest-duck/cu-vla-exp6-b0-ckpt` |

**Why a100-large:** dramatically better cost-time efficiency than L4 ($2.78 / 1.1 hr vs $5.36 / 6.7 hr). 80GB GPU enables `micro=16` (vs L4's 4) which cuts step overhead. 12 vCPU + 142 GB RAM removes any DataLoader bottleneck. **Confidence 8/10.**

H200 was considered but offers marginal gain at same total cost; A100 is the proven default.

## Evaluation design (B0)

Eval is structured for **diagnostic separability** (per GPT-5.5 review item 1) — many fine-grained slices feed into a typed-disposition gate (per item 8).

### Eval sets — diagnostic slices

All eval sets are filtered slices of the test split (1K episodes) plus a derived set for the wrong-instruction probe. No separate generators.

| Eval slice | Filter | Expected n | Purpose |
|---|---|---|---|
| **Phase-A holdout** | `n_buttons == 1 AND not is_scenario_error` | ~150 | Apples-to-apples Phase A regression check |
| **Multi-btn generic** | `n_buttons >= 2 AND composite_tier == 1 AND not is_adversarial` | ~360 | Single-attribute grounding |
| **Multi-btn composite** | `n_buttons >= 2 AND composite_tier >= 2 AND not is_adversarial` | ~240 | Two-/three-attribute grounding |
| **Scenario-recovery** | `is_scenario_error == True` | ~150 | Wrong-segment recovery |
| **Adversarial subset** | `is_adversarial == True` | ~250 | Disambiguation under stress |

(The "with-distractor main" metric reported in earlier sections = union of "multi-btn generic" + "multi-btn composite" + "adversarial".)

### Adversarial subset tiering (per review item 7)

Adversarial slice further reported by which disambiguator is required:
- **color-ambiguous** — multiple matching color, other attributes disambiguate
- **shape-ambiguous** — multiple matching shape
- **size-ambiguous** — multiple matching size
- **position-ambiguous** — clustered positions force position+attribute
- **2-attr-needed** — single-attribute insufficient on any axis
- **3-attr-needed** — two-attribute insufficient

Per-tier success rates surface which grounding relation is failing. Not gated individually but informs the typed-disposition routing.

### Instruction-grounding probes (per review item 6)

Three probes, run on the same scenes as Multi-btn generic + composite:

| Probe | Modification at inference | What it tests |
|---|---|---|
| **Zero-instruction** | Instruction embedding replaced with zero vector | Baseline reliance check |
| **Shuffled-instruction** | Instruction embedding replaced with embedding of a random other val-set instruction | Noise-robust language reliance (corrects for OOD-zero-embedding artifacts) |
| **Wrong-instruction** | Scene unchanged; instruction targets a different button than the rendered "correct" target | Hardest test: model must follow instruction even when the visually-salient or scene-prior target is the "wrong" answer |

All three should produce significant degradation if grounding is real. Zero and Shuffled measure noise-robustness; Wrong is the strongest test.

### Typed-disposition gate (per review item 8)

Replaces the previous "all-or-nothing 4-criterion" gate. Failure routes to a typed remediation path.

**Hard pass requirements (all must hold to ship B0 → B1):**

| Criterion | Threshold | Failure type if missed |
|---|---|---|
| **Phase-A holdout closed-loop** | ≥0.92 | **Phase-A-regression** |
| **Phase-A holdout l_press recall** | ≥0.90 | **Phase-A-regression** |
| **Multi-btn generic + composite (combined) closed-loop** | ≥0.85 | **Motor or grounding fail** (sub-routed by probes) |
| **Wrong-instruction degradation** | ≥40 pp from baseline | **Grounding-fail** |
| **Adversarial subset closed-loop** | ≥0.75 | **Grounding-fail** if probe degradations weak; **Motor-fail** if probe degradations strong |
| **Scenario-recovery closed-loop** | ≥0.80 | **Recovery-fail** |

**Soft warnings (don't block ship, but flag in write-up):**
- Zero-instruction degradation <30 pp: noise-robust grounding signal weak.
- Shuffled-instruction degradation <25 pp: language conditioning weak.
- Per-tier adversarial breakdown shows one tier <0.50: grounding gap on a specific relation.
- Bin-10 frequency on dx/dy >0.20 at frames where expert delta >5px: zero-motion attractor regime.

**Typed disposition table (failure routing):**

| Failure type | Trigger | Remediation path |
|---|---|---|
| **Phase-A-regression** | Phase-A holdout fails | Loss-fix bug or training instability — debug head/loss code first |
| **Motor-fail** | Multi-btn fails AND probe degradations strong (≥40 pp) | Click-head or dx/dy issue — sweep gamma, retrain |
| **Grounding-fail** | Multi-btn fails AND probe degradations weak | Text tower / fusion issue — investigate training-time instruction signal |
| **Recovery-fail** | Scenario-recovery fails specifically | Wrong-segment or DART design — increase fraction or σ |

### Tier-1.5 metrics (kept from Phase A's Q24 amendment)

Success @ {0, 3, 5, 10}px tolerance for click endpoints. Reported per eval slice as diagnostic curves; not gating.

### Closed-loop config

- n=200 rollouts per eval slice.
- **Probabilistic decode** (`E[bin_center]` under softmax) — Phase A default per amendments.
- Argmax decode optional via `--decode argmax` flag for reproducibility.

### Cost-aware eval cadence

**During training (A100, $2.50/hr):**
- **Tier-A only** every val pass: per-head accuracy, per-class click recall, bin-10 freq, instruction-zeroing val loss. ~30 sec per pass. GPU-native, batched.
- Early stopping on aggregate val loss (Tier-A). Patience=3.
- **No closed-loop eval during training.** Saves ~40 min job time / ~$1.70.

**Post-training (M1 MPS, $0 marginal):**
- Pull `best.pt` locally.
- Run full 4-criterion gate suite. ~50 min total wall-clock.

**Optional mid-training debug probe** (A100, ~$0.20 added if enabled):
- One small closed-loop sample (n=20, single eval set) at end of every 2nd epoch. Toggle via `--debug-rollout-every-n-epochs 2` flag. Default OFF for B0.

**Vectorized closed-loop eval** (deferred to B1/B2): vectorized env wrapper to amortize forward-pass cost ~10-20×. Implementation ~1 day; payoff scales with rollout count. Not for B0.

## Diagnostics during training

**Per-step W&B logging:**
- Aggregate loss (Phase A baseline).
- Per-head loss (Phase A baseline).
- **Per-class click recall** (NEW): idle/press/release for both left and right heads, computed on val mini-batch every step. Surfaces `l_press` recall before aggregate masks it.
- **Soft-CE diagnostic suite** (NEW, per review item 5): for dx and dy heads, log
  - **Sign accuracy** — fraction of frames where predicted expected-value sign matches expert sign. Catches "averaging across two valid wrong directions" failure mode.
  - **Softmax entropy** — mean entropy of dx/dy distributions per step. Spikes indicate the model is genuinely uncertain (could be bimodal).
  - **Mass on wrong-sign bins** — fraction of softmax mass on bins with opposite sign to expert. Diagnostic for bimodal collapse.
  - **Expected-value endpoint error** — L1 distance from `E[bin_center]` to expert continuous action. Tracks the metric probabilistic decode actually uses.
- **Bin-10 frequency** (NEW, demoted from gating): `frac(predicted_dx_bin == 10)` and `frac(predicted_dy_bin == 10)` on frames where expert delta > 5px. Early warning of zero-motion attractor.
- **Wrong-direction-first-3-frames rate** (NEW): at Tier-B eval cadence (post-training only), fraction of rollouts where cursor moves AWAY from instruction-target in first 3 frames. Spike B identified this as a 15-18% residual failure mode invisible to frame-level metrics.
- **Instruction-grounding probes during val passes** (NEW, expanded per review item 6):
  - Zero-instruction val loss (instruction embedding zeroed)
  - Shuffled-instruction val loss (random other instruction's embedding)
  - Wrong-instruction val accuracy (instruction targets different button than rendered correct one)
- Standard hygiene: per-param-group grad norms, LR schedule, NaN detector, weight EMA tracker.

**Optimizer + LR (unchanged from Phase A):** AdamW two-param-group (trunk lr 3e-4, LoRA lr 2e-4), betas (0.9, 0.95), wd 0.01, linear warmup 100 → cosine decay to 10%, bf16 autocast, grad clip 1.0.

## Implementation timeline

~7-8 working days, ~2 weeks calendar.

### Day 1-2: Generator infrastructure + scene/instruction system

- New `scene.py` (or extend `env.py`): scene generator with 1-6 buttons, attribute sampling, decorative shapes, adversarial flagging.
- New `instructions.py`: instruction-resolution from scene, attribute composition (60/30/10), 3-5 phrasing templates, ~3-5% typo injection.
- Extend `LClickExpert`: instruction-aware target selection, re-queryable from any cursor state.
- Extend `LClickEnv` rendering for new attribute palette + decorative shapes.
- Multiproc generator: worker-writes-shards, per-worker pygame init, atomic counter for tqdm.

### Day 3: Recovery trajectory generator

- Start-chunk wrong-segment generator: wrong-direction / overshoot / edge-bang. K∈[5,15], 50/30/20 mix.
- DART action-noise mechanism: per-frame Gaussian on dx/dy with p=0.08, σ=20 px. Excluded from click frames + scenario-error wrong segments. Expert re-query for clean labels.
- Loss-mask plumbing: parquet schema field, dataset loader passthrough, `losses.total_loss` masks per-head losses.
- `is_dart_noisy_frame` per-frame instrumentation flag (for ablation if needed).
- Unit tests for `actions_applied` vs `actions_labeled` invariants on both mechanisms.

### Day 4: Architecture + loss changes

- `heads.py`: replace 5-way click with two parallel 3-way heads; update flatten/forward/checkpoint loading.
- `losses.py`: add `soft_label_ce` for dx/dy/scroll; per-class recall; bin-10 freq metric.
- `train.py`: new diagnostic logging; instruction-zeroing val pass; early stopping unchanged.
- Unit tests for new losses + checkpoint compatibility (Phase A checkpoint should NOT load — architecture changed; verify clean error).

### Day 5: Data generation + upload

- Create HF dataset repo `PenTest-duck/cu-vla-exp6-b0-lclick`.
- Run multiproc generator: 10K episodes (~7 min on 8-core M1).
- Inspect random samples: instruction uniqueness, scenario-error correctness, loss-mask validity.
- Upload via existing `hf_sync.upload_parquet_dir`.

### Day 6: Training run on A100

- Update `scripts/launch_hf_job_exp6.py` for `a100-large` flavor.
- Launch: `--epochs 10 --micro-batch-episodes 16 --num-workers 8 --eval-every-steps 50` (no closed-loop in-training).
- ~1.1 hr wall-clock, ~$2.78.
- Pull `best.pt` to local.

### Day 7: Post-training closed-loop eval

- Extend `evaluate.py`: filter eval splits by metadata (n_buttons, is_scenario_error, is_adversarial, composite_tier, adversarial-tier). Implement zero/shuffled/wrong-instruction probes.
- Run all eval slices on M1 MPS, n=200, probabilistic decode. Slice count is now 5 (Phase-A holdout, multi-btn generic, multi-btn composite, scenario-recovery, adversarial) + 3 probes — total ~80 min.
- Compute per-slice + per-tier (adversarial sub-tiers) metrics + Tier-1.5 tolerance curves.
- Generate the typed-disposition table for the write-up.

### Day 8: Write-up + decision

- `docs/experiments/6-action-primitives-phase-b-results/spike-b0-combined.md`: methodology, results per gate, failure-mode visual sample, gate pass/fail.
- If all four gates pass → start B1 planning.
- If any fail → per-failure debug + remediation. No automatic fallback path.

## Branching + deliverables

**Branch:** `feat/exp6-phase-b` (long-running). Worktree `feat/exp6-phase-b0` for B0 implementation work. PR at end of B0 close-out.

**Deliverables at B0 close:**
- Updated experiment code in `experiments/action_primitives/`.
- New unit tests in `tests/action_primitives/`.
- 10K-episode dataset on HF Hub: `PenTest-duck/cu-vla-exp6-b0-lclick`.
- Trained checkpoint on HF Hub: `PenTest-duck/cu-vla-exp6-b0-ckpt`.
- Spike B0 write-up at `docs/experiments/6-action-primitives-phase-b-results/spike-b0-combined.md`.
- Design-doc amendments for B1 planning (next chapter).

## Open items deferred to B1+ planning

- Which 1-3 primitives to add in B1 (candidates from full-v1 list: type-string, scroll-to-target, drag, hover, copy-paste).
- Vectorized closed-loop eval implementation (worth implementing once rollout volume justifies it).
- Negative examples (impossibility detection) and corresponding head/loss design.
- Multimodality (multiple correct targets) loss design.
- Text labels and relative-position attributes.
- Themed scenes / icon UI mocks.

## Confidence summary

| Decision | Confidence |
|---|---|
| Three-stage Phase B (B0 → B1 → B2) | 8/10 |
| Combined B0 (loss + data fixes + V+L grounding in single push) | 7/10 |
| Click head: two parallel 3-way heads + focal γ | 7/10 (single-primitive); 7/10 (multi-primitive) |
| Soft-label CE on dx/dy/scroll, 2-bin triangular | 7/10 |
| Keep text tower for B0/B1/B2 | 8/10 |
| Distractor scene design (4 attributes, 60/30/10, 25% adversarial) | 7/10 |
| Recovery via start-chunk + DART action noise (no mid-traj) | 7/10 |
| Typed-disposition gate replacing all-or-nothing | 7/10 |
| Diagnostic eval slicing + grounding probes | 8/10 |
| 10K episodes for B0 | 7/10 |
| a100-large compute choice | 8/10 |
| In-training Tier-A only; post-training M1 closed-loop | 7/10 |
| Implementation timeline (7-8 days) | 6/10 |

## Lit grounding (consolidated)

- **VIMA / RT-2 / OpenVLA:** distractor + attribute-grounded instruction methodology is canonical in spirit for V+L grounding eval. **Confidence 7/10** — the methodology is well-precedented, but mapping it to our Pygame setup (rather than tabletop manipulation) deserves the hedge.
- **π0 / π0.5:** scaled broad multi-task data with strong priors; informs Phase B's eventual full-v1 ambition. **Confidence 6/10** the prior story applies to our context.
- **OpenVLA-OFT:** parallel decoding + bin smoothing improvements compound across tasks once architecture is locked. Argues for locking architecture before scaling. **Confidence 7/10.**
- **DART (Laskey et al., 2017):** action-noise + expert re-query is included in B0 (low-rate, p=0.08) for continuous off-trajectory coverage. Closed-loop DAgger deferred. **Confidence 7/10** the technique applies cleanly to our setup.
- **DDPM / OpenVLA-OFT bin smoothing:** soft-label CE on discrete-distribution heads is well-precedented. **Confidence 7/10** — supports the principle; doesn't prove 2-bin triangular sufficiency for our exponential-bin scheme.
- **Focal Loss (Lin et al., 2017):** class-imbalance fix; γ is somewhat domain-sensitive — γ=3 is a reasonable first bet, not a robust default. **Confidence 6/10.**
- **Robomimic / Diffusion Policy:** sub-0.95 single-task ceiling typically degrades multi-task transfer; argues for hardening single-task before multi-primitive. **Confidence 7/10.**

## References

- [Phase A summary](../experiments/6-action-primitives-phase-a-results/PHASE-A-SUMMARY.md)
- [Phase A Spike B writeup](../experiments/6-action-primitives-phase-a-results/spike-b-lclick-end-to-end.md)
- [Design doc + amendments](../experiments/6-action-primitives.md)
- [Design changelog](../experiments/6-action-primitives-changelog.md)
- [HF Jobs gotchas](../research/hf-jobs-gotchas.md)
