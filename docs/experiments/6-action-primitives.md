# Experiment 6: Action Primitives

Teaching the VLA how to perform a diverse set of fundamental action primitives.

In anticipation of potential dual-system operation where reasoning happens in a larger, lower-Hz VLM that breaks a task down into action primitives for our high-Hz VLA.

**Performance targets:** logical framerate 30Hz (aspirational). Real-time wall-clock on current M1 PyTorch/MPS stack measured at ~7–15Hz; INT8+MLX optimization could potentially reach 15–22Hz. For v1, training and within-env eval use pygame slowdown so logical 30Hz is achievable regardless of wall-clock. Real-time 30Hz deployment deferred to Q29 (inference optimization).

**Status:** design v2 (2026-04-23). All 43 design questions closed; pre-implementation brainstorm pass added Phase A feasibility spikes and several amendments (see `6-action-primitives-changelog.md`). Ready for Phase A; Phase B (full v1) runs after per-spike review.

---

## Phase A — feasibility spikes (run before Phase B)

Four spikes to de-risk load-bearing assumptions before committing to the full generator + training stack. Each spike produces a write-up; user reviews and approves proceeding on a per-spike basis. No pre-committed quantitative thresholds — judgment call per spike based on results.

| Spike | What it validates | Effort |
|---|---|---|
| **A — Typing legibility probe** | Can SigLIP2 naflex @ `max_num_patches=256` actually resolve 14pt+ rendered text? Linear-probe patch features on rendered character content, or cross-attention visualization. Highest architectural risk — informs Q5/Q6 decision on typing feedback mechanism. | 2–3 days |
| **B — L-click end-to-end** | Whole pipeline works: generator → data format → model → training loop → eval. ~3K episodes of L-click primitive, 3–5 epochs, closed-loop val pass. Subsumes dataloader throughput (Q21) and architecture assembly testing. | 5–7 days |
| **C — M1 closed-loop eval timing** | Real wall-clock for 100-episode rollouts on M1 fp16 MPS. Validates Tier-B eval budget (Q25). Run as a 1-hour follow-up to B. | 1 day |
| **E — Pygame generation throughput** | Generate 1K episodes each of click, drag, type; measure eps/sec, storage per episode, parquet shard size. Validates Q7's "≥200 eps/sec" claim. Runs in parallel with A. | 0.5 day |

**Dependencies on spike outcomes:** see "Open items" in `6-action-primitives-changelog.md` for which Q-decisions each spike may reopen.

---

## Action primitives — taxonomy

Grouped by capability so eval can separate grounding failures from motor failures, and so rare capabilities aren't drowned by common ones in training.

**Group A — Grounding + single-frame motor** (core grounding probe)
- Hover to target
- Hover-and-wait (tests "stop deliberately", separates hovering from impulsive click)
- Click-at-relative-position (spatial relational grounding — "just below the red button")
- Left-click target
- Right-click target
- Double-click target (two clicks within N frames)

**Group B — Grounding + stateful motor** (closed-loop probe)
- Click-and-hold at target (mouse button, zero motion)
- Drag A → B (two-target grounding, held state across frames)
- Scroll-to-target (observation changes mid-task; target initially off-screen, closed-loop visual termination)

**Group C — Blind keyboard** (key head sanity check, no screen grounding)
- Type a short string (conditioned on text, ≤5 chars)
- Type a medium string (6–15 chars)
- Type a long string (16+ chars)
- Press a chord, e.g. Shift+Cmd+X
- Press and hold key(s) — chord with duration

**Training window:** per-primitive-type fixed window (see Q8 for specifics). No-ops pad the remainder after primitive completion.

**Deferred from this experiment:**
- "Highlight X" — redundant with Drag on action side; grounding variant better addressed via richer target-referring expressions
- Composite grounding + modifier (Cmd+Click, Shift+Drag) — tackle after A + B work
- Triple-click, click-drag-with-pause, press-key-N-times — niche, not worth v1 complexity
- Type-then-Enter compound — System 2 primitive chaining concern

---

## Dataset

Custom pygame-based synthetic renderer (see Q7). ~24,500 episodes × ~55 frames avg = ~1.3M transitions, split 80/10/10 episode-level stratified by primitive type (see Q8, Q14, Q25). Full primitive-specific recipes in Q9, visual diversity in Q10, instruction diversity in Q12. Separate OOD slices (~2,200 additional episodes across 6 slice types) for generalization measurement (Q14, Q25).

---

## Model architecture

### MacBook action space — delta / event-based

Every discrete actuator emits **what happened this frame** rather than **what is currently held**. Held-state lives in proprio. This is architecturally uniform with mouse-delta and scroll (which are already per-frame rates), and matches the OS event layer 1:1 (see Q3 for the full decision analysis).

**Mouse (4 physical actuators):**
- `delta_x`, `delta_y` — per-frame cursor velocity → **21 exponential bins per axis** (see Q1)
- Click events — **5-way `{idle, L_press, L_release, R_press, R_release}`**. Single head, softmax enforces trackpad exclusivity natively (cannot simultaneously L and R).
- Scroll — signed per-frame delta → **21 signed exponential bins**. Horizontal scroll ignored.

**Keyboard (77 physical keys):**
- Alphabet: A–Z (26)
- Numbers: 0–9 (10)
- Symbols: `, . / ; ' [ ] \ - = \`` (11)
- Arrows: Up / Down / Left / Right (4)
- Special: Space, Return, Delete, LShift, RShift, Tab, CapsLock, Escape, Ctrl, LCmd, RCmd, LOpt, ROpt (13)
- Function: Fn, F1–F12 (13)
- Per key: **3-way softmax `{press, release, idle}`**. 77 independent heads. Softmax enforces "at most one state change per key per frame" — press+release of same key requires 2 frames (see Q3).

**Output head summary (per frame):**

| Head | Output shape | Loss | Sparsity handling |
|---|---|---|---|
| Mouse `delta_x` | 21 logits | Focal CE + label smoothing | Zero-bin focal |
| Mouse `delta_y` | 21 logits | Focal CE + label smoothing | Zero-bin focal |
| Click (delta) | 5 logits | Focal CE | Idle-biased |
| Scroll | 21 logits | Focal CE + label smoothing | Zero-bin focal |
| Keys (delta) | 77 × 3 = 231 logits | Focal CE per key + idle-biased smoothing | Idle-biased bias init + per-class inverse-frequency weighting |
| Done (auxiliary) | 1 logit | Focal BCE | Advisory at inference; logged for interpretability |

Total: **300 logits/frame** across 6 heads (5 action heads + 1 auxiliary done head). All single-pass, no autoregressive decoding, no output chunking — one forward pass produces one frame's outputs. Real-time control rate on M1 measured at ~7–15Hz wall-clock (Q6, Q16); logical 30Hz achieved in training via pygame slowdown.

### Proprio (83 dims)

State-as-input, integrated as an additional K/V token in the trunk (see Q15 for architecture). Proprio is the state; action heads predict changes.

| Component | Dims | Notes |
|---|---|---|
| Cursor `(x, y)` | 2 | Normalized to [0, 1] |
| Held keys | 77 | Fixed bitmap mirroring the 77-key action space |
| Held mouse buttons | 3 | {LCLICK, RCLICK, middle} |
| CapsLock mode | 1 | Modal state — differs from held CapsLock *key* (green LED indicator on Mac). Hardcoded Mac-specific bit. |

**Proprio is treated as clean/exact signal.** No training-time noise augmentation (bit-flip, Gaussian jitter). The OS provides deterministic proprio; we're not simulating any real phenomenon by adding noise. If training reveals brittle over-reliance on proprio, introduce dropout-style regularization then — not preemptively.

### Observation space

- **Source (task window):** 720×450 full-screen, 16:10 aspect ratio
- **Fed directly to naflex** — no explicit pre-downscale; naflex resizes internally to ~320×192 at `max_num_patches=256` (240 tokens)
- **Future embodiment targets (deferred to later experiments):** MacBook native 1440×900, other aspect ratios, higher resolutions (handled via naflex + higher `max_num_patches`)

### Embodiment notes

- Horizontal scroll ignored.
- Real-world embodiments vary: keyboard layouts (non-English, OS differences), hardware (trackpad can't LCLICK+RCLICK simultaneously, mouse can; mouse has middle click).

---

## Open design questions

### Q1 — How do we handle both continuous & discrete actuators?

**Reframed:** the axis isn't continuous-vs-discrete, it's "what head architecture and loss match the physical character of each actuator." Factored heterogeneous heads dissolve the problem by treating each actuator on its own terms.

**Decision:** factored heads, each matched to its actuator (all delta / event-based for architectural uniformity — see Q3):

| Actuator | Head | Loss | Notes |
|---|---|---|---|
| Mouse `delta_x` | 21-bin exponential CE | Focal CE (γ≈2) + label smoothing (σ≈1 bin) | 10+1+10 layout, α≈2.5, **±100px** (widened from ±50px to cover human flicks, 2026-04-23). Finest bin ≈0.75px; coarsest ≈100px. Expected-value readout at inference. |
| Mouse `delta_y` | 21-bin exponential CE | Same as above | Same layout. |
| Click (delta) | 5-way softmax CE | Focal CE | `{idle, L_press, L_release, R_press, R_release}`. Trackpad exclusivity built-in. |
| Scroll | 21-bin signed exponential CE | Focal CE + label smoothing | Signed scroll delta per frame. Mirrors mouse philosophy — captures velocity, matches human action space. |
| Keys (77) | 77 × 3-way softmax CE | Focal CE per key + idle-biased label smoothing | Each key independent: `{press, release, idle}`. Idle-biased bias init + per-class inverse-frequency weighting. |

**Why exponential binning beats L1/Huber regression on mouse:**
- No 0-delta collapse (bin 0 is a first-class class — no degenerate regress-to-zero solution)
- Native multimodality (softmax can be bimodal → coarse flick + fine settle)
- No tanh saturation
- Heavy-tailed resolution matches natural mouse distribution
- Validated at scale by FDM-1 (11M hours of video)
- Already implemented in Exp 2 of this codebase (at 49 bins; revised to 21 here)

**Why 21 bins per axis (not 49, not 11):**
- **Literature range:** VPT (2022) used 11 per axis for angular camera control in Minecraft; JARVIS-VLA (2025) used 21 per μ-law-encoded axis; FDM-1 (2026) used 49 (ambiguous; likely 49 per axis given text, though figure is suggestive of ~7 per axis).
- **Data-efficiency argument:** with ~400K transitions, 49 per axis leaves each tail bin with hundreds of examples — marginal training signal. 21 per axis doubles per-bin examples.
- **Precision argument:** at ±100px with α≈2.5, the finest non-zero bin is ≈0.75px; naflex's 2.25× internal downscale already floors useful precision at ~1–2 source px. Sub-pixel bins are spurious precision.
- **Multimodality argument:** practical mouse distributions have ≤3 modes; 15+ bins is more than sufficient. FDM-1 at 49 was matched to 11M hours — different regime.
- Alignment with scroll head at 21 bins keeps the architecture uniform.

**Why ±100px (widened from ±50px, 2026-04-23):** at 30Hz, deliberate cross-screen motion is ~50px/frame but fast human flicks reach 100–167px/frame. Q9 training data deliberately includes "superhuman burst" tempos; ±50px would clip that signal at the top bin. ±100px covers typical flicks with minor loss at the finest bins (still below naflex's useful precision floor).

Lost ordinal structure is mitigated by **label smoothing across adjacent bins** and **expected-value readout at inference** (softmax × bin_centers → continuous delta).

**Why scroll gets the same treatment as mouse:** from the OS's perspective, scroll-per-frame is a signed integer/float with a heavy-tailed distribution — structurally identical to mouse. A 3-way categorical imposes an assumption we're trying to avoid (loses velocity info, doesn't match trackpad momentum scrolling or variable-rate wheel scrolling).

**Deferred / rejected:**
- **Unified token head (RT-2 / OpenVLA v1 style):** autoregressive decoding over ~80 action dims per frame is latency-prohibitive at 30 Hz on M1. FAST-style DCT tokenization is structurally bad for step-function key signals (Gibbs phenomenon).
- **Flow matching / diffusion over full action vector (π0, π0.5):** multiple denoising steps per frame exceeds the M1 budget for whole-action generation.
- **Flow head for mouse only:** ~200–400 extra LOC, 4–10× mouse-head inference cost, harder to debug. Binning already gives multimodality. Revisit if eval reveals a concrete motion failure binning provably cannot fix.
- **L1 / Huber regression on mouse:** 0-delta collapse, saturation risk, loses multimodality. Strictly dominated by binning here.

**Inter-actuator correlations** are handled by the shared trunk (same feature vector feeds all heads); the independence assumption lives only at the output layer. Two distinct failure modes to design around:
- *Data coverage* for rare chord combinations within the key head — mitigation: varied chord coverage in data generation.
- *Temporal state* for stateful cross-actuator primitives like Shift+Drag — mitigation: proprioception (held-state bitmap + CapsLock mode) as input, integrated as a K/V token (see Q15). Full proprio spec in Model architecture section; memory beyond proprio deferred to Q5.

### Q2 — How do we balance loss across the 5 heads?

**Reframed:** with Q1 and Q3 settled (all heads are now focal CE variants over discrete bins or categoricals), the loss families are unified but the magnitudes still differ by output dimensionality and sparsity.

**Four underlying problems:**
1. *Init-loss magnitude imbalance.* Different heads have different log(K) init values: log(21) ≈ 3.04 for bin heads, log(5) ≈ 1.61 for click, log(3) ≈ 1.10 for each key head × 77 keys.
2. *Target sparsity.* On any given frame, most heads' correct prediction is "do nothing / idle." Model can reach near-zero loss by learning the easy default, diluting signal on actually-active frames.
3. *Training dynamics drift.* Focal CE collapses toward label-smoothing floor; loss weights chosen at init become wrong.
4. *Original keyboard/mouse exclusivity concern* — mostly a subset of (2).

**Decision — Tier 1 stack (cheap, principled, no adaptive methods):**

1. **Balanced primitive sampler via micro-batch grad accumulation.** Macro-batch = 8 micro-batches × 8 episodes, each micro-batch homogeneous by primitive sub-type; the 14 primitive sub-types cycle across consecutive macro-batches so each sub-type gets roughly equal exposure. Accumulate gradients across the 8 micro-batches; optimizer steps once per macro-batch. Clean tensor shapes per forward pass; blended multi-task gradient per step. Strongest lever; fixes sparsity at the data source rather than the loss downstream (see Q8 for the reconciliation with per-primitive fixed windows).

2. **Focal CE on all heads (mouse `dx`, `dy`, click, scroll, keyboard).** γ ≈ 2, α balancing the majority "idle/zero" class. Uniform focal treatment across all heads — the delta keyboard decision (Q3) makes this clean: all heads are softmax CE with focal weighting.

3. **Init-loss-normalized fixed weights.** `w_i = 1 / L_i^init`, with `L_i^init` measured empirically on the first few batches (not computed analytically — actual init values depend on bias init + label smoothing). Each head contributes ~20% to total loss at init.

4. **Per-head loss + gradient-norm monitoring.** Diagnostic, logged every step. Not a weighting mechanism. Required given prior burn where key head dominated global gradient norms before bias init fix.

5. **Per-head gradient clipping** in addition to global clip. Prevents one head from destabilizing the shared trunk on an outlier batch.

**Explicitly rejected:**
- **Loss masking for inactive actuators** (e.g., zeroing out mouse loss on keyboard-active frames). Breaks the learned "when to be quiet" behavior. Downweighting via focal CE is fine; zeroing is not.
- **Uncertainty weighting (Kendall 2018), GradNorm (Chen 2018), PCGrad, CAGrad** — deferred. Added complexity not justified until Tier 1 demonstrably fails. None of the scaled VLA work (OpenVLA, OpenVLA-OFT, FDM-1, JARVIS-VLA) uses adaptive multi-task balancing — they solve it via tokenization / data mixing instead.

**Deferred triggers — revisit if:**
- Per-head loss curves diverge unrecoverably (Tier 2: hand-retune weights or add uncertainty weighting).
- Per-head gradient norms stay imbalanced after a convergence plateau (Tier 3: GradNorm).

**Hyperparameter note:** Focal γ is a new choice in this codebase (prior exp2 used ASL on keys; neither focal CE on bin heads nor focal CE on delta key heads is validated here). Treat γ as a hyperparameter to sweep ({1.0, 1.5, 2.0}) rather than inheriting γ=2 from the detection literature blindly. May also want per-head γ if convergence rates diverge.

**Coupling with future questions:**
- Focal γ and weight normalization should be re-tuned per dataset-generation recipe (Q6, dataset), not treated as universal constants.

### Q3 — Keyboard head specifics

**Reframed:** the original multi-label-ASL plan was reconsidered against delta/event-based alternatives. After full comparison (see decision log below), keyboard moved from toggle (hold-state) to delta (event-based) for architectural uniformity with OS events, scaling-precedent fit (FDM-1, JARVIS-VLA), and IDM-compatibility for future pseudo-labeled data.

**Decision: delta-3way per key.**
- 77 independent softmax heads, each over `{press, release, idle}`
- Mutual exclusivity per key enforced by softmax (cannot press+release same key in one frame)
- Multiple keys can emit events in the same frame (rollover supported natively)

**Loss treatment:**
- Focal CE per key (γ ≈ 2, swept)
- Idle-biased label smoothing (smoothing mass concentrated on idle neighbor, not uniform across 3 classes)
- Idle-biased bias init matching ~99% natural idle rate
- Per-class inverse-frequency weighting to boost gradient for rare keys (F-keys etc.)
- Shared γ across keys; per-key γ/α deferred as a Tier-2 option

**Data-side support:**
- Balanced primitive sampler (from Q2) — stratifies across primitive types
- Curriculum sampling: uniform key frequency early, shift toward natural distribution

**Inference:**
- Argmax per 3-way head, no threshold tuning needed (softmax already calibrated via focal CE)
- No rollover cap (violates minimal-inductive-bias; real data exhibits natural cap)
- Optional deployment-time safety cap allowed (e.g., reject >10 simultaneous events)

**Known trade-offs accepted:**
- Same-key consecutive (`xx`) requires ≥3 frames, different-key consecutive (`xy`) requires ≥2 frames. Structural asymmetry, not a learning problem. Model learns per-frame policy; architecture forces the difference.
- Max same-key repeat rate = 15 Hz at 30 Hz frame rate (fine for human typing ~180 WPM; sufficient for gaming WASD-style tapping).
- Sub-frame events (<33ms) lost at 30 Hz. Future higher-Hz inference resolves this; keyboard hardware debounce (~5ms) is well below 33ms anyway so the loss is small.

**Rejected alternatives:**
- **77-way multi-label ASL (toggle / held-state)** — the prior plan. Strong on residual learning with proprio and dense supervision across long holds, but inconsistent with scaling precedent, requires external differ to produce OS events, and creates visible release+press event pairs on single-frame bit flips. See decision log.
- **4-way `{press, release, tap, idle}`** — compact for typing (1 frame/char vs 2), but introduces labeling ambiguity (below what duration threshold does press+release become `tap`?) that hurts IDM pseudo-labeling. Revisit only if primitives need >15 Hz same-key repeats or if typing throughput becomes a binding eval axis.
- **Paired multi-label (154 bits: press-mask + release-mask)** — loses mutual exclusivity structure; more logits without clear benefit.
- **Autoregressive event tokens (FDM-1 style)** — latency-prohibitive at 30 Hz with parallel decoding.
- **Rollover cap during training** — violates minimal-inductive-bias; data-distribution learning is sufficient.
- **Auxiliary toggle head during training** — dropped; the supervision-density concern it was meant to address turned out to be smaller than initially claimed (focal losses flatten the "dense vs sparse gradient" gap). Revisit as Tier-2 if eval shows drift on long stateful primitives.
- **Grouping modifier vs standard keys** — arbitrary boundary, violates minimal-inductive-bias. Per-key bias init + per-class weighting captures the statistical differences without imposing structure.

**CapsLock handled as proprio bit** (not inferred from action history). Mac-specific — the physical green LED indicator reflects a real modal state that held-state alone can't capture. Other modifiers (Shift/Cmd/Option/Ctrl) don't need a mode bit because their effect is scoped to their hold duration.

**LShift / RShift kept distinct.** Physically separate switches. Data generator decides convention.

**Embodiment generality note (v1 scope):** This experiment targets a specific MacBook keyboard embodiment — 77 Mac-specific keys, no middle mouse button (trackpad only), US-QWERTY layout. CU-VLA's eventual goal is generalist embodiment support (different layouts, resolutions, aspect ratios, hardware). For v1, we accept embodiment specialization to keep the experiment tractable. The vision encoder choice (SigLIP2 naflex in Q6) preserves aspect-ratio generality for future work; keyboard action space does not. See Q39 (forward-looking).

### Decision log: keyboard action space (toggle vs delta)

Full 20-dimension comparison between Option 1 (state-based / toggle with ASL) and Option 2+ (delta-3way with mitigations). Key dimensions that drove the choice:

**Option 2+ wins on:**
- Architectural uniformity (all discrete actuators are event-based)
- OS event-layer fidelity (1:1 match, no external differ)
- Scaling precedent (FDM-1 at 11M hr, JARVIS-VLA)
- IDM / pseudo-label compatibility (events are native to event-stream data)
- Loss family uniformity (all focal CE variants)
- Debug granularity (per-key confusion matrix vs per-bit F1)

**Option 1 wins on (and how Option 2+ mitigates):**
- Residual learning with proprio → mitigated by proprio-as-token integration (see Q15), idle-biased init/smoothing
- Self-correction on single-frame errors → smaller concern with delta-3way since OS events update proprio truthfully next frame
- Supervision density on long holds → smaller concern than initially framed (focal losses flatten the gap); auxiliary toggle head held in reserve
- Supervision density on rare keys → mitigated by per-class weighting + curriculum sampling

**Trade accepted:** more output logits (299 vs 143), more hyperparameter surface (focal γ, idle smoothing mass, class weights, curriculum schedule), less battle-tested specific combination. Net: reasonable for writing-afresh codebase with scaling goals.

### Q4 — Mouse head specifics

**Most items settled by Q1 and Q3.** Recording the remaining decision and implementation flags.

**Click/scroll coupling with mouse movement:** factored heads with shared trunk (no architectural coupling). Reasoning:
- The "don't click while flicking" correlation is a data-distribution fact, not a hard constraint (drag initiation is a valid counter-example).
- Shared trunk features capture the correlation natively.
- Autoregressive conditioning would add latency for a small expected gain.
- Empirical check: track click-timing accuracy during movement in eval. Revisit only if model fires clicks during fast flicks.

**Previously open items (now settled):**
- ~~Movement as vector field vs exponential binning~~ → settled in Q1 (21-bin exponential CE per axis; softmax can natively represent both coarse flicks and fine corrections as different output distributions).
- ~~Pure regression might be hard~~ → settled in Q1 (binning chosen).

**Implementation flags (not decisions, just flags for later):**
- *Trajectory smoothness.* Per-frame binned prediction doesn't enforce smoothness across frames. Default: accept and rely on smooth training data. Fallback if jitter appears: temporal smoothness regularizer `(dx[t] − dx[t−1])²` at training time.
- *Screen-bound clipping.* Cursor position is in proprio, so the model *can* learn to respect bounds. Cheap inference-time clip as a safety net.
- *Expected-value readout.* At inference: continuous `delta_x = softmax(logits) · bin_centers`. Gives continuous output from discrete prediction and averages multimodal softmax distributions.
- *macOS pointer acceleration.* Train and infer in pixel space (what the encoder sees), not device space (what the mouse sends). Skip modeling acceleration.

### Q5 — Memory and action chunking

**Reframed:** "memory" covers three different needs — state (solved by proprio), history (past events relevant to current decision), and intent persistence (knowing what primitive is in progress). Q5 is about the latter two.

**Memory requirements per primitive:**

| Primitive | Needs history? | Why / how handled |
|---|---|---|
| Hover | No | State + task suffices |
| L/R-click | No | State + task suffices |
| Double-click | **Yes** | Must know "just clicked once" to fire second click |
| Click-and-hold | No | Proprio's held-mouse bit suffices |
| Drag A → B | Yes, mildly | Must remember drag-start event and target semantics (which is A vs B) |
| Type string | **Yes** | Must know typing progress; proprio has no typing memory |
| Chord | No | State + task suffices |
| Press-and-hold | No | Fixed-window + proprio suffices |

**Decision — minimal memory stack (single-frame output):**

1. **Single-frame observation** (no stacked visual frames). Vision compute bounded.
2. **Single-frame output** (no action chunking). One forward pass produces one frame's actions. Simplest possible control loop.
3. **Action history: last 8 frames as discrete tokens.** Each frame's argmax action embedded into a small token, 8 tokens added to trunk input. Handles Double-click, drag-start events. Low cost (~8 tokens × ~32 dims).
4. **No recurrent state.** Fully stateless between forward passes — simpler debug, fully parallelizable training.
5. **Type-string handled via visual feedback + instruction-includes-target**, not via long action history. Model re-derives typing progress from rendered text + instruction.

**Action-history token format (spec'd in Q19):** per-frame composite action vector (~300-dim: head one-hots + key press/release masks) → MLP projection (300→256→768) → 768-dim token per past frame. Plain teacher forcing during training (no scheduled sampling — see Q19 for rationale).

**Why single-frame over chunking (reversed from earlier draft):**

Chunking was initially motivated by latency amortization: "30 Hz on M1 with SigLIP2 requires K=6 chunks." On honest re-examination:
- Target spec allows 15 Hz minimum. Single-frame at 40–70ms/forward fits 15 Hz cleanly.
- Encoder choice (Q6) determines whether single-frame 30 Hz is achievable — lighter encoders close the gap without needing chunking.
- Chunking's modelling benefits (double-click in one chunk, typing bursts, co-predicted drag start/end) are real but each has a cheaper single-frame equivalent via action history + proprio.
- Chunking costs: closed-loop responsiveness drops (committed to stale plan for K frames), added complexity (temporal transformer, temporal loss reweighting, partial-playback re-planning), train/inference distribution shift.

Single-frame's benefits: simpler, closer to minimal-inductive-bias, closed-loop responsive every frame, no train/inference mismatch, trivial to debug.

**Coupling with Q6 / Q16 reality (updated 2026-04-23):** earlier drafts of Q5 framed chunking as rejected because "Q6 encoder choice resolves latency." Q16's empirical measurement shows that's not true: current PyTorch MPS stack delivers ~7.6Hz wall-clock for the full pipeline on M1, not 30Hz. Single-frame output is nonetheless the right call on its independent merits:

- Closed-loop responsiveness every frame (chunking commits to a stale plan for K frames)
- Simpler debug and no train/inference distribution mismatch
- Action history (8 past frames) + proprio carry enough temporal context for Double-click, Drag, Click-and-hold without chunking's modeling wins
- Logical 30Hz is achievable for training and within-env eval via pygame slowdown (env paused during inference), decoupling wall-clock from logical clock

Real-time 30Hz deployment is a separate concern, tracked in Q29 (MLX/INT8 port, post-v1).

**Idle-frame handling during typing rhythm:**
Human typing at 120 WPM produces ~3 idle frames between keystrokes at 30 Hz. Handled by:
- Focal CE + idle-biased init (from Q3) — already set up for idle-dominant regime
- Per-class inverse-frequency weighting — upweights press/release
- Action-history tokens capture rhythm → model learns distribution over inter-keystroke gaps, not a point estimate
- Training data with varied tempos (see below) teaches the model that gaps are variable

**Data-generation flag — tempo variability for superhuman execution:**

Standard behavioral cloning will reproduce the training distribution's timing exactly. Training on Fitts-law human timing → model types at human speed.

To enable superhuman execution at inference while training on human-plausible data, the synthetic generator should produce primitives at **varied tempos**:
- Inter-keystroke gaps drawn from a wide distribution (e.g., 0–10 frames, not a fixed 3)
- Mouse-settle durations varied (0–5 frames between arrival and click)
- Chord hold durations varied
- Mouse velocities including super-human flicks (bins 18–20 used deliberately for some samples)

The model learns *sequence invariance* (press-before-release, release-before-next-press) rather than *timing memorization*. At inference, when the model is confident about the next event, it emits immediately — effective superhuman execution without RL or explicit speed conditioning.

Precedent: VPT achieved superhuman Minecraft crafting speed from human-demo data using this distributional-variety approach.

**Deferred / rejected:**
- **Action chunking (K=6 temporal action expert).** Latency motivation is resolved by Q6 encoder choice; modelling benefits addressable via action history. Revisit only if eval shows single-frame struggles with tightly-coupled event pairs (e.g., drag-start press + first-frame-motion co-prediction).
- **Stacked visual frames.** Too expensive on M1. Revisit at scaling.
- **Long video context (FDM-1 style, minutes to hours).** Scaling story.
- **Recurrent hidden state.** Debug cost exceeds value; action history covers the same capability more simply.
- **Longer action history (K=32+).** Diminishing returns. If 8 frames proves insufficient for a specific primitive, increase K before adding other mechanisms.
- **Reinforcement learning for speed.** Data-side tempo variation is cheaper and sufficient.
- **Explicit tempo conditioning.** Data variety alone teaches sequence invariance; no need for a separate tempo input.

**Cross-references:**
- Proprio (from Q3) + 8-frame action history (here) jointly solve "intent persistence."
- Single-frame output puts full weight on Q6 (encoder choice) for achieving 30 Hz control rate.
- Tempo variability is a data-generator requirement that belongs in the Dataset spec.

### Q6 — Vision and text encoder choice

**Elevated criticality post-Q5:** with action chunking rejected and single-frame output locked in, encoder latency is the *sole* determinant of achievable control rate on M1. No more amortization escape valve.

Three sub-decisions, entangled:
- **Q6a:** Vision encoder
- **Q6b:** Text encoder + conditioning mode
- **Q6c:** Image resolution

**Decision stack:**

**Vision: SigLIP2-B/16 naflex (`google/siglip2-base-patch16-naflex`) with `max_num_patches=256`.**
- ~90M params, contrastively pretrained with SigLIP2's multi-objective setup (captioning, masked prediction, self-distillation) for strong dense features *and* language grounding.
- **Naflex variant:** accepts arbitrary input resolution/aspect ratio, resizes internally to satisfy `max_num_patches`. Control knob is `max_num_patches`, not our input resolution.
- `max_num_patches=256` (default): for 16:10 input, internally resized to 20×12 patches = 240 tokens actually fed to the ViT. At 720×450 source → naflex downsamples to ~320×192 internally before patching.
- Preserves future embodiment generality (different aspect ratios) without re-training.
- Estimated M1 single-frame latency: 8–14ms (INT8-quantized via MLX) at 240 tokens.

**Text: SigLIP2 matched text tower with instruction caching.**
- Pre-aligned with vision tower from contrastive pretraining — no re-alignment cost during primitive training.
- Instruction text is *static per primitive* → encode once at primitive-start, cache tokens, reuse. Text-side cost amortizes to ~zero at steady state.

**Conditioning: decoder cross-attention.**
- Instruction acts as query/goal; vision features are keys/values.
- Asymmetric flow: visual tokens need rich instruction context (for grounding); instruction tokens don't need rich visual context.
- Token concatenation wastes attention capacity for our specific grounding-heavy task shape.

**Resolution / input pipeline:**

Render synthetic primitives at **720×450 full-screen task window** (16:10). Feed directly to the naflex processor — **no explicit pre-downscale step.** Naflex resizes internally to ~320×192 (20×12 patches at `max_num_patches=256`).

The earlier "512×320 pre-downscale" plan turned out to be redundant: naflex already internally resizes based on `max_num_patches`, so pre-downscaling just moves work around. At `max_num_patches=256` the effective resolution entering the ViT is ~320×192 regardless of whether we feed 720×450 or 512×320.

**Why `max_num_patches=256` (default):**
- Matches a SigLIP2 NaFlex training sequence length (trained on {128, 256, 576, 784, 1024}) → in-distribution use
- ~240 tokens keeps latency budget comfortable (8–14ms encoder)
- Self-attention is O(n²) so token count dominates latency — more patches gets expensive fast
- At this setting, an ~18 source-px region maps to ~1 patch → spatial resolution is coarse but adequate for 20-source-px-minimum UI targets (Q9)

**`max_num_patches` ablation (Q27):**
- 256 (default): ~240 tokens, ~10ms, coarse cursor visibility (~1 patch)
- 576: ~480 tokens, ~30ms, better spatial resolution, pushes 20 Hz limit
- 1024: ~960 tokens, ~60–100ms, best spatial resolution, likely breaks 15 Hz floor

Ablation measures primitive accuracy vs control rate tradeoff.

**Dataset element-sizing constraints (feeds Q9):** at `max_num_patches=256`, naflex internal resize is ~320×192 from 720×450 source (~2.25× downscale internally). Minimum source sizes for reliable grounding:
- UI targets: ≥ 20 source px (≈9 internal px ≈ 0.5–1 patch) for minimal discriminability; ≥ 40 source px (≈18 internal px ≈ ~1 patch) for comfortable grounding
- Text (for typing visual feedback): ≥ 14pt at default DPI for legibility after 2.25× downscale
- Cursor: macOS-default 32 source px → ~14 internal px ≈ ~0.9 patch (sub-patch — see Q11 for implications)
- WCAG AA-ish contrast for task-relevant elements vs background

**Deferred to future experiments:**
- Multi-resolution source data ({1280×720, 1920×1080, 2560×1440, 4K}) — embodiment-general but adds dataset complexity v1 doesn't need
- Multiple aspect ratios (16:9, 3:2, 4:3) at source — architecture (naflex) already supports; add at data-generation when scaling
- Higher `max_num_patches` at inference (576, 1024) — latency-expensive; revisit if grounding quality proves inadequate

**Trunk wiring (complete pipeline — see Q15 for architecture details):**
1. Screen render → 720×450 RGB frame
2. naflex processor resizes internally to ~320×192 → patch tokens (240 at 16px patch, max_num_patches=256)
3. SigLIP2-B/16 vision encoder (LoRA rank-8 adapted) processes 240 patch tokens → vision K/V
4. Instruction (cached from primitive-start) → SigLIP2 text tower (frozen) → text K/V (~16 tokens)
5. Proprio (83 dim) → 2-layer MLP → single 768-dim token → added to K/V
6. Action history: 8 past-frame composite action vectors → MLP `300→256→768` + temporal positional embedding → 8 tokens added to K/V (Q19)
7. 16 learnable query tokens cross-attend to the combined K/V pool (~265 tokens), then self-attend among themselves — 3 alternating cross+self blocks
8. 6 output heads (5 action + 1 auxiliary done) read from flattened query states (unpooled — Q16)

Total params: ~118M inference (86M SigLIP2-B + 32M trunk + heads). Trainable: ~22M (LoRA adapters + trunk + unpooled heads).

**Empirically measured latency (M1 MacBook Pro, PyTorch fp16 MPS):**

| Stage | Measured median |
|---|---|
| SigLIP2-B vision tower @ max_patches=256 | 80–92ms |
| Trunk A (3 blocks, 32M) | 40ms |
| 6 unpooled heads | negligible (included in trunk timing) |
| **Full pipeline (encoder + trunk)** | **~132ms → ~7.6 Hz wall-clock** |

**30Hz is aspirational, not achieved on current stack.** Reaching it requires migrating off PyTorch/MPS to MLX or CoreML with INT8 quantization — neither available as a pure-PyTorch flag. Projected best-case with INT8+MLX: ~45–65ms/frame = 15–22 Hz. Still below 30Hz.

**v1 framing:**
- **Logical framerate: 30Hz.** Model trained and evaluated as if running at 30Hz.
- **Wall-clock framerate: 7–15Hz on current stack.**
- **Approach:** pygame env paused during inference. Agent experiences 30Hz-stamped frames; model learns the 30Hz policy; wall-clock decoupled from logical clock.
- Legitimate for training + within-env eval; real-screen deployment requires Q29 inference optimization.

See Q16 for full sizing rationale and Q29 for deployment optimization roadmap.

**Explicitly rejected / deferred:**

- **SigLIP2-SO400M (400M params).** Too heavy — projected ~60–80ms/frame = ~13 Hz, misses the 15 Hz floor. Revisit at scaling when we have beefier compute.
- **C-RADIOv4.** Strictly better multi-task features than SigLIP2 in principle (distilled from SigLIP2-g + DINOv3 + SAM3), but negligible grounding-on-screenshots advantage at similar cost, and SigLIP2 is more mature on HF/MLX. Revisit if grounding quality becomes the binding constraint.
- **DINOv3 alone.** No language grounding; would need separate alignment layer. Complexity not worth it given SigLIP2 covers both.
- **E-RADIO-B.** 3–8× faster than competitors but NSCL non-commercial license disqualifies from an open trajectory.
- **FDM-1 custom video encoder (masked compression).** Scaling story; requires our own large-scale pretraining. Flagged in Q38.
- **FastViT-like custom encoder.** Fast but no pretrained weights in our domain — would need training from scratch, sacrificing the transfer advantage of SigLIP2.
- **Token concatenation** (alternative to cross-attention). Wastes attention capacity by mixing instruction and visual tokens at every layer.
- **Separate text encoder (MobileBERT / DistilBERT / SmolLM2).** Loses SigLIP2's contrastive pre-alignment. MobileBERT in particular was previously under consideration but is dominated by the matched SigLIP2 text tower for this use case.
- **Full 1440×900 resolution.** ~6600 tokens per frame, prohibitively expensive.

**Cross-references:**
- Q18 (vision token reduction): no trunk-side reduction; encoder-side via `max_num_patches` is the latency lever (deferred to Q29).
- Q29 (deployment stack): PyTorch fp16 + MPS is the v1 stack. MLX / CoreML / INT8 flagged as future work; not v1 scope. See Q29 for feasibility analysis.
- Q15 (trunk architecture): now constrained — SigLIP2-B/16 output dim is 768; trunk operates in that dim unless we project down.
- Q16 (model size): decided at ~118M inference / ~22M trainable (Trunk A, 3 blocks, unpooled heads).

---

## Backlog: Remaining questions

Broader design questions not yet resolved, grouped by area. Roughly prioritized: A blocks B–D, B blocks E, etc. None are blocking Q6 (encoder) directly.

### A. Dataset & environment (8 — highest priority, blocks most else)

**Q7 — Environment choice. [DECIDED]**

**Custom pygame-based synthetic renderer.** Full pixel-level control, deterministic, fast (≥200 episodes/sec target), perfect ground-truth labels by construction.

**Spec:**
- **Framework:** pygame
- **Canvas:** 720×450 full-screen task window (16:10)
- **Rendering style:** flat UI — colored buttons, simple text, basic backgrounds. Not mimicking any specific OS design.
- **Ground truth per frame:** target bboxes, cursor position, held-state, rendered text content, element identities — all emitted alongside the frame
- **Action execution:** pygame event loop accepts our 77×3-way key events, 5-way click events, 21-bin mouse deltas, 21-bin scroll, applies at 30 Hz
- **Primitive library:** one Python class per primitive (HoverPrimitive, ClickPrimitive, DragPrimitive, ChordPrimitive, TypePrimitive, etc.), parameterizable for target size/position/instruction-phrasing/pre-state/tempo
- **Determinism:** seeded random state per episode → reproducible
- **Storage:** frames as PNG or compressed numpy; actions as discrete event streams with frame indices; proprio as per-frame records; instructions as text; metadata (bboxes, target identity) for eval ground truth

**Why pygame over alternatives:**
- None of our primitives need real-app visual fidelity — click-the-red-button is learnable against a pygame-drawn button
- Speed: ~400K transitions target at 30 Hz ≈ 3.7 hours of screen time; pygame generates this in hours, browser in days, VM in weeks
- Ground truth by construction: no DOM parsing, no accessibility-tree inspection
- Visual diversity is controllable (Q10)
- Trivial to render at exactly 720×450

**Explicitly rejected / deferred:**
- **Browser-based (Playwright/Selenium):** too slow; DOM-to-bbox extraction adds complexity; not needed for primitives. Revisit for later web-specific primitive experiments.
- **macOS VM:** slow, complex state management, licensing, overkill. Revisit when real-world transfer eval matters (Q37).
- **Qt/Tkinter synthetic:** no meaningful advantage over pygame.
- **Hybrid (synthetic + real):** two pipelines for no v1 benefit.
- **Web canvas pre-rendering:** complexity without payoff.

**Known limitation to address:** visual distribution brittleness — model may overfit to pygame's visual style. Mitigated by aggressive generator-side visual diversity (Q10).

**Downstream implications:**
- Q8 (dataset composition): pure synthetic for v1; real-screen data becomes Q37 concern.
- Q9 (synthetic generator): scoped to pygame-class-per-primitive implementation.
- Q10 (visual diversity): fully generator-controlled; need aggressive variety in colors/fonts/backgrounds/distractors.
- Q11 (cursor rendering): trivial to render onto frame; decision becomes "do we want to?" not "can we?"
- Q13 (rendering latency): pygame has no natural latency; explicit modeling deferred.
- Q25 (eval): real-screen eval slice deferred; OOD eval uses pygame-variant rendering (different palettes, fonts, backgrounds).
- Q28 (eval infra): lightweight pygame-based rollout harness, not FDM-1-style forking VMs.

**Q8 — Dataset composition. [DECIDED]**

**Synthetic / real mix:** Pure pygame synthetic (from Q7). No real-screen data in v1. Real-data bridge deferred to Q37.

**Dataset size:** Target ~1M transitions (≈16K episodes at ~60 avg frames). Start with ~400K as a pilot to validate the pipeline end-to-end, then expand to 1M once the pipeline is validated.

**Per-primitive episode counts (target ~1M transitions):**

| Primitive | Episodes | Window | Rationale |
|---|---|---|---|
| Hover | ~1,500 | 30 | Simplest grounding probe |
| Hover-and-wait | ~1,000 | 45 | "Stop deliberately" variant |
| Click-at-relative-position | ~1,500 | 30 | Spatial relational grounding |
| Left-click | ~3,000 | 30 | Core skill, high variety needed |
| Right-click | ~1,500 | 30 | Transfer expected from left-click; explicit volume |
| Double-click | ~2,000 | 45 | Tests action history |
| Click-and-hold | ~1,500 | 60 | Stateful motor probe |
| Drag A→B | ~4,000 | 90 | Most complex Group A/B primitive |
| Scroll-to-target | ~2,000 | 90 | Closed-loop visual grounding |
| Type (short, ≤5 chars) | ~1,500 | 45 | |
| Type (medium, 6–15) | ~1,500 | 120 | |
| Type (long, 16+) | ~500 | 240 | |
| Chord | ~2,000 | 30 | Covers 77-key space |
| Press-and-hold | ~1,000 | 60 | Variant of chord |
| **Total** | **~24,500 episodes** | | **~1.3–1.5M transitions** |

**Window strategy: per-primitive-type fixed windows** (option B). Each primitive type has its own fixed window sized to natural length distribution. Within a type, episodes padded to that window with no-op continuations.

**Batching with micro-batch grad accumulation** (reconciles Q2's balanced sampler with per-primitive windows): each optimizer step consumes a macro-batch of 64 episodes split into 8 micro-batches × 8 episodes, where each micro-batch is homogeneous by primitive sub-type (same window length → clean tensor shapes). Gradients accumulate across the 8 micro-batches; optimizer steps once per macro-batch. Across consecutive macro-batches the 14 primitive sub-types cycle so each gets roughly equal exposure. This gives clean shapes per forward pass AND blended multi-task gradient per step — closer to π0 / OpenVLA-OFT patterns than either single-primitive-per-batch (literal earlier reading) or true variable-length masked batches (~200–500 LOC cost rejected below).

Window lengths per primitive are specified in Q9's recipe tables. Summary: Group A primitives (hover/click family): 30–45 frames. Group B primitives (drag, scroll, click-and-hold): 60–90 frames. Group C typing sub-categorized by string length: 45/120/240 frames. Chord: 30. Press-and-hold: 60.

**Why not fully dynamic windows:** ~200–500 lines of training-pipeline complexity (variable-length batching, attention/loss masking, episode-start tokens, eval-pipeline changes, debugging surface). For v1 this cost isn't justified. Can revisit at scaling.

**Why not single fixed window (60 frames):** wastes compute on short primitives (click at 30 frames padded to 60 means half-compute is no-op loss); truncates long primitives (>60-frame typing impossible).

**Done head (auxiliary task + inference telemetry):**

Add a single-sigmoid "done" head to the architecture, supervised during training by ground-truth primitive completion time from the generator. At inference, the head runs but is **advisory-only** — its output is logged for interpretability but does not control rollout termination. Termination is driven by external orchestration (fixed-window rollout for eval, future System-2 orchestrator for deployment).

Rationale:
- **Training benefit (unknown but likely modest):** auxiliary loss forces trunk to represent "task progress" explicitly, which may regularize representations. Especially relevant for long primitives (typing, drag) where progress-awareness is harder to learn implicitly.
- **Inference observability:** per-frame done probability logged for debugging. Lets us distinguish "model knew it was done but failed anyway" from "model never realized it was done." Hard-to-get diagnostic otherwise.
- **No inference risk:** since the head doesn't control anything at inference, mis-calibration produces misleading telemetry, not production failures.
- **Option-value for future System 2:** dynamic termination or failure detection using the done signal becomes available without retraining — just start listening to the head.

**Done-head supervision details:**
- Binary target: 1 for frames at or after the ground-truth completion point, 0 before
- Focal BCE consistent with other heads
- Included in the focal-CE weighting scheme from Q2
- Included in the auxiliary "done head on / off" ablation (Q27)

**Train/val/test split:** 80/10/10 at episode-seed level. Each episode is atomic — no frame-level splits.

**OOD slices (for generalization measurement, from the test set or held-out additions):**
- Held-out target colors (train: red/blue/green; test: orange/purple)
- Held-out instruction phrasings (train: "click the red button"; test: "press the crimson control")
- Held-out target sizes (train: 20–40px; test: 60–100px)
- Held-out target positions (train: central; test: screen corners)
- Held-out backgrounds (train: white/gray; test: textured)

Explicit goal: distinguish memorization from generalization. A model that scores high on in-distribution test but low on OOD slices has learned the training distribution, not the primitive.

**Q9 — Synthetic primitive generator design. [DECIDED]**

Each primitive type has a dedicated Python generator class that produces episodes. An episode is `(instruction, pre_state, frame_sequence, action_sequence, proprio_sequence, done_sequence, metadata)`.

**Shared generation pipeline (all primitives):**

1. Sample primitive parameters (target bbox, phrasing, pre-state noise, tempo)
2. Render pre-state (initial frame)
3. Compute ground-truth action trajectory
4. Roll forward frame-by-frame: apply action → update proprio → render → mark done
5. Pad to fixed window (from Q8) with no-op continuations
6. Emit metadata for eval (target bboxes, completion frame, primitive type/sub-type)

### Shared parameters

**Pre-state noise:**
- Cursor position: uniform over canvas
- Held keys: 0 keys (P=0.85), 1–2 modifier-only held (P=0.15)
- CapsLock mode: 95% off, 5% on (matches real-world base rate; 5% included so model learns to handle the mode, not because it's common)

**Tempo variability** (from Q5):
- Mouse velocity profile sampled per episode: slow / normal / fast / superhuman
- Settle before click: 0–5 frames
- Inter-keystroke gap: 0–6 frames (mix of distributions including superhuman-fast)
- Chord hold duration: 2–15 frames

**Behavioral noise (2–5% of episodes):**
- Mid-primitive hesitation, small cursor overshoot-correct, typed-then-backspaced char
- Primitive still succeeds — these simulate human-realistic self-correction, not task failure

### Primitive-specific recipes

**Group A — Grounding + single-frame motor**

| Primitive | Trajectory | Window | Notes |
|---|---|---|---|
| Hover | Minimum-jerk spline cursor → target | 30 | Tempo-varied |
| Hover-and-wait | Hover + hold cursor in target for N frames | 45 | Tests "stop deliberately" |
| Click-at-relative-position | Approach anchor + offset, settle, L_press/release | 30 | e.g. "click just below the red button" — tests spatial relational grounding |
| Left-click | Approach, settle, L_press, L_release | 30 | Core grounding + motor |
| Right-click | Approach, settle, R_press, R_release | 30 | Explicit volume to avoid "always L_click" prior |
| Double-click | Approach, settle, L_press, L_release, gap (3–10f), L_press, L_release | 45 | Tests action history |

**Group B — Grounding + stateful motor**

| Primitive | Trajectory | Window | Notes |
|---|---|---|---|
| Click-and-hold | Approach, settle, L_press, hold (10–40f), L_release | 60 | Hold duration varied |
| Drag A → B | Approach A, L_press, drag path to B, L_release | 90 | Two-target grounding, continuous closed-loop |
| Scroll-to-target | Scroll events until target enters viewport, then stop | 90 | **Spec (2026-04-23):** content type randomized per episode between (a) text-row list of 15–30 colored rows (reused/adapted from `miniwob_pygame/widgets.py` `ScrollableList`) and (b) scrollable page/document with a button below the fold. Input: scroll-wheel only (maps to 21-bin scroll head); no scrollbar-drag (would collide with Drag primitive grounding); no arrow-keys in v1. Target initially off-screen; closed-loop visual grounding as target enters viewport. |

**Group C — Blind keyboard**

| Primitive | Trajectory | Window | Notes |
|---|---|---|---|
| Type (short, ≤5) | Per-char press/release with inter-key gap | 45 | Mix natural and random strings (70/30) |
| Type (medium, 6–15) | Same, longer | 120 | |
| Type (long, 16+) | Same, longest | 240 | |
| Chord | Press modifiers in order, press trigger, hold (2–15f), release (synchronous) | 30 | Varied chord sizes (2–5 keys) |
| Press-and-hold | Chord with longer mid-hold (15–40f) | 60 | |

**Frame-limit validation at generation time:** the data generator MUST verify that every generated episode fits within its primitive's window budget (including all pre/settle/inter-key gaps at sampled tempos + action frames + post-completion padding). If a sampled trajectory would exceed the window, resample tempos or truncate string length (for typing). Concretely: Type-short window of 45 frames accommodates up to 5 chars at modest tempo (~2 press/release + 0–6 gap frames per char = ~16–40 frames total); aggressive tempos fit comfortably, slow tempos may need tempo resampling. Similar checks for all primitives. Assertion-level enforcement in generator code.

### Typing — Capslock & uppercase strategy

Length-based heuristic per continuous uppercase run in target string:

```
len == 1:  shift_letter  # single Shift+key, ergonomic shift side
len 2–4:   70% shift_held, 30% capslock
len 5+:    80% capslock, 20% shift_held
```

**Shift side (LShift vs RShift):** 60% ergonomic (opposite hand from letter), 40% random. Covers both touch-typists and hunt-and-peck variation.

**Symbol keys requiring shift:** always Shift+key, side chosen ergonomically with noise.

**Capslock-on-at-start + lowercase target:**
- 70%: model toggles Capslock off first, then types
- 30%: model uses Shift for each letter to avoid toggling

This gives the model a meaningful signal that proprio Capslock state should influence typing strategy.

### String content for Type primitives

**70% natural English** — words, phrases, sentences. Provides realistic character frequency distribution.
**30% random characters** — ensures rare-key coverage (Q, Z, J, symbols).
**String length distribution:**
- Short (≤5 chars): 40%
- Medium (6–15 chars): 45%
- Long (16+ chars): 15%

### Instruction phrasing

Per primitive, 10+ phrasing templates. Target-referring slot has varied expressions:
- Color + type: "red button"
- Color + label: "red 'Submit' button"
- Label only: "the button labeled 'Submit'"
- Position: "the button in the top-right"
- Combined: "the red 'Submit' button in the lower right"
- Relational (for relative primitives): "just below the red button"

### Diversity target

Approximates what a contracted-human / wild-scraped / internet-video dataset would look like:

**Visual diversity (target ≥20 distinct visual themes):**
- UI styles: flat, skeuomorphic, neobrutalist, minimalist, cluttered
- Color palettes: pastel, bold saturated, dark mode, light mode, high contrast
- Fonts: sans-serif (SF Pro / Inter / Helvetica-style), serif, monospace; varied weights
- Element shapes: rectangles, rounded rectangles, pill buttons, circular icons, irregular
- Backgrounds: solid, gradient, textured, noisy, pseudo-image content
- Distractors: similar-colored non-targets, overlapping elements, partial occlusions

**Behavioral diversity (5–10 tempo/style presets):**
- Tempo: slow beginner, average, fast power user, superhuman burst
- Trajectory: smooth min-jerk, hand-shake noise, overshoot-correction
- Click: quick, deliberate settle-then-click, rapid double-click
- Typing: touch-typist smooth vs hunt-and-peck with variable gaps
- Modifier release: synchronous (dominant) vs staggered
- Error recovery: 2–5% episodes include backspace-correct or overshoot-correct, primitive still succeeds

**Instruction diversity (≥10 phrasings per primitive):**
- Formality: "Please click..." / "click..." / "click the thing that..."
- Verbosity: terse / medium / verbose
- Ambiguity: unambiguous / requires disambiguation
- Pattern: imperative / declarative / conversational

### Ablation knobs exposed by generator (forward ref to Q27)

- Visual diversity level (low/med/high)
- Distractor count (0/low/med/high)
- Tempo variability (fixed/varied)
- Instruction phrasing count (1/few/many)
- Pre-state noise (none/mild/strong)
- Behavioral noise on/off

Each settable per data generation run for ablation sweeps.

### Kept deferred

- **Highlight primitives** — redundant with Drag on action side; grounding variant better addressed by improving target-referring expressions in other primitives. Revisit if text-selection becomes a specific need.
- **Composite modifier primitives** (Cmd+Click, Shift+Drag) — tackle after Group A/B validated.
- **Press-key-N-times** — covered by Chord + Typing patterns.
- **Type-then-Enter** — System 2 concern (primitive chaining), not a new primitive.
- **Triple-click, click-drag-with-pause** — niche, not worth v1 complexity.

**Q10 — Visual diversity strategy. [DECIDED]**

**Core principle:** prevent shortcut learning by ensuring every attribute the instruction could anchor on (color, shape, size, position, label, etc.) varies across the dataset, and by constructing adversarial scenes where simple single-attribute shortcuts fail.

### Theme-constrained sampling (Approach B)

Reject independent per-axis sampling (produces incoherent UIs). Instead: define coherent **themes** as consistent tuples of (background palette, target-shape preference, font family, color palette, style descriptors). Per episode, sample a theme → sample specific values within that theme's constraints.

**Theme-parameterized renderer:**
```
Theme = {
    background_palette: [...],
    target_shapes: [...],
    font_family: [...],
    color_palette: [...],
    border_style: {...},
    shadow_style: {...},
    ...
}
```
Adding a new theme is a config dict, not new code.

### ≥20 coherent themes

1. Flat modern — solid pastel bg, rounded rectangles, sans-serif
2. Flat minimal — white bg, sharp rectangles, high contrast
3. Neobrutalist — bright solids, thick borders, bold sans-serif
4. Dark mode modern — dark gray bg, muted accents
5. Dark mode high-contrast — pure black bg, bright accents
6. iOS-style — off-white bg, rounded pills, SF Pro-ish
7. Material Design — subtle gradients, rounded rectangles, Roboto-ish
8. Terminal/CLI — monospace, solid dark bg, green/amber text
9. Retro Windows 95 — gray bg, sharp borders, dropped shadows
10. Skeuomorphic — gradients, shadows, realistic textures
11. Pastel soft — low-contrast rounded shapes
12. Bold saturated — primary colors, sharp rectangles
13. Newspaper/document — serif, high information density
14. Dashboard/data-heavy — multiple panels, varied content
15. Form-heavy — input fields, labels, button rows
16. E-commerce — cards with images/prices/CTAs
17. Game UI — semi-transparent overlays, unconventional shapes
18. Accessibility high-contrast — WCAG AAA, bold outlines
19. Textured background — wood/paper/metallic behind UI
20. Noisy/cluttered — real-world distractor density

Per-theme randomization within constraints (hue, specific font, distractor count, target position) produces thousands of distinct compositions from these base themes.

### Diversity axes

**Scene-level:** background (solid/gradient/texture/pseudo-image), theme (20+), downscale artifact (bilinear/bicubic/nearest/area).

**Target-level:** shape (rectangle/rounded-rect/circle/icon/irregular), size (14–100 source px), color (full HSV), position (full canvas), label text (letter, font, weight, size), stroke/fill style.

**Distractor-level:** count (0–N), similarity to target (adversarial — same color/shape/size), spatial relationship (near/far/overlapping).

**Instruction-level:** phrasing template (10+ per primitive), referring expression strategy (color-only / shape-only / label-only / position-only / relational / compound), formality, verbosity.

**Cursor/proprio-level:** cursor start position (uniform over canvas), cursor sprite (standard / custom / high-contrast), keys held at start, CapsLock mode (from Q9).

### Adversarial scene generation (~20–30% of episodes)

Deliberately construct scenes where single-attribute shortcuts fail. Example:
- Instruction: "click the red Submit button"
- Scene: red Cancel (same color, wrong label), blue Submit (same label, wrong color), red Submit (correct), grey distractor
- Shortcuts fail: "click red" hits Cancel; "click Submit" hits blue Submit
- Only multi-attribute grounding succeeds

Forces model to attend to multiple attributes rather than exploit single correlations.

### Scene-instruction coupling

Critical: instruction phrasing must use only attributes that **uniquely disambiguate** the target within the scene.

**Generation flow:**
1. Sample theme and scene composition (target + distractors)
2. Identify which target attributes are unique in the scene (color, shape, size, position, label)
3. Sample an instruction phrasing strategy that uses only uniquely-disambiguating attributes
4. Generate the instruction text using that strategy

Teaches the model that different attributes can be the anchor depending on scene context, rather than always-all-attributes.

### Explicit anti-shortcut rules

Every axis the instruction could anchor on must vary:
- "Red button" → scene must contain red non-buttons and non-red buttons in other episodes
- "Top-right button" → scene must contain buttons in top-right that aren't labeled that way in other episodes
- "Submit button" → scene must contain Submit buttons of various colors in other episodes

Verified by dataset-level statistics: marginal distribution of each attribute should be ~uniform across primitives; joint distribution (target-attribute × instruction-attribute) should not show spurious correlations.

### Ablation knobs (forward ref Q27)

Diversity level settable at data generation:
- **Low:** 3–5 themes, ≤5 distractors, uniform sampling, no adversarial
- **Medium:** 10 themes, adversarial 10%, varied instructions
- **High:** 20+ themes, adversarial 25%, full phrasing strategy variety

Ablation measures OOD-slice degradation as diversity decreases — answers "what's the cheapest diversity level that still generalizes?"

### OOD slice construction (coupling with Q8)

OOD test slices draw on:
- Held-out themes (train on 16 of 20, test on 4)
- Held-out phrasing strategies (train on 5 of 10 templates per primitive, test on the other 5)
- Held-out color/size/position ranges
- Held-out distractor configurations

Measures whether diversity is producing genuine generalization vs memorization.

**Q11 — Cursor rendering. [DECIDED]**

**Cursor rendered onto frame AND in proprio** (redundant but reinforcing).

**Core rationale:** proprio carries exact cursor position (precise signal); rendered cursor provides a visual region-level cue and matches real-screen distribution. At `max_num_patches=256` the cursor spans ~1 patch (sub-patch at macOS default size), so vision provides coarse localization while proprio handles precision.

**Cursor sprite specs:**
- **Default:** macOS-accurate — black fill, white outline, subtle drop shadow (soft gray, offset down-right, blurred). The shadow is what makes the cursor visible against any background.
- **Size:** randomized 24–56 source pixels per episode, centered on macOS default (~32 px). Model learns cursor-as-concept across sizes rather than specializing to one sprite.
- **Variants (for Q10 diversity):**
  1. macOS default arrow (black + white outline + drop shadow)
  2. High-contrast arrow (pure black / pure white, no shadow)
  3. Bold arrow (thicker, accessibility variant)
  4. I-beam (text contexts — see below)
  5. Grab / grabbing (drag contexts — see below)

**Context-aware sprite swapping (v1):**
- **Default arrow:** most primitives
- **I-beam:** during typing primitives when cursor is over a text-input field (if rendered)
- **Grab:** during click-and-hold and drag primitives when cursor is over a draggable element (pre-press)
- **Grabbing:** during drag primitive while L is held and cursor is dragging

Matches real macOS behavior. Teaches the model cursor-type is a real signal tied to context.

**Rendering details:**
- Cursor drawn on top of all UI elements
- Anti-aliased edges and subtle shadow for visibility against any background
- No cursor trail or motion blur (unrealistic, adds complexity)

**Deferred to Q39 (embodiment generality):**
- Full macOS cursor-state taxonomy (pointing-hand, crosshair, wait/beachball, not-allowed, resize-ew/ns, etc.) — requires app-state modeling our synthetic env doesn't have in v1
- Non-Mac cursor sprites (Windows, Linux) — real embodiment generality

**No training-time proprio noise** (reversed earlier decision). Real proprio isn't noisy; training-time noise would have been addressing a non-existent problem. If eval reveals over-reliance on proprio, introduce dropout-style regularization then.

**Aspect-ratio generalization deferred to Q39** — normalizing cursor `(x, y)` to [0, 1] per-axis is a starting point but may not fully transfer across aspect ratios without training-time aspect-ratio variety. Not a v1 concern given single aspect.

**Cursor-sprite OOD ablation (Q27):** test-time evaluation using a cursor sprite the model never saw during training (different color, different shape). Measures whether the model learned cursor-as-concept or specialized to a specific sprite. If the model fails on a novel sprite, bump up Q10 cursor-sprite diversity.

**Q12 — Language instruction diversity. [DECIDED]**

Formalizes the instruction-diversity scope that Q9 and Q10 partially addressed. Core approach: **procedural combinatorial phrase generation** rather than hand-written templates.

### Procedural template system

Define pools per primitive; sample combinations per episode. Total phrasing space ~10⁴+ per primitive without hand-writing each:

**Verb pools (per primitive type, 5–10 verbs each).** Note: pools deliberately include cross-primitive overlap to prevent verb→primitive-type shortcuts (per Q12 anti-shortcut rules). "Press" appears in Click, Chord, and Press-and-hold. "Hit" appears in Click and Chord. "Open" appears in Double-click. The model must use context, not verbs, to distinguish primitives.
- Click: "click" / "press" / "tap" / "select" / "hit" / "activate"
- Right-click: "right-click" / "secondary-click" / "press with two fingers on" / "open the context menu for"
- Double-click: "double-click" / "dbl-click" / "open" (in some contexts) / "click twice on"
- Drag: "drag" / "move" / "pull" / "slide"
- Type: "type" / "enter" / "write" / "input"
- Chord: "press" / "use" / "hit the shortcut" / "activate"
- Hover: "hover over" / "move the cursor to" / "point at" / "put the mouse on"
- Scroll: "scroll to" / "scroll until" / "bring into view"

**Politeness prefix pool:** ∅ / "please" / "could you" / "I'd like you to" / "go ahead and" / "now"

**Referring expression strategies** (from Q10 scene-instruction coupling — sample based on what uniquely disambiguates the target):
- Color: "red button"
- Label: "'Submit' button"
- Shape: "circular icon"
- Position: "button in the top-right"
- Relational: "just below the red button"
- Compound: "red 'Submit' button in the lower-right"
- Superlative: "biggest button"
- Disambiguating: "leftmost of the two red buttons"

**Trailing context pool (optional):** ∅ / "to finalize the form" / "to close the dialog" / "when you're ready" / "if possible"

### Syntactic variety (sampled with weights per primitive)

| Pattern | Example | Weight |
|---|---|---|
| Imperative terse | "click red submit" | 15% |
| Imperative standard | "click the red Submit button" | 40% |
| Imperative polite | "please click the red Submit button" | 20% |
| Declarative | "I want you to click the red Submit button" | 10% |
| Question-as-command | "could you click the red Submit button?" | 10% |
| Fragmentary | "the red Submit button" | 5% |

### Typos / informal language

**5% rate** for click/drag target references only. Injection strategies:
- Character substitution / deletion / transposition at ~1 per 15 chars
- Informal contractions: "btn" for "button", "dbl-click" for "double-click"
- Lowercase-only ("click the red submit button")

**Not applied to typing target strings** (complicates the learning signal — if instruction says type 'hello' but 'hello' is misspelled in the instruction, does the model type the typo or the intended?). Typing target strings kept clean.

### Instruction length distribution

Natural distribution, not uniform:
- Short (≤6 words): 50%
- Medium (7–15 words): 35%
- Long (16+ words): 15%

Matches real System-2 planner output (mostly concise with occasional verbose).

### Instruction-target coupling (re-emphasis from Q10)

**Every instruction must be uniquely resolvable from the scene.** If phrasing is "click the red button," scene contains exactly one red button. If ambiguous, phrasing gets more specific until unique. Phrasing strategy is chosen based on what disambiguates the target in the current scene — never the other way around.

### Shortcut-prevention rules

Dataset-level statistics to monitor:
- **Verb-match shortcut:** verbs should occasionally overlap across primitives (e.g., "tap" for click, "press" for chord, "hit" for both)
- **Length shortcut:** instruction length distribution should not systematically correlate with primitive type
- **Saliency shortcut:** target visual saliency should not systematically exceed distractors (handled in Q10 adversarial scenes)
- **Position shortcut:** instructions using position references should cover all regions, not just "center"

### Not included in v1

- **Multilingual instructions.** SigLIP2's text tower supports it natively, but requires native/validated phrasings per language and interacts with keyboard layout (out of scope — Q39 concern).
- **Long natural-language contexts** (paragraph-length). Instructions stay focused on the primitive.
- **Markdown / formatting.** Plain text only.
- **Emoji / non-ASCII characters in instructions.** Clean ASCII for v1.

**Q13 — Rendering latency modeling. [DECIDED — DEFER]**

**Decision:** No rendering latency modeling in v1. Clean pygame — action at frame `t` produces visible change at frame `t+1`, no delay.

**Rationale:**
- Our primitives don't critically depend on "see my action reflected in the next frame" — proprio carries instant state feedback
- Target positions are stationary across frames in Groups A/B; rendering delay wouldn't affect grounding
- Typing primitive's visual feedback is secondary; proprio action history carries typing progress
- Adding 1-frame delay 5% of the time would be a half-measure — either too small to matter or trains for irrelevant noise

**Implementation flag:** verify the frame-reading order in training/inference pipeline. Common bug: model reads stale frame `t` (pre-action) instead of rendered frame `t+1` (post-action), which effectively bakes a 1-frame latency into the pipeline unintentionally.

**Deferred to Q37 (real-data bridge):** rendering latency becomes a genuine distribution-shift concern when transferring to real screens (30–100ms typical, 1–3 frames at 30 Hz). Mitigations available then:
- Latency-aware synthetic training (variable 1–3 frame rendering delays)
- Visual-consistency regularizer (predict future frame conditional on action)
- Action-effect time-shift augmentation
- Not needed for v1; address during real-screen transfer work.

**Q14 — Train/val/test splits. [DECIDED]**

Extends Q8's sketch with concrete construction strategy.

**In-distribution splits (episode-level, stratified by primitive type):**

For each primitive type independently:
- 80% train
- 10% val
- 10% test

Stratification ensures each split has representative coverage of every primitive (including rare ones like press-and-hold at only ~1000 episodes). Random 80/10/10 without stratification can yield val/test sets with few rare-primitive episodes → noisy per-primitive metrics.

**Leakage check at dataset construction:**
Flag any episode pair (train ↔ val/test) with:
- Same target position
- Same target color and label
- Same theme

Not automatically disqualifying — flags the degree of near-duplication. High duplication count means in-distribution test accuracy will overstate generalization even within the training distribution.

**OOD slices (generated as separate datasets, not carved from test):**

Each slice uses explicit held-out parameters so in-distribution test performance and OOD performance are cleanly separated.

| Slice | Train saw | OOD has |
|---|---|---|
| Themes | 16 of 20 themes | The other 4 themes |
| Phrasings | 70% of phrasing templates | Held-out 30% |
| Colors | Primary palette (e.g., R/G/B variants) | Unused hues (orange, purple, cyan) |
| Target sizes | 20–60 source px | 80–140 source px |
| Target positions | Center + standard edges | Extreme corners, unusual clusters |
| Distractor density | 0–5 distractors | 6–12 distractors |

Each OOD slice: ~1,500–3,000 episodes, stratified across primitive types so every primitive is measured on each slice.

**Val usage:** both HP tuning and early stopping. Accepts small optimism bias in v1; not worth splitting val into sub-sets at this scale.

**Test discipline:** measure on test **once** at the end, after all design decisions are locked. Iterate on OOD slices during development so test stays unseen. If we need multiple iteration rounds, use val + OOD slices for all decisions, touch test only for final reporting.

**Per-episode metadata** for filtering and slice-level analysis:
- Split assignment (train/val/test/ood_themes/ood_phrasings/ood_colors/...)
- Seed
- Primitive type
- Theme ID
- Phrasing template ID
- Any OOD slice memberships (an episode can be in only one slice)

Allows fine-grained eval decomposition during development.

### B. Model architecture details (5 — medium-high priority)

**Q15 — Trunk architecture. [DECIDED]**

**Action expert with learnable query tokens + cross-attention.** Matches the dominant modern pattern (OpenVLA-OFT, π0, GR00T-N1, SmolVLA).

### Core architecture

- **K = 16 learnable query tokens** — fixed "read heads" that extract task-relevant features
- **3 alternating cross-attention + self-attention blocks** (6 total attention layers)
- **Hidden dim: 768** (matches SigLIP2-B vision tower output — no projection needed)
- **Attention heads: 12**
- **FFN multiplier: 4×**

**K/V pool for cross-attention (≈265 tokens):**
- Vision tokens from SigLIP2-B: ~240
- Text tokens from SigLIP2 text tower (cached per primitive): ~16 for typical instruction
- Proprio token: 1
- Action history tokens: 8

**Queries (16) cross-attend to K/V pool, then self-attend among themselves** (so queries can coordinate — e.g., mouse queries and key queries share context when both are relevant).

**Output:** 16 final query hidden states. Heads pool/read from these.

### Proprio integration — token via 2-layer MLP

- 83-dim proprio (cursor xy + 77 held keys + 3 held mouse + CapsLock mode) → 2-layer MLP with GELU → 768-dim token
- Concatenated into K/V pool alongside vision/text/history
- **Not FiLM, not AdaLN** — token is simpler and matches OpenVLA-OFT / π0 / GR00T precedent
- Literature signal: most modern VLAs use MLP-encoded proprio tokens; FiLM/AdaLN dominant only in diffusion-style action heads (FLOWER, MDT)

### SigLIP2-B adaptation — LoRA on vision, frozen text

- **Vision tower:** LoRA rank 8 on Q/K/V/O projections (~0.5–1M extra trainable params)
- **Text tower:** fully frozen (instruction encoding cached per Q6 — no reason to adapt)
- **Rationale:** SigLIP2 is pretrained on natural photos (WebLI 10B); our domain is synthetic GUI screens with flat UI elements, specific palettes, pixel-perfect boundaries. LoRA is cheap insurance for small domain-specific adaptation without risking catastrophic forgetting of pretrained grounding features (at our ~1M transition data scale, full fine-tune is the riskier option per π0 knowledge-insulation findings)

### Action history integration (spec'd in Q19)

- 8 past-action frames, each a ~300-dim composite vector (head one-hots + key press/release masks + done bit)
- MLP projection `300 → 256 → 768` shared across timesteps (~273K params)
- Learned temporal positional embedding per timestep (~6K params)
- 8 resulting tokens concatenated into K/V pool

### Parameter breakdown (actual, empirically measured)

| Component | Params | Frozen? |
|---|---|---|
| SigLIP2-B vision tower | ~42M | LoRA rank 8 on attention projections |
| SigLIP2 text tower | ~44M | Fully frozen |
| LoRA adapters (vision) | ~0.5–1M | Trainable |
| Trunk (3 cross-attn + 3 self-attn blocks at dim 768, 12 heads, 4× FFN) | ~18M | Trainable |
| Proprio MLP (83→768, 2 layers) | ~0.2M | Trainable |
| Action history MLP + temporal embedding (Q19) | ~0.28M | Trainable |
| Output heads (unpooled — flatten(16×768) → logits for each of 6 heads) | ~3.7M | Trainable |
| **Total inference** | **~118M** | — |
| **Total trainable** | **~22M** | — |

Note: earlier "20M trainable / 106M total" estimate was optimistic — flattened unpooled heads add ~3.7M (discussed in Q16). See Q16 for rationale on keeping unpooled.

### Latency (empirically measured, see Q16 for full discussion)

| Stage | Measured (M1 fp16 MPS) |
|---|---|
| SigLIP2-B vision tower @ max_patches=256 | 80–92ms |
| Trunk A (3 blocks, 32M actual) | 40ms |
| 6 unpooled heads | negligible |
| **Full pipeline** | **~132ms → ~7.6 Hz wall-clock** |

Wall-clock 30Hz requires INT8+MLX or similar optimization (Q29). v1 uses pygame slowdown to achieve logical 30Hz during training.

### Q27 ablations queued

- **Query count K:** {4, 8, 16, 32}. Expected: 8–16 optimal.
- **Proprio integration:** token (default) vs FiLM on queries vs none. Expected: token wins.
- **SigLIP2-B adaptation:** frozen vs LoRA rank 8 (default) vs LoRA rank 16. Expected: LoRA 8 best on OOD; frozen best on pure efficiency.

### Rejected / deferred

- **Full fine-tune of SigLIP2-B** — risks catastrophic forgetting at our data scale; π0 knowledge-insulation work found this needs massive data to work cleanly
- **FiLM / AdaLN conditioning** — less common in transformer action experts; token is simpler and standard
- **Concat-all self-attention (no separate queries)** — wastes attention capacity per Q6 analysis; cross-attention with queries is more targeted
- **Separate query groups per head** (e.g., mouse queries vs keys queries) — adds architectural complexity for marginal interpretability gain; defer unless needed

**Q16 — Model size target. [DECIDED]**

**Decision: Trunk A (3 blocks, ~32M actual params), heads unpooled.** Total inference ~118M (SigLIP2-B 86M + LoRA ~1M + trunk 32M + projections/heads).

### Empirical latency measurements (M1 MacBook Pro, PyTorch fp16 MPS)

Medians from 100 timed forward passes (after 20 warmup):

| Component | Actual params | Median latency | Hz |
|---|---|---|---|
| SigLIP2-B naflex @ max_patches=256 | 375M (loaded with text tower) | 80–92ms | 10.8–12.5 |
| Trunk A (3 blocks) | 32M | 40ms | 25.1 |
| Trunk B (6 blocks) | 60M | 75ms | 13.4 |
| Full pipeline A | — | ~132ms | 7.6 |
| Full pipeline B | — | ~167ms | 6.0 |

**Honest confrontation with prior estimates:** earlier Q6 projections of 15–28ms/frame were fantasy. Real PyTorch+MPS+fp16 is 4–5× slower. INT8+MLX might get us 2–3× back (→ ~45–65ms/frame = 15–22 Hz), but that requires a migration to MLX or CoreML, not just a flag flip. PyTorch MPS has no INT8 path (`torch.ao.quantization` targets CPU/CUDA; bitsandbytes has no MPS backend).

### 30 Hz is aspirational, not achieved

Real-time 30Hz on M1 with this architecture and current (PyTorch/MPS) inference stack **is not reachable**. Even with aggressive naflex (max_patches=64) and INT8+MLX optimization, best-case projection is 15–22 Hz wall-clock.

**v1 framing:**
- **Logical framerate: 30Hz.** Model trained and evaluated as if running at 30Hz.
- **Wall-clock framerate: 7–15Hz on current stack, potentially 15–22Hz with INT8+MLX.**
- **Approach:** pygame env paused during inference. Agent experiences discrete 30Hz-stamped frames; model learns the 30Hz policy; wall-clock is decoupled.

This is legitimate because training and within-env eval are both slowdown-tolerant (we control the env). Real-screen deployment (deferred to Q37) would hit a wall here and require either running at achievable wall-clock Hz (~11 Hz currently) or a dedicated inference optimization project (Q29).

### Sizing rationale

**Why Trunk A (32M) over Trunk B (60M):**
- Trunk A adds ~40ms on top of encoder; Trunk B adds ~75ms (roughly doubles trunk cost for +28M params)
- At 1M transitions, 32M trainable is on the low end of my estimated compute-optimal range (30–60M); 60M is on the high end
- A preserves latency budget for deployment optimization later; B consumes it now
- If A underperforms on primitives, B is an obvious next step and we'll have ablation evidence for why
- Smaller means faster training iterations → more ablation runs feasible

Training data regime (~1M transitions) supports either size roughly equally on a Chinchilla-ish scaling heuristic. The deciding factor is "validate the smaller version first."

**Confidence note (from Q16 discussion):** My best guess at compute-optimal at 1M transitions was ~30–60M trainable trunk, with *low-to-medium* confidence. I don't know of a scaling-law study specific to VLA action experts at our data scale. Trunk A is on the lower end; plausibly modestly undersized. If primitives underperform, capacity is the first place to look.

### Why heads stay unpooled

The earlier pooling proposal (pool 16 queries → 768-dim vector before heads; save ~3.5M params) was a premature optimization. Empirical measurement showed **pooled and unpooled have the same Hz** on M1 — latency was not the reason to pool.

Critical-reflection analysis of pooling's learning impact:

**Pooling definitely loses information when:**
- Queries specialize by head (e.g., query 3 → mouse dx; query 11 → keyboard state). Mean-pool averages these; unpooled heads can select.
- Queries encode spatial positional signal. Pooling collapses it.
- Compound-attribute grounding needs multiple facets simultaneously ("red Submit button"). Pooling mixes.

**Pooling is roughly lossless when:**
- Queries are redundant (common under-trained or for simpler primitives)
- Heads need aggregate signal (done head, scalar judgments)
- Deep self-attention has already mixed queries into near-identical representations

**Verdict:** unpooled guarantees no information loss at ~3.5M param cost (~10% of trunk). Mean-pool probably costs low-single-digit % accuracy on compound-grounding primitives. Attention-pool with per-head weights (each head learns what to extract through a rank-768 bottleneck) is near-free but adds architecture.

**For v1, stay unpooled.** We're trying to validate the architecture; don't introduce capacity-reducing choices without reason. Pooling as an ablation in Q27.

### Concrete v1 sizing

| Component | Params | Trainable | Role |
|---|---|---|---|
| SigLIP2-B vision tower | ~42M | LoRA rank-8 only (~0.5M) | Grounding |
| SigLIP2 text tower | ~44M | Frozen | Instruction encoding (cached per primitive) |
| Trunk A (3 blocks, dim 768, 12 heads, 4× FFN) | ~18M | Full | Task reasoning |
| Query tokens (16) | ~0.01M | Full | Action expert |
| Proprio MLP (83→768) | ~0.2M | Full | State integration |
| Action history lookup | ~0.1M | Full | Temporal context |
| Output heads (unpooled, flatten 16×768→{21,21,5,21,231,1}) | ~3.7M | Full | Action prediction |
| **Total inference** | **~118M** | — | — |
| **Total trainable** | **~22M** | — | — |

### Q27 ablations queued for later

- Trunk A (3 blocks) vs Trunk B (6 blocks) — directly measures capacity-vs-speed tradeoff at our data scale
- Unpooled vs mean-pool vs attention-pool (per-head) heads — measures pooling's learning impact
- Both feasible as post-v1 ablations once the A/unpooled baseline is validated

**Q17 — Fusion/conditioning wiring. [RESOLVED BY Q15]** Action expert with learnable query tokens cross-attending to a unified K/V pool containing vision tokens, text tokens, proprio token, and action-history tokens. Queries then self-attend among themselves. 3 alternating cross+self blocks.

**Q18 — Vision token reduction. [DECIDED]**

**No trunk-side vision token reduction in v1.** All 240 vision tokens from SigLIP2-B naflex @ max_patches=256 feed directly into the cross-attention K/V pool.

**Rationale:**
- Token count is already small (240). ShowUI-style ~33% pruning was designed for thousand-token regimes; at our scale the latency savings are trivial (maybe 10–15% of a 40ms trunk = ~5ms, or ~3% of pipeline)
- Latency pressure is encoder-bound (80ms), not trunk-bound (40ms). Trunk-side reduction doesn't meaningfully move the pipeline
- Encoder-side reduction via `max_num_patches` is the right latency lever — already in Q29's optimization roadmap (max_patches=64 → ~27ms encoder)
- Cursor already at sub-patch coverage at max_patches=256; further spatial reduction directly hurts an already-marginal signal
- Pruning introduces a learning problem (which tokens to drop) we don't need to solve

**Deferred to Q29 / Q27:**
- Aggressive `max_num_patches` reduction (64, 128) as a deployment-time latency lever
- Token pooling ablation (240 / 180 / 120 via spatial pooling) as a lower-priority Q27 measurement if we ever want empirical capacity-vs-latency tradeoff data

**Q19 — Action history token embedding. [DECIDED]**

Per Q5: 8 past-action frames added as K/V tokens to the trunk's cross-attention pool. Q19 specifies the exact embedding format and training regime.

### Per-frame action embedding: MLP-projected composite vector

Each past-frame action is a 6-head composite: mouse dx/dy (21-way each), click (5-way), scroll (21-way), keys (77 × 3-way), done (1 binary).

**Raw past-action vector:** ~300 dims constructed per frame
- dx one-hot: 21
- dy one-hot: 21
- click one-hot: 5
- scroll one-hot: 21
- keys as press/release masks: 154 (77 press bits + 77 release bits)
- done: 1 scalar

**MLP projection:** `300 → 256 → 768` (two layers, GELU). Shared across all 8 past frames. ~273K params.

**Rejected alternatives:**
- **Flatten-to-single-categorical** — combinatorial explosion (21×21×5×21×3^77 possibilities)
- **Per-head lookup embeddings + sum** — ~177K params dominated by 77-key × 3-state lookup; redundant on mostly-idle frames
- **Per-head separate tokens** — explodes K/V pool from 265 to 313+ tokens

### Temporal positional encoding

Learned 768-dim embedding per timestep {−1, −2, …, −8}, added to each history token after MLP projection. 8 × 768 = 6K params. Standard and cheap.

### Training regime: plain teacher forcing (no scheduled sampling)

**Use ground-truth past actions throughout training.** Do not mix in model's own predictions.

**Rationale — scheduled sampling is contested in the literature and uncertain value for our setup:**

*Why I almost recommended scheduled sampling (Bengio et al. 2015):* exposure bias is a real phenomenon in autoregressive sequence models — training on ground-truth history but inferring with model-predicted history creates a distribution mismatch.

*Why literature evidence is mixed:*
- Huszár 2015 showed scheduled sampling is theoretically *inconsistent* — doesn't converge to true data distribution even in infinite-data limit
- Goyal et al. 2017 found gains fragile and hard to reproduce on transformer seq2seq
- Wang et al. 2020 and follow-ups: often neutral or negative in transformer settings
- Modern LLM training (GPT-3+) uses plain teacher forcing
- Modern VLA training (π0, OpenVLA, OpenVLA-OFT, ACT) does not use scheduled sampling on action history

*Why our specific setup is low-risk for exposure bias:*
- Per-frame policy (not multi-step autoregressive generation) — errors don't compound across long rollouts
- Discrete argmax actions are low-dim; a well-trained model's predictions should match ground-truth most of the time
- Proprio carries real state; action history is a hint, not the primary signal
- Current observation is always real at both train and inference time

**Teacher forcing is the safer default.** Add scheduled sampling only as a targeted fix if eval reveals train/inference mismatch (e.g., double-click underperforms in closed-loop rollouts vs offline eval).

**Better alternative if exposure bias becomes a concern:** periodic closed-loop rollout evaluation during training exposes train/inference gaps directly without requiring architectural change (Q28 evaluation infrastructure).

### Q27 ablations queued

- **Teacher forcing (default) vs scheduled sampling** — only if closed-loop rollout eval shows a specific gap. Low priority.
- **Action history length K** ({0, 4, 8} — from Q27 backlog) — validates whether history helps at all.

### Summary

- History embedding: MLP 300→256→768 over per-frame composite action vector (~273K params)
- Temporal positional embedding: learned per-timestep, 6K params
- Training: plain teacher forcing, no scheduled sampling
- 8 history tokens added to K/V pool alongside vision/text/proprio
- Total Q19 cost: ~279K params (trivial)

### C. Training (4 — medium priority, post-architecture)

**Q20 — Optimizer & learning rate schedule. [DECIDED]**

### Optimizer: AdamW

**Confidence: high.** AdamW is the established default for transformer fine-tuning, used in every modern VLA reference (OpenVLA, OpenVLA-OFT, π0, ACT, SmolVLA).

**Why not Muon (despite 2025–2026 hype):**
- Muon's documented wins (Moonshot's 2× compute efficiency, Kimi-2, Essential AI's Pareto frontier) are at **pretraining scale** (500M–1T+ params, trillion-token text corpora), not fine-tuning at ~1M transitions
- Muon requires careful parameter grouping: [literature consensus](https://kellerjordan.github.io/posts/muon/) is 2D hidden weights → Muon, embedding/output/1D params → AdamW. Adds implementation surface
- Our trunk is small (~18M of 2D weights). Muon's efficiency advantage is less pronounced at this scale
- Training dynamics are closer to fine-tuning (mostly-frozen encoder) than pretraining, where Muon's motivation applies most strongly
- **Confidence: low-medium** that Muon would help at our specific setup

**Queued as a Q27 ablation** if v1 trains successfully and we want to squeeze more compute efficiency. Not v1 default.

### Parameter groups & learning rates

Two groups with distinct learning rates:

**Group 1: LoRA adapters on SigLIP2 vision tower** (~1M params)
- LR: **2e-4** — standard LoRA fine-tuning default per Latitude / Unsloth 2025 guides
- Confidence: high
- Note: Thinking Machines "LoRA Without Regret" (Sept 2025) suggests ~10× full-FT LR for adapters-on-all-layers setups → could justify up to 2e-3. Stick with conservative 2e-4 for v1; ablate if desired.

**Group 2: Trunk + heads + proprio MLP + history MLP + query tokens** (~21M params, trained from scratch)
- LR: **3e-4** — standard from-scratch transformer default
- Confidence: high

**Shared:**
- Weight decay: 0.01 (standard AdamW)
- Betas: (0.9, 0.95) — β2=0.95 more common in recent transformer work than 0.999
- Confidence on defaults: medium-high

### LR schedule: Warmup + cosine decay

- **Warmup:** 500–1000 steps (linear ramp 0 → max LR). Prevents early-training instability.
- **Main:** cosine decay from max LR → 10% of max LR
- **Total duration:** tied to Q21 (batch size + training length)

**Confidence: medium-high.** Standard transformer schedule.

**Alternative noted but not chosen:** constant LR (used by Thinking Machines in LoRA study) is simpler and works fine when LR is well-tuned. Cosine is chosen because it's what every modern VLA reference uses and is robust to imperfect LR tuning.

### Gradient clipping

- Global gradient norm clip at **1.0** (standard transformer default)
- **Per-head gradient *monitoring*** (from Q2) — log, don't clip per-head. Diagnostic only.
- **Expected clipping rate: <10% of batches.** If clipping triggers every batch, loss scaling is wrong (exp2 burn — clipping masked a corrupted baseline).

**Confidence: high.**

### Precision: bf16 mixed precision

- bf16 on L40S (native support); avoids fp16-overflow issues that can bite with MPS
- Confidence: high

### Summary table

| Decision | Value | Confidence |
|---|---|---|
| Optimizer | AdamW | High |
| Weight decay | 0.01 | High |
| Betas | (0.9, 0.95) | Medium-high |
| Group 1 (LoRA) LR | 2e-4 | High |
| Group 2 (trunk+heads) LR | 3e-4 | High |
| Schedule | Warmup 500–1000 + cosine → 10% | Medium-high |
| Gradient clipping | Global norm 1.0 | High |
| Precision | bf16 mixed | High |

### Q27 ablations queued

- **Optimizer:** AdamW (default) vs Muon — low priority, revisit if v1 succeeds
- **Group 1 LR sweep:** {5e-5, 2e-4, 1e-3}
- **Group 2 LR sweep:** {1e-4, 3e-4, 1e-3}
- **Schedule:** cosine (default) vs constant — low priority

### Dependencies

LR values assume typical batch sizes (64–512). Q21 (batch size) may require LR rescaling (linear-scaling rule loose for AdamW; sqrt-scaling sometimes preferred at large batch).

**Q21 — Batch size, training duration, early stopping. [DECIDED]**

### Batch size: 64 episodes (episode-level sampling)

**Confidence: medium.**

Episode-level batches (not frame-level): each macro-batch contains 64 full episodes, each contributing its full window of frames to the loss (macro-batch size ~3,500 frames on average across primitives). Realized as **8 micro-batches × 8 episodes per macro-batch** with gradient accumulation (see Q2/Q8): each micro-batch is homogeneous by primitive sub-type for clean tensor shapes; grads accumulate across 8 micro-batches; optimizer steps once per macro-batch. Across consecutive macro-batches the 14 primitive sub-types cycle so each gets roughly equal exposure.

Rejected:
- Frame-level sampling — breaks stratification, harder to reason about per-primitive coverage
- Batch size 32 — too little per-primitive signal (14 sub-types means only 2–3 samples per primitive)
- Batch size 128 — memory pressure on L40S; diminishing returns on gradient noise
- Single-primitive homogeneous batches without accumulation — gradient magnitude swings step-to-step as trunk alternates between primitive types
- True mixed batches with padding + attention/loss masks — ~200–500 LOC cost rejected in Q8

Effective macro-batch size can be further increased (to 128 or 256 episodes) by adding more micro-batch steps per optimizer step if per-primitive gradient variance is problematic.

### Training duration: 20 epochs (~7,600 steps)

**Confidence: medium.**

At batch 64 and ~24,500 episodes, one epoch = ~380 steps. 20 epochs = ~7,600 steps total.

Rationale:
- Our trunk (~18M) trains from scratch; from-scratch transformers at this scale typically need 10–30 passes over data
- Tulu3-style fine-tuning uses 2–5 epochs, but that's fine-tuning a pretrained LLM, not from-scratch trunk
- OpenVLA base pretraining used ~27 epochs on 1M trajectories (comparable scale, larger model)
- 20 sits in the middle; best-checkpoint + early stopping handle under/over-specification
- Total compute: manageable on L40S (few hours)

Alternatives considered:
- 10 epochs — likely undercooked for rare primitives and cosine schedule has little cooldown
- 30+ epochs — risk of overfitting at 1M transitions given ~22M trainable

### Early stopping: patience-with-floor

**Confidence: medium-high.**

- **Active from step 3,800 (50% of training) onwards.** No ES during warmup or first-half cosine decay.
- **Patience: 5 eval rounds** on val loss
- **Eval frequency: every 500 steps** → ~15 total eval rounds, ~7 eligible for ES
- **Best-checkpoint saving throughout training** regardless of ES

**Rationale for floor:**
- Cosine schedule commits to a specific endpoint via LR curve; stopping early during warmup/cosine kills runs that are still genuinely learning with high LR
- Second-half ES catches genuine saturation — up to ~17% compute saved if model converges by step 5000
- Best-checkpoint saving provides overfitting protection regardless of ES firing

**Two different metrics for two different jobs:**
- **ES metric: val loss** (smoother signal, less sensitive to per-primitive saturation timing)
- **Best-checkpoint metric: mean primitive success rate** (metric we actually care about; Q24 defines exact formula)

Per-primitive success rates saturate at different speeds (simple primitives early, complex drag/typing late). Using val loss for ES gives cleaner convergence detection; using success rate for checkpoint selection targets what we optimize for.

### DataLoader throughput (implementation flag)

Prior exp2 had DataLoader bottleneck. For 1M frames:

- Full RAM cache infeasible (~400GB)
- Stream from disk with 4–8 workers
- Cache naflex-preprocessed frames on disk for faster re-reads across epochs
- Memory-mapped numpy arrays for per-episode frame sequences

**Validate during implementation:** if training time per step is >2× pure forward+backward compute, I/O is the bottleneck and needs optimization.

### Summary

| Decision | Value | Confidence |
|---|---|---|
| Batch size (episodes) | 64 | Medium |
| Effective frames per batch | ~3,500 avg | Derived |
| Sampling unit | Episode-level | Medium-high |
| Training duration | 20 epochs (~7,600 steps) | Medium |
| ES activation floor | Step 3,800 (50% mark) | Medium-high |
| ES patience | 5 eval rounds | Medium |
| Eval frequency | Every 500 steps | Medium |
| ES metric | Val loss | Medium |
| Best-checkpoint metric | Mean primitive success rate | High |
| Warmup | 500 steps linear (from Q20) | High |
| LR schedule | Cosine to 10% of max over 7,600 steps (from Q20) | High |
| Gradient accumulation | 8 micro-batches × 8 episodes per macro-batch (for per-primitive micro-batch homogeneity per Q2/Q8; blended multi-task grad per optimizer step) | High |

### Q27 ablations queued

- **Batch size sweep** (32/64/128) — low priority unless gradient noise is an issue
- **Training duration** (10/20/30 epochs) — naturally covered by best-checkpoint tracking across runs

### Uncertainty flags

- 20-epoch count is an informed estimate, not empirically validated; best-checkpoint + ES handle miscalibration
- 50% floor for ES is a heuristic; log curves to calibrate for future experiments
- Eval every 500 steps is arbitrary middle ground between 250 (fine-grained, more overhead) and 1000 (coarse)

**Q22 — Regularization. [DECIDED]**

### Decision: status quo + weight EMA

**No additional trunk-level regularization in v1.** Existing stack is substantial (13 mechanisms already):

- Focal CE (down-weights confident predictions)
- Label smoothing on binned heads
- Per-class inverse-frequency weighting on keys
- LoRA rank 8 on vision tower (inherent low-rank regularization)
- Frozen text tower (no overfitting to instruction distribution)
- Weight decay 0.01
- Gradient norm clipping at 1.0
- Visual diversity (20+ themes from Q10)
- Adversarial scenes (~20–30% from Q10)
- Stratified primitive sampling (Q2)
- Large effective batch size (~3,500 frames avg)
- Early stopping with 50% floor (Q21)
- Best-checkpoint by val metric (Q21)

**Add only: weight EMA (decay 0.9999)** on trainable params for eval checkpoints. Low-cost, consistently used in modern VLA work (π0, OpenVLA-OFT and siblings have variants), zero downside since non-EMA checkpoint also saved.

### Rationale for status quo

**ViT regularization literature (Steiner et al. "How to train your ViT" 2021):** best setting when regularization helps is dropout 0.1 + stochastic depth 0.1. BUT: "model regularization mainly helps larger models, and only when trained for long" — our 3-block trunk and 20-epoch training don't clearly qualify. Literature was on 12-layer ViTs for ImageNet classification; transfer to 3-block VLA trunk is uncertain.

**Specific concerns with dropout at our scale:**
- Stochastic depth at 3-block depth drops 33% of effective depth per drop event — too disruptive; safe only at 6+ blocks
- Dropout 0.1 interaction with focal CE not well-studied (both suppress gradients; may compound)
- Dropout interaction with LoRA not well-studied
- Adds a hyperparameter without clear signal for correct value
- Reduces effective model capacity — opposite of what we want given Q16 concern that Trunk A may be on the low end of compute-optimal

**Specific concerns with R-Drop / AttentionDrop:**
- R-Drop: 2× forward-pass cost not justified at uncertain gain (confidence: 4/10)
- AttentionDrop: too new, insufficient validation (2025 paper)
- LayerDrop: designed for 48+ layer transformers, wrong tool (confidence: 9/10 reject)

**Specific concerns with Mixup/CutMix:** inappropriate for UI screens — mixing two UI screenshots creates ungrounded content that doesn't reflect any real scene. Reject (confidence: 9/10).

### Confidence levels

| Component | Default | Confidence |
|---|---|---|
| Attention dropout | None | 5/10 that status-quo is right; 5/10 that 0.1 dropout would help |
| FFN dropout | None | 5/10 that status-quo is right |
| Stochastic depth | None | 7/10 that skipping is right at 3-block depth |
| R-Drop | None | 6/10 that skipping is right; 2× cost not justified |
| LayerDrop | None | 9/10 reject; wrong tool for small trunk |
| Weight EMA 0.9999 | **Yes** | 7/10 that it's worth adding |
| Mixup/CutMix | None | 9/10 reject; inappropriate for UI |

### Monitoring during training

Three signatures to watch for that would trigger adding dropout in a v2:

1. **Val loss diverging from train loss** — classic overfitting signature. If val/train loss gap widens >10–15% late in training, add dropout 0.05–0.1 for v2. **Definition:** "train loss" here is train-set eval loss (running pass over a held-in train subset with dropout off), not running training-batch loss (which is noisier and dropout-on). Compute on same cadence as val eval (every 500 steps) on a ~500-episode train subset for fair comparison.
2. **Train loss continues decreasing while val primitive success plateaus** — val metric saturation despite continued training. Suggests memorization rather than generalization. Add dropout.
3. **OOD slice (Q8 held-out themes) underperforms in-distribution by >15%** — distribution-shift robustness gap. Might indicate under-regularization OR under-diverse training data. Add dropout only if data-diversity fix doesn't close the gap.

Per-head train/val gaps logged (from Q2 gradient monitoring) — helps distinguish overfitting from under-capacity.

### Q27 ablations queued (low priority)

- **Dropout 0 (default) vs 0.05 vs 0.1** — only if v1 shows overfitting signatures above
- **Weight EMA on vs off** — medium priority; should show small but consistent EMA advantage

### Uncertainty flags

- ViT literature (Steiner et al.) is ImageNet classification, not VLA action prediction. Transfer is plausible but unproven.
- Our "trunk" is 3 blocks over frozen encoder — differs from 12-layer ViT trained end-to-end.
- If v1 trains without overfitting, status-quo validates. If v1 overfits, we have a targeted fix ready.

### Principle applied

Under epistemic uncertainty, fewer moving parts makes diagnosis cleaner. Adding uncertain-benefit mechanisms preemptively obscures future ablation signal. Regularization is a drop-in fix if overfitting appears — not necessary to prevent it speculatively.

**Q23 — Training-time data augmentation. [DECIDED]**

### Decision: no additional training-time augmentation for v1

**Rationale: data-generation-time diversity from Q9–Q12 already substantial.**

Q23 is distinct from Q10–Q12 in that Q10–Q12 cover *generation-time variety* (the dataset has 20K diverse episodes), whereas Q23 asks about *training-time transforms* applied per-step (each episode becomes effectively infinite views).

### What we already have from prior Qs (generation-time)

- **Q9:** per-primitive tempo variability, pre-state diversity, typing string mix
- **Q10:** 20+ themes, theme-constrained sampling, adversarial scenes (~20–30%), scene-instruction coupling
- **Q11:** randomized cursor size 24–56px, context-aware sprites
- **Q12:** combinatorial instruction generation (10⁴+ per primitive), 5% typos, natural length distribution

This is higher data-generation diversity than most published VLA benchmarks.

### Training-time augmentation candidates analyzed

| Component | Default | Confidence it's right |
|---|---|---|
| Visual augmentation (color jitter, crops) | None | 4/10 |
| Instruction runtime paraphrasing | None | 3/10 |
| Action/proprio noise | None (Q3 already decided) | 8/10 |
| Resolution/aspect augmentation | None for v1 (Q39 will need for multi-embodiment) | 4/10 |
| Temporal jitter on action history | None | 6/10 |

### Reasoning for status quo

1. **Our deployment target is clean UI, not noisy natural photos.** Standard ViT augmentation (color jitter, random crop) is tuned for ImageNet-style data, not UI screenshots.
2. **Generator diversity exceeds augmentation benefit.** Combinatorial instruction generation already produces 10⁴+ variants per primitive — runtime paraphrasing adds little on top.
3. **Frozen text tower means runtime paraphrasing risk.** New phrasings may land on text encodings the frozen tower handles poorly.
4. **Status-quo principle (from Q22) applies.** Under uncertainty about benefit, cleaner baseline makes ablation diagnosis easier.
5. **Cheap to add later.** If v1 OOD slice underperforms, augmentation is a drop-in fix.

### What's explicitly rejected and why

- **Color jitter / brightness / contrast** — UI colors are semantic signals (e.g., button colors encode state); jittering obscures this. 6/10 reject.
- **Random crops** — our naflex pipeline already handles aspect/resolution variance; further cropping can clip task-relevant UI. 6/10 reject.
- **JPEG compression simulation** — our deployment renders clean pixels, not compressed images. 8/10 reject.
- **Mixup/CutMix** — already rejected in Q22; mixing UI screenshots produces ungrounded content. 9/10 reject.
- **Instruction paraphrasing at runtime** — frozen text tower may not handle paraphrased encodings well; combinatorial generator already covers phrasing variation. 7/10 reject.
- **Proprio noise** — already rejected in Q3; real proprio isn't noisy. 8/10 reject.

### Q27 ablations queued (low priority)

- **Visual augmentation (color jitter at 0.1)** — only if OOD slice underperforms in v1
- **Resolution augmentation** — queued for Q39 (multi-embodiment expansion), not v1

### Signatures that would trigger adding augmentation in v2

Same as Q22's overfitting signatures plus:

1. **OOD theme slice accuracy <70% of in-distribution** — visual distribution-shift gap. Augmentation is one knob; Q10 adversarial-scene fraction is another.
2. **OOD instruction phrasing slice underperforms** — language distribution-shift gap. Paraphrasing is one knob; Q12 instruction combinatorial coverage is another.

### Principle applied

Distinct from Q10–Q12's generation-time diversity, Q23's per-step augmentation offers diminishing returns when generation diversity is already high. The cases where training-time augmentation matters most (natural photos with limited data) don't apply to synthetic UI with procedurally-generated diversity.

### D. Evaluation (5 — medium priority, parallel with training)

**Q24 — Metrics per primitive. [DECIDED]**

### Two-tier evaluation system

**Tier 1: Per-primitive binary success** — drives best-checkpoint selection and final reporting.
**Tier 2: Continuous sub-metrics** — logged always, used to diagnose *why* episodes fail.

### Tier 1 — per-primitive success criteria

| Primitive | Success criterion | Tolerance | Confidence |
|---|---|---|---|
| Hover | Cursor inside target bbox at episode end | 0 px (bbox-inclusive) | 8/10 |
| Hover-and-wait | Cursor inside bbox for ≥10 contiguous frames at end | 0 px | 7/10 |
| Click-at-relative-position | Click fires with cursor inside relative bbox | 0 px | 8/10 |
| Left-click | L-click fires with cursor inside target bbox | 0 px | 9/10 |
| Right-click | R-click fires with cursor inside target bbox | 0 px | 9/10 |
| Double-click | Two L-clicks within 15-frame window at same target | 0 px, ≤15 frames | 7/10 |
| Click-and-hold | L-press at target, held ≥N frames without release | 0 px, N=primitive-spec | 8/10 |
| Drag A→B | L-press at A, L-release at B, cursor inside B bbox at release | 0 px endpoints; trajectory not required | 8/10 |
| Scroll-to-target | Target bbox entirely inside viewport at episode end AND no scroll events for last ≥5 frames (tests "recognize I'm done") | Bbox visibility + 5-frame quiescence | 6/10 |
| Type-short/med/long | Final visible text matches target exactly | Exact string match | 8/10 (6/10 on exact-match choice) |
| Chord | All keys in chord held simultaneously within 3-frame window | N=3 frames | 6/10 |
| Press-and-hold | Key held for ≥target duration without intermediate release | Within ±3 frames of target | 6/10 |

**Rationale for 0px tolerance:** our pygame env generates bbox ground-truth directly — no annotation noise unlike human-labeled ScreenSpot (which uses ~3-5px tolerance). Model clicks either inside or outside the designed target.

**Literature anchor:** ScreenSpot click accuracy (proportion of predicted clicks falling inside GT bbox) is the dominant grounding metric in 2025 GUI-agent literature. Our Tier-1 for click/hover primitives matches this convention.

### Tier 1.5 — tolerance / edit-distance curves (added 2026-04-23)

Binary Tier-1 doesn't distinguish "1px outside" from "100px off" or "1 typo in 16 chars" from "every char wrong." Tier-1.5 reports the underlying curves at negligible extra cost (data already captured in Tier-2):

| Primitive family | Tier-1 (unchanged) | Tier-1.5 curve |
|---|---|---|
| Click / hover / drag endpoints | Inside bbox (0px) | Success @ {0, 3, 5, 10}px cumulative tolerance |
| Type-short / med / long | Exact string match | Success at edit-distance {≤0, ≤1, ≤2} and normalized Levenshtein ratio |

**Role:** diagnostic granularity only. Best-checkpoint selection still uses Tier-1 (0px / exact-match) so physical-keyboard and bbox-strict semantics remain the primary metric. Tier-1.5 answers "is the model clicking near?" and "is the model mistyping by one key?" without changing what we optimize for.

### Tier 2 — diagnostic sub-metrics (always logged)

**Grounding sub-metrics (click/hover/drag):**
- Mean pixel distance from predicted click to target bbox center
- % episodes where cursor *ever* entered target bbox (vs never)
- % episodes where cursor entered a *wrong* bbox (distractor hits)

**Trajectory sub-metrics (drag/hover-and-wait):**
- Trajectory smoothness: mean curvature of cursor path
- Trajectory efficiency: path length / direct distance ratio
- Overshoot count: times cursor left target bbox after first entry

**Timing sub-metrics (temporal primitives):**
- Double-click inter-click latency (target: 10–15 frames)
- Press-and-hold duration vs target
- Chord simultaneity window (all keys held overlap)

**Keyboard sub-metrics (typing/chord):**
- Edit distance between typed output and target (Levenshtein)
- Per-key press/release F1 vs ground truth
- % of keys with wrong modifier state (e.g., shift-off when should be on)

### Aggregate metrics

**Best-checkpoint metric (from Q21): unweighted mean primitive success rate across types.**

Each primitive type contributes equally regardless of training data count. Rejected alternatives:
- Weighted by data proportion — biases toward well-represented primitives; redundant with training signal
- Weighted by assumed deployment frequency — we don't know deployment distribution at v1

**Confidence: 7/10.**

### Saturated-primitive handling

"Task difficulty modulates metric utility: binary success becomes uninformative for very easy or very hard tasks."

Rule of thumb:
- Primitive saturating >95% success: switch to Tier-2 metrics for finer signal (mean pixel error on clicks)
- Primitive <10% success: primitive is broken — investigate before caring about metric
- Middle range (10–95%): binary success is primary

**Confidence: 6/10** — captures the right idea; exact thresholds somewhat arbitrary.

### OOD generalization reporting

Not new metrics, but explicit per-slice breakdown:

For each OOD slice from Q8 (unseen themes, phrasings, colors, sizes, positions, distractor density), compute full Tier-1 + Tier-2 metrics. Report per-slice success rates.

**Key derived metric: OOD-to-InD ratio.** If ID success = 85% and OOD-theme success = 60%, ratio = 0.71. Higher = better generalization.

Signature to trigger Q22/Q23 review (add dropout/augmentation): ratio <0.70 consistently across OOD slices.

**Confidence: 8/10.**

### Not measured in v1

- **Efficiency / steps-to-completion (MMBench-GUI EQA):** we train on fixed-window episodes; efficiency is baked in
- **Execution quality / smoothness (Eval-Actions):** important in robotics (joint safety); cosmetic for GUI
- **Human preference rankings:** deferred; requires human eval
- **Behavioral diversity:** deferred

### Latency measurement (separate axis)

Per Q16/Q29, reported alongside success metrics:
- Inference ms/frame wall-clock on M1
- Training ms/step on L40S (diagnostic only)

Latency is a binary-gated constraint (must hit target), not a primary success metric.

### Logging cadence

**Every 500 steps (eval frequency from Q21):**
- Per-primitive Tier-1 success on val set (11–14 rows)
- Mean primitive success rate (best-checkpoint signal)
- Val loss total + per-head (7 scalars)
- Train loss + per-head (7 scalars)
- Grad norm per head (6 scalars)
- LR per group (2 scalars)

**Every 2000 steps (deeper eval):**
- Per-primitive Tier-2 sub-metrics
- OOD slice performance

**Final eval:**
- Full Tier-1 + Tier-2 on test set + all OOD slices
- Latency on M1 wall-clock

### Uncertainty flags

1. Drag success criterion doesn't assess whether trajectory passed through sensible intermediate points. Deferred to Tier-2 sub-metrics.
2. Chord / press-and-hold 3-frame tolerance (~100ms @ 30Hz) is a guess. Validate empirically.
3. Scroll-to-target "visible in viewport" needs operational definition in pygame (implementation decision).
4. Type exact-match may be too strict ("Hello world" vs "hello world" fails). Confidence 6/10 on exact-match — could soften to edit-distance threshold if over-strict in practice.

### Summary

| Component | Decision | Confidence |
|---|---|---|
| Tier-1 success criteria | Binary bbox / text-match per primitive | 8/10 |
| Tier-2 diagnostics | Continuous sub-metrics, always logged | 7/10 |
| Best-checkpoint metric | Unweighted mean primitive success rate | 7/10 |
| Bbox tolerance | 0px (bbox-inclusive) for synthetic data | 8/10 |
| Saturated-primitive handling | Switch to Tier-2 metric | 6/10 |
| OOD reporting | Per-slice + OOD-to-InD ratio | 8/10 |
| Efficiency metric | Not v1 | 7/10 |
| Smoothness metric | Not primary; Tier-2 only | 7/10 |

**Q25 — Eval set composition. [DECIDED]**

### Train / val / test split (episode-level, stratified by primitive type)

80/10/10 split as decided in Q8, refined here with explicit per-primitive counts.

| Primitive | Train | Val | Test |
|---|---|---|---|
| Hover | 1,200 | 150 | 150 |
| Hover-wait | 800 | 100 | 100 |
| Click-rel | 1,200 | 150 | 150 |
| L-click | 2,400 | 300 | 300 |
| R-click | 1,200 | 150 | 150 |
| Double-click | 1,600 | 200 | 200 |
| Click-hold | 1,200 | 150 | 150 |
| Drag | 3,200 | 400 | 400 |
| Scroll | 1,600 | 200 | 200 |
| Type-short | 1,200 | 150 | 150 |
| Type-med | 1,200 | 150 | 150 |
| **Type-long** | **400** | **50** | **50** |
| Chord | 1,600 | 200 | 200 |
| Press-hold | 800 | 100 | 100 |
| **Totals** | **~19,600** | **~2,450** | **~2,450** |

**Frame count rough estimate: ~1.3M frames** (within Q8's 1M–1.5M target).

**Type-long kept at 400/50/50 despite ±14% CI at n=50 val/test.** Rationale: Type-long episodes are ~5× longer per-episode (240 frames vs 45 for Type-short), so even 400 Type-long episodes represent a meaningful share of total frames. Doubling Type-long would push it to ~40% of the corpus, disproportionate to its role. Accept wider CI on Type-long success rate; sufficient for detecting gross failures.

**Confidence: 7/10** on this sizing (acknowledging Type-long CI caveat).

### Six OOD slices

Each slice is a separately-generated set of episodes using held-out attributes along one axis. Not part of the 80/10/10 split — generated additionally for OOD measurement.

**Slice 1: Unseen themes**
- Hold out 2–3 themes during training (e.g., skeuomorphic, retro-Win95, neobrutalist)
- Test primitives against scenes rendered in held-out themes
- Measures visual distribution-shift robustness
- **Size: 500 episodes**, stratified across primitive types

**Slice 2: Unseen instruction phrasings**
- Hold out instruction templates during training (e.g., certain verb families or politeness patterns); test uses held-out templates with in-distribution scenes
- Measures instruction-form robustness
- **Size: 500 episodes**

**Slice 3: Unseen target colors**
- Train on constrained color palette; test on held-out colors (e.g., train mostly blue/green, test red/purple/yellow)
- Measures color-independence of grounding
- **Size: 300 episodes**

**Slice 4: Unseen target sizes**
- Hold out extreme sizes during training (tiny <20px or huge >200px)
- Measures scale robustness
- **Size: 300 episodes**

**Slice 5: Unseen target positions**
- Hold out specific screen regions during training (e.g., top-right quadrant)
- Measures position-invariance
- **Size: 300 episodes**

**Slice 6: Higher distractor density**
- Train with 3–8 distractors per scene; test with 15–25
- Measures robustness to visual clutter
- **Size: 300 episodes**

**Total OOD: 2,200 episodes.** Additional cost beyond main 25K.

**Confidence: 7/10** that these 6 slices capture the right axes. COLOSSEUM (robotics) uses 14 perturbation axes; we have 6, narrower but sufficient for v1.

### Rejected / deferred slices

- **Unseen primitive × theme combinations** (compositional generalization) — hard to construct cleanly; compositional generalization often fails in imitation learning and may make v1 look worse than it is. Post-v1. (Confidence: 6/10 on skipping.)
- **Multi-language typing** (non-ASCII, non-English) — out of scope for v1 per Q12 (English-only). Post-v1.
- **Unseen cursor sprites** — Q11 already randomizes; separate OOD slice would just be a tighter cursor randomization test. Covered by Q11 ablation.

### Three-tier evaluation cadence

**Tier A — Offline val (every 500 steps during training):**
- Full val set: ~2,450 episodes
- Per-step prediction vs ground-truth action
- Produces: val loss (ES metric from Q21) + per-primitive offline accuracy (best-checkpoint signal)
- Cost: few minutes per round × 15 rounds ≈ 1 hour cumulative

**Tier B — Closed-loop val-subsample (every 2,000 steps during training):**
- ~30 episodes × 14 primitive sub-types = ~400 rollouts
- Model drives pygame env; actions affect next observation
- Produces: closed-loop primitive success rate (verifies offline metric matches real rollout performance)
- Cost: ~15–20 min/round × 4 rounds ≈ 1 hour cumulative
- Purpose: catch offline-closed-loop gaps (exposure bias signal)

**Tier C — Final eval (one-time at training end):**
- Full closed-loop on val + test + all 6 OOD slices
- Total: ~7,100 rollouts (~2,450 val + 2,450 test + 2,200 OOD)
- Cost: ~4 hours one-time

**In-training eval overhead: ~2 hours cumulative** on top of training compute. Acceptable for a few-hour L40S training run.

**Confidence: 8/10.**

### ES and best-checkpoint logic (refined from Q21)

With this cadence:

- **ES metric:** val loss from offline eval (every 500 steps). Smooth signal, frequent enough for patience-5.
- **Best-checkpoint metric:** offline per-primitive accuracy as primary signal. Verified every 2,000 steps against closed-loop primitive success.
- **If offline-best diverges from closed-loop-best**, trust closed-loop; investigate why (usually exposure bias). Flag for v2 targeted fix.

**Confidence: 8/10.**

### Eval determinism protocol

**Deterministic rollouts:**
- Argmax actions at eval time (from Q1 decision on discrete actions)
- No sampling/stochasticity at action level
- Environment seeds fixed per eval set

**Seeds:**
- Val seed: fixed across all training runs (same val episodes every eval)
- Test seed: fixed but *different* from val; not touched during training
- OOD seeds: fixed per slice; not touched during training

**Episode caching:**
- Pre-generate val and test episodes once, store to disk, replay identically on every eval run
- OOD episodes generated once at start of experiment
- **Confidence: 9/10** on pre-generation approach

### Test discipline

Test set evaluated **once** at end of training (or small number of times for final experiment report). Val is for decisions; test is for reporting.

**Rule:** if we make architecture/HP changes based on test-set performance, we've contaminated the test. Revert to reporting Tier A val numbers only in that case.

**Confidence: 10/10.**

### Statistical inference caveats

Rollout count per primitive per slice limits CI resolution:

| Scenario | Typical n | CI at p=0.5 |
|---|---|---|
| Full val/test per primitive | 50–400 | ±5–14% |
| OOD slice total | 300–500 | ±5–6% pooled |
| OOD slice per primitive (14 types) | 20–40 | ±14–15% |
| Type-long val/test | 50 | ±14% |

**Implications:**
- Per-primitive OOD stats are under-powered for fine-grained claims; sufficient for detecting gross failures (e.g., 90% → 60%)
- Pooled-across-primitives OOD metrics have better CI
- Type-long is the worst under-powered in-distribution case; accept the caveat

**Confidence: 7/10** that this resolution is acceptable for v1. Generate more OOD data post-v1 if specific claims need tighter CI.

### Summary

| Component | Value | Confidence |
|---|---|---|
| Train/val/test split | 80/10/10 stratified (~19.6K/2.45K/2.45K) | 8/10 |
| Type-long sizing | 400/50/50 (kept; long episodes balance proportion) | 7/10 |
| OOD slices | 6 slices, 2,200 total episodes | 7/10 |
| Compositional / multi-language slices | Deferred post-v1 | 6/10 |
| Training-time offline val | Every 500 steps, full val | 8/10 |
| Training-time closed-loop val | Every 2,000 steps, ~400 rollouts subsample | 8/10 |
| Final eval | One-time closed-loop on val+test+OOD | 8/10 |
| Determinism | Argmax actions + fixed env seeds | 8/10 |
| Episode caching | Pre-generate + replay | 9/10 |
| Test discipline | Evaluated once, val drives decisions | 10/10 |

### Uncertainty flags

1. **OOD per-primitive stats under-powered** at ~20–40 episodes per primitive per slice. Accept for v1.
2. **6 OOD slices may miss important axes** (compositional, multi-language, unseen rhythms). Revisit post-v1.
3. **OOD generation cost** (~2,200 additional episodes) adds to data-gen timeline; explicitly budgeted.
4. **Closed-loop rollout time budget** assumes ~2s per rollout at 30Hz logical. Validate empirically during implementation; scale Tier-B cadence down if too slow.

**Q26 — Baselines. [DECIDED]**

### Decision: no baselines in v1

**Rationale:**
- v1 validates the core architecture. Absolute success rates on our own eval sets are the meaningful signal, not relative comparison to other systems.
- Action-space mismatch with existing GUI agents makes direct comparison honestly incomplete — UI-TARS, ShowUI, Operator emit high-level `click(x,y)`/`drag(A,B)` at 0.2–2 Hz; we emit per-frame factored deltas at 30Hz. Primitives like Hover-and-wait, Chord, Press-and-hold have no equivalent in those agents' action spaces.
- Setting up comparable eval (inference infrastructure, prompt formatting, output-to-metric mapping) is meaningful engineering cost for uncertain value.
- Our own diagnostics (Q24 Tier-1/Tier-2 metrics) are rich enough to assess whether the model is learning what it should without external reference points.

**Confidence: 7/10** that skipping baselines for v1 is the right call.

### Deferred / future consideration

Post-v1 (if needed for publication, ablation against field, or deployment-readiness assessment):

- **ShowUI-2B locally** — closest open-source architectural reference at similar scale. Could be run locally for direct grounding-primitive comparison (click, drag endpoints). Practical to set up on our hardware.
- **UI-TARS-1.5-7B via HuggingFace Inference Endpoint** — open-source SOTA GUI agent. HF endpoint avoids local 7B inference overhead. Good for SOTA context when we want to position against the field.

Both limited by action-space mismatch; comparison would need to restrict to primitives where action spaces translate (single-click, hover, drag endpoints).

**Rejected alternatives:**
- **Random / idle policies** — trivial sanity baselines. Our absolute success rates on Q24 metrics already tell us if the model is doing meaningful work; sanity baselines add little.
- **Rule-based oracle** — per-primitive scripted policy with ground-truth access. Useful as upper bound but requires per-primitive scripting effort that may not faithfully represent achievable performance.
- **Claude Computer Use / GPT-4V** — API cost prohibitive for sweep evaluation.

### What replaces baselines as the sanity check

- **Q24 Tier-1 success rates** on in-distribution val/test sets — direct measure of "is the model doing the task"
- **Q24 OOD-to-InD ratio** — measures robustness beyond rote memorization
- **Q27 ablations** — stripped versions of our own model (no proprio, no action history, no LoRA, etc.) provide the most informative "baselines" by isolating which components matter
- **Chance-level reference in reporting** — for each primitive, report the random-click probability (e.g., for L-click with one 40×40 target on 720×450 screen, chance ≈ 0.5%) as a text-only reference, no rollout needed

### Uncertainty flags

- Skipping baselines means v1 results stand alone without field context. If v1 succeeds and we want to publish or justify scaling, we'll need at minimum a ShowUI-2B comparison.
- The "action-space mismatch" argument is honest but could be used to rationalize skipping any external comparison; worth revisiting post-v1 whether restricted-primitive comparison is actually impractical or just inconvenient.

**Q27 — Ablation plan. [DECIDED]**

### Decision: no dedicated ablation runs in v1

**Rationale:**
- v1's core claim is "the architecture learns primitives to meaningful success rates" — validatable from absolute Q24 Tier-1/Tier-2 metrics and OOD slice performance (Q25) on a single training run.
- No individual ablation listed below is load-bearing for establishing the v1 claim.
- 22 accumulated ablation candidates × compute cost of main experiment = 22× compute budget for questions that are follow-up ("what contributed?") not foundational.
- Richer approach: let v1 results *drive* which ablations matter. If Drag underperforms, ablate action history + trunk depth. If OOD slices gap wide, ablate augmentation. Data-driven ablation plan > speculative sweep.

**Confidence: 8/10** that skipping dedicated ablation runs for v1 is correct.

### Inference-time diagnostic probes (cheap substitutes, included in v1 eval)

Three free probes run at final eval time — no separate training required:

**Probe 1: Zero-out action history at inference.** During closed-loop eval on test set, run half the episodes with real action history tokens and half with all-zeros history tokens. Compare per-primitive success rates. Tests whether model *relies on* action history.

**Probe 2: Zero-out proprio at inference.** Same idea on proprio token. Tests whether model *relies on* state input.

**Probe 3: Zero-out instruction at inference (added 2026-04-23).** Same pattern on the cached instruction tokens. Measures how much the model actually conditions on the instruction vs exploits scene-level correlations (e.g., "there's only one red thing, click it regardless of what was asked"). Particularly informative for scenes with natural attribute-target alignments; failure to use instruction means Q10's adversarial-scene coverage may be insufficient.

All three tell us about reliance, not about counterfactual performance if trained without these inputs. Enough diagnostic value for v1.

**Confidence: 8/10** these probes are worth including.

### Full ablation backlog (future experiments, prioritized)

**P1 — highest expected information gain:**

| Ablation | Question | Est. compute |
|---|---|---|
| Trunk A (3 blocks, 32M) vs Trunk B (6 blocks, 60M) | Is our trunk capacity right? (Q16 low-medium confidence) | 1× |
| Action history K sweep (0 / 4 / 8) | Is history meaningfully useful? How much? | 2× |
| Delta vs toggle keyboard | Was foundational keyboard-repr choice right? | 1× |
| OOD slice deep-dives | Which generalization axes are weakest? | 0× (already in v1 eval) |

**P2 — medium value:**

| Ablation | Question | Est. compute |
|---|---|---|
| Proprio-as-token vs FiLM vs none | Was state-integration choice right? | 2× |
| Query count K (4 / 8 / 16 / 32) | Is K=16 well-calibrated? | 3× |
| LoRA rank (frozen / 8 / 16) | Does vision-tower adaptation help? | 2× |
| Dropout 0 / 0.05 / 0.1 | Only if v1 shows overfitting signatures | 2× |
| Weight EMA on vs off | Does EMA help? | 0× (can compare checkpoints within v1) |
| `max_num_patches` sweep (64 / 128 / 256) | Grounding-quality vs latency tradeoff | 2× |
| Head format (unpooled / mean-pool / attention-pool) | Does pooling hurt? (Q16 worried it might) | 2× |

**P3 — lower priority:**

| Ablation | Question | Est. compute |
|---|---|---|
| Optimizer (AdamW vs Muon) | Potential compute-efficiency win at our scale | 1× |
| LR sweeps per group | Are our LR defaults well-tuned? | 2–3× |
| Schedule (cosine vs constant) | Does schedule matter at our duration? | 1× |
| Scheduled sampling | Only if closed-loop eval shows exposure bias gap | 1× |
| Visual augmentation | Only if OOD slice underperforms | 1× |
| Tempo variability fixed vs varied | Is tempo diversity helping? | 1× |
| Visual diversity low/med/high | How much diversity is enough? | 2× |
| Cursor-sprite OOD test | Is model learning "cursor-as-concept"? | 0× (can add as extra OOD slice in v1) |
| Focal γ sweep (1.0 / 1.5 / 2.0) | Is γ well-tuned? | 2× |
| Per-class weighting on/off | Does key-frequency weighting help? | 1× |
| Done-head aux loss on/off | Does done-signal regularize trunk features? | 1× |
| Encoder alternatives | Only if SigLIP2-B underperforms | 1× |

### Execution principle

v1 results will reveal which P1/P2/P3 questions actually matter. Don't speculate on all 22; run the 2–5 most informative ones based on v1 diagnostic signal.

If v1 is broadly successful, Trunk A vs B and action history K ablations become the obvious next experiment.

If v1 shows specific failure modes, the ablation priorities shift accordingly (e.g., overfitting → dropout; OOD gap → augmentation; exposure bias → scheduled sampling).

### Two "free" additions to v1 eval that capture cheap ablation value

Beyond the inference-time probes above, two low-cost additions to v1 eval protocol:

1. **Add cursor-sprite-OOD as a 7th OOD slice** in Q25's eval set. Test-time uses a novel cursor sprite unseen in training. Measures cursor-as-concept learning. Small data-gen cost, no training cost.

2. **Record both EMA and non-EMA checkpoints.** Compare final metrics on test set. Zero extra compute — just evaluate two checkpoints instead of one.

**Confidence: 9/10** these cheap additions are worth including.

### Summary

| Component | Decision | Confidence |
|---|---|---|
| Dedicated ablation runs in v1 | None | 8/10 |
| Inference-time probe: zero action history | Include | 8/10 |
| Inference-time probe: zero proprio | Include | 8/10 |
| Inference-time probe: zero instruction (2026-04-23) | Include | 8/10 |
| Cursor-sprite OOD as 7th slice | Include | 9/10 |
| EMA vs non-EMA checkpoint comparison | Include (zero cost) | 9/10 |
| Full ablation backlog | Documented, prioritized, driven by v1 results | 8/10 |

**Q28 — Eval infrastructure & error taxonomy. [DECIDED]**

### What Q28 adds beyond prior Qs

Most original Q28 sub-topics were absorbed:
- Online vs offline eval → Q25 (offline during training, closed-loop at end)
- Rollout counts for statistical significance → Q25 (binomial CI analysis)
- Grounding/timing/keyboard failure diagnostics → Q24 Tier-2

Q28 pins down what remained: pygame eval env operational spec, cross-cutting error taxonomy, diagnostic visualizations, Q4 residual flags, inference-time ablation probes.

### Pygame eval environment operational spec

**Per-episode instantiation:**
- Fresh pygame scene per episode (no state leakage between rollouts)
- Seed fixed from (episode_id, slice_name) for reproducibility
- Render output + ground-truth state captured in lockstep

**Non-termination handling:**
- Each primitive has fixed window (30–240 frames per Q8)
- If success condition met mid-window: log success, continue to end (captures post-success errors like unnecessary clicks)
- If success condition not met at end-of-window: log failure

**Success detection:**
- Check success condition after each frame
- Log first-success frame and final-frame success separately (captures overshoot)

**Parallel rollout execution:**
- Pygame is CPU-only, multi-process clean
- N worker processes = min(available CPU cores, total episodes / 10)
- Typical: 8–16 workers on L40S host

**Confidence: 8/10.**

### Cross-cutting error taxonomy

Aggregates Q24 per-primitive Tier-2 into 6 failure classes spanning primitives:

| Error class | Definition | Primitives affected |
|---|---|---|
| **Grounding failure** | Cursor never entered target bbox, or entered wrong bbox | Hover, Click, Drag, Scroll |
| **Motor failure** | Cursor overshoots / jitters / wrong direction despite correct intent | Hover, Drag, Scroll |
| **Timing failure** | Action fires at wrong time (premature release, missed double-click window, wrong chord sync) | Double, Click-hold, Drag, Chord, Press-hold |
| **Keyboard failure** | Wrong keys, wrong modifier state | Type, Chord, Press-hold |
| **Completion failure** | Success briefly achieved then corrupted | Hover-wait, long primitives |
| **No-action failure** | Idle throughout episode despite instruction | All |

Per eval run: % of failures in each class (sums to 100% of failures). Helps identify systemic weaknesses vs per-primitive weaknesses.

**Confidence: 7/10** on this taxonomy being reasonably complete; may overlap in edge cases (motor-vs-timing ambiguity for late clicks).

### Diagnostic visualizations

**Always generated per eval run:**
- Per-primitive success-rate dashboard: single plot, all primitives, Tier-1 success with OOD overlay
- Failure-class distribution: bar chart per primitive showing which error classes dominate
- Cursor trajectory overlays: 20 random failed episodes per primitive, predicted trajectory overlaid on pygame scene with GT target bbox

**On-demand (investigation mode):**
- Rollout replay videos: MP4 with cursor + action annotations
- Cross-attention heatmaps: visualize what query tokens attended to (vision / text / history / proprio)
- Per-frame action prediction plots: argmax predictions vs GT over time, per head

**Confidence: 8/10.**

### Absorbed Q4 residual flags

Added as additional Tier-2 sub-metrics for all primitives (extending Q24):

- **Screen-bound violations:** frames where cursor predicted to move outside screen edges. Diagnostic, not a failure per se — indicates broken motor control
- **Click-during-flick rate:** clicks emitted during high-velocity cursor motion (above threshold). Indicates timing mismatch between cursor-movement head and click head
- **Trajectory smoothness:** already in Q24 as "mean curvature" for drag/hover-wait; no change

**Confidence: 7/10** on thresholds (need empirical calibration in v1).

### Inference-time diagnostic probes (from Q27)

Cheap substitutes for dedicated ablation training runs:

**Action-history ablation probe:**
- On 20% of each primitive's val set, replace action-history tokens with zeros
- Measure success-rate delta vs matched episodes with real history
- Answers: "does the model actually use action history?"

**Proprio ablation probe:**
- Same pattern but zero-out proprio token
- Particularly informative for Chord / Press-hold (theoretically impossible without held-key state)
- Answers: "does the model actually use proprio state?"
- Note: cursor remains rendered on the visual frame (it's part of the screen); only the explicit proprio `(x, y)` and held-key bits are zeroed. So this probe tests reliance on explicit state input, not cursor-concept learning. For the latter, use the Q11 cursor-sprite OOD test.

**Instruction ablation probe (added 2026-04-23):**
- Same pattern but zero-out cached instruction tokens
- Tests whether the model actually conditions on the instruction or exploits scene-level correlations
- Particularly informative for scenes where target attributes are naturally unique (and instruction is therefore theoretically redundant). A model that succeeds here without the instruction is pattern-matching on the scene, not following instructions.

**Run cadence:** once per major eval milestone (end of training), not every 500 steps.

**Confidence: 8/10.**

### Not covered (deferred)

- **Automated LLM-judge scoring for rollouts** — our success criteria are deterministic; no judge needed
- **Human-rater eval** — deferred per Q24
- **Real-screen eval** — deferred per Q29

### Summary

| Component | Decision | Confidence |
|---|---|---|
| Pygame env instantiation | Fresh per episode, seeded reproducibly | 8/10 |
| Parallel rollouts | 8–16 worker processes | 8/10 |
| Non-termination handling | Run to full window, log success at first hit | 8/10 |
| Error taxonomy | 6 cross-cutting failure classes | 7/10 |
| Default visualizations | Dashboard + failure distribution + trajectory overlays | 8/10 |
| On-demand visualizations | Replay videos, attention heatmaps, action plots | 8/10 |
| Screen-bound + click-during-flick | Added as Tier-2 sub-metrics | 7/10 |
| Inference-time probes | Action-history zero-out, proprio zero-out | 8/10 |

### Uncertainty flags

1. Error taxonomy may need refinement if v1 reveals overlap (motor vs timing for late clicks)
2. Cross-attention heatmap visualization requires model surgery (small cost, not zero)
3. Multi-process pygame rollouts need empirical validation — pygame can be weird in multi-process contexts

### E. Inference on M1 (4 — unlocks 30 Hz aspiration)

**Q29 — Deployment stack. [DECIDED]**

### Decision: PyTorch fp16 + MPS for v1. No porting.

Post-Q16 empirical measurement: current PyTorch fp16+MPS stack delivers ~7.6Hz wall-clock for full pipeline on M1 (SigLIP2-B 80–92ms + Trunk A 40ms = ~132ms). 30Hz is *not* achievable on this stack and likely requires significant porting effort that is out of scope for v1.

**Rationale:**
- Same stack as training — no conversion needed, no validation burden
- Wall-clock 30Hz requires MLX or CoreML port, each with meaningful engineering cost (see feasibility below)
- Logical 30Hz achievable for training and closed-loop eval via pygame slowdown (env paused during inference)
- Accept ~7–15Hz wall-clock for v1 deployment measurement

**Confidence: 8/10** that this is the right call for v1 scope.

### Feasibility check on porting options (why we're not doing them in v1)

**MLX port of SigLIP2-naflex:**
- `mlx-transformers` library exists with HuggingFace-style interface, but SigLIP/SigLIP2 is not in their current model set (BertModel, Qwen3-VL, and others are)
- Would require reimplementing SigLIP2-naflex in MLX. The naflex variant specifically handles variable-patch tokens with position interpolation — non-standard, not a trivial port
- Realistic estimate: 1–2 weeks of focused engineering effort for someone familiar with MLX, plus ongoing maintenance
- Confidence MLX port is feasible if prioritized: 7/10. Confidence it's quick: 3/10

**CoreML → ANE path:**
- Apple's guidance for ANE requires fp16 weights, specific tensor shapes (BCHW with sequence on last axis), static shapes preferred via EnumeratedShapes (up to 128 discrete shapes allowed)
- naflex's continuous variable-patch design is in direct conflict with ANE's static-shape preference
- Apple's reference ANE transformer implementation is only DistilBERT; other models require their own rewrite
- Realistic: significant model rewrite, unclear if naflex even works cleanly on ANE
- Confidence CoreML → ANE works without rewrite: 2/10
- Confidence CoreML → GPU backend (as alternative to MPS) gives meaningful speedup: 5/10

**INT8 on M1 (without MLX/CoreML):**
- PyTorch `torch.ao.quantization` targets CPU/CUDA — no MPS backend
- bitsandbytes: no MPS backend
- Realistic INT8 requires committing to MLX or CoreML first
- Confidence INT8 achievable without MLX/CoreML port: 1/10

### Single v1-feasible optimization flagged (not adopted)

**`max_num_patches=64` (from 256):** empirically tested, drops encoder to ~27ms, projected pipeline ~80ms = 12.5Hz. Flag change, not a port.

**v1 decision:** stick with `max_num_patches=256` for training + eval. The latency win isn't worth risking grounding quality degradation when we haven't validated the architecture yet. Queue as Q27-style future ablation: measure grounding-quality vs latency tradeoff post-v1.

**Confidence: 7/10** on keeping 256; 5/10 that dropping to 64 wouldn't meaningfully degrade grounding (depends on empirical).

### Future work (post-v1, prioritized)

If/when real-time 30Hz deployment becomes a binding constraint:

1. **`max_num_patches` sweep** — cheapest; flag change + retrain. Low hanging.
2. **MLX port of SigLIP2-naflex + trunk** — medium cost; needed if MPS is the binding bottleneck
3. **MLX INT8 quantization** — stackable on top of MLX port
4. **Encoder distillation** to a smaller UI-specific encoder — high cost, high upside
5. **CoreML/ANE exploration** — highest cost, uncertain feasibility due to naflex shape mismatch

None of these are v1 scope.

---

**Q30 — Quantization strategy. [DECIDED]**

**None for v1. bf16 during training on L40S, fp16 during inference on M1 MPS.**

**Rationale:**
- INT8 on M1 requires committing to MLX or CoreML port, which we've deferred in Q29
- PyTorch MPS has no native INT8 path (see Q29 feasibility)
- Accept fp16 latency floor for v1
- Revisit post-v1 if/when MLX port happens

**Confidence: 8/10.**

Training precision (bf16 on L40S) unchanged from Q20.

---

**Q31 — Hardware target on M1. [DECIDED]**

**GPU via Metal Performance Shaders (MPS). Not ANE.**

**Rationale:**
- MPS is natively supported by PyTorch — matches our training stack
- ANE requires CoreML conversion + Apple-specific transformer rewrites (see Q29 feasibility). naflex variable shapes make this particularly hard
- CPU would be dramatically slower than both — not considered
- MPS empirical measurement (Q16): 80–92ms for SigLIP2-B encoder, 40ms for trunk. Good enough to proceed

**Confidence: 8/10.**

Future work: if MLX port happens, MLX runs across GPU/ANE automatically depending on op compatibility. That's the natural path to ANE acceleration without explicit CoreML conversion.

---

**Q32 — End-to-end system latency breakdown. [DECIDED]**

Empirically measured on M1 MacBook Pro with PyTorch fp16 MPS:

| Stage | Measured latency | Notes |
|---|---|---|
| Screen capture (pygame surface readback) | 1–2ms | Trivial in pygame; real-screen will add ~5–15ms |
| naflex preprocessing (resize + patchification) | 2–3ms | |
| SigLIP2-B vision tower @ max_patches=256 | 80–92ms | **Dominant cost** |
| Trunk A (3 blocks, 32M params) | 40ms | Second-largest cost |
| 6 unpooled heads | <1ms | Negligible |
| Action execution (pygame apply) | 1ms | Trivial in sim; real-screen adds ~2–5ms |
| **Total per frame (pygame sim)** | **~130–140ms → ~7.6 Hz** | |

**Identified bottleneck:** vision encoder dominates at ~60% of total latency. Trunk second at ~30%. Other stages are negligible.

**v1 deployment latency estimate for real MacBook (not pygame):** add ~10–20ms for real screen capture, event dispatch, and system overhead → **~150ms/frame, ~6–7Hz real-screen wall-clock.**

### Optimization priority ordering (for post-v1)

Ranked by expected improvement per effort:

1. **`max_num_patches` reduction (flag change)** — 65% encoder reduction if 64 vs 256 works for grounding. ~3 days engineering.
2. **MLX port of encoder** — 2–3× MPS speedup empirically claimed. 1–2 weeks engineering.
3. **MLX INT8 quantization** — additional 2–4× on quantized ops. Stackable on #2.
4. **Encoder distillation to smaller model** — limited by distillation training time (weeks) but potentially best end-state.
5. **CoreML/ANE export** — highest uncertainty; naflex variable shapes likely require substantial rewrite.

### Uncertainty flags

1. **Real-screen overhead not measured.** Pygame numbers underestimate production MacBook latency by ~10–20ms.
2. **Thermal throttling not tested** at continuous 7–15Hz for long sessions — could degrade over time.
3. **Naflex grounding quality at max_patches=64 unknown** — need empirical measurement before committing to optimization path #1.
4. **MLX port feasibility for SigLIP2-naflex specifically unproven** — estimates assume similar-complexity ViT ports; naflex's variable-patch logic may complicate.

**Confidence: 9/10** on empirical pygame latency breakdown (actually measured).
**Confidence: 6/10** on real-screen extrapolation.
**Confidence: 5/10** on post-v1 optimization ranking (depends on engineering team's MLX familiarity).

### F. Methodology & experimental practice (3 — lower priority)

**Q33 — Success criteria. [DECIDED]**

### "Promising signs" gate, not a rigid threshold

v1 is exploratory validation, not production acceptance. Success = evidence that the architecture works and warrants continued investment, not a specific numerical target.

### Heuristics for go/no-go decision after v1

**Strong signals (any one → continue):**
- ≥3 primitives at >60% Tier-1 success rate in-distribution
- Any primitive showing clean scaling curve (train loss down, val loss down, val success up monotonically)
- OOD-to-InD ratio ≥0.70 on any slice for any primitive (shows generalization)
- Simple primitives (Hover, L-click) at >80% while complex primitives still learning (normal convergence profile)

**Weak signals (combination suggests continue):**
- Multiple primitives at 30–60% in-distribution — working but undercooked
- Train loss decreasing steadily without val loss explosion
- Failure classes concentrated (e.g., all timing failures) — suggests fixable issues

**Kill signals (reconsider or pivot):**
- All primitives near chance-level after 20 epochs → architecture doesn't work
- Train loss plateaus high → capacity or optimization problem
- Val loss diverges catastrophically → overfitting we can't fix with current stack
- Inference latency > 300ms/frame making eval impractical

**Decision format:** explicit go/no-go assessment after v1 eval, articulating which signals fired. Forcing function for honest evaluation.

**Confidence: 7/10** this captures the right principle — rigorous enough to prevent wishful thinking, flexible enough for exploratory work.

**Uncertainty flag:** "promising signs" is subjective. Heuristics help but don't eliminate judgment call.

---

**Q34 — Hyperparameter search strategy. [DECIDED]**

### Decision: no HP search for v1. Single run with chosen defaults.

v1 uses our best-guess defaults across all HPs. HP search is future work only if v1 results warrant it.

### Compute cost context (why we're not searching)

Minimal informative sweep:
- Trunk LR × LoRA LR (3×3 = 9 combinations)
- At ~4 hours per L40S run → ~36 L40S-hours for 2D LR sweep alone

Broader sweep:
- LR × focal γ × dropout × augmentation (3×3×2×2 = 36 combinations)
- ~144 L40S-hours

Bayesian optimization adaptive sampling: cuts 30–50% but still 70–100 L40S-hours.

**Realistic full HP search cost: 5–50× v1 alone.** Not worth it before validating the architecture.

### v1 defaults (consolidated from prior Qs)

| HP | v1 Default | Source | Confidence |
|---|---|---|---|
| Trunk LR | 3e-4 | Q20 | 7/10 |
| LoRA LR | 2e-4 | Q20 | 7/10 |
| Weight decay | 0.01 | Q20 | 8/10 |
| Betas | (0.9, 0.95) | Q20 | 7/10 |
| Warmup | 500 steps | Q20 | 8/10 |
| Schedule | cosine to 10% | Q20 | 8/10 |
| Gradient clip | 1.0 | Q20 | 9/10 |
| Batch size | 64 episodes | Q21 | 7/10 |
| Epochs | 20 | Q21 | 6/10 |
| Focal γ | 2.0 | Q2 | 6/10 |
| Label smoothing | 0.05 | Q2 | 6/10 |
| LoRA rank | 8 | Q15 | 7/10 |
| Query count K | 16 | Q15 | 6/10 |
| Action history K | 8 | Q5 | 7/10 |
| Dropout | 0 | Q22 | 5/10 |
| Weight EMA | 0.9999 | Q22 | 7/10 |
| `max_num_patches` | 256 | Q6/Q29 | 7/10 |

**Confidence: 8/10** on single-run approach for v1.

### Post-v1 HP exploration priority (if warranted)

1. Trunk LR + LoRA LR joint — most likely to move the needle
2. Focal γ + label smoothing — head-loss tuning
3. Dropout + weight decay — only if overfitting observed
4. Batch size × LR — if throughput becomes constraint
5. Duration × schedule — if saturation behavior unclear

---

**Q35 — Reproducibility practices. [DECIDED]**

### Infrastructure stack

| Purpose | Tool |
|---|---|
| Dataset hosting | HuggingFace Datasets |
| Model checkpoint hosting | HuggingFace Model Hub |
| Training compute | HuggingFace Jobs (v1 sole platform) |
| Training compute — future fallback | AWS SageMaker — deferred post-v1 per 2026-04-23 simplification; added only if/when HF credits exhaust |
| Training diagnostics | Weights & Biases |
| Code repository | GitHub |

**Confidence: 8/10** — standard open-source ML infra, good for reproducibility and sharing.

### Seed management

- Global seed fixed per training run, logged to W&B
- Data-generation seed fixed and logged (1.3M training transitions reproducibly generated)
- Eval seeds fixed per slice (val, test, per-OOD-slice) from Q25
- NumPy, PyTorch, Python `random`, and pygame all seeded at init
- **Not enforcing bit-exact determinism** (`torch.use_deterministic_algorithms(True)` adds ~10–30% overhead). Accept run-to-run variance from non-deterministic CUDA ops
- **Confidence: 9/10** on seed strategy; 7/10 on skipping full determinism

### Dependency pinning

- `pyproject.toml` with exact versions
- Pin: torch, transformers, datasets, peft (LoRA), wandb, pygame, numpy
- Pin Python version (3.11 or 3.12)
- Pin CUDA version for L40S

### Config versioning

- All HPs (Q34 table) in a single YAML config file
- Config committed to repo with each training run
- W&B run captures full config as metadata
- Git commit hash logged alongside W&B run

### Resume-ready checkpoint protocol (crash recovery on HF Jobs)

**Checkpoint contents (everything needed to resume without loss):**
- Model weights (trunk + LoRA + heads + EMA)
- Optimizer state (AdamW momentum, variance buffers)
- LR scheduler state (current step, current LR)
- RNG state (PyTorch, NumPy, Python `random`, pygame)
- Training state (current epoch, global step, best-val-metric seen)
- W&B run ID (for reattaching logs)
- Data loader state (position in epoch, worker seeds)

**Checkpoint frequency:**
- Every 500 steps (aligned with eval cadence from Q21)
- Keep: last 3 checkpoints + best-by-val checkpoint
- Upload to HF Hub automatically per checkpoint (async background to avoid blocking training)
- Retry queue for network failures

**Resume protocol (same-platform crash recovery):**
1. Download latest checkpoint from HF Hub
2. Restore all state
3. Continue training from last step
4. W&B reattaches to same run ID for continuous logging

**Smoke test (pre-v1):** 1000-step training run → save checkpoint → kill process → resume on fresh process; verify loss curves continue smoothly without visible discontinuity. Validates the resume protocol for HF-Jobs crash recovery. Cross-platform (HF Jobs → SageMaker) switch validation deferred post-v1 per Q35 simplification.

### Checkpoint format

- PyTorch `state_dict` + config JSON
- Separately loadable: trunk weights, LoRA adapters, head weights (enables partial loading)
- Both "best" (by val primitive success rate) and "final" (end of training)

### HF Jobs → SageMaker switch readiness (post-v1 only)

Cross-platform switching is out of scope for v1 per the 2026-04-23 simplification. When/if we need it (HF credits exhaust), the main risks are dependency parity (SageMaker deep-learning AMIs vs HF Jobs env) and data access (HF Datasets streaming should work from both). Checkpoints on HF Hub are already platform-portable (PyTorch state_dict), so the artifact side is fine. Address as its own small project when triggered.

### W&B logging cadence (from Q21)

- Every step: train/val loss (total + per-head), gradient norms per head, LR per group
- Every 500 steps: per-primitive success rate
- Every 2000 steps: cross-attention visualizations (if implemented)
- End of training: full eval tables as W&B artifacts

### Artifact registry

- Dataset uploaded as `{username}/cu-vla-primitives-v1`
- Model checkpoints uploaded per milestone
- Model card with training config, eval results, limitations
- README includes W&B run link for full diagnostics

### Minimum reproducibility checklist

1. Pin all dependencies (`pyproject.toml`)
2. Log full config per run (YAML + W&B)
3. Fix all seeds
4. Upload dataset + final model to HF Hub
5. W&B run link in README
6. **Test same-platform checkpoint-resume protocol before v1 full run** (crash recovery on HF Jobs; cross-platform switch deferred post-v1)

**Confidence: 9/10** that this is sufficient for v1.

### Uncertainty flags

1. HF Jobs compute limit unknown until we hit it — may force switch mid-v1
2. Full ~1.3M transitions upload to HF Hub may have size limits (likely fine, verify)
3. W&B free-tier project limits — verify if sharing publicly
4. "Promising signs" success criterion remains subjective; heuristics help but don't eliminate judgment call
5. Mid-training platform switch adds a failure mode — test resume protocol early, don't depend on it working first try

### G. Forward-looking (3 — defer unless blocking)

**Q36 — Primitive-to-System-2 interface. [DECIDED for v1 scope]**

**v1 interface: natural language instruction string only.** System 2 produces a string like "click the blue Submit button", System 1 executes. Matches Q12's training interface — no mismatch between training and deployment.

**Rejected for v1:** richer protocols (target bbox, completion condition) would require System 2 to do visual grounding, defeating the specialization of System 1 for grounding.

**Future work (post-v1):**
- Context passing: should System 2 also pass "what just happened" and "what we're trying to accomplish overall"? Opens questions about dual-encoder architecture.
- Completion-condition hints: System 2 might say "click Submit and wait for confirmation dialog" — would need action heads aware of completion criteria.
- Bbox hints: hybrid where System 2 narrows to a region, System 1 does precise grounding within it.

**Confidence: 7/10.** Language-only is clean v1 interface; richer protocols are future work.

**Q37 — Real-data bridge / IDM pseudo-labeling. [DECIDED: deferred]**

**v1 scope: synthetic pygame data only.** No real data.

**Future direction (post-v1):** IDM-style pseudo-labeling as the primary path to scale beyond synthetic.

**Mechanism (VPT-style):**
1. Train an Inverse Dynamics Model (IDM) on our synthetic data: (prev_frame, next_frame, held_key_state) → action
2. Apply IDM to unlabeled real screen recordings (YouTube-scale video) to generate pseudo-labels
3. Use pseudo-labels as additional training data for the main System 1 model

**Expected benefits:**
- Order-of-magnitude data scaling (YouTube screen recordings vastly exceed what we can synthesize)
- Real-distribution training without human demonstrations
- Natural coverage of real UI diversity (fonts, themes, apps, OSes)

**Challenges:**
- Real screens: higher resolution, variable frame rate, real cursor trails vs our synthetic sprites
- Distribution shift: IDM trained on synthetic may mis-label real videos
- Action-space alignment: real videos have no 30Hz frame alignment guarantee
- Cursor event detection: inferring click/drag from pixel changes is non-trivial

**v2 candidate.** Flagged in user memories as "highest-leverage data-side opportunity."

**Confidence: 7/10** on IDM being the right direction — well-validated by VPT on Minecraft; applicability to GUI-recording scale seems plausible but unproven.

**Q38 — Future encoder options. [DECIDED: SigLIP2-B for v1; roadmap flagged]**

**v1 scope: SigLIP2-B as per Q6.** No encoder experiments in v1.

**Future encoder roadmap (post-v1):**

1. **C-RADIOv4 as drop-in upgrade** — distilled from SigLIP2-g + DINOv3 + SAM3. Strictly better multi-task features. Low-risk swap if/when it matures and is well-supported.

2. **Encoder distillation to UI-specific smaller model** — distill SigLIP2-B → ~20–40M encoder trained specifically on UI screens. Best latency + UI-domain-specific features. Biggest effort but biggest upside.

3. **FDM-1-style masked-compression video encoder** — for scaling story. Treats screen video as compressible spatiotemporal signal. Dramatically reduces per-frame token cost. Requires our own large-scale pretraining.

4. **Custom FastViT-style encoder trained from scratch** — fastest inference, but loses SigLIP2's transfer benefits. Probably dominated by distillation path.

**Path forward:** if v1 architecture validates, consider encoder experiments when (a) latency becomes binding, or (b) grounding quality on real UI screens is insufficient.

**Confidence: 7/10** on this priority ordering — C-RADIOv4 swap is safest first step; distillation is highest-upside; FDM-1 is most speculative.

**Q39 — Embodiment generality roadmap. [DECIDED: v1 MacBook-specific; full roadmap below]** CU-VLA's eventual goal is supporting arbitrary computer embodiments. This experiment specializes to MacBook (77 Mac keys, trackpad only, US-QWERTY, 1440×900 16:10). Future work should consider:
- **Keyboard action space:** switch from 77 Mac-specific keys to ~104 HID scancodes for OS/layout generality. Each scancode → one 3-way delta head. Modest architectural cost (~30 more keys × 3 logits), significant generality win. Language-specific character mappings (AZERTY, QWERTZ) handled at data-generation level since scancodes are physical.
- **Mouse action space:** add middle-click if targeting mice rather than trackpad-only embodiments (currently excluded — standard MacBook has no middle click).
- **Aspect ratios:** SigLIP2 naflex already supports natively. Just needs training data at varied aspect ratios (16:9, 3:2, 4:3, portrait, ultrawide).
- **Resolutions:** train with source data at multiple resolutions ({1280×720, 1440×900, 1920×1080, 2560×1440, 4K}) so the model learns robustness to varied downscale ratios.
- **Multi-touch gestures:** new action modality (two-finger scroll, pinch, three-finger swipe). Architecturally separate from current heads.
- **OS-specific shortcuts:** Cmd vs Ctrl for standard shortcuts. Handled at System-2 planning level, not System-1 motor level.
- **Force Touch / pressure-sensitive input:** new action dimension, Mac-specific, defer.

### H. Operational (post-design addenda)

**Q40 — Epoch-1 live diagnostics. [DECIDED]**

**No separate smoke test.** Instead: rich in-band diagnostics during epoch 1 streamed to the HF Jobs dashboard for live monitoring. If something is broken, the diagnostics expose it and the run is killed manually.

**Streamed per-step:**
- Train loss (total + per-head — 7 lines)
- Grad norm (total + per-head — 7 lines)
- LR per param group (2 lines: LoRA, trunk)
- Per-head sparsity stats (% idle frames predicted for each action head) — catches idle-collapse early
- NaN/Inf detector (log step number + kill signal if detected)
- Throughput: steps/sec, samples/sec, wall-clock time
- VRAM usage on L40S

**Streamed at first val eval (~step 500, mid-epoch 1):**
- Per-primitive val success rate (11 rows)
- Per-head val loss breakdown (which primitives / heads learning vs stuck)
- Train-set eval loss on held-in subset (for train/val gap tracking)

**Epoch-1 pass criteria (what you're watching for):**
- Loss decreasing monotonically, not diverging, not stuck
- No NaN/Inf anywhere
- Grad norms stable; not hitting clip every step (the exp2 burn)
- No single head dominating total loss (prior keys-head-dominance failure mode)
- Throughput matches expected (~380 steps per epoch in reasonable wall-clock on L40S)
- By end of epoch 1 (step ~380), at least one primitive showing non-zero val success

**Action if any check fails:** kill the run manually from the HF Jobs console, investigate, fix, relaunch. Expected to catch pipeline bugs without a dedicated smoke-test step.

**Confidence: 8/10** — in-band diagnostics replace dedicated smoke test cleanly. Prior experience (exp2) confirms most pipeline bugs reveal themselves within ~50 steps of training.

---

**Q41 — Failure mode recovery. [DECIDED]**

**Dataset generation is a separate upfront effort.** Generated once, uploaded to HF Datasets, consumed by all training runs. No regeneration risk mid-training.

**Checkpointing to HF Hub:**
- Save `best` (by val primitive success rate) — used for final eval and downstream use
- Save `latest` (most recent step) — used for crash recovery
- Save cadence: every 500 steps (matches val eval cadence from Q21)
- HF Hub handles versioning; same filenames overwrite cleanly

**Mid-training crash recovery:**
1. HF Jobs reports job failure
2. Restart the HF job with the same config
3. Job detects `latest` checkpoint exists on HF Hub, loads: model state_dict + optimizer state + LR scheduler state + step counter
4. Training resumes from that step

**Not strictly preserved (acceptable losses):**
- W&B run continuity: if crash-restart, start a fresh W&B run rather than stitching. Diagnostic history is preserved per-run; we don't need a single continuous plot across a crash event.
- RNG state exactness: resumed run's batch order differs from hypothetical no-crash run. Minor effect on training dynamics; not meaningful.
- Sub-500-step progress: worst-case lose ~500 steps of training (<10% of epoch).

**If HF Jobs compute budget exhausts mid-run:** out of scope for v1 per Q35 (2026-04-23). If it happens, treat it as its own small project — HF-Hub checkpoints are platform-portable, so the artifact side is already fine; the work is SageMaker environment setup + dependency parity.

**Confidence: 9/10** — simple, pragmatic, matches available infrastructure.

---

**Q42 — Joint vs staged training. [DECIDED]**

**v1: joint training on all primitives from step 0.**

Stratified batching from Q2 ensures balanced exposure per primitive. All heads trained together.

**Alternative considered and rejected: staged curriculum training.**
- Stage 1: train on grounding-only primitives (Hover, Click-family) for N epochs
- Stage 2: add stateful primitives (Drag, Click-hold)
- Stage 3: add keyboard primitives (Type, Chord)
- Rationale for staging: might help if complex primitives overwhelm early optimization dynamics

**Rejected because:**
- Complicates implementation (multi-stage training, HP tuning per stage, checkpoint handoffs)
- Not obviously beneficial — modern large-model training is overwhelmingly joint/multi-task from step 0 (OpenVLA, π0, VPT all train jointly)
- Our stratified sampling already balances per-primitive exposure
- If joint training fails on a subset of primitives, staging is one candidate fix for v2 — but let v1 data decide

**Confidence: 7/10** that joint training is right for v1. If v1 reveals specific primitives collapse (e.g., Drag stuck at chance), staged curriculum is a v2 candidate.

---

**Q43 — Go/no-go review cadence. [DECIDED]**

**Single manual review after full run completion.** No intermediate go/no-go checkpoints during training.

**During training:**
- Epoch-1 live diagnostics (Q40) serve as the live check — if something looks broken, kill the run manually and investigate
- Otherwise let training run to completion (or early stopping from Q21)

**At run completion:**
- Final eval runs automatically (test set + all 6 OOD slices per Q25)
- Results logged to W&B dashboard + summary artifact uploaded to HF Hub
- Best and final checkpoints persisted

**Manual review (you, post-run):**
- Apply Q33 "promising signs" framework to final results
- Decision: continue CU-VLA development / pivot / redesign

**What this trades:** early-cancel compute savings for simplicity. If a run is going to fail, Q40 diagnostics catch it in epoch 1 (~30 min of wall-clock). If it passes epoch 1 but fails later, we've spent a few hours of L40S — acceptable cost for workflow simplicity.

**Confidence: 8/10** — matches your manual-review-after-full-run workflow preference.

---

## Amendments (Phase A findings, 2026-04-24)

Revisions to the questions above based on Spike A, B, C, E results. Phase B planning should treat these as authoritative overrides. Full derivations in [docs/experiments/6-action-primitives-phase-a-results/](6-action-primitives-phase-a-results/).

### Q7 — Pygame generator throughput (revised)

- **Was:** "pygame env generates at ≥200 eps/sec on consumer CPU."
- **Now:** **≥20 eps/s single-process on M1** (measured 3.61 eps/s at N=1000 in Spike E). Naive `multiprocessing.Pool.imap` at default `chunksize=1` is *slower* than serial (1.79 eps/s at workers=4 on M1) because of spawn+pygame init cost per worker and IPC payload size. The 200 eps/s target is only achievable with a **worker-writes-shards redesign** where workers own episode ranges and write their own parquet shards, eliminating the IPC round-trip.
- **Phase B implication:** schedule the multiproc redesign before Phase B data gen (24,500 eps × 2.5 eps/s serial = 2.7 h). With redesign, target 6-8× speedup on 8-core M1.

### Q8 — Per-primitive window length (revised)

- **Was:** `max_frames_lclick = 30`.
- **Now:** `max_frames_lclick = 45`. At 30, slow-tempo + long-distance episodes overflowed the window (~5% label-noise rate: expert never reached press, `done_gt=0` on all rows — model would learn "never predict done" from these). At 45, rate drops to ~0.25%.
- **Phase B implication:** size other primitives' windows by the same "slow-tempo + long-distance + settle + press/release" worst case, with ~15% headroom. Document per-primitive window lengths in Q9.

### Q15 — Trainable parameter budget (revised)

- **Was:** "~118M total params, ~30M trainable."
- **Now:** **408.8M total** (vision 93M frozen + text 282M frozen + trunk 28M + heads 4M + encoders 1M + LoRA 0.6M), **33.6M trainable**.
- **L4 fits** `micro_batch_episodes=4` with bf16 autocast (14.5 GB peak). `micro=8` OOMs (43 GB activations).
- **Phase B note:** the **text tower is 282M of frozen dead weight** for Phase A (no language variation in L-click). Phase B consideration: cache instruction embeddings offline and drop the text tower from the runtime model. Saves 1.1 GB GPU memory. Zero quality cost if the instruction set is enumerable at training time, which it is.

### Q2/Q8 — Micro batch sizing (new constraint)

- **New default:** `micro_batch_episodes=4` on L4 24 GB, `num_workers=4` for DataLoader prefetch, `macro_batch_episodes=64` (unchanged → 16 micros/step).
- Larger GPUs (L40S 48 GB, A100 80 GB) can run `micro=8` or `16`; adjust `num_workers` accordingly.
- The default in config.py (`micro_batch_episodes=8`) is now documented as "development-time default, override on L4 via `--micro-batch-episodes 4`".

### Q5/Q6 — Typing legibility at `max_num_patches=256` (confirmed)

- **Design stands.** Per-patch char identity probe gives 74% top-1 at 14pt on 62-way identity (46× chance, 46× better than random). SigLIP2 preserves per-patch char identity — the original string-presence probe was pooling-limited, not feature-limited.
- **Caveats for Phase B:**
  - Use the per-patch probe, not mean-pool or attention-pool presence, if re-measuring legibility for a new font/size regime.
  - Residual ~12-15% error even at 32pt is consistent with patch-boundary aliasing. If typing primitives underperform, first hypothesis is glyphs straddling patch edges; test by snapping rendered text to patch centers.

### Q40 — Epoch-1 diagnostics (add new signal)

- Existing list (loss descending, no NaN, grad norms stable, no head dominating) stands.
- **Add:** log `frac(dx_bin == 10)` and `frac(dy_bin == 10)` per step. Bin 10 is zero-motion. If the model outputs bin 10 on frames where the expert was moving >5 px/frame, it's collapsing to a "safe" zero-motion attractor. Spike B visual n=20 suggests this is the dominant "stuck far away" failure mode — confirm/refute with n=200 data.

### Q25 — Eval cadence on M1 (revised)

- Design assumed Spike C M1 timing ≈ 7.6 Hz per-frame effective rate.
- **Measured (extracted from Spike B closed-loop wall-clock):** ~180 ms/frame with `--decode expected`, ~370 ms/frame with `--decode argmax` on M1 MPS. That's **2.7-5.5 Hz** depending on decode mode — meaningfully below the 7.6 Hz target.
- **Phase B implication:** Tier-B eval cadence (400 rollouts every 2000 steps as originally scoped) is too expensive at the measured rate. Adjust to one of:
  - Drop cadence to every 4000 training steps (halves cost).
  - Trim rollout count to 200 (halves cost again).
  - Use probabilistic decode in Phase B eval (~2× faster than argmax — fewer max-frame timeouts).
- A dedicated Spike C measurement (`m1_eval_timing.py`) was implemented but not run separately; Spike B's wall-clock data is sufficient to revise this.

### Convention notes (not parameter changes, but tooling)

- **pygame → pygame-ce.** Project dep swapped because pygame 2.6.1 + Python 3.14 has a circular-import bug on `pygame.font`. pygame-ce 2.5.7 is a drop-in replacement.
- **`--num-workers` DataLoader prefetch + dataset-side SigLIP2 preprocessing.** Training loop hides JPEG decode + naflex processor + history-vector construction in DataLoader workers. Without this, GPU util oscillates 0-30 ↔ 100% on L4 because data prep is synchronous with the forward. Default on L4: `--num-workers 4`.
- **HF Jobs entry script workflow.** `scripts/hf_job_train_exp6.py` skips `pip install -e .` (UV env has no pip), clones an explicit branch via `CU_VLA_BRANCH` env var, and sets `HF_HUB_DISABLE_PROGRESS_BARS` + `HF_DATASETS_DISABLE_PROGRESS_BARS` to keep HF Jobs logs readable. See `docs/research/hf-jobs-gotchas.md` for the full debugging journey.