# Experiment 6 — Design Changelog

Delta log for `6-action-primitives.md`. Each entry records a design change made after the original 43-question close-out, with the decision, rationale, and sections touched in the main doc.

---

## 2026-04-23 — Pre-implementation brainstorm pass

Context: post-design review before writing the implementation plan. Goal was to surface ambiguities, correctness concerns, and scope risk before committing to generator + training infrastructure.

### Structural change — Phase A feasibility spikes added before Phase B (full v1)

**Decision:** split v1 into two phases. Phase A is a set of cheap feasibility spikes that de-risk load-bearing assumptions; Phase B is the full v1 as originally designed, contingent on per-spike review.

**Phase A spikes (review-and-decide per spike):**

| Spike | What it validates | Effort |
|---|---|---|
| **A — Typing legibility probe** | Can SigLIP2 naflex @ `max_num_patches=256` actually resolve 14pt+ rendered text? Linear-probe patch features on rendered character content, or cross-attention visualization. | 2–3 days |
| **B — L-click end-to-end** | Whole pipeline works: generator → data format → model → training loop → eval. 3K episodes, 3–5 epochs, target >50% success. | 5–7 days |
| **C — M1 closed-loop eval timing** | Real wall-clock for 100-episode rollouts on M1 fp16 MPS. Validates Tier-B eval budget. | 1 day (after B) |
| **E — Pygame generation throughput** | Measure eps/sec and per-episode storage across primitive types. Validates Q7's "≥200 eps/sec" claim. | 0.5 day |

**Pass criteria:** each spike produces a write-up; user reviews and explicitly approves proceeding. No rigid quantitative thresholds pre-committed — captures judgment that fixed thresholds miss.

**Main doc impact:** new "Phase A" section inserted after preamble; Status updated from "design complete" to "design v2, Phase A before Phase B."

---

### Q1 — Mouse delta cap widened from ±50px to ±100px

**Decision:** mouse `delta_x`/`delta_y` head ranges are now ±100px/frame (was ±50px). 10+1+10 exponential bins, α retained at ≈2.5 → finest bin ≈0.75px, coarsest ≈100px.

**Rationale:**
- At 30Hz, deliberate cross-screen motion is ~50px/frame; fast human flicks reach 100–167px/frame
- Q9 explicitly seeds "superhuman burst" tempo in training data; ±50px clipped that signal at the top bin
- Naflex 2.25× internal downscale already floors useful precision at ~1–2 source px, so the finer sub-px bins the ±50px cap preserved were effectively spurious
- ±100px gives flick headroom without meaningful fine-precision loss

**Main doc impact:** Q1 table and rationale updated; Q9 tempo language aligned; any other references to "±50px" or "50 source px" corrected.

---

### Q2 / Q8 — Batching strategy clarified: micro-batch grad accumulation

**Decision:** 64-episode macro-batch = 8 micro-batches × 8 episodes, each micro-batch homogeneous by primitive type. Gradients accumulate across the 8 micro-batches; optimizer steps once per macro-batch.

**Rationale:**
- Q8 literally specified "same-window-length within a batch, clean tensor shapes" (→ homogeneous micro-batches) but Q2 implied mixing primitive types within a batch. Micro-batch accumulation reconciles: clean shapes per forward pass, blended multi-task gradient per step.
- Single-primitive-per-batch (literal Q8) causes gradient magnitude swings as the trunk alternates between primitive types step-to-step.
- True mixed batches with padding + attention masks were Q8-rejected for cost; this achieves most of their benefit with much less plumbing.
- Aligned with π0 / OpenVLA-OFT multi-task training patterns.

**Main doc impact:** Q2 and Q8 both updated to reference micro-batch grad accumulation explicitly; remove the ambiguous "stratified within batch" language.

---

### Q5 — Action-chunking rejection reframed

**Decision:** no change to the outcome (single-frame output, no chunking). Only the *justification* is corrected.

**Rationale:** Q5 originally rejected chunking on "Q6 encoder choice resolves latency." Q16's empirical measurement (7.6Hz wall-clock on PyTorch MPS) shows Q6 did *not* resolve latency. The decision still stands on its independent merits (closed-loop responsiveness, simpler debug, action-history covers chunking's modeling wins, pygame slowdown achieves logical 30Hz for training + within-env eval), but Q5 should say that rather than leaning on a false premise.

**Main doc impact:** Q5's "Coupling with Q6" paragraph rewritten.

---

### Q8 / Q9 — Scroll-to-target environment specified

**Decision:** Scroll primitive is specified as:
- **Content:** both text-row list AND scrollable page/document, randomized per episode (widget reused/adapted from `miniwob_pygame/widgets.py`)
- **Input:** scroll-wheel only (mapped to 21-bin scroll head); no scrollbar-drag (conflicts with Drag primitive grounding), no arrow-keys for v1
- **Success criterion:** target bbox entirely inside viewport at episode end **and** no scroll events for last ≥5 frames (exercises "recognize I'm done")

**Main doc impact:** Q9 Group B recipe row for `Scroll-to-target` expanded with content + input + success spec; Q24's "Scroll" criterion refined from "visible in viewport + no more scroll needed" to the precise version above.

---

### Q24 — Tier-1.5 tolerance / edit-distance reporting added

**Decision:** keep Tier-1 binary success metrics unchanged. Add a Tier-1.5 layer for more informative diagnostic curves:

| Primitive family | Tier-1 (unchanged) | Tier-1.5 (new) |
|---|---|---|
| Click / hover / drag endpoints | Inside bbox (0px) | Success @ {0, 3, 5, 10}px cumulative |
| Type-short / med / long | Exact string match | Success at edit-distance {≤0, ≤1, ≤2}, and normalized Levenshtein ratio |

**Rationale:** binary 0px and exact-match don't distinguish "1px off" from "100px off" or "1 typo in 16 chars" from "every char wrong." Tier-2 already has the raw data (pixel distance, Levenshtein); we're just surfacing a curve at negligible cost. Keeps the physical-keyboard / bbox-strict semantics as primary, adds diagnostic granularity.

**Main doc impact:** Q24 table adds a Tier-1.5 sub-section; no change to best-checkpoint selection metric.

---

### Q27 — Fifth inference-time probe: zero-out instruction

**Decision:** add a zero-out-instruction probe alongside the existing zero-history and zero-proprio probes. 20% of each primitive's val set, instruction token zeroed.

**Rationale:** measures how much the model actually conditions on the instruction vs exploits scene-level correlations (e.g., "there's only one red thing, click it regardless of what was asked"). Complements the existing reliance probes; zero additional compute. Answers: "is instruction-following real, or is the scene picking the target?"

**Main doc impact:** Q27 probe list gains a third item; Q28 reference updated.

---

### Q35 — Simplified to HF Jobs only for v1

**Decision:** HF Jobs is the sole training compute for v1. SageMaker deferred until HF credits exhaust (per user memory note). Checkpoint-to-HF-Hub protocol stays (crash recovery works on same platform).

**Rationale:** SageMaker was originally positioned as a credits-exhaustion path, not a reliability fallback. Pre-v1 "mid-training platform switch validation" was ~1–2 days of plumbing for a scenario that may not occur during v1. When credits actually exhaust, add SageMaker as its own small project.

**Main doc impact:** Q35 infra stack table keeps HF Jobs as primary, drops SageMaker from v1 scope. "Resume-ready checkpoint protocol" simplifies to single-platform resume. Cross-platform switch validation removed.

---

## Open items tracked for revisiting after Phase A spike results

- **Spike A outcome → Q6 decision:** if typing text is illegible at `max_num_patches=256`, either bump to 576 for typing primitives specifically, or shift the typing-progress signal from visual feedback to action history (reverses Q5's current call on "Type-string handled via visual feedback").
- **Spike B outcome → Q16 decision:** if L-click single-primitive training fails to converge, capacity (Trunk A vs Trunk B) is the first place to look per Q16 confidence flag.
- **Spike C outcome → Q25 decision:** if M1 closed-loop eval is slower than projected, Tier-B cadence (every 2000 steps) may need reduction or the 400-rollout subsample may need trimming.
- **Spike E outcome → Q7 / Q21 decision:** if pygame generation is slower than 200 eps/sec, data generation timeline extends; dataloader cache strategy (Q21) may need upgrade.
