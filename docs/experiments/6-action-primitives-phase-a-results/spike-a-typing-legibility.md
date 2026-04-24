# Spike A — Typing legibility probe

**Ran:** 2026-04-24 (first run + methodology improvement + re-run)
**Script:** `experiments/action_primitives/probes/typing_legibility.py`
**Data (run 1, mean-pool):** `spike-a-typing-legibility.json`
**Data (run 2, attention-pool):** `spike-a-typing-legibility-attention.json`

## Question

Does SigLIP2-B-naflex at `max_num_patches=256` (the Phase B design choice in Q6/Q29) preserve enough character-identity information to support visual typing feedback at the design floor of 14pt (Q5)? If the answer is clearly "no", Phase B's typing-primitive design needs revision (bump patches to 576 for typing, or move typing progress signal to action history instead of vision).

## Method

Rendered N random 3–12-char strings (charset: A-Z, a-z, 0-9, space; 63 classes) at each font size in {8, 10, 12, 14, 16, 20, 24, 32}pt on white 720×450 pygame-sized surfaces (rendered via PIL due to a Python 3.14 + pygame 2.6.1 font circular-import bug; pygame-ce fixes this and we swap in T6). Encoded through frozen SigLIP2-B-naflex @ `max_num_patches=256`. Trained a linear probe from pooled patch features (d=768) to a 63-class multi-label char-presence vector. Reports macro-F1 on an 80/20 test split.

**Two runs:**

- **Run 1** (mean-pool, N=80, commit [179bf4c](../../experiments/action_primitives/probes/typing_legibility.py)) — pool = global average over all real (non-padded) patches. JSON file `spike-a-typing-legibility.json` was written with an old schema where the field is named `top1_accuracy` but its value is macro-F1 (schema fixed in [7b403b8](../../experiments/action_primitives/probes/typing_legibility.py); value is unchanged).
- **Run 2** (attention-pool, N=500, commit [7b403b8](../../experiments/action_primitives/probes/typing_legibility.py)) — pool = softmax-weighted sum over patches via a single learnable query vector `q` (jointly trained with the linear head, padded patches masked with -inf before softmax).

The attention pool is a closer approximation to what the Phase B model's trunk will do (16 cross-attention queries over patches + 3 transformer blocks), so Run 2 is the methodologically stronger result. Run 1 is preserved as a baseline for reference.

Neither probe tests **per-patch** char identity — we are asking "does the *pooled* feature still carry char presence", not "can we localize each character". Per-patch is available as follow-up if the Phase B design needs a stricter gate.

## Results

### Run 1 — mean-pool, N=80

| Font size (pt) | Macro-F1 (test) |
|---|---|
| 8  | 0.000 |
| 10 | 0.009 |
| 12 | 0.061 |
| 14 | 0.051 ← Q6 design floor |
| 16 | 0.038 |
| 20 | 0.123 |
| 24 | 0.085 |
| 32 | 0.188 |

### Run 2 — attention-pool, N=500

| Font size (pt) | Train F1 | Test F1 | Train-test gap |
|---|---|---|---|
| 8  | 0.378 | 0.092 | 0.286 |
| 10 | 0.509 | 0.135 | 0.374 |
| 12 | 0.696 | 0.233 | 0.463 |
| 14 | **0.717** | **0.329** | **0.388** ← Q6 design floor |
| 16 | 0.769 | 0.363 | 0.406 |
| 20 | 0.822 | 0.443 | 0.379 |
| 24 | 0.846 | 0.473 | 0.373 |
| 32 | 0.859 | 0.503 | 0.356 |

## Interpretation

**The attention-pool run shifts the reading of this spike materially:**

1. **The mean-pool run was methodology-limited, not SigLIP2-limited.** Moving from mean-pool to attention-pool produces a 6× jump at 14pt (0.05 → 0.33) and monotonizes the curve (Run 1 had noise dominating signal at N=80: 12pt > 14pt > 16pt, 20pt > 24pt). With N=500 the curve is strictly monotonic.

2. **SigLIP2 features do encode char identity at 14pt.** Train F1 @ 14pt = 0.72 (≥ 0.7). The information is present in the (pooled) 768-d feature; a linear head can fit it given the training distribution.

3. **The linear probe does not generalize well.** Train-test gaps of ~0.3–0.5 at every size indicate the bottleneck is the probe's ability to generalize over a 63-class multi-label target with only 400 training strings, not SigLIP2's ability to see characters. Each string has 3–12 unique chars out of 63, so positive signal per class is sparse; the probe over-fits to co-occurrence structure in the training set.

4. **Test F1 @ 14pt = 0.33 is below the rubric's 0.5 "proceed" threshold**, but the rubric was designed around a simpler mental model of the probe (test F1 ≈ model's ceiling). Given (a) the train-test gap, (b) that the real trunk has 16 learned queries + 3 transformer blocks + end-to-end supervision on the typing task (strictly more expressive than our single-query linear probe), and (c) that train F1 clears 0.7 at 14pt, **the probe's test F1 is a loose lower bound on what the real model can learn**, not a tight ceiling.

### Against the literal rubric

- F1 @ 14pt ≥ 0.7 → **not met on test (0.33); met on train (0.72)**
- 0.5 ≤ F1 @ 14pt < 0.7 → **not in this range on test**
- F1 @ 14pt < 0.5 → **met on test (0.33)**

If we apply the rubric strictly (test F1 only), the verdict is: Phase B revision needed — either bump `max_num_patches` to 576 for typing primitives or shift typing progress signal to action history.

### Against a nuanced reading

The rubric is a heuristic. The probe as specified is a **lower bound** on what the model can learn, not the ceiling. A per-patch probe (stricter than presence) would produce a harder test; a larger MLP head or more training strings would produce an easier test. The current result is consistent with several scenarios:

- **Scenario A (optimistic):** SigLIP2@256 provides enough signal at 14pt; the probe's generalization gap is driven by small-N + high-class-count + sparse labels, not by feature quality. The real ACT trunk, trained end-to-end on typing-primitive task loss, will comfortably use the visual feedback. No Phase B design change needed.
- **Scenario B (pessimistic):** Even with a more expressive probe, test F1 stays below what's needed for robust typing progress. Phase B typing primitives struggle to learn without either bumped patches or an action-history fallback.

**We can't distinguish A from B from this probe alone.** The cheapest way to resolve is during early Phase B training itself: if typing-primitive BC loss plateaus or generalization stalls, pivot to one of the fallbacks.

## Recommendation

**Proceed to Phase A as designed.** Phase A is L-click-only; typing is not on the critical path. The probe result is inconclusive — it rules out the mean-pool "worst case" (Run 1 would have forced a revision) but does not cleanly pass the attention-pool "strict case" (Run 2 is below the rubric but above what a pure-noise result would give, with clear evidence of information being present in the features).

**For Phase B typing primitives**, adopt an adaptive strategy rather than pre-committing a design change:

1. **Keep the Phase A/B baseline** at `max_num_patches=256` with visual typing feedback (Q5/Q6 as-written).
2. **Add an early-training checkpoint**: after the first few epochs of typing-primitive BC, measure per-primitive typing success rate on a dev slice. If typing primitives are learning ≥ 70% of the non-typing success rate at that checkpoint, continue as-designed. If typing primitives are clearly lagging (< 40%), trigger the fallback.
3. **Pre-commit the fallback mechanics so the pivot is cheap when it fires:**
   - **Fallback-1 (vision):** re-train typing-primitive instances at `max_num_patches=576` while keeping other primitives at 256 (costs wall-clock but doesn't revise the architecture).
   - **Fallback-2 (proprio):** add typing progress / buffer contents to the proprio/action-history token stream so typing primitives can succeed without relying on visual feedback.
4. **Optionally run a per-patch stricter probe** during Phase B prep if time permits — it would test char localization rather than presence and would more directly simulate what the cross-attention queries need to discriminate.

This shifts the risk-mitigation from "pre-commit a design change on an ambiguous probe" to "build a cheap pivot path and decide empirically in Phase B". I'm flagging this shift as a deliberate deviation from the Phase A plan's rubric language, because the amended methodology changed what the rubric was measuring against. If you prefer the stricter literal read, say so and I'll open a Phase B amendment to switch typing primitives to `max_num_patches=576` by default.

## Next steps

- [ ] Proceed to T6 (LClickEnv) with the current Phase A plan.
- [ ] Add the early-training typing-legibility checkpoint to the Phase B plan when it's written.
- [ ] Optional: per-patch probe (stricter) as a dedicated Phase B feasibility spike before typing primitives land.
