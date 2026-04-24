# Spike B — L-click end-to-end

**Ran:** 2026-04-24
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
| Wall-clock | [pending — fill from wandb/HF jobs] |
| Optimizer | AdamW, trunk lr 3e-4, LoRA lr 2e-4, betas (0.9, 0.95), wd 0.01 |
| LR schedule | Linear warmup 100 steps → cosine decay to 10% |
| Precision | bf16 autocast |
| Grad clip | 1.0 |

Model total params: **408.8M** (vision 93M frozen + text 282M frozen + trunk 28M + heads 4M + encoders 1M + LoRA 0.6M trainable). Trainable total: **33.6M**.

## Offline eval (val split, per-head top-1 accuracy)

Ran: `uv run python -m experiments.action_primitives.evaluate --checkpoint final.pt --data-dir data/phase-a-lclick --n-rollouts 200 --device mps`

| Head | Top-1 accuracy | Notes |
|---|---|---|
| dx | [pending] | 21-bin exponential; chance = 0.048 |
| dy | [pending] | 21-bin exponential; chance = 0.048 |
| click | [pending] | 5-way; chance = 0.200 |
| scroll | [pending] | 21-bin; scroll is always idle in L-click primitive, expect ~1.0 |
| keys | [pending] | 77 × 3-way; always idle in L-click, expect ~1.0 |
| done | [pending] | binary; chance = 0.500 |

## Closed-loop eval (200 rollouts)

| Metric | Rate |
|---|---|
| **success_rate** | [pending] ← Spike B gate: ≥0.50 |
| click_within_0px | [pending] |
| click_within_3px | [pending] |
| click_within_5px | [pending] |
| click_within_10px | [pending] |

### Quick-look visual eval (n=20, live pygame window)

Before the headless n=200 run, did an `--visual --n-rollouts 20` pass to confirm the agent is actually acting sensibly (not, e.g., stuck at origin):

| Metric | Value |
|---|---|
| success_rate | 0.70 |
| click_within_0px | 0.00 |
| click_within_3px | 0.05 |
| click_within_5px | 0.25 |
| click_within_10px | 0.60 |

Sample trajectory: 14/20 clean presses, frames 7-34, end-distance 0.4-12.6 px. Agent adapts to tempo and consistently moves toward the button.

## Failure-mode analysis

Observed in the n=20 visual pass:

**Mode A — "stuck far away" (5/6 failures, dist ≥230 px):** episodes 1, 13, 14, 18, 20. Hit max_frames=45 with cursor still ~250 px from target. The 45-frame × 18 px/frame (slow-tempo peak) = 810 px of reachable motion is more than enough for the ~850 px canvas diagonal, so the physical budget exists. Hypothesis: model sometimes outputs `dx`/`dy` near the middle bin (zero-motion attractor under discretized regression + focal loss), stalling instead of moving. [confirm / refute with n=200 data]

**Mode B — "close but didn't click" (1/6, dist=32.9 px on ep 15):** cursor arrived near target but didn't complete press+release cleanly. Could be (a) click head didn't fire at the right cursor position, (b) pressed but released off-target, (c) misjudged bbox edge on a small button. Rare at n=20.

## Interpretation

- **[pass / borderline / fail — pick one based on n=200 success rate]**
- Phase B target >50% ✓/✗
- If ≥70%: pipeline solidly validated, proceed to Phase B as-designed.
- If 50-70%: pipeline works but with meaningful failure modes; Phase B should include the early-training typing checkpoint (from Spike A recommendation) and revisit the "zero-motion attractor" hypothesis.
- If <50%: diagnose before Phase B. Most likely culprits: (a) insufficient training (5 epochs may be too few), (b) loss weighting (done head is tiny, keys head is huge — could dominate gradient), (c) capacity (trunk + LoRA rank-8 may be under-powered for the combined vision + motor decoding task).

### Tolerance-curve shape

If `click_within_10px` is much higher than binary `success_rate`, the model lands near target but either (a) overshoots the press boundary, or (b) presses late and cursor drifts during settle. Diagnostic for Phase B UX concerns — Phase A success criterion is binary-in-bbox so this isn't a pass/fail, just a planning signal.

## Recommendation

[fill in after n=200 numbers land — example framings:]

- **Pass (≥70%):** "Phase B plan stands. Proceed to generate multi-primitive data + expand architecture per Phase B design."
- **Borderline (50-70%):** "Phase B starts with a 1-epoch sanity check on this same L-click config + the early-training typing checkpoint from Spike A; if L-click drops on re-run, re-examine loss weights before expanding primitives."
- **Fail (<50%):** "Block Phase B. Run diagnostic experiments first: (a) train 20 epochs on the same data to see if 5 was undertraining; (b) try loss-weight rebalancing per Q2 (head-wise L_i^init normalization); (c) bump trunk from 3 blocks → 6 blocks if capacity is the issue."

## Follow-ups (not blocking Phase A)

- **Mode A root cause:** if n=200 confirms the stuck-far-away pattern, add a Phase B diagnostic logging the bin-10 frequency on `dx`/`dy` heads per frame. A high rate of bin-10 predictions on frames where the expert was moving = confirmation.
- **Per-primitive success breakdown for Phase B:** Spike B is L-click only, so the success rate is a single scalar. Phase B will need per-primitive numbers.
- **Best.pt vs final.pt:** this eval used `final.pt` (step 190). `best.pt` (lowest val loss) is also in the HF repo — worth a quick comparison eval to see whether training was over-fitting in the last ~20 steps.

## Next steps

- [ ] Plug in n=200 numbers (offline + closed-loop + tolerance curves) once the eval completes.
- [ ] Decide verdict (pass/borderline/fail) based on `success_rate`.
- [ ] Commit write-up.
- [ ] Proceed to Spike C (M1 eval timing) with the same checkpoint.
