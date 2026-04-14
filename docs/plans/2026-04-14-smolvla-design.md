# Experiment 4: SmolVLA for Computer Use — Design Doc

## Primary Research Question

**Does SmolVLA, a VLA designed for robotics, also work for computer use?**

This is a direct test of the project's founding hypothesis: that the robotics VLA paradigm (perceive screen, understand instruction, output continuous actions at high frequency) transfers to desktop interaction. SmolVLA (450M params, HuggingFace) was built for robot manipulation — we adapt it for mouse and keyboard control at 30Hz.

## Secondary Questions

- **Robot pre-training transfer:** Does robot manipulation pre-training in the action expert transfer to computer use, or is a fresh expert equally good / better? (Ablation: pretrained vs fresh action expert.)
- **Language grounding:** Does processing task instructions as language tokens (via VLM) improve over reading them from rendered pixels? (Comparison: SmolVLA vs ACT baseline.)
- **Control frequency:** Can we achieve 30Hz action execution with ~4Hz closed-loop replanning on consumer hardware (M1 MacBook) via action chunking? Note: this is 30Hz *execution* rate, not 30Hz *closed-loop feedback*. The VLM replans at ~4Hz; actions execute open-loop between replanning cycles.

## Environment

Exp 3 Phase 1 — 4 core primitive tasks from the MiniWoB-Pygame suite:

| Task | Motor primitive | Language relevance |
|------|----------------|-------------------|
| click-target | Point + click | Low (color is visual) |
| drag-to-zone | Sustained drag | Low (color matching) |
| use-slider | Fine 1D drag | Medium (must read target number) |
| type-field | Click + type sequence | High (must read target string) |

Pygame window: **512×512** (changed from 400×400 to match SigLIP's native input resolution — no resize or padding needed). Exp 3 data to be regenerated at this resolution.

## Architecture

### Overview

Three components, mirroring SmolVLA's robotics design:

```
Screenshot (512×512) ──→ SigLIP Vision Encoder ──→ 64 visual tokens ─┐
                                                                      ├──→ Frozen VLM (16 layers) ──→ fused features
Instruction text ──────→ SmolLM2 Tokenizer ──────→ ~20 text tokens ──┘          │
                                                                                 │
State (cursor + keys) ─→ Linear projection ──────→ 1 prefix token ──────────────┘
                                                                                 │
                                                              ┌──────────────────┘
                                                              ▼
                                                    Action Expert (~100M)
                                                    Interleaved CA + SA
                                                              │
                                              ┌───────────────┴───────────────┐
                                              ▼                               ▼
                                    Flow Matching Head              Classification Head
                                    dx, dy (continuous)          mouse_left + keys (binary)
                                    10 Euler denoise steps              BCE loss
                                    MSE velocity loss
                                              │                               │
                                              └───────────┬───────────────────┘
                                                          ▼
                                                Action chunk (10, 43)
```

### 1. Frozen VLM Backbone (~350M params, not trained)

- **Model:** SmolVLM2-500M-Video-Instruct, truncated to 16/32 layers
- **Vision:** SigLIP encoder processes 512×512 screenshot → 64 visual tokens via PixelShuffle
- **Language:** SmolLM2 tokenizer + text embeddings process instruction string
- **State:** Cursor position + held keys projected to 1 token, prepended as VLM prefix
- **Output:** Fused vision+language+state features, used as keys/values for action expert cross-attention
- **Loaded from:** `transformers` library (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`)

### 2. Flow Matching Action Expert (~100M params, trained)

- **Architecture:** Transformer with interleaved cross-attention (attend to VLM features) and self-attention (causal, between action timesteps). Pattern: CA, SA, CA, SA, ...
- **Expert width:** 0.75× VLM hidden dimension
- **Hybrid output heads:**
  - **Continuous (flow matching):** dx, dy — predicted via 10 Euler denoising steps. Loss: MSE on velocity field.
  - **Discrete (classification):** mouse_left + 40 key channels — predicted via sigmoid. Loss: Asymmetric Loss (ASL) to handle severe class imbalance (most keys are 0 most of the time).
- **Head coupling:** Independent — both heads read from shared transformer hidden states but don't see each other's outputs. Avoids train-test distribution shift where discrete head would be conditioned on ground-truth continuous values during training but denoised values at inference.
- **Loss balancing:** GradNorm (gradient normalization) to automatically balance flow matching MSE and ASL gradients during training. Prevents one head from dominating.
- **Time sampling:** Beta(1.5, 1.0), clamped to [0.001, 1.0)
- **Inference:** Start from Gaussian noise, 10 Euler steps for dx/dy; direct prediction for discrete actions from final hidden states. Binary threshold at 0.5 for all discrete channels.

### 3. Async Inference Loop

- VLM processes one screenshot every 10 steps (~333ms)
- Action expert generates 10-action chunk
- Actions dispatched from queue at 30Hz
- Effective: ~4Hz inference, 30Hz control

## Action Space

Hybrid continuous + binary, matching Exp 3's held-state design (minus scroll, ctrl, shift, alt — not needed for Phase 1):

| Dims | Name | Type | Loss |
|------|------|------|------|
| 0–1 | dx, dy | continuous | Flow matching (MSE velocity) |
| 2 | mouse_left | binary | ASL (Asymmetric Loss) |
| 3–28 | A–Z | binary | ASL |
| 29 | space | binary | ASL |
| 30 | enter | binary | ASL |
| 31 | backspace | binary | ASL |
| 32 | tab | binary | ASL |
| 33–42 | 0–9 | binary | ASL |

Total: 2 continuous + 41 binary = **43 dimensions**.

Scroll: excluded (YAGNI — no Phase 1 task uses it). When needed for Phase 3 (scroll-and-click), add as a 3-class categorical head (none/up/down, CE loss).

## State (Proprioception)

Projected to 1 VLM prefix token via linear layer:

- `cursor_pos`: (2,) float32, normalized [0, 1]
- `mouse_left`: (1,) binary
- `keys_held`: (40,) binary (same key channels as action space)

Total: 43 dims → Linear → 1 token of VLM hidden dimension.

## Chunk Configuration

- **Chunk size:** 10 actions (SmolVLA ablation optimal: 84% at chunk=10 vs 80.3% at chunk=50)
- **Observation frequency:** Every 10 steps (333ms). SmolVLA showed this actually outperforms every-step observation (82.8% vs 80.3%).
- **Temporal ensembling:** Not used (simple queue: generate 10, pop 1 per step, refill when empty).

## Training

### Data

Expert demonstrations from Exp 3's scripted Fitts's Law experts:

| Task | Episodes | ~Steps/episode |
|------|----------|----------------|
| click-target | 5,000 | ~30 |
| drag-to-zone | 5,000 | ~60 |
| use-slider | 5,000 | ~50 |
| type-field | 5,000 | ~80 |

Total: ~20K episodes, ~1.1M frames. Stored as Parquet via HF datasets (consistent with Exp 2/3). Each row: image (PNG), action (43D), state (43D), instruction (string), episode_id, timestep.

### Training Runs

| Run | VLM Backbone | Action Expert | Purpose |
|-----|-------------|---------------|---------|
| **SmolVLA-fresh** | Frozen SmolVLM2-500M | Trained from scratch | Primary: does SmolVLA work for computer use? |
| **SmolVLA-robot** | Frozen SmolVLM2-500M | Fine-tuned from `lerobot/smolvla_base` robot weights | Ablation: does robot pre-training transfer? |
| **ACT baseline** | None (SigLIP2) | CVAE + transformer (Exp 3 architecture) | Baseline: instruction as rendered pixels, 512×512, pooled to 49 tokens |

### Hyperparameters

Starting from SmolVLA simulation fine-tuning defaults:

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10) |
| Warmup | 100 steps |
| LR schedule | Cosine decay to 2.5e-6 |
| Batch size | 64 |
| Steps | 100K |
| Precision | bfloat16 (float16 on MPS) |
| Grad clip | 10 |
| Trainable params | Action expert + state projection only (~100M) |

### Hardware

- **Training:** HF Jobs with L4 GPU. Estimated ~4–6 hours per run, ~12–18 GPU hours total for 3 runs.
- **Evaluation:** Local M1 MacBook (need to benchmark inference latency here anyway).

## Evaluation

### Protocol

- 200 episodes per task per model (consistent with Exp 1/2)
- Deterministic seeds for reproducibility
- Both headless (metrics) and visual (qualitative) modes

### Metrics

**Per-task:**
- Success rate (%)
- Completion time (steps → ms at 30Hz)
- Per-primitive: hit rate (click), drop accuracy (drag), value error (slider), character accuracy (type)

**Aggregate:**
- Average success across 4 tasks
- Inference latency: VLM forward (ms), denoising steps (ms), total per-chunk (ms)
- Effective control Hz

### Baselines

All models evaluated on the same 512×512 Pygame environment:

| Baseline | Purpose |
|----------|---------|
| **Expert (scripted)** | Performance ceiling — what's the best possible? |
| **Random policy** | Performance floor — what does chance look like? |
| **ACT (Exp 3, re-trained at 512×512)** | Instruction-as-pixels baseline. SigLIP2 backbone, pooled to 49 tokens via adaptive avg pooling. |

### Success Criteria

Success is defined relative to expert ceiling, not an arbitrary threshold:

- SmolVLA-fresh achieves **>50% of expert success rate** on each of the 4 tasks → **"yes, SmolVLA works for computer use"**
- SmolVLA matches or exceeds ACT baseline → language-as-tokens provides advantage over language-as-pixels
- 30Hz action execution with ~4Hz replanning achieved on M1

## Codebase

### Approach

Extract SmolVLA model from LeRobot (3 files), write our own training loop and data pipeline:

**Extracted from LeRobot** (with trivial import replacements):
- `smolvlm_with_expert.py` — VLM + action expert integration
- `modeling_smolvla.py` — SmolVLAPolicy, VLAFlowMatching
- `configuration_smolvla.py` — config dataclass

**Written ourselves** (reusing Exp 2/3 patterns):
- `generate_data.py` — run Exp 3 experts, save in our Parquet format
- `train.py` — hybrid loss (flow matching + BCE), AdamW, AMP
- `evaluate.py` — per-task and aggregate metrics
- `config.py` — experiment hyperparameters

**Not extracted** (replaced with our own ~30 lines):
- `processor_smolvla.py` — depends on 15+ file LeRobot processor module. We write our own preprocessing: resize image, normalize, tokenize text with `AutoProcessor`.

### Dependencies

- `torch` — model and training
- `transformers` — SmolVLM2 backbone loading
- `datasets` — Parquet data pipeline (already used in Exp 2/3)
- `safetensors` — weight loading from `lerobot/smolvla_base`

### Directory Structure

```
experiments/smolvla/
├── __init__.py
├── config.py              # All hyperparameters
├── smolvlm_with_expert.py # Extracted + cleaned from LeRobot
├── modeling_smolvla.py    # Extracted + cleaned from LeRobot
├── configuration_smolvla.py # Extracted + cleaned from LeRobot
├── generate_data.py       # Expert demo generation
├── train.py               # Training loop (hybrid loss)
├── evaluate.py            # Evaluation harness
└── hf_sync.py             # Upload/download checkpoints
```

## Key Risks

1. **Hybrid action head engineering.** Splitting flow matching (continuous) from classification (discrete) within SmolVLA's denoising loop is the hardest task. The continuous dims go through Euler steps while discrete dims need direct prediction — these share the same expert transformer but diverge at the output.

2. **MPS compatibility.** SmolVLM2 + SigLIP on Apple Silicon may have operator gaps. Early smoke test needed.

3. **Vision encoder frozen = can't learn Pygame visual features.** SmolVLM2's SigLIP was pre-trained on natural images and documents. Pygame's simple vector graphics may not activate useful features. If results are poor, consider LoRA on the vision encoder or unfreezing the connector layers.

4. **Inference speed on M1.** Estimated ~200–230ms per chunk (50–80ms VLM + 10×15ms expert). With chunk=10 at 30Hz, budget is 333ms — should fit, but needs empirical verification.

## Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Model | Full SmolVLA adaptation | Directly tests project thesis |
| Primary question | Does SmolVLA work for computer use? | Core hypothesis of entire project |
| Tasks | Exp 3 Phase 1 (4 tasks) | Motor diversity without scope explosion |
| Pre-training | Ablate both (robot vs fresh) | Novel research finding either way |
| Action space | Hybrid: flow matching + ASL | Principled for mixed continuous/discrete |
| Loss balancing | GradNorm | Auto-balance flow matching MSE vs ASL gradients |
| Key imbalance | Asymmetric Loss (ASL) | Handles severe class imbalance in binary key channels |
| Head coupling | Independent (shared hidden states) | Avoids train-test skew from conditioning discrete on continuous |
| ACT baseline | Re-train at 512×512 with adaptive pooling to 49 tokens | Fair resolution-matched comparison |
| Frame history | Single frame (no action history) | Matches SmolVLA design; state + screenshot captures past implicitly |
| Scroll | Excluded (YAGNI) | No Phase 1 task uses it |
| Chunk size | 10, observe every 10 steps | SmolVLA ablation optimal |
| VLM backbone | SmolVLM2-500M-Video-Instruct | Weight compatibility for robot ablation |
| Image resolution | 512×512 Pygame window | Native SigLIP resolution, no padding waste |
| Codebase | Extract 3 files from LeRobot | Weight compatibility + less engineering than rewrite |
| Training hardware | HF Jobs (L4) | M1 too slow for 100M params × 100K steps |

## Open Questions

1. **Frozen vision encoder:** Will SigLIP's pre-trained features work for Pygame screenshots? May need LoRA or unfreezing if results are poor.
2. **Multi-task vs per-task:** Design assumes per-task training first. Multi-task (single model on all 4 tasks) is a stretch goal — requires task-conditioned training and may surface negative transfer.
3. **Video backbone for static frames:** SmolVLM2-500M-Video-Instruct was pre-trained on video, but we feed single static frames. The video priors (temporal attention, frame position IDs) may be inactive or mildly harmful. We chose this backbone for weight compatibility with `lerobot/smolvla_base` (required for robot pre-training ablation). If single-frame performance is poor, consider feeding 2-frame history as an ablation.
4. **Number of training seeds:** Single run per condition is fragile. Budget permitting, run 2-3 seeds for credible comparisons.
5. **Denoising step count:** 10 Euler steps is SmolVLA's default. Fewer steps (5) would be faster; more (20) might be more accurate. No sensitivity analysis planned yet.

## Reviewer Feedback (Incorporated)

An independent AI reviewer flagged the following issues (addressed above):
- Success criteria should be relative to expert ceiling, not arbitrary → fixed (>50% of expert rate)
- Need expert and random baselines → added to evaluation
- Loss weighting unspecified → GradNorm
- Key label imbalance → Asymmetric Loss (ASL)
- ACT baseline at different resolution confounds comparison → re-train at 512×512
- Hybrid head train-test skew → independent heads (shared hidden states, no cross-conditioning)
- "30Hz" is execution rate, not closed-loop bandwidth → clarified as ~4Hz replanning + 30Hz execution
- Video backbone for static frames → documented as open question
- Multiple seeds needed → documented as open question
