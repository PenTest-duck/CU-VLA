# Experiment 2: ACT Drag-and-Label — Design Document

## Goal

Validate that ACT (Action Chunking with Transformers) — originally designed for robotic manipulation — can produce coherent multi-step mouse trajectories and keyboard sequences for desktop computer use, at 30Hz on consumer hardware (M1 MacBook, 8GB RAM).

## Core Hypothesis

Action chunking decouples inference frequency from control frequency, enabling a transformer-based policy to output smooth, temporally coherent desktop actions. Specifically, chunking should provide a genuine advantage over reactive single-step models for:
- **Sustained mouse drag** (hold button pressed for 10-20 consecutive steps)
- **Multi-character typing** (plan and execute a 3-keystroke word as a sequence)

## Scope Limitations

This experiment validates ACT-style action chunking with multiple vision backbones on a synthetic drag-and-label task. It does **not** validate:
- Language grounding or task instructions (no VLM reasoning)
- Real screen capture (synthetic Pygame environment)
- Complex multi-application workflows
- Temporal context / frame history (single-frame conditioning)
- Keyboard modifier combos (Ctrl+C, Shift+A)
- Right click, scroll, or other mouse modalities

A positive result means: a transformer can plan multi-step drag trajectories and keystroke sequences from a single screen observation, outperforming reactive baselines on tasks requiring temporal coherence. That validates chunking as a mechanism for desktop interaction.

## Why Drag-and-Label (Not Click-and-Type)

An earlier design considered click-and-type targets (move to circle, click, press one key). That task was rejected because:
- The mouse phase is already solved reactively (Exp 1: 96.5% hit rate at 569Hz)
- A single key press doesn't require sequential planning
- A reactive single-step model could solve click-and-type without chunking
- Positive results would not demonstrate that chunking is necessary

Drag-and-label is better because:
- **Drag requires sustained button state** across many frames — one slip releases the drag
- **Typing 3 characters requires sequential planning** — the model commits to a word
- A reactive model must independently decide "keep holding" every frame and "which key next" every step
- Action chunking naturally encodes both as planned sequences

## Task Environment

### Click-and-Type Targets → Drag-and-Label (Pygame, 400x400)

**Layout:**
- Dark gray background (30, 30, 30)
- **Left half:** 1-3 colored shapes (rounded rectangles, 60-80px wide) with 3-character uppercase text labels rendered in white bold font (~24-30px)
- **Right half:** Matching drop zones (outlined rectangles, color-coded to match shapes)
- White cursor dot (3px radius), controlled by agent's delta actions
- Cursor visible in the observation frame

**Matching signal:** Color matching. Red shape → red-bordered zone. Agent learns to associate shape color with zone color. Text label is for the typing phase, not for matching.

**Colors:** Distinct palette — red, blue, green (for 3-shape variant). Additional colors added if needed.

**Word vocabulary (~20 words):**
CAT, DOG, RED, BOX, SUN, CUP, HAT, PEN, MAP, BUS, FAN, JAR, KEY, LOG, NET, OWL, RUG, TOP, VAN, WAX

All uppercase. Curated for visually distinct characters. No O/0 or I/1 ambiguity.

### Episode Flow (1 shape)

1. Shape and matching drop zone appear (randomized positions within their halves)
2. Agent moves cursor to shape (~5-15 steps)
3. Agent presses mouse_down (left_click 0→1) on shape — shape "grabbed"
4. Agent drags shape to matching zone while holding button (left_click=1 for ~10-20 steps)
5. Agent releases mouse_up (left_click 1→0) in the zone — shape "dropped"
6. Agent pauses (2-5 steps) — simulates visual confirmation
7. Agent types the 3-character label: key_down/key_up for each character (~6 steps)
8. Typed characters appear in the drop zone as visual feedback
9. Episode ends on correct final character. Wrong character = failure. Timeout at 300 steps (~10s).

### Episode Flow (3 shapes)

1. Three shapes + three matching zones appear simultaneously
2. Agent completes drag-and-label for each shape in any order
3. Completed shapes disappear. Drop zone shows typed text.
4. Expert uses nearest-first ordering heuristic
5. Episode ends when all 3 are done, or timeout at 900 steps (~30s)

### Two Modes

- **Gym mode (headless):** Off-screen rendering, `surfarray` read, resized to 224x224. For data generation, training, and primary evaluation.
- **Visual mode:** Pygame window visible. Same `surfarray` observation path. For debugging and demos.

## Action Space

State-based representation. Each timestep outputs a complete state snapshot.

| Component | Type | Range | Notes |
|-----------|------|-------|-------|
| dx | float32 regression | [-50, +50] px | tanh x max_delta_px |
| dy | float32 regression | [-50, +50] px | tanh x max_delta_px |
| left_click | binary sigmoid | {0, 1} | State: 0=released, 1=pressed. Events derived from transitions. |
| key | softmax classification | 28 classes | [no_key, A-Z, space]. Mutually exclusive. |

**Shift from Experiment 1:** State-based click (0/1) replaces event-based (no_change/down/up). Each chunk step is a self-contained state snapshot — cleaner for action chunking. This matches how ACT in robotics outputs joint positions (states) not velocities (events).

**Action chunks:** Model outputs `(chunk_size, action_dim)` — a sequence of future states. At 30Hz with chunk_size=10, each chunk covers 333ms of planned action.

## Model Architecture

### Overview

ACT = Vision Backbone + CVAE + Transformer Encoder-Decoder.

```
Observation (224x224 RGB) ──→ Vision Backbone ──→ spatial features
                                                      ↓ pool/project to 49 tokens
                                                 49 vision tokens (d=256)
                                                      ↓
Proprioception (~30D) ──────→ Linear ──────────→ 1 proprio token (d=256)
                                                      ↓
                              [49 vision + 1 proprio + 1 CVAE latent] = 51 tokens
                                                      ↓
                                        Transformer Encoder (4 layers)
                                                      ↓
                                                 encoded memory
                                                      ↓
                              chunk_size learnable query tokens (d=256)
                                                      ↓
                                        Transformer Decoder (7 layers)
                                                      ↓
                              chunk_size action predictions
                                                      ↓
                                        Action heads (per step):
                                          Linear → tanh×50 → dx
                                          Linear → tanh×50 → dy
                                          Linear → sigmoid → left_click
                                          Linear → softmax(28) → key
                                        Auxiliary head:
                                          Linear → sigmoid → is_pad
```

### Vision Backbones (Phase 1 Ablation)

| Backbone | Params | Pretrained | Output at 224x224 | Pooled to | Frozen | Feature dim |
|----------|--------|------------|-------------------|-----------|--------|-------------|
| ResNet18 | 11M | ImageNet | 7x7x512 | 49 tokens (native) | No (lr=1e-5) | 512→256 projection |
| DINOv2 ViT-S/14 | 21M | DINOv2 | 16x16x384 | 7x7=49 tokens (adaptive avg pool) | Yes | 384→256 projection |
| SigLIP2 base | 86M | SigLIP2 | 16x16x768 | 7x7=49 tokens (adaptive avg pool) | Yes | 768→256 projection |

All backbones project to d_model=256 via a learned linear layer. ViT backbones are frozen (pretrained features should capture shapes, colors, and text). ResNet18 is fine-tuned with separate lr=1e-5 (matching original ACT).

Pooling ViT tokens to 49 keeps the ACT transformer identical across all backbones — fair comparison with same sequence length and attention cost.

### CVAE Component

- **Training:** Encoder sees [observation tokens, ground-truth action chunk] → Linear(action_dim, 256) per step + Linear(proprio_dim, 256) for proprioception → CLS token → Transformer encoder → Linear(256, 64) → split to μ (32D) and log σ² (32D) → reparameterize z → Linear(32, 256) → fed as 1 token to main transformer encoder
- **Inference:** z = 0 (mean of prior). Deterministic. No sampling.
- **KL loss weight (β):** Annealed linearly from 0 → 0.1 over first 20% of training epochs
- **Latent dimension:** 32

### Transformer Configuration (Matching Original ACT)

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| Encoder layers | 4 |
| Decoder layers | 7 |
| Attention heads | 8 |
| Feedforward dim | 2048 |
| Dropout | 0.1 |
| Activation | ReLU |
| Positional encoding | Learned embeddings for proprio + latent tokens; sinusoidal for vision tokens and decoder queries |

### is_pad Head

Auxiliary `Linear(256, 1) → sigmoid` predicts whether each chunk step is padding. Handles variable-length episodes at chunk boundaries. Trained with BCE loss. At inference, pad predictions can be used to stop chunk execution early.

### Proprioception Vector

| Field | Dim | Description |
|-------|-----|-------------|
| mouse_x | 1 | Normalized cursor x (0-1) |
| mouse_y | 1 | Normalized cursor y (0-1) |
| left_click_state | 1 | Current button state (0/1) |
| key_state | 28 | One-hot of currently pressed key (or zeros if no key) |

Total: 31 dimensions. Embedded via Linear(31, 256) as one token with a learned positional embedding.

### Estimated Size and Speed

- ResNet18 variant: ~35M params. Estimated 15-25ms inference on M1 MPS.
- DINOv2 ViT-S variant: ~45M params (21M frozen + 24M transformer). Feature extraction ~40ms, transformer ~10ms. Borderline 30Hz.
- SigLIP2 variant: ~110M params (86M frozen + 24M transformer). Feature extraction potentially slow. Must profile.

## Action Chunking and Temporal Ensemble

### Chunk Execution

- **Query frequency:** k=1 (inference every step, matching original ACT paper)
- **Chunk sizes (Phase 2 ablation):** 5, 10, 20 steps at 30Hz (167ms, 333ms, 667ms)
- At every step, a new chunk is predicted. The executed action blends predictions from all active chunks.

### Temporal Ensemble

Exponential weighting with decay k=0.01 (matching ACT codebase):

```python
# For each active chunk prediction at current timestep:
weight_i = exp(-0.01 * age_i)  # age_i = how many steps since chunk_i was predicted
action_t = sum(weight_i * chunk_i_prediction_for_t) / sum(weight_i)
```

Newer predictions weighted more heavily. Smooths chunk boundary transitions while favoring more recent (better-informed) predictions.

## Expert Policy

### Navigate Phase (Fitts's Law)

Same as Experiment 1:
- Movement time: `MT = a + b * log2(D/W + 1)` where D = distance, W = shape width
- Trajectory: fast acceleration, smooth deceleration, Gaussian noise on path
- Parameters: fitts_a=0.05, fitts_b=0.15, noise_std=2.0

### Drag Phase

- mouse_down when cursor enters shape bounds
- Fitts's Law trajectory from shape center to drop zone center, button held (left_click=1) throughout
- mouse_up when cursor enters drop zone bounds
- Total drag: ~10-20 steps depending on distance

### Pause Phase

- 2-5 steps (random) of no action after dropping
- Simulates human visual confirmation before typing

### Type Phase

- For each character in the 3-char label:
  - key_down on correct key (1 step)
  - key_up (1 step)
- Total typing: 6 steps (~200ms)
- Characters appear on screen as typed (visual feedback)

### Multi-Shape Expert (3-shape variant)

- After completing one shape's drag-and-label, move to nearest remaining shape
- Small random pause (1-3 steps) between shapes
- Nearest-first ordering heuristic

### Data Recording

Recording starts when shapes appear. Idle pre-delay frames excluded (same as Exp 1). Each episode captures the full sequence from first movement to final keystroke.

## Data Format

```
data/
  episode_NNNN.hdf5
    /observations       (T, 224, 224, 3) uint8
    /actions_dx         (T,) float32
    /actions_dy         (T,) float32
    /actions_click      (T,) int8        # 0 or 1 (state-based)
    /actions_key        (T,) int8        # 0-27 (softmax class index)
    /proprio_mouse_x    (T,) float32     # normalized 0-1
    /proprio_mouse_y    (T,) float32
    /proprio_click      (T,) int8
    /proprio_key        (T, 28) int8     # one-hot
    attrs: {
      num_shapes, labels: [...], shape_colors: [...],
      shape_positions: [...], zone_positions: [...],
      drag_success: [bool, ...], type_success: [bool, ...],
      total_steps, episode_success: bool
    }
```

## Training

### Method

Behavior cloning — supervised learning from expert demonstrations.

### Chunk Sampling

Each training sample: `(observation_t, proprio_t, action_chunk_{t:t+chunk_size})`. The model predicts the full chunk from a single observation. Chunks sampled with stride 1 from episodes (every valid starting timestep). Chunks extending past episode end are padded with zeros and `is_pad=1`.

### Loss Function

```
L = L1(dx_pred, dx_true) + L1(dy_pred, dy_true)       # movement regression
  + w_click * BCE(click_pred, click_true)               # click state
  + w_key * CE(key_pred, key_true)                      # key classification
  + w_pad * BCE(pad_pred, pad_true)                     # padding prediction
  + β * D_KL(q(z|s,a) || N(0,I))                       # CVAE regularization
```

Loss weights (initial, tune empirically):
- w_click = 5.0 (sparse event, needs upweighting)
- w_key = 5.0 (sparse event, most steps are no_key)
- w_pad = 1.0
- β: annealed 0 → 0.1 over first 20% of epochs

### Optimizer

- AdamW, lr=1e-4 (transformer + action heads + projections)
- Separate backbone lr=1e-5 (ResNet18 only; ViTs frozen)
- Cosine decay schedule
- Weight decay: 1e-4

### Training Schedule

- Epochs: 500 (early stopping with patience=50 on validation loss)
- Batch size: 64
- Mixed precision (AMP) for memory efficiency on 8GB M1
- Val split: 20% by episode (held-out scene configurations, not random frames)

### Data Scale

- Initial: 10,000 single-shape episodes (~40 steps each = ~400K frames)
- Scale up if underfitting
- 3-shape episodes generated separately when progressing

## Ablation Strategy

### Phase 1: Vision Backbone (fix chunk_size=10, softmax keys)

| Config | Backbone | Chunk | Keys |
|--------|----------|-------|------|
| P1-A | ResNet18 (11M, fine-tuned) | 10 | softmax |
| P1-B | DINOv2 ViT-S/14 (21M, frozen) | 10 | softmax |
| P1-C | SigLIP2 base (86M, frozen) | 10 | softmax |

**Decision:** Best backbone (highest sequence completion) advances to Phase 2.

### Phase 2: Chunk Size (fix best backbone, softmax keys)

| Config | Backbone | Chunk | Keys |
|--------|----------|-------|------|
| P2-A | Best from P1 | 5 | softmax |
| P2-B | Best from P1 | 10 | softmax (reuse P1 result) |
| P2-C | Best from P1 | 20 | softmax |

**Total training runs:** 5 (3 in Phase 1 + 2 new in Phase 2) + baselines.

## Evaluation

### Primary Evaluation

200 episodes per config, headless Gym mode.

### Execution Protocol

- Inference every step (k=1)
- Temporal ensemble with exponential weighting (decay k=0.01)
- Actions executed at 30Hz control rate

### Metrics (Decomposed by Phase)

| Metric | Definition | Target |
|--------|-----------|--------|
| **Navigate** | | |
| Navigate success | Cursor reaches shape | >90% |
| Navigate time | Steps from start to shape contact | Report p50/p95 |
| **Drag** | | |
| Drag success | Shape dropped in correct zone | >70% |
| Drag smoothness | Mean jerk of mouse trajectory during drag | Report (lower = smoother) |
| Drag duration | Steps from grab to drop | Report p50/p95 |
| Button hold accuracy | % of drag steps with correct left_click=1 | Report |
| **Type** | | |
| Character accuracy | % individual characters typed correctly | Report |
| Word accuracy | % complete 3-char words typed correctly (given successful drag) | >60% |
| Type latency | Steps from drop to first keystroke | Report p50/p95 |
| **Combined** | | |
| Sequence completion | Full success: navigate + drag + type all correct | >50% (1-shape) |
| Multi-target completion | All 3 shapes completed correctly | Report (3-shape) |
| **System** | | |
| Per-step loop latency | Wall time per observe→infer→act cycle | <33ms p95 (30Hz) |
| Effective Hz (p95) | 1 / p95_latency | >=30Hz |

### Baselines

1. **Scripted expert** — upper bound (~100% across all metrics)
2. **Extended TinyCNN + key head** — single-step reactive model with same action space. No chunking. Same training data. Direct comparison: does chunking help?
3. **Random agent** — lower bound (sanity check)

### Diagnostic Outputs

- Trajectory visualizations with drag paths, click points, and keystroke markers
- Per-phase timing breakdown (navigate / drag / pause / type)
- Per-ablation comparison tables (Phase 1: backbone, Phase 2: chunk size)
- Training loss curves (per head + KL divergence)
- Action chunk heatmaps: visualize predicted chunk contents at key moments (grab, mid-drag, drop, typing)
- Button hold analysis: plot left_click state over time for representative episodes
- Temporal ensemble effect: compare best config with vs without ensemble

## Project Structure

```
experiments/
  act_drag_label/
    config.py           # All hyperparameters (env, action, model, training, eval, expert)
    env.py              # DragLabelEnv — Pygame Gym-style env (1-shape + 3-shape modes)
    expert.py           # Fitts's Law movement + drag + pause + type expert
    baseline_cnn.py     # Extended TinyCNN with key head (no-chunking baseline)
    generate_data.py    # Run expert → HDF5 episodes
    model.py            # ACT model: vision backbone + CVAE + Transformer encoder-decoder
    backbones.py        # ResNet18, DINOv2, SigLIP2 feature extractors with pooling
    train.py            # BC training loop with chunk sampling + KL annealing + AMP
    evaluate.py         # Run all agents, compute decomposed metrics
    data/               # HDF5 episodes (gitignored)
    checkpoints/        # Model weights (gitignored)
    results/            # Plots, metrics, ablation tables
```

## Dependencies

`torch`, `torchvision` (ResNet18), `transformers` (DINOv2, SigLIP2), `pygame`, `h5py`, `numpy`, `matplotlib`

## Run Sequence

```bash
# Data generation
uv run python experiments/act_drag_label/generate_data.py                          # 10K 1-shape episodes
uv run python experiments/act_drag_label/generate_data.py --num-shapes 3           # 3-shape episodes (later)

# Phase 1: Vision backbone ablation
uv run python experiments/act_drag_label/train.py --backbone resnet18 --chunk-size 10
uv run python experiments/act_drag_label/train.py --backbone dinov2-vits14 --chunk-size 10
uv run python experiments/act_drag_label/train.py --backbone siglip2-base --chunk-size 10

# Phase 2: Chunk size ablation (with best backbone from Phase 1)
uv run python experiments/act_drag_label/train.py --backbone <best> --chunk-size 5
uv run python experiments/act_drag_label/train.py --backbone <best> --chunk-size 20

# Evaluation
uv run python experiments/act_drag_label/evaluate.py                               # all configs + baselines
uv run python experiments/act_drag_label/evaluate.py --visual                      # with Pygame window
uv run python experiments/act_drag_label/evaluate.py --num-shapes 3                # 3-shape variant
```
