# Experiment 1: Reactive Clicks

Validates the core visuomotor pointing loop: a tiny CNN (2.2M params) observes a 128x128 frame and outputs discretized delta mouse movements + click at 30 Hz, trained via behavior cloning from a scripted expert.

A red circle appears at a random position on a dark background. The agent must move a cursor to it and click. Episode ends on first click (hit or miss).

**What a positive result means:** a small neural network can close a perception-to-action loop at 30 Hz on consumer hardware for a simple pointing task. That's the foundation for higher-level VLA computer use, not the building itself. See the [design doc](../../docs/plans/2026-04-11-reactive-clicks-design.md) for full scope limitations.

## Quick Start

All commands run from the repo root.

```bash
# 1. Generate 1000 expert demonstration episodes (~40s)
uv run python experiments/reactive_clicks/generate_data.py

# 2. Train the CNN via behavior cloning (~50 epochs)
uv run python experiments/reactive_clicks/train.py

# 3. Evaluate all agents (expert, baseline, CNN) and report metrics
uv run python experiments/reactive_clicks/evaluate.py

# 4. (Optional) Watch the CNN play in a visible Pygame window
uv run python experiments/reactive_clicks/evaluate.py --visual
```

Flags: `--device mps` for M1 GPU acceleration, `-n 500` to change episode count.

## How It Works

### Environment

A 400x400 Pygame window (or headless surface). Red circle (radius 20-40px) appears at a random position. A white cursor dot is rendered in the frame. The agent controls the cursor via relative delta movements.

Two modes, identical observation path:
- **Headless** (default): renders to off-screen surface. Used for data generation, training, and fast evaluation.
- **Visual** (`--visual`): opens a Pygame window so you can watch. Same pixel data.

### Action Space

Each timestep the agent outputs three things:

| Output | Encoding | Details |
|--------|----------|---------|
| Δx | Continuous float | tanh-scaled to [-50, +50] pixels |
| Δy | Continuous float | Same as Δx |
| Button | 3 classes | 0 = no change, 1 = mouse_down, 2 = mouse_up |

Movement is continuous regression (L1 loss), not discretized. The model outputs raw values through `tanh * max_delta` for bounded, differentiable output.

### Expert (Fitts's Law)

The scripted expert generates training demonstrations:
- Uses Fitts's Law (`MT = a + b * log2(D/W + 1)`) to determine movement duration
- Bell-shaped velocity profile: accelerate, then decelerate
- Online correction: each step steers toward remaining distance to target
- Gaussian noise on path for variability

### Baseline (No Learning)

A deterministic controller for comparison:
- Detects red pixels via color threshold
- Computes centroid of red region
- Beelines cursor toward centroid
- Clicks when overlapping

Establishes the timing floor for the pipeline.

### Model (TinyCNN)

```
128x128x3 input
  -> 4x Conv2d(3x3, stride 2) with ReLU  -> 8x8x128
  -> Flatten -> 8192
  -> Linear(256) + ReLU
  -> 3 heads: dx(1)->tanh*50, dy(1)->tanh*50, btn(3)

~2.2M parameters, <2ms inference on M1 CPU
```

### Training

Behavior cloning (supervised) on expert demonstrations:
- L1 loss on dx/dy (regression), cross-entropy on btn (classification)
- Class weights on button head (no_change dominates)
- AdamW optimizer, cosine LR schedule
- 80/20 train/val split by episode

### Data Format

Each episode is an HDF5 file:
```
episode_NNNN.hdf5
  /observations   (T, 128, 128, 3) uint8
  /actions_dx     (T,) float32
  /actions_dy     (T,) float32
  /actions_btn    (T,) int8
  attrs: {reaction_time_ms, circle_x, circle_y, circle_radius, hit}
```

Recording starts at target onset — idle waiting frames are excluded.

## Success Criteria

| Metric | Target |
|--------|--------|
| CNN mean reaction time | ≤ 2x expert mean |
| Hit rate | > 90% |
| Per-step loop latency (p95) | < 33ms (30 Hz) |

Metrics are decomposed into onset detection latency (target appear to first movement) and movement time (first movement to click).

## Files

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters |
| `env.py` | `ReactiveClicksEnv` — Gym-style environment |
| `expert.py` | Fitts's Law trajectory generator |
| `baseline.py` | Deterministic threshold + centroid controller |
| `generate_data.py` | Run expert, save HDF5 episodes to `data/` |
| `model.py` | `TinyCNN` policy network |
| `train.py` | Behavior cloning training loop |
| `evaluate.py` | Run all agents, report decomposed metrics |
