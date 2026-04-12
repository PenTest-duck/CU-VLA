# Experiment 1: Reactive Clicks — Design Document

## Goal

Validate the core visuomotor pointing loop: a tiny CNN observes a 128x128 frame (red circle + cursor on dark gray) and outputs discretized delta mouse movements + button state at 30 Hz, trained via behavior cloning from Fitts's Law scripted demonstrations.

## Scope Limitations

This experiment validates a toy visuomotor pointing loop on synthetic visuals. It does **not** validate:
- General computer use (no real applications, no window chrome, no complex backgrounds)
- Language grounding (no text instructions, no VLM reasoning)
- Multi-step tasks (single target, single click)
- Keyboard input (mouse only)
- Action chunking or temporal abstraction
- Real screen capture pipeline (deferred to a future experiment)

A positive result means: a small neural network can close a perception→action loop at 30 Hz on consumer hardware for a simple pointing task. That's the foundation, not the building.

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Observation method | Gym mode (headless) + visual mode (Pygame window, `surfarray` read) | Both read the same framebuffer; visual mode is for debugging, not screen capture |
| Observation resolution | 128x128 RGB | Balance of detail and speed |
| Action: movement | Delta (Δx, Δy), continuous float pixels with tanh scaling | Relative displacement; L1 regression loss; no discretization needed |
| Action: buttons | 3 classes: no_change, mouse_down, mouse_up | Minimal event-level primitives |
| Action format | Single multi-head output per step (dx, dy, btn) | One forward pass, three outputs, every 33ms |
| Click semantics | Hit/miss decided at mouse_down; episode terminates on first click | Clean evaluation, no compounding error from off-distribution recovery |
| Cursor visibility | Rendered in observation frame | Agent needs to see its own position to decide movement direction |
| Mouse trajectories | Fitts's Law with acceleration/deceleration + Gaussian noise | Realistic human-like training signal |
| Model | Tiny CNN (~2.2M params) | Comfortably meets 30 Hz on M1 CPU |
| Training | Behavior cloning (supervised) from scripted demos | Simple, deterministic, debuggable first step |
| Training data | Recording starts at target onset; idle pre-delay frames excluded | Avoids overwhelming class imbalance from blank waiting frames |
| ML framework | PyTorch (MPS for M1 GPU) | Best ecosystem, VLA code reuse |
| GUI framework | Pygame | Precise frame timing, easy numpy/surfarray extraction |
| Data format | HDF5 chunked files | Standard in VLA research, fast random access |
| Code structure | Flat experiment directory | YAGNI — refactor when experiment 2 shares code |

## Environment

**Test GUI (Pygame, 400x400):**
- Dark gray background (30, 30, 30)
- Red circle (255, 0, 0) appears at random position after 0.5–3.0s random delay
- Circle radius: 20–40px, random per trial
- White cursor dot (3px radius) rendered in frame, controlled by agent's delta actions
- Episode terminates on first mouse_down: hit if cursor inside circle, miss otherwise
- Retina scaling disabled (`SDL_VIDEO_HIGHDPI_DISABLED`) to keep coordinate mapping 1:1

**Two modes, same observation interface:**
- **Gym mode (headless):** renders to an off-screen surface, read via `pygame.surfarray.array3d`, resized to 128x128. No window. Used for data generation, training, and primary evaluation.
- **Visual mode:** opens a Pygame window for human observation/debugging. Same `surfarray` read path — the agent sees identical pixels. Used for visual inspection and demos.

Real screen capture (MSS + Quartz) is deferred to a future experiment that specifically validates the capture pipeline.

**Gym API:** `step(action) → (obs, done, info)` — no reward (training is pure BC).

## Fitts's Law Expert

Generates scripted demonstration trajectories:
- Movement time: `MT = a + b * log2(D/W + 1)` where D = distance to circle center, W = circle diameter
- Trajectory profile: fast acceleration, smooth deceleration, slight Gaussian noise on path
- Issues `mouse_down` when cursor enters circle, then `mouse_up` on the next frame
- Target reaction times: ~150–300ms depending on distance (fast human range)

## Action Space Detail

**Delta movement (Δx, Δy):** continuous float pixels, tanh-scaled to [-50, +50].
- Model outputs raw values through `tanh * max_delta_px`
- Max single-step displacement: 50px at native 400x400 resolution (covers full field within Fitts's Law RT window: 9 steps × 50px = 450px)
- L1 regression loss (robust to outliers, no discretization artifacts)

**Button state:** 3 classes — `0` no_change, `1` mouse_down, `2` mouse_up

## Model

```
Input: 128x128x3 (normalized float32)
  ↓ Conv2d(3, 32, 3x3, stride=2, padding=1)  + ReLU  → 64x64x32
  ↓ Conv2d(32, 64, 3x3, stride=2, padding=1)  + ReLU  → 32x32x64
  ↓ Conv2d(64, 64, 3x3, stride=2, padding=1)  + ReLU  → 16x16x64
  ↓ Conv2d(64, 128, 3x3, stride=2, padding=1) + ReLU  → 8x8x128
  ↓ Flatten → 8192
  ↓ Linear(8192, 256) + ReLU
  ↓ Three heads:
      Linear(256, 1) → tanh → ×50  → dx (pixels)
      Linear(256, 1) → tanh → ×50  → dy (pixels)
      Linear(256, 3)                → btn_logits
```

~2.2M parameters (dominated by the 8192→256 FC layer). Estimated <5ms inference on M1 CPU.

## Training

- **Method:** behavior cloning (supervised learning)
- **Loss:** `L1(dx_pred, dx_true) + L1(dy_pred, dy_true) + CE(btn_pred, btn_true)` with class weights on btn (since `no_change` heavily dominates)
- **Optimizer:** AdamW, lr ~1e-3, cosine decay schedule
- **Data:** HDF5 episodes loaded into PyTorch Dataset with random frame sampling
- **Split:** 80/20 train/val by episode
- **Scale:** ~10000 episodes, ~15-30 steps avg per episode (from target onset to click) ≈ 115K training frames
- **Recording:** starts at target onset, idle pre-delay frames excluded

## Data Format

```
data/
  episode_0000.hdf5
    /observations   (T, 128, 128, 3) uint8
    /actions_dx     (T,) float32    # pixel delta
    /actions_dy     (T,) float32
    /actions_btn    (T,) int8       # 0=no_change, 1=down, 2=up
    attrs: {reaction_time_ms, circle_x, circle_y, circle_radius, hit}
  episode_0001.hdf5
  ...
```

## Evaluation

**Primary eval (Gym mode):** 100+ episodes headless. Fast iteration loop.

**Visual eval:** 100+ episodes with Pygame window visible, same `surfarray` observation path. Full quantitative metrics, not just "does it look right."

### Metrics (decomposed)

| Metric | Definition | Target |
|--------|-----------|--------|
| Onset detection latency | target_appear → first nonzero delta | Report p50/p95 |
| Movement time | first nonzero delta → click | Report p50/p95 |
| Total reaction time | target_appear → click | ≤ 2× expert mean |
| Hit rate | % episodes where click lands on circle | > 90% |
| Per-step loop latency | wall time per observe→infer→act cycle | < 33ms p95 (30 Hz) |
| Gym vs visual gap | RT difference between modes | Report (should be negligible) |

### Diagnostic Outputs

- Reaction time distribution (agent vs expert), broken down by onset + movement
- Sample trajectory visualizations (overlay agent path on environment)
- Per-step inference latency histogram (p50/p95/p99)
- Training loss curves (per-head)

## Baselines

A deterministic color-threshold + centroid controller (no learning) should be built alongside the learned policy as a comparison point:
- Detect red pixels → compute centroid → beeline delta toward centroid → click when overlapping
- Establishes the timing floor for the pipeline
- ~20 lines of code, not a separate experiment phase

## Project Structure

```
experiments/
  reactive_clicks/
    config.py           # All hyperparameters
    env.py              # ReactiveClicksEnv (Gym API + Pygame rendering)
    expert.py           # Fitts's Law trajectory generator
    baseline.py         # Deterministic threshold + centroid controller
    generate_data.py    # Run expert → HDF5 episodes
    model.py            # TinyCNN policy
    train.py            # Behavior cloning training loop
    evaluate.py         # Gym eval + visual eval + metrics
    data/               # HDF5 episodes (gitignored)
    checkpoints/        # Model weights (gitignored)
    results/            # Plots, logs, metrics
```

## Dependencies

`torch`, `pygame`, `h5py`, `numpy`
