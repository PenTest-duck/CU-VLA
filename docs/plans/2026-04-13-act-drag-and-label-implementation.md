# ACT Drag-and-Label Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement ACT (Action Chunking with Transformers) for a drag-and-label desktop task, with vision backbone ablation (ResNet18, DINOv2, SigLIP2) and chunk size ablation (5, 10, 20).

**Architecture:** ACT transformer encoder-decoder with CVAE, operating on vision tokens from one of three backbones (pooled to 49 tokens) + proprioception token. Predicts action chunks (dx, dy, click, key) with temporal ensemble at inference. Trained via behavior cloning on scripted expert demonstrations.

**Tech Stack:** PyTorch, torchvision (ResNet18), transformers (DINOv2, SigLIP2), Pygame, h5py, numpy, matplotlib

**Design doc:** `docs/plans/2026-04-13-act-drag-and-label-design.md`

**Reference code:** `experiments/reactive_clicks/` -- follow same patterns (Gym API, HDF5 format, config dataclasses, evaluation structure)

---

### Task 1: Dependencies and Config

**Files:**
- Modify: `pyproject.toml`
- Create: `experiments/act_drag_label/__init__.py`
- Create: `experiments/act_drag_label/config.py`

**Step 1: Add dependencies to pyproject.toml**

Add `torchvision>=0.22` and `transformers>=4.52` to the dependencies list in `pyproject.toml`.

**Step 2: Create config.py with all hyperparameters**

Follow the pattern from `experiments/reactive_clicks/config.py` -- frozen dataclasses, one per concern. Create empty `__init__.py`.

```python
"""All hyperparameters for Experiment 2: ACT Drag-and-Label."""

from dataclasses import dataclass, field


# --- Word vocabulary ---
VOCAB = [
    "CAT", "DOG", "RED", "BOX", "SUN",
    "CUP", "HAT", "PEN", "MAP", "BUS",
    "FAN", "JAR", "KEY", "LOG", "NET",
    "OWL", "RUG", "TOP", "VAN", "WAX",
]

# Key index mapping: 0=no_key, 1=A, 2=B, ..., 26=Z, 27=space (unused but reserved)
NUM_KEY_CLASSES = 28  # no_key + A-Z + space


@dataclass(frozen=True)
class EnvConfig:
    window_size: int = 400
    obs_size: int = 224
    bg_color: tuple[int, int, int] = (30, 30, 30)
    cursor_color: tuple[int, int, int] = (255, 255, 255)
    cursor_radius: int = 3
    shape_width_min: int = 60
    shape_width_max: int = 80
    shape_height: int = 50
    zone_width: int = 90
    zone_height: int = 60
    zone_border_width: int = 3
    font_size: int = 28
    control_hz: int = 30
    # Colors for shapes/zones (RGB)
    shape_colors: tuple[tuple[int, int, int], ...] = (
        (220, 60, 60),    # red
        (60, 120, 220),   # blue
        (60, 180, 80),    # green
    )
    max_shapes: int = 3
    # Layout: shapes on left half, zones on right half
    shape_x_min: int = 30
    shape_x_max: int = 170
    zone_x_min: int = 230
    zone_x_max: int = 340


@dataclass(frozen=True)
class ActionConfig:
    max_delta_px: float = 50.0
    num_key_classes: int = NUM_KEY_CLASSES  # 0=no_key, 1-26=A-Z, 27=space


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256
    encoder_layers: int = 4
    decoder_layers: int = 7
    nheads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    latent_dim: int = 32          # CVAE latent
    num_vision_tokens: int = 49   # all backbones pool to 7x7
    backbone_feature_dims: dict[str, int] = field(default_factory=lambda: {
        "resnet18": 512,
        "dinov2-vits14": 384,
        "siglip2-base": 768,
    })


@dataclass(frozen=True)
class ChunkConfig:
    default_chunk_size: int = 10
    query_frequency: int = 1       # inference every step
    ensemble_decay: float = 0.01   # temporal ensemble exponential decay


@dataclass(frozen=True)
class TrainConfig:
    num_episodes: int = 10000
    batch_size: int = 64
    lr: float = 1e-4
    backbone_lr: float = 1e-5     # ResNet18 only; ViTs are frozen
    weight_decay: float = 1e-4
    epochs: int = 500
    early_stop_patience: int = 50
    val_fraction: float = 0.2
    kl_weight_max: float = 0.1
    kl_anneal_fraction: float = 0.2  # anneal over first 20% of epochs
    loss_weight_click: float = 5.0
    loss_weight_key: float = 5.0
    loss_weight_pad: float = 1.0
    use_amp: bool = True


@dataclass(frozen=True)
class EvalConfig:
    num_episodes: int = 200
    max_steps_per_episode: int = 300   # single shape
    max_steps_multi: int = 900         # 3 shapes


@dataclass(frozen=True)
class ExpertConfig:
    fitts_a: float = 0.05
    fitts_b: float = 0.15
    noise_std: float = 2.0
    pause_min: int = 2       # steps between drop and typing
    pause_max: int = 5
    inter_shape_pause_min: int = 1
    inter_shape_pause_max: int = 3


ENV = EnvConfig()
ACTION = ActionConfig()
MODEL = ModelConfig()
CHUNK = ChunkConfig()
TRAIN = TrainConfig()
EVAL_CFG = EvalConfig()
EXPERT = ExpertConfig()
```

**Step 3: Verify config loads**

```bash
uv run python -c "from experiments.act_drag_label.config import *; print('OK', ENV.obs_size, len(VOCAB))"
```

Expected: `OK 224 20`

**Step 4: Commit**

```bash
git add pyproject.toml experiments/act_drag_label/__init__.py experiments/act_drag_label/config.py
git commit -m "exp2: add config and dependencies for ACT drag-and-label"
```

---

### Task 2: Environment -- Core Rendering and Reset

**Files:**
- Create: `experiments/act_drag_label/env.py`

Build incrementally: rendering + reset first, step logic in Task 3.

**Step 1: Write env.py with DragLabelEnv -- init, reset, rendering**

Follow `experiments/reactive_clicks/env.py` patterns:
- Gym-style API: `reset() -> obs`, `step(action) -> (obs, done, info)`
- Headless (surfarray) + visual (Pygame window) modes
- SDL_VIDEO_HIGHDPI_DISABLED, surfarray read path

Key differences from Exp 1:
- Shapes are rounded rectangles with text labels
- Drop zones are outlined rectangles with matching colors
- Layout: shapes on left, zones on right
- `num_shapes` parameter (1 or 3)
- Font rendering with `pygame.font.SysFont("monospace", font_size, bold=True)`
- Typed text appears in drop zones

The env tracks state per shape:
- `_shapes`: list of dicts {x, y, width, color, label, grabbed, dropped, typed_so_far, complete}
- `_zones`: list of dicts {x, y, color, target_label, typed_text}
- `_cursor_x, _cursor_y`, `_click_state`, `_current_key`, `_grabbed_shape_idx`

Non-overlapping placement: retry random y positions until no overlap (max 100 attempts).

**Step 2: Smoke test rendering**

```bash
uv run python -c "
from experiments.act_drag_label.env import DragLabelEnv
env = DragLabelEnv(visual=False, num_shapes=3)
obs = env.reset(seed=42)
print('Obs shape:', obs.shape, 'dtype:', obs.dtype)
print('Shapes:', [(s['label'], s['color']) for s in env.shapes])
env.close()
print('OK')
"
```

Expected: `Obs shape: (224, 224, 3) dtype: uint8` and 3 shapes.

**Step 3: Visual smoke test**

```bash
uv run python -c "
import pygame, time
from experiments.act_drag_label.env import DragLabelEnv
env = DragLabelEnv(visual=True, num_shapes=3)
obs = env.reset(seed=42)
time.sleep(3)
env.close()
"
```

Visually verify: dark background, 3 colored shapes on left with text, 3 matching zones on right.

**Step 4: Commit**

```bash
git add experiments/act_drag_label/env.py
git commit -m "exp2: add DragLabelEnv with rendering (no step logic yet)"
```

---

### Task 3: Environment -- Step Logic (Drag and Type)

**Files:**
- Modify: `experiments/act_drag_label/env.py`

**Step 1: Add step() method to DragLabelEnv**

The step method handles:
- Apply movement (dx, dy) and clamp to window bounds
- Move grabbed shape with cursor if dragging
- Click transitions: 0->1 grabs shape under cursor, 1->0 drops in zone if color matches
- Key presses: key>0 types character for first dropped-but-not-complete shape
- Wrong character = episode failure
- All shapes complete = episode success
- Timeout check

Also import `EVAL_CFG` at the top.

Key detail: when a shape is dropped in its matching zone, snap it visually above the zone. When the agent types characters, update the zone's `typed_text` for visual feedback.

**Step 2: Test with scripted sequence**

```bash
uv run python -c "
from experiments.act_drag_label.env import DragLabelEnv
env = DragLabelEnv(visual=False, num_shapes=1)
obs = env.reset(seed=42)
shape = env.shapes[0]
zone = [z for z in env.zones if z['color'] == shape['color']][0]
# Move to shape, grab, drag to zone, drop, type label
# (scripted movement sequence -- details in implementation)
print('Step logic test: verify grab, drag, drop, type phases work')
env.close()
"
```

**Step 3: Commit**

```bash
git add experiments/act_drag_label/env.py
git commit -m "exp2: add step logic -- drag, drop, and type handling"
```

---

### Task 4: Expert Policy

**Files:**
- Create: `experiments/act_drag_label/expert.py`

**Step 1: Write expert.py**

Port from `experiments/reactive_clicks/expert.py`. Structure:
- `_fitts_trajectory()`: reusable Fitts's Law movement (accepts `click_held` flag for drag phase)
- `generate_trajectory()`: full episode trajectory for all shapes (nearest-first heuristic)
  - Navigate to shape (Fitts's Law, click=0)
  - mouse_down (grab)
  - Drag to zone (Fitts's Law, click=1 throughout)
  - mouse_up (drop)
  - Pause (2-5 steps, no action)
  - Type label (key_down/key_up pairs for each character)
  - Inter-shape pause if more shapes remain
- `run_episode()`: plays trajectory in env, records observations and actions

**Step 2: Test expert success rate**

```bash
uv run python -c "
from experiments.act_drag_label.env import DragLabelEnv
from experiments.act_drag_label.expert import run_episode
env = DragLabelEnv(visual=False, num_shapes=1)
successes = 0
for i in range(50):
    obs_list, actions, info = run_episode(env, seed=i)
    if info.get('success'): successes += 1
env.close()
print(f'Expert success: {successes}/50')
"
```

Expected: near 100% success rate.

**Step 3: Visual test**

```bash
uv run python -c "
from experiments.act_drag_label.env import DragLabelEnv
from experiments.act_drag_label.expert import run_episode
env = DragLabelEnv(visual=True, num_shapes=1, fps=30)
for i in range(3):
    _, actions, info = run_episode(env, seed=i)
    print(f'Episode {i}: {len(actions)} steps, success={info.get(\"success\", False)}')
env.close()
"
```

**Step 4: Commit**

```bash
git add experiments/act_drag_label/expert.py
git commit -m "exp2: add Fitts's Law expert with drag and typing phases"
```

---

### Task 5: Data Generation

**Files:**
- Create: `experiments/act_drag_label/generate_data.py`

**Step 1: Write generate_data.py**

Follow `experiments/reactive_clicks/generate_data.py` pattern. HDF5 schema:

```
episode_NNNNN.hdf5
  /observations     (T, 224, 224, 3) uint8
  /actions_dx       (T,) float32
  /actions_dy       (T,) float32
  /actions_click    (T,) int8
  /actions_key      (T,) int8
  attrs: {num_shapes, success, num_steps, shapes_completed}
```

CLI: `-n` episodes, `--num-shapes`, `-o` output dir, `-s` seed.

Note: proprioception recording (cursor position per frame) needs to be added to `run_episode` in expert.py. Flag with a TODO during initial implementation -- the cursor position can be reconstructed from cumulative deltas for now.

**Step 2: Generate small test batch and verify**

```bash
uv run python experiments/act_drag_label/generate_data.py -n 100
uv run python -c "
import h5py, os
path = os.path.join('experiments', 'act_drag_label', 'data', 'episode_00000.hdf5')
with h5py.File(path, 'r') as f:
    print('Keys:', list(f.keys()))
    print('Obs shape:', f['observations'].shape)
    print('Click range:', f['actions_click'][:].min(), f['actions_click'][:].max())
    print('Key range:', f['actions_key'][:].min(), f['actions_key'][:].max())
"
```

**Step 3: Commit**

```bash
git add experiments/act_drag_label/generate_data.py
git commit -m "exp2: add data generation script"
```

---

### Task 6: Vision Backbones

**Files:**
- Create: `experiments/act_drag_label/backbones.py`

**Step 1: Write backbones.py**

Three backbone wrappers, all outputting `(batch, 49, d_model)`:

- **ResNet18Backbone**: `torchvision.models.resnet18` with last 2 layers removed. Native 7x7 spatial at 224x224. `Linear(512, 256)` projection. Fine-tuned (requires_grad=True).

- **DINOv2Backbone**: `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`. Frozen. Extract `x_norm_patchtokens` (16x16x384). `AdaptiveAvgPool2d(7,7)`. `Linear(384, 256)` projection.

- **SigLIP2Backbone**: `AutoModel.from_pretrained("google/siglip2-base-patch16-224")`. Frozen. Extract patch tokens from `vision_model`. Pool 14x14 to 7x7. `Linear(768, 256)` projection.

Factory: `build_backbone(name: str) -> nn.Module`

**Step 2: Test each backbone shape**

```bash
uv run python -c "
import torch
from experiments.act_drag_label.backbones import build_backbone
x = torch.randn(2, 3, 224, 224)
bb = build_backbone('resnet18')
out = bb(x)
print(f'ResNet18: {out.shape}')
assert out.shape == (2, 49, 256)
print('OK')
"
```

Test DINOv2 and SigLIP2 separately (they download weights on first run).

**Step 3: Commit**

```bash
git add experiments/act_drag_label/backbones.py
git commit -m "exp2: add vision backbones with 49-token pooling"
```

---

### Task 7: ACT Model

**Files:**
- Create: `experiments/act_drag_label/model.py`

**Step 1: Write model.py**

Components:
1. `SinusoidalPositionEncoding` -- for vision tokens and decoder queries
2. `ACT` class with:
   - Vision backbone (from backbones.py)
   - Proprioception embedding: `Linear(proprio_dim, d_model)` + learned positional embedding
   - CVAE encoder: CLS token + proprio + action tokens -> TransformerEncoder -> mu, logvar -> reparameterize z -> project to d_model
   - Main TransformerEncoder: fuses [vision_tokens, proprio_tok, latent_tok] (51 tokens)
   - TransformerDecoder: chunk_size learned query embeddings cross-attend to encoder memory
   - Action heads: dx (tanh*50), dy (tanh*50), click (sigmoid logit), key (softmax logits), is_pad (sigmoid logit)
3. `count_parameters()` utility

Key config: d_model=256, 4 encoder + 7 decoder layers, 8 heads, ffn=2048, dropout=0.1, latent_dim=32.

At inference (actions=None): z=0 (zeros), no CVAE encoder pass.

**Step 2: Test forward pass**

```bash
uv run python -c "
import torch
from experiments.act_drag_label.model import ACT, count_parameters
model = ACT(backbone_name='resnet18', chunk_size=10)
print(f'Trainable: {count_parameters(model, True):,}')
B = 4
images = torch.randn(B, 3, 224, 224)
proprio = torch.randn(B, 31)
actions = torch.randn(B, 10, 31)
out = model(images, proprio, actions)
print('dx:', out['dx'].shape)
print('key_logits:', out['key_logits'].shape)
out_inf = model(images, proprio)
print('Inference OK')
"
```

Expected: dx (4,10), key_logits (4,10,28), no errors.

**Step 3: Commit**

```bash
git add experiments/act_drag_label/model.py
git commit -m "exp2: add ACT model (CVAE + transformer encoder-decoder)"
```

---

### Task 8: Training Loop

**Files:**
- Create: `experiments/act_drag_label/train.py`

**Step 1: Write train.py**

Key components:
- `ChunkDataset`: loads HDF5 episodes, samples `(obs_t, proprio_t, action_chunk_{t:t+chunk_size})` with stride-1. Zero-pads chunks near episode end with `is_pad=1` mask.
- `train()` function with:
  - Separate optimizer param groups (backbone_lr vs lr)
  - AMP via `torch.amp.GradScaler`
  - KL annealing: `kl_weight = kl_weight_max * min(1, epoch / (epochs * kl_anneal_fraction))`
  - Loss: L1(dx,dy) + weighted BCE(click) + weighted CE(key) + BCE(pad) + kl_weight*KL
  - Pad masking: movement/click/key losses only computed on non-padded positions
  - Early stopping on val loss
  - Saves best.pt and final.pt to checkpoint_dir

CLI: `--backbone`, `--chunk-size`, `--data-dir`, `--checkpoint-dir`, `--device`

Proprioception note: the dataset builds proprio from action state at time t. Initially use placeholder mouse position (0.5, 0.5) -- proper cursor tracking added after end-to-end works.

**Step 2: Test training runs for 1 epoch**

```bash
uv run python experiments/act_drag_label/train.py --backbone resnet18 --chunk-size 5 --device cpu 2>&1 | head -20
```

Requires data from Task 5. Expected: prints loss, no crashes.

**Step 3: Commit**

```bash
git add experiments/act_drag_label/train.py
git commit -m "exp2: add BC training loop with chunk sampling, KL annealing, AMP"
```

---

### Task 9: Baseline CNN

**Files:**
- Create: `experiments/act_drag_label/baseline_cnn.py`

**Step 1: Write baseline_cnn.py**

Extend TinyCNN from Exp 1 for 224x224 input with click + key heads:
- 4 conv layers (stride 2): 224 -> 14x14 spatial
- Flatten -> Linear(128*14*14, 256) -> ReLU
- Heads: dx (tanh*50), dy (tanh*50), click (sigmoid logit), key (softmax 28)

No chunking, no transformer, no proprioception. Single-step reactive baseline.

**Step 2: Test forward pass**

```bash
uv run python -c "
import torch
from experiments.act_drag_label.baseline_cnn import BaselineCNN
model = BaselineCNN()
x = torch.randn(2, 3, 224, 224)
dx, dy, click, key = model(x)
print(f'dx: {dx.shape}, click: {click.shape}, key: {key.shape}')
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
"
```

**Step 3: Commit**

```bash
git add experiments/act_drag_label/baseline_cnn.py
git commit -m "exp2: add BaselineCNN (single-step, no chunking)"
```

---

### Task 10: Evaluation Script

**Files:**
- Create: `experiments/act_drag_label/evaluate.py`

**Step 1: Write evaluate.py**

Components:
- `ACTAgent`: loads model, implements temporal ensemble at inference. Maintains buffer of active chunk predictions. Each step: run inference -> add new chunk to buffer -> blend all active chunks with exp(-0.01 * age) weighting -> age and prune expired chunks.
- `BaselineCNNAgent`: single-step inference wrapper.
- `ExpertAgent`: wraps scripted expert.
- `RandomAgent`: random actions.
- `run_agent()`: runs any agent, tracks phase-decomposed metrics (navigate/drag/type).
- `print_metrics()`: formatted per-phase output.
- CLI: `--backbone`, `--chunk-size`, `--num-shapes`, `--visual`, `--device`, `-n`

Phase tracking in run_agent: monitor click state transitions and key presses to identify navigate (pre-grab), drag (grab to drop), and type (post-drop) phases.

**Step 2: Test evaluation pipeline**

```bash
uv run python experiments/act_drag_label/evaluate.py --backbone resnet18 --chunk-size 10 -n 20
```

**Step 3: Commit**

```bash
git add experiments/act_drag_label/evaluate.py
git commit -m "exp2: add evaluation with temporal ensemble and decomposed metrics"
```

---

### Task 11: End-to-End Integration Test

**Step 1: Generate small dataset**

```bash
uv run python experiments/act_drag_label/generate_data.py -n 200 --seed 0
```

**Step 2: Train briefly**

```bash
uv run python experiments/act_drag_label/train.py --backbone resnet18 --chunk-size 10 --device mps
```

Watch for: no crashes, loss decreasing, memory under 8GB.

**Step 3: Evaluate**

```bash
uv run python experiments/act_drag_label/evaluate.py --backbone resnet18 --chunk-size 10 -n 50
uv run python experiments/act_drag_label/evaluate.py --backbone resnet18 --chunk-size 10 -n 10 --visual
```

**Step 4: Fix any issues and commit**

```bash
git add -A experiments/act_drag_label/
git commit -m "exp2: end-to-end integration fixes"
```

---

### Task 12: Full Training Runs (Experiment Execution)

Not code -- actual experiment runs.

**Phase 1: Vision backbone ablation (chunk_size=10)**

```bash
uv run python experiments/act_drag_label/generate_data.py -n 10000 --seed 0
uv run python experiments/act_drag_label/train.py --backbone resnet18 --chunk-size 10 --device mps
uv run python experiments/act_drag_label/train.py --backbone dinov2-vits14 --chunk-size 10 --device mps
uv run python experiments/act_drag_label/train.py --backbone siglip2-base --chunk-size 10 --device mps
uv run python experiments/act_drag_label/evaluate.py --backbone resnet18 --chunk-size 10
uv run python experiments/act_drag_label/evaluate.py --backbone dinov2-vits14 --chunk-size 10
uv run python experiments/act_drag_label/evaluate.py --backbone siglip2-base --chunk-size 10
```

**Phase 2: Chunk size ablation (best backbone)**

```bash
uv run python experiments/act_drag_label/train.py --backbone <best> --chunk-size 5 --device mps
uv run python experiments/act_drag_label/train.py --backbone <best> --chunk-size 20 --device mps
uv run python experiments/act_drag_label/evaluate.py --backbone <best> --chunk-size 5
uv run python experiments/act_drag_label/evaluate.py --backbone <best> --chunk-size 20
```

**3-shape variant (best config)**

```bash
uv run python experiments/act_drag_label/generate_data.py -n 10000 --num-shapes 3 -o experiments/act_drag_label/data_3shape
uv run python experiments/act_drag_label/train.py --backbone <best> --chunk-size <best> --data-dir experiments/act_drag_label/data_3shape --device mps
uv run python experiments/act_drag_label/evaluate.py --backbone <best> --chunk-size <best> --num-shapes 3
```
