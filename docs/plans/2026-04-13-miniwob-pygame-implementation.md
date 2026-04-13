# Experiment 3: MiniWoB-Pygame Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a unified multi-task Pygame environment suite with 12 computer use tasks (Phases 1-3), scripted experts, multi-task ACT training, and evaluation framework.

**Architecture:** Fresh `experiments/miniwob_pygame/` directory. `BaseTaskEnv` renders instruction text in a 40px header bar and provides shared cursor/button/key mechanics. 12 task subclasses implement specific layouts and success criteria. Each task has a paired scripted expert for data generation. Multi-task ACT model (expanded from Exp 2) trained via behavior cloning on mixed expert demonstrations. Instruction is rendered as pixels in the frame -- no explicit task conditioning at ACT stage.

**Tech Stack:** Pygame, PyTorch, Parquet, NumPy, torchvision (backbones), pytest

**Design doc:** `docs/experiments/3-miniwob-pygame.md`

**Key changes from Exp 2:**
- Action: All inputs represented as **held state** (multi-binary), not edge events. The env detects all transitions (0->1 = press, 1->0 = release). This matches the physical reality of input devices: at any instant, each button/key is either pressed or not.
- Mouse: `click` (binary held, edge-detected) -> `mouse_left` (binary held, edge-detected). Same semantics, renamed for clarity. No 3-class button head.
- Keyboard: single `key` class (28 softmax) -> `keys_held` (43 independent binary sigmoids). Each key is an independent held-state toggle. Multiple keys can be held simultaneously (Ctrl+C = ctrl held + c held). Matches VPT's approach for Minecraft (20Hz BC on multi-binary keys).
- Key classes: 28 -> 43 (added enter, backspace, tab, digits 0-9, ctrl, shift, alt). Design doc says 41; we add ctrl/shift/alt as real modifier keys instead of atomic Ctrl+C/V classes.
- Observation: adds `cursor_pos` as separate (2,) float32 normalized to [0,1]
- Proprio dim: 31 -> **46** (2 cursor + 1 mouse_left + 43 keys_held). Uses real cursor_pos, not placeholder (0.5, 0.5).
- Instruction bar: 40px header with rendered text instruction

**Action space philosophy:** "The only assumptions we allow are those set by the laws of physics or physical limitations of a modern conventional computer." No high-level abstractions (click, type), no "one key at a time" constraint, no edge-event encoding. Just the raw held state of every input device at each 30Hz timestep.

**MVP strategy:** Phase 1 tasks (click-target, drag-to-zone, use-slider, type-field) are the first end-to-end pass (generate -> train -> eval). Remaining tasks added incrementally.

**Exp 2 -> Exp 3 migration note:** Exp 2 uses held click (click=1 for entire drag), detecting edges from prev/current state. Exp 3 uses the same held-state approach for mouse_left, so the conversion is straightforward (rename click -> mouse_left). For keys, Exp 2's single key class must be converted to a held-state vector (key=5 for one step -> keys_held[5]=1 for that step, 0 otherwise).

**Class imbalance:** Most keys are 0 most of the time. Mouse is mostly 0 (not pressed). Log per-key and mouse accuracy during training. Add focal loss or per-class weights only if imbalance causes problems. VPT evidence suggests this is manageable with BC.

---

## Directory Structure

```
experiments/miniwob_pygame/
  __init__.py
  config.py                # All hyperparameters
  base_env.py              # BaseTaskEnv
  widgets.py               # TextInput, Slider, ScrollableList, TextBlock
  task_registry.py         # Task name to env class mapping
  tasks/
    __init__.py
    click_target.py      # Phase 1
    drag_to_zone.py
    use_slider.py
    type_field.py
    click_sequence.py    # Phase 2
    draw_path.py
    highlight_text.py
    drag_sort.py
    form_fill.py         # Phase 3
    drag_and_label.py
    scroll_and_click.py
    copy_paste.py
  experts/
    __init__.py
    common.py            # Fitts's Law trajectory, shared utilities
    [one .py per task]   # Each exports generate_trajectory() + run_expert_episode()
  backbones.py             # Copied from Exp 2 (ResNet18, DINOv2, SigLIP2)
  model.py                 # Multi-task ACT (expanded action heads)
  baseline_cnn.py          # Single-step baseline
  generate_data.py         # Multi-task data generation
  train.py                 # Multi-task training loop
  evaluate.py              # Unified evaluation framework
  hf_sync.py               # HF Hub upload/download
tests/miniwob_pygame/
  conftest.py              # Shared fixtures
  test_config.py
  test_base_env.py
  test_widgets.py
  test_tasks.py            # Parametrized across all tasks
  test_experts.py          # Expert completion tests
  test_model.py            # Forward pass shape tests
  test_training_smoke.py   # 1-epoch smoke test
```

---

## Unified Action Space

All inputs are represented as **held state** — binary toggles indicating whether each button/key is currently pressed. The env detects transitions (0->1 = down, 1->0 = up) to trigger events. This matches hardware reality.

```python
# 43 key indices (for the keys_held binary vector)
KEY_A, KEY_B, ..., KEY_Z = 0..25    # 26 letter keys
KEY_SPACE = 26
KEY_ENTER = 27
KEY_BACKSPACE = 28
KEY_TAB = 29
KEY_0, KEY_1, ..., KEY_9 = 30..39   # 10 digit keys
KEY_CTRL = 40                        # modifier keys (held state)
KEY_SHIFT = 41
KEY_ALT = 42

NUM_KEYS = 43

# Action dict format (all tasks)
action = {
    "dx": float,           # cursor delta x, continuous [-max_delta, max_delta]
    "dy": float,           # cursor delta y, continuous [-max_delta, max_delta]
    "mouse_left": int,     # 0=released, 1=pressed (held state)
    "keys_held": list[int], # length-43 binary vector, 1=held 0=released
}

# Examples:
# Idle:        dx=0, dy=0, mouse_left=0, keys_held=[0]*43
# Typing "A":  dx=0, dy=0, mouse_left=0, keys_held=[1,0,0,...] (KEY_A=1)
# Ctrl+C:      dx=0, dy=0, mouse_left=0, keys_held=[..., 0,0,1,0,...,1,0,0]
#              (KEY_C=1 at index 2, KEY_CTRL=1 at index 40)
# Dragging:    dx=5, dy=3, mouse_left=1, keys_held=[0]*43

# Helper functions
def char_to_key_index(ch: str) -> int:
    if "A" <= ch <= "Z": return ord(ch) - ord("A")
    if "a" <= ch <= "z": return ord(ch) - ord("a")
    if "0" <= ch <= "9": return ord(ch) - ord("0") + 30
    if ch == " ": return KEY_SPACE
    raise ValueError(f"Unsupported: {ch!r}")
```

**Proprio vector (46-dim):**
```
[cursor_x, cursor_y, mouse_left, key_0, key_1, ..., key_42]
 |--- 2 float ---| |-- 1 bin --| |------- 43 binary -------|
```

**CVAE action vector (46-dim):**
```
[dx, dy, mouse_left, key_0, key_1, ..., key_42]
 |- 2 float -| |-- 1 bin --| |--- 43 binary ---|
```

**Model output heads (per chunk timestep):**
- `dx`: 1 regression (tanh * max_delta)
- `dy`: 1 regression (tanh * max_delta)
- `mouse_left`: 1 sigmoid (BCE loss)
- `keys_held`: 43 sigmoids (43 independent BCE losses)
- `pad`: 1 sigmoid (BCE loss)

**Env edge detection:**
```python
# In BaseTaskEnv.step():
prev_mouse = self._mouse_pressed
self._mouse_pressed = bool(action["mouse_left"])
if not prev_mouse and self._mouse_pressed:
    self._handle_mouse_down()
if prev_mouse and not self._mouse_pressed:
    self._handle_mouse_up()

# For keys: compare prev_keys_held vs current keys_held
for i, (prev, curr) in enumerate(zip(self._prev_keys, action["keys_held"])):
    if prev == 0 and curr == 1:
        self._handle_key_down(i)
    elif prev == 1 and curr == 0:
        self._handle_key_up(i)
self._prev_keys = list(action["keys_held"])
```

---

## Task 1: Project Scaffold + Config

**Files:**
- Create: `experiments/miniwob_pygame/__init__.py`
- Create: `experiments/miniwob_pygame/config.py`
- Create: `tests/miniwob_pygame/__init__.py`
- Create: `tests/miniwob_pygame/conftest.py`
- Create: `tests/miniwob_pygame/test_config.py`

**Step 1: Write test for config constants**

```python
# tests/miniwob_pygame/test_config.py
import pytest

def test_num_keys():
    from experiments.miniwob_pygame.config import NUM_KEYS
    assert NUM_KEYS == 43

def test_key_indices():
    from experiments.miniwob_pygame.config import (
        KEY_A, KEY_Z, KEY_SPACE, KEY_ENTER,
        KEY_BACKSPACE, KEY_TAB, KEY_0, KEY_9,
        KEY_CTRL, KEY_SHIFT, KEY_ALT,
    )
    assert KEY_A == 0
    assert KEY_Z == 25
    assert KEY_SPACE == 26
    assert KEY_ENTER == 27
    assert KEY_BACKSPACE == 28
    assert KEY_TAB == 29
    assert KEY_0 == 30
    assert KEY_9 == 39
    assert KEY_CTRL == 40
    assert KEY_SHIFT == 41
    assert KEY_ALT == 42

def test_env_config_defaults():
    from experiments.miniwob_pygame.config import ENV
    assert ENV.window_size == 400
    assert ENV.obs_size == 224
    assert ENV.instruction_bar_height == 40
    assert ENV.control_hz == 30

def test_action_config():
    from experiments.miniwob_pygame.config import ACTION
    assert ACTION.max_delta_px == 50.0
    assert ACTION.num_keys == 43

def test_proprio_dim():
    """Proprio = 2 (cursor) + 1 (mouse_left) + 43 (keys_held) = 46."""
    from experiments.miniwob_pygame.config import ACTION
    expected = 2 + 1 + ACTION.num_keys  # 46
    assert expected == 46

def test_task_names():
    from experiments.miniwob_pygame.config import TASK_NAMES
    assert len(TASK_NAMES) == 12
    assert "click-target" in TASK_NAMES
    assert "copy-paste" in TASK_NAMES
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/miniwob_pygame/test_config.py -v`
Expected: FAIL (module not found)

**Step 3: Implement config.py**

```python
# experiments/miniwob_pygame/config.py
"""All hyperparameters for Experiment 3: MiniWoB-Pygame."""

from dataclasses import dataclass, field

# --- Key indices (for keys_held binary vector) ---
# A-Z = 0..25
KEY_A, KEY_B, KEY_C, KEY_D, KEY_E = 0, 1, 2, 3, 4
KEY_F, KEY_G, KEY_H, KEY_I, KEY_J = 5, 6, 7, 8, 9
KEY_K, KEY_L, KEY_M, KEY_N, KEY_O = 10, 11, 12, 13, 14
KEY_P, KEY_Q, KEY_R, KEY_S, KEY_T = 15, 16, 17, 18, 19
KEY_U, KEY_V, KEY_W, KEY_X, KEY_Y, KEY_Z = 20, 21, 22, 23, 24, 25
KEY_SPACE = 26
KEY_ENTER = 27
KEY_BACKSPACE = 28
KEY_TAB = 29
# Digits 0-9 = 30..39
KEY_0, KEY_1, KEY_2, KEY_3, KEY_4 = 30, 31, 32, 33, 34
KEY_5, KEY_6, KEY_7, KEY_8, KEY_9 = 35, 36, 37, 38, 39
# Modifier keys (held state)
KEY_CTRL = 40
KEY_SHIFT = 41
KEY_ALT = 42

NUM_KEYS = 43

# --- Key helpers ---
def char_to_key_index(ch: str) -> int:
    """Convert a character to its key index in keys_held vector."""
    if "A" <= ch <= "Z":
        return ord(ch) - ord("A")
    if "a" <= ch <= "z":
        return ord(ch) - ord("a")  # treat as uppercase
    if "0" <= ch <= "9":
        return ord(ch) - ord("0") + KEY_0
    if ch == " ":
        return KEY_SPACE
    raise ValueError(f"Unsupported character: {ch!r}")

def key_index_to_char(idx: int) -> str | None:
    """Convert key index to character, or None for non-printable."""
    if 0 <= idx <= 25:
        return chr(ord("A") + idx)
    if KEY_0 <= idx <= KEY_9:
        return chr(ord("0") + idx - KEY_0)
    if idx == KEY_SPACE:
        return " "
    return None

# --- Task registry ---
TASK_NAMES = [
    # Phase 1
    "click-target",
    "drag-to-zone",
    "use-slider",
    "type-field",
    # Phase 2
    "click-sequence",
    "draw-path",
    "highlight-text",
    "drag-sort",
    # Phase 3
    "form-fill",
    "drag-and-label",
    "scroll-and-click",
    "copy-paste",
]

# --- Vocab for labeling tasks ---
VOCAB = [
    "CAT", "DOG", "RED", "BOX", "SUN",
    "CUP", "HAT", "PEN", "MAP", "BUS",
    "FAN", "JAR", "KEY", "LOG", "NET",
    "OWL", "RUG", "TOP", "VAN", "WAX",
]

# --- Config dataclasses ---

@dataclass(frozen=True)
class EnvConfig:
    window_size: int = 400
    obs_size: int = 224
    bg_color: tuple[int, int, int] = (30, 30, 30)
    cursor_color: tuple[int, int, int] = (255, 255, 255)
    cursor_radius: int = 3
    instruction_bar_height: int = 40
    instruction_bg_color: tuple[int, int, int] = (50, 50, 50)
    instruction_font_size: int = 24
    font_size: int = 28
    control_hz: int = 30
    shape_colors: tuple[tuple[int, int, int], ...] = (
        (220, 60, 60),    # red
        (60, 120, 220),   # blue
        (60, 180, 80),    # green
        (240, 180, 40),   # yellow
    )

@dataclass(frozen=True)
class ActionConfig:
    max_delta_px: float = 50.0
    num_keys: int = NUM_KEYS

@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256
    encoder_layers: int = 4
    decoder_layers: int = 7
    nheads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    latent_dim: int = 32
    num_vision_tokens: int = 49
    proprio_dim: int = 46  # 2 cursor + 1 mouse_left + 43 keys_held
    backbone_feature_dims: dict[str, int] = field(default_factory=lambda: {
        "resnet18": 512,
        "dinov2-vits14": 384,
        "siglip2-base": 768,
    })

@dataclass(frozen=True)
class ChunkConfig:
    default_chunk_size: int = 10
    query_frequency: int = 1
    ensemble_decay: float = 0.01

@dataclass(frozen=True)
class TrainConfig:
    num_episodes_per_task: int = 5000
    batch_size: int = 64
    lr: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4
    epochs: int = 500
    early_stop_patience: int = 50
    val_fraction: float = 0.2
    kl_weight_max: float = 0.1
    kl_anneal_fraction: float = 0.2
    loss_weight_button: float = 5.0
    loss_weight_key: float = 5.0
    loss_weight_pad: float = 1.0
    use_amp: bool = True

@dataclass(frozen=True)
class EvalConfig:
    num_episodes: int = 200
    max_steps_per_episode: int = 300
    max_steps_multi: int = 600   # for multi-step tasks (drag-sort, form-fill, etc.)
    max_steps_long: int = 900    # for long tasks (copy-paste, scroll-and-click)

@dataclass(frozen=True)
class ExpertConfig:
    fitts_a: float = 0.05
    fitts_b: float = 0.15
    noise_std: float = 2.0
    pause_min: int = 2
    pause_max: int = 5

ENV = EnvConfig()
ACTION = ActionConfig()
MODEL = ModelConfig()
CHUNK = ChunkConfig()
TRAIN = TrainConfig()
EVAL_CFG = EvalConfig()
EXPERT = ExpertConfig()
```

**Step 4: Create __init__.py and conftest.py**

```python
# experiments/miniwob_pygame/__init__.py
# (empty)

# tests/miniwob_pygame/__init__.py
# (empty)

# tests/miniwob_pygame/conftest.py
"""Shared test fixtures for MiniWoB-Pygame tests."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
```

**Step 5: Run tests, verify pass**

Run: `uv run pytest tests/miniwob_pygame/test_config.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add experiments/miniwob_pygame/__init__.py experiments/miniwob_pygame/config.py \
      tests/miniwob_pygame/__init__.py tests/miniwob_pygame/conftest.py \
      tests/miniwob_pygame/test_config.py
git commit -m "exp3: add project scaffold and config module"
```

---

## Task 2: BaseTaskEnv

**Files:**
- Create: `experiments/miniwob_pygame/base_env.py`
- Create: `tests/miniwob_pygame/test_base_env.py`

**Step 1: Write test for BaseTaskEnv**

```python
# tests/miniwob_pygame/test_base_env.py
import numpy as np
import pytest

from experiments.miniwob_pygame.config import ENV, NUM_KEYS
from experiments.miniwob_pygame.base_env import BaseTaskEnv


class ConcreteTask(BaseTaskEnv):
    """Minimal concrete subclass for testing."""
    task_name = "test-task"

    def _setup_task(self, rng):
        self.task_instruction = "Test instruction"  # Set dynamically
        self._target_x = 200
        self._target_y = 200

    def _check_success(self) -> tuple[bool, dict]:
        return False, {}


class TestBaseTaskEnv:
    def test_reset_returns_obs_dict(self):
        env = ConcreteTask()
        obs = env.reset(seed=0)
        assert "screenshot" in obs
        assert "cursor_pos" in obs
        assert obs["screenshot"].shape == (ENV.obs_size, ENV.obs_size, 3)
        assert obs["screenshot"].dtype == np.uint8
        assert obs["cursor_pos"].shape == (2,)
        assert obs["cursor_pos"].dtype == np.float32
        # cursor_pos should be normalized [0, 1]
        assert 0.0 <= obs["cursor_pos"][0] <= 1.0
        assert 0.0 <= obs["cursor_pos"][1] <= 1.0
        env.close()

    def _noop(self, **overrides):
        """Helper: default idle action with overrides."""
        action = {
            "dx": 0.0, "dy": 0.0,
            "mouse_left": 0, "keys_held": [0] * NUM_KEYS,
        }
        action.update(overrides)
        return action

    def test_step_applies_cursor_delta(self):
        env = ConcreteTask()
        env.reset(seed=42)
        start_x, start_y = env._cursor_x, env._cursor_y
        obs, done, info = env.step(self._noop(dx=10.0, dy=-5.0))
        expected_x = np.clip(start_x + 10.0, 0, ENV.window_size - 1)
        expected_y = np.clip(start_y - 5.0, 0, ENV.window_size - 1)
        assert abs(env._cursor_x - expected_x) < 0.01
        assert abs(env._cursor_y - expected_y) < 0.01
        env.close()

    def test_step_clamps_delta(self):
        env = ConcreteTask()
        env.reset(seed=0)
        env._cursor_x = 200.0
        env._cursor_y = 200.0
        env.step(self._noop(dx=999.0, dy=-999.0))
        assert env._cursor_x == 250.0
        assert env._cursor_y == 150.0
        env.close()

    def test_step_clamps_cursor_to_window(self):
        env = ConcreteTask()
        env.reset(seed=0)
        env._cursor_x = 5.0
        env._cursor_y = 5.0
        env.step(self._noop(dx=-50.0, dy=-50.0))
        assert env._cursor_x == 0.0
        assert env._cursor_y == 0.0
        env.close()

    def test_mouse_held_state_tracking(self):
        env = ConcreteTask()
        env.reset(seed=0)
        assert env._mouse_pressed is False
        env.step(self._noop(mouse_left=1))
        assert env._mouse_pressed is True
        env.step(self._noop(mouse_left=1))  # still held
        assert env._mouse_pressed is True
        env.step(self._noop(mouse_left=0))
        assert env._mouse_pressed is False
        env.close()

    def test_key_held_state_tracking(self):
        from experiments.miniwob_pygame.config import KEY_A, KEY_CTRL
        env = ConcreteTask()
        env.reset(seed=0)
        # Press A + Ctrl simultaneously
        keys = [0] * NUM_KEYS
        keys[KEY_A] = 1
        keys[KEY_CTRL] = 1
        env.step(self._noop(keys_held=keys))
        assert env._keys_held[KEY_A] == 1
        assert env._keys_held[KEY_CTRL] == 1
        # Release A, keep Ctrl
        keys[KEY_A] = 0
        env.step(self._noop(keys_held=keys))
        assert env._keys_held[KEY_A] == 0
        assert env._keys_held[KEY_CTRL] == 1
        env.close()

    def test_instruction_rendered_in_header(self):
        """Instruction bar occupies top 40px with distinct bg color."""
        env = ConcreteTask()
        obs = env.reset(seed=0)
        screenshot = obs["screenshot"]
        # Scale factor: 224/400 = 0.56, so 40px header -> ~22px in obs
        header_rows = int(ENV.instruction_bar_height * ENV.obs_size / ENV.window_size)
        # Check that the header bar has the instruction_bg_color (50,50,50)
        # not the main bg_color (30,30,30). Sample a pixel in the header
        # away from text (left edge, middle of header bar).
        header_pixel = screenshot[header_rows // 2, 5, :]
        # Should be close to instruction_bg_color, not main bg_color
        assert header_pixel[0] > 40, f"Header pixel too dark: {header_pixel}"
        env.close()

    def test_step_after_done_raises(self):
        env = ConcreteTask()
        env.reset(seed=0)
        env._done = True
        with pytest.raises(AssertionError):
            env.step({"dx": 0, "dy": 0, "button": 0, "key": 0})
        env.close()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/miniwob_pygame/test_base_env.py -v`
Expected: FAIL (import error)

**Step 3: Implement BaseTaskEnv**

```python
# experiments/miniwob_pygame/base_env.py
"""Base environment for all MiniWoB-Pygame tasks.

Provides shared Pygame rendering, cursor mechanics, button/key state tracking,
instruction bar rendering, and observation construction. Subclasses implement
_setup_task() and _check_success().
"""

import os
from abc import ABC, abstractmethod

import numpy as np
import pygame
import pygame._freetype as _ft

from .config import ENV, ACTION, EVAL_CFG, NUM_KEYS

os.environ.setdefault("SDL_VIDEO_HIGHDPI_DISABLED", "1")


class BaseTaskEnv(ABC):
    """Abstract base for all MiniWoB-Pygame task environments.

    Subclasses MUST define:
        task_name: str          -- unique task identifier
        task_instruction: str   -- text rendered in header bar
        _setup_task(rng)        -- place task-specific elements
        _check_success()        -- return (done, info) based on task state
    Subclasses MAY override:
        _render_task(surface)   -- draw task-specific elements
        _handle_button_down()   -- respond to mouse_down event
        _handle_button_up()     -- respond to mouse_up event
        _handle_drag()          -- called every step while mouse held
        _handle_key(key: int)   -- respond to key press
        _get_max_steps()        -- override default max steps
    """

    task_name: str = ""

    def __init__(self, visual: bool = False, fps: int | None = None):
        self.visual = visual
        self.fps = fps if fps is not None else (ENV.control_hz if visual else 0)
        self.task_instruction: str = ""  # Set dynamically in _setup_task()

        self._pygame_initialized = False
        self._surface: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: _ft.Font | None = None
        self._instruction_font: _ft.Font | None = None

        # Episode state
        self._cursor_x: float = 0.0
        self._cursor_y: float = 0.0
        self._mouse_pressed: bool = False
        self._keys_held: list[int] = [0] * NUM_KEYS
        self._step_count: int = 0
        self._done: bool = False
        self._rng: np.random.Generator = np.random.default_rng()

    def _init_pygame(self) -> None:
        if self._pygame_initialized:
            return
        if self.visual:
            pygame.init()
            self._surface = pygame.display.set_mode(
                (ENV.window_size, ENV.window_size)
            )
            pygame.display.set_caption(f"MiniWoB: {self.task_name}")
        else:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.init()
            self._surface = pygame.Surface((ENV.window_size, ENV.window_size))
        self._clock = pygame.time.Clock()
        _ft.init()
        self._font = _ft.Font(None, ENV.font_size)
        self._font.strong = True
        self._instruction_font = _ft.Font(None, ENV.instruction_font_size)
        self._instruction_font.strong = True
        self._pygame_initialized = True

    def reset(self, seed: int | None = None) -> dict:
        """Reset environment for a new episode.

        Returns:
            Observation dict with 'screenshot' and 'cursor_pos'.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._init_pygame()

        # Random cursor start in task area (below instruction bar)
        task_top = ENV.instruction_bar_height
        self._cursor_x = float(self._rng.integers(0, ENV.window_size))
        self._cursor_y = float(self._rng.integers(task_top, ENV.window_size))

        self._mouse_pressed = False
        self._keys_held = [0] * NUM_KEYS
        self._step_count = 0
        self._done = False

        self._setup_task(self._rng)

        return self._get_observation()

    def step(self, action: dict) -> tuple[dict, bool, dict]:
        """Advance environment by one timestep.

        Canonical order: (1) move cursor, (2) apply mouse, (3) apply keys.
        All inputs are held state; env detects transitions.

        Args:
            action: dict with 'dx', 'dy' (float), 'mouse_left' (int 0/1),
                    'keys_held' (list[int] length NUM_KEYS).

        Returns:
            (observation, done, info) tuple.
        """
        assert not self._done, "Episode already done. Call reset()."

        info: dict = {}
        max_d = ACTION.max_delta_px

        # Parse action
        dx = float(np.clip(action["dx"], -max_d, max_d))
        dy = float(np.clip(action["dy"], -max_d, max_d))
        mouse_left = bool(action["mouse_left"])
        keys_held = list(action["keys_held"])

        self._step_count += 1

        # 1. Move cursor
        self._cursor_x = float(
            np.clip(self._cursor_x + dx, 0, ENV.window_size - 1)
        )
        self._cursor_y = float(
            np.clip(self._cursor_y + dy, 0, ENV.window_size - 1)
        )

        # 2. Mouse edge detection
        prev_mouse = self._mouse_pressed
        self._mouse_pressed = mouse_left
        if not prev_mouse and mouse_left:
            self._handle_mouse_down()
        elif prev_mouse and not mouse_left:
            self._handle_mouse_up()
        if self._mouse_pressed:
            self._handle_drag()

        # 3. Key edge detection
        for i in range(NUM_KEYS):
            prev = self._keys_held[i]
            curr = keys_held[i]
            if prev == 0 and curr == 1:
                self._handle_key_down(i)
            elif prev == 1 and curr == 0:
                self._handle_key_up(i)
        self._keys_held = keys_held

        # 4. Check success/failure
        if not self._done:
            done, task_info = self._check_success()
            info.update(task_info)
            if done:
                self._done = True

        # 5. Timeout
        max_steps = self._get_max_steps()
        if not self._done and self._step_count >= max_steps:
            self._done = True
            info["timeout"] = True

        # 6. Terminal info
        if self._done:
            info["steps"] = self._step_count

        # 7. Render + handle pygame events
        obs = self._get_observation()
        if self.visual:
            self._clock.tick(self.fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._done = True

        return obs, self._done, info

    # --- Subclass hooks (override as needed) ---

    @abstractmethod
    def _setup_task(self, rng: np.random.Generator) -> None:
        """Initialize task-specific state (shapes, targets, etc.)."""

    @abstractmethod
    def _check_success(self) -> tuple[bool, dict]:
        """Check if the task is complete. Return (done, info_dict)."""

    def _render_task(self, surface: pygame.Surface) -> None:
        """Draw task-specific elements. Override in subclass."""

    def _handle_mouse_down(self) -> None:
        """Called on mouse 0->1 transition. Override in subclass."""

    def _handle_mouse_up(self) -> None:
        """Called on mouse 1->0 transition. Override in subclass."""

    def _handle_drag(self) -> None:
        """Called every step while mouse is held. Override in subclass."""

    def _handle_key_down(self, key_index: int) -> None:
        """Called when key transitions 0->1. Override in subclass."""

    def _handle_key_up(self, key_index: int) -> None:
        """Called when key transitions 1->0. Override in subclass."""

    def _get_max_steps(self) -> int:
        """Override to set task-specific step limit."""
        return EVAL_CFG.max_steps_per_episode

    # --- Rendering ---

    def _render_instruction(self, surface: pygame.Surface) -> None:
        """Render instruction text in header bar."""
        bar_rect = pygame.Rect(
            0, 0, ENV.window_size, ENV.instruction_bar_height
        )
        pygame.draw.rect(surface, ENV.instruction_bg_color, bar_rect)
        if self.task_instruction:
            text_surf, text_rect = self._instruction_font.render(
                self.task_instruction, (255, 255, 255)
            )
            text_rect.center = bar_rect.center
            surface.blit(text_surf, text_rect)

    def _render_cursor(self, surface: pygame.Surface) -> None:
        pygame.draw.circle(
            surface,
            ENV.cursor_color,
            (int(round(self._cursor_x)), int(round(self._cursor_y))),
            ENV.cursor_radius,
        )

    def _get_observation(self) -> dict:
        """Render frame and return observation dict."""
        surface = self._surface

        # Background
        surface.fill(ENV.bg_color)

        # Instruction bar
        self._render_instruction(surface)

        # Task-specific rendering
        self._render_task(surface)

        # Cursor
        self._render_cursor(surface)

        if self.visual:
            pygame.display.flip()

        # Read pixels
        pixels = pygame.surfarray.array3d(surface).transpose(1, 0, 2)

        # Resize to obs_size
        if pixels.shape[0] != ENV.obs_size or pixels.shape[1] != ENV.obs_size:
            obs_surface = pygame.transform.scale(
                surface, (ENV.obs_size, ENV.obs_size)
            )
            pixels = pygame.surfarray.array3d(obs_surface).transpose(1, 0, 2)

        # Cursor position normalized to [0, 1]
        cursor_pos = np.array([
            self._cursor_x / (ENV.window_size - 1),
            self._cursor_y / (ENV.window_size - 1),
        ], dtype=np.float32)

        return {
            "screenshot": pixels.copy(),
            "cursor_pos": cursor_pos,
        }

    # --- Utilities ---

    @property
    def cursor_pos(self) -> tuple[float, float]:
        return (self._cursor_x, self._cursor_y)

    def close(self) -> None:
        if self._pygame_initialized:
            pygame.quit()
            self._pygame_initialized = False
```

**Step 4: Run tests, verify pass**

Run: `uv run pytest tests/miniwob_pygame/test_base_env.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add experiments/miniwob_pygame/base_env.py tests/miniwob_pygame/test_base_env.py
git commit -m "exp3: add BaseTaskEnv with instruction rendering and cursor mechanics"
```

---

## Task 3: Shared Widgets

**Files:**
- Create: `experiments/miniwob_pygame/widgets.py`
- Create: `tests/miniwob_pygame/test_widgets.py`

Reusable Pygame widgets that multiple tasks share. Each widget knows how to render itself and respond to events.

**Widgets to implement:**

1. **TextInput** -- clickable text field with cursor, accepts typed characters. Used by: type-field, form-fill, drag-and-label, copy-paste.
2. **Slider** -- horizontal track with draggable handle. Used by: use-slider.
3. **ScrollableList** -- list of items taller than viewport, with scrollbar. Used by: scroll-and-click.
4. **TextBlock** -- rendered paragraph of selectable text. Used by: highlight-text, copy-paste.

**Step 1: Write tests for TextInput widget**

```python
# tests/miniwob_pygame/test_widgets.py
import pytest
from experiments.miniwob_pygame.widgets import TextInput
from experiments.miniwob_pygame.config import KEY_A, KEY_B, KEY_BACKSPACE


class TestTextInput:
    def test_initial_state(self):
        ti = TextInput(x=10, y=10, width=100, height=30)
        assert ti.text == ""
        assert ti.focused is False

    def test_click_focuses(self):
        ti = TextInput(x=10, y=10, width=100, height=30)
        assert not ti.focused
        ti.handle_click(50, 20)  # inside
        assert ti.focused

    def test_click_outside_unfocuses(self):
        ti = TextInput(x=10, y=10, width=100, height=30)
        ti.focused = True
        ti.handle_click(200, 200)  # outside
        assert not ti.focused

    def test_type_character(self):
        ti = TextInput(x=10, y=10, width=100, height=30)
        ti.focused = True
        ti.handle_key(KEY_A)
        assert ti.text == "A"
        ti.handle_key(KEY_B)
        assert ti.text == "AB"

    def test_type_when_unfocused_ignored(self):
        ti = TextInput(x=10, y=10, width=100, height=30)
        ti.handle_key(KEY_A)
        assert ti.text == ""

    def test_backspace(self):
        ti = TextInput(x=10, y=10, width=100, height=30)
        ti.focused = True
        ti.handle_key(KEY_A)
        ti.handle_key(KEY_B)
        ti.handle_key(KEY_BACKSPACE)
        assert ti.text == "A"

    def test_backspace_empty_noop(self):
        ti = TextInput(x=10, y=10, width=100, height=30)
        ti.focused = True
        ti.handle_key(KEY_BACKSPACE)  # should not crash
        assert ti.text == ""

    def test_contains(self):
        ti = TextInput(x=10, y=10, width=100, height=30)
        assert ti.contains(50, 20)
        assert not ti.contains(200, 200)
        assert ti.contains(10, 10)  # edge
        assert not ti.contains(111, 20)  # just outside
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/miniwob_pygame/test_widgets.py -v`

**Step 3: Implement widgets.py**

Implement `TextInput`, `Slider`, `ScrollableList`, and `TextBlock`. Each widget has:
- `__init__` with position/size
- `render(surface, font)` -- draw to surface
- `contains(x, y)` -- hit test
- Event handlers (`handle_click`, `handle_key`, `handle_drag`)

Key implementation notes:
- **Slider:** `handle_drag(x, y)` updates value based on x position along track. `value` property returns float in `[min_val, max_val]`.
- **ScrollableList:** `handle_drag(x, y)` scrolls if dragging the scrollbar. `visible_items()` returns items in current viewport.
- **TextBlock:** Stores list of `(char, x, y, width)` for hit testing highlight start/end positions.

**Step 4: Run tests, verify pass**

Run: `uv run pytest tests/miniwob_pygame/test_widgets.py -v`

**Step 5: Commit**

```bash
git add experiments/miniwob_pygame/widgets.py tests/miniwob_pygame/test_widgets.py
git commit -m "exp3: add reusable Pygame widgets (TextInput, Slider, ScrollableList, TextBlock)"
```

---

## Task 4: click-target (Phase 1a)

**Files:**
- Create: `experiments/miniwob_pygame/tasks/__init__.py`
- Create: `experiments/miniwob_pygame/tasks/click_target.py`
- Create: `experiments/miniwob_pygame/experts/__init__.py`
- Create: `experiments/miniwob_pygame/experts/common.py`
- Create: `experiments/miniwob_pygame/experts/click_target.py`
- Create: `tests/miniwob_pygame/test_tasks.py`
- Create: `tests/miniwob_pygame/test_experts.py`

**Step 1: Write test for ClickTargetEnv**

```python
# tests/miniwob_pygame/test_tasks.py
"""Parametrized tests for all task environments."""
import numpy as np
import pytest

from experiments.miniwob_pygame.config import ENV, NUM_KEYS

def _noop(**overrides):
    action = {"dx": 0.0, "dy": 0.0, "mouse_left": 0, "keys_held": [0] * NUM_KEYS}
    action.update(overrides)
    return action


class TestClickTarget:
    def test_reset_places_target(self):
        from experiments.miniwob_pygame.tasks.click_target import ClickTargetEnv
        env = ClickTargetEnv()
        obs = env.reset(seed=0)
        assert hasattr(env, "_target")
        assert "x" in env._target and "y" in env._target
        env.close()

    def test_click_on_target_succeeds(self):
        from experiments.miniwob_pygame.tasks.click_target import ClickTargetEnv
        env = ClickTargetEnv()
        env.reset(seed=0)
        # Teleport cursor to target center
        t = env._target
        env._cursor_x = float(t["x"] + t["width"] // 2)
        env._cursor_y = float(t["y"] + t["height"] // 2)
        # Press mouse (held state = 1), then release (held state = 0)
        env.step(_noop(mouse_left=1))
        _, done, info = env.step(_noop(mouse_left=0))
        assert done
        assert info.get("success") is True
        env.close()

    def test_click_off_target_does_not_succeed(self):
        from experiments.miniwob_pygame.tasks.click_target import ClickTargetEnv
        env = ClickTargetEnv()
        env.reset(seed=0)
        env._cursor_x = 0.0
        env._cursor_y = 0.0
        env.step(_noop(mouse_left=1))
        _, done, info = env.step(_noop(mouse_left=0))
        assert not done
        env.close()

    def test_timeout(self):
        from experiments.miniwob_pygame.tasks.click_target import ClickTargetEnv
        env = ClickTargetEnv()
        env.reset(seed=0)
        for _ in range(env._get_max_steps()):
            _, done, info = env.step(_noop())
            if done:
                break
        assert done
        assert info.get("timeout") is True
        env.close()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/miniwob_pygame/test_tasks.py::TestClickTarget -v`

**Step 3: Implement ClickTargetEnv**

Task logic:
- `_setup_task`: Place a colored shape at random position. Optionally add distractors.
- `_handle_button_down`: Record click start position.
- `_check_success`: On button_up, check if cursor is on target. Success if yes.
- `_render_task`: Draw target shape (and distractors).
- `task_instruction`: "Click the {COLOR} shape"

**Step 4: Implement expert common utilities**

```python
# experiments/miniwob_pygame/experts/common.py
# Contains:
# - noop_action() -> dict
# - fitts_trajectory(cx, cy, tx, ty, target_width, rng, button) -> list[dict]
# - pause_actions(rng) -> list[dict]
# - simulate_cursor(actions, start_x, start_y) -> (float, float)
# - run_episode(env, trajectory) -> (observations, actions, final_info)
```

Fitts's Law trajectory is identical to Exp 2's `_fitts_trajectory()` but uses `button` instead of `click` key in action dicts.

**Step 5: Implement click_target expert**

Navigate to target center -> mouse_down -> mouse_up.

**Step 6: Write expert completion test**

```python
# tests/miniwob_pygame/test_experts.py
class TestClickTargetExpert:
    def test_expert_completes_task(self):
        import numpy as np
        from experiments.miniwob_pygame.tasks.click_target import ClickTargetEnv
        from experiments.miniwob_pygame.experts.click_target import (
            generate_trajectory,
        )
        from experiments.miniwob_pygame.experts.common import run_episode

        env = ClickTargetEnv()
        successes = 0
        for seed in range(20):
            env.reset(seed=seed)
            traj = generate_trajectory(
                *env.cursor_pos, env._target, np.random.default_rng(seed)
            )
            _, _, info = run_episode(env, traj)
            if info.get("success"):
                successes += 1
        assert successes >= 18, f"Expert only succeeded {successes}/20"
        env.close()
```

**Step 7: Run all tests, verify pass**

Run: `uv run pytest tests/miniwob_pygame/ -v`

**Step 8: Commit**

```bash
git add experiments/miniwob_pygame/tasks/ experiments/miniwob_pygame/experts/ \
      tests/miniwob_pygame/test_tasks.py tests/miniwob_pygame/test_experts.py
git commit -m "exp3: add click-target task + expert + shared expert utilities"
```

---

## Task 5: drag-to-zone (Phase 1b)

Follows the same pattern as Task 4. Key differences:

**Files:**
- Create: `experiments/miniwob_pygame/tasks/drag_to_zone.py`
- Create: `experiments/miniwob_pygame/experts/drag_to_zone.py`
- Modify: `tests/miniwob_pygame/test_tasks.py` -- add `TestDragToZone`
- Modify: `tests/miniwob_pygame/test_experts.py` -- add expert completion test

**Task logic (drag_to_zone.py):**
- `_setup_task`: Place 1-3 colored shapes on left, matching zones on right (reuse Exp 2 layout logic)
- `_handle_button_down`: If cursor on un-grabbed shape, grab it
- `_handle_drag`: Move grabbed shape with cursor
- `_handle_button_up`: If cursor on matching zone, drop and snap shape
- `_check_success`: All shapes dropped in matching zones
- `task_instruction`: "Drag the {color} shape to the {color} zone" (single) or "Drag each shape to its matching zone" (multi)

**Expert logic (drag_to_zone.py):**
- Nearest-first shape ordering (same as Exp 2 expert)
- Per shape: navigate -> mouse_down -> drag to zone center -> mouse_up -> pause
- Uses `fitts_trajectory()` with `button=BUTTON_DOWN` for drag segments

**Tests:**
- `test_drag_to_correct_zone_succeeds`: Teleport + manual drag sequence
- `test_expert_completes_task`: 20 episodes, >= 90% success

**Commit:** `"exp3: add drag-to-zone task + expert"`

---

## Task 6: use-slider (Phase 1c)

**Files:**
- Create: `experiments/miniwob_pygame/tasks/use_slider.py`
- Create: `experiments/miniwob_pygame/experts/use_slider.py`
- Modify: tests

**Task logic (use_slider.py):**
- `_setup_task`:
  - Horizontal Slider widget (x=50..350, y=200)
  - Draggable handle at random initial position
  - Target value (0-100) displayed as number + tick mark
  - Tolerance: +/- 5 (configurable)
- `_handle_button_down`: If cursor on handle, begin dragging
- `_handle_drag`: Constrain handle to track x-axis, update value
- `_handle_button_up`: Release handle
- `_check_success`: |slider_value - target_value| <= tolerance
- `task_instruction`: "Set the slider to {value}"

**Expert logic:**
1. Navigate to handle center
2. Mouse down (grab handle)
3. Compute target x position from target value
4. Drag handle horizontally to target_x using fitts_trajectory with button=BUTTON_DOWN
5. Mouse up

**Tests:**
- `test_slider_at_target_succeeds`: Set handle directly to target
- `test_expert_completes_task`: 20 episodes, >= 85% success

**Commit:** `"exp3: add use-slider task + expert"`

---

## Task 7: type-field (Phase 1d)

**Files:**
- Create: `experiments/miniwob_pygame/tasks/type_field.py`
- Create: `experiments/miniwob_pygame/experts/type_field.py`
- Modify: tests

**Task logic (type_field.py):**
- `_setup_task`:
  - Display target word (from VOCAB) as large text above input
  - TextInput widget centered below
- `_handle_button_down` / `_handle_button_up`: Delegate to TextInput
- `_handle_key`: Delegate to TextInput
- `_check_success`: TextInput.text == target_word -> success. Wrong char (text diverges from target prefix) -> failure.
- `task_instruction`: "Type: {word}"

**Expert logic:**
1. Navigate to TextInput center
2. Mouse down + up (click to focus)
3. Pause
4. For each char in target: emit key=char_to_key(ch), then key=0 (release)

**Tests:**
- `test_correct_typing_succeeds`: Focus + type correct word
- `test_wrong_character_fails`: Type wrong char
- `test_expert_completes_task`: 20 episodes, >= 95% success

**Commit:** `"exp3: add type-field task + expert"`

---

## Task 8: click-sequence (Phase 2a)

**Files:**
- Create: `experiments/miniwob_pygame/tasks/click_sequence.py`
- Create: `experiments/miniwob_pygame/experts/click_sequence.py`
- Modify: tests

**Task logic:**
- `_setup_task`: Place 3-5 numbered buttons at random positions. Generate target click order.
- Track which buttons clicked and in what order.
- `_handle_button_up`: If cursor on a button, check if it is the next expected in sequence. Correct -> advance. Wrong -> failure.
- `_check_success`: All buttons clicked in correct order.
- `task_instruction`: "Click in order: 3, 1, 4, 2"

**Expert logic:** Navigate to each button in sequence order, click each.

**Commit:** `"exp3: add click-sequence task + expert"`

---

## Task 9: draw-path (Phase 2b)

**Files:**
- Create: `experiments/miniwob_pygame/tasks/draw_path.py`
- Create: `experiments/miniwob_pygame/experts/draw_path.py`
- Modify: tests

**Task logic:**
- `_setup_task`: Generate a reference path (line between two points, or circle outline). Render as dotted/faded guide. Track drawn path while mouse held.
- `_handle_button_down`: Start recording drawn path
- `_handle_drag`: Append cursor position to drawn path
- `_handle_button_up`: Stop recording, evaluate path similarity
- `_check_success`: Mean distance from drawn path to reference path < threshold (15px)
- `task_instruction`: "Draw a line from A to B" or "Trace the circle"

**Expert logic:**
1. Navigate to path start
2. Mouse down
3. Follow reference path with small noisy deltas
4. Mouse up at path end

**Commit:** `"exp3: add draw-path task + expert"`

---

## Task 10: highlight-text (Phase 2c)

**Files:**
- Create: `experiments/miniwob_pygame/tasks/highlight_text.py`
- Create: `experiments/miniwob_pygame/experts/highlight_text.py`
- Modify: tests

**Task logic:**
- `_setup_task`: Render paragraph text using TextBlock. Pick random target word. Track highlight via mouse_down/drag/up positions.
- `_handle_button_down`: Record highlight start
- `_handle_drag`: Update highlight end (render blue highlight rect)
- `_handle_button_up`: Check if highlighted range covers target word
- `_check_success`: Highlighted text matches target word
- `task_instruction`: "Highlight the word '{word}'"

**Expert logic:**
1. Get target word pixel bounds from TextBlock
2. Navigate to word start x
3. Mouse down
4. Drag to word end x
5. Mouse up

**Commit:** `"exp3: add highlight-text task + expert"`

---

## Task 11: drag-sort (Phase 2d)

**Files:**
- Create: `experiments/miniwob_pygame/tasks/drag_sort.py`
- Create: `experiments/miniwob_pygame/experts/drag_sort.py`
- Modify: tests

**Task logic:**
- `_setup_task`: Place 3-5 numbered cards in shuffled order in a horizontal row. Each card shows a number.
- Dragging a card to a slot position inserts it there (other cards shift).
- `_check_success`: Cards in ascending order left to right.
- `task_instruction`: "Sort the numbers in ascending order"

**Expert logic:** Insertion sort -- find first out-of-place card, drag to correct position. Repeat.

**Commit:** `"exp3: add drag-sort task + expert"`

---

## Task 12: form-fill (Phase 3a)

**Files:**
- Create: `experiments/miniwob_pygame/tasks/form_fill.py`
- Create: `experiments/miniwob_pygame/experts/form_fill.py`
- Modify: tests

**Task logic:**
- `_setup_task`: 2-3 labeled TextInput fields + "Submit" button. Target values in instruction.
- `_handle_key(KEY_TAB)`: Move focus to next field
- `_handle_button_up`: If cursor on Submit button, check all fields
- `_check_success`: Submit clicked AND all fields match target values
- `task_instruction`: "Username: alice  Password: x7k"

**Expert logic:** For each field: navigate -> click (focus) -> type value -> pause. Then navigate to Submit -> click.

**Commit:** `"exp3: add form-fill task + expert"`

---

## Task 13: drag-and-label (Phase 3b)

**Files:**
- Create: `experiments/miniwob_pygame/tasks/drag_and_label.py`
- Create: `experiments/miniwob_pygame/experts/drag_and_label.py`
- Modify: tests

Exp 2 reimplemented on BaseTaskEnv. Key changes:
- Uses `button` (3-class) instead of `click` (binary)
- Instruction bar at top
- Returns observation dict instead of raw array
- Extended key classes (43 instead of 28)

Expert logic is the same as Exp 2's `expert.py` using the new action format.

**Commit:** `"exp3: add drag-and-label task + expert (Exp 2 on BaseTaskEnv)"`

---

## Task 14: scroll-and-click (Phase 3c)

**Files:**
- Create: `experiments/miniwob_pygame/tasks/scroll_and_click.py`
- Create: `experiments/miniwob_pygame/experts/scroll_and_click.py`
- Modify: tests

**Task logic:**
- `_setup_task`: ScrollableList widget with 15-20 items, viewport shows ~6. Target item off-screen. Scrollbar on right.
- Agent must drag scrollbar to scroll down, find target, click on it.
- `_check_success`: Target item clicked.
- `task_instruction`: "Click on '{item_name}'"

**No action space change** -- scroll is scrollbar drag.

**Expert logic:**
1. Navigate to scrollbar handle
2. Mouse down
3. Drag scrollbar down (compute distance from item index)
4. Mouse up
5. Navigate to target item
6. Click (mouse_down + mouse_up)

**Commit:** `"exp3: add scroll-and-click task + expert"`

---

## Task 15: copy-paste (Phase 3d)

**Files:**
- Create: `experiments/miniwob_pygame/tasks/copy_paste.py`
- Create: `experiments/miniwob_pygame/experts/copy_paste.py`
- Modify: tests

**Task logic:**
- `_setup_task`: Source TextBlock (top half) + target TextInput (bottom half). Internal clipboard.
- `_handle_key_down`: Detect Ctrl+C (KEY_CTRL and KEY_C both held -> copy) and Ctrl+V (KEY_CTRL and KEY_V both held -> paste)
- Copy: copies highlighted text to internal clipboard
- Paste: inserts clipboard into focused TextInput
- `_check_success`: Target field text matches source text
- `task_instruction`: "Copy the text and paste it into the field"

**Expert logic:**
1. Navigate to source text start, mouse_left=1, drag to end (highlight), mouse_left=0
2. Hold keys_held[KEY_CTRL]=1 + keys_held[KEY_C]=1 for one step, then release both
3. Navigate to TextInput, mouse_left=1 then 0 (click to focus)
4. Hold keys_held[KEY_CTRL]=1 + keys_held[KEY_V]=1 for one step, then release both

This is the key task that validates multi-key support. Ctrl+C is naturally represented as two keys held simultaneously, not an atomic class.

**Commit:** `"exp3: add copy-paste task + expert"`

---

## Task 16: Task Registry + Multi-Task Data Generation

**Files:**
- Create: `experiments/miniwob_pygame/task_registry.py`
- Create: `experiments/miniwob_pygame/generate_data.py`
- Create: `tests/miniwob_pygame/test_task_registry.py`

**Step 1: Implement task registry**

Maps task name strings to env classes and expert functions via lazy imports.
Functions: `get_env_class(task_name)`, `get_expert_fn(task_name)`.

**Step 2: Implement multi-task data generation**

Uses HuggingFace `datasets.Dataset.from_generator()` with one row per timestep (matching Exp 2 pattern). Data saved via `save_to_disk()` (Arrow/Parquet) and optionally `push_to_hub()`.

Data layout per task:
```
data/{task_name}/          # Arrow dataset directory (save_to_disk output)
  dataset_info.json
  data-00000-of-00010.arrow   # sharded Arrow files
  ...
```

Each row (one timestep) contains:
- `episode_id`: int32
- `timestep`: int32
- `image`: Image (PIL, stored as compressed bytes by HF datasets)
- `cursor_x`: float32 (normalized 0-1)
- `cursor_y`: float32 (normalized 0-1)
- `action_dx`: float32
- `action_dy`: float32
- `action_mouse_left`: int8
- `action_keys_held`: Sequence(int8, length=43)
- `episode_length`: int32
- `task_name`: string
- `success`: bool

CLI:
```bash
uv run python experiments/miniwob_pygame/generate_data.py \
    --tasks click-target drag-to-zone use-slider type-field -n 5000
uv run python experiments/miniwob_pygame/generate_data.py \
    --tasks click-target -n 1000 --push-to-hub PenTest-duck/cu-vla-exp3-data
```

Each expert module exports `run_expert_episode(env, rng)` as the standardized interface. This function resets the env internally (if not already reset), generates a trajectory, and replays it. Returns `(observations, actions, final_info)`.

**Step 3: Test task registry**

```python
import pytest
from experiments.miniwob_pygame.config import TASK_NAMES

@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_task_registered(task_name):
    try:
        cls = get_env_class(task_name)
        fn = get_expert_fn(task_name)
        assert cls is not None
        assert callable(fn)
    except (ImportError, KeyError):
        pytest.skip(f"Task {task_name} not yet implemented")
```

**Commit:** `"exp3: add task registry and multi-task data generation"`

---

## Task 17: Multi-Task ACT Model

**Files:**
- Create: `experiments/miniwob_pygame/model.py`
- Create: `experiments/miniwob_pygame/backbones.py` (copy from Exp 2)
- Create: `experiments/miniwob_pygame/baseline_cnn.py`
- Create: `tests/miniwob_pygame/test_model.py`

**Step 1: Write model shape tests**

```python
# tests/miniwob_pygame/test_model.py
import torch
import pytest

def test_act_forward_pass_shapes():
    from experiments.miniwob_pygame.model import ACT
    from experiments.miniwob_pygame.config import MODEL, ACTION, CHUNK

    model = ACT(backbone_name="resnet18", chunk_size=10)
    B = 4
    images = torch.randn(B, 3, 224, 224)
    proprio = torch.randn(B, MODEL.proprio_dim)  # 46
    action_dim = 2 + 1 + ACTION.num_keys  # 46: dx,dy + mouse + 43 keys
    actions = torch.randn(B, 10, action_dim)

    out = model(images, proprio, actions)

    assert out["dx"].shape == (B, 10)
    assert out["dy"].shape == (B, 10)
    assert out["mouse_left"].shape == (B, 10)   # single sigmoid logit
    assert out["keys_held"].shape == (B, 10, ACTION.num_keys)  # 43 sigmoid logits
    assert out["pad_logits"].shape == (B, 10)
    assert out["mu"].shape == (B, MODEL.latent_dim)
    assert out["logvar"].shape == (B, MODEL.latent_dim)

def test_act_inference_no_actions():
    from experiments.miniwob_pygame.model import ACT
    from experiments.miniwob_pygame.config import MODEL

    model = ACT(backbone_name="resnet18", chunk_size=10)
    model.eval()
    B = 2
    images = torch.randn(B, 3, 224, 224)
    proprio = torch.randn(B, MODEL.proprio_dim)  # 46

    with torch.no_grad():
        out = model(images, proprio, actions=None)

    assert out["dx"].shape == (B, 10)
    assert out["mouse_left"].shape == (B, 10)
    assert out["keys_held"].shape == (B, 10, 43)
    # mu and logvar should be zeros (no CVAE at inference)
    assert torch.all(out["mu"] == 0)

def test_baseline_cnn_forward():
    from experiments.miniwob_pygame.baseline_cnn import BaselineCNN

    model = BaselineCNN()
    x = torch.randn(4, 3, 224, 224)
    dx, dy, mouse_logit, key_logits = model(x)
    assert dx.shape == (4, 1)
    assert dy.shape == (4, 1)
    assert mouse_logit.shape == (4, 1)    # single sigmoid
    assert key_logits.shape == (4, 43)    # 43 independent sigmoids
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/miniwob_pygame/test_model.py -v`

**Step 3: Implement model.py**

Adapt from Exp 2's `model.py` with these changes:
- **Proprio dim:** 31 -> 46 (2 cursor + 1 mouse + 43 keys)
- **Action dim for CVAE:** 31 -> 46 (2 dx/dy + 1 mouse + 43 keys)
- **head_click -> head_mouse:** `nn.Linear(d_model, 1)` (single sigmoid, BCE loss)
- **head_key -> head_keys:** `nn.Linear(d_model, 43)` (43 independent sigmoids, per-key BCE)
- **Output dict:** `click` -> `mouse_left`, `key_logits` -> `keys_held`
- **Loss:** All binary heads use BCE. No CE/softmax anywhere for mouse/keys.

The architecture (CVAE encoder, transformer encoder/decoder, decoder queries) stays identical to Exp 2.

**Step 4: Implement baseline_cnn.py**

Adapt from Exp 2:
- `head_click` -> `head_mouse: nn.Linear(256, 1)` (sigmoid)
- `head_key` -> `head_keys: nn.Linear(256, 43)` (43 independent sigmoids)

**Step 5: Copy backbones.py from Exp 2**

```bash
cp experiments/act_drag_label/backbones.py experiments/miniwob_pygame/backbones.py
```

Update import: `from .config import MODEL` (same interface, compatible).

**Step 6: Run tests, verify pass**

Run: `uv run pytest tests/miniwob_pygame/test_model.py -v`

**Step 7: Commit**

```bash
git add experiments/miniwob_pygame/model.py \
      experiments/miniwob_pygame/baseline_cnn.py \
      experiments/miniwob_pygame/backbones.py \
      tests/miniwob_pygame/test_model.py
git commit -m "exp3: add multi-task ACT model and baseline CNN"
```

---

## Task 18: Multi-Task Training Loop

**Files:**
- Create: `experiments/miniwob_pygame/train.py`
- Create: `tests/miniwob_pygame/test_training_smoke.py`

**Step 1: Write smoke test**

```python
# tests/miniwob_pygame/test_training_smoke.py
"""Smoke test: 1 epoch of training on synthetic data."""
import os, tempfile
import numpy as np
import pytest
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage


@pytest.fixture
def synthetic_data_dir():
    """Create minimal HF datasets for two tasks."""
    features = Features({
        "episode_id": Value("int32"),
        "timestep": Value("int32"),
        "image": Image(),
        "cursor_x": Value("float32"),
        "cursor_y": Value("float32"),
        "action_dx": Value("float32"),
        "action_dy": Value("float32"),
        "action_mouse_left": Value("int8"),
        "action_keys_held": Sequence(Value("int8"), length=43),
        "episode_length": Value("int32"),
        "task_name": Value("string"),
        "success": Value("bool"),
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        for task in ["click-target", "type-field"]:
            rows = []
            for ep in range(4):
                T = 20
                for t in range(T):
                    rows.append({
                        "episode_id": ep,
                        "timestep": t,
                        "image": PILImage.fromarray(
                            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                        ),
                        "cursor_x": float(np.random.rand()),
                        "cursor_y": float(np.random.rand()),
                        "action_dx": float(np.random.randn()),
                        "action_dy": float(np.random.randn()),
                        "action_mouse_left": int(np.random.randint(0, 2)),
                        "action_keys_held": list(np.random.randint(0, 2, 43).astype(int)),
                        "episode_length": T,
                        "task_name": task,
                        "success": True,
                    })
            ds = Dataset.from_list(rows, features=features)
            ds.save_to_disk(os.path.join(tmpdir, task))
        yield tmpdir


def test_training_one_epoch(synthetic_data_dir):
    from experiments.miniwob_pygame.train import train
    train(
        backbone="resnet18",
        chunk_size=5,
        batch_size=4,
        data_dir=synthetic_data_dir,
        device="cpu",
        max_epochs=1,
    )
```

**Step 2: Implement train.py**

Adapt from Exp 2's `train.py` with these changes:

- **ChunkDataset:**
  - Loads Parquet episodes from `data/{task_name}/` directories
  - `actions_click` -> `actions_mouse_left` (binary held state)
  - `actions_key` (single class) -> `actions_keys_held` (43-dim binary vector)
  - Proprio: 2 (cursor_pos) + 1 (mouse_left) + 43 (keys_held) = 46
  - CVAE action vector: 2 (dx,dy) + 1 (mouse_left) + 43 (keys_held) = 46
  - `--tasks` CLI arg to select which tasks

- **Loss function changes:**
  - `loss_click` (BCE) -> `loss_mouse` (BCE, single sigmoid, masked)
  - `loss_key` (28-class CE) -> `loss_keys` (43 independent BCE sigmoids, masked, summed)
  - All binary heads use BCE. No softmax/CE anywhere for mouse/keys.
  - Everything else (loss_dx, loss_dy L1, loss_pad BCE, KL) stays the same

- **Data loading:**
  - Discovers episodes across task subdirectories
  - Shuffles across all tasks (mixed-task batches)
  - `--tasks` flag to subset

- **New CLI args:**
  - `--tasks`: list of task names (default: all found in data_dir)
  - `--max-epochs`: override epochs (useful for smoke tests)

**Step 3: Run smoke test**

Run: `uv run pytest tests/miniwob_pygame/test_training_smoke.py -v`

**Step 4: Commit**

```bash
git add experiments/miniwob_pygame/train.py \
      tests/miniwob_pygame/test_training_smoke.py
git commit -m "exp3: add multi-task training loop with mixed-task batching"
```

---

## Task 19: Unified Evaluation Framework

**Files:**
- Create: `experiments/miniwob_pygame/evaluate.py`

Adapt from Exp 2's `evaluate.py`:

- **ACTAgent:**
  - Mouse: sigmoid threshold > 0.5 -> mouse_left=1 (same as Exp 2 click)
  - Keys: 43 independent sigmoid thresholds > 0.5 -> keys_held binary vector
  - Temporal ensemble: blend sigmoid probabilities independently for mouse and each key
  - `build_proprio()` returns 46-dim vector [cursor_x, cursor_y, mouse_left, keys_held...]

- **Per-task evaluation:**
  - Loop over `--tasks` (default: all registered tasks)
  - Run each agent on each task separately
  - Report per-task metrics AND aggregate metrics

- **Metrics reported per task:**
  - Success rate (%)
  - Mean steps to completion
  - p50/p95 step counts
  - Loop latency p50/p95/p99 (ms)
  - Effective Hz

- **Aggregate metrics:**
  - Overall success rate (weighted average across tasks)
  - Motor score (per-primitive: click hit rate, drag accuracy, type accuracy)
  - Composition score (Phase 3 success vs Phase 1 average)

- **CLI:**
  ```bash
  uv run python experiments/miniwob_pygame/evaluate.py \
      --backbone resnet18 --chunk-size 10 \
      --tasks click-target drag-to-zone use-slider type-field \
      --device mps
  ```

**Commit:** `"exp3: add unified evaluation with per-task and aggregate metrics"`

---

## Task 20: HF Sync + Integration

**Files:**
- Create: `experiments/miniwob_pygame/hf_sync.py`
- Modify: `AGENTS.md` -- add Exp 3 section

**hf_sync.py:** Copy from Exp 2, adapt data paths to `data/{task_name}/{shard}/`.

**AGENTS.md update:** Add Experiment 3 section with run sequence and code layout.

**Run sequence:**
```bash
# Generate expert demos for Phase 1
uv run python experiments/miniwob_pygame/generate_data.py \
    --tasks click-target drag-to-zone use-slider type-field -n 5000

# Train multi-task ACT on Phase 1
uv run python experiments/miniwob_pygame/train.py \
    --backbone resnet18 --chunk-size 10 --device mps \
    --tasks click-target drag-to-zone use-slider type-field

# Evaluate
uv run python experiments/miniwob_pygame/evaluate.py \
    --backbone resnet18 --chunk-size 10 --device mps \
    --tasks click-target drag-to-zone use-slider type-field

# Visual evaluation
uv run python experiments/miniwob_pygame/evaluate.py --visual \
    --tasks click-target
```

**Commit:** `"exp3: add HF sync and update AGENTS.md with Exp 3 docs"`

---

## Implementation Order + Dependencies

```
Task 1 (config) ---------+
Task 2 (BaseTaskEnv) -----+
Task 3 (widgets) ---------+
     |                    |
     +-- Task 4  (click-target)      --+
     +-- Task 5  (drag-to-zone)        |-- Phase 1 (parallelizable)
     +-- Task 6  (use-slider)          |
     +-- Task 7  (type-field)        --+
     |                    |
     +-- Task 8  (click-sequence)    --+
     +-- Task 9  (draw-path)           |-- Phase 2 (parallelizable)
     +-- Task 10 (highlight-text)      |
     +-- Task 11 (drag-sort)         --+
     |                    |
     +-- Task 12 (form-fill)         --+
     +-- Task 13 (drag-and-label)      |-- Phase 3 (parallelizable)
     +-- Task 14 (scroll-and-click)    |
     +-- Task 15 (copy-paste)        --+
     |                    |
     +-- Task 16 (registry + datagen)  |-- depends on all tasks
                          |
Task 17 (model) ----------+-- depends on config only
Task 18 (training) --------+-- depends on model + datagen
Task 19 (evaluation) ------+-- depends on model + tasks
Task 20 (HF sync + docs) ----- depends on all
```

**Parallelizable groups:**
- Tasks 4-7 can be implemented in parallel (independent task envs)
- Tasks 8-11 can be implemented in parallel
- Tasks 12-15 can be implemented in parallel
- Task 17 (model) can be done in parallel with tasks 4-15

**Critical path:** 1 -> 2 -> 3 -> (4-7 parallel) -> 16 -> 18

---

## Open Decisions (from design doc, address during implementation)

1. **Observation frame rate vs action frame rate:** Start with both at 30Hz (same as Exp 2). Decoupling is a future experiment.
2. **Task difficulty curriculum within each task:** Implement as constructor params (e.g., `num_distractors`, `target_size`, `num_shapes`). Default to easiest variant for initial training.
3. **Multi-task negative transfer mitigation:** Start with uniform mixing. If gradient domination observed, add task-weighted sampling.
4. **Cursor_pos storage:** Store in each episode alongside observations for dense proprio signal.
5. **Key independence assumption:** Multi-binary treats each key independently (no cross-key correlation in loss). VPT evidence suggests this works for BC. Monitor if simultaneous-key tasks (copy-paste) show lower accuracy; if so, consider adding a correlation-aware loss or autoregressive key prediction.
6. **Key sparsity:** Most keys are 0 most of the time. If BCE loss is dominated by true negatives, consider focal loss or positive-class upweighting for keys.

## Research References

Action space design was informed by:
- **VPT (Baker et al.):** Multi-binary keys at 20Hz for Minecraft BC. Validated independent binary per key.
- **AlphaStar (Vinyals et al.):** Autoregressive action decomposition for combinatorial spaces.
- **FAST (Pertsch et al., 2025):** Naive binning fails at high frequency; compression essential.
- **pi0 (Physical Intelligence):** Flow matching action head, 50Hz, continuous actions.
- **ACT (Zhao et al.):** CVAE + transformer decoder, action chunking, L1 regression.
- **CogAgent:** Only GUI agent with KEY_DOWN/KEY_UP event-level primitives.
- **FightLadder:** Factored categorical outperforms flat binary for fighting game actions.
