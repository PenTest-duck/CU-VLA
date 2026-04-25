# Phase B0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Phase B0 of Experiment 6 — combined L-click hardening + V+L grounding — per the [B0 design doc](2026-04-25-action-primitives-phase-b0-design.md).

**Architecture:** Extend Phase A's `experiments/action_primitives/` codebase with: (a) distractor scenes with attribute-grounded instructions, (b) recovery trajectories via start-chunk scenarios + DART action noise, (c) two parallel 3-way click heads + soft-label CE, (d) diagnostic eval slicing + instruction probes. Single primitive (L-click) but full V+L stack. Train on HF Jobs `a100-large`; eval on M1 MPS.

**Tech Stack:** Python 3.14, PyTorch (MPS + bf16 autocast), HuggingFace `datasets` / `transformers` / `huggingface_hub`, pygame-ce 2.5.7, parquet shards, W&B for diagnostics, HF Jobs for training.

**Predecessor:** Phase A merged on `feat/exp6-phase-a` (PR #1).
**Branch:** `feat/exp6-phase-b` (already created and pushed).
**Worktree (recommended):** `feat/exp6-phase-b0` for B0 implementation work, PR'd back into `feat/exp6-phase-b`.

---

## Pre-flight checks

Before starting Task 1:

- [ ] **Confirm branch + worktree.** Working in a worktree from `feat/exp6-phase-b`:
```bash
git -C /Users/pentest-duck/Desktop/CU-VLA worktree add ../CU-VLA-b0 -b feat/exp6-phase-b0 origin/feat/exp6-phase-b
cd /Users/pentest-duck/Desktop/CU-VLA-b0
```

- [ ] **Verify environment.** `uv run python -c "import torch; import pygame; import datasets; print('OK')"`

- [ ] **Verify Phase A tests pass on the new branch:** `uv run pytest tests/action_primitives/ -q`. Expected: all pass (Phase A test suite was 71 passing).

- [ ] **Confirm HF + W&B credentials.** `echo $HF_TOKEN | head -c 10` and `echo $WANDB_API_KEY | head -c 10` should both produce non-empty output.

---

## Phase 1 — Attribute palettes + scene primitives

Lay the foundation for distractor scenes. Define attribute palettes, button/decorative-shape data classes.

### Task 1: Attribute palettes in config

**Files:**
- Modify: `experiments/action_primitives/config.py` (append new section)
- Test: `tests/action_primitives/test_config.py` (extend)

- [ ] **Step 1: Write the failing test**

Add to `tests/action_primitives/test_config.py`:
```python
def test_b0_attribute_palettes_defined():
    from experiments.action_primitives.config import (
        B0_COLORS, B0_SHAPES, B0_SIZES, B0_POSITION_GRID,
    )
    assert len(B0_COLORS) >= 8 and len(B0_COLORS) <= 10
    assert all(isinstance(c, tuple) and len(c) == 3 for c in B0_COLORS.values())
    assert set(B0_SHAPES) >= {"rect", "circle", "triangle", "square", "hexagon"}
    assert set(B0_SIZES) == {"small", "medium", "large"}
    assert B0_POSITION_GRID == (3, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/action_primitives/test_config.py::test_b0_attribute_palettes_defined -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement minimal palettes**

Append to `experiments/action_primitives/config.py`:
```python
# ---------- Phase B0 attribute palettes ----------
B0_COLORS: dict[str, tuple[int, int, int]] = {
    "red":    (220, 60,  60),
    "blue":   (60,  120, 230),
    "green":  (70,  180, 90),
    "yellow": (240, 220, 70),
    "orange": (240, 140, 50),
    "purple": (160, 80,  200),
    "pink":   (240, 130, 180),
    "cyan":   (80,  200, 220),
    "white":  (245, 245, 245),
    "black":  (30,  30,  30),
}

B0_SHAPES: tuple[str, ...] = ("rect", "circle", "triangle", "square", "hexagon")

B0_SIZES: dict[str, tuple[int, int]] = {
    "small":  (30, 50),
    "medium": (60, 90),
    "large":  (100, 140),
}

B0_POSITION_GRID: tuple[int, int] = (3, 3)  # 3 cols × 3 rows of zones

B0_BG_COLORS: tuple[tuple[int, int, int], ...] = (
    (245, 245, 248),
    (240, 246, 240),
    (245, 240, 240),
    (240, 245, 250),
    (250, 248, 235),
    (235, 240, 248),
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/action_primitives/test_config.py::test_b0_attribute_palettes_defined -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/action_primitives/config.py tests/action_primitives/test_config.py
git commit -m "feat(exp6/b0): attribute palettes — colors, shapes, sizes, position grid"
```

---

### Task 2: Button + decorative shape data classes

**Files:**
- Create: `experiments/action_primitives/scene.py`
- Test: `tests/action_primitives/test_scene.py`

- [ ] **Step 1: Write the failing test**

Create `tests/action_primitives/test_scene.py`:
```python
import pytest
from experiments.action_primitives.scene import Button, DecorativeShape


def test_button_has_attributes():
    b = Button(
        button_id=0, color="red", shape="circle", size="medium",
        pos_zone=(1, 1), x=100, y=100, w=80, h=60,
    )
    assert b.color == "red"
    assert b.shape == "circle"
    assert b.size == "medium"
    assert b.pos_zone == (1, 1)
    assert (b.x, b.y, b.w, b.h) == (100, 100, 80, 60)
    assert b.is_clickable is True


def test_decorative_shape_is_not_clickable():
    d = DecorativeShape(
        shape="triangle", color="cyan", x=200, y=200, w=40, h=30,
    )
    assert d.is_clickable is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/action_primitives/test_scene.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement scene primitives**

Create `experiments/action_primitives/scene.py`:
```python
"""Scene primitives for Phase B0: distractor-aware buttons and decorative shapes."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Button:
    """A clickable button with attribute-tagged identity."""
    button_id: int
    color: str           # B0_COLORS key
    shape: str           # B0_SHAPES element
    size: str            # B0_SIZES key
    pos_zone: tuple[int, int]  # (col, row) on B0_POSITION_GRID
    x: int               # top-left pixel x
    y: int               # top-left pixel y
    w: int
    h: int
    is_clickable: bool = True

    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h


@dataclass(frozen=True)
class DecorativeShape:
    """A non-clickable decorative shape used as a visual distractor."""
    shape: str
    color: str           # may use B0_COLORS but rendered semi-transparent / no border
    x: int
    y: int
    w: int
    h: int
    is_clickable: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/action_primitives/test_scene.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/action_primitives/scene.py tests/action_primitives/test_scene.py
git commit -m "feat(exp6/b0): Button and DecorativeShape scene primitives"
```

---

### Task 3: Scene generator (1-6 buttons + decorative shapes + bg)

**Files:**
- Modify: `experiments/action_primitives/scene.py`
- Modify: `tests/action_primitives/test_scene.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/action_primitives/test_scene.py`:
```python
import numpy as np
from experiments.action_primitives.scene import generate_scene, Scene


def test_generate_scene_basic():
    rng = np.random.default_rng(42)
    scene = generate_scene(rng=rng)
    assert isinstance(scene, Scene)
    assert 1 <= len(scene.buttons) <= 6
    assert 0 <= len(scene.decorative_shapes) <= 3
    # All buttons have unique button_ids
    ids = [b.button_id for b in scene.buttons]
    assert len(ids) == len(set(ids))
    # All buttons fit on canvas
    from experiments.action_primitives.config import ENV
    for b in scene.buttons:
        assert b.x >= 0 and b.y >= 0
        assert b.x + b.w <= ENV.canvas_w
        assert b.y + b.h <= ENV.canvas_h


def test_generate_scene_seeded_reproducible():
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    s1 = generate_scene(rng=rng1)
    s2 = generate_scene(rng=rng2)
    assert len(s1.buttons) == len(s2.buttons)
    assert s1.bg_color == s2.bg_color


def test_generate_scene_n_buttons_override():
    rng = np.random.default_rng(0)
    scene = generate_scene(rng=rng, n_buttons=4)
    assert len(scene.buttons) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/action_primitives/test_scene.py -v`
Expected: FAIL on the new tests.

- [ ] **Step 3: Implement scene generator**

Append to `experiments/action_primitives/scene.py`:
```python
import numpy as np

from experiments.action_primitives.config import (
    B0_BG_COLORS, B0_COLORS, B0_POSITION_GRID, B0_SHAPES, B0_SIZES, ENV,
)


@dataclass(frozen=True)
class Scene:
    """A rendered scene: bg color + buttons + decorative shapes."""
    buttons: tuple[Button, ...]
    decorative_shapes: tuple[DecorativeShape, ...]
    bg_color: tuple[int, int, int]


def _zone_to_xy(col: int, row: int, w: int, h: int, rng: np.random.Generator) -> tuple[int, int]:
    """Sample a (x, y) inside the (col, row) zone of B0_POSITION_GRID."""
    cols, rows = B0_POSITION_GRID
    zone_w = ENV.canvas_w // cols
    zone_h = ENV.canvas_h // rows
    margin = 10
    x_low = col * zone_w + margin
    x_high = max(x_low + 1, (col + 1) * zone_w - w - margin)
    y_low = row * zone_h + margin
    y_high = max(y_low + 1, (row + 1) * zone_h - h - margin)
    return int(rng.integers(x_low, x_high)), int(rng.integers(y_low, y_high))


def generate_scene(
    rng: np.random.Generator,
    n_buttons: int | None = None,
    n_decorative: int | None = None,
) -> Scene:
    """Generate a scene with 1-6 buttons + 0-3 decorative shapes + random bg."""
    if n_buttons is None:
        n_buttons = int(rng.integers(1, 7))  # 1..6 inclusive
    if n_decorative is None:
        n_decorative = int(rng.integers(0, 4))  # 0..3 inclusive

    bg_color = tuple(B0_BG_COLORS[rng.integers(0, len(B0_BG_COLORS))])

    # Sample distinct (col, row) zones (or random if not enough zones)
    cols, rows = B0_POSITION_GRID
    all_zones = [(c, r) for c in range(cols) for r in range(rows)]
    rng.shuffle(all_zones)
    chosen_zones = all_zones[:n_buttons]

    buttons: list[Button] = []
    for i, (col, row) in enumerate(chosen_zones):
        color = list(B0_COLORS.keys())[rng.integers(0, len(B0_COLORS))]
        shape = B0_SHAPES[rng.integers(0, len(B0_SHAPES))]
        size = list(B0_SIZES.keys())[rng.integers(0, len(B0_SIZES))]
        w_low, w_high = B0_SIZES[size]
        w = int(rng.integers(w_low, w_high))
        h = int(rng.integers(w_low, w_high))
        x, y = _zone_to_xy(col, row, w, h, rng)
        buttons.append(Button(
            button_id=i, color=color, shape=shape, size=size,
            pos_zone=(col, row), x=x, y=y, w=w, h=h,
        ))

    decorative_shapes: list[DecorativeShape] = []
    for _ in range(n_decorative):
        shape = B0_SHAPES[rng.integers(0, len(B0_SHAPES))]
        color = list(B0_COLORS.keys())[rng.integers(0, len(B0_COLORS))]
        w = int(rng.integers(20, 50))
        h = int(rng.integers(20, 50))
        x = int(rng.integers(10, ENV.canvas_w - w - 10))
        y = int(rng.integers(10, ENV.canvas_h - h - 10))
        decorative_shapes.append(DecorativeShape(shape=shape, color=color, x=x, y=y, w=w, h=h))

    return Scene(
        buttons=tuple(buttons),
        decorative_shapes=tuple(decorative_shapes),
        bg_color=bg_color,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/action_primitives/test_scene.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/action_primitives/scene.py tests/action_primitives/test_scene.py
git commit -m "feat(exp6/b0): scene generator with 1-6 buttons + decorative shapes"
```

---

### Task 4: Adversarial scene flagging

**Files:**
- Modify: `experiments/action_primitives/scene.py`
- Modify: `tests/action_primitives/test_scene.py`

A scene is **adversarial** if no single attribute uniquely identifies any button (forcing composite disambiguation).

- [ ] **Step 1: Write the failing test**

Append to `tests/action_primitives/test_scene.py`:
```python
from experiments.action_primitives.scene import compute_disambiguators, is_adversarial


def test_compute_disambiguators_single_button():
    rng = np.random.default_rng(0)
    scene = generate_scene(rng=rng, n_buttons=1)
    dis = compute_disambiguators(scene)
    # 1-button scene: any attribute trivially disambiguates
    assert len(dis) == 1
    assert "color" in dis[0] and "shape" in dis[0]


def test_is_adversarial_two_same_color_diff_shape():
    """Two red buttons differing in shape — not adversarial (shape disambiguates)."""
    scene = Scene(
        buttons=(
            Button(0, "red", "circle", "medium", (0, 0), 10, 10, 60, 60),
            Button(1, "red", "square", "medium", (1, 0), 100, 10, 60, 60),
        ),
        decorative_shapes=(),
        bg_color=(245, 245, 248),
    )
    assert is_adversarial(scene) is False


def test_is_adversarial_all_unique_on_some_attribute():
    """Three buttons each with a unique color — not adversarial."""
    scene = Scene(
        buttons=(
            Button(0, "red", "circle", "medium", (0, 0), 10, 10, 60, 60),
            Button(1, "blue", "circle", "medium", (1, 0), 100, 10, 60, 60),
            Button(2, "green", "circle", "medium", (2, 0), 200, 10, 60, 60),
        ),
        decorative_shapes=(),
        bg_color=(245, 245, 248),
    )
    # Each button has unique color → single-attribute disambiguates → not adversarial.
    assert is_adversarial(scene) is False


def test_is_adversarial_shared_attributes():
    """Two red circles in different sizes — single attribute insufficient for either."""
    scene = Scene(
        buttons=(
            Button(0, "red", "circle", "small", (0, 0), 10, 10, 40, 40),
            Button(1, "red", "circle", "large", (1, 0), 100, 10, 100, 100),
        ),
        decorative_shapes=(),
        bg_color=(245, 245, 248),
    )
    assert is_adversarial(scene) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/action_primitives/test_scene.py::test_compute_disambiguators_single_button -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement disambiguator computation**

Append to `experiments/action_primitives/scene.py`:
```python
ATTRIBUTE_KEYS: tuple[str, ...] = ("color", "shape", "size", "position")


def _attr_value(b: Button, attr: str) -> str | tuple[int, int]:
    if attr == "color": return b.color
    if attr == "shape": return b.shape
    if attr == "size":  return b.size
    if attr == "position": return b.pos_zone
    raise ValueError(f"unknown attribute {attr}")


def compute_disambiguators(scene: Scene) -> list[set[str]]:
    """For each button, return the set of single attributes that uniquely identify it.

    Empty set means no single attribute is sufficient (adversarial w.r.t. that button).
    """
    out: list[set[str]] = []
    for i, b in enumerate(scene.buttons):
        unique_attrs: set[str] = set()
        for attr in ATTRIBUTE_KEYS:
            val = _attr_value(b, attr)
            if all(_attr_value(other, attr) != val for j, other in enumerate(scene.buttons) if j != i):
                unique_attrs.add(attr)
        out.append(unique_attrs)
    return out


def is_adversarial(scene: Scene) -> bool:
    """A scene is adversarial if NO button has any single attribute disambiguator.

    (i.e., every button requires composite attributes to identify uniquely.)
    """
    if len(scene.buttons) <= 1:
        return False
    dis = compute_disambiguators(scene)
    # Adversarial iff at least one button requires composite (no single-attribute disambiguator).
    # Stronger version: ALL buttons require composite. We use the weaker "any button" version
    # because that's enough to flag the scene as composite-grounding-required.
    return any(len(d) == 0 for d in dis)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/action_primitives/test_scene.py -v`
Expected: PASS on all 4 new tests.

- [ ] **Step 5: Commit**

```bash
git add experiments/action_primitives/scene.py tests/action_primitives/test_scene.py
git commit -m "feat(exp6/b0): disambiguator computation + adversarial-scene flagging"
```

---

## Phase 2 — Instruction system

### Task 5: Instruction template registry

**Files:**
- Create: `experiments/action_primitives/instructions.py`
- Test: `tests/action_primitives/test_instructions.py`

- [ ] **Step 1: Write the failing test**

Create `tests/action_primitives/test_instructions.py`:
```python
import pytest
from experiments.action_primitives.instructions import (
    SINGLE_ATTR_TEMPLATES,
    DOUBLE_ATTR_TEMPLATES,
    TRIPLE_ATTR_TEMPLATES,
    render_template,
)


def test_template_registry_nonempty():
    assert len(SINGLE_ATTR_TEMPLATES) >= 3
    assert len(DOUBLE_ATTR_TEMPLATES) >= 3
    assert len(TRIPLE_ATTR_TEMPLATES) >= 3


def test_render_single_attr_color():
    template = "click the {color} button"
    out = render_template(template, color="red")
    assert out == "click the red button"


def test_render_uses_position_label():
    out = render_template("click the button in the {position}", position="top-left")
    assert "top-left" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/action_primitives/test_instructions.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement template registry**

Create `experiments/action_primitives/instructions.py`:
```python
"""Instruction templates and instruction rendering for Phase B0."""
from __future__ import annotations

# Single-attribute templates (60% of instructions)
SINGLE_ATTR_TEMPLATES: tuple[str, ...] = (
    "click the {color} button",
    "select the {color} one",
    "press the {color} button",
    "tap the {shape}",
    "click the {shape}",
    "press the {size} button",
    "select the {size} one",
    "click the button in the {position}",
    "tap the button on the {position}",
    "click the {position} button",
)

# Double-attribute templates (30%)
DOUBLE_ATTR_TEMPLATES: tuple[str, ...] = (
    "click the {color} {shape}",
    "select the {size} {color} button",
    "press the {color} {shape}",
    "tap the {color} button on the {position}",
    "click the {size} {shape}",
    "select the {shape} on the {position}",
)

# Triple-attribute templates (10%)
TRIPLE_ATTR_TEMPLATES: tuple[str, ...] = (
    "click the {size} {color} {shape}",
    "select the {color} {shape} on the {position}",
    "press the {size} {color} button on the {position}",
)

# Position labels (3x3 grid)
POSITION_LABELS: dict[tuple[int, int], str] = {
    (0, 0): "top-left",     (1, 0): "top-center",    (2, 0): "top-right",
    (0, 1): "middle-left",  (1, 1): "center",        (2, 1): "middle-right",
    (0, 2): "bottom-left",  (1, 2): "bottom-center", (2, 2): "bottom-right",
}


def render_template(template: str, **kwargs: str) -> str:
    """Format a template with attribute slot values."""
    return template.format(**kwargs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/action_primitives/test_instructions.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/action_primitives/instructions.py tests/action_primitives/test_instructions.py
git commit -m "feat(exp6/b0): instruction template registry (single/double/triple attr)"
```

---

### Task 6: Instruction generator (scene → unique instruction)

**Files:**
- Modify: `experiments/action_primitives/instructions.py`
- Modify: `tests/action_primitives/test_instructions.py`

The generator picks a target button + composite tier + template, emits instruction string. Must guarantee the resolved attribute set uniquely identifies the target button.

- [ ] **Step 1: Write the failing test**

Append to `tests/action_primitives/test_instructions.py`:
```python
import numpy as np
from experiments.action_primitives.scene import Scene, Button, generate_scene
from experiments.action_primitives.instructions import (
    generate_instruction,
    InstructionResult,
)


def test_generate_instruction_returns_unique_target():
    rng = np.random.default_rng(0)
    for _ in range(20):
        scene = generate_scene(rng=rng)
        result = generate_instruction(scene, rng=rng)
        assert isinstance(result, InstructionResult)
        assert isinstance(result.instruction, str)
        assert len(result.instruction) > 0
        assert 0 <= result.target_button_id < len(scene.buttons)
        assert result.composite_tier in (1, 2, 3)


def test_generate_instruction_single_button_scene():
    rng = np.random.default_rng(0)
    scene = generate_scene(rng=rng, n_buttons=1)
    result = generate_instruction(scene, rng=rng)
    # 1-button scene: target_id must be 0; instruction can be generic
    assert result.target_button_id == 0


def test_generate_instruction_uniqueness_invariant():
    """The chosen attribute combination should uniquely identify the target."""
    rng = np.random.default_rng(42)
    for _ in range(10):
        scene = generate_scene(rng=rng, n_buttons=4)
        result = generate_instruction(scene, rng=rng)
        target = scene.buttons[result.target_button_id]
        # The result's used_attrs must distinguish the target from all other buttons
        for j, other in enumerate(scene.buttons):
            if j == result.target_button_id:
                continue
            differs_on_any = any(
                _get_attr(target, a) != _get_attr(other, a) for a in result.used_attrs
            )
            assert differs_on_any, (
                f"Instruction {result.instruction!r} cannot distinguish target "
                f"button {result.target_button_id} from button {j}: used_attrs={result.used_attrs}"
            )


def _get_attr(b: Button, a: str):
    if a == "color": return b.color
    if a == "shape": return b.shape
    if a == "size": return b.size
    if a == "position": return b.pos_zone
    raise ValueError(a)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/action_primitives/test_instructions.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement instruction generator**

Append to `experiments/action_primitives/instructions.py`:
```python
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from experiments.action_primitives.scene import (
    ATTRIBUTE_KEYS, Scene, compute_disambiguators, _attr_value,
)


@dataclass(frozen=True)
class InstructionResult:
    instruction: str
    target_button_id: int
    used_attrs: tuple[str, ...]
    composite_tier: int  # 1, 2, or 3


# Composite tier sampling probabilities
TIER_PROBS = {1: 0.6, 2: 0.3, 3: 0.1}


def _attr_value_human(button, attr: str) -> str:
    """Convert an attribute value to its human-readable form for templates."""
    val = _attr_value(button, attr)
    if attr == "position":
        return POSITION_LABELS[val]
    return val  # color/shape/size are already strings


def _candidate_attr_sets(scene: Scene, target_idx: int, k: int) -> list[tuple[str, ...]]:
    """All k-attribute combinations that uniquely identify the target."""
    target = scene.buttons[target_idx]
    out: list[tuple[str, ...]] = []
    for combo in combinations(ATTRIBUTE_KEYS, k):
        target_vals = tuple(_attr_value(target, a) for a in combo)
        if all(
            tuple(_attr_value(other, a) for a in combo) != target_vals
            for j, other in enumerate(scene.buttons) if j != target_idx
        ):
            out.append(combo)
    return out


def _matching_templates(used_attrs: tuple[str, ...]) -> list[str]:
    """Templates whose slots exactly match the used_attrs set."""
    if len(used_attrs) == 1:
        pool = SINGLE_ATTR_TEMPLATES
    elif len(used_attrs) == 2:
        pool = DOUBLE_ATTR_TEMPLATES
    else:
        pool = TRIPLE_ATTR_TEMPLATES
    used_set = set(used_attrs)
    out: list[str] = []
    for tpl in pool:
        # Slot names in template
        slots = {s for s in ("color", "shape", "size", "position") if "{" + s + "}" in tpl}
        if slots == used_set:
            out.append(tpl)
    return out


def generate_instruction(scene: Scene, rng: np.random.Generator) -> InstructionResult:
    """Sample a target + composite tier + template; render unique instruction."""
    n_buttons = len(scene.buttons)
    target_idx = int(rng.integers(0, n_buttons))

    if n_buttons == 1:
        # 1-button scene: trivially unique. Pick any single attribute or generic.
        target = scene.buttons[0]
        attr = ATTRIBUTE_KEYS[rng.integers(0, len(ATTRIBUTE_KEYS))]
        templates = _matching_templates((attr,))
        if not templates:
            # Fallback (shouldn't happen with current registry)
            templates = ["click the button"]
            instruction = templates[0]
            return InstructionResult(
                instruction=instruction, target_button_id=0,
                used_attrs=(), composite_tier=1,
            )
        tpl = templates[rng.integers(0, len(templates))]
        slot_vals = {a: _attr_value_human(target, a) for a in (attr,)}
        return InstructionResult(
            instruction=render_template(tpl, **slot_vals),
            target_button_id=0,
            used_attrs=(attr,),
            composite_tier=1,
        )

    # Sample composite tier weighted by TIER_PROBS, falling back to higher tier if needed
    tier_order = [1, 2, 3]
    weights = np.array([TIER_PROBS[t] for t in tier_order])
    weights = weights / weights.sum()
    primary_tier = int(rng.choice(tier_order, p=weights))

    # Try primary tier first; fall back to next tier if no unique attribute set exists
    used_attrs: tuple[str, ...] = ()
    chosen_tier = 0
    for tier in [primary_tier] + [t for t in tier_order if t != primary_tier]:
        candidates = _candidate_attr_sets(scene, target_idx, tier)
        # Filter further: only candidates whose template pool is non-empty
        candidates = [c for c in candidates if len(_matching_templates(c)) > 0]
        if candidates:
            used_attrs = candidates[rng.integers(0, len(candidates))]
            chosen_tier = tier
            break
    if chosen_tier == 0:
        raise RuntimeError(
            f"No uniqueness-preserving attribute combination found for scene "
            f"with {n_buttons} buttons targeting button {target_idx}."
        )

    target = scene.buttons[target_idx]
    templates = _matching_templates(used_attrs)
    tpl = templates[rng.integers(0, len(templates))]
    slot_vals = {a: _attr_value_human(target, a) for a in used_attrs}
    return InstructionResult(
        instruction=render_template(tpl, **slot_vals),
        target_button_id=target_idx,
        used_attrs=used_attrs,
        composite_tier=chosen_tier,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/action_primitives/test_instructions.py -v`
Expected: PASS on all tests.

- [ ] **Step 5: Commit**

```bash
git add experiments/action_primitives/instructions.py tests/action_primitives/test_instructions.py
git commit -m "feat(exp6/b0): instruction generator with uniqueness invariant + tier sampling"
```

---

### Task 7: Typo injection (3-5%)

**Files:**
- Modify: `experiments/action_primitives/instructions.py`
- Modify: `tests/action_primitives/test_instructions.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/action_primitives/test_instructions.py`:
```python
from experiments.action_primitives.instructions import inject_typo


def test_inject_typo_changes_string():
    rng = np.random.default_rng(0)
    s = "click the red button"
    out = inject_typo(s, rng=rng)
    assert out != s  # at least one char changed
    assert len(out) >= len(s) - 1  # bounded growth/shrinkage


def test_inject_typo_idempotent_at_zero_rate():
    """Calling with no perturbation should be a no-op."""
    rng = np.random.default_rng(0)
    s = "click the red button"
    out = inject_typo(s, rng=rng, n_changes=0)
    assert out == s
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/action_primitives/test_instructions.py::test_inject_typo_changes_string -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement typo injection**

Append to `experiments/action_primitives/instructions.py`:
```python
TYPO_RATE: float = 0.04  # 4% of instructions get a typo


def inject_typo(s: str, rng: np.random.Generator, n_changes: int = 1) -> str:
    """Inject n_changes character-level perturbations: swap/drop/insert."""
    if n_changes == 0 or len(s) < 2:
        return s
    chars = list(s)
    for _ in range(n_changes):
        if len(chars) < 2:
            break
        op = rng.integers(0, 3)  # 0=swap, 1=drop, 2=insert
        idx = int(rng.integers(0, len(chars)))
        if op == 0 and idx + 1 < len(chars):
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        elif op == 1:
            del chars[idx]
        else:  # insert
            insert_char = chr(int(rng.integers(ord("a"), ord("z") + 1)))
            chars.insert(idx, insert_char)
    return "".join(chars)


def maybe_typo(s: str, rng: np.random.Generator, rate: float = TYPO_RATE) -> str:
    """With probability `rate`, inject a typo; else return s unchanged."""
    if rng.random() < rate:
        return inject_typo(s, rng=rng, n_changes=1)
    return s
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/action_primitives/test_instructions.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/action_primitives/instructions.py tests/action_primitives/test_instructions.py
git commit -m "feat(exp6/b0): typo injection (swap/drop/insert) at ~4% rate"
```

---

## Phase 3 — Env extension + instruction-aware expert

### Task 8: Extend env to render distractor scenes

**Files:**
- Modify: `experiments/action_primitives/env.py`
- Modify: `tests/action_primitives/test_env.py`

The existing `LClickEnv.reset()` generates one button using its own logic. We replace this with a Scene + render Buttons + DecorativeShapes.

- [ ] **Step 1: Write the failing test**

Append to `tests/action_primitives/test_env.py`:
```python
import numpy as np
from experiments.action_primitives.scene import generate_scene
from experiments.action_primitives.env import LClickEnv


def test_env_accepts_scene_and_target():
    rng = np.random.default_rng(0)
    scene = generate_scene(rng=rng, n_buttons=3)
    env = LClickEnv(scene=scene, target_button_id=1, seed=0)
    obs, info = env.reset(seed=0)
    assert info["target_button_id"] == 1
    target_button = scene.buttons[1]
    tx, ty, tw, th = info["target_bbox"]
    assert (tx, ty, tw, th) == (target_button.x, target_button.y, target_button.w, target_button.h)
    # Image should render all buttons + decorative shapes
    img = obs["image"]
    assert img.size == (720, 450)


def test_env_step_with_target_collidepoint():
    """Cursor on target + L_press → L_release should set done."""
    from experiments.action_primitives.env import Action
    rng = np.random.default_rng(0)
    scene = generate_scene(rng=rng, n_buttons=2)
    env = LClickEnv(scene=scene, target_button_id=0, seed=0)
    target = scene.buttons[0]
    cx, cy = target.center()
    # Move cursor to target
    env.cursor_x, env.cursor_y = cx, cy
    obs, done, info = env.step(Action(click=1))  # L_press on target
    assert not done
    obs, done, info = env.step(Action(click=2))  # L_release on target
    assert done is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/action_primitives/test_env.py::test_env_accepts_scene_and_target -v`
Expected: FAIL — `LClickEnv` doesn't accept `scene` kwarg.

- [ ] **Step 3: Modify LClickEnv to accept scene**

Modify `experiments/action_primitives/env.py`:

(a) Update `__init__` signature and `reset()`:
```python
class LClickEnv:
    def __init__(self, scene=None, target_button_id: int | None = None,
                 theme: str = "flat-modern", seed: int = 0,
                 visual: bool = False, fps: int = 30) -> None:
        # ... existing display init code unchanged ...
        self.rng = np.random.default_rng(seed)
        self.theme = THEMES[theme]
        self._scene = scene  # may be None for legacy single-button mode
        self._target_button_id = target_button_id
        self.reset(seed=seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self._scene is not None:
            return self._reset_with_scene()
        return self._reset_legacy()

    def _reset_with_scene(self):
        scene = self._scene
        target_id = self._target_button_id
        assert target_id is not None and 0 <= target_id < len(scene.buttons), (
            f"target_button_id={target_id} invalid for scene with {len(scene.buttons)} buttons"
        )
        target = scene.buttons[target_id]
        import pygame
        self.target_rect = pygame.Rect(target.x, target.y, target.w, target.h)
        self.target_color = self.theme["button"]  # rendering uses scene colors per button
        # Cursor starts at random position not on target.
        for _ in range(100):
            cx = int(self.rng.integers(10, ENV.canvas_w - 10))
            cy = int(self.rng.integers(10, ENV.canvas_h - 10))
            if not self.target_rect.collidepoint(cx, cy):
                break
        else:
            raise RuntimeError("Could not place cursor off-target after 100 tries")
        self.cursor_x = float(cx)
        self.cursor_y = float(cy)
        self.held_keys = np.zeros(NUM_KEYS, dtype=bool)
        self.held_mouse = np.zeros(3, dtype=bool)
        self.capslock = False
        self.done_flag = False
        self._press_frame = None
        self._release_frame = None
        self.frame_idx = 0
        obs = self._render_obs()
        info = self._info()
        return obs, info

    def _reset_legacy(self):
        # ... existing reset() body, renamed ...
```

(b) Update `_render_obs()` to draw the scene:
```python
def _render_obs(self):
    if self._scene is not None:
        self.screen.fill(self._scene.bg_color)
        # Decorative shapes (semi-transparent, no border)
        for d in self._scene.decorative_shapes:
            self._draw_shape(d.shape, d.color, d.x, d.y, d.w, d.h, alpha=128)
        # Buttons
        for b in self._scene.buttons:
            from experiments.action_primitives.config import B0_COLORS
            self._draw_shape(b.shape, B0_COLORS[b.color], b.x, b.y, b.w, b.h, alpha=255)
    else:
        # Legacy single-button rendering
        self.screen.fill(self.theme["bg"])
        pygame.draw.rect(self.screen, self.target_color, self.target_rect, border_radius=6)
    # Cursor (unchanged from legacy)
    cx, cy = int(self.cursor_x), int(self.cursor_y)
    pygame.draw.polygon(
        self.screen, (0, 0, 0),
        [(cx, cy), (cx + 14, cy + 10), (cx + 8, cy + 12), (cx + 12, cy + 20),
         (cx + 9, cy + 21), (cx + 5, cy + 13), (cx, cy + 16)],
    )
    arr = pygame.surfarray.array3d(self.screen).transpose(1, 0, 2).copy()
    img = Image.fromarray(arr, mode="RGB")
    proprio = Proprio(
        cursor_x=self.cursor_x / ENV.canvas_w,
        cursor_y=self.cursor_y / ENV.canvas_h,
        held_keys=self.held_keys.copy(),
        held_mouse=self.held_mouse.copy(),
        capslock=self.capslock,
    )
    if self.visual:
        pygame.event.pump()
        pygame.display.flip()
        if self._clock is not None:
            self._clock.tick(self.fps)
    return {"image": img, "proprio": proprio}


def _draw_shape(self, shape: str, color, x, y, w, h, alpha: int = 255):
    import pygame
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    rgba = (*color, alpha)
    if shape == "rect" or shape == "square":
        pygame.draw.rect(surf, rgba, (0, 0, w, h), border_radius=6)
    elif shape == "circle":
        pygame.draw.ellipse(surf, rgba, (0, 0, w, h))
    elif shape == "triangle":
        pygame.draw.polygon(surf, rgba, [(w // 2, 0), (0, h), (w, h)])
    elif shape == "hexagon":
        pts = [(w * 0.25, 0), (w * 0.75, 0), (w, h * 0.5),
               (w * 0.75, h), (w * 0.25, h), (0, h * 0.5)]
        pygame.draw.polygon(surf, rgba, pts)
    self.screen.blit(surf, (x, y))
```

(c) Update `_info()` to include `target_button_id`:
```python
def _info(self):
    info = {
        "target_bbox": (self.target_rect.x, self.target_rect.y, self.target_rect.w, self.target_rect.h),
        "target_color": self.target_color,
        "cursor_xy": (self.cursor_x, self.cursor_y),
        "frame_idx": self.frame_idx,
        "success": self.done_flag,
    }
    if self._scene is not None:
        info["target_button_id"] = self._target_button_id
        info["scene"] = self._scene
    return info
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/action_primitives/test_env.py -v`
Expected: PASS on the new tests AND existing legacy tests.

- [ ] **Step 5: Commit**

```bash
git add experiments/action_primitives/env.py tests/action_primitives/test_env.py
git commit -m "feat(exp6/b0): LClickEnv accepts Scene + target_button_id; renders distractor shapes"
```

---

### Task 9: Instruction-aware expert

**Files:**
- Modify: `experiments/action_primitives/expert.py`
- Modify: `tests/action_primitives/test_expert.py`

The existing `LClickExpert` takes `target_center` directly. We add a wrapper that takes `(scene, target_button_id)` and delegates to it. The wrapper is needed because: (1) recovery training queries the expert from arbitrary cursor states; (2) future scenario errors need to compute "wrong direction" relative to the instruction-specified target.

- [ ] **Step 1: Write the failing test**

Append to `tests/action_primitives/test_expert.py`:
```python
import numpy as np
from experiments.action_primitives.scene import generate_scene
from experiments.action_primitives.expert import (
    LClickExpertConfig, InstructionAwareLClickExpert,
)


def test_instruction_aware_expert_targets_specified_button():
    rng = np.random.default_rng(0)
    scene = generate_scene(rng=rng, n_buttons=3)
    target_id = 1
    target = scene.buttons[target_id]
    cx, cy = 10.0, 10.0
    cfg = LClickExpertConfig(tempo="normal", seed=0)
    expert = InstructionAwareLClickExpert(
        cfg=cfg, scene=scene, target_button_id=target_id, cursor_xy=(cx, cy),
    )
    # First action's dx, dy should head toward target
    a0 = next(iter(expert))
    target_dx = target.center()[0] - cx
    target_dy = target.center()[1] - cy
    assert np.sign(a0.dx) == np.sign(target_dx)
    assert np.sign(a0.dy) == np.sign(target_dy)


def test_expert_re_query_from_arbitrary_state():
    """Re-querying the expert from a new cursor position should produce a fresh trajectory."""
    rng = np.random.default_rng(0)
    scene = generate_scene(rng=rng, n_buttons=2)
    target_id = 0
    cfg = LClickExpertConfig(tempo="normal", seed=0)
    expert1 = InstructionAwareLClickExpert(cfg, scene, target_id, cursor_xy=(10.0, 10.0))
    expert2 = InstructionAwareLClickExpert(cfg, scene, target_id, cursor_xy=(700.0, 400.0))
    a1 = next(iter(expert1))
    a2 = next(iter(expert2))
    # Two starting positions on opposite sides → different signs in at least one of dx, dy
    assert (np.sign(a1.dx) != np.sign(a2.dx)) or (np.sign(a1.dy) != np.sign(a2.dy))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/action_primitives/test_expert.py::test_instruction_aware_expert_targets_specified_button -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement instruction-aware expert wrapper**

Append to `experiments/action_primitives/expert.py`:
```python
class InstructionAwareLClickExpert:
    """Wrapper that grounds expert to a specific button id within a scene.

    Re-queryable: construct fresh from (cursor_xy, scene, target_button_id) at
    any time to get a fresh recovery trajectory from a new cursor state.
    """
    def __init__(
        self,
        cfg: LClickExpertConfig,
        scene,                     # Scene
        target_button_id: int,
        cursor_xy: tuple[float, float],
    ) -> None:
        target = scene.buttons[target_button_id]
        self._inner = LClickExpert(
            cfg=cfg,
            cursor_xy=cursor_xy,
            target_center=target.center(),
        )
        self._scene = scene
        self._target_button_id = target_button_id

    def __iter__(self):
        return iter(self._inner)

    def __next__(self):
        return next(self._inner)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/action_primitives/test_expert.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/action_primitives/expert.py tests/action_primitives/test_expert.py
git commit -m "feat(exp6/b0): InstructionAwareLClickExpert wrapper grounded to scene + target_id"
```

---

## Phase 4 — Recovery mechanisms

### Task 10: Start-chunk wrong-segment generator

**Files:**
- Create: `experiments/action_primitives/recovery.py`
- Test: `tests/action_primitives/test_recovery.py`

- [ ] **Step 1: Write the failing test**

Create `tests/action_primitives/test_recovery.py`:
```python
import numpy as np
from experiments.action_primitives.scene import generate_scene
from experiments.action_primitives.recovery import (
    sample_wrong_segment_type, generate_wrong_segment, WrongSegment,
)
from experiments.action_primitives.env import Action


def test_sample_wrong_segment_type_distribution():
    """Verify 50/30/20 wrong-direction/overshoot/edge-bang sampling."""
    rng = np.random.default_rng(0)
    counts = {"wrong-direction": 0, "overshoot": 0, "edge-bang": 0}
    n = 10000
    for _ in range(n):
        t = sample_wrong_segment_type(rng=rng)
        counts[t] += 1
    # Allow ±2 percentage points
    assert abs(counts["wrong-direction"] / n - 0.50) < 0.02
    assert abs(counts["overshoot"] / n - 0.30) < 0.02
    assert abs(counts["edge-bang"] / n - 0.20) < 0.02


def test_generate_wrong_segment_returns_actions_and_state():
    rng = np.random.default_rng(0)
    scene = generate_scene(rng=rng, n_buttons=2)
    target = scene.buttons[0]
    seg = generate_wrong_segment(
        scene=scene, target_button_id=0, cursor_xy=(360.0, 225.0),
        segment_type="wrong-direction", k_frames=10, rng=rng,
    )
    assert isinstance(seg, WrongSegment)
    assert len(seg.actions) == 10
    assert all(isinstance(a, Action) for a in seg.actions)
    assert seg.k_frames == 10
    assert seg.segment_type == "wrong-direction"
    # Final cursor should be off-target (we deliberately moved away)
    final_x, final_y = seg.final_cursor_xy
    assert not (target.x <= final_x < target.x + target.w
                and target.y <= final_y < target.y + target.h)


def test_wrong_segment_overshoot_passes_target():
    rng = np.random.default_rng(0)
    scene = generate_scene(rng=rng, n_buttons=1)
    target = scene.buttons[0]
    cx, cy = 10.0, 10.0  # far from target
    seg = generate_wrong_segment(
        scene=scene, target_button_id=0, cursor_xy=(cx, cy),
        segment_type="overshoot", k_frames=10, rng=rng,
    )
    final_x, final_y = seg.final_cursor_xy
    target_cx, target_cy = target.center()
    # Overshoot should travel past target (start to final dist > start to target dist)
    start_to_target = ((target_cx - cx)**2 + (target_cy - cy)**2)**0.5
    start_to_final = ((final_x - cx)**2 + (final_y - cy)**2)**0.5
    assert start_to_final >= start_to_target * 0.8  # passed by or near target
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/action_primitives/test_recovery.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement wrong-segment generators**

Create `experiments/action_primitives/recovery.py`:
```python
"""Recovery trajectory generation for Phase B0.

Two mechanisms:
1. Start-chunk wrong segments: K=5-15 frames of deliberately-wrong actions at
   episode start, then clean expert recovery. 50/30/20 wrong-direction/overshoot/edge.
2. DART action noise: per-frame Gaussian on dx/dy in clean episodes.

Both produce internally-consistent trajectories (env steps deterministically from
each action). Wrong-segment frames have loss_mask=0; DART noise frames have
loss_mask=1 (label is correct expert action from resulting state).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from experiments.action_primitives.config import ENV, MOUSE_CAP_PX, NUM_KEYS
from experiments.action_primitives.env import Action
from experiments.action_primitives.scene import Scene


SegmentType = Literal["wrong-direction", "overshoot", "edge-bang"]


@dataclass(frozen=True)
class WrongSegment:
    actions: tuple[Action, ...]   # K actions to apply
    k_frames: int
    segment_type: SegmentType
    final_cursor_xy: tuple[float, float]


# Distribution: 50% wrong-direction, 30% overshoot, 20% edge-bang.
_TYPE_PROBS = np.array([0.50, 0.30, 0.20])
_TYPE_LABELS: tuple[SegmentType, ...] = ("wrong-direction", "overshoot", "edge-bang")


def sample_wrong_segment_type(rng: np.random.Generator) -> SegmentType:
    return _TYPE_LABELS[rng.choice(len(_TYPE_LABELS), p=_TYPE_PROBS)]


def _idle_keys() -> np.ndarray:
    return np.full(NUM_KEYS, 2, dtype=np.int64)


def _step_cursor(cx: float, cy: float, dx: float, dy: float) -> tuple[float, float]:
    """Apply action delta + clip to canvas, like LClickEnv.step()."""
    new_x = float(np.clip(cx + dx, 0, ENV.canvas_w - 1))
    new_y = float(np.clip(cy + dy, 0, ENV.canvas_h - 1))
    return new_x, new_y


def _wrong_direction_segment(
    cursor_xy: tuple[float, float],
    target_center: tuple[float, float],
    k_frames: int,
    rng: np.random.Generator,
) -> tuple[list[Action], tuple[float, float]]:
    """Cursor heads ~90-180° away from target for k_frames."""
    cx, cy = cursor_xy
    tx, ty = target_center
    # Direction from cursor toward target
    to_target = np.array([tx - cx, ty - cy], dtype=np.float64)
    norm = np.linalg.norm(to_target)
    if norm < 1.0:
        to_target = np.array([1.0, 0.0])  # arbitrary
    else:
        to_target = to_target / norm
    # Wrong direction: rotate by 90-180 degrees
    angle = float(rng.uniform(np.pi / 2, np.pi)) * float(rng.choice([-1.0, 1.0]))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    wrong_dir = np.array([
        cos_a * to_target[0] - sin_a * to_target[1],
        sin_a * to_target[0] + cos_a * to_target[1],
    ])
    # Step magnitude per frame (moderate)
    step_mag = 25.0
    actions: list[Action] = []
    for _ in range(k_frames):
        dx, dy = wrong_dir * step_mag
        dx = float(np.clip(dx, -MOUSE_CAP_PX, MOUSE_CAP_PX))
        dy = float(np.clip(dy, -MOUSE_CAP_PX, MOUSE_CAP_PX))
        actions.append(Action(dx=dx, dy=dy, key_events=_idle_keys()))
        cx, cy = _step_cursor(cx, cy, dx, dy)
    return actions, (cx, cy)


def _overshoot_segment(
    cursor_xy: tuple[float, float],
    target_center: tuple[float, float],
    target_size: tuple[int, int],
    k_frames: int,
    rng: np.random.Generator,
) -> tuple[list[Action], tuple[float, float]]:
    """Cursor passes target by 1.5-2× its size, then needs to reverse."""
    cx, cy = cursor_xy
    tx, ty = target_center
    tw, th = target_size
    overshoot_factor = float(rng.uniform(1.5, 2.0))
    end_x = tx + (tx - cx) / max(np.linalg.norm([tx - cx, ty - cy]), 1.0) * tw * overshoot_factor
    end_y = ty + (ty - cy) / max(np.linalg.norm([tx - cx, ty - cy]), 1.0) * th * overshoot_factor
    # Linear interpolation cursor → end_x, end_y over k_frames
    actions: list[Action] = []
    for i in range(k_frames):
        t = (i + 1) / k_frames
        next_x = cx + (end_x - cx) * t
        next_y = cy + (end_y - cy) * t
        prev_x = cx + (end_x - cx) * (i / k_frames)
        prev_y = cy + (end_y - cy) * (i / k_frames)
        dx = float(np.clip(next_x - prev_x, -MOUSE_CAP_PX, MOUSE_CAP_PX))
        dy = float(np.clip(next_y - prev_y, -MOUSE_CAP_PX, MOUSE_CAP_PX))
        actions.append(Action(dx=dx, dy=dy, key_events=_idle_keys()))
    final_x, final_y = _step_cursor(cx, cy, sum(a.dx for a in actions), sum(a.dy for a in actions))
    return actions, (float(np.clip(end_x, 0, ENV.canvas_w - 1)),
                     float(np.clip(end_y, 0, ENV.canvas_h - 1)))


def _edge_bang_segment(
    cursor_xy: tuple[float, float],
    k_frames: int,
    rng: np.random.Generator,
) -> tuple[list[Action], tuple[float, float]]:
    """Cursor heads to canvas edge, pinned for several frames before recovery point."""
    cx, cy = cursor_xy
    edge = rng.choice(["left", "right", "top", "bottom"])
    if edge == "left":
        end_x, end_y = 5.0, cy
    elif edge == "right":
        end_x, end_y = ENV.canvas_w - 5.0, cy
    elif edge == "top":
        end_x, end_y = cx, 5.0
    else:
        end_x, end_y = cx, ENV.canvas_h - 5.0
    actions: list[Action] = []
    travel_frames = max(1, k_frames - 4)  # remaining frames are "pinned at edge"
    for i in range(k_frames):
        if i < travel_frames:
            t = (i + 1) / travel_frames
            next_x = cx + (end_x - cx) * t
            next_y = cy + (end_y - cy) * t
            prev_x = cx + (end_x - cx) * (i / travel_frames)
            prev_y = cy + (end_y - cy) * (i / travel_frames)
            dx = float(np.clip(next_x - prev_x, -MOUSE_CAP_PX, MOUSE_CAP_PX))
            dy = float(np.clip(next_y - prev_y, -MOUSE_CAP_PX, MOUSE_CAP_PX))
        else:
            # Pinned: tiny no-op, env clip keeps cursor at edge
            dx, dy = 0.5, 0.5
        actions.append(Action(dx=dx, dy=dy, key_events=_idle_keys()))
    return actions, (end_x, end_y)


def generate_wrong_segment(
    scene: Scene,
    target_button_id: int,
    cursor_xy: tuple[float, float],
    segment_type: SegmentType,
    k_frames: int,
    rng: np.random.Generator,
) -> WrongSegment:
    target = scene.buttons[target_button_id]
    if segment_type == "wrong-direction":
        actions, final = _wrong_direction_segment(cursor_xy, target.center(), k_frames, rng)
    elif segment_type == "overshoot":
        actions, final = _overshoot_segment(
            cursor_xy, target.center(), (target.w, target.h), k_frames, rng,
        )
    elif segment_type == "edge-bang":
        actions, final = _edge_bang_segment(cursor_xy, k_frames, rng)
    else:
        raise ValueError(f"Unknown segment_type: {segment_type}")
    return WrongSegment(
        actions=tuple(actions), k_frames=k_frames,
        segment_type=segment_type, final_cursor_xy=final,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/action_primitives/test_recovery.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/action_primitives/recovery.py tests/action_primitives/test_recovery.py
git commit -m "feat(exp6/b0): start-chunk wrong-segment generator (3 types, 50/30/20 mix)"
```

---

### Task 11: DART action-noise mechanism

**Files:**
- Modify: `experiments/action_primitives/recovery.py`
- Modify: `tests/action_primitives/test_recovery.py`

DART perturbs the expert's emitted dx/dy with Gaussian noise. Excluded on click-event frames (and 1 frame before/after).

- [ ] **Step 1: Write the failing test**

Append to `tests/action_primitives/test_recovery.py`:
```python
from experiments.action_primitives.recovery import (
    apply_dart_noise, DART_PROB_DEFAULT, DART_SIGMA_DEFAULT,
)


def test_dart_noise_perturbs_at_expected_rate():
    rng = np.random.default_rng(0)
    n = 10000
    perturbed = 0
    for _ in range(n):
        clean_action = Action(dx=10.0, dy=5.0, click=0)
        noisy_action, was_perturbed = apply_dart_noise(
            clean_action, rng=rng, p=0.10, sigma=20.0,
        )
        if was_perturbed:
            perturbed += 1
    rate = perturbed / n
    assert abs(rate - 0.10) < 0.015  # ~10% with allowance


def test_dart_noise_skips_click_frames():
    rng = np.random.default_rng(0)
    # Force "always perturb" by setting p=1.0; click frame should still be skipped
    a_press = Action(dx=10.0, dy=5.0, click=1)
    a_out, was_perturbed = apply_dart_noise(a_press, rng=rng, p=1.0, sigma=20.0)
    assert was_perturbed is False
    assert a_out.dx == 10.0 and a_out.dy == 5.0


def test_dart_noise_preserves_other_fields():
    rng = np.random.default_rng(0)
    a = Action(dx=10.0, dy=5.0, click=0, scroll=0.0, key_events=_idle_keys_test())
    noisy, _ = apply_dart_noise(a, rng=rng, p=1.0, sigma=20.0)
    assert noisy.click == a.click
    assert noisy.scroll == a.scroll


def _idle_keys_test():
    return np.full(NUM_KEYS, 2, dtype=np.int64)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/action_primitives/test_recovery.py::test_dart_noise_perturbs_at_expected_rate -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement DART noise**

Append to `experiments/action_primitives/recovery.py`:
```python
DART_PROB_DEFAULT: float = 0.08
DART_SIGMA_DEFAULT: float = 20.0


def apply_dart_noise(
    action: Action,
    rng: np.random.Generator,
    p: float = DART_PROB_DEFAULT,
    sigma: float = DART_SIGMA_DEFAULT,
) -> tuple[Action, bool]:
    """With prob p, add Gaussian(0, sigma²) to dx/dy. Skip click event frames.

    Returns (possibly_perturbed_action, was_perturbed).
    """
    # Skip click event frames (preserve click semantics)
    if action.click != 0:
        return action, False
    if rng.random() >= p:
        return action, False
    noise_dx = float(rng.normal(0.0, sigma))
    noise_dy = float(rng.normal(0.0, sigma))
    new_dx = float(np.clip(action.dx + noise_dx, -MOUSE_CAP_PX, MOUSE_CAP_PX))
    new_dy = float(np.clip(action.dy + noise_dy, -MOUSE_CAP_PX, MOUSE_CAP_PX))
    return Action(
        dx=new_dx, dy=new_dy,
        click=action.click, scroll=action.scroll, key_events=action.key_events,
    ), True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/action_primitives/test_recovery.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/action_primitives/recovery.py tests/action_primitives/test_recovery.py
git commit -m "feat(exp6/b0): DART action noise (p=0.08, σ=20px) excluding click frames"
```

---

## Phase 5 — Generator integration + multiproc

### Task 12: Single-episode B0 generator

**Files:**
- Modify: `experiments/action_primitives/generator.py`
- Modify: `tests/action_primitives/test_generator.py`

Replace Phase A's single-button generator with a B0 generator that takes (scene, instruction, target, recovery config) and emits per-frame rows. Wraps the env + instruction-aware expert + recovery mechanisms.

- [ ] **Step 1: Write the failing test**

Append to `tests/action_primitives/test_generator.py`:
```python
import numpy as np
from experiments.action_primitives.generator import generate_one_b0_episode


def test_generate_b0_episode_no_recovery():
    rng = np.random.default_rng(42)
    rows = generate_one_b0_episode(episode_id=0, seed=42)
    assert len(rows) > 0
    # First row has all expected B0 fields
    r = rows[0]
    for field in ("episode_id", "frame_idx", "image_bytes",
                  "instruction", "target_button_id", "n_buttons",
                  "composite_tier", "is_adversarial",
                  "is_scenario_error", "scenario_type", "k_wrong_frames",
                  "is_dart_noisy_frame", "loss_mask"):
        assert field in r, f"missing field: {field}"
    # Episode-level metadata is consistent across rows
    assert all(r["instruction"] == rows[0]["instruction"] for r in rows)
    assert all(r["target_button_id"] == rows[0]["target_button_id"] for r in rows)


def test_generate_b0_scenario_error_episode():
    rng = np.random.default_rng(0)
    rows = generate_one_b0_episode(
        episode_id=1, seed=0, force_scenario_error=True, force_k_frames=10,
    )
    assert rows[0]["is_scenario_error"] == 1
    assert rows[0]["k_wrong_frames"] == 10
    # First k frames have loss_mask=0
    assert all(rows[i]["loss_mask"] == 0 for i in range(10))
    # Frames after k have loss_mask=1
    assert any(rows[i]["loss_mask"] == 1 for i in range(10, len(rows)))


def test_generate_b0_dart_noise_in_clean_episode():
    rng = np.random.default_rng(0)
    rows = generate_one_b0_episode(
        episode_id=2, seed=0, force_scenario_error=False, dart_p=0.5,
    )
    # With p=0.5, expect ~50% of nav frames to be DART-noisy
    nav_frames = [r for r in rows if r["loss_mask"] == 1]
    noisy_frames = [r for r in nav_frames if r["is_dart_noisy_frame"] == 1]
    assert len(noisy_frames) > 0, "expected at least some DART-noisy frames"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/action_primitives/test_generator.py::test_generate_b0_episode_no_recovery -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement B0 generator**

Append to `experiments/action_primitives/generator.py`:
```python
from experiments.action_primitives.scene import Scene, generate_scene, is_adversarial
from experiments.action_primitives.instructions import generate_instruction, maybe_typo
from experiments.action_primitives.expert import (
    InstructionAwareLClickExpert, LClickExpertConfig,
)
from experiments.action_primitives.recovery import (
    DART_PROB_DEFAULT, DART_SIGMA_DEFAULT, apply_dart_noise,
    generate_wrong_segment, sample_wrong_segment_type,
)


SCENARIO_ERROR_RATE_DEFAULT: float = 0.18  # ~15-20%


def _b0_action_to_row(a: Action) -> dict:
    return {
        "action_dx": float(a.dx),
        "action_dy": float(a.dy),
        "action_click": int(a.click),
        "action_scroll": float(a.scroll),
        "action_key_events": a.key_events.astype(np.int8).tolist(),
    }


def generate_one_b0_episode(
    episode_id: int,
    seed: int,
    primitive: str = "lclick",
    tempo: str | None = None,
    max_frames: int = ENV.max_frames_lclick,
    scenario_error_rate: float = SCENARIO_ERROR_RATE_DEFAULT,
    force_scenario_error: bool | None = None,
    force_k_frames: int | None = None,
    dart_p: float = DART_PROB_DEFAULT,
    dart_sigma: float = DART_SIGMA_DEFAULT,
) -> list[dict]:
    """Generate one B0 episode: scene + instruction + (optional) wrong segment + DART + clean expert.

    Per-frame rows include all metadata fields needed for B0 eval slicing and loss masking.
    """
    rng = np.random.default_rng(seed)
    tempo = tempo if tempo is not None else rng.choice(list(TEMPO_CHOICES))

    # Generate scene + instruction
    scene = generate_scene(rng=rng)
    instr_result = generate_instruction(scene, rng=rng)
    instruction = maybe_typo(instr_result.instruction, rng=rng)

    # Decide if this is a scenario-error episode
    if force_scenario_error is None:
        is_scenario = rng.random() < scenario_error_rate
    else:
        is_scenario = force_scenario_error

    # Build environment
    from experiments.action_primitives.env import LClickEnv
    env = LClickEnv(scene=scene, target_button_id=instr_result.target_button_id, seed=seed)
    obs, info = env.reset(seed=seed)
    cursor_xy = info["cursor_xy"]

    rows: list[dict] = []
    frame_idx = 0
    expert_iter = None
    done_frame: int | None = None
    env_done_frame: int | None = None

    # If scenario-error, run wrong segment first
    k_wrong = 0
    scenario_type = "none"
    if is_scenario:
        seg_type = sample_wrong_segment_type(rng=rng)
        if force_k_frames is not None:
            k_wrong = force_k_frames
        else:
            k_wrong = int(rng.integers(5, 16))  # [5, 15] inclusive
        scenario_type = seg_type
        seg = generate_wrong_segment(
            scene=scene,
            target_button_id=instr_result.target_button_id,
            cursor_xy=cursor_xy,
            segment_type=seg_type,
            k_frames=k_wrong,
            rng=rng,
        )
        for action in seg.actions:
            if frame_idx >= max_frames:
                break
            row = _b0_row(
                episode_id=episode_id, frame_idx=frame_idx,
                obs=obs, action=action, instruction=instruction,
                instr_result=instr_result, scene=scene,
                tempo=tempo, primitive=primitive,
                is_scenario_error=True, scenario_type=scenario_type, k_wrong_frames=k_wrong,
                is_dart_noisy=False, loss_mask=0,
                done_frame=done_frame,
            )
            rows.append(row)
            obs, env_done, info = env.step(action)
            if env_done and env_done_frame is None:
                env_done_frame = frame_idx
            frame_idx += 1

    # Build instruction-aware expert from current state
    cursor_xy = info["cursor_xy"]
    expert_cfg = LClickExpertConfig(tempo=tempo, seed=seed + 1)
    expert = InstructionAwareLClickExpert(
        cfg=expert_cfg, scene=scene,
        target_button_id=instr_result.target_button_id,
        cursor_xy=cursor_xy,
    )
    expert_iter = iter(expert)

    # Run clean expert with DART noise (loss_mask=1 throughout)
    while frame_idx < max_frames:
        if done_frame is None:
            try:
                clean_action = next(expert_iter)
            except StopIteration:
                done_frame = frame_idx
                clean_action = Action()
        else:
            clean_action = Action()  # no-op padding

        # DART noise (only on clean-phase, non-click frames)
        applied_action = clean_action
        was_dart_noisy = False
        if not is_scenario or frame_idx >= k_wrong:  # past wrong segment
            applied_action, was_dart_noisy = apply_dart_noise(
                clean_action, rng=rng, p=dart_p, sigma=dart_sigma,
            )

        row = _b0_row(
            episode_id=episode_id, frame_idx=frame_idx,
            obs=obs, action=applied_action, instruction=instruction,
            instr_result=instr_result, scene=scene,
            tempo=tempo, primitive=primitive,
            is_scenario_error=is_scenario, scenario_type=scenario_type,
            k_wrong_frames=k_wrong,
            is_dart_noisy=was_dart_noisy, loss_mask=1,
            done_frame=done_frame,
            # Note: training target is the CLEAN expert action, not the applied (noisy) one.
            # We store applied as action_*; clean as action_label_*
            clean_action=clean_action,
        )
        rows.append(row)
        obs, env_done, info = env.step(applied_action)
        if env_done and env_done_frame is None:
            env_done_frame = frame_idx
        frame_idx += 1

    # Stamp env_done_frame onto all rows
    sentinel = env_done_frame if env_done_frame is not None else -1
    for row in rows:
        row["env_done_frame"] = int(sentinel)

    return rows


def _b0_row(
    *, episode_id: int, frame_idx: int, obs: dict, action: Action,
    instruction: str, instr_result, scene: Scene,
    tempo: str, primitive: str,
    is_scenario_error: bool, scenario_type: str, k_wrong_frames: int,
    is_dart_noisy: bool, loss_mask: int,
    done_frame: int | None,
    clean_action: Action | None = None,
) -> dict:
    """Construct one parquet row with all B0 fields."""
    target = scene.buttons[instr_result.target_button_id]
    row = {
        "episode_id": int(episode_id),
        "frame_idx": int(frame_idx),
        "image_bytes": _image_to_jpeg_bytes(obs["image"]),
        "primitive_type": primitive,
        "tempo": tempo,
        "instruction": instruction,
        "target_button_id": int(instr_result.target_button_id),
        "target_bbox_x": int(target.x),
        "target_bbox_y": int(target.y),
        "target_bbox_w": int(target.w),
        "target_bbox_h": int(target.h),
        "n_buttons": int(len(scene.buttons)),
        "composite_tier": int(instr_result.composite_tier),
        "is_adversarial": int(is_adversarial(scene)),
        "is_scenario_error": int(is_scenario_error),
        "scenario_type": scenario_type,
        "k_wrong_frames": int(k_wrong_frames),
        "is_dart_noisy_frame": int(is_dart_noisy),
        "loss_mask": int(loss_mask),
        "done_gt": 1 if (done_frame is not None and frame_idx >= done_frame) else 0,
    }
    # action_applied (in env)
    row.update(_b0_action_to_row(action))
    # action_label (training target — clean expert action). On wrong-segment
    # frames there is no clean label; we use action zeros + loss_mask=0.
    if clean_action is not None:
        row["action_label_dx"] = float(clean_action.dx)
        row["action_label_dy"] = float(clean_action.dy)
        row["action_label_click"] = int(clean_action.click)
        row["action_label_scroll"] = float(clean_action.scroll)
        row["action_label_key_events"] = clean_action.key_events.astype(np.int8).tolist()
    else:
        # Placeholder for wrong-segment frames (loss_mask=0 means these aren't trained on)
        row["action_label_dx"] = 0.0
        row["action_label_dy"] = 0.0
        row["action_label_click"] = 0
        row["action_label_scroll"] = 0.0
        row["action_label_key_events"] = np.full(NUM_KEYS, 2, dtype=np.int8).tolist()
    # Proprio
    row.update(_proprio_to_row(obs["proprio"]))
    return row
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/action_primitives/test_generator.py -v`
Expected: PASS on the new tests; existing Phase A `generate_one_episode` tests should still pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/action_primitives/generator.py tests/action_primitives/test_generator.py
git commit -m "feat(exp6/b0): single-episode B0 generator (scene+instruction+wrong-segment+DART)"
```

---

### Task 13: Multiproc generator (worker-writes-shards)

**Files:**
- Modify: `experiments/action_primitives/generate_data.py`
- Test: `tests/action_primitives/test_generate_data.py` (create if missing)

- [ ] **Step 1: Write the failing test**

Create `tests/action_primitives/test_generate_data.py` (or extend existing):
```python
import os
from pathlib import Path
import pytest
import pandas as pd

from experiments.action_primitives.generate_data import (
    generate_dataset_multiproc,
)


def test_multiproc_generates_n_episodes_to_shards(tmp_path):
    out_dir = tmp_path / "test_b0_dataset"
    out_dir.mkdir()
    generate_dataset_multiproc(
        n_episodes=20, output_dir=out_dir, n_workers=2, seed=0,
        episodes_per_shard=10,
    )
    shards = sorted(out_dir.glob("shard_*.parquet"))
    assert len(shards) >= 2  # 20 eps / 10 per shard = 2 shards
    # All episodes present
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    assert df["episode_id"].nunique() == 20


def test_multiproc_rows_contain_b0_fields(tmp_path):
    out_dir = tmp_path / "test_b0_fields"
    out_dir.mkdir()
    generate_dataset_multiproc(
        n_episodes=5, output_dir=out_dir, n_workers=1, seed=0,
        episodes_per_shard=10,
    )
    shards = sorted(out_dir.glob("shard_*.parquet"))
    df = pd.read_parquet(shards[0])
    for field in ("instruction", "target_button_id", "n_buttons",
                  "composite_tier", "is_adversarial", "is_scenario_error",
                  "loss_mask", "is_dart_noisy_frame"):
        assert field in df.columns, f"missing field: {field}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/action_primitives/test_generate_data.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement worker-writes-shards generator**

Modify `experiments/action_primitives/generate_data.py`. Add at the top:
```python
import multiprocessing as mp
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from experiments.action_primitives.generator import generate_one_b0_episode
```

Append to file:
```python
def _worker_run_episodes(
    worker_id: int,
    episode_id_start: int,
    n_episodes_for_worker: int,
    output_dir: Path,
    base_seed: int,
    episodes_per_shard: int,
) -> tuple[int, int]:
    """Run a worker that generates n episodes and writes them as parquet shards.

    Each worker initializes its own pygame instance ONCE on first call to
    generate_one_b0_episode (which constructs LClickEnv, which initializes
    pygame). Subsequent episodes reuse the same pygame state.

    Returns (worker_id, n_episodes_written).
    """
    buffer: list[dict] = []
    shard_idx = 0
    episodes_in_shard = 0
    n_done = 0
    for offset in range(n_episodes_for_worker):
        episode_id = episode_id_start + offset
        seed = base_seed + episode_id  # deterministic per episode
        rows = generate_one_b0_episode(episode_id=episode_id, seed=seed)
        buffer.extend(rows)
        episodes_in_shard += 1
        n_done += 1
        if episodes_in_shard >= episodes_per_shard:
            _flush_shard(buffer, output_dir, worker_id, shard_idx)
            buffer = []
            episodes_in_shard = 0
            shard_idx += 1
    if buffer:
        _flush_shard(buffer, output_dir, worker_id, shard_idx)
    return worker_id, n_done


def _flush_shard(buffer: list[dict], output_dir: Path, worker_id: int, shard_idx: int) -> None:
    df = pd.DataFrame(buffer)
    out_path = output_dir / f"shard_w{worker_id:02d}_s{shard_idx:04d}.parquet"
    df.to_parquet(out_path, compression="zstd", index=False)


def generate_dataset_multiproc(
    n_episodes: int,
    output_dir: Path,
    n_workers: int = 4,
    seed: int = 0,
    episodes_per_shard: int = 200,
) -> None:
    """Generate n_episodes B0 episodes via worker-writes-shards multiproc.

    No IPC for episode payloads — each worker writes its own parquet shards.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eps_per_worker = n_episodes // n_workers
    remainder = n_episodes % n_workers

    args_list: list[tuple] = []
    cursor = 0
    for w in range(n_workers):
        n_for_w = eps_per_worker + (1 if w < remainder else 0)
        args_list.append((w, cursor, n_for_w, output_dir, seed, episodes_per_shard))
        cursor += n_for_w

    if n_workers == 1:
        # Skip multiprocessing for single-worker case — useful for tests/debug
        for args in args_list:
            _worker_run_episodes(*args)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.starmap(_worker_run_episodes, args_list)
            print(f"Workers completed: {results}")


# CLI entry point — extend existing argparse if present
def _main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-episodes", type=int, default=10000)
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("data/phase-b0-lclick"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes-per-shard", type=int, default=200)
    args = parser.parse_args()
    print(f"Generating {args.n_episodes} eps with {args.workers} workers → {args.output_dir}")
    t0 = time.time()
    generate_dataset_multiproc(
        n_episodes=args.n_episodes, output_dir=args.output_dir,
        n_workers=args.workers, seed=args.seed,
        episodes_per_shard=args.episodes_per_shard,
    )
    elapsed = time.time() - t0
    eps_per_sec = args.n_episodes / elapsed
    print(f"Done in {elapsed:.1f}s ({eps_per_sec:.1f} eps/s)")


if __name__ == "__main__":
    _main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/action_primitives/test_generate_data.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/action_primitives/generate_data.py tests/action_primitives/test_generate_data.py
git commit -m "feat(exp6/b0): worker-writes-shards multiproc generator (eliminates IPC bottleneck)"
```

---

## Phase 6 — Architecture changes (heads + losses)

### Task 14: Two parallel 3-way click heads

**Files:**
- Modify: `experiments/action_primitives/heads.py`
- Modify: `experiments/action_primitives/config.py`
- Modify: `tests/action_primitives/test_heads.py`

Replace the single 5-way click head with two parallel 3-way heads (left, right). Update `HEAD_LOGITS` accordingly.

- [ ] **Step 1: Write the failing test**

Modify/append to `tests/action_primitives/test_heads.py`:
```python
import torch
from experiments.action_primitives.heads import ActionHeads
from experiments.action_primitives.config import HEAD_LOGITS, MODEL


def test_action_heads_outputs_two_click_heads():
    heads = ActionHeads()
    queries = torch.randn(2, MODEL.n_queries, MODEL.d_model)
    out = heads(queries)
    # New B0 architecture: dx, dy, click_left, click_right, scroll, keys, done
    assert "click_left" in out
    assert "click_right" in out
    assert out["click_left"].shape == (2, 3)
    assert out["click_right"].shape == (2, 3)
    assert "click" not in out, "5-way click head should be removed"


def test_head_logits_total_matches_new_arch():
    # 21+21+3+3+21+231+1 = 301
    expected = 21 + 21 + 3 + 3 + 21 + 231 + 1
    actual = sum(HEAD_LOGITS.values())
    assert actual == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/action_primitives/test_heads.py -v`
Expected: FAIL — `click_left` not in output.

- [ ] **Step 3: Update HEAD_LOGITS and ActionHeads**

Modify `experiments/action_primitives/config.py`:
```python
# Click event split into two parallel 3-way heads (B0): {idle, press, release} per button
NUM_CLICK_EVENTS_PER_BUTTON: int = 3
CLICK_BTN_IDLE, CLICK_BTN_PRESS, CLICK_BTN_RELEASE = range(3)

# Phase A's combined 5-way constants kept for backward-compat in legacy code paths
# (e.g., env.step still emits 0-4 click event ids); training-side code uses
# CLICK_BTN_* with separate click_left/click_right logits.

HEAD_LOGITS = {
    "dx":          NUM_BINS_MOUSE,                # 21
    "dy":          NUM_BINS_MOUSE,                # 21
    "click_left":  NUM_CLICK_EVENTS_PER_BUTTON,   # 3
    "click_right": NUM_CLICK_EVENTS_PER_BUTTON,   # 3
    "scroll":      NUM_BINS_SCROLL,               # 21
    "keys":        NUM_KEYS * 3,                  # 231
    "done":        1,                             # binary
}
TOTAL_LOGITS: int = sum(HEAD_LOGITS.values())  # 301
```

`heads.py` requires no changes (it iterates HEAD_LOGITS dict).

Update `experiments/action_primitives/config.py` LossConfig to use focal_gamma=3.0:
```python
@dataclass(frozen=True)
class LossConfig:
    focal_gamma: float = 3.0           # B0 default (Phase A used 2.0)
    label_smoothing_mouse: float = 0.05
    idle_smoothing_keys: float = 0.05
    placeholder_weights: dict = field(default_factory=lambda: {
        "dx": 1.0, "dy": 1.0,
        "click_left": 1.0, "click_right": 1.0,
        "scroll": 1.0, "keys": 1.0, "done": 1.0,
    })
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/action_primitives/test_heads.py -v`
Expected: PASS. Some Phase A tests in test_losses.py / test_model.py may now fail because they reference the old "click" head name — those will be updated in subsequent tasks.

- [ ] **Step 5: Commit**

```bash
git add experiments/action_primitives/heads.py experiments/action_primitives/config.py tests/action_primitives/test_heads.py
git commit -m "feat(exp6/b0): split click head into click_left + click_right (3-way each); focal γ=3"
```

---

### Task 15: Soft-label CE for dx/dy/scroll

**Files:**
- Modify: `experiments/action_primitives/losses.py`
- Modify: `tests/action_primitives/test_losses.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/action_primitives/test_losses.py`:
```python
import torch
import numpy as np
from experiments.action_primitives.losses import soft_label_ce
from experiments.action_primitives.config import MOUSE_BIN_CENTERS


def test_soft_label_ce_concentrates_at_target_bin():
    """When expert is exactly on a bin center, soft label = hard label."""
    centers = torch.tensor(MOUSE_BIN_CENTERS, dtype=torch.float32)
    bin_idx = 10  # center bin (zero)
    expert_continuous = torch.tensor([centers[bin_idx].item()])
    # Use a uniform-ish logits, expect soft label to match hard target
    logits = torch.zeros(1, 21)
    logits[0, bin_idx] = 5.0  # heavily favor target bin
    loss = soft_label_ce(logits, expert_continuous, centers)
    # Loss should be very small (logits favor the correct bin)
    assert loss.item() < 0.5


def test_soft_label_ce_interpolates_between_bins():
    """When expert is between two bins, soft label is bimodal (2-bin triangular)."""
    centers = torch.tensor(MOUSE_BIN_CENTERS, dtype=torch.float32)
    # Pick a value between bin 10 (0.0) and bin 11 (~0.026)
    expert_continuous = torch.tensor([centers[10].item() * 0.5 + centers[11].item() * 0.5])
    # Logits: zero everywhere → uniform softmax → high loss
    logits1 = torch.zeros(1, 21)
    loss_uniform = soft_label_ce(logits1, expert_continuous, centers)
    # Logits matching the soft target (bins 10 + 11)
    logits2 = torch.zeros(1, 21)
    logits2[0, 10] = 3.0
    logits2[0, 11] = 3.0
    loss_matched = soft_label_ce(logits2, expert_continuous, centers)
    assert loss_matched < loss_uniform  # matching distribution gets lower loss
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/action_primitives/test_losses.py::test_soft_label_ce_concentrates_at_target_bin -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement soft_label_ce**

Append to `experiments/action_primitives/losses.py`:
```python
def soft_label_ce(
    logits: torch.Tensor,         # (B, num_bins)
    expert_continuous: torch.Tensor,  # (B,) float — expert action in continuous units
    bin_centers: torch.Tensor,    # (num_bins,) float — sorted ascending
) -> torch.Tensor:
    """2-bin triangular soft-label cross-entropy.

    For each sample, find the two bins bracketing expert_continuous and assign
    weights linearly interpolated between them. Example: bins [0, 1, 4, 10],
    expert=2.5 → bracketing bins {1, 2}, weight on bin 1 = (4-2.5)/(4-1) = 0.5,
    weight on bin 2 = (2.5-1)/(4-1) = 0.5.
    """
    B = logits.size(0)
    num_bins = bin_centers.size(0)
    # For each sample, find the rightmost bin <= expert (lower bracket)
    # `searchsorted` with side="right" gives the index of the first bin > expert.
    upper_idx = torch.searchsorted(bin_centers, expert_continuous, right=True)
    upper_idx = torch.clamp(upper_idx, 1, num_bins - 1)
    lower_idx = upper_idx - 1
    lower_centers = bin_centers[lower_idx]
    upper_centers = bin_centers[upper_idx]
    span = upper_centers - lower_centers
    span = torch.where(span > 1e-6, span, torch.ones_like(span))
    upper_w = (expert_continuous - lower_centers) / span
    lower_w = 1.0 - upper_w
    upper_w = torch.clamp(upper_w, 0.0, 1.0)
    lower_w = torch.clamp(lower_w, 0.0, 1.0)

    # Construct soft target (B, num_bins)
    soft_target = torch.zeros_like(logits)
    soft_target.scatter_(-1, lower_idx.unsqueeze(-1), lower_w.unsqueeze(-1))
    soft_target.scatter_add_(-1, upper_idx.unsqueeze(-1), upper_w.unsqueeze(-1))

    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(soft_target * log_probs).sum(dim=-1).mean()
    return loss
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/action_primitives/test_losses.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/action_primitives/losses.py tests/action_primitives/test_losses.py
git commit -m "feat(exp6/b0): soft_label_ce — 2-bin triangular soft-label CE for dx/dy/scroll"
```

---

### Task 16: Update total_loss for B0 architecture + loss_mask

**Files:**
- Modify: `experiments/action_primitives/losses.py`
- Modify: `tests/action_primitives/test_losses.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/action_primitives/test_losses.py`:
```python
def test_total_loss_b0_architecture():
    """total_loss should compute losses for click_left + click_right (no 'click')."""
    from experiments.action_primitives.losses import total_loss_b0
    B = 4
    head_logits = {
        "dx":          torch.randn(B, 21),
        "dy":          torch.randn(B, 21),
        "click_left":  torch.randn(B, 3),
        "click_right": torch.randn(B, 3),
        "scroll":      torch.randn(B, 21),
        "keys":        torch.randn(B, 231),
        "done":        torch.randn(B, 1),
    }
    targets = {
        "dx_continuous":      torch.randn(B),     # for soft-label CE
        "dy_continuous":      torch.randn(B),
        "scroll_continuous":  torch.randn(B),
        "click_left":         torch.randint(0, 3, (B,)),
        "click_right":        torch.randint(0, 3, (B,)),
        "keys":               torch.randint(0, 3, (B, 77)),
        "done":               torch.randint(0, 2, (B,)).float(),
    }
    loss_mask = torch.ones(B)
    head_weights = {n: 1.0 for n in head_logits}
    total, per_head = total_loss_b0(head_logits, targets, head_weights, loss_mask)
    assert torch.isfinite(total)
    assert "click_left" in per_head
    assert "click_right" in per_head


def test_total_loss_b0_zero_mask_zeros_loss():
    """All-zero loss_mask should produce zero total loss."""
    from experiments.action_primitives.losses import total_loss_b0
    B = 4
    head_logits = {
        "dx":          torch.randn(B, 21),
        "dy":          torch.randn(B, 21),
        "click_left":  torch.randn(B, 3),
        "click_right": torch.randn(B, 3),
        "scroll":      torch.randn(B, 21),
        "keys":        torch.randn(B, 231),
        "done":        torch.randn(B, 1),
    }
    targets = {
        "dx_continuous":      torch.randn(B),
        "dy_continuous":      torch.randn(B),
        "scroll_continuous":  torch.randn(B),
        "click_left":         torch.randint(0, 3, (B,)),
        "click_right":        torch.randint(0, 3, (B,)),
        "keys":               torch.randint(0, 3, (B, 77)),
        "done":               torch.randint(0, 2, (B,)).float(),
    }
    loss_mask = torch.zeros(B)
    head_weights = {n: 1.0 for n in head_logits}
    total, per_head = total_loss_b0(head_logits, targets, head_weights, loss_mask)
    assert total.item() == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/action_primitives/test_losses.py::test_total_loss_b0_architecture -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement total_loss_b0**

Append to `experiments/action_primitives/losses.py`:
```python
from experiments.action_primitives.config import (
    MOUSE_BIN_CENTERS, SCROLL_BIN_CENTERS, LOSS,
)


def _focal_ce_masked(
    logits: torch.Tensor, target: torch.Tensor,
    loss_mask: torch.Tensor, gamma: float, label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Per-sample focal CE multiplied by loss_mask; mean over masked samples."""
    logp = F.log_softmax(logits, dim=-1)
    if label_smoothing > 0.0:
        C = logits.size(-1)
        smooth = torch.full_like(logp, label_smoothing / (C - 1))
        smooth.scatter_(-1, target.unsqueeze(-1), 1.0 - label_smoothing)
        ce = -(smooth * logp).sum(dim=-1)
    else:
        ce = F.nll_loss(logp, target, reduction="none")
    p_true = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1).exp()
    focal_weight = (1.0 - p_true) ** gamma
    per_sample = focal_weight * ce  # (B,)
    masked = per_sample * loss_mask
    n_active = loss_mask.sum().clamp(min=1)
    return masked.sum() / n_active


def _soft_label_ce_masked(
    logits: torch.Tensor, expert_continuous: torch.Tensor,
    bin_centers: torch.Tensor, loss_mask: torch.Tensor,
) -> torch.Tensor:
    B = logits.size(0)
    num_bins = bin_centers.size(0)
    upper_idx = torch.searchsorted(bin_centers, expert_continuous, right=True)
    upper_idx = torch.clamp(upper_idx, 1, num_bins - 1)
    lower_idx = upper_idx - 1
    lower_centers = bin_centers[lower_idx]
    upper_centers = bin_centers[upper_idx]
    span = upper_centers - lower_centers
    span = torch.where(span > 1e-6, span, torch.ones_like(span))
    upper_w = torch.clamp((expert_continuous - lower_centers) / span, 0.0, 1.0)
    lower_w = 1.0 - upper_w
    soft_target = torch.zeros_like(logits)
    soft_target.scatter_(-1, lower_idx.unsqueeze(-1), lower_w.unsqueeze(-1))
    soft_target.scatter_add_(-1, upper_idx.unsqueeze(-1), upper_w.unsqueeze(-1))
    log_probs = F.log_softmax(logits, dim=-1)
    per_sample = -(soft_target * log_probs).sum(dim=-1)
    masked = per_sample * loss_mask
    n_active = loss_mask.sum().clamp(min=1)
    return masked.sum() / n_active


def _done_loss_masked(logits_done, target_done, loss_mask):
    logits = logits_done.squeeze(-1)
    bce = F.binary_cross_entropy_with_logits(logits, target_done.float(), reduction="none")
    p_true = torch.sigmoid(logits)
    p_t = torch.where(target_done.bool(), p_true, 1 - p_true)
    focal_weight = (1.0 - p_t) ** LOSS.focal_gamma
    per_sample = focal_weight * bce
    masked = per_sample * loss_mask
    n_active = loss_mask.sum().clamp(min=1)
    return masked.sum() / n_active


def _keys_focal_loss_masked(
    logits_keys, target_keys, loss_mask,
    gamma=2.0, idle_smoothing=0.05,
):
    B = logits_keys.size(0)
    logits = logits_keys.view(B, NUM_KEYS, 3)
    logp = F.log_softmax(logits, dim=-1)
    smooth = torch.full_like(logp, 0.0)
    smooth.scatter_(-1, target_keys.unsqueeze(-1), 1.0 - idle_smoothing)
    idle_slot = smooth[..., KEY_STATE_IDLE]
    mask_target_not_idle = (target_keys != KEY_STATE_IDLE).float()
    smooth[..., KEY_STATE_IDLE] = idle_slot + idle_smoothing * mask_target_not_idle
    target_is_idle = (target_keys == KEY_STATE_IDLE)
    smooth[..., KEY_STATE_IDLE] = torch.where(
        target_is_idle, torch.ones_like(smooth[..., KEY_STATE_IDLE]),
        smooth[..., KEY_STATE_IDLE],
    )
    ce = -(smooth * logp).sum(dim=-1)  # (B, NUM_KEYS)
    p_true = logp.gather(-1, target_keys.unsqueeze(-1)).squeeze(-1).exp()
    focal_weight = (1.0 - p_true) ** gamma
    per_sample = (focal_weight * ce).mean(dim=-1)  # (B,)
    masked = per_sample * loss_mask
    n_active = loss_mask.sum().clamp(min=1)
    return masked.sum() / n_active


def total_loss_b0(
    head_logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    head_weights: dict[str, float],
    loss_mask: torch.Tensor,            # (B,) float — 0 means skip, 1 means train
    bin_centers_mouse: torch.Tensor | None = None,
    bin_centers_scroll: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """B0 total loss: soft-CE on dx/dy/scroll, focal CE on click_left/click_right/keys, focal BCE on done."""
    if bin_centers_mouse is None:
        bin_centers_mouse = torch.tensor(MOUSE_BIN_CENTERS, device=head_logits["dx"].device)
    if bin_centers_scroll is None:
        bin_centers_scroll = torch.tensor(SCROLL_BIN_CENTERS, device=head_logits["scroll"].device)
    per_head = {
        "dx":          _soft_label_ce_masked(head_logits["dx"], targets["dx_continuous"], bin_centers_mouse, loss_mask),
        "dy":          _soft_label_ce_masked(head_logits["dy"], targets["dy_continuous"], bin_centers_mouse, loss_mask),
        "click_left":  _focal_ce_masked(head_logits["click_left"], targets["click_left"], loss_mask, LOSS.focal_gamma),
        "click_right": _focal_ce_masked(head_logits["click_right"], targets["click_right"], loss_mask, LOSS.focal_gamma),
        "scroll":      _soft_label_ce_masked(head_logits["scroll"], targets["scroll_continuous"], bin_centers_scroll, loss_mask),
        "keys":        _keys_focal_loss_masked(head_logits["keys"], targets["keys"], loss_mask, LOSS.focal_gamma, LOSS.idle_smoothing_keys),
        "done":        _done_loss_masked(head_logits["done"], targets["done"], loss_mask),
    }
    total = sum(head_weights[n] * per_head[n] for n in per_head)
    return total, per_head
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/action_primitives/test_losses.py -v`
Expected: PASS on new tests; existing Phase A `total_loss` tests should still pass (kept for backward-compat).

- [ ] **Step 5: Commit**

```bash
git add experiments/action_primitives/losses.py tests/action_primitives/test_losses.py
git commit -m "feat(exp6/b0): total_loss_b0 with loss_mask + soft-CE + dual click heads"
```

---

## Phase 7 — Dataset loader updates

### Task 17: Update PhaseAEpisodeDataset → PhaseB0EpisodeDataset

**Files:**
- Modify: `experiments/action_primitives/dataset.py`
- Modify: `tests/action_primitives/test_dataset.py`

The dataset must now yield: image, proprio, action_history (built from `actions_applied`), action_label (clean expert), loss_mask, instruction, plus all metadata fields.

- [ ] **Step 1: Write the failing test**

Modify `tests/action_primitives/test_dataset.py`:
```python
import torch
from pathlib import Path
import pandas as pd
from experiments.action_primitives.dataset import PhaseB0EpisodeDataset


def test_b0_dataset_yields_b0_fields(tmp_path):
    # Generate a tiny dataset
    from experiments.action_primitives.generate_data import generate_dataset_multiproc
    out_dir = tmp_path / "tiny"
    out_dir.mkdir()
    generate_dataset_multiproc(n_episodes=2, output_dir=out_dir, n_workers=1, seed=0,
                               episodes_per_shard=10)
    ds = PhaseB0EpisodeDataset(data_dir=out_dir, split="all")
    assert len(ds) == 2  # 2 episodes
    sample = ds[0]
    for field in ("images", "proprio", "action_history", "action_label",
                  "loss_mask", "instruction"):
        assert field in sample, f"missing field: {field}"
    # action_label should have separate click_left, click_right targets
    assert "click_left" in sample["action_label"]
    assert "click_right" in sample["action_label"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/action_primitives/test_dataset.py::test_b0_dataset_yields_b0_fields -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement PhaseB0EpisodeDataset**

The existing `PhaseAEpisodeDataset` in `experiments/action_primitives/dataset.py` is the structural template. Read it first, then add a new `PhaseB0EpisodeDataset` class adjacent to it. Key transformations from Phase A's logic to B0's:

```python
class PhaseB0EpisodeDataset(Dataset):
    """B0 episode dataset: distractor scenes + grounded instructions + recovery + loss masking."""

    # __init__ — same shard loading + episode_id hash split as PhaseAEpisodeDataset.
    # No new constructor args needed.

    def __getitem__(self, idx):
        # 1. Load episode rows (sorted by frame_idx) — same as Phase A.
        ep_rows = self._load_episode_rows(idx)
        T = len(ep_rows)

        # 2. Decode images — same as Phase A.
        images = self._decode_images(ep_rows)  # PIL list or pre-encoded tensors

        # 3. Build proprio tensor — same as Phase A.
        proprio = self._build_proprio(ep_rows)  # (T, 83)

        # 4. Build action_history vector — same as Phase A, but built from
        #    action_* columns (action_applied, possibly DART-noisy). DO NOT use
        #    action_label_* columns for history.
        action_history = self._build_history(ep_rows)  # (T, 8, 223) — uses action_* cols

        # 5. NEW — instruction string per episode (same value across rows; pull from row 0).
        instruction = ep_rows[0]["instruction"]

        # 6. NEW — loss_mask per frame.
        loss_mask = torch.tensor(
            [r["loss_mask"] for r in ep_rows], dtype=torch.float32,
        )  # (T,)

        # 7. Build LABEL targets from action_label_* columns (clean expert).
        #    On wrong-segment frames these are zero placeholders (loss_mask=0 means
        #    they don't contribute to loss).
        dx_continuous = torch.tensor(
            [r["action_label_dx"] for r in ep_rows], dtype=torch.float32,
        )
        dy_continuous = torch.tensor(
            [r["action_label_dy"] for r in ep_rows], dtype=torch.float32,
        )
        scroll_continuous = torch.tensor(
            [r["action_label_scroll"] for r in ep_rows], dtype=torch.float32,
        )

        # 8. Split the legacy 5-way action_label_click into two 3-way labels.
        click_left = []
        click_right = []
        for r in ep_rows:
            c = r["action_label_click"]  # 0=idle, 1=L_press, 2=L_release, 3=R_press, 4=R_release
            click_left.append(0 if c not in (1, 2) else (1 if c == 1 else 2))
            click_right.append(0 if c not in (3, 4) else (1 if c == 3 else 2))
        click_left = torch.tensor(click_left, dtype=torch.long)
        click_right = torch.tensor(click_right, dtype=torch.long)

        # 9. Keys + done labels (same as Phase A, from action_label_key_events / done_gt).
        keys = torch.tensor(
            [r["action_label_key_events"] for r in ep_rows], dtype=torch.long,
        )  # (T, 77)
        done = torch.tensor([r["done_gt"] for r in ep_rows], dtype=torch.float32)  # (T,)

        return {
            "images":         images,
            "proprio":        proprio,
            "action_history": action_history,
            "instruction":    instruction,
            "loss_mask":      loss_mask,
            "action_label": {
                "dx_continuous":     dx_continuous,
                "dy_continuous":     dy_continuous,
                "scroll_continuous": scroll_continuous,
                "click_left":        click_left,
                "click_right":       click_right,
                "keys":              keys,
                "done":              done,
            },
            # Episode-level metadata for eval-time slicing (passed through unchanged):
            "episode_id":       ep_rows[0]["episode_id"],
            "n_buttons":        ep_rows[0]["n_buttons"],
            "is_scenario_error": ep_rows[0]["is_scenario_error"],
            "is_adversarial":   ep_rows[0]["is_adversarial"],
            "composite_tier":   ep_rows[0]["composite_tier"],
        }
```

**Implementation notes:**

- The methods `_load_episode_rows`, `_decode_images`, `_build_proprio`, `_build_history` already exist in `PhaseAEpisodeDataset`. Either inherit from it (`class PhaseB0EpisodeDataset(PhaseAEpisodeDataset)`) and override `__getitem__`, OR copy the helper methods.
- The action-history vector dimension (223) is unchanged from Phase A. The history encoding uses `action_*` columns (applied actions), so DART-noisy frames produce noisy history entries — exactly the design.
- The `instruction` string flows to the model's text tower; the existing model forward already accepts an instruction kwarg in Phase A's text-fusion path (verify by reading `model.py`).
- For the action_label_click → (click_left, click_right) split: the helper logic above maps the 5-way enum cleanly. If the env later supports right-click, this same split works because we just look at which button is being pressed/released.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/action_primitives/test_dataset.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/action_primitives/dataset.py tests/action_primitives/test_dataset.py
git commit -m "feat(exp6/b0): PhaseB0EpisodeDataset — instruction + loss_mask + dual click labels"
```

---

## Phase 8 — Diagnostics

### Task 18: Per-class click recall metric

**Files:**
- Modify: `experiments/action_primitives/losses.py` (or create `metrics.py`)
- Test: `tests/action_primitives/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/action_primitives/test_metrics.py`:
```python
import torch
from experiments.action_primitives.metrics import per_class_click_recall


def test_per_class_click_recall():
    # 6 frames: 2 idle, 2 press, 2 release
    targets = torch.tensor([0, 0, 1, 1, 2, 2])
    # Predictions: idle correct on both, press correct 1/2, release correct 2/2
    preds = torch.tensor([0, 0, 1, 0, 2, 2])
    out = per_class_click_recall(preds, targets)
    assert out["recall_idle"] == 1.0
    assert out["recall_press"] == 0.5
    assert out["recall_release"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/action_primitives/test_metrics.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement metrics module**

Create `experiments/action_primitives/metrics.py`:
```python
"""Diagnostic metrics for Phase B0 training + eval."""
from __future__ import annotations

import torch


CLICK_BTN_CLASS_NAMES = ("idle", "press", "release")


def per_class_click_recall(
    preds: torch.Tensor,    # (N,) int 0..2
    targets: torch.Tensor,  # (N,) int 0..2
) -> dict[str, float]:
    """Per-class recall for the 3-way click button head."""
    out: dict[str, float] = {}
    for cls_idx, name in enumerate(CLICK_BTN_CLASS_NAMES):
        mask = targets == cls_idx
        n = int(mask.sum().item())
        if n == 0:
            out[f"recall_{name}"] = float("nan")
            continue
        correct = int(((preds == targets) & mask).sum().item())
        out[f"recall_{name}"] = correct / n
    return out


def soft_ce_diagnostics(
    logits: torch.Tensor,        # (N, num_bins)
    expert_continuous: torch.Tensor,  # (N,)
    bin_centers: torch.Tensor,   # (num_bins,)
) -> dict[str, float]:
    """Diagnostics for soft-CE failure modes."""
    probs = torch.softmax(logits, dim=-1)
    expected_value = (probs * bin_centers.unsqueeze(0)).sum(dim=-1)  # (N,)
    # Sign accuracy
    sign_match = (torch.sign(expected_value) == torch.sign(expert_continuous))
    sign_acc = sign_match.float().mean().item()
    # Entropy
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean().item()
    # Mass on wrong-sign bins
    expert_sign = torch.sign(expert_continuous)
    bin_signs = torch.sign(bin_centers).unsqueeze(0).expand_as(probs)
    wrong_sign_mass = torch.where(
        bin_signs != expert_sign.unsqueeze(-1), probs, torch.zeros_like(probs),
    ).sum(dim=-1).mean().item()
    # Expected-value endpoint error
    ev_l1 = (expected_value - expert_continuous).abs().mean().item()
    return {
        "sign_acc": sign_acc,
        "entropy_mean": entropy,
        "wrong_sign_mass_mean": wrong_sign_mass,
        "ev_l1_mean": ev_l1,
    }


def bin_10_frequency(
    pred_bins: torch.Tensor,         # (N,) int
    expert_continuous: torch.Tensor, # (N,) float
    bin_centers: torch.Tensor,       # (num_bins,)
    threshold_px: float = 5.0,
) -> float:
    """Fraction of frames where pred==bin 10 (zero) and |expert| > threshold_px."""
    significant_motion = expert_continuous.abs() > threshold_px
    if not significant_motion.any():
        return 0.0
    n_significant = int(significant_motion.sum().item())
    n_bin_10 = int(((pred_bins == 10) & significant_motion).sum().item())
    return n_bin_10 / n_significant
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/action_primitives/test_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/action_primitives/metrics.py tests/action_primitives/test_metrics.py
git commit -m "feat(exp6/b0): diagnostic metrics — per-class recall, soft-CE diagnostics, bin-10 freq"
```

---

### Task 19: Wire diagnostics into training val pass

**Files:**
- Modify: `experiments/action_primitives/train.py`

- [ ] **Step 1: Write the failing test**

(Test via integration — see Task 20's training run.)

- [ ] **Step 2: Add diagnostics to val pass**

Modify the validation pass section of `experiments/action_primitives/train.py` to compute and log the new metrics. The exact code depends on the current val-pass structure; key changes:

1. After computing per-head logits in val, run:
```python
from experiments.action_primitives.metrics import (
    per_class_click_recall, soft_ce_diagnostics, bin_10_frequency,
)
from experiments.action_primitives.config import MOUSE_BIN_CENTERS

# ... inside val pass ...
left_preds = head_logits["click_left"].argmax(dim=-1)
right_preds = head_logits["click_right"].argmax(dim=-1)
left_recall = per_class_click_recall(left_preds, targets["click_left"])
right_recall = per_class_click_recall(right_preds, targets["click_right"])
soft_dx = soft_ce_diagnostics(
    head_logits["dx"], targets["dx_continuous"],
    torch.tensor(MOUSE_BIN_CENTERS, device=head_logits["dx"].device),
)
soft_dy = soft_ce_diagnostics(
    head_logits["dy"], targets["dy_continuous"],
    torch.tensor(MOUSE_BIN_CENTERS, device=head_logits["dy"].device),
)
bin10_dx = bin_10_frequency(
    head_logits["dx"].argmax(dim=-1), targets["dx_continuous"],
    torch.tensor(MOUSE_BIN_CENTERS, device=head_logits["dx"].device),
)
bin10_dy = bin_10_frequency(
    head_logits["dy"].argmax(dim=-1), targets["dy_continuous"],
    torch.tensor(MOUSE_BIN_CENTERS, device=head_logits["dy"].device),
)

# Log to W&B
wandb.log({
    "val/click_left/recall_idle":     left_recall["recall_idle"],
    "val/click_left/recall_press":    left_recall["recall_press"],
    "val/click_left/recall_release":  left_recall["recall_release"],
    "val/click_right/recall_idle":    right_recall["recall_idle"],
    "val/click_right/recall_press":   right_recall["recall_press"],
    "val/click_right/recall_release": right_recall["recall_release"],
    "val/dx/sign_acc":             soft_dx["sign_acc"],
    "val/dx/entropy":              soft_dx["entropy_mean"],
    "val/dx/wrong_sign_mass":      soft_dx["wrong_sign_mass_mean"],
    "val/dx/ev_l1":                soft_dx["ev_l1_mean"],
    "val/dy/sign_acc":             soft_dy["sign_acc"],
    "val/dy/entropy":              soft_dy["entropy_mean"],
    "val/dy/wrong_sign_mass":      soft_dy["wrong_sign_mass_mean"],
    "val/dy/ev_l1":                soft_dy["ev_l1_mean"],
    "val/dx/bin_10_freq":          bin10_dx,
    "val/dy/bin_10_freq":          bin10_dy,
}, step=global_step)
```

2. Add instruction-grounding probes (zero/shuffled/wrong) — see Task 23 for code; bind into val pass when probes exist.

- [ ] **Step 3: Smoke test the val-pass code**

Run a 2-episode mini-run to verify metrics log without errors:
```bash
# Use the multiproc generator to make a tiny dataset, then call train.py with --max-steps 5
```

- [ ] **Step 4: Commit**

```bash
git add experiments/action_primitives/train.py
git commit -m "feat(exp6/b0): wire per-class recall + soft-CE diagnostics + bin-10 freq into val pass"
```

---

## Phase 9 — Eval extensions

### Task 20: Metadata-based eval slicing

**Files:**
- Modify: `experiments/action_primitives/evaluate.py`
- Modify: `tests/action_primitives/` (extend existing or create test_evaluate.py)

- [ ] **Step 1: Write the failing test**

Add to `tests/action_primitives/test_evaluate.py`:
```python
import pandas as pd
from experiments.action_primitives.evaluate import filter_eval_split


def test_filter_phase_a_holdout(tmp_path):
    # Build minimal df with required fields
    df = pd.DataFrame([
        {"episode_id": 0, "n_buttons": 1, "is_scenario_error": 0, "is_adversarial": 0, "composite_tier": 1},
        {"episode_id": 1, "n_buttons": 3, "is_scenario_error": 0, "is_adversarial": 0, "composite_tier": 1},
        {"episode_id": 2, "n_buttons": 1, "is_scenario_error": 1, "is_adversarial": 0, "composite_tier": 1},
    ])
    out = filter_eval_split(df, slice_name="phase_a_holdout")
    assert set(out["episode_id"]) == {0}


def test_filter_multi_btn_generic(tmp_path):
    df = pd.DataFrame([
        {"episode_id": 0, "n_buttons": 1, "is_scenario_error": 0, "is_adversarial": 0, "composite_tier": 1},
        {"episode_id": 1, "n_buttons": 3, "is_scenario_error": 0, "is_adversarial": 0, "composite_tier": 1},
        {"episode_id": 2, "n_buttons": 3, "is_scenario_error": 0, "is_adversarial": 0, "composite_tier": 2},
        {"episode_id": 3, "n_buttons": 3, "is_scenario_error": 0, "is_adversarial": 1, "composite_tier": 1},
    ])
    out = filter_eval_split(df, slice_name="multi_btn_generic")
    assert set(out["episode_id"]) == {1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/action_primitives/test_evaluate.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement filter_eval_split**

Append to `experiments/action_primitives/evaluate.py`:
```python
def filter_eval_split(df, slice_name: str):
    """Filter a parquet DataFrame to a B0 eval slice."""
    if slice_name == "phase_a_holdout":
        return df[(df["n_buttons"] == 1) & (df["is_scenario_error"] == 0)]
    elif slice_name == "multi_btn_generic":
        return df[(df["n_buttons"] >= 2) & (df["composite_tier"] == 1) & (df["is_adversarial"] == 0)]
    elif slice_name == "multi_btn_composite":
        return df[(df["n_buttons"] >= 2) & (df["composite_tier"] >= 2) & (df["is_adversarial"] == 0)]
    elif slice_name == "scenario_recovery":
        return df[df["is_scenario_error"] == 1]
    elif slice_name == "adversarial":
        return df[df["is_adversarial"] == 1]
    else:
        raise ValueError(f"Unknown slice: {slice_name}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/action_primitives/test_evaluate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/action_primitives/evaluate.py tests/action_primitives/test_evaluate.py
git commit -m "feat(exp6/b0): metadata-based eval slicing (5 slices)"
```

---

### Task 21: Adversarial tier sub-slicing

Identify which attribute(s) are required to disambiguate the target.

**Files:**
- Modify: `experiments/action_primitives/evaluate.py` (and possibly add `adversarial_tier` field at gen time)
- Test: extend `test_evaluate.py`

- [ ] **Step 1: Write the failing test**

Append to `test_evaluate.py`:
```python
def test_adversarial_tier_color_ambiguous():
    """A scene where 2 buttons share color but differ on shape → color is ambiguous; shape disambiguates."""
    from experiments.action_primitives.evaluate import classify_adversarial_tier
    from experiments.action_primitives.scene import Scene, Button
    scene = Scene(
        buttons=(
            Button(0, "red", "circle", "small", (0, 0), 10, 10, 40, 40),
            Button(1, "red", "square", "small", (1, 0), 100, 10, 40, 40),
        ),
        decorative_shapes=(),
        bg_color=(245, 245, 248),
    )
    target_id = 0
    # Used attrs: shape (since color is ambiguous)
    used_attrs = ("shape",)
    tier = classify_adversarial_tier(scene, target_id, used_attrs)
    assert tier == "color-ambiguous"
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL with ImportError.

- [ ] **Step 3: Implement classifier**

```python
def classify_adversarial_tier(scene, target_id, used_attrs):
    """Identify which attribute caused the need for composite (i.e., is ambiguous within scene)."""
    target = scene.buttons[target_id]
    ambiguous_attrs = []
    for attr in ("color", "shape", "size", "position"):
        target_val = _attr_value_human_or_zone(target, attr)
        for j, other in enumerate(scene.buttons):
            if j == target_id:
                continue
            if _attr_value_human_or_zone(other, attr) == target_val:
                ambiguous_attrs.append(attr)
                break
    # Single-attribute composite: target needs one specific attribute used
    if len(used_attrs) == 1 and len(ambiguous_attrs) == 0:
        return "single-unique"
    if len(used_attrs) == 1:
        return f"{ambiguous_attrs[0]}-ambiguous"
    if len(used_attrs) == 2:
        return "2-attr-needed"
    if len(used_attrs) >= 3:
        return "3-attr-needed"
    return "uncategorized"


def _attr_value_human_or_zone(b, attr):
    if attr == "position": return b.pos_zone
    if attr == "color": return b.color
    if attr == "shape": return b.shape
    if attr == "size": return b.size
```

Also add `used_attrs` and `adversarial_tier` columns to per-episode metadata in `_b0_row` (modify Task 12's row construction):
- Add `used_attrs` (str, joined with ",")
- Compute `adversarial_tier` once per episode (when generating instruction) and stamp on each row.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/action_primitives/test_evaluate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/action_primitives/evaluate.py experiments/action_primitives/generator.py tests/action_primitives/test_evaluate.py
git commit -m "feat(exp6/b0): adversarial tier classification (color/shape/size/position-ambiguous, n-attr-needed)"
```

---

### Task 22: Wrong-direction-first-3-frames metric

**Files:**
- Modify: `experiments/action_primitives/evaluate.py`

- [ ] **Step 1: Write the failing test**

Append to `test_evaluate.py`:
```python
def test_wrong_direction_first_3_frames():
    from experiments.action_primitives.evaluate import compute_wrong_direction_first_3_frames
    cursor_xys_3frames = [(100.0, 100.0), (90.0, 90.0), (80.0, 80.0)]  # heading away from target at (200, 200)
    target_xy = (200.0, 200.0)
    out = compute_wrong_direction_first_3_frames(cursor_xys_3frames, target_xy)
    assert out is True

    # Heading toward target
    cursor_xys_correct = [(100.0, 100.0), (130.0, 130.0), (160.0, 160.0)]
    out2 = compute_wrong_direction_first_3_frames(cursor_xys_correct, target_xy)
    assert out2 is False
```

- [ ] **Step 2: Implement metric**

```python
def compute_wrong_direction_first_3_frames(
    cursor_xys: list[tuple[float, float]],
    target_xy: tuple[float, float],
) -> bool:
    """Returns True if cursor moves AWAY from target (net) over the first 3 frames."""
    if len(cursor_xys) < 3:
        return False
    start = cursor_xys[0]
    end = cursor_xys[2]
    start_dist = ((target_xy[0] - start[0])**2 + (target_xy[1] - start[1])**2)**0.5
    end_dist = ((target_xy[0] - end[0])**2 + (target_xy[1] - end[1])**2)**0.5
    return end_dist > start_dist  # got farther from target
```

- [ ] **Step 3: Wire into closed-loop eval**

In `evaluate.py`'s closed-loop rollout function, capture cursor positions and compute the metric across all rollouts in a slice.

- [ ] **Step 4: Run test + commit**

```bash
uv run pytest tests/action_primitives/test_evaluate.py -v
git add experiments/action_primitives/evaluate.py tests/action_primitives/test_evaluate.py
git commit -m "feat(exp6/b0): wrong-direction-first-3-frames metric for closed-loop eval"
```

---

### Task 23: Instruction probes (zero/shuffled/wrong)

**Files:**
- Modify: `experiments/action_primitives/evaluate.py`
- Modify: `experiments/action_primitives/model.py` (add `instruction_override` kwarg to forward)
- Test: `tests/action_primitives/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

```python
def test_instruction_probe_zero():
    """Zero-instruction probe replaces text embedding with zeros."""
    from experiments.action_primitives.evaluate import build_zero_instruction_embedding
    emb_dim = 768
    out = build_zero_instruction_embedding(emb_dim=emb_dim)
    import torch
    assert out.shape == (1, emb_dim)
    assert torch.allclose(out, torch.zeros_like(out))
```

- [ ] **Step 2: Implement probes**

Append to `evaluate.py`:
```python
def build_zero_instruction_embedding(emb_dim: int = 768) -> torch.Tensor:
    return torch.zeros(1, emb_dim)


def build_shuffled_instruction(rng, val_instructions: list[str]) -> str:
    """Return a random other instruction from val set."""
    return val_instructions[rng.randint(0, len(val_instructions))]


def build_wrong_instruction(scene, target_id, rng) -> str:
    """Return an instruction that targets a DIFFERENT button than the rendered correct one."""
    other_buttons = [i for i in range(len(scene.buttons)) if i != target_id]
    if not other_buttons:
        return ""  # only one button, can't construct wrong
    fake_target_id = rng.choice(other_buttons)
    from experiments.action_primitives.instructions import generate_instruction
    # Force generate_instruction to pick fake_target_id — we'd need to extend it for this
    # For B0, simpler: re-use the existing instruction format with different button's attrs
    fake_target = scene.buttons[fake_target_id]
    return f"click the {fake_target.color} {fake_target.shape}"
```

In closed-loop eval, run all three probes by passing different instructions/embeddings to the model.

- [ ] **Step 3: Run test + commit**

```bash
uv run pytest tests/action_primitives/test_evaluate.py -v
git add experiments/action_primitives/evaluate.py experiments/action_primitives/model.py tests/action_primitives/test_evaluate.py
git commit -m "feat(exp6/b0): instruction probes — zero / shuffled / wrong"
```

---

## Phase 10 — HF Jobs setup

### Task 24: Update HF Jobs launcher for a100-large

**Files:**
- Modify: `scripts/launch_hf_job_exp6.py`
- Modify: `scripts/hf_job_train_exp6.py`

- [ ] **Step 1: Update launcher defaults**

Modify `scripts/launch_hf_job_exp6.py` to default `--flavor a100-large`. Update any docstrings to mention B0.

- [ ] **Step 2: Update training entry script branch**

In `scripts/hf_job_train_exp6.py`, change the default cloned branch from `feat/exp6-phase-a` to `feat/exp6-phase-b0` (or `feat/exp6-phase-b` if implementing on the long-running branch directly).

- [ ] **Step 3: Verify launcher can be invoked**

Dry-run:
```bash
uv run python scripts/launch_hf_job_exp6.py --dry-run --flavor a100-large -- --epochs 10 --micro-batch-episodes 16 --num-workers 8
```
Expected: prints the resolved job spec without launching.

- [ ] **Step 4: Commit**

```bash
git add scripts/launch_hf_job_exp6.py scripts/hf_job_train_exp6.py
git commit -m "chore(exp6/b0): default HF Jobs launcher to a100-large + clone phase-b branch"
```

---

## Phase 11 — Data generation + upload

### Task 25: Generate 10K episodes (multiproc)

- [ ] **Step 1: Run small smoke test (50 episodes, 2 workers)**

```bash
uv run python -m experiments.action_primitives.generate_data \
  -n 50 -o data/phase-b0-smoke --workers 2 --episodes-per-shard 25
```
Expected: completes in ~30 sec, produces 2-4 parquet shards. No errors.

- [ ] **Step 2: Inspect smoke-test output**

```bash
uv run python -c "
import pandas as pd
df = pd.read_parquet('data/phase-b0-smoke', engine='pyarrow')
print(df.columns.tolist())
print('Episodes:', df.episode_id.nunique())
print('Scenario error frac:', df.groupby('episode_id').is_scenario_error.first().mean())
print('DART noisy frame frac:', df.is_dart_noisy_frame.mean())
print('Sample instruction:', df.instruction.iloc[0])
print('Adversarial frac:', df.groupby('episode_id').is_adversarial.first().mean())
print('n_buttons distribution:', df.groupby('episode_id').n_buttons.first().value_counts())
"
```

Expected: scenario error ~15-20%, DART noisy ~6-8% of frames, adversarial ~25%, n_buttons uniform 1-6.

- [ ] **Step 3: Generate full 10K episodes**

```bash
uv run python -m experiments.action_primitives.generate_data \
  -n 10000 -o data/phase-b0-lclick --workers 8 --episodes-per-shard 500
```
Expected: completes in ~7-10 min on 8-core M1.

- [ ] **Step 4: Verify integrity**

```bash
uv run python -c "
from pathlib import Path
import pandas as pd
shards = sorted(Path('data/phase-b0-lclick').glob('shard_*.parquet'))
print(f'{len(shards)} shards')
df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
print(f'Total episodes: {df.episode_id.nunique()}')
print(f'Total frames: {len(df)}')
"
```

Expected: 10000 episodes, ~300K-450K frames.

- [ ] **Step 5: Commit (no code change, just record the dataset stats)**

(No commit; data is gitignored. Record stats in next step's PR description.)

---

### Task 26: Create HF dataset repo + upload

- [ ] **Step 1: Create the repo**

```bash
uv run python -c "from huggingface_hub import create_repo; create_repo('PenTest-duck/cu-vla-exp6-b0-lclick', repo_type='dataset', exist_ok=True)"
```

- [ ] **Step 2: Upload via hf_sync**

```bash
uv run python -c "
from experiments.action_primitives.hf_sync import upload_parquet_dir
from pathlib import Path
upload_parquet_dir(Path('data/phase-b0-lclick'), 'PenTest-duck/cu-vla-exp6-b0-lclick', 'dataset')
"
```

- [ ] **Step 3: Verify upload**

```bash
uv run python -c "
from datasets import load_dataset
ds = load_dataset('PenTest-duck/cu-vla-exp6-b0-lclick', streaming=True)
sample = next(iter(ds['train']))
print(list(sample.keys()))
"
```

Expected: schema includes all B0 fields.

---

## Phase 12 — Training

### Task 27: Launch training run on a100-large

- [ ] **Step 1: Create checkpoint repo**

```bash
uv run python -c "from huggingface_hub import create_repo; create_repo('PenTest-duck/cu-vla-exp6-b0-ckpt', repo_type='model', exist_ok=True)"
```

- [ ] **Step 2: Set environment variables**

```bash
export WANDB_API_KEY=<your_key>
export HF_TOKEN=<your_token>
```

- [ ] **Step 3: Launch HF Job**

```bash
uv run python scripts/launch_hf_job_exp6.py --flavor a100-large --timeout 4h \
  -- \
  --hf-data-repo PenTest-duck/cu-vla-exp6-b0-lclick \
  --epochs 10 \
  --micro-batch-episodes 16 \
  --num-workers 8 \
  --hf-upload-repo PenTest-duck/cu-vla-exp6-b0-ckpt \
  --wandb-run-name phase-b0-combined \
  --ckpt-every-steps 100 \
  --eval-every-steps 50 \
  --early-stop-patience 3
```

- [ ] **Step 4: Monitor W&B during run**

Watch for:
- Aggregate train loss decreasing
- Per-class click recall climbing (`val/click_left/recall_press` is the headline)
- Soft-CE diagnostics: `sign_acc` ≥ 0.95, `wrong_sign_mass` < 0.05
- Bin-10 freq < 0.20 on dx/dy
- Instruction-probe val losses: zero/shuffled/wrong should be MUCH higher than baseline

- [ ] **Step 5: Pull best.pt to local**

```bash
uv run python -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download('PenTest-duck/cu-vla-exp6-b0-ckpt', 'best.pt')
print(p)
"
```

---

## Phase 13 — Closed-loop evaluation on M1

### Task 28: Run all eval slices

Run sequentially for each slice. Probabilistic decode is the default.

- [ ] **Step 1: Phase-A holdout slice**

```bash
uv run python -m experiments.action_primitives.evaluate \
  --checkpoint <local_path_to_best.pt> \
  --data-dir data/phase-b0-lclick \
  --slice phase_a_holdout \
  --n-rollouts 200 \
  --device mps \
  --decode expected
```

Expected: ~12 min wall-clock. Output: `closed_loop_success`, `l_press_recall`, `tolerance_curve_at_{0,3,5,10}px`.

- [ ] **Step 2: Multi-btn generic slice**

```bash
uv run python -m experiments.action_primitives.evaluate \
  --checkpoint <local_path> --data-dir data/phase-b0-lclick \
  --slice multi_btn_generic --n-rollouts 200 --device mps --decode expected
```

- [ ] **Step 3: Multi-btn composite slice**

Same as above with `--slice multi_btn_composite`.

- [ ] **Step 4: Scenario-recovery slice**

Same with `--slice scenario_recovery`.

- [ ] **Step 5: Adversarial subset (with tier breakdown)**

```bash
uv run python -m experiments.action_primitives.evaluate \
  --checkpoint <local_path> --data-dir data/phase-b0-lclick \
  --slice adversarial --n-rollouts 200 --device mps --decode expected \
  --report-by-tier
```

Expected: per-tier success rates printed for color-ambiguous / shape-ambiguous / size-ambiguous / position-ambiguous / 2-attr-needed / 3-attr-needed.

- [ ] **Step 6: Instruction probes**

```bash
# Zero
uv run python -m experiments.action_primitives.evaluate \
  --checkpoint <local_path> --data-dir data/phase-b0-lclick \
  --slice multi_btn_generic --n-rollouts 200 --device mps --decode expected \
  --instruction-probe zero

# Shuffled
uv run python -m experiments.action_primitives.evaluate \
  ... --instruction-probe shuffled

# Wrong
uv run python -m experiments.action_primitives.evaluate \
  ... --instruction-probe wrong
```

- [ ] **Step 7: Save all results to a JSON for the writeup**

```bash
mkdir -p docs/experiments/6-action-primitives-phase-b-results
# (each eval invocation writes to a slice-specific JSON file in this dir)
```

---

### Task 29: Apply typed-disposition gate

- [ ] **Step 1: Compute pass/fail per criterion**

Based on results from Task 28, evaluate against B0 design doc gates:

| Criterion | Threshold | Result | Pass? |
|---|---|---|---|
| Phase-A holdout closed-loop | ≥0.92 | __ | __ |
| Phase-A holdout l_press recall | ≥0.90 | __ | __ |
| Multi-btn (generic + composite) | ≥0.85 | __ | __ |
| Wrong-instruction degradation | ≥40 pp | __ | __ |
| Adversarial subset | ≥0.75 | __ | __ |
| Scenario-recovery | ≥0.80 | __ | __ |

- [ ] **Step 2: Determine disposition**

If ALL pass → ship to B1.
If Phase-A holdout fails → **Phase-A-regression**: investigate loss/head bug.
If multi-btn fails AND probes show strong degradation → **Motor-fail**: try focal γ sweep / longer training / larger dataset.
If multi-btn fails AND probes show weak degradation → **Grounding-fail**: investigate text-tower fusion path.
If only scenario-recovery fails → **Recovery-fail**: bump scenario error rate / DART σ.

---

## Phase 14 — Write-up + PR

### Task 30: Write spike-b0-combined.md

**Files:**
- Create: `docs/experiments/6-action-primitives-phase-b-results/spike-b0-combined.md`

- [ ] **Step 1: Draft the writeup**

Follow the structure of Phase A's `spike-b-lclick-end-to-end.md`:
1. Question + run metadata (commit, W&B run, HF Jobs run id)
2. Training config table
3. Offline eval results (per-head accuracy)
4. Closed-loop eval results (5 slices + 3 probes + adversarial tier breakdown)
5. Failure-mode analysis (visual sample of wrong-direction / press-timing / etc.)
6. Verdict (PASS / FAIL by typed-disposition table)
7. Recommendations for B1 (or remediation if failed)

- [ ] **Step 2: Commit**

```bash
git add docs/experiments/6-action-primitives-phase-b-results/spike-b0-combined.md
git commit -m "results(exp6/b0): combined L-click + V+L grounding spike — [PASS|FAIL]"
```

---

### Task 31: Update design doc with Phase B0 amendments

**Files:**
- Modify: `docs/experiments/6-action-primitives.md` (Amendments section)

- [ ] **Step 1: Add Phase B0 findings**

Append a new "## Amendments (Phase B0 findings, YYYY-MM-DD)" section to the design doc with any design changes informed by B0 results (e.g., "click head split worked / didn't work, here's the revised approach for B1").

- [ ] **Step 2: Commit**

```bash
git add docs/experiments/6-action-primitives.md
git commit -m "docs(exp6/b0): design doc amendments from B0 findings"
```

---

### Task 32: Open PR for Phase B0

- [ ] **Step 1: Push final state**

```bash
git push -u origin feat/exp6-phase-b0
```

- [ ] **Step 2: Open PR**

```bash
gh pr create --base feat/exp6-phase-b --head feat/exp6-phase-b0 \
  --title "Phase B0: combined L-click hardening + V+L grounding" \
  --body "$(cat <<'EOF'
## Summary
Phase B0 implementation per [B0 design doc](docs/plans/2026-04-25-action-primitives-phase-b0-design.md).

- Two parallel 3-way click heads (left, right) replacing 5-way
- Soft-label CE on dx/dy/scroll
- Distractor scenes (1-6 buttons, 4 attributes, 25% adversarial)
- Instruction-aware expert + grounding via text tower
- Recovery via start-chunk scenarios + DART action noise
- 5 diagnostic eval slices + 3 instruction probes + typed-disposition gate

## Results
[See spike-b0-combined.md for detail]

## Test plan
- [ ] Verify all unit tests pass: `uv run pytest tests/action_primitives/`
- [ ] Verify offline per-head accuracy ≥ 0.9 (loss-side fixes)
- [ ] Verify closed-loop on Phase-A holdout ≥ 0.92
- [ ] Verify wrong-instruction degradation ≥ 40 pp

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review checklist

Before merging, verify:

- [ ] All 71 Phase A unit tests still pass on `feat/exp6-phase-b0`.
- [ ] All new B0 unit tests pass.
- [ ] No references to the old `click` head in B0 code (should be `click_left` / `click_right` everywhere).
- [ ] `loss_mask` is honored in `total_loss_b0` (verified by zero-mask test).
- [ ] DART noise excludes click frames (verified by test).
- [ ] Instruction generator produces unique-target instructions (verified by uniqueness invariant test).
- [ ] Multiproc generator produces 25+ eps/s on 8-core M1 (verified during smoke test).
- [ ] Eval slicing produces non-empty subsets for all 5 slices (verified during eval).

---

## References

- [Phase B0 design doc](2026-04-25-action-primitives-phase-b0-design.md)
- [Phase A Spike B writeup](../experiments/6-action-primitives-phase-a-results/spike-b-lclick-end-to-end.md)
- [Phase A summary](../experiments/6-action-primitives-phase-a-results/PHASE-A-SUMMARY.md)
- [HF Jobs gotchas](../research/hf-jobs-gotchas.md)
