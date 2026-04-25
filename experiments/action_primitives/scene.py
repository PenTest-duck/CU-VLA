"""Scene primitives for Phase B0: distractor-aware buttons and decorative shapes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.action_primitives.config import (
    B0_BG_COLORS, B0_COLORS, B0_POSITION_GRID, B0_SHAPES, B0_SIZES, ENV,
)


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

    def __post_init__(self) -> None:
        if self.color not in B0_COLORS:
            raise ValueError(
                f"Button color {self.color!r} not in B0_COLORS "
                f"(allowed: {sorted(B0_COLORS)})"
            )
        if self.shape not in B0_SHAPES:
            raise ValueError(
                f"Button shape {self.shape!r} not in B0_SHAPES "
                f"(allowed: {list(B0_SHAPES)})"
            )
        if self.size not in B0_SIZES:
            raise ValueError(
                f"Button size {self.size!r} not in B0_SIZES "
                f"(allowed: {sorted(B0_SIZES)})"
            )
        cols, rows = B0_POSITION_GRID
        col, row = self.pos_zone
        if not (0 <= col < cols and 0 <= row < rows):
            raise ValueError(
                f"Button pos_zone {self.pos_zone!r} out of bounds for "
                f"B0_POSITION_GRID {B0_POSITION_GRID} "
                f"(must satisfy 0 <= col < {cols} and 0 <= row < {rows})"
            )
        if self.shape == "square" and self.w != self.h:
            raise ValueError(
                f"Button shape='square' requires w == h; got w={self.w}, h={self.h}"
            )

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

    def __post_init__(self) -> None:
        if self.color not in B0_COLORS:
            raise ValueError(
                f"DecorativeShape color {self.color!r} not in B0_COLORS "
                f"(allowed: {sorted(B0_COLORS)})"
            )
        if self.shape not in B0_SHAPES:
            raise ValueError(
                f"DecorativeShape shape {self.shape!r} not in B0_SHAPES "
                f"(allowed: {list(B0_SHAPES)})"
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


def _overlaps_any_button(x: int, y: int, w: int, h: int, buttons: list[Button]) -> bool:
    """Axis-aligned bounding box overlap test."""
    for b in buttons:
        if (x < b.x + b.w and x + w > b.x and
            y < b.y + b.h and y + h > b.y):
            return True
    return False


def generate_scene(
    rng: np.random.Generator,
    n_buttons: int | None = None,
    n_decorative: int | None = None,
) -> Scene:
    """Generate a scene with 1-6 buttons + 0-3 decorative shapes + random bg."""
    cols, rows = B0_POSITION_GRID
    max_buttons = cols * rows
    if n_buttons is not None and not (1 <= n_buttons <= max_buttons):
        raise ValueError(f"n_buttons must be in [1, {max_buttons}], got {n_buttons}")
    if n_decorative is not None and n_decorative < 0:
        raise ValueError(f"n_decorative must be >= 0, got {n_decorative}")

    if n_buttons is None:
        n_buttons = int(rng.integers(1, 7))  # 1..6 inclusive
    if n_decorative is None:
        n_decorative = int(rng.integers(0, 4))  # 0..3 inclusive

    bg_color = tuple(B0_BG_COLORS[rng.integers(0, len(B0_BG_COLORS))])

    # Sample distinct (col, row) zones (or random if not enough zones)
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
        h = w if shape == "square" else int(rng.integers(w_low, w_high))
        x, y = _zone_to_xy(col, row, w, h, rng)
        buttons.append(Button(
            button_id=i, color=color, shape=shape, size=size,
            pos_zone=(col, row), x=x, y=y, w=w, h=h,
        ))

    decorative_shapes: list[DecorativeShape] = []
    for _ in range(n_decorative):
        for _attempt in range(20):
            shape = B0_SHAPES[rng.integers(0, len(B0_SHAPES))]
            color = list(B0_COLORS.keys())[rng.integers(0, len(B0_COLORS))]
            w = int(rng.integers(20, 50))
            h = int(rng.integers(20, 50))
            x = int(rng.integers(10, ENV.canvas_w - w - 10))
            y = int(rng.integers(10, ENV.canvas_h - h - 10))
            if not _overlaps_any_button(x, y, w, h, buttons):
                decorative_shapes.append(DecorativeShape(shape=shape, color=color, x=x, y=y, w=w, h=h))
                break
        # If 20 attempts fail, silently skip this decoration (expected for very crowded scenes)

    return Scene(
        buttons=tuple(buttons),
        decorative_shapes=tuple(decorative_shapes),
        bg_color=bg_color,
    )


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


NON_POSITION_ATTRIBUTES: tuple[str, ...] = ("color", "shape", "size")


def is_adversarial(scene: Scene) -> bool:
    """A scene is adversarial if NO button has any non-position single-attribute disambiguator.

    Position is excluded because `generate_scene` enforces distinct zones — position
    trivially disambiguates and would make adversarial scenes vanishingly rare. We
    care about scenes where the categorical (non-spatial) attributes alone don't
    suffice — those are the ones that exercise compositional V+L grounding.
    """
    if len(scene.buttons) <= 1:
        return False
    dis = compute_disambiguators(scene)
    # Each button's disambiguator set; restrict to non-position attributes.
    non_pos_dis = [d & set(NON_POSITION_ATTRIBUTES) for d in dis]
    return any(len(d) == 0 for d in non_pos_dis)
