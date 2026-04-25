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
