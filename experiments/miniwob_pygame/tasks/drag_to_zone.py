"""Drag-to-zone task: drag colored shapes to matching drop zones."""

from __future__ import annotations

import numpy as np

from ..base_env import BaseTaskEnv
from ..config import ENV, EVAL_CFG

# Color names matching ENV.shape_colors order
_COLOR_NAMES = ["RED", "BLUE", "GREEN", "YELLOW"]


class DragToZoneEnv(BaseTaskEnv):
    """Agent must drag colored shapes to their matching drop zones.

    Colored shapes appear on the left half; outlined drop zones of
    matching colors appear on the right half.  The agent drags each
    shape onto the zone with the same color.
    """

    task_name = "drag-to-zone"

    def __init__(self, num_shapes: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_shapes = min(num_shapes, len(ENV.shape_colors))
        self._shapes: list[dict] = []
        self._zones: list[dict] = []
        self._grabbed_idx: int | None = None

    # ------------------------------------------------------------------
    # Properties (for expert access)
    # ------------------------------------------------------------------

    @property
    def shapes(self) -> list[dict]:
        return self._shapes

    @property
    def zones(self) -> list[dict]:
        return self._zones

    # ------------------------------------------------------------------
    # Task setup
    # ------------------------------------------------------------------

    def _setup_task(self, rng: np.random.Generator) -> None:
        ibh = ENV.instruction_bar_height
        colors = list(ENV.shape_colors)

        # Pick which colors to use (shuffle and take num_shapes)
        color_indices = list(range(len(colors)))
        rng.shuffle(color_indices)
        chosen = color_indices[: self.num_shapes]

        # --- Place shapes on LEFT half ---
        self._shapes = []
        for idx in chosen:
            w = int(rng.integers(60, 80))
            h = 50
            x = int(rng.integers(30, 170 - w + 1))
            y = self._find_non_overlapping_y(
                self._shapes, h, gap=h + 15
            )
            self._shapes.append({
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "color": colors[idx],
                "grabbed": False,
                "dropped": False,
                "complete": False,
                "original_x": x,
                "original_y": y,
            })

        # --- Place matching zones on RIGHT half ---
        self._zones = []
        for idx in chosen:
            w = 90
            h = 60
            x = int(rng.integers(230, 340 - w + 1))
            y = self._find_non_overlapping_y(
                self._zones, h, gap=h + 15
            )
            self._zones.append({
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "color": colors[idx],
                "border_width": 3,
            })

        # Instruction
        if self.num_shapes == 1:
            color_name = _COLOR_NAMES[chosen[0]]
            self.task_instruction = (
                f"Drag the {color_name} shape to the {color_name} zone"
            )
        else:
            self.task_instruction = "Drag each shape to its matching zone"

        self._grabbed_idx = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_non_overlapping_y(
        self, placed: list[dict], item_height: int, gap: int
    ) -> int:
        """Find a random y that doesn't overlap with already-placed items."""
        ibh = ENV.instruction_bar_height
        margin = ibh + 10
        y_max = ENV.window_size - item_height - 10
        for _ in range(100):
            y = int(self._rng.integers(margin, y_max + 1))
            overlap = False
            for other in placed:
                if (y < other["y"] + other["height"] + gap
                        and y + item_height + gap > other["y"]):
                    overlap = True
                    break
            if not overlap:
                return y
        # Fallback: stack below the last placed item
        if placed:
            return min(placed[-1]["y"] + placed[-1]["height"] + gap, y_max)
        return margin

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------

    def _handle_mouse_down(self) -> None:
        cx, cy = self._cursor_x, self._cursor_y
        for i, shape in enumerate(self._shapes):
            if shape["grabbed"] or shape["dropped"] or shape["complete"]:
                continue
            if (shape["x"] <= cx <= shape["x"] + shape["width"]
                    and shape["y"] <= cy <= shape["y"] + shape["height"]):
                shape["grabbed"] = True
                self._grabbed_idx = i
                break

    def _handle_drag(self) -> None:
        if self._grabbed_idx is not None:
            shape = self._shapes[self._grabbed_idx]
            if shape["grabbed"]:
                # Move shape center to cursor position
                shape["x"] = int(self._cursor_x - shape["width"] / 2)
                shape["y"] = int(self._cursor_y - shape["height"] / 2)

    def _handle_mouse_up(self) -> None:
        if self._grabbed_idx is not None:
            shape = self._shapes[self._grabbed_idx]
            if shape["grabbed"]:
                # Find matching zone (same color)
                matching_zone = None
                for zone in self._zones:
                    if zone["color"] == shape["color"]:
                        matching_zone = zone
                        break

                if matching_zone is not None:
                    cx, cy = self._cursor_x, self._cursor_y
                    zx, zy = matching_zone["x"], matching_zone["y"]
                    zw, zh = matching_zone["width"], matching_zone["height"]
                    if zx <= cx <= zx + zw and zy <= cy <= zy + zh:
                        shape["dropped"] = True
                        shape["complete"] = True
                        # Snap shape above zone (centered horizontally)
                        shape["x"] = int(
                            zx + (zw - shape["width"]) / 2
                        )
                        shape["y"] = int(
                            zy + (zh - shape["height"]) / 2
                        )

                shape["grabbed"] = False
            self._grabbed_idx = None

    # ------------------------------------------------------------------
    # Success check
    # ------------------------------------------------------------------

    def _check_success(self) -> tuple[bool, dict]:
        n_dropped = sum(1 for s in self._shapes if s["dropped"])
        if n_dropped == self.num_shapes:
            return True, {
                "success": True,
                "shapes_dropped": n_dropped,
                "num_shapes": self.num_shapes,
            }
        return False, {}

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_task(self, surface) -> None:
        import pygame

        # Draw zones as outlined rects (color border, 3px width)
        for zone in self._zones:
            rect = pygame.Rect(zone["x"], zone["y"], zone["width"], zone["height"])
            pygame.draw.rect(
                surface, zone["color"], rect, width=zone["border_width"]
            )

        # Draw shapes as filled rounded rects with label text
        for shape in self._shapes:
            rect = pygame.Rect(
                shape["x"], shape["y"], shape["width"], shape["height"]
            )
            pygame.draw.rect(surface, shape["color"], rect, border_radius=8)

            # Label text (color name)
            if self._font is not None:
                color_idx = list(ENV.shape_colors).index(shape["color"])
                label = _COLOR_NAMES[color_idx]
                text_surf, text_rect = self._font.render(
                    label, fgcolor=(255, 255, 255)
                )
                tx = shape["x"] + (shape["width"] - text_rect.width) // 2
                ty = shape["y"] + (shape["height"] - text_rect.height) // 2
                surface.blit(text_surf, (tx, ty))

    # ------------------------------------------------------------------
    # Max steps
    # ------------------------------------------------------------------

    def _get_max_steps(self) -> int:
        if self.num_shapes == 1:
            return EVAL_CFG.max_steps_per_episode
        return EVAL_CFG.max_steps_multi
