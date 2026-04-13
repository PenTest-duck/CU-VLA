"""Drag-and-label task: drag colored shapes to zones, then type labels."""

from __future__ import annotations

import numpy as np

from ..base_env import BaseTaskEnv
from ..config import ENV, EVAL_CFG, VOCAB, char_to_key_index, key_index_to_char


class DragAndLabelEnv(BaseTaskEnv):
    """Agent must drag colored shapes to matching zones and type their labels.

    Combines drag (from drag-to-zone) with typing (from type-field).
    Colored shapes with 3-letter labels appear on the left; outlined
    drop zones of matching colors appear on the right.  After dropping
    a shape onto its zone, the agent types the shape's label.
    """

    task_name = "drag-and-label"

    def __init__(self, num_shapes: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_shapes = min(num_shapes, len(ENV.shape_colors))
        self._shapes: list[dict] = []
        self._zones: list[dict] = []
        self._grabbed_shape_idx: int = -1

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
        colors = list(ENV.shape_colors)

        # Pick which colors to use (shuffle and take num_shapes)
        color_indices = list(range(len(colors)))
        rng.shuffle(color_indices)
        chosen = color_indices[: self.num_shapes]

        # Pick random labels from VOCAB
        vocab_indices = list(range(len(VOCAB)))
        rng.shuffle(vocab_indices)

        # --- Place shapes on LEFT half ---
        self._shapes = []
        for si, idx in enumerate(chosen):
            w = int(rng.integers(60, 80))
            h = 50
            x = int(rng.integers(30, 170 - w + 1))
            y = self._find_non_overlapping_y(self._shapes, h, gap=h + 15)
            label = VOCAB[vocab_indices[si % len(vocab_indices)]]
            self._shapes.append({
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "color": colors[idx],
                "label": label,
                "grabbed": False,
                "dropped": False,
                "complete": False,
                "typed_so_far": "",
            })

        # --- Place matching zones on RIGHT half ---
        self._zones = []
        for si, idx in enumerate(chosen):
            w = 90
            h = 60
            x = int(rng.integers(230, 340 - w + 1))
            y = self._find_non_overlapping_y(self._zones, h, gap=h + 15)
            self._zones.append({
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "color": colors[idx],
                "target_label": self._shapes[si]["label"],
                "typed_text": "",
            })

        self._grabbed_shape_idx = -1
        self.task_instruction = "Drag shapes to zones and type labels"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_non_overlapping_y(
        self, placed: list[dict], item_height: int, gap: int
    ) -> int:
        """Find a random y that doesn't overlap with already-placed items."""
        margin = ENV.instruction_bar_height + 10
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

    def _find_zone_for_shape(self, shape: dict) -> dict | None:
        """Find the zone with matching color."""
        for zone in self._zones:
            if zone["color"] == shape["color"]:
                return zone
        return None

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------

    def _handle_mouse_down(self) -> None:
        cx, cy = self._cursor_x, self._cursor_y
        for i, shape in enumerate(self._shapes):
            if shape["grabbed"] or shape["dropped"]:
                continue
            if (shape["x"] <= cx <= shape["x"] + shape["width"]
                    and shape["y"] <= cy <= shape["y"] + shape["height"]):
                shape["grabbed"] = True
                self._grabbed_shape_idx = i
                break

    def _handle_drag(self) -> None:
        if self._grabbed_shape_idx >= 0:
            shape = self._shapes[self._grabbed_shape_idx]
            if shape["grabbed"]:
                shape["x"] = int(self._cursor_x - shape["width"] / 2)
                shape["y"] = int(self._cursor_y - shape["height"] / 2)

    def _handle_mouse_up(self) -> None:
        if self._grabbed_shape_idx >= 0:
            shape = self._shapes[self._grabbed_shape_idx]
            if shape["grabbed"]:
                zone = self._find_zone_for_shape(shape)
                if zone is not None:
                    cx, cy = self._cursor_x, self._cursor_y
                    zx, zy = zone["x"], zone["y"]
                    zw, zh = zone["width"], zone["height"]
                    if zx <= cx <= zx + zw and zy <= cy <= zy + zh:
                        shape["dropped"] = True
                        # Snap shape above zone (centered)
                        shape["x"] = int(zx + (zw - shape["width"]) / 2)
                        shape["y"] = int(zy - shape["height"] - 5)

                shape["grabbed"] = False
            self._grabbed_shape_idx = -1

    def _handle_key_down(self, key_index: int) -> None:
        if not (0 <= key_index <= 25):
            return  # Only A-Z

        ch = chr(ord("A") + key_index)

        # Find the first dropped-but-not-complete shape
        target_shape = None
        target_zone = None
        for shape in self._shapes:
            if shape["dropped"] and not shape["complete"]:
                target_shape = shape
                target_zone = self._find_zone_for_shape(shape)
                break

        if target_shape is None:
            return

        # Append character
        new_typed = target_shape["typed_so_far"] + ch
        target_shape["typed_so_far"] = new_typed
        if target_zone is not None:
            target_zone["typed_text"] = new_typed

        # Check if label matches
        if new_typed == target_shape["label"]:
            target_shape["complete"] = True
        elif not target_shape["label"].startswith(new_typed):
            # Wrong key — mark as failure via done flag
            self._done = True
            self._failure_info = {
                "failure": "wrong_key",
                "typed": new_typed,
                "expected": target_shape["label"],
            }

    # ------------------------------------------------------------------
    # Success check
    # ------------------------------------------------------------------

    def _check_success(self) -> tuple[bool, dict]:
        # Check for failure set by _handle_key_down
        if hasattr(self, "_failure_info") and self._failure_info:
            info = dict(self._failure_info)
            self._failure_info = {}
            return True, info

        n_complete = sum(1 for s in self._shapes if s["complete"])
        n_dropped = sum(1 for s in self._shapes if s["dropped"])
        if n_complete == self.num_shapes:
            return True, {
                "success": True,
                "shapes_completed": n_complete,
                "shapes_dropped": n_dropped,
                "num_shapes": self.num_shapes,
            }
        return False, {}

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_task(self, surface) -> None:
        import pygame

        # Draw zones as outlined rects with typed text inside
        for zone in self._zones:
            rect = pygame.Rect(zone["x"], zone["y"], zone["width"], zone["height"])
            pygame.draw.rect(surface, zone["color"], rect, width=3)

            # Show typed text inside zone
            if zone["typed_text"] and self._font is not None:
                text_surf, text_rect = self._font.render(
                    zone["typed_text"], fgcolor=(255, 255, 255)
                )
                tx = zone["x"] + (zone["width"] - text_rect.width) // 2
                ty = zone["y"] + (zone["height"] - text_rect.height) // 2
                surface.blit(text_surf, (tx, ty))

        # Draw shapes as filled rounded rects with label text (skip if complete)
        for shape in self._shapes:
            if shape["complete"]:
                continue
            rect = pygame.Rect(
                shape["x"], shape["y"], shape["width"], shape["height"]
            )
            pygame.draw.rect(surface, shape["color"], rect, border_radius=8)

            # Label text
            if self._font is not None:
                text_surf, text_rect = self._font.render(
                    shape["label"], fgcolor=(255, 255, 255)
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

    # ------------------------------------------------------------------
    # Reset override to clear failure state
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> dict:
        self._failure_info: dict = {}
        return super().reset(seed=seed)
