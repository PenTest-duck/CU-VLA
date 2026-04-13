"""Click-target task: navigate cursor to a colored shape and click it."""

from __future__ import annotations

import numpy as np

from ..base_env import BaseTaskEnv
from ..config import ENV

# Color names matching ENV.shape_colors order
_COLOR_NAMES = ["RED", "BLUE", "GREEN", "YELLOW"]


class ClickTargetEnv(BaseTaskEnv):
    """Agent must navigate cursor to a colored shape and click it."""

    task_name = "click-target"

    def __init__(self, num_distractors: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_distractors = num_distractors
        self._target: dict = {}
        self._distractors: list[dict] = []
        self._clicked_on_target: bool = False

    # ------------------------------------------------------------------
    # Task setup
    # ------------------------------------------------------------------

    def _setup_task(self, rng: np.random.Generator) -> None:
        ws = ENV.window_size
        ibh = ENV.instruction_bar_height
        colors = list(ENV.shape_colors)

        # Pick target color
        target_color_idx = int(rng.integers(0, len(colors)))
        target_color = colors[target_color_idx]
        color_name = _COLOR_NAMES[target_color_idx]
        self.task_instruction = f"Click the {color_name} shape"

        # Place target
        w = int(rng.integers(40, 80))
        h = int(rng.integers(40, 80))
        x = int(rng.integers(10, ws - w - 10))
        y = int(rng.integers(ibh + 10, ws - h - 10))
        self._target = {"x": x, "y": y, "width": w, "height": h, "color": target_color}

        # Place distractors with different colors
        available_colors = [c for i, c in enumerate(colors) if i != target_color_idx]
        self._distractors = []
        for i in range(self.num_distractors):
            dw = int(rng.integers(40, 80))
            dh = int(rng.integers(40, 80))
            dx = int(rng.integers(10, ws - dw - 10))
            dy = int(rng.integers(ibh + 10, ws - dh - 10))
            dc = available_colors[i % len(available_colors)]
            self._distractors.append(
                {"x": dx, "y": dy, "width": dw, "height": dh, "color": dc}
            )

        self._clicked_on_target = False

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------

    def _handle_mouse_up(self) -> None:
        t = self._target
        cx, cy = self._cursor_x, self._cursor_y
        if (t["x"] <= cx <= t["x"] + t["width"]
                and t["y"] <= cy <= t["y"] + t["height"]):
            self._clicked_on_target = True
            self._done = True

    # ------------------------------------------------------------------
    # Success check
    # ------------------------------------------------------------------

    def _check_success(self) -> tuple[bool, dict]:
        if self._clicked_on_target:
            return True, {"success": True}
        return False, {}

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_task(self, surface) -> None:
        import pygame

        # Draw distractors first (behind target)
        for d in self._distractors:
            rect = pygame.Rect(d["x"], d["y"], d["width"], d["height"])
            pygame.draw.rect(surface, d["color"], rect, border_radius=8)

        # Draw target
        t = self._target
        rect = pygame.Rect(t["x"], t["y"], t["width"], t["height"])
        pygame.draw.rect(surface, t["color"], rect, border_radius=8)
