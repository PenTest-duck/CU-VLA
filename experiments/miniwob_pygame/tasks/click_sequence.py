"""Click-sequence task: click numbered buttons in a specified order."""

from __future__ import annotations

import numpy as np

from ..base_env import BaseTaskEnv
from ..config import ENV


class ClickSequenceEnv(BaseTaskEnv):
    """Agent must click numbered buttons in a specified order.

    Multiple numbered buttons are placed at random non-overlapping positions.
    The instruction specifies the order in which they should be clicked.
    Clicking the wrong button ends the episode with failure.
    """

    task_name = "click-sequence"

    # Button dimensions
    _BUTTON_W = 60
    _BUTTON_H = 40

    def __init__(self, num_buttons: int = 4, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_buttons = num_buttons
        self._buttons: list[dict] = []
        self._target_order: list[int] = []
        self._next_idx: int = 0

    # ------------------------------------------------------------------
    # Task setup
    # ------------------------------------------------------------------

    def _setup_task(self, rng: np.random.Generator) -> None:
        ws = ENV.window_size
        ibh = ENV.instruction_bar_height
        margin = 10
        bw, bh = self._BUTTON_W, self._BUTTON_H

        # Place buttons at random non-overlapping positions
        self._buttons = []
        max_attempts = 200

        for i in range(self.num_buttons):
            for _ in range(max_attempts):
                x = int(rng.integers(margin, ws - bw - margin))
                y = int(rng.integers(ibh + margin, ws - bh - margin))
                # Check overlap with existing buttons
                overlap = False
                for btn in self._buttons:
                    if (x < btn["x"] + btn["width"] + margin
                            and x + bw + margin > btn["x"]
                            and y < btn["y"] + btn["height"] + margin
                            and y + bh + margin > btn["y"]):
                        overlap = True
                        break
                if not overlap:
                    self._buttons.append({
                        "x": x,
                        "y": y,
                        "width": bw,
                        "height": bh,
                        "number": i + 1,
                        "clicked": False,
                    })
                    break

        # Generate a random target click order (permutation of 1..num_buttons)
        self._target_order = list(
            rng.permutation(self.num_buttons) + 1
        )
        self._next_idx = 0

        self.task_instruction = (
            f"Click in order: {', '.join(str(n) for n in self._target_order)}"
        )

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------

    def _handle_mouse_up(self) -> None:
        cx, cy = self._cursor_x, self._cursor_y

        # Find which button (if any) was clicked
        clicked_button = None
        for btn in self._buttons:
            if (btn["x"] <= cx <= btn["x"] + btn["width"]
                    and btn["y"] <= cy <= btn["y"] + btn["height"]
                    and not btn["clicked"]):
                clicked_button = btn
                break

        if clicked_button is None:
            return

        expected_number = self._target_order[self._next_idx]

        if clicked_button["number"] == expected_number:
            # Correct button
            clicked_button["clicked"] = True
            self._next_idx += 1
        else:
            # Wrong button — failure
            self._done = True

    # ------------------------------------------------------------------
    # Success check
    # ------------------------------------------------------------------

    def _check_success(self) -> tuple[bool, dict]:
        if self._next_idx >= self.num_buttons:
            return True, {"success": True}
        # Check if we failed (done set by wrong click)
        if self._done and self._next_idx < self.num_buttons:
            return False, {"failure": "wrong_button"}
        return False, {}

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_task(self, surface) -> None:
        import pygame

        for btn in self._buttons:
            rect = pygame.Rect(btn["x"], btn["y"], btn["width"], btn["height"])

            if btn["clicked"]:
                # Grayed out for clicked buttons
                color = (100, 100, 100)
                text_color = (160, 160, 160)
            else:
                # Active button — blue-ish
                color = (60, 120, 220)
                text_color = (255, 255, 255)

            pygame.draw.rect(surface, color, rect, border_radius=6)

            # Draw number label centered
            if self._font is not None:
                label = str(btn["number"])
                text_surf, text_rect = self._font.render(
                    label, fgcolor=text_color
                )
                tx = btn["x"] + (btn["width"] - text_rect.width) // 2
                ty = btn["y"] + (btn["height"] - text_rect.height) // 2
                surface.blit(text_surf, (tx, ty))
