"""Use-slider task: drag a slider handle to match a target value."""

from __future__ import annotations

import numpy as np

from ..base_env import BaseTaskEnv
from ..widgets import Slider


class UseSliderEnv(BaseTaskEnv):
    """Agent must drag a slider handle to a target value."""

    task_name = "use-slider"

    def __init__(self, tolerance: float = 5.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._tolerance = tolerance
        self._slider: Slider | None = None

    # ------------------------------------------------------------------
    # Task setup
    # ------------------------------------------------------------------

    def _setup_task(self, rng: np.random.Generator) -> None:
        # Create slider centered in task area
        slider = Slider(x=50, y=220, width=300, height=10,
                        handle_width=20, handle_height=30)

        # Random initial value
        initial_value = float(rng.uniform(0, 100))
        slider.value = initial_value

        # Target at least 15 away from initial
        while True:
            target_value = float(rng.uniform(0, 100))
            if abs(target_value - initial_value) >= 15:
                break

        slider.target_value = target_value
        self._slider = slider
        self.task_instruction = f"Set the slider to {int(target_value)}"

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------

    def _handle_mouse_down(self) -> None:
        assert self._slider is not None
        self._slider.handle_mouse_down(
            int(self._cursor_x), int(self._cursor_y)
        )

    def _handle_drag(self) -> None:
        assert self._slider is not None
        self._slider.handle_drag(int(self._cursor_x), int(self._cursor_y))

    def _handle_mouse_up(self) -> None:
        assert self._slider is not None
        self._slider.handle_mouse_up()

    # ------------------------------------------------------------------
    # Success check
    # ------------------------------------------------------------------

    def _check_success(self) -> tuple[bool, dict]:
        assert self._slider is not None
        if (
            not self._slider.dragging
            and self._slider.target_value is not None
            and abs(self._slider.value - self._slider.target_value)
            <= self._tolerance
        ):
            return True, {
                "success": True,
                "value": self._slider.value,
                "target": self._slider.target_value,
            }
        return False, {}

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_task(self, surface) -> None:
        assert self._slider is not None
        self._slider.render(surface, self._font)
