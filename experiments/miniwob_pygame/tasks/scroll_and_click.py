"""Scroll-and-click task: scroll a list to find a target item, then click it."""

from __future__ import annotations

import numpy as np

from ..base_env import BaseTaskEnv
from ..config import EVAL_CFG, VOCAB
from ..widgets import ScrollableList


class ScrollAndClickEnv(BaseTaskEnv):
    """Agent must scroll a list to find and click a target item."""

    task_name = "scroll-and-click"

    def __init__(self, num_items: int = 15, **kwargs) -> None:
        super().__init__(**kwargs)
        self._num_items = num_items
        self._scroll_list: ScrollableList | None = None
        self._target_item: str = ""
        self._target_index: int = 0
        self._clicked_item: str | None = None

    # ------------------------------------------------------------------
    # Task setup
    # ------------------------------------------------------------------

    def _setup_task(self, rng: np.random.Generator) -> None:
        # Generate unique item labels: pick random words, add numbers
        words = list(rng.choice(VOCAB, size=self._num_items, replace=True))
        items = [f"{w}-{i}" for i, w in enumerate(words)]

        # Create scrollable list widget
        scroll_list = ScrollableList(
            x=50, y=60, width=250, height=250,
            items=items, scrollbar_width=25,
        )

        # Pick target from the BOTTOM half (guaranteed off-screen initially)
        bottom_half_start = self._num_items // 2
        self._target_index = int(rng.integers(bottom_half_start, self._num_items))
        self._target_item = items[self._target_index]

        self._scroll_list = scroll_list
        self._clicked_item = None
        self.task_instruction = f"Click on '{self._target_item}'"

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------

    def _handle_mouse_down(self) -> None:
        assert self._scroll_list is not None
        self._scroll_list.handle_mouse_down(
            int(self._cursor_x), int(self._cursor_y)
        )

    def _handle_drag(self) -> None:
        assert self._scroll_list is not None
        self._scroll_list.handle_drag(
            int(self._cursor_x), int(self._cursor_y)
        )

    def _handle_mouse_up(self) -> None:
        assert self._scroll_list is not None
        self._scroll_list.handle_mouse_up()
        clicked_idx = self._scroll_list.handle_click(
            int(self._cursor_x), int(self._cursor_y)
        )
        if clicked_idx is not None:
            self._clicked_item = self._scroll_list.items[clicked_idx]

    # ------------------------------------------------------------------
    # Success check
    # ------------------------------------------------------------------

    def _check_success(self) -> tuple[bool, dict]:
        if self._clicked_item is not None:
            if self._clicked_item == self._target_item:
                return True, {"success": True}
            else:
                return True, {
                    "failure": "wrong_item",
                    "clicked": self._clicked_item,
                    "expected": self._target_item,
                }
        return False, {}

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_task(self, surface) -> None:
        assert self._scroll_list is not None
        self._scroll_list.render(surface, self._font)

    def _get_max_steps(self) -> int:
        return EVAL_CFG.max_steps_long
