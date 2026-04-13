"""Type-field task: click a text input field and type a target word."""

from __future__ import annotations

import numpy as np

from ..base_env import BaseTaskEnv
from ..config import ENV, VOCAB
from ..widgets import TextInput


class TypeFieldEnv(BaseTaskEnv):
    """Agent must click a text input field and type a target word."""

    task_name = "type-field"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._target_word: str = ""
        self._text_input: TextInput | None = None

    # ------------------------------------------------------------------
    # Task setup
    # ------------------------------------------------------------------

    def _setup_task(self, rng: np.random.Generator) -> None:
        # Pick random word from vocabulary
        idx = int(rng.integers(0, len(VOCAB)))
        self._target_word = VOCAB[idx]
        self.task_instruction = f"Type: {self._target_word}"

        # Create text input field centered horizontally
        self._text_input = TextInput(x=120, y=200, width=160, height=40)

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------

    def _handle_mouse_down(self) -> None:
        assert self._text_input is not None
        self._text_input.handle_click(int(self._cursor_x), int(self._cursor_y))

    def _handle_mouse_up(self) -> None:
        assert self._text_input is not None
        self._text_input.handle_click(int(self._cursor_x), int(self._cursor_y))

    def _handle_key_down(self, key_index: int) -> None:
        assert self._text_input is not None
        self._text_input.handle_key_down(key_index)

    # ------------------------------------------------------------------
    # Success check
    # ------------------------------------------------------------------

    def _check_success(self) -> tuple[bool, dict]:
        assert self._text_input is not None
        typed = self._text_input.text
        target = self._target_word

        if typed == target:
            return True, {"success": True}

        # Wrong prefix — typed something that doesn't match the start of target
        if typed and not target.startswith(typed):
            return True, {
                "failure": "wrong_key",
                "typed": typed,
                "expected": target,
            }

        return False, {}

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_task(self, surface) -> None:
        assert self._text_input is not None
        assert self._font is not None

        # Render target word as large text above the input field
        text_surf, text_rect = self._font.render(
            self._target_word, fgcolor=(255, 255, 255)
        )
        x = (ENV.window_size - text_rect.width) // 2
        surface.blit(text_surf, (x, 130))

        # Render the text input widget
        self._text_input.render(surface, self._font)
