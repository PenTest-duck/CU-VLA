"""Highlight-text task: click-drag to highlight a specific word in a paragraph."""

from __future__ import annotations

import numpy as np

from ..base_env import BaseTaskEnv
from ..config import VOCAB
from ..widgets import TextBlock


class HighlightTextEnv(BaseTaskEnv):
    """Agent must click-drag to highlight a target word in a paragraph."""

    task_name = "highlight-text"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._text_block: TextBlock | None = None
        self._target_word: str = ""
        self._highlight_start_char: int | None = None
        self._selection_made: bool = False

    # ------------------------------------------------------------------
    # Task setup
    # ------------------------------------------------------------------

    def _setup_task(self, rng: np.random.Generator) -> None:
        # Generate a short paragraph: 4-6 random words from VOCAB
        num_words = int(rng.integers(4, 7))  # 4, 5, or 6
        words = list(rng.choice(VOCAB, size=num_words, replace=False))
        paragraph = " ".join(words)

        # Pick a random target word
        target_idx = int(rng.integers(0, num_words))
        self._target_word = words[target_idx]

        # Create TextBlock widget
        self._text_block = TextBlock(x=30, y=80, width=340, text=paragraph)

        self.task_instruction = f"Highlight the word '{self._target_word}'"
        self._highlight_start_char = None
        self._selection_made = False

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------

    def _handle_mouse_down(self) -> None:
        assert self._text_block is not None
        char_idx = self._text_block.char_at(
            int(self._cursor_x), int(self._cursor_y)
        )
        if char_idx is not None:
            self._text_block.set_highlight(char_idx, char_idx)
        self._highlight_start_char = char_idx

    def _handle_drag(self) -> None:
        assert self._text_block is not None
        char_idx = self._text_block.char_at(
            int(self._cursor_x), int(self._cursor_y)
        )
        if char_idx is not None and self._highlight_start_char is not None:
            start = min(self._highlight_start_char, char_idx)
            end = max(self._highlight_start_char, char_idx)
            self._text_block.set_highlight(start, end + 1)

    def _handle_mouse_up(self) -> None:
        self._selection_made = True

    # ------------------------------------------------------------------
    # Success check
    # ------------------------------------------------------------------

    def _check_success(self) -> tuple[bool, dict]:
        assert self._text_block is not None
        if not self._selection_made:
            return False, {}

        highlighted = self._text_block.highlighted_text().strip()
        if highlighted == self._target_word:
            return True, {"success": True}

        # Wrong selection — episode ends with failure
        return True, {
            "failure": "wrong_selection",
            "selected": highlighted,
            "expected": self._target_word,
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_task(self, surface) -> None:
        assert self._text_block is not None
        self._text_block.render(surface, self._font)
