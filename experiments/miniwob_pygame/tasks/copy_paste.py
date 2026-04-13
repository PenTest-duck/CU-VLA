"""Copy-paste task: highlight source text, Ctrl+C, click target field, Ctrl+V.

Validates multi-key simultaneous input (modifier + letter combos).
"""

from __future__ import annotations

import numpy as np

from ..base_env import BaseTaskEnv
from ..config import ENV, EVAL_CFG, KEY_C, KEY_CTRL, KEY_V, NUM_KEYS, VOCAB
from ..widgets import TextBlock, TextInput


class CopyPasteEnv(BaseTaskEnv):
    """Agent must copy source text and paste it into a text input field."""

    task_name = "copy-paste"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._text_block: TextBlock | None = None
        self._text_input: TextInput | None = None
        self._source_text: str = ""
        self._clipboard: str = ""
        self._highlight_start_char: int | None = None
        # Edge flags so combos fire only once per press
        self._ctrl_c_handled: bool = False
        self._ctrl_v_handled: bool = False

    # ------------------------------------------------------------------
    # Task setup
    # ------------------------------------------------------------------

    def _setup_task(self, rng: np.random.Generator) -> None:
        # Pick a random word from VOCAB as source text
        idx = int(rng.integers(0, len(VOCAB)))
        self._source_text = VOCAB[idx]

        # Create source TextBlock
        self._text_block = TextBlock(x=50, y=70, width=300, text=self._source_text)

        # Create target TextInput below
        self._text_input = TextInput(x=100, y=250, width=200, height=40)

        self._clipboard = ""
        self._highlight_start_char = None
        self._ctrl_c_handled = False
        self._ctrl_v_handled = False
        self.task_instruction = "Copy the text and paste it into the field"

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------

    def _handle_mouse_down(self) -> None:
        assert self._text_block is not None
        assert self._text_input is not None

        cx, cy = int(self._cursor_x), int(self._cursor_y)

        # Check if cursor is on TextBlock: start highlight
        char_idx = self._text_block.char_at(cx, cy)
        if char_idx is not None:
            self._text_block.set_highlight(char_idx, char_idx)
            self._highlight_start_char = char_idx
            return

        # Check if cursor is on TextInput: focus it
        self._text_input.handle_click(cx, cy)

    def _handle_drag(self) -> None:
        assert self._text_block is not None
        if self._highlight_start_char is not None:
            char_idx = self._text_block.char_at(
                int(self._cursor_x), int(self._cursor_y)
            )
            if char_idx is not None:
                start = min(self._highlight_start_char, char_idx)
                end = max(self._highlight_start_char, char_idx)
                self._text_block.set_highlight(start, end + 1)

    def _handle_mouse_up(self) -> None:
        assert self._text_input is not None
        # If we were highlighting, stop
        self._highlight_start_char = None
        # Also check for TextInput click (focus on release)
        self._text_input.handle_click(
            int(self._cursor_x), int(self._cursor_y)
        )

    def _handle_key_down(self, key_index: int) -> None:
        assert self._text_input is not None
        # Ctrl+C and Ctrl+V are handled in step() post-processing
        # to properly detect simultaneous key combos.
        # Here we only delegate non-modifier, non-combo keys to TextInput.
        if key_index == KEY_CTRL:
            return
        if key_index in (KEY_C, KEY_V):
            # These might be part of a combo -- will be detected in step().
            # But if Ctrl is NOT held (in the new keys_held, checked in step),
            # we should type the letter. We handle this in step() too.
            return
        self._text_input.handle_key_down(key_index)

    # ------------------------------------------------------------------
    # Step override: detect Ctrl+C / Ctrl+V combos
    # ------------------------------------------------------------------

    def step(self, action: dict) -> tuple[dict, bool, dict]:
        """Override step to detect key combos after base env updates _keys_held."""
        obs, done, info = super().step(action)

        # After super().step(), self._keys_held reflects the NEW state.
        # Detect Ctrl+C combo (fire once per press)
        ctrl_c_active = bool(
            self._keys_held[KEY_CTRL] and self._keys_held[KEY_C]
        )
        if ctrl_c_active and not self._ctrl_c_handled:
            assert self._text_block is not None
            self._clipboard = self._text_block.highlighted_text()
            self._ctrl_c_handled = True
        elif not ctrl_c_active:
            self._ctrl_c_handled = False

        # Detect Ctrl+V combo (fire once per press)
        ctrl_v_active = bool(
            self._keys_held[KEY_CTRL] and self._keys_held[KEY_V]
        )
        if ctrl_v_active and not self._ctrl_v_handled:
            assert self._text_input is not None
            self._text_input.paste_text(self._clipboard)
            self._ctrl_v_handled = True
        elif not ctrl_v_active:
            self._ctrl_v_handled = False

        # Re-check success after paste (paste may have completed the task)
        if not done:
            success, info2 = self._check_success()
            if success:
                self._done = True
                info.update(info2)
                info["steps"] = self._step_count
                done = True

        return obs, done, info

    # ------------------------------------------------------------------
    # Success check
    # ------------------------------------------------------------------

    def _check_success(self) -> tuple[bool, dict]:
        assert self._text_input is not None
        if self._text_input.text == self._source_text:
            return True, {"success": True}
        return False, {}

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_task(self, surface) -> None:
        assert self._text_block is not None
        assert self._text_input is not None
        assert self._font is not None

        # Label: "Source:"
        label_surf, _ = self._font.render("Source:", fgcolor=(180, 180, 180))
        surface.blit(label_surf, (50, 50))

        # Source text block
        self._text_block.render(surface, self._font)

        # Label: "Target:"
        label_surf2, _ = self._font.render("Target:", fgcolor=(180, 180, 180))
        surface.blit(label_surf2, (100, 225))

        # Target text input
        self._text_input.render(surface, self._font)

    # ------------------------------------------------------------------
    # Max steps
    # ------------------------------------------------------------------

    def _get_max_steps(self) -> int:
        return EVAL_CFG.max_steps_long
