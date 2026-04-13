"""Form-fill task: fill labeled text fields with target values and submit."""

from __future__ import annotations

import numpy as np

from ..base_env import BaseTaskEnv
from ..config import ENV, EVAL_CFG, KEY_TAB, VOCAB
from ..widgets import TextInput


class FormFillEnv(BaseTaskEnv):
    """Agent must fill 2-3 labeled text fields and click Submit."""

    task_name = "form-fill"

    _LABELS = ["Username", "Password", "Code"]

    def __init__(self, num_fields: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        assert 2 <= num_fields <= 3
        self.num_fields = num_fields
        self._fields: list[TextInput] = []
        self._target_values: list[str] = []
        self._labels: list[str] = []
        self._submit_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._submitted: bool = False

    # ------------------------------------------------------------------
    # Task setup
    # ------------------------------------------------------------------

    def _setup_task(self, rng: np.random.Generator) -> None:
        labels = self._LABELS[: self.num_fields]

        # Pick random 3-letter target values
        indices = rng.choice(len(VOCAB), size=self.num_fields, replace=False)
        values = [VOCAB[int(i)] for i in indices]

        # Create TextInput widgets vertically stacked
        fields: list[TextInput] = []
        for i in range(self.num_fields):
            y = 100 + i * 50
            fields.append(TextInput(x=150, y=y, width=160, height=35))

        # Submit button rect: bottom center below last field
        submit_y = 100 + self.num_fields * 50 + 30
        submit_rect = (150, submit_y, 100, 35)

        self._fields = fields
        self._target_values = values
        self._labels = labels
        self._submit_rect = submit_rect
        self._submitted = False
        self.task_instruction = " | ".join(
            f"{label}: {value}" for label, value in zip(labels, values)
        )

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------

    def _handle_mouse_down(self) -> None:
        cx, cy = int(self._cursor_x), int(self._cursor_y)
        for field in self._fields:
            field.handle_click(cx, cy)

    def _handle_mouse_up(self) -> None:
        cx, cy = int(self._cursor_x), int(self._cursor_y)
        for field in self._fields:
            field.handle_click(cx, cy)

        # Check submit button
        sx, sy, sw, sh = self._submit_rect
        if sx <= cx <= sx + sw and sy <= cy <= sy + sh:
            self._submitted = True

    def _handle_key_down(self, key_index: int) -> None:
        if key_index == KEY_TAB:
            # Move focus to next field
            focused_idx = None
            for i, field in enumerate(self._fields):
                if field.focused:
                    focused_idx = i
                    break
            if focused_idx is not None:
                self._fields[focused_idx].focused = False
                next_idx = (focused_idx + 1) % len(self._fields)
                self._fields[next_idx].focused = True
            else:
                # No field focused — focus the first one
                self._fields[0].focused = True
            return

        # Delegate to whichever field is focused
        for field in self._fields:
            if field.focused:
                field.handle_key_down(key_index)
                break

    # ------------------------------------------------------------------
    # Success check
    # ------------------------------------------------------------------

    def _check_success(self) -> tuple[bool, dict]:
        if not self._submitted:
            return False, {}

        all_correct = all(
            field.text == target
            for field, target in zip(self._fields, self._target_values)
        )
        if all_correct:
            return True, {"success": True}

        return True, {
            "failure": "wrong_values",
            "typed": [f.text for f in self._fields],
            "expected": self._target_values,
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_task(self, surface) -> None:
        import pygame

        assert self._font is not None

        # Render labels and fields
        for i, (label, field) in enumerate(zip(self._labels, self._fields)):
            # Label to the left of each field
            label_surf, label_rect = self._font.render(
                label, fgcolor=(200, 200, 200)
            )
            label_x = field.x - label_rect.width - 10
            label_y = field.y + (field.height - label_rect.height) // 2
            surface.blit(label_surf, (label_x, label_y))

            # Field widget
            field.render(surface, self._font)

        # Submit button
        sx, sy, sw, sh = self._submit_rect
        btn_rect = pygame.Rect(sx, sy, sw, sh)
        pygame.draw.rect(surface, (80, 140, 80), btn_rect)
        pygame.draw.rect(surface, (120, 200, 120), btn_rect, 2)

        btn_surf, btn_text_rect = self._font.render("Submit", fgcolor=(255, 255, 255))
        tx = sx + (sw - btn_text_rect.width) // 2
        ty = sy + (sh - btn_text_rect.height) // 2
        surface.blit(btn_surf, (tx, ty))

    # ------------------------------------------------------------------
    # Max steps
    # ------------------------------------------------------------------

    def _get_max_steps(self) -> int:
        return EVAL_CFG.max_steps_multi
