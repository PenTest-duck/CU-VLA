"""Drag-sort task: drag numbered cards into ascending left-to-right order."""

from __future__ import annotations

import numpy as np

from ..base_env import BaseTaskEnv
from ..config import ENV, EVAL_CFG

# Palette for card backgrounds (distinct, readable with white text)
_CARD_COLORS = [
    (70, 130, 210),   # blue
    (210, 80, 80),    # red
    (60, 170, 90),    # green
    (200, 160, 50),   # amber
    (150, 80, 190),   # purple
    (50, 170, 170),   # teal
    (200, 110, 60),   # orange
    (140, 140, 140),  # gray
]


class DragSortEnv(BaseTaskEnv):
    """Agent must drag numbered cards into ascending order (left to right).

    Cards with values 1..num_cards are arranged in a horizontal row in
    shuffled order.  The agent drags cards between slots to sort them.
    Dropping a card on an occupied slot swaps the two cards.
    """

    task_name = "drag-sort"

    def __init__(self, num_cards: int = 4, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_cards = num_cards
        self._cards: list[dict] = []
        self._slots: list[int] = []  # x-center for each slot position
        self._grabbed_card: int | None = None

    # ------------------------------------------------------------------
    # Properties (for expert access)
    # ------------------------------------------------------------------

    @property
    def cards(self) -> list[dict]:
        return self._cards

    @property
    def slots(self) -> list[int]:
        return self._slots

    # ------------------------------------------------------------------
    # Task setup
    # ------------------------------------------------------------------

    def _setup_task(self, rng: np.random.Generator) -> None:
        ibh = ENV.instruction_bar_height
        ws = ENV.window_size

        card_w = 70
        card_h = 50
        spacing = (ws - 60) // self.num_cards
        start_x = 30
        card_y = ibh + (ws - ibh - card_h) // 2  # centered vertically in task area

        # Compute slot x-centers
        self._slots = [
            start_x + i * spacing + card_w // 2
            for i in range(self.num_cards)
        ]

        # Shuffled card values
        values = list(range(1, self.num_cards + 1))
        rng.shuffle(values)

        self._cards = []
        for i, val in enumerate(values):
            x = self._slots[i] - card_w // 2
            self._cards.append({
                "x": x,
                "y": card_y,
                "width": card_w,
                "height": card_h,
                "value": val,
                "slot_index": i,
                "color": _CARD_COLORS[(val - 1) % len(_CARD_COLORS)],
            })

        self._grabbed_card = None
        self.task_instruction = "Sort the numbers in ascending order"

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------

    def _handle_mouse_down(self) -> None:
        cx, cy = self._cursor_x, self._cursor_y
        for i, card in enumerate(self._cards):
            if (card["x"] <= cx <= card["x"] + card["width"]
                    and card["y"] <= cy <= card["y"] + card["height"]):
                self._grabbed_card = i
                break

    def _handle_drag(self) -> None:
        if self._grabbed_card is not None:
            card = self._cards[self._grabbed_card]
            card["x"] = int(self._cursor_x - card["width"] / 2)
            card["y"] = int(self._cursor_y - card["height"] / 2)

    def _handle_mouse_up(self) -> None:
        if self._grabbed_card is not None:
            grabbed = self._cards[self._grabbed_card]

            # Find nearest slot by x distance
            best_slot = 0
            best_dist = float("inf")
            for si, slot_cx in enumerate(self._slots):
                d = abs(self._cursor_x - slot_cx)
                if d < best_dist:
                    best_dist = d
                    best_slot = si

            # Check if another card occupies the target slot
            other_idx = None
            for ci, card in enumerate(self._cards):
                if ci != self._grabbed_card and card["slot_index"] == best_slot:
                    other_idx = ci
                    break

            # Swap if occupied
            if other_idx is not None:
                old_slot = grabbed["slot_index"]
                other_card = self._cards[other_idx]
                # Move the other card to the grabbed card's old slot
                other_card["slot_index"] = old_slot
                other_card["x"] = self._slots[old_slot] - other_card["width"] // 2
                # Restore other card's y to standard position
                ibh = ENV.instruction_bar_height
                ws = ENV.window_size
                other_card["y"] = ibh + (ws - ibh - other_card["height"]) // 2

            # Snap grabbed card to target slot
            grabbed["slot_index"] = best_slot
            grabbed["x"] = self._slots[best_slot] - grabbed["width"] // 2
            ibh = ENV.instruction_bar_height
            ws = ENV.window_size
            grabbed["y"] = ibh + (ws - ibh - grabbed["height"]) // 2

            self._grabbed_card = None

    # ------------------------------------------------------------------
    # Success check
    # ------------------------------------------------------------------

    def _check_success(self) -> tuple[bool, dict]:
        # Build value sequence left-to-right by slot_index
        slot_values = [0] * self.num_cards
        for card in self._cards:
            slot_values[card["slot_index"]] = card["value"]

        if slot_values == list(range(1, self.num_cards + 1)):
            return True, {"success": True}
        return False, {}

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_task(self, surface) -> None:
        import pygame

        ibh = ENV.instruction_bar_height
        ws = ENV.window_size
        card_h = 50

        standard_y = ibh + (ws - ibh - card_h) // 2

        # Draw slot outlines (light gray dashed-style rects)
        for slot_cx in self._slots:
            slot_rect = pygame.Rect(
                slot_cx - 35, standard_y, 70, card_h
            )
            pygame.draw.rect(surface, (80, 80, 80), slot_rect, width=2)

        # Draw cards (grabbed card last so it renders on top)
        draw_order = list(range(len(self._cards)))
        if self._grabbed_card is not None:
            draw_order.remove(self._grabbed_card)
            draw_order.append(self._grabbed_card)

        for ci in draw_order:
            card = self._cards[ci]
            is_grabbed = ci == self._grabbed_card

            # Slightly larger if grabbed
            w = card["width"] + (6 if is_grabbed else 0)
            h = card["height"] + (6 if is_grabbed else 0)
            x = card["x"] - (3 if is_grabbed else 0)
            y = card["y"] - (3 if is_grabbed else 0)

            rect = pygame.Rect(x, y, w, h)
            pygame.draw.rect(surface, card["color"], rect, border_radius=8)

            # Highlight border if grabbed
            if is_grabbed:
                pygame.draw.rect(
                    surface, (255, 255, 255), rect, width=2, border_radius=8
                )

            # Number text centered
            if self._font is not None:
                label = str(card["value"])
                text_surf, text_rect = self._font.render(
                    label, fgcolor=(255, 255, 255)
                )
                tx = x + (w - text_rect.width) // 2
                ty = y + (h - text_rect.height) // 2
                surface.blit(text_surf, (tx, ty))

    # ------------------------------------------------------------------
    # Max steps
    # ------------------------------------------------------------------

    def _get_max_steps(self) -> int:
        return EVAL_CFG.max_steps_multi
