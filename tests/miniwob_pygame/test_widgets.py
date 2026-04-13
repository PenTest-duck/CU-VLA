"""Tests for experiments.miniwob_pygame.widgets."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

import pygame
import pygame._freetype as _ft

from experiments.miniwob_pygame.config import (
    KEY_A,
    KEY_B,
    KEY_BACKSPACE,
    KEY_SPACE,
    KEY_0,
    KEY_1,
    KEY_CTRL,
)
from experiments.miniwob_pygame.widgets import (
    ScrollableList,
    Slider,
    TextBlock,
    TextInput,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _init_pygame():
    """Initialise Pygame + freetype once for the whole test session."""
    pygame.init()
    _ft.init()
    yield
    pygame.quit()


@pytest.fixture()
def surface() -> pygame.Surface:
    return pygame.Surface((400, 400))


@pytest.fixture()
def font() -> _ft.Font:
    f = _ft.Font(None, 24)
    f.strong = True
    return f


# ==================================================================
# TextInput
# ==================================================================


class TestTextInput:
    def test_initial_state(self):
        ti = TextInput(10, 10, 200, 30)
        assert ti.text == ""
        assert ti.focused is False

    def test_click_focuses(self):
        ti = TextInput(10, 10, 200, 30)
        ti.handle_click(50, 20)
        assert ti.focused is True

    def test_click_outside_unfocuses(self):
        ti = TextInput(10, 10, 200, 30)
        ti.focused = True
        ti.handle_click(300, 300)
        assert ti.focused is False

    def test_type_character(self):
        ti = TextInput(10, 10, 200, 30)
        ti.focused = True
        ti.handle_key_down(KEY_A)
        assert ti.text == "A"
        ti.handle_key_down(KEY_B)
        assert ti.text == "AB"

    def test_type_digit(self):
        ti = TextInput(10, 10, 200, 30)
        ti.focused = True
        ti.handle_key_down(KEY_1)
        assert ti.text == "1"

    def test_type_space(self):
        ti = TextInput(10, 10, 200, 30)
        ti.focused = True
        ti.handle_key_down(KEY_A)
        ti.handle_key_down(KEY_SPACE)
        ti.handle_key_down(KEY_B)
        assert ti.text == "A B"

    def test_type_when_unfocused_ignored(self):
        ti = TextInput(10, 10, 200, 30)
        ti.handle_key_down(KEY_A)
        assert ti.text == ""

    def test_backspace(self):
        ti = TextInput(10, 10, 200, 30)
        ti.focused = True
        ti.handle_key_down(KEY_A)
        ti.handle_key_down(KEY_B)
        ti.handle_key_down(KEY_BACKSPACE)
        assert ti.text == "A"

    def test_backspace_empty_noop(self):
        ti = TextInput(10, 10, 200, 30)
        ti.focused = True
        ti.handle_key_down(KEY_BACKSPACE)  # should not crash
        assert ti.text == ""

    def test_contains(self):
        ti = TextInput(10, 10, 200, 30)
        assert ti.contains(10, 10) is True
        assert ti.contains(110, 25) is True
        assert ti.contains(210, 40) is True  # inclusive right/bottom
        assert ti.contains(9, 10) is False
        assert ti.contains(10, 9) is False
        assert ti.contains(211, 25) is False

    def test_paste_text(self):
        ti = TextInput(10, 10, 200, 30)
        ti.text = "HI"
        ti.paste_text(" WORLD")
        assert ti.text == "HI WORLD"

    def test_modifier_key_ignored(self):
        ti = TextInput(10, 10, 200, 30)
        ti.focused = True
        ti.handle_key_down(KEY_CTRL)
        assert ti.text == ""

    def test_render_no_crash(self, surface, font):
        ti = TextInput(10, 10, 200, 30)
        ti.focused = True
        ti.text = "HELLO"
        ti.render(surface, font)  # should not raise


# ==================================================================
# Slider
# ==================================================================


class TestSlider:
    def test_initial_value_in_range(self):
        s = Slider(50, 100, 200, min_val=0.0, max_val=100.0)
        assert s.min_val <= s.value <= s.max_val

    def test_drag_updates_value(self):
        s = Slider(50, 100, 200, min_val=0.0, max_val=100.0)
        s.value = 0.0
        # Simulate drag: press on handle, drag to midpoint
        hx = int(s.handle_x)
        track_cy = s.y + s.height // 2
        s.handle_mouse_down(hx, track_cy)
        assert s.dragging is True
        # Drag to center of track
        mid_x = s.x + s.width // 2
        s.handle_drag(mid_x, track_cy)
        assert abs(s.value - 50.0) < 1.0

    def test_handle_position(self):
        s = Slider(50, 100, 200, min_val=0.0, max_val=100.0)
        s.value = 50.0
        expected_x = 50 + 100  # x + width/2
        assert abs(s.handle_x - expected_x) < 1.0

    def test_handle_position_at_extremes(self):
        s = Slider(50, 100, 200, min_val=0.0, max_val=100.0)
        s.value = 0.0
        assert abs(s.handle_x - 50.0) < 0.1
        s.value = 100.0
        assert abs(s.handle_x - 250.0) < 0.1

    def test_mouse_up_stops_drag(self):
        s = Slider(50, 100, 200)
        s.dragging = True
        s.handle_mouse_up()
        assert s.dragging is False

    def test_contains_track(self):
        s = Slider(50, 100, 200, height=10, handle_height=30)
        # Center of track
        assert s.contains_track(150, 105) is True
        # Outside
        assert s.contains_track(40, 105) is False

    def test_render_no_crash(self, surface, font):
        s = Slider(50, 100, 200)
        s.value = 30.0
        s.target_value = 70.0
        s.render(surface, font)


# ==================================================================
# ScrollableList
# ==================================================================


class TestScrollableList:
    def _make_list(self, n_items: int = 30) -> ScrollableList:
        items = [f"Item {i}" for i in range(n_items)]
        return ScrollableList(10, 10, 200, 150, items, item_height=35)

    def test_visible_items(self):
        sl = self._make_list()
        vis = sl.visible_items()
        # With height=150, item_height=35, we expect ~4-5 visible items
        assert len(vis) >= 4
        assert len(vis) <= 6
        # First visible should be index 0
        assert vis[0][0] == 0

    def test_scroll_changes_visible(self):
        sl = self._make_list()
        vis_before = sl.visible_items()
        first_before = vis_before[0][0]

        sl.scroll_offset = 200.0  # scroll well down
        vis_after = sl.visible_items()
        first_after = vis_after[0][0]
        assert first_after > first_before

    def test_item_at(self):
        sl = self._make_list()
        # Click in the middle of the first item
        idx = sl.item_at(50, 10 + 17)  # ~center of first item
        assert idx == 0
        # Click lower
        idx = sl.item_at(50, 10 + 35 + 17)  # ~center of second item
        assert idx == 1

    def test_item_at_returns_none_outside(self):
        sl = self._make_list()
        assert sl.item_at(5, 50) is None  # left of list
        assert sl.item_at(50, 5) is None  # above list

    def test_max_scroll(self):
        sl = self._make_list(30)
        expected = 30 * 35 - 150
        assert abs(sl.max_scroll - expected) < 0.1

    def test_handle_click_returns_index(self):
        sl = self._make_list()
        idx = sl.handle_click(50, 10 + 17)
        assert idx == 0
        assert sl.highlighted_index == 0

    def test_scrollbar_rect_valid(self):
        sl = self._make_list()
        sx, sy, sw, sh = sl.scrollbar_rect()
        assert sw == sl.scrollbar_width
        assert sh > 0
        assert sy >= sl.y

    def test_render_no_crash(self, surface, font):
        sl = self._make_list()
        sl.highlighted_index = 2
        sl.render(surface, font)


# ==================================================================
# TextBlock
# ==================================================================


class TestTextBlock:
    def test_word_bounds(self):
        tb = TextBlock(10, 10, 300, "THE DOG RAN")
        bounds = tb.word_bounds("DOG")
        assert bounds == (4, 7)

    def test_word_bounds_not_found(self):
        tb = TextBlock(10, 10, 300, "THE DOG RAN")
        assert tb.word_bounds("CAT") is None

    def test_highlighted_text(self):
        tb = TextBlock(10, 10, 300, "THE DOG RAN")
        tb.set_highlight(4, 7)
        assert tb.highlighted_text() == "DOG"

    def test_highlighted_text_empty_when_cleared(self):
        tb = TextBlock(10, 10, 300, "THE DOG RAN")
        tb.set_highlight(4, 7)
        tb.clear_highlight()
        assert tb.highlighted_text() == ""

    def test_char_at(self, surface, font):
        tb = TextBlock(10, 10, 380, "ABCDEF")
        tb.render(surface, font)
        # After render, _char_rects should be populated
        assert len(tb._char_rects) == 6
        # First character should be at (10, 10, ...)
        cx, cy, cw, ch = tb._char_rects[0]
        assert cx == 10
        assert cy == 10
        # char_at the center of the first char rect should return 0
        result = tb.char_at(cx + cw // 2, cy + ch // 2)
        assert result == 0

    def test_char_at_second_char(self, surface, font):
        tb = TextBlock(10, 10, 380, "ABCDEF")
        tb.render(surface, font)
        cx, cy, cw, ch = tb._char_rects[1]
        result = tb.char_at(cx + cw // 2, cy + ch // 2)
        assert result == 1

    def test_char_at_none_outside(self, surface, font):
        tb = TextBlock(10, 10, 380, "AB")
        tb.render(surface, font)
        assert tb.char_at(390, 390) is None

    def test_get_char_x(self, surface, font):
        tb = TextBlock(10, 10, 380, "ABCDEF")
        tb.render(surface, font)
        x0 = tb.get_char_x(0)
        x1 = tb.get_char_x(1)
        assert x0 == 10.0
        assert x1 > x0

    def test_render_no_crash(self, surface, font):
        tb = TextBlock(10, 10, 380, "HELLO WORLD")
        tb.set_highlight(0, 5)
        tb.render(surface, font)

    def test_render_empty_text(self, surface, font):
        tb = TextBlock(10, 10, 380, "")
        tb.render(surface, font)  # should not crash
