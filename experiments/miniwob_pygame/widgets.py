"""Reusable Pygame widgets for MiniWoB-Pygame task environments.

Each widget knows how to render itself onto a surface and respond to
mouse/keyboard events.  Widgets do NOT initialise Pygame — they receive
a surface and a ``pygame._freetype.Font`` from the owning environment.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from .config import KEY_BACKSPACE, KEY_SPACE, key_index_to_char

if TYPE_CHECKING:
    import pygame
    import pygame._freetype as _ft


# -----------------------------------------------------------------------
# TextInput
# -----------------------------------------------------------------------

class TextInput:
    """Clickable single-line text field."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        bg_color: tuple[int, int, int] = (60, 60, 60),
        text_color: tuple[int, int, int] = (255, 255, 255),
        border_color: tuple[int, int, int] = (120, 120, 120),
        focused_border_color: tuple[int, int, int] = (80, 180, 255),
    ) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.text_color = text_color
        self.border_color = border_color
        self.focused_border_color = focused_border_color

        self.text: str = ""
        self.focused: bool = False

    # -- hit test -------------------------------------------------------
    def contains(self, px: int, py: int) -> bool:
        """Return True if (px, py) is inside the field (inclusive)."""
        return (
            self.x <= px <= self.x + self.width
            and self.y <= py <= self.y + self.height
        )

    # -- events ---------------------------------------------------------
    def handle_click(self, px: int, py: int) -> None:
        """Focus if click is inside, unfocus otherwise."""
        self.focused = self.contains(px, py)

    def handle_key_down(self, key_index: int) -> None:
        """Append printable character or handle backspace (only when focused)."""
        if not self.focused:
            return
        if key_index == KEY_BACKSPACE:
            self.text = self.text[:-1]
            return
        ch = key_index_to_char(key_index)
        if ch is not None:
            self.text += ch

    def paste_text(self, text: str) -> None:
        """Append *text* to current text (Ctrl+V support)."""
        self.text += text

    # -- rendering ------------------------------------------------------
    def render(self, surface: pygame.Surface, font: _ft.Font) -> None:
        """Draw the text field onto *surface*."""
        import pygame

        rect = pygame.Rect(self.x, self.y, self.width, self.height)

        # Background
        pygame.draw.rect(surface, self.bg_color, rect)

        # Border
        border_col = self.focused_border_color if self.focused else self.border_color
        pygame.draw.rect(surface, border_col, rect, 2)

        # Text (left-padded 4 px, vertically centered)
        if self.text:
            text_surf, text_rect = font.render(self.text, fgcolor=self.text_color)
            ty = self.y + (self.height - text_rect.height) // 2
            surface.blit(text_surf, (self.x + 4, ty))

        # Blinking cursor (visible half the time)
        if self.focused and int(time.time() * 2) % 2 == 0:
            if self.text:
                cursor_x = self.x + 4 + font.get_rect(self.text).width
            else:
                cursor_x = self.x + 4
            cy_top = self.y + 4
            cy_bot = self.y + self.height - 4
            pygame.draw.line(surface, self.text_color, (cursor_x, cy_top), (cursor_x, cy_bot), 1)


# -----------------------------------------------------------------------
# Slider
# -----------------------------------------------------------------------

class Slider:
    """Horizontal slider with draggable handle."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int = 10,
        handle_width: int = 20,
        handle_height: int = 30,
        min_val: float = 0.0,
        max_val: float = 100.0,
        track_color: tuple[int, int, int] = (80, 80, 80),
        handle_color: tuple[int, int, int] = (200, 200, 200),
        target_color: tuple[int, int, int] = (255, 200, 50),
    ) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.handle_width = handle_width
        self.handle_height = handle_height
        self.min_val = min_val
        self.max_val = max_val
        self.track_color = track_color
        self.handle_color = handle_color
        self.target_color = target_color

        self.value: float = min_val
        self.target_value: float | None = None
        self.dragging: bool = False

    # -- derived positions ---------------------------------------------
    @property
    def handle_x(self) -> float:
        """X position of handle center based on current value."""
        frac = (self.value - self.min_val) / max(self.max_val - self.min_val, 1e-9)
        return self.x + frac * self.width

    def _value_from_x(self, px: float) -> float:
        frac = (px - self.x) / max(self.width, 1)
        frac = max(0.0, min(1.0, frac))
        return self.min_val + frac * (self.max_val - self.min_val)

    # -- hit tests -----------------------------------------------------
    def contains_handle(self, px: int, py: int) -> bool:
        hx = self.handle_x
        track_cy = self.y + self.height / 2
        hx0 = hx - self.handle_width / 2
        hy0 = track_cy - self.handle_height / 2
        return (
            hx0 <= px <= hx0 + self.handle_width
            and hy0 <= py <= hy0 + self.handle_height
        )

    def contains_track(self, px: int, py: int) -> bool:
        track_cy = self.y + self.height / 2
        return (
            self.x <= px <= self.x + self.width
            and track_cy - self.handle_height / 2 <= py <= track_cy + self.handle_height / 2
        )

    # -- events --------------------------------------------------------
    def handle_mouse_down(self, px: int, py: int) -> None:
        if self.contains_handle(px, py):
            self.dragging = True

    def handle_drag(self, px: int, py: int) -> None:
        if self.dragging:
            self.value = self._value_from_x(px)

    def handle_mouse_up(self) -> None:
        self.dragging = False

    # -- rendering -----------------------------------------------------
    def render(self, surface: pygame.Surface, font: _ft.Font) -> None:
        import pygame

        track_cy = self.y + self.height // 2

        # Track
        track_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(surface, self.track_color, track_rect)

        # Target tick
        if self.target_value is not None:
            t_frac = (self.target_value - self.min_val) / max(self.max_val - self.min_val, 1e-9)
            tx = int(self.x + t_frac * self.width)
            pygame.draw.line(
                surface, self.target_color,
                (tx, track_cy - self.handle_height // 2),
                (tx, track_cy + self.handle_height // 2),
                2,
            )
            label_surf, _ = font.render(f"{self.target_value:.0f}", fgcolor=self.target_color)
            surface.blit(label_surf, (tx - 10, track_cy - self.handle_height // 2 - 18))

        # Handle
        hx = int(self.handle_x)
        handle_rect = pygame.Rect(
            hx - self.handle_width // 2,
            track_cy - self.handle_height // 2,
            self.handle_width,
            self.handle_height,
        )
        pygame.draw.rect(surface, self.handle_color, handle_rect)

        # Value label below handle
        val_surf, _ = font.render(f"{self.value:.0f}", fgcolor=self.handle_color)
        surface.blit(val_surf, (hx - 10, track_cy + self.handle_height // 2 + 4))


# -----------------------------------------------------------------------
# ScrollableList
# -----------------------------------------------------------------------

class ScrollableList:
    """Scrollable list of text items with a scrollbar."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        items: list[str],
        item_height: int = 35,
        scrollbar_width: int = 20,
        bg_color: tuple[int, int, int] = (40, 40, 40),
        item_color: tuple[int, int, int] = (200, 200, 200),
        highlight_color: tuple[int, int, int] = (80, 180, 255),
    ) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.items = items
        self.item_height = item_height
        self.scrollbar_width = scrollbar_width
        self.bg_color = bg_color
        self.item_color = item_color
        self.highlight_color = highlight_color

        self.scroll_offset: float = 0.0
        self.scrollbar_dragging: bool = False
        self.highlighted_index: int | None = None
        self._drag_anchor_y: float = 0.0
        self._drag_anchor_offset: float = 0.0

    # -- derived -------------------------------------------------------
    @property
    def total_content_height(self) -> float:
        return self.item_height * len(self.items)

    @property
    def max_scroll(self) -> float:
        return max(0.0, self.total_content_height - self.height)

    def visible_items(self) -> list[tuple[int, str, int, int]]:
        """Return (index, text, y_start, y_end) for items visible in viewport."""
        result: list[tuple[int, str, int, int]] = []
        for i, text in enumerate(self.items):
            y_start = int(self.y + i * self.item_height - self.scroll_offset)
            y_end = y_start + self.item_height
            # Visible if overlaps viewport [self.y, self.y + self.height)
            if y_end > self.y and y_start < self.y + self.height:
                result.append((i, text, y_start, y_end))
        return result

    def item_at(self, px: int, py: int) -> int | None:
        """Return item index at pixel position, or None."""
        content_x_end = self.x + self.width - self.scrollbar_width
        if not (self.x <= px <= content_x_end and self.y <= py < self.y + self.height):
            return None
        content_y = py - self.y + self.scroll_offset
        idx = int(content_y / self.item_height)
        if 0 <= idx < len(self.items):
            return idx
        return None

    # -- scrollbar geometry --------------------------------------------
    def scrollbar_rect(self) -> tuple[int, int, int, int]:
        """Return (x, y, w, h) of the scrollbar handle."""
        sb_x = self.x + self.width - self.scrollbar_width
        total = self.total_content_height
        if total <= self.height:
            return (sb_x, self.y, self.scrollbar_width, self.height)
        handle_h = max(20, int(self.height * (self.height / total)))
        scrollable = self.height - handle_h
        frac = self.scroll_offset / self.max_scroll if self.max_scroll > 0 else 0.0
        handle_y = int(self.y + frac * scrollable)
        return (sb_x, handle_y, self.scrollbar_width, handle_h)

    def contains_scrollbar(self, px: int, py: int) -> bool:
        sx, sy, sw, sh = self.scrollbar_rect()
        return sx <= px <= sx + sw and sy <= py <= sy + sh

    # -- events --------------------------------------------------------
    def handle_mouse_down(self, px: int, py: int) -> None:
        if self.contains_scrollbar(px, py):
            self.scrollbar_dragging = True
            self._drag_anchor_y = py
            self._drag_anchor_offset = self.scroll_offset

    def handle_drag(self, px: int, py: int) -> None:
        if not self.scrollbar_dragging:
            return
        total = self.total_content_height
        if total <= self.height:
            return
        _, _, _, handle_h = self.scrollbar_rect()
        scrollable = self.height - handle_h
        if scrollable <= 0:
            return
        dy = py - self._drag_anchor_y
        new_offset = self._drag_anchor_offset + dy * (self.max_scroll / scrollable)
        self.scroll_offset = max(0.0, min(self.max_scroll, new_offset))

    def handle_mouse_up(self) -> None:
        self.scrollbar_dragging = False

    def handle_click(self, px: int, py: int) -> int | None:
        """Return clicked item index (content area only, not scrollbar)."""
        idx = self.item_at(px, py)
        if idx is not None:
            self.highlighted_index = idx
        return idx

    # -- rendering -----------------------------------------------------
    def render(self, surface: pygame.Surface, font: _ft.Font) -> None:
        import pygame

        # Background
        bg_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(surface, self.bg_color, bg_rect)

        # Clip to viewport
        clip_rect = pygame.Rect(self.x, self.y, self.width - self.scrollbar_width, self.height)
        prev_clip = surface.get_clip()
        surface.set_clip(clip_rect)

        for idx, text, ys, _ye in self.visible_items():
            # Highlight
            if idx == self.highlighted_index:
                hl_rect = pygame.Rect(self.x, ys, self.width - self.scrollbar_width, self.item_height)
                pygame.draw.rect(surface, self.highlight_color, hl_rect)

            text_surf, text_rect = font.render(text, fgcolor=self.item_color)
            ty = ys + (self.item_height - text_rect.height) // 2
            surface.blit(text_surf, (self.x + 6, ty))

        surface.set_clip(prev_clip)

        # Scrollbar track
        sb_x = self.x + self.width - self.scrollbar_width
        track_rect = pygame.Rect(sb_x, self.y, self.scrollbar_width, self.height)
        pygame.draw.rect(surface, (60, 60, 60), track_rect)

        # Scrollbar handle
        sx, sy, sw, sh = self.scrollbar_rect()
        handle_rect = pygame.Rect(sx, sy, sw, sh)
        pygame.draw.rect(surface, (140, 140, 140), handle_rect)


# -----------------------------------------------------------------------
# TextBlock
# -----------------------------------------------------------------------

class TextBlock:
    """Rendered paragraph with character-level hit testing and highlighting."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        text: str,
        font_size: int = 24,
        text_color: tuple[int, int, int] = (220, 220, 220),
        bg_color: tuple[int, int, int] = (45, 45, 45),
        highlight_color: tuple[int, int, int] = (50, 100, 200),
    ) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.text = text
        self.font_size = font_size
        self.text_color = text_color
        self.bg_color = bg_color
        self.highlight_color = highlight_color

        self.highlight_start: int | None = None
        self.highlight_end: int | None = None
        self._char_rects: list[tuple[int, int, int, int]] = []

    # -- layout --------------------------------------------------------
    def _compute_char_rects(self, font: _ft.Font) -> None:
        """Compute (x, y, w, h) for each character using font metrics."""
        rects: list[tuple[int, int, int, int]] = []
        cx = self.x
        cy = self.y
        line_h = font.get_sized_height()

        for i, ch in enumerate(self.text):
            if ch == "\n":
                rects.append((cx, cy, 0, line_h))
                cx = self.x
                cy += line_h
                continue

            r = font.get_rect(ch)
            cw = r.width if r.width > 0 else int(line_h * 0.4)

            # Word wrap: if this char would exceed block width AND it's not
            # the first char on this line, wrap.
            if cx + cw > self.x + self.width and cx > self.x:
                cx = self.x
                cy += line_h

            rects.append((cx, cy, cw, line_h))
            cx += cw

        self._char_rects = rects

    # -- queries -------------------------------------------------------
    def char_at(self, px: int, py: int) -> int | None:
        """Return character index at pixel, or None."""
        for i, (cx, cy, cw, ch) in enumerate(self._char_rects):
            if cx <= px < cx + max(cw, 1) and cy <= py < cy + ch:
                return i
        return None

    def word_bounds(self, word: str) -> tuple[int, int] | None:
        """Return (start, end) of first occurrence of *word* in text."""
        idx = self.text.find(word)
        if idx == -1:
            return None
        return (idx, idx + len(word))

    def highlighted_text(self) -> str:
        """Return the currently highlighted substring."""
        if self.highlight_start is None or self.highlight_end is None:
            return ""
        lo = min(self.highlight_start, self.highlight_end)
        hi = max(self.highlight_start, self.highlight_end)
        return self.text[lo:hi]

    def set_highlight(self, start: int, end: int) -> None:
        self.highlight_start = start
        self.highlight_end = end

    def clear_highlight(self) -> None:
        self.highlight_start = None
        self.highlight_end = None

    def get_char_x(self, idx: int) -> float:
        """X position of character at *idx* (for expert use)."""
        if 0 <= idx < len(self._char_rects):
            return float(self._char_rects[idx][0])
        # Past end — return right edge of last char
        if self._char_rects:
            lx, _ly, lw, _lh = self._char_rects[-1]
            return float(lx + lw)
        return float(self.x)

    # -- rendering -----------------------------------------------------
    def render(self, surface: pygame.Surface, font: _ft.Font) -> None:
        import pygame

        # Compute layout on first render (or if text changed)
        if len(self._char_rects) != len(self.text):
            self._compute_char_rects(font)

        # Background — compute bounding box from char rects
        if self._char_rects:
            max_y = max(cy + ch for (_, cy, _, ch) in self._char_rects)
        else:
            max_y = self.y + font.get_sized_height()
        bg_rect = pygame.Rect(self.x, self.y, self.width, max_y - self.y)
        pygame.draw.rect(surface, self.bg_color, bg_rect)

        # Highlight rects
        if self.highlight_start is not None and self.highlight_end is not None:
            lo = min(self.highlight_start, self.highlight_end)
            hi = max(self.highlight_start, self.highlight_end)
            for i in range(lo, min(hi, len(self._char_rects))):
                cx, cy, cw, ch = self._char_rects[i]
                pygame.draw.rect(surface, self.highlight_color, (cx, cy, max(cw, 1), ch))

        # Draw characters
        for i, ch in enumerate(self.text):
            if ch == "\n" or i >= len(self._char_rects):
                continue
            cx, cy, _cw, _ch = self._char_rects[i]
            font.render_to(surface, (cx, cy), ch, fgcolor=self.text_color)
