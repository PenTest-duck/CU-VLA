"""MiniEditorEnv — Pygame text editor environment for VLA training.

640x480 window, 512x384 observation. White background, thin border,
monospace text. No scrolling, no chrome. Physical keyboard model
with 53-key held-state edge detection.
"""

from __future__ import annotations

import os

import numpy as np

from .config import (
    ENV,
    KEY_DELETE,
    KEY_LSHIFT,
    KEY_RETURN,
    KEY_RSHIFT,
    KEY_SPACE,
    KEY_TAB,
    NUM_KEYS,
    PROPRIO_DIM,
    SHIFTED_CHARS,
    UNSHIFTED_CHARS,
)


class MiniEditorEnv:
    """Pygame text editor environment for VLA training.

    640x480 window, 512x384 observation. White background, thin border,
    monospace text. No scrolling, no chrome. Physical keyboard model
    with 53-key held-state edge detection.
    """

    def __init__(self, visual: bool = False, fps: int | None = None) -> None:
        self.visual = visual
        self.fps = fps if fps is not None else (ENV.control_hz if visual else 0)

        # Pygame state (lazy init)
        self._pygame_initialized: bool = False
        self._surface = None
        self._clock = None
        self._font = None

        # Font metrics (set in _init_pygame)
        self._char_width: int = 0
        self._line_height: int = 0
        self._chars_per_line: int = 0

        # Text state
        self._text: str = ""
        self._lines: list[str] = []
        self._text_cursor: int | None = None  # flat index, None = no cursor
        self._selection_start: int | None = None
        self._selection_end: int | None = None

        # Mouse state
        self._cursor_x: float = 0.0
        self._cursor_y: float = 0.0

        # Edge detection state
        self._prev_mouse_left: int = 0
        self._prev_keys_held: list[int] = [0] * NUM_KEYS

        # Episode state
        self._step_count: int = 0
        self._expected_text: str | None = None
        self._rng: np.random.Generator = np.random.default_rng()

    # ------------------------------------------------------------------
    # Pygame initialization
    # ------------------------------------------------------------------

    def _init_pygame(self) -> None:
        import pygame
        import pygame._freetype as _ft

        if self.visual:
            pygame.init()
            self._surface = pygame.display.set_mode(
                (ENV.window_w, ENV.window_h)
            )
            pygame.display.set_caption("Mini Editor")
        else:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.init()
            self._surface = pygame.Surface((ENV.window_w, ENV.window_h))

        self._clock = pygame.time.Clock()

        # Monospace font via freetype
        _ft.init()
        self._font = _ft.Font(None, ENV.font_size)
        self._font.strong = True

        # Measure font metrics
        rect = self._font.get_rect("A")
        self._char_width = rect.width
        self._line_height = self._font.get_sized_height()
        self._chars_per_line = (ENV.window_w - 2 * ENV.margin) // self._char_width

        self._pygame_initialized = True

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, text: str = "", seed: int | None = None) -> dict:
        """Reset with specific text content."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if not self._pygame_initialized:
            self._init_pygame()

        self._text = text
        self._rewrap()

        # Mouse cursor at center of editor area
        self._cursor_x = float(ENV.window_w // 2)
        self._cursor_y = float(ENV.window_h // 2)

        # No text cursor, no selection
        self._text_cursor = None
        self._selection_start = None
        self._selection_end = None

        # Reset edge detection
        self._prev_mouse_left = 0
        self._prev_keys_held = [0] * NUM_KEYS

        # Reset episode
        self._step_count = 0
        self._expected_text = None

        return self._get_observation()

    def step(self, action: dict) -> tuple[dict, bool, dict]:
        """Process one frame."""
        max_d = 50.0  # max delta px per frame
        dx = float(np.clip(action.get("dx", 0.0), -max_d, max_d))
        dy = float(np.clip(action.get("dy", 0.0), -max_d, max_d))
        mouse_left = int(action.get("mouse_left", 0))
        keys_held = list(action.get("keys_held", [0] * NUM_KEYS))

        self._step_count += 1

        # (1) Apply mouse delta, clamp to window
        self._cursor_x = float(np.clip(self._cursor_x + dx, 0, ENV.window_w - 1))
        self._cursor_y = float(np.clip(self._cursor_y + dy, 0, ENV.window_h - 1))

        # (2) Mouse edge detection
        prev_ml = self._prev_mouse_left
        if prev_ml == 0 and mouse_left == 1:
            # mouseDown
            shift_held = (
                self._prev_keys_held[KEY_LSHIFT] == 1
                or self._prev_keys_held[KEY_RSHIFT] == 1
            )
            click_pos = self._pixel_to_char_pos(self._cursor_x, self._cursor_y)
            if shift_held and self._text_cursor is not None:
                # Extend selection
                if self._selection_start is None:
                    self._selection_start = self._text_cursor
                self._selection_end = click_pos
                self._text_cursor = click_pos
            else:
                # Set cursor, clear selection
                self._text_cursor = click_pos
                self._selection_start = None
                self._selection_end = None
        # mouseUp: no action needed

        self._prev_mouse_left = mouse_left

        # (3) Key edge detection
        shift_held = keys_held[KEY_LSHIFT] == 1 or keys_held[KEY_RSHIFT] == 1

        for i in range(NUM_KEYS):
            prev_k = self._prev_keys_held[i]
            new_k = keys_held[i]
            if prev_k == 0 and new_k == 1:
                # keyDown transition
                if i in (KEY_LSHIFT, KEY_RSHIFT):
                    # Modifier: just note state change
                    continue

                if self._text_cursor is None:
                    # No cursor set, ignore key input
                    continue

                if i == KEY_DELETE:
                    self._delete_at_cursor()
                elif i == KEY_RETURN:
                    self._insert_text("\n")
                elif i == KEY_SPACE:
                    self._insert_text(" ")
                elif i == KEY_TAB:
                    self._insert_text("\t")
                else:
                    # Character key
                    ch = SHIFTED_CHARS.get(i) if shift_held else UNSHIFTED_CHARS.get(i)
                    if ch is not None:
                        self._insert_text(ch)

        self._prev_keys_held = list(keys_held)

        # (4) Check done
        done = False
        info: dict = {}
        if self._step_count >= ENV.max_steps:
            done = True
            info["timeout"] = True
        elif self._expected_text is not None and self._text == self._expected_text:
            done = True
            info["success"] = True

        info["steps"] = self._step_count

        # Render & observation
        obs = self._get_observation()

        if self.visual:
            import pygame

            self._clock.tick(self.fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

        return obs, done, info

    def set_expected_text(self, expected: str) -> None:
        """Set expected text for early termination check."""
        self._expected_text = expected

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_observation(self) -> dict:
        """Build observation dict."""
        import pygame
        import pygame.surfarray

        assert self._surface is not None

        # Render frame
        self._render()

        # Flip display if visual
        if self.visual:
            pygame.display.flip()

        # Read pixels: surfarray gives (W, H, 3), transpose to (H, W, 3)
        pixels = pygame.surfarray.array3d(self._surface)  # (W, H, 3)
        pixels = pixels.transpose(1, 0, 2)  # (H, W, 3)

        # Downscale to observation size
        obs_surface = pygame.transform.smoothscale(
            self._surface, (ENV.obs_w, ENV.obs_h)
        )
        obs_pixels = pygame.surfarray.array3d(obs_surface)  # (obs_w, obs_h, 3)
        obs_pixels = obs_pixels.transpose(1, 0, 2)  # (obs_h, obs_w, 3)
        screenshot = obs_pixels.copy().astype(np.uint8)

        # Proprio: [cursor_x_norm, cursor_y_norm, mouse_left, *keys_held]
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        proprio[0] = self._cursor_x / max(ENV.window_w - 1, 1)
        proprio[1] = self._cursor_y / max(ENV.window_h - 1, 1)
        proprio[2] = float(self._prev_mouse_left)
        for i in range(NUM_KEYS):
            proprio[3 + i] = float(self._prev_keys_held[i])

        return {
            "screenshot": screenshot,
            "proprio": proprio,
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> np.ndarray:
        """Render the full frame to the internal surface."""
        import pygame

        assert self._surface is not None
        assert self._font is not None

        surface = self._surface
        margin = ENV.margin

        # (1) White background
        surface.fill(ENV.bg_color)

        # (2) Thin gray border
        pygame.draw.rect(
            surface,
            ENV.border_color,
            (0, 0, ENV.window_w, ENV.window_h),
            ENV.border_width,
        )

        # (3) Selection highlight (drawn behind text)
        if (
            self._selection_start is not None
            and self._selection_end is not None
            and self._selection_start != self._selection_end
        ):
            sel_lo = min(self._selection_start, self._selection_end)
            sel_hi = max(self._selection_start, self._selection_end)
            self._draw_selection(surface, sel_lo, sel_hi)

        # (4) Text lines
        for line_idx, line_text in enumerate(self._lines):
            if not line_text:
                continue
            text_surf, _text_rect = self._font.render(line_text, fgcolor=(0, 0, 0))
            y = margin + line_idx * self._line_height
            surface.blit(text_surf, (margin, y))

        # (5) Text cursor (blinking at 2Hz)
        if self._text_cursor is not None:
            blink_period = ENV.control_hz / ENV.cursor_blink_hz
            if blink_period > 0 and (self._step_count % int(blink_period)) < (int(blink_period) // 2):
                cx, cy = self._char_pos_to_pixel(self._text_cursor)
                pygame.draw.line(
                    surface,
                    ENV.cursor_color,
                    (int(cx), int(cy)),
                    (int(cx), int(cy + self._line_height)),
                    1,
                )

        # (6) Mouse cursor (small dark triangle)
        mx, my = int(self._cursor_x), int(self._cursor_y)
        sz = ENV.mouse_cursor_size
        points = [
            (mx, my),
            (mx, my + sz),
            (mx + sz * 0.7, my + sz * 0.7),
        ]
        pygame.draw.polygon(surface, ENV.mouse_cursor_color, points)

        # Return raw pixels
        pixels = pygame.surfarray.array3d(surface)  # (W, H, 3)
        return pixels.transpose(1, 0, 2)  # (H, W, 3)

    def _draw_selection(self, surface, sel_lo: int, sel_hi: int) -> None:
        """Draw blue highlight rectangles for selected text."""
        import pygame

        margin = ENV.margin
        lo_line, lo_col = self._flat_to_line_col(sel_lo)
        hi_line, hi_col = self._flat_to_line_col(sel_hi)

        for line_idx in range(lo_line, hi_line + 1):
            if line_idx >= len(self._lines):
                break
            line_text = self._lines[line_idx]

            if line_idx == lo_line:
                start_col = lo_col
            else:
                start_col = 0

            if line_idx == hi_line:
                end_col = hi_col
            else:
                end_col = len(line_text)

            if start_col >= end_col and line_idx != hi_line:
                # Include the newline character visually as a small rect
                end_col = max(start_col + 1, len(line_text))

            x1 = margin + self._text_width(line_text[:start_col])
            x2 = margin + self._text_width(line_text[:end_col])
            y = margin + line_idx * self._line_height

            if x2 > x1:
                sel_rect = pygame.Rect(x1, y, x2 - x1, self._line_height)
                sel_surface = pygame.Surface((sel_rect.width, sel_rect.height), pygame.SRCALPHA)
                sel_surface.fill((*ENV.selection_color, 100))
                surface.blit(sel_surface, sel_rect.topleft)

    # ------------------------------------------------------------------
    # Text state helpers
    # ------------------------------------------------------------------

    def _text_width(self, text: str) -> int:
        """Measure the pixel width of a string using the freetype font."""
        if not text:
            return 0
        return self._font.get_rect(text).width

    def _flat_to_line_col(self, flat_idx: int) -> tuple[int, int]:
        """Convert flat character index to (line, col) in wrapped lines."""
        remaining = flat_idx
        for line_idx, line_text in enumerate(self._lines):
            line_len = len(line_text)
            # Account for newline: each line except the last has an implicit
            # newline that takes 1 position in flat space
            if line_idx < len(self._lines) - 1:
                # This line + its newline
                if remaining <= line_len:
                    return (line_idx, remaining)
                remaining -= line_len + 1  # +1 for newline
            else:
                # Last line, no trailing newline
                return (line_idx, min(remaining, line_len))
        return (max(0, len(self._lines) - 1), 0)

    def _line_col_to_flat(self, line: int, col: int) -> int:
        """Convert (line, col) to flat character index."""
        flat = 0
        for i in range(min(line, len(self._lines))):
            flat += len(self._lines[i])
            if i < len(self._lines) - 1:
                flat += 1  # newline
        if line < len(self._lines):
            flat += min(col, len(self._lines[line]))
        return min(flat, len(self._text))

    def _char_pos_to_pixel(self, flat_idx: int) -> tuple[float, float]:
        """Convert flat char index to pixel (x, y) position on screen."""
        line_idx, col = self._flat_to_line_col(flat_idx)
        line_text = self._lines[line_idx] if line_idx < len(self._lines) else ""
        x = ENV.margin + self._text_width(line_text[:col])
        y = ENV.margin + line_idx * self._line_height
        return (float(x), float(y))

    def _pixel_to_char_pos(self, px: float, py: float) -> int:
        """Convert pixel position to nearest flat character index."""
        margin = ENV.margin

        # Find closest line by y
        line_idx = int((py - margin) / self._line_height)
        line_idx = max(0, min(line_idx, len(self._lines) - 1))

        if not self._lines:
            return 0

        line_text = self._lines[line_idx]

        # Find closest character by x using font metrics
        best_col = 0
        best_dist = abs(px - margin)
        for col in range(1, len(line_text) + 1):
            char_x = margin + self._text_width(line_text[:col])
            dist = abs(px - char_x)
            if dist < best_dist:
                best_dist = dist
                best_col = col

        return self._line_col_to_flat(line_idx, best_col)

    def _insert_text(self, text_to_insert: str) -> None:
        """Insert text at current text cursor position."""
        if self._text_cursor is None:
            return

        # If selection active, delete selected range first
        if (
            self._selection_start is not None
            and self._selection_end is not None
            and self._selection_start != self._selection_end
        ):
            lo = min(self._selection_start, self._selection_end)
            hi = max(self._selection_start, self._selection_end)
            self._text = self._text[:lo] + self._text[hi:]
            self._text_cursor = lo
            self._selection_start = None
            self._selection_end = None

        # Insert at cursor
        pos = self._text_cursor
        self._text = self._text[:pos] + text_to_insert + self._text[pos:]
        self._text_cursor = pos + len(text_to_insert)
        self._rewrap()

    def _delete_at_cursor(self) -> None:
        """Delete selection if active, else delete one char before text cursor."""
        if self._text_cursor is None:
            return

        if (
            self._selection_start is not None
            and self._selection_end is not None
            and self._selection_start != self._selection_end
        ):
            lo = min(self._selection_start, self._selection_end)
            hi = max(self._selection_start, self._selection_end)
            self._text = self._text[:lo] + self._text[hi:]
            self._text_cursor = lo
            self._selection_start = None
            self._selection_end = None
        elif self._text_cursor > 0:
            pos = self._text_cursor
            self._text = self._text[: pos - 1] + self._text[pos:]
            self._text_cursor = pos - 1

        self._rewrap()

    def _rewrap(self) -> None:
        """Re-wrap self._text into self._lines using word wrapping."""
        if self._chars_per_line <= 0:
            self._lines = [self._text]
            return

        self._lines = []
        # First split by actual newlines
        paragraphs = self._text.split("\n")
        for para in paragraphs:
            if not para:
                self._lines.append("")
                continue
            # Word wrap each paragraph
            remaining = para
            while len(remaining) > self._chars_per_line:
                # Try to break at a space
                break_at = remaining.rfind(" ", 0, self._chars_per_line)
                if break_at <= 0:
                    # No space found, hard break
                    break_at = self._chars_per_line
                self._lines.append(remaining[:break_at])
                # Skip the space if we broke at one
                if break_at < len(remaining) and remaining[break_at] == " ":
                    remaining = remaining[break_at + 1 :]
                else:
                    remaining = remaining[break_at:]
            self._lines.append(remaining)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def text(self) -> str:
        """Current editor text content (flat string)."""
        return self._text

    @property
    def cursor_pos(self) -> tuple[float, float]:
        """Mouse cursor pixel position."""
        return (self._cursor_x, self._cursor_y)

    def close(self) -> None:
        """Shut down Pygame."""
        import pygame

        pygame.quit()
        self._pygame_initialized = False
