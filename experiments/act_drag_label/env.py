"""Drag-and-Label environment: drag shapes to matching zones, then type labels.

Gym-style API with two rendering modes:
- Headless: renders to off-screen surface, returns numpy observations.
- Visual: opens a Pygame window for human observation/debugging.

Both modes use the same surfarray read path, so the agent sees identical pixels.
"""

import os

import numpy as np
import pygame
import pygame._freetype as _ft

from .config import ENV, ACTION, VOCAB

# Disable Retina scaling to keep coordinates 1:1
os.environ.setdefault("SDL_VIDEO_HIGHDPI_DISABLED", "1")

# Click state constants
CLICK_NONE = 0
CLICK_DOWN = 1
CLICK_UP = 2


class DragLabelEnv:
    """Drag-and-label environment.

    Observation: (obs_size, obs_size, 3) uint8 RGB numpy array.
    Action: dict with keys 'dx', 'dy' (float px), 'click' (int 0/1/2), 'key' (int).
    """

    def __init__(self, visual: bool = False, fps: int | None = None,
                 num_shapes: int = 1):
        """
        Args:
            visual: If True, open a visible Pygame window. If False, headless.
            fps: If set, cap the frame rate. None = uncapped (as fast as possible).
            num_shapes: Number of shapes to place (max ENV.max_shapes).
        """
        self.visual = visual
        self.fps = fps if fps is not None else (ENV.control_hz if visual else 0)
        self.num_shapes = min(num_shapes, ENV.max_shapes)

        self._pygame_initialized = False
        self._surface: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: _ft.Font | None = None

        # Episode state
        self._cursor_x: float = 0.0
        self._cursor_y: float = 0.0
        self._click_state: int = CLICK_NONE
        self._current_key: int = 0
        self._grabbed_shape_idx: int = -1
        self._shapes: list[dict] = []
        self._zones: list[dict] = []
        self._rng = np.random.default_rng()

    def _init_pygame(self) -> None:
        if self._pygame_initialized:
            return
        if self.visual:
            pygame.init()
            self._surface = pygame.display.set_mode(
                (ENV.window_size, ENV.window_size)
            )
            pygame.display.set_caption("Drag & Label")
        else:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.init()
            self._surface = pygame.Surface((ENV.window_size, ENV.window_size))
        self._clock = pygame.time.Clock()
        _ft.init()
        self._font = _ft.Font(None, ENV.font_size)
        self._font.strong = True
        self._pygame_initialized = True

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset environment for a new episode.

        Returns:
            Initial observation with shapes on left, zones on right.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._init_pygame()

        # Random cursor start
        self._cursor_x = float(self._rng.integers(0, ENV.window_size))
        self._cursor_y = float(self._rng.integers(0, ENV.window_size))

        self._click_state = CLICK_NONE
        self._current_key = 0
        self._grabbed_shape_idx = -1

        # Pick labels (no replacement)
        label_indices = self._rng.choice(
            len(VOCAB), size=self.num_shapes, replace=False
        )
        labels = [VOCAB[i] for i in label_indices]

        # --- Place shapes on left half (non-overlapping) ---
        self._shapes = []
        for i in range(self.num_shapes):
            w = int(self._rng.integers(ENV.shape_width_min, ENV.shape_width_max + 1))
            h = ENV.shape_height
            x = int(self._rng.integers(ENV.shape_x_min, ENV.shape_x_max - w + 1))
            y = self._find_non_overlapping_y(
                self._shapes, h, gap=ENV.shape_height + 15
            )
            shape = {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "color": ENV.shape_colors[i],
                "label": labels[i],
                "grabbed": False,
                "dropped": False,
                "typed_so_far": "",
                "complete": False,
                "original_x": x,
                "original_y": y,
            }
            self._shapes.append(shape)

        # --- Place zones on right half (non-overlapping) ---
        self._zones = []
        for i in range(self.num_shapes):
            w = ENV.zone_width
            h = ENV.zone_height
            x = int(self._rng.integers(ENV.zone_x_min, ENV.zone_x_max - w + 1))
            y = self._find_non_overlapping_y(
                self._zones, h, gap=ENV.zone_height + 15
            )
            zone = {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "color": ENV.shape_colors[i],
                "target_label": labels[i],
                "typed_text": "",
            }
            self._zones.append(zone)

        return self._render_obs()

    def _find_non_overlapping_y(
        self, placed: list[dict], item_height: int, gap: int
    ) -> int:
        """Find a random y that doesn't overlap with already-placed items.

        Args:
            placed: List of already-placed dicts (each has 'y' and 'height').
            item_height: Height of the new item.
            gap: Minimum vertical gap between items.

        Returns:
            A valid y coordinate.
        """
        margin = 20
        y_max = ENV.window_size - item_height - margin
        for _ in range(100):
            y = int(self._rng.integers(margin, y_max + 1))
            overlap = False
            for other in placed:
                if (y < other["y"] + other["height"] + gap
                        and y + item_height + gap > other["y"]):
                    overlap = True
                    break
            if not overlap:
                return y
        # Fallback: stack below the last placed item
        if placed:
            return placed[-1]["y"] + placed[-1]["height"] + gap
        return margin

    def _render_obs(self) -> np.ndarray:
        """Render the current state to the surface and return as numpy array."""
        surface = self._surface

        # Background
        surface.fill(ENV.bg_color)

        # --- Drop zones: outlined rectangles with colored borders ---
        for zone in self._zones:
            rect = pygame.Rect(zone["x"], zone["y"], zone["width"], zone["height"])
            pygame.draw.rect(surface, zone["color"], rect, ENV.zone_border_width)

            # Typed text inside zone (white, centered)
            if zone["typed_text"]:
                text_surf, text_rect = self._font.render(
                    zone["typed_text"], (255, 255, 255)
                )
                text_rect.center = rect.center
                surface.blit(text_surf, text_rect)

        # --- Shapes: filled rounded rectangles with white bold text ---
        for shape in self._shapes:
            if shape["complete"]:
                continue
            rect = pygame.Rect(
                shape["x"], shape["y"], shape["width"], shape["height"]
            )
            pygame.draw.rect(surface, shape["color"], rect, border_radius=8)

            # White bold monospace label centered
            text_surf, text_rect = self._font.render(
                shape["label"], (255, 255, 255)
            )
            text_rect.center = rect.center
            surface.blit(text_surf, text_rect)

        # --- Cursor: white dot ---
        pygame.draw.circle(
            surface,
            ENV.cursor_color,
            (int(round(self._cursor_x)), int(round(self._cursor_y))),
            ENV.cursor_radius,
        )

        if self.visual:
            pygame.display.flip()

        # Read pixels: surfarray gives (W, H, 3), we need (H, W, 3)
        pixels = pygame.surfarray.array3d(surface).transpose(1, 0, 2)

        # Resize to observation size
        if pixels.shape[0] != ENV.obs_size or pixels.shape[1] != ENV.obs_size:
            obs_surface = pygame.transform.scale(
                surface, (ENV.obs_size, ENV.obs_size)
            )
            pixels = pygame.surfarray.array3d(obs_surface).transpose(1, 0, 2)

        return pixels.copy()

    def close(self) -> None:
        if self._pygame_initialized:
            pygame.quit()
            self._pygame_initialized = False

    @property
    def cursor_pos(self) -> tuple[float, float]:
        return (self._cursor_x, self._cursor_y)

    @property
    def shapes(self) -> list[dict]:
        return self._shapes

    @property
    def zones(self) -> list[dict]:
        return self._zones
