"""Reactive Clicks environment: red circle appears, agent must click it.

Gym-style API with two rendering modes:
- Headless: renders to off-screen surface, returns numpy observations.
- Visual: opens a Pygame window for human observation/debugging.

Both modes use the same surfarray read path, so the agent sees identical pixels.
"""

import os
import math
import time

import numpy as np
import pygame

from .config import ENV, ACTION

# Disable Retina scaling to keep coordinates 1:1
os.environ.setdefault("SDL_VIDEO_HIGHDPI_DISABLED", "1")

# Action constants
BTN_NONE = 0
BTN_DOWN = 1
BTN_UP = 2


class ReactiveClicksEnv:
    """Reactive click reaction-time environment.

    Observation: (obs_size, obs_size, 3) uint8 RGB numpy array.
    Action: dict with keys 'dx' (float px), 'dy' (float px), 'btn' (int 0/1/2).
    """

    def __init__(self, visual: bool = False, fps: int | None = None):
        """
        Args:
            visual: If True, open a visible Pygame window. If False, headless.
            fps: If set, cap the frame rate. None = uncapped (as fast as possible).
        """
        self.visual = visual
        self.fps = fps if fps is not None else (ENV.control_hz if visual else 0)

        self._pygame_initialized = False
        self._surface: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None

        # Episode state
        self._cursor_x: float = 0.0
        self._cursor_y: float = 0.0
        self._circle_x: int = 0
        self._circle_y: int = 0
        self._circle_radius: int = 0
        self._target_visible: bool = False
        self._onset_time: float = 0.0
        self._step_count: int = 0
        self._done: bool = True
        self._rng = np.random.default_rng()

    def _init_pygame(self) -> None:
        if self._pygame_initialized:
            return
        if self.visual:
            pygame.init()
            self._surface = pygame.display.set_mode(
                (ENV.window_size, ENV.window_size)
            )
            pygame.display.set_caption("Reactive Clicks")
        else:
            # Headless: only init the display module with a dummy driver
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.init()
            self._surface = pygame.Surface((ENV.window_size, ENV.window_size))
        self._clock = pygame.time.Clock()
        self._pygame_initialized = True

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset environment for a new episode.

        Returns:
            Initial observation (circle already visible, cursor at random position).
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._init_pygame()

        # Random cursor start position
        self._cursor_x = float(self._rng.integers(0, ENV.window_size))
        self._cursor_y = float(self._rng.integers(0, ENV.window_size))

        # Random circle
        self._circle_radius = int(
            self._rng.integers(ENV.circle_radius_min, ENV.circle_radius_max + 1)
        )
        pad = self._circle_radius
        self._circle_x = int(
            self._rng.integers(pad, ENV.window_size - pad)
        )
        self._circle_y = int(
            self._rng.integers(pad, ENV.window_size - pad)
        )

        self._target_visible = True
        self._onset_time = time.perf_counter()
        self._step_count = 0
        self._done = False

        return self._render_obs()

    def step(self, action: dict) -> tuple[np.ndarray, bool, dict]:
        """Execute one action and return (obs, done, info).

        Args:
            action: Dict with 'dx' (float pixels), 'dy' (float pixels), 'btn' (0/1/2).

        Returns:
            obs: RGB observation array.
            done: True if episode ended (click occurred or max steps).
            info: Dict with timing/hit metadata on episode end.
        """
        assert not self._done, "Episode is done. Call reset()."

        dx_px = float(np.clip(action["dx"], -ACTION.max_delta_px, ACTION.max_delta_px))
        dy_px = float(np.clip(action["dy"], -ACTION.max_delta_px, ACTION.max_delta_px))
        btn = action["btn"]

        # Apply movement
        self._cursor_x = float(np.clip(
            self._cursor_x + dx_px, 0, ENV.window_size - 1
        ))
        self._cursor_y = float(np.clip(
            self._cursor_y + dy_px, 0, ENV.window_size - 1
        ))

        self._step_count += 1
        info: dict = {}

        # Check for click
        if btn == BTN_DOWN:
            click_time = time.perf_counter()
            dist = math.hypot(
                self._cursor_x - self._circle_x,
                self._cursor_y - self._circle_y,
            )
            hit = dist <= self._circle_radius
            self._done = True
            info = {
                "hit": hit,
                "reaction_time_s": click_time - self._onset_time,
                "steps": self._step_count,
                "circle_x": self._circle_x,
                "circle_y": self._circle_y,
                "circle_radius": self._circle_radius,
                "cursor_x": self._cursor_x,
                "cursor_y": self._cursor_y,
                "distance_to_center": dist,
            }

        obs = self._render_obs()

        if self.visual and self.fps > 0:
            self._clock.tick(self.fps)

        # Handle Pygame events to prevent window from becoming unresponsive
        if self.visual:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._done = True
                    info["quit"] = True

        return obs, self._done, info

    def _render_obs(self) -> np.ndarray:
        """Render the current state to the surface and return as numpy array."""
        surface = self._surface

        # Background
        surface.fill(ENV.bg_color)

        # Circle (if visible)
        if self._target_visible:
            pygame.draw.circle(
                surface,
                ENV.circle_color,
                (self._circle_x, self._circle_y),
                self._circle_radius,
            )

        # Cursor
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
    def circle_pos(self) -> tuple[int, int]:
        return (self._circle_x, self._circle_y)

    @property
    def circle_radius(self) -> int:
        return self._circle_radius
