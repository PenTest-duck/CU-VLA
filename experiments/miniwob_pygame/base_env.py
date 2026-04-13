"""Abstract base environment for all MiniWoB-Pygame tasks."""

from __future__ import annotations

import os

os.environ["SDL_VIDEO_HIGHDPI_DISABLED"] = "1"

from abc import ABC, abstractmethod

import numpy as np

from .config import ACTION, ENV, EVAL_CFG, NUM_KEYS


class BaseTaskEnv(ABC):
    """Base class that all 12 MiniWoB-Pygame tasks extend.

    Handles shared Pygame rendering, cursor mechanics, held-state edge
    detection for mouse and keys, instruction bar rendering, and
    observation construction.
    """

    task_name: str = ""

    def __init__(self, visual: bool = False, fps: int | None = None) -> None:
        self.visual = visual
        self.fps = fps if fps is not None else (ENV.control_hz if visual else 0)
        self.task_instruction: str = ""

        # Pygame state (initialized lazily in _init_pygame)
        self._pygame_initialized: bool = False
        self._surface = None
        self._clock = None
        self._font = None
        self._instruction_font = None

        # Episode state
        self._cursor_x: float = 0.0
        self._cursor_y: float = 0.0
        self._mouse_pressed: bool = False
        self._keys_held: list[int] = [0] * NUM_KEYS
        self._step_count: int = 0
        self._done: bool = False
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
                (ENV.window_size, ENV.window_size)
            )
            pygame.display.set_caption(f"MiniWoB: {self.task_name}")
        else:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.init()
            self._surface = pygame.Surface(
                (ENV.window_size, ENV.window_size)
            )

        self._clock = pygame.time.Clock()
        _ft.init()
        self._font = _ft.Font(None, ENV.font_size)
        self._font.strong = True
        self._instruction_font = _ft.Font(None, ENV.instruction_font_size)
        self._instruction_font.strong = True
        self._pygame_initialized = True

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> dict:
        """Reset the environment and return the initial observation."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if not self._pygame_initialized:
            self._init_pygame()

        # Random cursor start
        ws = ENV.window_size
        self._cursor_x = float(self._rng.integers(0, ws))
        self._cursor_y = float(
            self._rng.integers(ENV.instruction_bar_height, ws)
        )

        # Reset state
        self._mouse_pressed = False
        self._keys_held = [0] * NUM_KEYS
        self._step_count = 0
        self._done = False

        self._setup_task(self._rng)
        return self._get_observation()

    def step(self, action: dict) -> tuple[dict, bool, dict]:
        """Execute one step and return (obs, done, info)."""
        assert not self._done, "Cannot step after episode is done"

        import pygame

        # Parse action
        max_d = ACTION.max_delta_px
        dx = float(np.clip(action.get("dx", 0.0), -max_d, max_d))
        dy = float(np.clip(action.get("dy", 0.0), -max_d, max_d))
        mouse_left = bool(action.get("mouse_left", 0))
        keys_held = list(action.get("keys_held", [0] * NUM_KEYS))

        self._step_count += 1

        # (1) Move cursor
        ws = ENV.window_size
        self._cursor_x = float(np.clip(self._cursor_x + dx, 0, ws - 1))
        self._cursor_y = float(np.clip(self._cursor_y + dy, 0, ws - 1))

        # (2) Mouse edge detection
        prev_mouse = self._mouse_pressed
        self._mouse_pressed = mouse_left
        if not prev_mouse and mouse_left:
            self._handle_mouse_down()
        if prev_mouse and not mouse_left:
            self._handle_mouse_up()
        if mouse_left:
            self._handle_drag()

        # (3) Key edge detection
        for i in range(NUM_KEYS):
            prev_key = self._keys_held[i]
            new_key = keys_held[i]
            if prev_key == 0 and new_key == 1:
                self._handle_key_down(i)
            elif prev_key == 1 and new_key == 0:
                self._handle_key_up(i)
        self._keys_held = list(keys_held)

        # (4) Check success
        success, info = self._check_success()
        if success:
            self._done = True
            info["steps"] = self._step_count

        # (5) Timeout
        if not self._done and self._step_count >= self._get_max_steps():
            self._done = True
            info["steps"] = self._step_count
            info["timeout"] = True

        # (6) Render
        obs = self._get_observation()

        if self.visual:
            self._clock.tick(self.fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._done = True

        return obs, self._done, info

    # ------------------------------------------------------------------
    # Abstract methods (subclass MUST implement)
    # ------------------------------------------------------------------

    @abstractmethod
    def _setup_task(self, rng: np.random.Generator) -> None:
        """Initialize task-specific state for a new episode."""
        ...

    @abstractmethod
    def _check_success(self) -> tuple[bool, dict]:
        """Return (success, info_dict) for the current state."""
        ...

    # ------------------------------------------------------------------
    # Optional hooks (subclass MAY override)
    # ------------------------------------------------------------------

    def _render_task(self, surface) -> None:  # noqa: ARG002
        """Draw task-specific elements onto the surface."""

    def _handle_mouse_down(self) -> None:
        """Called on mouse 0->1 transition."""

    def _handle_mouse_up(self) -> None:
        """Called on mouse 1->0 transition."""

    def _handle_drag(self) -> None:
        """Called every step while mouse is held."""

    def _handle_key_down(self, key_index: int) -> None:  # noqa: ARG002
        """Called on key 0->1 transition."""

    def _handle_key_up(self, key_index: int) -> None:  # noqa: ARG002
        """Called on key 1->0 transition."""

    def _get_max_steps(self) -> int:
        """Maximum steps before timeout."""
        return EVAL_CFG.max_steps_per_episode

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_instruction(self, surface) -> None:
        """Draw the instruction header bar."""
        import pygame

        # Background rect
        pygame.draw.rect(
            surface,
            ENV.instruction_bg_color,
            (0, 0, ENV.window_size, ENV.instruction_bar_height),
        )

        # Center instruction text
        if self.task_instruction and self._instruction_font is not None:
            text_surf, text_rect = self._instruction_font.render(
                self.task_instruction, fgcolor=(255, 255, 255)
            )
            x = (ENV.window_size - text_rect.width) // 2
            y = (ENV.instruction_bar_height - text_rect.height) // 2
            surface.blit(text_surf, (x, y))

    def _render_cursor(self, surface) -> None:
        """Draw the cursor as a white circle."""
        import pygame

        pygame.draw.circle(
            surface,
            ENV.cursor_color,
            (int(self._cursor_x), int(self._cursor_y)),
            ENV.cursor_radius,
        )

    def _get_observation(self) -> dict:
        """Build and return the observation dict."""
        import pygame.surfarray

        assert self._surface is not None

        # Fill background
        self._surface.fill(ENV.bg_color)

        # Render layers
        self._render_instruction(self._surface)
        self._render_task(self._surface)
        self._render_cursor(self._surface)

        # Flip display if visual
        if self.visual:
            import pygame
            pygame.display.flip()

        # Read pixels: surfarray gives (W, H, 3), transpose to (H, W, 3)
        pixels = pygame.surfarray.array3d(self._surface)  # (W, H, 3)
        pixels = pixels.transpose(1, 0, 2)  # (H, W, 3)

        # Resize to obs_size if needed
        ws = ENV.window_size
        obs_size = ENV.obs_size
        if ws != obs_size:
            # Use simple nearest-neighbor resize via numpy
            row_idx = (np.arange(obs_size) * ws / obs_size).astype(int)
            col_idx = (np.arange(obs_size) * ws / obs_size).astype(int)
            pixels = pixels[np.ix_(row_idx, col_idx)]

        return {
            "screenshot": pixels.copy().astype(np.uint8),
            "cursor_pos": np.array(
                [
                    self._cursor_x / max(ws - 1, 1),
                    self._cursor_y / max(ws - 1, 1),
                ],
                dtype=np.float32,
            ),
        }

    # ------------------------------------------------------------------
    # Properties & cleanup
    # ------------------------------------------------------------------

    @property
    def cursor_pos(self) -> tuple[float, float]:
        """Current cursor position in pixel coordinates."""
        return (self._cursor_x, self._cursor_y)

    def close(self) -> None:
        """Shut down Pygame."""
        import pygame

        pygame.quit()
        self._pygame_initialized = False
