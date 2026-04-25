"""Pygame L-click environment for Experiment 6 Phase A.

Canvas: 720x450 white with one colored button.
Goal: cursor over button, L_press → L_release → success.
Headless by default; pass visual=True to LClickEnv(...) to open a
pygame display window (for --visual eval).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pygame
from PIL import Image

from experiments.action_primitives.config import ENV, NUM_KEYS


# Theme palettes (minimal Phase A set; full 20-theme system is Phase B)
THEMES = {
    "flat-modern":  {"bg": (245, 245, 248), "button": (80, 130, 230), "label": (255, 255, 255)},
    "flat-minimal": {"bg": (255, 255, 255), "button": (30, 30, 30),   "label": (255, 255, 255)},
    "dark-mode":    {"bg": (28, 30, 36),    "button": (100, 200, 140),"label": (10, 10, 10)},
}


@dataclass
class Action:
    """Per-frame action. Matches 6-head structure except done is training-only."""
    dx: float = 0.0
    dy: float = 0.0
    click: int = 0                          # 0=idle, 1=L_press, 2=L_release, 3=R_press, 4=R_release
    scroll: float = 0.0
    key_events: np.ndarray = field(         # (77,) int: 0=press, 1=release, 2=idle
        default_factory=lambda: np.full(NUM_KEYS, 2, dtype=np.int64)
    )


@dataclass
class Proprio:
    cursor_x: float
    cursor_y: float
    held_keys: np.ndarray                   # (77,) bool
    held_mouse: np.ndarray                  # (3,)  bool  [L, R, middle]
    capslock: bool


class LClickEnv:
    """L-click primitive environment.

    reset() returns initial observation + info.
    step(action) returns (obs, done, info).
    """

    def __init__(self, theme: str = "flat-modern", seed: int = 0,
                 visual: bool = False, fps: int = 30) -> None:
        """Create a new L-click environment.

        Args:
            theme: Palette key in THEMES.
            seed: RNG seed for target + cursor randomization.
            visual: If True, open a real pygame display window (for live eval).
                If False, render to an offscreen Surface with the SDL dummy
                driver (headless, used for data generation + training).
            fps: Frame rate cap when ``visual=True``. Ignored otherwise.
        """
        self.visual = visual
        self.fps = fps
        if visual:
            # Remove a possibly-stale dummy driver so SDL picks the real one.
            if os.environ.get("SDL_VIDEODRIVER") == "dummy":
                del os.environ["SDL_VIDEODRIVER"]
            pygame.init()
            self.screen = pygame.display.set_mode((ENV.canvas_w, ENV.canvas_h))
            pygame.display.set_caption("CU-VLA Exp6 · L-click eval")
            self._clock: pygame.time.Clock | None = pygame.time.Clock()
        else:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.init()
            self.screen = pygame.Surface((ENV.canvas_w, ENV.canvas_h))
            self._clock = None
        self.rng = np.random.default_rng(seed)
        self.theme = THEMES[theme]
        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None) -> tuple[dict, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Randomize button position and size
        btn_w = int(self.rng.integers(40, 120))
        btn_h = int(self.rng.integers(30, 80))
        margin = 20
        btn_x = int(self.rng.integers(margin, ENV.canvas_w - btn_w - margin))
        btn_y = int(self.rng.integers(margin, ENV.canvas_h - btn_h - margin))
        self.target_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
        self.target_color = self.theme["button"]
        # Cursor starts at random position far from target. Bound the retry
        # loop to avoid an unbounded spin in degenerate cases (very large
        # button, small canvas).
        for _ in range(100):
            cx = int(self.rng.integers(10, ENV.canvas_w - 10))
            cy = int(self.rng.integers(10, ENV.canvas_h - 10))
            if not self.target_rect.collidepoint(cx, cy):
                break
        else:
            raise RuntimeError(
                f"Could not find a cursor-off-target position after 100 tries. "
                f"Target rect {self.target_rect} vs canvas ({ENV.canvas_w}x{ENV.canvas_h})."
            )
        self.cursor_x = float(cx)
        self.cursor_y = float(cy)
        self.held_keys = np.zeros(NUM_KEYS, dtype=bool)
        self.held_mouse = np.zeros(3, dtype=bool)
        self.capslock = False
        self.done_flag = False
        self._press_frame: Optional[int] = None
        self._release_frame: Optional[int] = None
        self.frame_idx = 0

        obs = self._render_obs()
        info = self._info()
        return obs, info

    def step(self, action: Action) -> tuple[dict, bool, dict]:
        # Apply mouse delta (clipped to canvas)
        self.cursor_x = float(np.clip(self.cursor_x + action.dx, 0, ENV.canvas_w - 1))
        self.cursor_y = float(np.clip(self.cursor_y + action.dy, 0, ENV.canvas_h - 1))
        # Apply click event
        if action.click == 1:  # L_press
            self.held_mouse[0] = True
            if self._press_frame is None and self.target_rect.collidepoint(self.cursor_x, self.cursor_y):
                self._press_frame = self.frame_idx
        elif action.click == 2:  # L_release
            self.held_mouse[0] = False
            if self._release_frame is None and self._press_frame is not None:
                self._release_frame = self.frame_idx
                if self.target_rect.collidepoint(self.cursor_x, self.cursor_y):
                    self.done_flag = True
        # (R_press / R_release ignored for L-click primitive success)
        # Apply key events (press=0, release=1, idle=2)
        press_mask = action.key_events == 0
        release_mask = action.key_events == 1
        self.held_keys[press_mask] = True
        self.held_keys[release_mask] = False
        self.frame_idx += 1
        obs = self._render_obs()
        info = self._info()
        return obs, self.done_flag, info

    def _render_obs(self) -> dict:
        self.screen.fill(self.theme["bg"])
        pygame.draw.rect(self.screen, self.target_color, self.target_rect, border_radius=6)
        # Cursor: simple arrow (macOS default-ish ~32px sprite)
        cx, cy = int(self.cursor_x), int(self.cursor_y)
        pygame.draw.polygon(
            self.screen,
            (0, 0, 0),
            [(cx, cy), (cx + 14, cy + 10), (cx + 8, cy + 12), (cx + 12, cy + 20), (cx + 9, cy + 21), (cx + 5, cy + 13), (cx, cy + 16)],
        )
        # PIL Image for SigLIP2 naflex compatibility
        arr = pygame.surfarray.array3d(self.screen).transpose(1, 0, 2).copy()
        img = Image.fromarray(arr, mode="RGB")
        proprio = Proprio(
            cursor_x=self.cursor_x / ENV.canvas_w,
            cursor_y=self.cursor_y / ENV.canvas_h,
            held_keys=self.held_keys.copy(),
            held_mouse=self.held_mouse.copy(),
            capslock=self.capslock,
        )
        if self.visual:
            # Pump the event queue so macOS doesn't flag the window as "not
            # responding", flush the drawn surface to the visible display, and
            # cap the frame rate so humans can watch.
            pygame.event.pump()
            pygame.display.flip()
            if self._clock is not None:
                self._clock.tick(self.fps)
        return {"image": img, "proprio": proprio}

    def _info(self) -> dict:
        return {
            "target_bbox": (self.target_rect.x, self.target_rect.y, self.target_rect.w, self.target_rect.h),
            "target_color": self.target_color,
            "cursor_xy": (self.cursor_x, self.cursor_y),
            "frame_idx": self.frame_idx,
            "success": self.done_flag,
        }
