"""Draw-path task: draw a line or trace a shape by holding mouse and moving."""

from __future__ import annotations

import math

import numpy as np

from ..base_env import BaseTaskEnv
from ..config import ENV, EVAL_CFG


class DrawPathEnv(BaseTaskEnv):
    """Agent must draw a line or trace a shape by holding mouse and moving.

    The environment shows a reference path (dotted gray line) and the agent
    must hold the mouse button and trace along it.  Quality is measured by
    the mean nearest-point distance between the drawn path and the reference.
    """

    task_name = "draw-path"

    def __init__(
        self,
        path_type: str = "line",
        distance_threshold: float = 20.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.path_type = path_type
        self.distance_threshold = distance_threshold

        self._reference_path: list[tuple[float, float]] = []
        self._drawn_path: list[tuple[float, float]] = []
        self._drawing: bool = False
        self._evaluated: bool = False
        self._mean_distance: float = float("inf")

    # ------------------------------------------------------------------
    # Properties (for expert access)
    # ------------------------------------------------------------------

    @property
    def reference_path(self) -> list[tuple[float, float]]:
        return self._reference_path

    @property
    def drawn_path(self) -> list[tuple[float, float]]:
        return self._drawn_path

    # ------------------------------------------------------------------
    # Task setup
    # ------------------------------------------------------------------

    def _setup_task(self, rng: np.random.Generator) -> None:
        ibh = ENV.instruction_bar_height
        ws = ENV.window_size
        margin = 30

        self._drawn_path = []
        self._drawing = False
        self._evaluated = False
        self._mean_distance = float("inf")

        if self.path_type == "line":
            self._setup_line(rng, ibh, ws, margin)
        elif self.path_type == "circle":
            self._setup_circle(rng, ibh, ws, margin)
        else:
            raise ValueError(f"Unknown path_type: {self.path_type!r}")

    def _setup_line(
        self,
        rng: np.random.Generator,
        ibh: int,
        ws: int,
        margin: int,
    ) -> None:
        """Pick two random points at least 150px apart and sample path."""
        min_dist = 150.0
        for _ in range(200):
            ax = float(rng.integers(margin, ws - margin))
            ay = float(rng.integers(ibh + margin, ws - margin))
            bx = float(rng.integers(margin, ws - margin))
            by = float(rng.integers(ibh + margin, ws - margin))
            if math.hypot(bx - ax, by - ay) >= min_dist:
                break

        # Sample 20-40 evenly spaced points along the line
        num_pts = int(rng.integers(20, 41))
        self._reference_path = [
            (ax + (bx - ax) * t / (num_pts - 1),
             ay + (by - ay) * t / (num_pts - 1))
            for t in range(num_pts)
        ]
        self._start = (ax, ay)
        self._end = (bx, by)
        self.task_instruction = "Draw a line from A to B"

    def _setup_circle(
        self,
        rng: np.random.Generator,
        ibh: int,
        ws: int,
        margin: int,
    ) -> None:
        """Pick center and radius, generate reference points around circle."""
        radius = float(rng.integers(50, 101))
        # Ensure circle fits within the task area
        min_xy = margin + radius
        max_x = ws - margin - radius
        max_y = ws - margin - radius
        cx = float(rng.integers(int(min_xy), int(max_x) + 1))
        cy = float(rng.integers(int(max(min_xy, ibh + margin + radius)),
                                int(max_y) + 1))

        # Sample 30 evenly spaced points around the circle
        num_pts = 30
        self._reference_path = [
            (cx + radius * math.cos(2 * math.pi * t / num_pts),
             cy + radius * math.sin(2 * math.pi * t / num_pts))
            for t in range(num_pts)
        ]
        self._start = self._reference_path[0]
        self._end = self._reference_path[-1]
        self.task_instruction = "Trace the circle"

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------

    def _handle_mouse_down(self) -> None:
        self._drawing = True

    def _handle_drag(self) -> None:
        if self._drawing:
            self._drawn_path.append((self._cursor_x, self._cursor_y))

    def _handle_mouse_up(self) -> None:
        if self._drawing:
            self._drawing = False
            self._evaluate_path()

    # ------------------------------------------------------------------
    # Path evaluation
    # ------------------------------------------------------------------

    def _evaluate_path(self) -> None:
        """Compute mean nearest-point distance from drawn to reference."""
        self._evaluated = True
        if len(self._drawn_path) == 0:
            self._mean_distance = float("inf")
            return

        # Convert to numpy for efficient computation
        drawn = np.array(self._drawn_path)        # (N, 2)
        ref = np.array(self._reference_path)       # (M, 2)

        # For each drawn point, find the nearest reference point
        # Use broadcasting: (N, 1, 2) - (1, M, 2) -> (N, M, 2)
        diffs = drawn[:, None, :] - ref[None, :, :]  # (N, M, 2)
        dists = np.sqrt((diffs ** 2).sum(axis=2))     # (N, M)
        min_dists = dists.min(axis=1)                  # (N,)
        self._mean_distance = float(min_dists.mean())

    # ------------------------------------------------------------------
    # Success check
    # ------------------------------------------------------------------

    def _check_success(self) -> tuple[bool, dict]:
        if self._evaluated:
            min_points = len(self._reference_path) // 2
            if (self._mean_distance <= self.distance_threshold
                    and len(self._drawn_path) >= min_points):
                return True, {
                    "success": True,
                    "mean_distance": self._mean_distance,
                    "num_points": len(self._drawn_path),
                }
            else:
                return True, {
                    "failure": "path_too_far",
                    "mean_distance": self._mean_distance,
                    "num_points": len(self._drawn_path),
                }
        return False, {}

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_task(self, surface) -> None:
        import pygame

        # Draw reference path as dotted gray line
        gray = (120, 120, 120)
        for i, (rx, ry) in enumerate(self._reference_path):
            if i % 2 == 0:  # every other point for dotted effect
                pygame.draw.circle(surface, gray, (int(rx), int(ry)), 2)

        # Draw start/end markers
        if self.path_type == "line":
            # Start marker (green circle with "A")
            start_color = (60, 200, 80)
            end_color = (200, 60, 60)
            sx, sy = int(self._start[0]), int(self._start[1])
            ex, ey = int(self._end[0]), int(self._end[1])
            pygame.draw.circle(surface, start_color, (sx, sy), 12)
            pygame.draw.circle(surface, end_color, (ex, ey), 12)
            if self._font is not None:
                a_surf, a_rect = self._font.render("A", fgcolor=(255, 255, 255))
                surface.blit(a_surf, (sx - a_rect.width // 2,
                                      sy - a_rect.height // 2))
                b_surf, b_rect = self._font.render("B", fgcolor=(255, 255, 255))
                surface.blit(b_surf, (ex - b_rect.width // 2,
                                      ey - b_rect.height // 2))
        else:
            # Circle outline
            if len(self._reference_path) >= 2:
                pts = [(int(x), int(y)) for x, y in self._reference_path]
                pygame.draw.lines(surface, gray, closed=True, points=pts, width=1)

        # Draw agent's drawn path as solid colored line
        if len(self._drawn_path) >= 2:
            draw_color = (80, 180, 255)
            pts = [(int(x), int(y)) for x, y in self._drawn_path]
            pygame.draw.lines(surface, draw_color, closed=False, points=pts, width=2)

    # ------------------------------------------------------------------
    # Max steps
    # ------------------------------------------------------------------

    def _get_max_steps(self) -> int:
        return EVAL_CFG.max_steps_per_episode
