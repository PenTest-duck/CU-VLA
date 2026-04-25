"""Fitts-law expert for L-click primitive.

Generates a trajectory of per-frame actions that drive the cursor to the target,
settles for a sampled number of frames, then emits L_press and L_release.
Includes tempo variability (slow / normal / fast / superhuman) per Q5/Q9.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from experiments.action_primitives.config import MOUSE_CAP_PX, NUM_KEYS
from experiments.action_primitives.env import Action


TEMPO_PROFILES = {
    "slow":       {"peak_speed_px": 18.0, "settle_frames": (2, 5)},
    "normal":     {"peak_speed_px": 35.0, "settle_frames": (1, 3)},
    "fast":       {"peak_speed_px": 60.0, "settle_frames": (0, 2)},
    "superhuman": {"peak_speed_px": 95.0, "settle_frames": (0, 1)},
}
# `settle_frames` = (low, high) interpreted as half-open `rng.integers(low, high)`;
# the state machine adds +1 to the sample so effective settle count is in
# [low+1, high]. New profiles must satisfy high > low (else `rng.integers(n, n)`
# raises `ValueError: high <= low`).


@dataclass
class LClickExpertConfig:
    tempo: str = "normal"          # "slow" | "normal" | "fast" | "superhuman"
    overshoot_prob: float = 0.1     # probability of a human-like overshoot-correct
    seed: int = 0


def _idle_keys() -> np.ndarray:
    return np.full(NUM_KEYS, 2, dtype=np.int64)  # 2 == idle


class LClickExpert:
    """Iterator yielding per-frame Actions that drive L-click completion."""

    def __init__(
        self,
        cfg: LClickExpertConfig,
        cursor_xy: tuple[float, float],
        target_center: tuple[float, float],
    ) -> None:
        self.rng = np.random.default_rng(cfg.seed)
        self.cfg = cfg
        self.cursor = np.array(cursor_xy, dtype=np.float64)
        self.target = np.array(target_center, dtype=np.float64)
        self.profile = TEMPO_PROFILES[cfg.tempo]
        # Guard the half-open interval convention; rng.integers(n, n) raises.
        low, high = self.profile["settle_frames"]
        if high <= low:
            raise ValueError(
                f"Tempo '{cfg.tempo}' has settle_frames={self.profile['settle_frames']}; "
                "high must be strictly greater than low (rng.integers is half-open)."
            )
        # State machine: move -> settle -> press -> release -> done.
        # `_move_step()` already returns one zero-motion frame on the
        # transition from "move" → "settle" (the arrival frame), so
        # settle_remaining counts ONLY the additional idle frames after that.
        # Earlier code added an extra `+ 1` here, which made every episode
        # pause 1 frame longer than the tempo profile specifies. Phase A's
        # uploaded checkpoint was trained with the +1 behaviour; this fix
        # only affects future Phase B data regenerations.
        self.state = "move"
        self.settle_remaining = int(self.rng.integers(low, high))
        self._overshoot_done = False

    def _move_step(self) -> Action:
        to_target = self.target - self.cursor
        dist = np.linalg.norm(to_target)
        if dist < 1.0:
            # Arrived; transition to settle
            self.state = "settle"
            return Action(dx=0.0, dy=0.0, key_events=_idle_keys())
        # Velocity profile: minimum-jerk-ish — peak in middle, taper ends
        peak = self.profile["peak_speed_px"]
        step_mag = min(peak, dist)
        direction = to_target / dist
        # Random overshoot near end
        if (not self._overshoot_done
            and dist < peak * 2
            and self.rng.random() < self.cfg.overshoot_prob):
            step_mag = min(dist * 1.4, MOUSE_CAP_PX)
            self._overshoot_done = True
        dx, dy = direction * step_mag
        # Clip to mouse cap
        dx = float(np.clip(dx, -MOUSE_CAP_PX, MOUSE_CAP_PX))
        dy = float(np.clip(dy, -MOUSE_CAP_PX, MOUSE_CAP_PX))
        self.cursor = self.cursor + np.array([dx, dy])
        return Action(dx=dx, dy=dy, key_events=_idle_keys())

    def __iter__(self) -> Iterator[Action]:
        return self

    def __next__(self) -> Action:
        if self.state == "move":
            return self._move_step()
        if self.state == "settle":
            if self.settle_remaining > 0:
                self.settle_remaining -= 1
                return Action(dx=0.0, dy=0.0, key_events=_idle_keys())
            self.state = "press"
            return Action(dx=0.0, dy=0.0, click=1, key_events=_idle_keys())  # L_press
        if self.state == "press":
            self.state = "release"
            return Action(dx=0.0, dy=0.0, click=2, key_events=_idle_keys())  # L_release
        # After release, stop iterating
        raise StopIteration
