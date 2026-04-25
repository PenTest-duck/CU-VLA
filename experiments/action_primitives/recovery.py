"""Recovery trajectory generation for Phase B0.

Two mechanisms:
1. Start-chunk wrong segments: K=5-15 frames of deliberately-wrong actions at
   episode start, then clean expert recovery. 50/30/20 wrong-direction/overshoot/edge.
2. DART action noise: per-frame Gaussian on dx/dy in clean episodes (Task 11).

Both produce internally-consistent trajectories (env steps deterministically from
each action). Wrong-segment frames have loss_mask=0; DART noise frames have
loss_mask=1 (label is correct expert action from resulting state).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from experiments.action_primitives.config import ENV, MOUSE_CAP_PX, NUM_KEYS
from experiments.action_primitives.env import Action
from experiments.action_primitives.scene import Scene


SegmentType = Literal["wrong-direction", "overshoot", "edge-bang"]


@dataclass(frozen=True)
class WrongSegment:
    actions: tuple[Action, ...]   # K actions to apply
    k_frames: int
    segment_type: SegmentType
    final_cursor_xy: tuple[float, float]


# Distribution: 50% wrong-direction, 30% overshoot, 20% edge-bang.
_TYPE_PROBS = np.array([0.50, 0.30, 0.20])
_TYPE_LABELS: tuple[SegmentType, ...] = ("wrong-direction", "overshoot", "edge-bang")


def sample_wrong_segment_type(rng: np.random.Generator) -> SegmentType:
    return _TYPE_LABELS[rng.choice(len(_TYPE_LABELS), p=_TYPE_PROBS)]


def _idle_keys() -> np.ndarray:
    return np.full(NUM_KEYS, 2, dtype=np.int64)


def _step_cursor(cx: float, cy: float, dx: float, dy: float) -> tuple[float, float]:
    """Apply action delta + clip to canvas, like LClickEnv.step()."""
    new_x = float(np.clip(cx + dx, 0, ENV.canvas_w - 1))
    new_y = float(np.clip(cy + dy, 0, ENV.canvas_h - 1))
    return new_x, new_y


def _wrong_direction_segment(
    cursor_xy: tuple[float, float],
    target_center: tuple[float, float],
    k_frames: int,
    rng: np.random.Generator,
) -> tuple[list[Action], tuple[float, float]]:
    """Cursor heads ~90-180° away from target for k_frames."""
    cx, cy = cursor_xy
    tx, ty = target_center
    # Direction from cursor toward target
    to_target = np.array([tx - cx, ty - cy], dtype=np.float64)
    norm = np.linalg.norm(to_target)
    if norm < 1.0:
        to_target = np.array([1.0, 0.0])  # arbitrary
    else:
        to_target = to_target / norm
    # Wrong direction: rotate by 90-180 degrees
    angle = float(rng.uniform(np.pi / 2, np.pi)) * float(rng.choice([-1.0, 1.0]))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    wrong_dir = np.array([
        cos_a * to_target[0] - sin_a * to_target[1],
        sin_a * to_target[0] + cos_a * to_target[1],
    ])
    # Step magnitude per frame (moderate)
    step_mag = 25.0
    actions: list[Action] = []
    for _ in range(k_frames):
        dx, dy = wrong_dir * step_mag
        dx = float(np.clip(dx, -MOUSE_CAP_PX, MOUSE_CAP_PX))
        dy = float(np.clip(dy, -MOUSE_CAP_PX, MOUSE_CAP_PX))
        actions.append(Action(dx=dx, dy=dy, key_events=_idle_keys()))
        cx, cy = _step_cursor(cx, cy, dx, dy)
    return actions, (cx, cy)


def _overshoot_segment(
    cursor_xy: tuple[float, float],
    target_center: tuple[float, float],
    target_size: tuple[int, int],
    k_frames: int,
    rng: np.random.Generator,
) -> tuple[list[Action], tuple[float, float]]:
    """Cursor passes target by 1.5-2× its size, then needs to reverse."""
    cx, cy = cursor_xy
    tx, ty = target_center
    tw, th = target_size
    overshoot_factor = float(rng.uniform(1.5, 2.0))
    end_x = tx + (tx - cx) / max(np.linalg.norm([tx - cx, ty - cy]), 1.0) * tw * overshoot_factor
    end_y = ty + (ty - cy) / max(np.linalg.norm([tx - cx, ty - cy]), 1.0) * th * overshoot_factor
    # Linear interpolation cursor → end_x, end_y over k_frames
    actions: list[Action] = []
    for i in range(k_frames):
        t = (i + 1) / k_frames
        next_x = cx + (end_x - cx) * t
        next_y = cy + (end_y - cy) * t
        prev_x = cx + (end_x - cx) * (i / k_frames)
        prev_y = cy + (end_y - cy) * (i / k_frames)
        dx = float(np.clip(next_x - prev_x, -MOUSE_CAP_PX, MOUSE_CAP_PX))
        dy = float(np.clip(next_y - prev_y, -MOUSE_CAP_PX, MOUSE_CAP_PX))
        actions.append(Action(dx=dx, dy=dy, key_events=_idle_keys()))
    return actions, (float(np.clip(end_x, 0, ENV.canvas_w - 1)),
                     float(np.clip(end_y, 0, ENV.canvas_h - 1)))


def _edge_bang_segment(
    cursor_xy: tuple[float, float],
    k_frames: int,
    rng: np.random.Generator,
) -> tuple[list[Action], tuple[float, float]]:
    """Cursor heads to canvas edge, pinned for several frames before recovery point."""
    cx, cy = cursor_xy
    edge = rng.choice(["left", "right", "top", "bottom"])
    if edge == "left":
        end_x, end_y = 5.0, cy
    elif edge == "right":
        end_x, end_y = ENV.canvas_w - 5.0, cy
    elif edge == "top":
        end_x, end_y = cx, 5.0
    else:
        end_x, end_y = cx, ENV.canvas_h - 5.0
    actions: list[Action] = []
    travel_frames = max(1, k_frames - 4)  # remaining frames are "pinned at edge"
    for i in range(k_frames):
        if i < travel_frames:
            t = (i + 1) / travel_frames
            next_x = cx + (end_x - cx) * t
            next_y = cy + (end_y - cy) * t
            prev_x = cx + (end_x - cx) * (i / travel_frames)
            prev_y = cy + (end_y - cy) * (i / travel_frames)
            dx = float(np.clip(next_x - prev_x, -MOUSE_CAP_PX, MOUSE_CAP_PX))
            dy = float(np.clip(next_y - prev_y, -MOUSE_CAP_PX, MOUSE_CAP_PX))
        else:
            # Pinned: tiny no-op, env clip keeps cursor at edge
            dx, dy = 0.5, 0.5
        actions.append(Action(dx=dx, dy=dy, key_events=_idle_keys()))
    return actions, (end_x, end_y)


def generate_wrong_segment(
    scene: Scene,
    target_button_id: int,
    cursor_xy: tuple[float, float],
    segment_type: SegmentType,
    k_frames: int,
    rng: np.random.Generator,
) -> WrongSegment:
    target = scene.buttons[target_button_id]
    if segment_type == "wrong-direction":
        actions, final = _wrong_direction_segment(cursor_xy, target.center(), k_frames, rng)
    elif segment_type == "overshoot":
        actions, final = _overshoot_segment(
            cursor_xy, target.center(), (target.w, target.h), k_frames, rng,
        )
    elif segment_type == "edge-bang":
        actions, final = _edge_bang_segment(cursor_xy, k_frames, rng)
    else:
        raise ValueError(f"Unknown segment_type: {segment_type}")
    return WrongSegment(
        actions=tuple(actions), k_frames=k_frames,
        segment_type=segment_type, final_cursor_xy=final,
    )
