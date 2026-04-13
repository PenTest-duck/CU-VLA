"""Expert policy for the draw-path task.

Navigates to the first reference point, then traces along the
reference path using small delta steps with slight noise.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import ACTION, NUM_KEYS
from .common import fitts_trajectory, noop_action, pause_actions, run_episode


def generate_trajectory(
    cursor_x: float,
    cursor_y: float,
    reference_path: list[tuple[float, float]],
    rng: np.random.Generator,
) -> list[dict]:
    """Generate expert trajectory for the draw-path task.

    1. Navigate to reference_path[0] using fitts_trajectory (mouse up).
    2. mouse_left=1 (start drawing).
    3. For each consecutive pair of reference points, produce 3-5 small
       dx/dy steps with slight noise to move between them (mouse held).
    4. mouse_left=0 (finish drawing).

    Args:
        cursor_x, cursor_y: Current cursor position.
        reference_path: List of (x, y) reference points to trace.
        rng: Numpy random generator.

    Returns:
        List of action dicts.
    """
    if not reference_path:
        return [noop_action()]

    actions: list[dict] = []
    max_delta = ACTION.max_delta_px

    # 1. Navigate to start of path
    sx, sy = reference_path[0]
    move_actions = fitts_trajectory(
        cursor_x, cursor_y, sx, sy, 20.0, rng, mouse_held=False
    )
    actions.extend(move_actions)

    # Track simulated cursor
    cx, cy = cursor_x, cursor_y
    for a in move_actions:
        cx = max(0.0, min(cx + a["dx"], 399.0))
        cy = max(0.0, min(cy + a["dy"], 399.0))

    # 2. Mouse down (start drawing)
    actions.append({
        "dx": 0.0,
        "dy": 0.0,
        "mouse_left": 1,
        "keys_held": [0] * NUM_KEYS,
    })

    # 3. Trace along reference path with small steps
    for i in range(len(reference_path) - 1):
        px, py = reference_path[i]
        qx, qy = reference_path[i + 1]

        seg_dx = qx - px
        seg_dy = qy - py
        seg_len = math.hypot(seg_dx, seg_dy)

        # 3-5 steps per segment
        num_steps = int(rng.integers(3, 6))

        for s in range(num_steps):
            # Target position for this micro-step
            frac = (s + 1) / num_steps
            target_x = px + seg_dx * frac
            target_y = py + seg_dy * frac

            # Compute delta from current simulated position
            dx = target_x - cx + rng.normal(0, 1.0)
            dy = target_y - cy + rng.normal(0, 1.0)

            # Clamp to max delta
            dx = float(np.clip(dx, -max_delta, max_delta))
            dy = float(np.clip(dy, -max_delta, max_delta))

            actions.append({
                "dx": dx,
                "dy": dy,
                "mouse_left": 1,
                "keys_held": [0] * NUM_KEYS,
            })
            cx = max(0.0, min(cx + dx, 399.0))
            cy = max(0.0, min(cy + dy, 399.0))

    # 4. Mouse up (finish drawing)
    actions.append({
        "dx": 0.0,
        "dy": 0.0,
        "mouse_left": 0,
        "keys_held": [0] * NUM_KEYS,
    })

    return actions


def run_expert_episode(
    env,
    rng: np.random.Generator,
    seed: int | None = None,
) -> tuple[list[np.ndarray], list[dict], dict]:
    """Run a full expert episode for draw-path.

    Args:
        env: DrawPathEnv instance.
        rng: Numpy random generator for trajectory noise.
        seed: Optional seed for env.reset().

    Returns:
        (observations, actions, final_info)
    """
    env.reset(seed=seed)
    trajectory = generate_trajectory(
        *env.cursor_pos, env.reference_path, rng
    )
    return run_episode(env, trajectory)
