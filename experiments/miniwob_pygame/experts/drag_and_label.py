"""Expert policy for the drag-and-label task.

Navigates to each shape, drags it to the matching zone, drops it,
then types the shape's label.  Uses nearest-first ordering when
multiple shapes are present.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import NUM_KEYS, char_to_key_index
from .common import fitts_trajectory, noop_action, pause_actions, run_episode


def generate_trajectory(
    cursor_x: float,
    cursor_y: float,
    shapes: list[dict],
    zones: list[dict],
    rng: np.random.Generator,
) -> list[dict]:
    """Generate expert trajectory for drag-and-label task.

    For each shape (nearest-first):
      1. Navigate to shape center (mouse not held)
      2. mouse_left=1 (grab)
      3. Drag to matching zone center (mouse held)
      4. mouse_left=0 (drop)
      5. Pause
      6. Type label: for each char, keys_held[char_to_key_index(ch)]=1
         for one step, then all 0 for the next step
      7. Inter-shape pause if more shapes remain

    Args:
        cursor_x, cursor_y: Current cursor position.
        shapes: List of shape dicts with x, y, width, height, color, label, etc.
        zones: List of zone dicts with x, y, width, height, color.
        rng: Numpy random generator.

    Returns:
        List of action dicts.
    """
    # Build zone lookup by color (tuple for hashability)
    zone_by_color: dict[tuple, dict] = {}
    for zone in zones:
        color_key = tuple(zone["color"]) if isinstance(zone["color"], (list, tuple)) else zone["color"]
        zone_by_color[color_key] = zone

    # Determine ordering: nearest-first from current cursor position
    remaining = [
        (i, s) for i, s in enumerate(shapes)
        if not s.get("dropped") and not s.get("complete")
    ]

    actions: list[dict] = []
    cx, cy = cursor_x, cursor_y

    while remaining:
        # Find nearest shape
        best_idx = 0
        best_dist = float("inf")
        for ri, (_, s) in enumerate(remaining):
            sx = s["x"] + s["width"] / 2
            sy = s["y"] + s["height"] / 2
            d = math.hypot(sx - cx, sy - cy)
            if d < best_dist:
                best_dist = d
                best_idx = ri

        _, shape = remaining.pop(best_idx)

        # Shape center
        sx = shape["x"] + shape["width"] / 2
        sy = shape["y"] + shape["height"] / 2

        # 1. Navigate to shape center
        move_actions = fitts_trajectory(
            cx, cy, sx, sy, shape["width"], rng, mouse_held=False
        )
        actions.extend(move_actions)
        for a in move_actions:
            cx = max(0, min(cx + a["dx"], 399))
            cy = max(0, min(cy + a["dy"], 399))

        # 2. Mouse down (grab)
        actions.append({
            "dx": 0.0,
            "dy": 0.0,
            "mouse_left": 1,
            "keys_held": [0] * NUM_KEYS,
        })

        # 3. Find matching zone and drag to its center
        color_key = tuple(shape["color"]) if isinstance(shape["color"], (list, tuple)) else shape["color"]
        zone = zone_by_color[color_key]
        zx = zone["x"] + zone["width"] / 2
        zy = zone["y"] + zone["height"] / 2

        drag_actions = fitts_trajectory(
            cx, cy, zx, zy, zone["width"], rng, mouse_held=True
        )
        actions.extend(drag_actions)
        for a in drag_actions:
            cx = max(0, min(cx + a["dx"], 399))
            cy = max(0, min(cy + a["dy"], 399))

        # 4. Mouse up (drop)
        actions.append({
            "dx": 0.0,
            "dy": 0.0,
            "mouse_left": 0,
            "keys_held": [0] * NUM_KEYS,
        })

        # 5. Pause before typing
        actions.extend(pause_actions(rng))

        # 6. Type label
        label = shape["label"]
        for ch in label:
            key_idx = char_to_key_index(ch)
            keys_held = [0] * NUM_KEYS
            keys_held[key_idx] = 1
            actions.append({
                "dx": 0.0,
                "dy": 0.0,
                "mouse_left": 0,
                "keys_held": keys_held,
            })
            # Release all keys
            actions.append(noop_action())

        # 7. Inter-shape pause if more remain
        if remaining:
            actions.extend(pause_actions(rng))

    return actions


def run_expert_episode(
    env,
    rng: np.random.Generator,
    seed: int | None = None,
) -> tuple[list[np.ndarray], list[dict], dict]:
    """Run a full expert episode for drag-and-label.

    Args:
        env: DragAndLabelEnv instance.
        rng: Numpy random generator for trajectory noise.
        seed: Optional seed for env.reset().

    Returns:
        (observations, actions, final_info)
    """
    env.reset(seed=seed)
    trajectory = generate_trajectory(
        *env.cursor_pos, env.shapes, env.zones, rng
    )
    return run_episode(env, trajectory)
