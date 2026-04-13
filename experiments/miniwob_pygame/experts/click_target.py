"""Expert policy for the click-target task.

Navigates to the target center using Fitts's Law trajectory,
then presses and releases mouse to register a click.
"""

from __future__ import annotations

import numpy as np

from ..config import NUM_KEYS
from .common import fitts_trajectory, run_episode


def generate_trajectory(
    cursor_x: float,
    cursor_y: float,
    target: dict,
    rng: np.random.Generator,
) -> list[dict]:
    """Generate expert trajectory for click-target task.

    Steps:
      1. Navigate to target center using fitts_trajectory
      2. Emit mouse_left=1 (press)
      3. Emit mouse_left=0 (release = click complete)

    Args:
        cursor_x, cursor_y: Current cursor position.
        target: Target dict with x, y, width, height, color.
        rng: Numpy random generator.

    Returns:
        List of action dicts.
    """
    # Target center
    tx = target["x"] + target["width"] / 2
    ty = target["y"] + target["height"] / 2

    actions: list[dict] = []

    # 1. Navigate to target
    move_actions = fitts_trajectory(
        cursor_x, cursor_y, tx, ty, target["width"], rng, mouse_held=False
    )
    actions.extend(move_actions)

    # 2. Mouse down (press)
    actions.append({
        "dx": 0.0,
        "dy": 0.0,
        "mouse_left": 1,
        "keys_held": [0] * NUM_KEYS,
    })

    # 3. Mouse up (release = click complete)
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
    """Run a full expert episode for click-target.

    Args:
        env: ClickTargetEnv instance.
        rng: Numpy random generator for trajectory noise.
        seed: Optional seed for env.reset().

    Returns:
        (observations, actions, final_info)
    """
    env.reset(seed=seed)
    trajectory = generate_trajectory(
        *env.cursor_pos, env._target, rng
    )
    return run_episode(env, trajectory)
