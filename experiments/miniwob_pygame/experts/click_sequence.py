"""Expert policy for the click-sequence task.

Navigates to each button in the target order using Fitts's Law trajectory,
then presses and releases mouse to register a click.
"""

from __future__ import annotations

import numpy as np

from ..config import ENV, NUM_KEYS
from .common import fitts_trajectory, pause_actions, run_episode


def generate_trajectory(
    cursor_x: float,
    cursor_y: float,
    buttons: list[dict],
    target_order: list[int],
    rng: np.random.Generator,
) -> list[dict]:
    """Generate expert trajectory for click-sequence task.

    For each number in target_order:
      1. Find the button with that number
      2. Navigate to its center (fitts_trajectory)
      3. mouse_left=1 (press)
      4. mouse_left=0 (release)
      5. Short pause

    Args:
        cursor_x, cursor_y: Current cursor position.
        buttons: List of button dicts with x, y, width, height, number, clicked.
        target_order: List of button numbers in the order to click.
        rng: Numpy random generator.

    Returns:
        List of action dicts.
    """
    # Build lookup by number
    btn_by_number: dict[int, dict] = {}
    for btn in buttons:
        btn_by_number[btn["number"]] = btn

    actions: list[dict] = []
    cx, cy = cursor_x, cursor_y
    ws = ENV.window_size

    for number in target_order:
        btn = btn_by_number[number]

        # Button center
        bx = btn["x"] + btn["width"] / 2
        by = btn["y"] + btn["height"] / 2

        # 1. Navigate to button center
        move_actions = fitts_trajectory(
            cx, cy, bx, by, btn["width"], rng, mouse_held=False
        )
        actions.extend(move_actions)
        # Update simulated cursor position
        for a in move_actions:
            cx = float(max(0, min(cx + a["dx"], ws - 1)))
            cy = float(max(0, min(cy + a["dy"], ws - 1)))

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

        # 4. Short pause
        actions.extend(pause_actions(rng))

    return actions


def run_expert_episode(
    env,
    rng: np.random.Generator,
    seed: int | None = None,
) -> tuple[list[np.ndarray], list[dict], dict]:
    """Run a full expert episode for click-sequence.

    Args:
        env: ClickSequenceEnv instance.
        rng: Numpy random generator for trajectory noise.
        seed: Optional seed for env.reset().

    Returns:
        (observations, actions, final_info)
    """
    env.reset(seed=seed)
    trajectory = generate_trajectory(
        *env.cursor_pos, env._buttons, env._target_order, rng
    )
    return run_episode(env, trajectory)
