"""Expert policy for the form-fill task.

Navigates to each text field, clicks to focus, types the target value,
then navigates to the submit button and clicks it.
"""

from __future__ import annotations

import numpy as np

from ..config import NUM_KEYS, char_to_key_index
from .common import fitts_trajectory, noop_action, pause_actions, run_episode, simulate_cursor


def generate_trajectory(
    cursor_x: float,
    cursor_y: float,
    fields: list,
    target_values: list[str],
    submit_rect: tuple[int, int, int, int],
    rng: np.random.Generator,
) -> list[dict]:
    """Generate expert trajectory for form-fill task.

    Steps per field:
      1. Navigate to field center (fitts_trajectory)
      2. Click to focus (mouse_left=1 then 0)
      3. Pause
      4. Type target value char by char

    After all fields:
      5. Navigate to submit button center
      6. Click (mouse_left=1 then 0)

    Args:
        cursor_x, cursor_y: Current cursor position.
        fields: List of TextInput widget instances.
        target_values: Target string for each field.
        submit_rect: (x, y, width, height) of the submit button.
        rng: Numpy random generator.

    Returns:
        List of action dicts.
    """
    actions: list[dict] = []
    cx, cy = cursor_x, cursor_y

    for field, target in zip(fields, target_values):
        # 1. Navigate to field center
        tx = field.x + field.width / 2
        ty = field.y + field.height / 2
        target_size = min(field.width, field.height)
        move_actions = fitts_trajectory(cx, cy, tx, ty, target_size, rng, mouse_held=False)
        actions.extend(move_actions)
        cx, cy = simulate_cursor(move_actions, cx, cy)

        # 2. Click to focus
        actions.append({
            "dx": 0.0,
            "dy": 0.0,
            "mouse_left": 1,
            "keys_held": [0] * NUM_KEYS,
        })
        actions.append({
            "dx": 0.0,
            "dy": 0.0,
            "mouse_left": 0,
            "keys_held": [0] * NUM_KEYS,
        })

        # 3. Pause before typing
        actions.extend(pause_actions(rng))

        # 4. Type each character
        for ch in target:
            key_idx = char_to_key_index(ch)
            keys_held = [0] * NUM_KEYS
            keys_held[key_idx] = 1
            actions.append({
                "dx": 0.0,
                "dy": 0.0,
                "mouse_left": 0,
                "keys_held": keys_held,
            })
            # Release
            actions.append(noop_action())

    # 5. Navigate to submit button center
    sx, sy, sw, sh = submit_rect
    btn_cx = sx + sw / 2
    btn_cy = sy + sh / 2
    btn_size = min(sw, sh)
    move_actions = fitts_trajectory(cx, cy, btn_cx, btn_cy, btn_size, rng, mouse_held=False)
    actions.extend(move_actions)

    # 6. Click submit
    actions.append({
        "dx": 0.0,
        "dy": 0.0,
        "mouse_left": 1,
        "keys_held": [0] * NUM_KEYS,
    })
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
    """Run a full expert episode for form-fill.

    Args:
        env: FormFillEnv instance.
        rng: Numpy random generator for trajectory noise.
        seed: Optional seed for env.reset().

    Returns:
        (observations, actions, final_info)
    """
    env.reset(seed=seed)
    trajectory = generate_trajectory(
        *env.cursor_pos, env._fields, env._target_values, env._submit_rect, rng
    )
    return run_episode(env, trajectory)
