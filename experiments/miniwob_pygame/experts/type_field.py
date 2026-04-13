"""Expert policy for the type-field task.

Navigates to the text input, clicks to focus, then types
each character of the target word using held-state key actions.
"""

from __future__ import annotations

import numpy as np

from ..config import NUM_KEYS, char_to_key_index
from .common import fitts_trajectory, noop_action, pause_actions, run_episode


def generate_trajectory(
    cursor_x: float,
    cursor_y: float,
    text_input,
    target_word: str,
    rng: np.random.Generator,
) -> list[dict]:
    """Generate expert trajectory for type-field task.

    Steps:
      1. Navigate to text input center using fitts_trajectory
      2. Click to focus (mouse_left=1 then mouse_left=0)
      3. Pause (2-5 noop steps)
      4. For each character: emit keys_held with that key=1, then noop (release)

    Args:
        cursor_x, cursor_y: Current cursor position.
        text_input: TextInput widget instance.
        target_word: The word to type.
        rng: Numpy random generator.

    Returns:
        List of action dicts.
    """
    actions: list[dict] = []

    # 1. Navigate to text input center
    tx = text_input.x + text_input.width / 2
    ty = text_input.y + text_input.height / 2
    # Use min(width, height) as target size so Fitts's noise doesn't
    # overshoot the narrow dimension of the text input.
    target_size = min(text_input.width, text_input.height)
    move_actions = fitts_trajectory(
        cursor_x, cursor_y, tx, ty, target_size, rng, mouse_held=False
    )
    actions.extend(move_actions)

    # 2. Click to focus: mouse down then mouse up
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
    for ch in target_word:
        key_idx = char_to_key_index(ch)

        # Key press: set the key bit to 1
        keys_held = [0] * NUM_KEYS
        keys_held[key_idx] = 1
        actions.append({
            "dx": 0.0,
            "dy": 0.0,
            "mouse_left": 0,
            "keys_held": keys_held,
        })

        # Key release: all keys 0
        actions.append(noop_action())

    return actions


def run_expert_episode(
    env,
    rng: np.random.Generator,
    seed: int | None = None,
) -> tuple[list[np.ndarray], list[dict], dict]:
    """Run a full expert episode for type-field.

    Args:
        env: TypeFieldEnv instance.
        rng: Numpy random generator for trajectory noise.
        seed: Optional seed for env.reset().

    Returns:
        (observations, actions, final_info)
    """
    env.reset(seed=seed)
    trajectory = generate_trajectory(
        *env.cursor_pos, env._text_input, env._target_word, rng
    )
    return run_episode(env, trajectory)
