"""Expert policy for the use-slider task.

Navigates to the slider handle, grabs it, drags horizontally
to the target value position, then releases.
"""

from __future__ import annotations

import numpy as np

from ..config import NUM_KEYS
from ..widgets import Slider
from .common import fitts_trajectory, run_episode


def generate_trajectory(
    cursor_x: float,
    cursor_y: float,
    slider: Slider,
    rng: np.random.Generator,
) -> list[dict]:
    """Generate expert trajectory for use-slider task.

    Steps:
      1. Navigate to slider handle center (fitts_trajectory)
      2. mouse_left=1 (grab handle)
      3. Drag horizontally to target_x (fitts_trajectory with mouse_held=True)
      4. mouse_left=0 (release)

    Args:
        cursor_x, cursor_y: Current cursor position.
        slider: Slider widget instance.
        rng: Numpy random generator.

    Returns:
        List of action dicts.
    """
    actions: list[dict] = []

    # Handle center position
    handle_cx = slider.handle_x
    track_cy = slider.y + slider.height / 2

    # 1. Navigate to handle center
    move_actions = fitts_trajectory(
        cursor_x, cursor_y, handle_cx, track_cy,
        slider.handle_width, rng, mouse_held=False,
    )
    actions.extend(move_actions)

    # 2. Mouse down (grab handle)
    actions.append({
        "dx": 0.0,
        "dy": 0.0,
        "mouse_left": 1,
        "keys_held": [0] * NUM_KEYS,
    })

    # 3. Compute target x from target_value
    assert slider.target_value is not None
    target_x = slider.x + (slider.target_value / 100.0) * slider.width

    # Drag horizontally to target_x, keeping y near handle center
    drag_actions = fitts_trajectory(
        handle_cx, track_cy, target_x, track_cy,
        slider.handle_width, rng, mouse_held=True,
    )
    # Constrain dy to keep drag near the track
    for a in drag_actions:
        a["dy"] = a["dy"] * 0.1  # dampen vertical drift
    actions.extend(drag_actions)

    # 4. Mouse up (release)
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
    """Run a full expert episode for use-slider.

    Args:
        env: UseSliderEnv instance.
        rng: Numpy random generator for trajectory noise.
        seed: Optional seed for env.reset().

    Returns:
        (observations, actions, final_info)
    """
    env.reset(seed=seed)
    trajectory = generate_trajectory(
        *env.cursor_pos, env._slider, rng
    )
    return run_episode(env, trajectory)
