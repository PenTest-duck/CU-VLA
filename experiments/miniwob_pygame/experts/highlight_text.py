"""Expert policy for the highlight-text task.

Navigates to the start of the target word, presses mouse,
drags to the end of the word, then releases.
"""

from __future__ import annotations

import numpy as np

from ..config import NUM_KEYS
from ..widgets import TextBlock
from .common import fitts_trajectory, run_episode


def generate_trajectory(
    cursor_x: float,
    cursor_y: float,
    text_block: TextBlock,
    target_word: str,
    rng: np.random.Generator,
) -> list[dict]:
    """Generate expert trajectory for highlight-text task.

    Steps:
      1. Get word bounds in the text
      2. Navigate to the start character position
      3. mouse_left=1 (begin highlight)
      4. Drag to the end character position
      5. mouse_left=0 (finish highlight)

    Args:
        cursor_x, cursor_y: Current cursor position.
        text_block: TextBlock widget (must have been rendered so _char_rects exist).
        target_word: The word to highlight.
        rng: Numpy random generator.

    Returns:
        List of action dicts.
    """
    actions: list[dict] = []

    # 1. Get word bounds
    bounds = text_block.word_bounds(target_word)
    assert bounds is not None, f"Word '{target_word}' not found in text"
    start_idx, end_idx = bounds

    # 2. Get pixel position of start character
    # _char_rects stores absolute positions (includes text_block.x/y offsets)
    assert len(text_block._char_rects) > 0, "TextBlock must be rendered first"
    start_rect = text_block._char_rects[start_idx]
    start_x = start_rect[0] + start_rect[2] * 0.3  # slightly into the char
    char_y = start_rect[1] + start_rect[3] * 0.5  # vertical center

    # 3. Navigate to start position
    move_actions = fitts_trajectory(
        cursor_x, cursor_y, start_x, char_y,
        max(start_rect[2], 5), rng, mouse_held=False,
    )
    actions.extend(move_actions)

    # Update simulated cursor position
    from .common import simulate_cursor
    cur_x, cur_y = simulate_cursor(actions, cursor_x, cursor_y)

    # 4. Mouse down (begin highlight)
    actions.append({
        "dx": 0.0,
        "dy": 0.0,
        "mouse_left": 1,
        "keys_held": [0] * NUM_KEYS,
    })

    # 5. Get pixel position of end character (last char of the word)
    last_char_idx = end_idx - 1
    end_rect = text_block._char_rects[last_char_idx]
    end_x = end_rect[0] + end_rect[2] * 0.7  # past center of last char
    end_y = end_rect[1] + end_rect[3] * 0.5

    # 6. Drag to end position
    drag_actions = fitts_trajectory(
        cur_x, cur_y, end_x, end_y,
        max(end_rect[2], 5), rng, mouse_held=True,
    )
    # Dampen vertical drift to stay on the same line
    for a in drag_actions:
        a["dy"] = a["dy"] * 0.1
    actions.extend(drag_actions)

    # 7. Mouse up (finish highlight)
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
    """Run a full expert episode for highlight-text.

    Args:
        env: HighlightTextEnv instance.
        rng: Numpy random generator for trajectory noise.
        seed: Optional seed for env.reset().

    Returns:
        (observations, actions, final_info)
    """
    env.reset(seed=seed)
    trajectory = generate_trajectory(
        *env.cursor_pos, env._text_block, env._target_word, rng
    )
    return run_episode(env, trajectory)
