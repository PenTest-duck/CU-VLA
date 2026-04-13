"""Expert policy for the copy-paste task.

Highlights source text, presses Ctrl+C, clicks target field,
presses Ctrl+V. Validates multi-key simultaneous input.
"""

from __future__ import annotations

import numpy as np

from ..config import KEY_C, KEY_CTRL, KEY_V, NUM_KEYS
from ..widgets import TextBlock, TextInput
from .common import (
    fitts_trajectory,
    noop_action,
    pause_actions,
    run_episode,
    simulate_cursor,
)


def generate_trajectory(
    cursor_x: float,
    cursor_y: float,
    text_block: TextBlock,
    text_input: TextInput,
    source_text: str,
    rng: np.random.Generator,
) -> list[dict]:
    """Generate expert trajectory for copy-paste task.

    Steps:
      1. Navigate to start of source text in TextBlock
      2. Mouse down (begin highlight)
      3. Drag to end of source text
      4. Mouse up (finish highlight)
      5. Pause
      6. Ctrl+C (both keys held for one step, then release)
      7. Pause
      8. Navigate to TextInput center
      9. Click to focus (mouse down, mouse up)
     10. Pause
     11. Ctrl+V (both keys held for one step, then release)

    Args:
        cursor_x, cursor_y: Current cursor position.
        text_block: TextBlock widget with source text (must have _char_rects).
        text_input: TextInput widget (target field).
        source_text: The text to copy.
        rng: Numpy random generator.

    Returns:
        List of action dicts.
    """
    actions: list[dict] = []

    # 1. Get word bounds in the text block
    bounds = text_block.word_bounds(source_text)
    assert bounds is not None, f"Source text '{source_text}' not found in TextBlock"
    start_idx, end_idx = bounds

    # Get pixel positions of start and end characters
    assert len(text_block._char_rects) > 0, "TextBlock must be rendered first"
    start_rect = text_block._char_rects[start_idx]
    start_x = start_rect[0] + start_rect[2] * 0.3  # slightly into the char
    char_y = start_rect[1] + start_rect[3] * 0.5  # vertical center

    # 2. Navigate to start of source text
    move_actions = fitts_trajectory(
        cursor_x, cursor_y, start_x, char_y,
        max(start_rect[2], 5), rng, mouse_held=False,
    )
    actions.extend(move_actions)
    cur_x, cur_y = simulate_cursor(actions, cursor_x, cursor_y)

    # 3. Mouse down (begin highlight)
    actions.append({
        "dx": 0.0, "dy": 0.0, "mouse_left": 1,
        "keys_held": [0] * NUM_KEYS,
    })

    # 4. Drag to end of source text
    last_char_idx = end_idx - 1
    end_rect = text_block._char_rects[last_char_idx]
    end_x = end_rect[0] + end_rect[2] * 0.7  # past center of last char
    end_y = end_rect[1] + end_rect[3] * 0.5

    drag_actions = fitts_trajectory(
        cur_x, cur_y, end_x, end_y,
        max(end_rect[2], 5), rng, mouse_held=True,
    )
    # Dampen vertical drift to stay on the same line
    for a in drag_actions:
        a["dy"] = a["dy"] * 0.1
    actions.extend(drag_actions)

    # 5. Mouse up (finish highlight)
    actions.append({
        "dx": 0.0, "dy": 0.0, "mouse_left": 0,
        "keys_held": [0] * NUM_KEYS,
    })
    cur_x, cur_y = simulate_cursor(actions, cursor_x, cursor_y)

    # 6. Pause before Ctrl+C
    actions.extend(pause_actions(rng))

    # 7. Ctrl+C: both KEY_CTRL and KEY_C held for one step
    keys_ctrl_c = [0] * NUM_KEYS
    keys_ctrl_c[KEY_CTRL] = 1
    keys_ctrl_c[KEY_C] = 1
    actions.append({
        "dx": 0.0, "dy": 0.0, "mouse_left": 0,
        "keys_held": keys_ctrl_c,
    })
    # Release all keys
    actions.append(noop_action())

    # 8. Pause before navigating to TextInput
    actions.extend(pause_actions(rng))

    # 9. Navigate to TextInput center
    cur_x, cur_y = simulate_cursor(actions, cursor_x, cursor_y)
    ti_cx = text_input.x + text_input.width / 2
    ti_cy = text_input.y + text_input.height / 2
    target_size = min(text_input.width, text_input.height)
    move_to_input = fitts_trajectory(
        cur_x, cur_y, ti_cx, ti_cy, target_size, rng, mouse_held=False,
    )
    actions.extend(move_to_input)

    # 10. Click to focus: mouse down, mouse up
    actions.append({
        "dx": 0.0, "dy": 0.0, "mouse_left": 1,
        "keys_held": [0] * NUM_KEYS,
    })
    actions.append({
        "dx": 0.0, "dy": 0.0, "mouse_left": 0,
        "keys_held": [0] * NUM_KEYS,
    })

    # 11. Pause before Ctrl+V
    actions.extend(pause_actions(rng))

    # 12. Ctrl+V: both KEY_CTRL and KEY_V held for one step
    keys_ctrl_v = [0] * NUM_KEYS
    keys_ctrl_v[KEY_CTRL] = 1
    keys_ctrl_v[KEY_V] = 1
    actions.append({
        "dx": 0.0, "dy": 0.0, "mouse_left": 0,
        "keys_held": keys_ctrl_v,
    })
    # Release all keys
    actions.append(noop_action())

    return actions


def run_expert_episode(
    env,
    rng: np.random.Generator,
    seed: int | None = None,
) -> tuple[list[np.ndarray], list[dict], dict]:
    """Run a full expert episode for copy-paste.

    Args:
        env: CopyPasteEnv instance.
        rng: Numpy random generator for trajectory noise.
        seed: Optional seed for env.reset().

    Returns:
        (observations, actions, final_info)
    """
    env.reset(seed=seed)
    trajectory = generate_trajectory(
        *env.cursor_pos,
        env._text_block,
        env._text_input,
        env._source_text,
        rng,
    )
    return run_episode(env, trajectory)
