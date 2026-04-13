"""Expert policy for the scroll-and-click task.

Navigates to the scrollbar handle, drags it down to reveal the target
item, then navigates to the target item and clicks it.

Uses a two-phase approach: the scroll phase is executed first, then the
click phase is computed based on the *actual* scroll offset achieved,
making the expert robust to imprecise scrollbar dragging.
"""

from __future__ import annotations

import numpy as np

from ..config import NUM_KEYS
from ..widgets import ScrollableList
from .common import fitts_trajectory, pause_actions, run_episode, simulate_cursor


def _generate_scroll_phase(
    cursor_x: float,
    cursor_y: float,
    scroll_list: ScrollableList,
    target_index: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Generate actions to scroll the list so the target item is visible."""
    actions: list[dict] = []

    # Navigate to scrollbar handle center
    sb_x, sb_y, sb_w, sb_h = scroll_list.scrollbar_rect()
    handle_cx = sb_x + sb_w / 2
    handle_cy = sb_y + sb_h / 2

    move_actions = fitts_trajectory(
        cursor_x, cursor_y, handle_cx, handle_cy,
        sb_w, rng, mouse_held=False,
    )
    actions.extend(move_actions)
    cx, cy = simulate_cursor(move_actions, cursor_x, cursor_y)

    # Mouse down (grab scrollbar)
    actions.append({
        "dx": 0.0,
        "dy": 0.0,
        "mouse_left": 1,
        "keys_held": [0] * NUM_KEYS,
    })

    # Compute desired scroll position: center the target in viewport
    target_y_in_content = target_index * scroll_list.item_height
    desired_offset = target_y_in_content - scroll_list.height / 2
    desired_offset = max(0.0, min(scroll_list.max_scroll, desired_offset))

    # Convert desired_offset to scrollbar handle position
    _, _, _, handle_h = scroll_list.scrollbar_rect()
    scrollable_track = scroll_list.height - handle_h
    if scroll_list.max_scroll > 0 and scrollable_track > 0:
        target_frac = desired_offset / scroll_list.max_scroll
        target_handle_y = scroll_list.y + target_frac * scrollable_track + handle_h / 2
    else:
        target_handle_y = cy

    drag_actions = fitts_trajectory(
        cx, cy, handle_cx, target_handle_y,
        sb_w, rng, mouse_held=True,
    )
    # Dampen horizontal drift to stay on scrollbar
    for a in drag_actions:
        a["dx"] = a["dx"] * 0.1
    actions.extend(drag_actions)

    # Mouse up (release scrollbar)
    actions.append({
        "dx": 0.0,
        "dy": 0.0,
        "mouse_left": 0,
        "keys_held": [0] * NUM_KEYS,
    })

    # Pause
    actions.extend(pause_actions(rng))

    return actions


def _generate_click_phase(
    cursor_x: float,
    cursor_y: float,
    scroll_list: ScrollableList,
    target_index: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Generate actions to click on the target item at its current screen position.

    This must be called *after* the scroll phase has executed so that
    scroll_list.scroll_offset reflects the actual scroll position.
    """
    actions: list[dict] = []

    # Compute target item's current screen position using the actual scroll_offset
    item_screen_y = (
        scroll_list.y
        + target_index * scroll_list.item_height
        - scroll_list.scroll_offset
    )
    content_width = scroll_list.width - scroll_list.scrollbar_width
    item_cx = scroll_list.x + content_width / 2
    item_cy = item_screen_y + scroll_list.item_height / 2

    # Clamp to viewport bounds
    item_cy = max(
        float(scroll_list.y + scroll_list.item_height / 2),
        min(float(scroll_list.y + scroll_list.height - scroll_list.item_height / 2), item_cy),
    )

    # Navigate to target item center — use item_height as target_width so that
    # the Fitts's Law random offset stays within the item bounds
    move_actions = fitts_trajectory(
        cursor_x, cursor_y, item_cx, item_cy,
        scroll_list.item_height, rng, mouse_held=False,
    )
    actions.extend(move_actions)

    # Click (press then release)
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
    """Run a full expert episode for scroll-and-click.

    Two-phase execution:
      1. Execute scroll phase (navigate to scrollbar, drag, release)
      2. Read actual scroll_offset, then compute and execute click phase

    Args:
        env: ScrollAndClickEnv instance.
        rng: Numpy random generator for trajectory noise.
        seed: Optional seed for env.reset().

    Returns:
        (observations, actions, final_info)
    """
    env.reset(seed=seed)

    # Phase 1: Scroll
    scroll_actions = _generate_scroll_phase(
        *env.cursor_pos, env._scroll_list, env._target_index, rng
    )
    observations, actions, scroll_info = run_episode(env, scroll_actions)

    # If somehow done during scroll phase (shouldn't happen), return
    if scroll_info.get("success") or scroll_info.get("timeout"):
        return observations, actions, scroll_info

    # Phase 2: Click (uses actual scroll_offset from the widget)
    click_actions = _generate_click_phase(
        *env.cursor_pos, env._scroll_list, env._target_index, rng
    )

    # Continue episode
    for action in click_actions:
        if env._done:
            break
        obs = env._get_observation()
        observations.append(obs)
        obs, done, info = env.step(action)
        actions.append(action)
        if done:
            # Trim observations to match actions
            observations = observations[: len(actions)]
            return observations, actions, info

    # Trim observations
    observations = observations[: len(actions)]
    final_info = info if click_actions else scroll_info  # type: ignore[possibly-undefined]
    return observations, actions, final_info
