"""Expert policy for the drag-sort task.

Uses selection sort: for each slot 0..n-1, find the card with the
correct value and drag it into place (swapping with whatever is there).
"""

from __future__ import annotations

import numpy as np

from ..config import ENV, NUM_KEYS
from .common import fitts_trajectory, pause_actions, run_episode


def generate_trajectory(
    cursor_x: float,
    cursor_y: float,
    cards: list[dict],
    slots: list[int],
    rng: np.random.Generator,
) -> list[dict]:
    """Generate expert trajectory for drag-sort task.

    Selection sort strategy:
      For each slot position 0..n-1:
        1. Find card with value (slot+1)
        2. If already in the right slot, skip
        3. Navigate to that card's center
        4. mouse_left=1 (grab)
        5. Drag to target slot center
        6. mouse_left=0 (drop — env will swap)
        7. Pause

    Args:
        cursor_x, cursor_y: Current cursor position.
        cards: List of card dicts with x, y, width, height, value, slot_index.
        slots: List of x-center positions for each slot.
        rng: Numpy random generator.

    Returns:
        List of action dicts.
    """
    # Build a mutable mapping: value -> card dict
    # (we track slot_index changes as we simulate swaps)
    card_slots = {card["value"]: card["slot_index"] for card in cards}
    card_w = cards[0]["width"] if cards else 70
    card_h = cards[0]["height"] if cards else 50

    # Compute standard y-center for cards
    ibh = ENV.instruction_bar_height
    ws = ENV.window_size
    standard_y = ibh + (ws - ibh - card_h) // 2 + card_h // 2

    actions: list[dict] = []
    cx, cy = cursor_x, cursor_y
    n = len(slots)

    for target_slot in range(n):
        target_value = target_slot + 1

        # Find which slot this value is currently in
        current_slot = card_slots[target_value]
        if current_slot == target_slot:
            continue  # Already in place

        # Card center position (based on current slot)
        card_cx = float(slots[current_slot])
        card_cy = float(standard_y)

        # 1. Navigate to the card
        move_actions = fitts_trajectory(
            cx, cy, card_cx, card_cy, card_w, rng, mouse_held=False
        )
        actions.extend(move_actions)
        for a in move_actions:
            cx = float(max(0, min(cx + a["dx"], ws - 1)))
            cy = float(max(0, min(cy + a["dy"], ws - 1)))

        # 2. Mouse down (grab)
        actions.append({
            "dx": 0.0,
            "dy": 0.0,
            "mouse_left": 1,
            "keys_held": [0] * NUM_KEYS,
        })

        # 3. Drag to target slot center
        target_cx = float(slots[target_slot])
        target_cy = float(standard_y)

        drag_actions = fitts_trajectory(
            cx, cy, target_cx, target_cy, card_w, rng, mouse_held=True
        )
        actions.extend(drag_actions)
        for a in drag_actions:
            cx = float(max(0, min(cx + a["dx"], ws - 1)))
            cy = float(max(0, min(cy + a["dy"], ws - 1)))

        # 4. Mouse up (drop — env swaps cards)
        actions.append({
            "dx": 0.0,
            "dy": 0.0,
            "mouse_left": 0,
            "keys_held": [0] * NUM_KEYS,
        })

        # Simulate the swap in our tracking
        # Find which value was in the target slot
        displaced_value = None
        for val, slot in card_slots.items():
            if slot == target_slot and val != target_value:
                displaced_value = val
                break

        if displaced_value is not None:
            card_slots[displaced_value] = current_slot
        card_slots[target_value] = target_slot

        # 5. Pause
        actions.extend(pause_actions(rng))

    return actions


def run_expert_episode(
    env,
    rng: np.random.Generator,
    seed: int | None = None,
) -> tuple[list[np.ndarray], list[dict], dict]:
    """Run a full expert episode for drag-sort.

    Args:
        env: DragSortEnv instance.
        rng: Numpy random generator for trajectory noise.
        seed: Optional seed for env.reset().

    Returns:
        (observations, actions, final_info)
    """
    env.reset(seed=seed)
    trajectory = generate_trajectory(
        *env.cursor_pos, env.cards, env.slots, rng
    )
    return run_episode(env, trajectory)
