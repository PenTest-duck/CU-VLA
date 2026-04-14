"""Visualize expert demos and interactively control the text editor.

Usage:
    # Watch expert demos
    uv run python -m experiments.mini_editor.visualize expert -n 10
    uv run python -m experiments.mini_editor.visualize expert --operation select_delete --fps 15

    # Control the editor yourself with mouse and keyboard
    uv run python -m experiments.mini_editor.visualize interactive
    uv run python -m experiments.mini_editor.visualize interactive --text "Hello world."
"""

from __future__ import annotations

import argparse

import numpy as np


def expert_demo(
    num_episodes: int = 10,
    operation: str | None = None,
    seed: int = 42,
    fps: int = 30,
) -> None:
    """Watch expert episodes in a Pygame window."""
    from .corpus import extract_words, load_corpus, make_passage
    from .env import MiniEditorEnv
    from .expert import generate_episode_trajectory, run_episode
    from .instructions import generate_instruction

    corpus = load_corpus()
    print(f"Loaded corpus: {len(corpus)} sentences")

    env = MiniEditorEnv(visual=True, fps=fps)
    rng = np.random.default_rng(seed)

    successes = 0
    for i in range(num_episodes):
        passage = None
        for _ in range(100):
            passage = make_passage(corpus, rng)
            if passage is not None:
                break
        if passage is None:
            continue

        words = extract_words(passage)
        inst = generate_instruction(passage, words, rng)

        # Filter by operation if requested
        if operation is not None:
            for _ in range(50):
                if inst.operation == operation:
                    break
                inst = generate_instruction(passage, words, rng)
            else:
                continue

        print(f"Episode {i}: [{inst.operation}] {inst.instruction_text}")

        env.reset(passage, seed=int(rng.integers(0, 2**31)))
        env.set_expected_text(inst.expected_text)
        traj = generate_episode_trajectory(env, inst, rng)

        # Replay
        env.reset(passage, seed=int(rng.integers(0, 2**31)))
        env.set_expected_text(inst.expected_text)
        for action in traj:
            obs, done, info = env.step(action)
            if done:
                break

        success = env.text == inst.expected_text
        if success:
            successes += 1
        print(f"  {len(traj)} steps, success={success}")

    print(f"\nTotal: {successes}/{num_episodes} successes")
    env.close()


# ---------------------------------------------------------------------------
# Pygame key code → our 53-key index mapping
# ---------------------------------------------------------------------------

def _build_pygame_key_map() -> dict[int, int]:
    """Map pygame key constants to our 53-key indices."""
    import pygame
    from .config import (
        KEY_A, KEY_0, KEY_BACKTICK, KEY_MINUS, KEY_EQUALS,
        KEY_LBRACKET, KEY_RBRACKET, KEY_BACKSLASH, KEY_SEMICOLON,
        KEY_APOSTROPHE, KEY_COMMA, KEY_PERIOD, KEY_SLASH,
        KEY_LSHIFT, KEY_RSHIFT, KEY_SPACE, KEY_DELETE, KEY_RETURN, KEY_TAB,
    )

    m: dict[int, int] = {}

    # Letters A-Z
    for i in range(26):
        m[pygame.K_a + i] = KEY_A + i

    # Digits 0-9
    for i in range(10):
        m[pygame.K_0 + i] = KEY_0 + i

    # Symbol keys
    m[pygame.K_BACKQUOTE] = KEY_BACKTICK
    m[pygame.K_MINUS] = KEY_MINUS
    m[pygame.K_EQUALS] = KEY_EQUALS
    m[pygame.K_LEFTBRACKET] = KEY_LBRACKET
    m[pygame.K_RIGHTBRACKET] = KEY_RBRACKET
    m[pygame.K_BACKSLASH] = KEY_BACKSLASH
    m[pygame.K_SEMICOLON] = KEY_SEMICOLON
    m[pygame.K_QUOTE] = KEY_APOSTROPHE
    m[pygame.K_COMMA] = KEY_COMMA
    m[pygame.K_PERIOD] = KEY_PERIOD
    m[pygame.K_SLASH] = KEY_SLASH

    # Modifiers
    m[pygame.K_LSHIFT] = KEY_LSHIFT
    m[pygame.K_RSHIFT] = KEY_RSHIFT

    # Special
    m[pygame.K_SPACE] = KEY_SPACE
    m[pygame.K_BACKSPACE] = KEY_DELETE
    m[pygame.K_DELETE] = KEY_DELETE
    m[pygame.K_RETURN] = KEY_RETURN
    m[pygame.K_TAB] = KEY_TAB

    return m


def interactive(text: str | None = None, fps: int = 30) -> None:
    """Control the editor with your real mouse and keyboard.

    Mouse movement, clicks, shift+click, and typing all work.
    Press Escape to quit.
    """
    import pygame

    from .config import ENV, NUM_KEYS
    from .corpus import extract_words, load_corpus, make_passage
    from .env import MiniEditorEnv
    from .instructions import generate_instruction

    if text is None:
        corpus = load_corpus()
        rng = np.random.default_rng(42)
        passage = None
        for _ in range(100):
            passage = make_passage(corpus, rng)
            if passage is not None:
                break
        text = passage or "The quick brown fox jumps over the lazy dog."

    # Generate a random instruction for context
    words = extract_words(text)
    rng = np.random.default_rng(0)
    inst = generate_instruction(text, words, rng)

    env = MiniEditorEnv(visual=True, fps=fps)
    env.reset(text, seed=0)

    print(f"Text: {text!r}")
    print(f"Instruction: {inst.instruction_text}")
    print(f"Expected result: {inst.expected_text!r}")
    print()
    print("Controls: mouse to move, click to set cursor, shift+click to select,")
    print("          type to insert, Delete to delete. Escape to quit.")
    print()

    key_map = _build_pygame_key_map()
    clock = pygame.time.Clock()

    # Track held state to produce action dicts
    prev_mouse_pos = pygame.mouse.get_pos()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if not running:
            break

        # Build action from current pygame input state
        mouse_pos = pygame.mouse.get_pos()
        dx = float(mouse_pos[0] - prev_mouse_pos[0])
        dy = float(mouse_pos[1] - prev_mouse_pos[1])
        prev_mouse_pos = mouse_pos

        mouse_buttons = pygame.mouse.get_pressed()
        mouse_left = 1 if mouse_buttons[0] else 0

        keys_pressed = pygame.key.get_pressed()
        keys_held = [0] * NUM_KEYS
        for pg_key, our_idx in key_map.items():
            if keys_pressed[pg_key]:
                keys_held[our_idx] = 1

        action = {
            "dx": dx,
            "dy": dy,
            "mouse_left": mouse_left,
            "keys_held": keys_held,
        }

        obs, done, info = env.step(action)

        # Override env's cursor position with actual mouse position
        # so the rendered cursor tracks the real mouse
        env._cursor_x = float(mouse_pos[0])
        env._cursor_y = float(mouse_pos[1])

        if done:
            success = env.text == inst.expected_text
            print(f"Done! text={env.text!r}, success={success}")
            break

        clock.tick(fps)

    # Check result
    if env.text == inst.expected_text:
        print("SUCCESS — text matches expected result!")
    else:
        print(f"Current text: {env.text!r}")
        print(f"Expected:     {inst.expected_text!r}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize expert demos or interactively control the editor"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # Expert demo mode
    p_expert = sub.add_parser("expert", help="Watch expert demonstrations")
    p_expert.add_argument("-n", "--num-episodes", type=int, default=10)
    p_expert.add_argument(
        "--operation", type=str, default=None,
        choices=["click", "click_type", "select_delete", "replace"],
    )
    p_expert.add_argument("--seed", type=int, default=42)
    p_expert.add_argument("--fps", type=int, default=30)

    # Interactive mode
    p_inter = sub.add_parser("interactive", help="Control the editor yourself")
    p_inter.add_argument("--text", type=str, default=None, help="Starting text (random if omitted)")
    p_inter.add_argument("--fps", type=int, default=30)

    args = parser.parse_args()

    if args.mode == "expert":
        expert_demo(
            num_episodes=args.num_episodes,
            operation=args.operation,
            seed=args.seed,
            fps=args.fps,
        )
    elif args.mode == "interactive":
        interactive(text=args.text, fps=args.fps)
