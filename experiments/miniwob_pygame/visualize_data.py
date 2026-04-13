"""Replay recorded expert demonstrations through task environments.

Loads actions from HDF5, replays through the actual Pygame env with
a visible window. Shows HUD overlay with episode info and current action.

Usage:
    uv run python experiments/miniwob_pygame/visualize_data.py --task click-target
    uv run python experiments/miniwob_pygame/visualize_data.py --task drag-to-zone -n 5
    uv run python experiments/miniwob_pygame/visualize_data.py --task type-field --episode 3
    uv run python experiments/miniwob_pygame/visualize_data.py --task use-slider --speed 0.5

Controls:
    SPACE  - pause/resume
    RIGHT  - next episode
    LEFT   - restart current episode
    Q/ESC  - quit
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np
import pygame
import pygame._freetype as _ft

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.miniwob_pygame.config import ENV, NUM_KEYS, key_index_to_char
from experiments.miniwob_pygame.task_registry import get_env_class


def load_episode(path: str) -> dict:
    """Load a single episode's actions and metadata from HDF5."""
    with h5py.File(path, "r") as f:
        return {
            "actions_dx": f["actions_dx"][:],
            "actions_dy": f["actions_dy"][:],
            "actions_mouse_left": f["actions_mouse_left"][:],
            "actions_keys_held": f["actions_keys_held"][:],
            "task_name": f.attrs["task_name"],
            "success": f.attrs.get("success", False),
            "num_steps": f.attrs.get("num_steps", len(f["actions_dx"])),
        }


def render_hud(
    surface: pygame.Surface,
    font: _ft.Font,
    ep_idx: int,
    total_eps: int,
    step: int,
    total_steps: int,
    ep_data: dict,
    paused: bool,
    done: bool,
) -> None:
    """Draw heads-up display overlay with episode and action info."""
    success = ep_data["success"]
    status = "PAUSED" if paused else ("DONE" if done else "PLAYING")

    lines = [
        f"Episode {ep_idx+1}/{total_eps}  |  Step {step}/{total_steps}  |  {status}",
        f"Task: {ep_data['task_name']}  |  Success: {'YES' if success else 'NO'}",
    ]

    if 0 < step <= total_steps:
        idx = step - 1
        dx = ep_data["actions_dx"][idx]
        dy = ep_data["actions_dy"][idx]
        mouse = ep_data["actions_mouse_left"][idx]
        keys = ep_data["actions_keys_held"][idx]

        # Show active keys
        active_keys: list[str] = []
        for i, v in enumerate(keys):
            if v:
                ch = key_index_to_char(i)
                if ch is not None:
                    active_keys.append(ch if ch != " " else "SPC")
                elif i == 27:
                    active_keys.append("ENT")
                elif i == 28:
                    active_keys.append("BS")
                elif i == 29:
                    active_keys.append("TAB")
                elif i == 40:
                    active_keys.append("CTRL")
                elif i == 41:
                    active_keys.append("SHIFT")
                elif i == 42:
                    active_keys.append("ALT")
        keys_str = "+".join(active_keys) if active_keys else "-"
        lines.append(
            f"dx={dx:+5.1f}  dy={dy:+5.1f}  mouse={'DOWN' if mouse else 'UP'}  keys={keys_str}"
        )

    y = 5
    for line in lines:
        text_surf, text_rect = font.render(line, (200, 200, 200), (0, 0, 0))
        surface.blit(text_surf, (5, y))
        y += text_rect.height + 2


def replay(
    task_name: str,
    data_dir: str | None = None,
    episode: int | None = None,
    num_episodes: int | None = None,
    speed: float = 1.0,
) -> None:
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "data")

    task_dir = os.path.join(data_dir, task_name)
    episode_files = sorted(
        glob.glob(os.path.join(task_dir, "**", "episode_*.hdf5"), recursive=True)
    )

    if not episode_files:
        print(f"No episodes found in {task_dir}")
        return

    if episode is not None:
        # Find specific episode
        target = f"episode_{episode:05d}.hdf5"
        matches = [f for f in episode_files if f.endswith(target)]
        if not matches:
            print(f"Episode {episode} not found")
            return
        episode_files = matches
    elif num_episodes is not None:
        episode_files = episode_files[:num_episodes]

    print(f"Loaded {len(episode_files)} episodes for task '{task_name}'")
    print("Controls: SPACE=pause, RIGHT=next, LEFT=restart, Q/ESC=quit")

    EnvClass = get_env_class(task_name)
    env = EnvClass(visual=True, fps=int(ENV.control_hz * speed))
    _ft.init()
    hud_font = _ft.Font(None, 13)

    ep_idx = 0
    running = True

    while running and ep_idx < len(episode_files):
        ep_data = load_episode(episode_files[ep_idx])
        total_steps = ep_data["num_steps"]

        # Extract episode number from filename for seed
        fname = os.path.basename(episode_files[ep_idx])
        seed = int(fname.replace("episode_", "").replace(".hdf5", ""))

        env.reset(seed=seed)
        step = 0
        paused = False
        done = False

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_RIGHT:
                        break  # next episode
                    elif event.key == pygame.K_LEFT:
                        # Restart: re-reset env
                        env.reset(seed=seed)
                        step = 0
                        done = False
            else:
                # No break from event loop
                if not paused and not done and step < total_steps:
                    action = {
                        "dx": float(ep_data["actions_dx"][step]),
                        "dy": float(ep_data["actions_dy"][step]),
                        "mouse_left": int(ep_data["actions_mouse_left"][step]),
                        "keys_held": list(ep_data["actions_keys_held"][step].astype(int)),
                    }
                    obs, done_env, info = env.step(action)
                    step += 1
                    done = done_env

                    # Draw HUD on top of env's rendered frame
                    render_hud(
                        env._surface, hud_font, ep_idx, len(episode_files),
                        step, total_steps, ep_data, paused, done,
                    )
                    pygame.display.flip()

                elif done or step >= total_steps:
                    # Show final state for a moment, then advance
                    render_hud(
                        env._surface, hud_font, ep_idx, len(episode_files),
                        step, total_steps, ep_data, paused, True,
                    )
                    pygame.display.flip()
                    pygame.time.wait(800)
                    break

                elif paused:
                    render_hud(
                        env._surface, hud_font, ep_idx, len(episode_files),
                        step, total_steps, ep_data, paused, done,
                    )
                    pygame.display.flip()
                    pygame.time.wait(33)

                continue
            # Break from event loop (RIGHT key) -> next episode
            break

        ep_idx += 1

    env.close()
    print(f"Replayed {ep_idx} episodes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay expert demonstrations")
    parser.add_argument("--task", type=str, required=True, help="Task name to replay")
    parser.add_argument("-d", "--data-dir", type=str, default=None)
    parser.add_argument(
        "--episode", type=int, default=None, help="Replay specific episode index"
    )
    parser.add_argument(
        "-n", "--num-episodes", type=int, default=None, help="Max episodes to replay"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Playback speed (0.5=slow, 2=fast)"
    )
    args = parser.parse_args()

    replay(
        task_name=args.task,
        data_dir=args.data_dir,
        episode=args.episode,
        num_episodes=args.num_episodes,
        speed=args.speed,
    )
