"""Replay recorded expert demonstrations live through the environment.

Loads actions from HDF5, replays them through DragLabelEnv with a visible
Pygame window. This shows the true environment state including the terminal
frame (which isn't saved in the data since it has no action pair).

Usage:
    uv run python experiments/act_drag_label/visualize_data.py              # replay all episodes
    uv run python experiments/act_drag_label/visualize_data.py -n 5         # first 5 episodes
    uv run python experiments/act_drag_label/visualize_data.py --episode 42 # specific episode
    uv run python experiments/act_drag_label/visualize_data.py --speed 0.5  # half speed

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

from experiments.act_drag_label.config import ENV
from experiments.act_drag_label.env import DragLabelEnv


def load_actions(path: str) -> dict:
    """Load actions and metadata from an HDF5 episode."""
    with h5py.File(path, "r") as f:
        return {
            "actions_dx": f["actions_dx"][:],
            "actions_dy": f["actions_dy"][:],
            "actions_click": f["actions_click"][:],
            "actions_key": f["actions_key"][:],
            "attrs": dict(f.attrs),
        }


def extract_seed(path: str) -> int:
    """Extract episode seed from filename (episode_NNNNN.hdf5 -> NNNNN)."""
    basename = os.path.basename(path)
    return int(basename.replace("episode_", "").replace(".hdf5", ""))


def render_hud(surface: pygame.Surface, font: _ft.Font,
               episode_idx: int, total: int, step: int, total_steps: int,
               ep_data: dict, paused: bool, done: bool) -> None:
    """Draw heads-up display overlay."""
    attrs = ep_data["attrs"]
    success = attrs.get("success", False)

    status = "PAUSED" if paused else ("DONE" if done else "PLAYING")
    lines = [
        f"Episode {episode_idx+1}/{total}  |  Step {step}/{total_steps}  |  {status}",
        f"Success: {'YES' if success else 'NO'}  |  Steps: {attrs.get('num_steps', '?')}",
    ]

    if step > 0 and step <= total_steps:
        idx = step - 1
        dx = ep_data["actions_dx"][idx]
        dy = ep_data["actions_dy"][idx]
        click = ep_data["actions_click"][idx]
        key = ep_data["actions_key"][idx]
        key_char = chr(ord('A') + key - 1) if 1 <= key <= 26 else "-"
        lines.append(f"dx={dx:+5.1f}  dy={dy:+5.1f}  click={click}  key={key_char}")

    y = 5
    for line in lines:
        text_surf, text_rect = font.render(line, (200, 200, 200), (0, 0, 0))
        surface.blit(text_surf, (5, y))
        y += text_rect.height + 2


def replay(
    data_dir: str | None = None,
    episode: int | None = None,
    num_episodes: int | None = None,
    speed: float = 1.0,
) -> None:
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "data")

    episode_files = sorted(glob.glob(os.path.join(data_dir, "**", "episode_*.hdf5"), recursive=True))
    if not episode_files:
        print(f"No episodes found in {data_dir}")
        return

    if episode is not None:
        pattern = f"episode_{episode:05d}.hdf5"
        matches = [f for f in episode_files if f.endswith(pattern)]
        if not matches:
            print(f"Episode {episode} not found")
            return
        episode_files = matches
    elif num_episodes is not None:
        episode_files = episode_files[:num_episodes]

    print(f"Loaded {len(episode_files)} episodes from {data_dir}")
    print("Controls: SPACE=pause, RIGHT=next, LEFT=restart, Q/ESC=quit")

    env = DragLabelEnv(visual=True, fps=int(ENV.control_hz * speed))
    _ft.init()
    font = _ft.Font(None, 13)

    ep_idx = 0
    running = True

    while running and ep_idx < len(episode_files):
        ep_data = load_actions(episode_files[ep_idx])
        seed = extract_seed(episode_files[ep_idx])
        total_steps = len(ep_data["actions_dx"])

        # Reset env with same seed used during generation
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
                        "click": int(ep_data["actions_click"][step]),
                        "key": int(ep_data["actions_key"][step]),
                    }
                    obs, done, info = env.step(action)
                    step += 1

                    # Draw HUD on top of env's rendered frame
                    render_hud(env._surface, font, ep_idx, len(episode_files),
                               step, total_steps, ep_data, paused, done)
                    pygame.display.flip()

                elif done or step >= total_steps:
                    # Show final state for a moment, then advance
                    render_hud(env._surface, font, ep_idx, len(episode_files),
                               step, total_steps, ep_data, paused, True)
                    pygame.display.flip()
                    pygame.time.wait(800)
                    break

                elif paused:
                    render_hud(env._surface, font, ep_idx, len(episode_files),
                               step, total_steps, ep_data, paused, done)
                    pygame.display.flip()
                    pygame.time.wait(33)

                continue
            # Break from event loop (RIGHT key) → next episode
            break

        ep_idx += 1

    env.close()
    print(f"Replayed {ep_idx} episodes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay expert demonstrations")
    parser.add_argument("-d", "--data-dir", type=str, default=None)
    parser.add_argument("--episode", type=int, default=None, help="Replay specific episode index")
    parser.add_argument("-n", "--num-episodes", type=int, default=None, help="Max episodes to replay")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (0.5=slow, 2=fast)")
    args = parser.parse_args()

    replay(
        data_dir=args.data_dir,
        episode=args.episode,
        num_episodes=args.num_episodes,
        speed=args.speed,
    )
