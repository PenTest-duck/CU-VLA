"""Replay recorded expert demonstrations live through the environment.

Loads actions from HF datasets (parquet), replays them through DragLabelEnv
with a visible Pygame window. This shows the true environment state including
the terminal frame (which isn't saved in the data since it has no action pair).

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
import os
import sys

import numpy as np
import pygame
import pygame._freetype as _ft

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.act_drag_label.config import ENV
from experiments.act_drag_label.env import DragLabelEnv
from experiments.act_drag_label.train import build_episode_offsets


def load_actions(ds, start_row: int, length: int) -> dict:
    """Load actions and metadata for a single episode from the dataset."""
    ep_rows = ds.select(range(start_row, start_row + length))
    return {
        "actions_dx": np.array(ep_rows["action_dx"], dtype=np.float32),
        "actions_dy": np.array(ep_rows["action_dy"], dtype=np.float32),
        "actions_click": np.array(ep_rows["action_click"], dtype=np.int8),
        "actions_key": np.array(ep_rows["action_key"], dtype=np.int8),
        "attrs": {
            "num_shapes": ep_rows[0]["num_shapes"],
            "success": ep_rows[0]["success"],
            "num_steps": length,
            "shapes_completed": ep_rows[0]["shapes_completed"],
        },
    }


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
    hf_data_repo: str | None = None,
    episode: int | None = None,
    num_episodes: int | None = None,
    speed: float = 1.0,
) -> None:
    from datasets import load_dataset, load_from_disk

    if hf_data_repo:
        print(f"Loading dataset from {hf_data_repo} ...")
        ds = load_dataset(hf_data_repo, split="train")
    else:
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data")
        print(f"Loading dataset from {data_dir} ...")
        ds = load_from_disk(data_dir)

    episode_offsets = build_episode_offsets(np.array(ds["episode_id"], dtype=np.int32))
    all_episode_ids = sorted(episode_offsets.keys())

    if not all_episode_ids:
        print("No episodes found in dataset")
        return

    if episode is not None:
        if episode not in episode_offsets:
            print(f"Episode {episode} not found")
            return
        all_episode_ids = [episode]
    elif num_episodes is not None:
        all_episode_ids = all_episode_ids[:num_episodes]

    print(f"Loaded {len(all_episode_ids)} episodes")
    print("Controls: SPACE=pause, RIGHT=next, LEFT=restart, Q/ESC=quit")

    env = DragLabelEnv(visual=True, fps=int(ENV.control_hz * speed))
    _ft.init()
    font = _ft.Font(None, 13)

    ep_idx = 0
    running = True

    while running and ep_idx < len(all_episode_ids):
        eid = all_episode_ids[ep_idx]
        start_row, length = episode_offsets[eid]
        ep_data = load_actions(ds, start_row, length)
        seed = eid  # episode seed = episode_id in generate_data.py
        total_steps = length

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
                    render_hud(env._surface, font, ep_idx, len(all_episode_ids),
                               step, total_steps, ep_data, paused, done)
                    pygame.display.flip()

                elif done or step >= total_steps:
                    # Show final state for a moment, then advance
                    render_hud(env._surface, font, ep_idx, len(all_episode_ids),
                               step, total_steps, ep_data, paused, True)
                    pygame.display.flip()
                    pygame.time.wait(800)
                    break

                elif paused:
                    render_hud(env._surface, font, ep_idx, len(all_episode_ids),
                               step, total_steps, ep_data, paused, done)
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
    parser.add_argument("-d", "--data-dir", type=str, default=None)
    parser.add_argument("--hf-data-repo", type=str, default=None,
                        help="HF dataset repo (e.g. PenTest-duck/cu-vla-data)")
    parser.add_argument("--episode", type=int, default=None, help="Replay specific episode index")
    parser.add_argument("-n", "--num-episodes", type=int, default=None, help="Max episodes to replay")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (0.5=slow, 2=fast)")
    args = parser.parse_args()

    replay(
        data_dir=args.data_dir,
        hf_data_repo=args.hf_data_repo,
        episode=args.episode,
        num_episodes=args.num_episodes,
        speed=args.speed,
    )
