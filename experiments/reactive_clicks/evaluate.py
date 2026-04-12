"""Evaluation script: run agents in the environment and report decomposed metrics.

Compares: expert, deterministic baseline, and learned CNN policy.
Reports: onset latency, movement time, total RT, hit rate, loop latency.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.reactive_clicks.config import ENV, EVAL
from experiments.reactive_clicks.env import ReactiveClicksEnv, BTN_NONE, BTN_DOWN
from experiments.reactive_clicks.baseline import BaselineController
from experiments.reactive_clicks.model import TinyCNN


class CNNAgent:
    """Wraps a trained TinyCNN for inference in the environment."""

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = TinyCNN().to(self.device)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> dict:
        # (H, W, C) uint8 -> (1, C, H, W) float32
        x = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = x.to(self.device)

        dx, dy, btn_logits = self.model(x)

        return {
            "dx": float(dx.item()),
            "dy": float(dy.item()),
            "btn": int(btn_logits.argmax(1).item()),
        }


class ExpertAgent:
    """Wraps the Fitts's Law expert for use in the evaluation loop."""

    def __init__(self) -> None:
        self._trajectory: list[dict] = []
        self._step = 0
        self._env_ref: ReactiveClicksEnv | None = None

    def bind_env(self, env: ReactiveClicksEnv) -> None:
        self._env_ref = env

    def reset(self) -> None:
        from experiments.reactive_clicks.expert import generate_trajectory

        e = self._env_ref
        self._trajectory = generate_trajectory(
            *e.cursor_pos,
            *e.circle_pos,
            e.circle_radius,
            rng=np.random.default_rng(),
        )
        self._step = 0

    def act(self, obs: np.ndarray) -> dict:
        if self._step < len(self._trajectory):
            a = self._trajectory[self._step]
            self._step += 1
            return a
        return {"dx": 0.0, "dy": 0.0, "btn": BTN_DOWN}


def run_agent(
    agent,
    env: ReactiveClicksEnv,
    num_episodes: int,
    max_steps: int,
) -> dict:
    """Run agent in environment and collect decomposed metrics.

    Returns dict with lists of per-episode metrics.
    """
    results = {
        "onset_latency_steps": [],
        "movement_steps": [],
        "total_steps": [],
        "total_rt_s": [],
        "hit": [],
        "loop_times_ms": [],
    }

    for ep in range(num_episodes):
        obs = env.reset(seed=ep + 10000)

        if hasattr(agent, "reset"):
            agent.reset()

        first_move_step = None
        step_times: list[float] = []
        done = False

        for step in range(max_steps):
            t_step = time.perf_counter()

            action = agent.act(obs)

            step_elapsed = time.perf_counter() - t_step
            step_times.append(step_elapsed * 1000)

            # Track onset detection (non-trivial movement threshold: >0.5px)
            is_move = abs(action["dx"]) > 0.5 or abs(action["dy"]) > 0.5
            if first_move_step is None and is_move:
                first_move_step = step

            obs, done, info = env.step(action)

            if done:
                total_steps = info.get("steps", step + 1)
                results["total_steps"].append(total_steps)
                results["total_rt_s"].append(info.get("reaction_time_s", 0))
                results["hit"].append(info.get("hit", False))

                if first_move_step is not None:
                    results["onset_latency_steps"].append(first_move_step)
                    results["movement_steps"].append(total_steps - first_move_step)
                else:
                    results["onset_latency_steps"].append(total_steps)
                    results["movement_steps"].append(0)

                results["loop_times_ms"].extend(step_times)
                break

        if not done:
            results["total_steps"].append(max_steps)
            results["total_rt_s"].append(max_steps / ENV.control_hz)
            results["hit"].append(False)
            results["onset_latency_steps"].append(max_steps)
            results["movement_steps"].append(0)
            results["loop_times_ms"].extend(step_times)

    return results


def print_metrics(results: dict, name: str) -> None:
    """Print formatted metrics for one agent."""
    hits = sum(results["hit"])
    total = len(results["hit"])
    hit_rate = hits / total * 100 if total > 0 else 0

    total_steps = np.array(results["total_steps"])
    onset = np.array(results["onset_latency_steps"])
    movement = np.array(results["movement_steps"])
    loop_ms = np.array(results["loop_times_ms"])

    step_ms = 1000.0 / ENV.control_hz

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Episodes:          {total}")
    print(f"  Hit rate:          {hits}/{total} ({hit_rate:.1f}%)")
    print()
    print(
        f"  Total steps:       mean={total_steps.mean():.1f}"
        f"  p50={np.median(total_steps):.0f}"
        f"  p95={np.percentile(total_steps, 95):.0f}"
    )
    print(f"  Onset latency:     mean={onset.mean():.1f} steps ({onset.mean()*step_ms:.0f}ms)")
    print(f"  Movement time:     mean={movement.mean():.1f} steps ({movement.mean()*step_ms:.0f}ms)")
    print(f"  Total RT (steps):  mean={total_steps.mean():.1f} ({total_steps.mean()*step_ms:.0f}ms)")

    if len(loop_ms) > 0:
        print()
        print(
            f"  Loop latency:      p50={np.percentile(loop_ms, 50):.2f}ms"
            f"  p95={np.percentile(loop_ms, 95):.2f}ms"
            f"  p99={np.percentile(loop_ms, 99):.2f}ms"
        )
        p95 = np.percentile(loop_ms, 95)
        if p95 > 0:
            print(f"  Effective Hz (p95): {1000.0 / p95:.0f} Hz")


def main(
    checkpoint: str | None = None,
    num_episodes: int = EVAL.num_episodes,
    visual: bool = False,
    device: str = "cpu",
) -> None:
    base = os.path.dirname(__file__)
    if checkpoint is None:
        checkpoint = os.path.join(base, "checkpoints", "best.pt")

    env = ReactiveClicksEnv(visual=visual, fps=ENV.control_hz if visual else 0)
    max_steps = EVAL.max_steps_per_episode

    # --- Expert ---
    print("Running expert evaluation...")
    expert = ExpertAgent()
    expert.bind_env(env)
    expert_results = run_agent(expert, env, num_episodes, max_steps)
    print_metrics(expert_results, "Expert (Fitts's Law)")

    # --- Baseline ---
    print("\nRunning baseline evaluation...")
    baseline = BaselineController()
    baseline_results = run_agent(baseline, env, num_episodes, max_steps)
    print_metrics(baseline_results, "Baseline (threshold + centroid)")

    # --- CNN Agent ---
    if os.path.exists(checkpoint):
        print(f"\nRunning CNN evaluation (checkpoint: {checkpoint})...")
        cnn_agent = CNNAgent(checkpoint, device=device)
        cnn_results = run_agent(cnn_agent, env, num_episodes, max_steps)
        print_metrics(cnn_results, "CNN (TinyCNN)")

        # --- Comparison summary ---
        step_ms = 1000.0 / ENV.control_hz
        expert_rt = np.mean(expert_results["total_steps"]) * step_ms
        cnn_rt = np.mean(cnn_results["total_steps"]) * step_ms
        cnn_hit = sum(cnn_results["hit"]) / len(cnn_results["hit"]) * 100

        print(f"\n{'='*60}")
        print(f"  SUMMARY: CNN vs Expert")
        print(f"{'='*60}")
        print(f"  Expert mean RT:    {expert_rt:.0f}ms")
        print(f"  CNN mean RT:       {cnn_rt:.0f}ms  ({cnn_rt/expert_rt:.1f}x expert)")
        print(f"  CNN hit rate:      {cnn_hit:.1f}%")

        target_rt = expert_rt * 2
        target_hit = 90
        rt_pass = cnn_rt <= target_rt
        hit_pass = cnn_hit >= target_hit
        print()
        print(f"  RT target (<=2x):  {'PASS' if rt_pass else 'FAIL'} ({cnn_rt:.0f}ms <= {target_rt:.0f}ms)")
        print(f"  Hit target (>90%%): {'PASS' if hit_pass else 'FAIL'} ({cnn_hit:.1f}% >= {target_hit}%)")

        cnn_loop_p95 = np.percentile(cnn_results["loop_times_ms"], 95)
        hz_pass = cnn_loop_p95 < 33.3
        print(f"  30Hz target:       {'PASS' if hz_pass else 'FAIL'} (p95={cnn_loop_p95:.2f}ms < 33.3ms)")
    else:
        print(f"\nNo checkpoint found at {checkpoint}. Skipping CNN evaluation.")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agents on reactive clicks")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("-n", "--num-episodes", type=int, default=EVAL.num_episodes)
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(
        checkpoint=args.checkpoint,
        num_episodes=args.num_episodes,
        visual=args.visual,
        device=args.device,
    )
