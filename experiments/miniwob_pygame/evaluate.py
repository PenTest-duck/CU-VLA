"""Unified evaluation: run agents across MiniWoB-Pygame tasks and report metrics.

Compares: expert, ACT (with temporal ensemble), baseline CNN, and random agent.
Reports per-task and aggregate: success rate, mean steps, p50/p95, loop latency, Hz.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.miniwob_pygame.config import (
    ACTION, CHUNK, ENV, EVAL_CFG, NUM_KEYS, TASK_NAMES,
)
from experiments.miniwob_pygame.model import ACT
from experiments.miniwob_pygame.baseline_cnn import BaselineCNN
from experiments.miniwob_pygame.task_registry import get_env_class, get_expert_fn


# ---------------------------------------------------------------------------
# Proprioception helper
# ---------------------------------------------------------------------------

def build_proprio(env) -> np.ndarray:
    """Construct 46-dim proprio from env state.

    Layout: [cursor_x/399, cursor_y/399, mouse_left, keys_held_0..42]
    """
    cx, cy = env.cursor_pos
    mouse = float(env._mouse_pressed)
    keys = [float(k) for k in env._keys_held]
    return np.array(
        [cx / (ENV.window_size - 1), cy / (ENV.window_size - 1), mouse] + keys,
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Agent wrappers
# ---------------------------------------------------------------------------

class ACTAgent:
    """Wraps a trained ACT model with temporal ensemble for inference."""

    def __init__(
        self,
        checkpoint: str,
        backbone: str = "resnet18",
        chunk_size: int = CHUNK.default_chunk_size,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.chunk_size = chunk_size
        self.model = ACT(
            backbone_name=backbone,
            chunk_size=chunk_size,
        ).to(self.device)
        state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()
        self.active_chunks: list[dict] = []

    def reset(self) -> None:
        self.active_chunks = []

    @torch.no_grad()
    def act(self, obs: dict, proprio: np.ndarray) -> dict:
        # Prepare tensors
        screenshot = obs["screenshot"]
        x = (
            torch.from_numpy(screenshot).permute(2, 0, 1).unsqueeze(0).float()
            / 255.0
        ).to(self.device)
        p = torch.from_numpy(proprio).unsqueeze(0).float().to(self.device)

        # Forward (inference: no actions, z=0)
        out = self.model(x, p, actions=None)

        # Extract new chunk as numpy arrays
        dx_chunk = out["dx"][0].cpu().numpy()                         # (chunk_size,)
        dy_chunk = out["dy"][0].cpu().numpy()                         # (chunk_size,)
        mouse_chunk = torch.sigmoid(out["mouse_left"][0]).cpu().numpy()  # (chunk_size,)
        keys_chunk = torch.sigmoid(out["keys_held"][0]).cpu().numpy()    # (chunk_size, 43)

        new_chunk = {
            "dx": dx_chunk,
            "dy": dy_chunk,
            "mouse": mouse_chunk,
            "keys": keys_chunk,
            "age": 0,
        }
        self.active_chunks.append(new_chunk)

        # Temporal ensemble: blend active chunks for current step
        # w_i = exp(-decay * age_i)
        total_w = 0.0
        blended_dx = 0.0
        blended_dy = 0.0
        blended_mouse = 0.0
        blended_keys = np.zeros(ACTION.num_keys, dtype=np.float32)

        for chunk in self.active_chunks:
            age = chunk["age"]
            if age >= self.chunk_size:
                continue
            w = np.exp(-CHUNK.ensemble_decay * age)
            total_w += w
            blended_dx += w * chunk["dx"][age]
            blended_dy += w * chunk["dy"][age]
            blended_mouse += w * chunk["mouse"][age]
            blended_keys += w * chunk["keys"][age]

        if total_w > 0:
            blended_dx /= total_w
            blended_dy /= total_w
            blended_mouse /= total_w
            blended_keys /= total_w

        mouse_out = 1 if blended_mouse > 0.5 else 0
        keys_out = [1 if blended_keys[i] > 0.5 else 0 for i in range(ACTION.num_keys)]

        # Increment ages and prune expired chunks
        for chunk in self.active_chunks:
            chunk["age"] += 1
        self.active_chunks = [
            c for c in self.active_chunks if c["age"] < self.chunk_size
        ]

        return {
            "dx": float(blended_dx),
            "dy": float(blended_dy),
            "mouse_left": mouse_out,
            "keys_held": keys_out,
        }


class BaselineCNNAgent:
    """Wraps a trained BaselineCNN for single-step inference."""

    def __init__(self, checkpoint: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = BaselineCNN().to(self.device)
        state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def act(self, obs: dict, proprio: np.ndarray) -> dict:
        screenshot = obs["screenshot"]
        x = (
            torch.from_numpy(screenshot).permute(2, 0, 1).unsqueeze(0).float()
            / 255.0
        ).to(self.device)

        dx, dy, mouse_logit, key_logits = self.model(x)

        mouse_prob = float(torch.sigmoid(mouse_logit).item())
        key_probs = torch.sigmoid(key_logits).squeeze(0).cpu().numpy()

        return {
            "dx": float(dx.item()),
            "dy": float(dy.item()),
            "mouse_left": 1 if mouse_prob > 0.5 else 0,
            "keys_held": [1 if key_probs[i] > 0.5 else 0 for i in range(ACTION.num_keys)],
        }


class ExpertAgent:
    """Wraps a task-specific expert for use in the assessment loop.

    The expert's run_expert_episode function drives the env directly,
    so this agent operates differently: it runs the full episode in
    _generate_trajectory() and replays the recorded actions step by step.
    """

    def __init__(self, task_name: str) -> None:
        self._task_name = task_name
        self._expert_fn = get_expert_fn(task_name)
        self._trajectory: list[dict] = []
        self._step = 0
        self._env_ref = None

    def bind_env(self, env) -> None:
        self._env_ref = env

    def reset(self) -> None:
        self._trajectory = []
        self._step = 0

    def _generate_trajectory(self, seed: int) -> None:
        """Run expert on a fresh episode to capture the trajectory."""
        rng = np.random.default_rng(seed + 99999)
        _, actions, _ = self._expert_fn(self._env_ref, rng, seed=seed)
        self._trajectory = actions
        self._step = 0

    def act(self, obs: dict, proprio: np.ndarray) -> dict:
        if self._step < len(self._trajectory):
            a = self._trajectory[self._step]
            self._step += 1
            return a
        return {
            "dx": 0.0,
            "dy": 0.0,
            "mouse_left": 0,
            "keys_held": [0] * NUM_KEYS,
        }


class RandomAgent:
    """Random baseline: random deltas, random mouse, sparse random keys."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng or np.random.default_rng()

    def reset(self) -> None:
        pass

    def act(self, obs: dict, proprio: np.ndarray) -> dict:
        return {
            "dx": float(self._rng.uniform(-10, 10)),
            "dy": float(self._rng.uniform(-10, 10)),
            "mouse_left": int(self._rng.integers(0, 2)),
            "keys_held": [
                int(self._rng.random() < 0.05) for _ in range(NUM_KEYS)
            ],
        }


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

def run_agent(
    agent,
    env,
    num_episodes: int,
    max_steps: int,
) -> dict:
    """Run agent in environment and collect per-episode metrics.

    Returns dict with lists of per-episode metrics.
    """
    results = {
        "total_steps": [],
        "success": [],
        "loop_times_ms": [],
    }

    is_expert = isinstance(agent, ExpertAgent)

    from tqdm import tqdm
    for ep in tqdm(range(num_episodes), desc="  Episodes", leave=False):
        seed = ep + 10000

        if is_expert:
            # Expert needs to run its own episode to capture trajectory,
            # then we replay in a fresh episode with the same seed.
            agent.bind_env(env)
            agent._generate_trajectory(seed)

        obs = env.reset(seed=seed)

        if not is_expert and hasattr(agent, "bind_env"):
            agent.bind_env(env)
        if hasattr(agent, "reset"):
            agent.reset()

        proprio = build_proprio(env)
        step_times: list[float] = []
        done = False

        for step in range(max_steps):
            t_step = time.perf_counter()

            action = agent.act(obs, proprio)

            step_elapsed = time.perf_counter() - t_step
            step_times.append(step_elapsed * 1000)

            obs, done, info = env.step(action)
            proprio = build_proprio(env)

            if done:
                results["total_steps"].append(info.get("steps", step + 1))
                results["success"].append(info.get("success", False))
                results["loop_times_ms"].extend(step_times)
                break

        if not done:
            results["total_steps"].append(max_steps)
            results["success"].append(False)
            results["loop_times_ms"].extend(step_times)

    return results


# ---------------------------------------------------------------------------
# Metrics display
# ---------------------------------------------------------------------------

def print_metrics(results: dict, name: str) -> None:
    """Print formatted metrics block for one agent."""
    total_eps = len(results["success"])
    successes = sum(results["success"])
    completion_rate = successes / total_eps * 100 if total_eps > 0 else 0

    total_steps = np.array(results["total_steps"])
    loop_ms = np.array(results["loop_times_ms"])

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Episodes:            {total_eps}")
    print(
        f"  Success rate:        {successes}/{total_eps}"
        f" ({completion_rate:.1f}%)"
    )
    print(
        f"  Mean steps:          {total_steps.mean():.1f}"
        f"  p50={np.median(total_steps):.0f}"
        f"  p95={np.percentile(total_steps, 95):.0f}"
    )

    if len(loop_ms) > 0:
        print()
        print(
            f"  Loop latency:        "
            f"p50={np.percentile(loop_ms, 50):.2f}ms  "
            f"p95={np.percentile(loop_ms, 95):.2f}ms  "
            f"p99={np.percentile(loop_ms, 99):.2f}ms"
        )
        p95 = np.percentile(loop_ms, 95)
        if p95 > 0:
            print(f"  Effective Hz (p95):  {1000.0 / p95:.0f} Hz")


def print_aggregate(
    all_results: dict[str, dict[str, dict]],
    agent_names: list[str],
) -> None:
    """Print aggregate summary across all tasks for each agent."""
    print(f"\n{'#'*60}")
    print(f"  AGGREGATE SUMMARY ACROSS ALL TASKS")
    print(f"{'#'*60}")

    for name in agent_names:
        total_eps = 0
        total_successes = 0
        all_steps: list[int] = []
        all_latencies: list[float] = []

        for task_name, agents in all_results.items():
            if name not in agents:
                continue
            r = agents[name]
            total_eps += len(r["success"])
            total_successes += sum(r["success"])
            all_steps.extend(r["total_steps"])
            all_latencies.extend(r["loop_times_ms"])

        if total_eps == 0:
            continue

        rate = total_successes / total_eps * 100
        steps_arr = np.array(all_steps)

        print(f"\n  {name}:")
        print(f"    Episodes:      {total_eps}")
        print(f"    Success rate:  {total_successes}/{total_eps} ({rate:.1f}%)")
        print(
            f"    Mean steps:    {steps_arr.mean():.1f}"
            f"  p50={np.median(steps_arr):.0f}"
            f"  p95={np.percentile(steps_arr, 95):.0f}"
        )
        if all_latencies:
            lat = np.array(all_latencies)
            p95 = np.percentile(lat, 95)
            print(
                f"    Latency:       "
                f"p50={np.percentile(lat, 50):.2f}ms  "
                f"p95={p95:.2f}ms  "
                f"p99={np.percentile(lat, 99):.2f}ms"
            )
            if p95 > 0:
                hz_pass = p95 < 33.3
                print(
                    f"    30Hz target:   "
                    f"{'PASS' if hz_pass else 'FAIL'} "
                    f"(p95={p95:.2f}ms {'<' if hz_pass else '>='} 33.3ms)"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    tasks: list[str] | None = None,
    backbone: str = "resnet18",
    chunk_size: int = CHUNK.default_chunk_size,
    checkpoint: str | None = None,
    num_episodes: int = EVAL_CFG.num_episodes,
    visual: bool = False,
    device: str = "cpu",
) -> None:
    if tasks is None:
        tasks = list(TASK_NAMES)

    base = os.path.dirname(__file__)
    max_steps = EVAL_CFG.max_steps_per_episode

    # Track results across tasks for aggregate summary
    all_results: dict[str, dict[str, dict]] = {}
    agent_names_seen: list[str] = []

    for task_name in tasks:
        print(f"\n{'*'*60}")
        print(f"  TASK: {task_name}")
        print(f"{'*'*60}")

        EnvClass = get_env_class(task_name)
        env = EnvClass(visual=visual, fps=ENV.control_hz if visual else 0)
        all_results[task_name] = {}

        # --- Expert ---
        expert_name = "Expert"
        print(f"\nRunning expert on {task_name}...")
        expert = ExpertAgent(task_name)
        expert_results = run_agent(expert, env, num_episodes, max_steps)
        print_metrics(expert_results, f"Expert ({task_name})")
        all_results[task_name][expert_name] = expert_results
        if expert_name not in agent_names_seen:
            agent_names_seen.append(expert_name)

        # --- ACT ---
        act_ckpt = checkpoint or os.path.join(
            base, "checkpoints", "act_best.pt"
        )
        act_name = f"ACT (backbone={backbone}, chunk={chunk_size})"
        if os.path.exists(act_ckpt):
            print(f"\nRunning ACT on {task_name} ({act_ckpt})...")
            act_agent = ACTAgent(
                act_ckpt,
                backbone=backbone,
                chunk_size=chunk_size,
                device=device,
            )
            act_results = run_agent(act_agent, env, num_episodes, max_steps)
            print_metrics(act_results, f"ACT ({task_name})")
            all_results[task_name][act_name] = act_results
            if act_name not in agent_names_seen:
                agent_names_seen.append(act_name)
        else:
            print(f"\nNo ACT checkpoint at {act_ckpt}. Skipping.")

        # --- Baseline CNN ---
        baseline_ckpt = checkpoint or os.path.join(
            base, "checkpoints", "baseline_best.pt"
        )
        baseline_name = "BaselineCNN"
        if os.path.exists(baseline_ckpt) and checkpoint != act_ckpt:
            print(f"\nRunning BaselineCNN on {task_name} ({baseline_ckpt})...")
            baseline_agent = BaselineCNNAgent(baseline_ckpt, device=device)
            baseline_results = run_agent(baseline_agent, env, num_episodes, max_steps)
            print_metrics(baseline_results, f"BaselineCNN ({task_name})")
            all_results[task_name][baseline_name] = baseline_results
            if baseline_name not in agent_names_seen:
                agent_names_seen.append(baseline_name)
        elif not checkpoint:
            print(f"\nNo BaselineCNN checkpoint at {baseline_ckpt}. Skipping.")

        # --- Random ---
        random_name = "Random"
        print(f"\nRunning random baseline on {task_name}...")
        random_agent = RandomAgent()
        random_results = run_agent(random_agent, env, num_episodes, max_steps)
        print_metrics(random_results, f"Random ({task_name})")
        all_results[task_name][random_name] = random_results
        if random_name not in agent_names_seen:
            agent_names_seen.append(random_name)

        env.close()

    # Aggregate summary
    if len(tasks) > 1:
        print_aggregate(all_results, agent_names_seen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assess agents across MiniWoB-Pygame tasks"
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Task names to assess (default: all 12 tasks)"
    )
    parser.add_argument(
        "--backbone", type=str, default="resnet18",
        help="Vision backbone for ACT model"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK.default_chunk_size,
        help="Action chunk size for ACT model"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "-n", "--num-episodes", type=int, default=EVAL_CFG.num_episodes,
        help="Number of assessment episodes per task"
    )
    parser.add_argument(
        "--visual", action="store_true",
        help="Show Pygame window during assessment"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device (cpu, cuda, mps)"
    )
    args = parser.parse_args()

    main(
        tasks=args.tasks,
        backbone=args.backbone,
        chunk_size=args.chunk_size,
        checkpoint=args.checkpoint,
        num_episodes=args.num_episodes,
        visual=args.visual,
        device=args.device,
    )
