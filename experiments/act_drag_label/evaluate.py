"""Evaluation script: run agents in the drag-and-label environment and report metrics.

Compares: expert, baseline CNN, ACT (with temporal ensemble), and random agent.
Reports: sequence completion rate, shapes dropped/typed/completed, loop latency.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.act_drag_label.config import (
    ACTION, CHUNK, ENV, EVAL_CFG,
)

from experiments.act_drag_label.env import DragLabelEnv
from experiments.act_drag_label.model import ACT
from experiments.act_drag_label.baseline_cnn import BaselineCNN


# ---------------------------------------------------------------------------
# Proprioception helper
# ---------------------------------------------------------------------------

def build_proprio(env: DragLabelEnv) -> np.ndarray:
    """Construct proprioception vector from environment state.

    Layout: [cx/400, cy/400, click_state, key_one_hot(28)]  -> 31-dim
    """
    cx, cy = env.cursor_pos
    click = env._click_state
    key = env._current_key
    key_oh = np.zeros(28, dtype=np.float32)
    key_oh[key] = 1.0
    return np.concatenate([
        [cx / ENV.window_size, cy / ENV.window_size, float(click)],
        key_oh,
    ])


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
    def act(self, obs: np.ndarray, proprio: np.ndarray) -> dict:
        # Prepare tensors
        x = (
            torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        ).to(self.device)
        p = torch.from_numpy(proprio).unsqueeze(0).float().to(self.device)

        # Forward (inference: no actions, z=0)
        out = self.model(x, p, actions=None)

        # Extract new chunk as numpy arrays
        dx_chunk = out["dx"][0].cpu().numpy()          # (chunk_size,)
        dy_chunk = out["dy"][0].cpu().numpy()          # (chunk_size,)
        click_chunk = torch.sigmoid(out["click"][0]).cpu().numpy()  # (chunk_size,)
        key_chunk = out["key_logits"][0].cpu().numpy()  # (chunk_size, 28)

        new_chunk = {
            "dx": dx_chunk,
            "dy": dy_chunk,
            "click": click_chunk,
            "key_logits": key_chunk,
            "age": 0,
        }
        self.active_chunks.append(new_chunk)

        # Temporal ensemble: blend active chunks for current step
        # w_i = exp(-0.01 * age_i)
        total_w = 0.0
        blended_dx = 0.0
        blended_dy = 0.0
        blended_click = 0.0
        blended_key = np.zeros(ACTION.num_key_classes, dtype=np.float32)

        for chunk in self.active_chunks:
            age = chunk["age"]
            if age >= self.chunk_size:
                continue
            w = np.exp(-CHUNK.ensemble_decay * age)
            total_w += w
            blended_dx += w * chunk["dx"][age]
            blended_dy += w * chunk["dy"][age]
            blended_click += w * chunk["click"][age]
            blended_key += w * chunk["key_logits"][age]

        if total_w > 0:
            blended_dx /= total_w
            blended_dy /= total_w
            blended_click /= total_w
            blended_key /= total_w

        click_out = 1 if blended_click > 0.5 else 0
        key_out = int(np.argmax(blended_key))

        # Increment ages and prune expired chunks
        for chunk in self.active_chunks:
            chunk["age"] += 1
        self.active_chunks = [
            c for c in self.active_chunks if c["age"] < self.chunk_size
        ]

        return {
            "dx": float(blended_dx),
            "dy": float(blended_dy),
            "click": click_out,
            "key": key_out,
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
    def act(self, obs: np.ndarray, proprio: np.ndarray) -> dict:
        x = (
            torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        ).to(self.device)

        dx, dy, click_logit, key_logits = self.model(x)

        return {
            "dx": float(dx.item()),
            "dy": float(dy.item()),
            "click": 1 if float(torch.sigmoid(click_logit).item()) > 0.5 else 0,
            "key": int(key_logits.argmax(1).item()),
        }


class ExpertAgent:
    """Wraps the Fitts's Law expert for use in the loop."""

    def __init__(self) -> None:
        self._trajectory: list[dict] = []
        self._step = 0
        self._env_ref: DragLabelEnv | None = None

    def bind_env(self, env: DragLabelEnv) -> None:
        self._env_ref = env

    def reset(self) -> None:
        from experiments.act_drag_label.expert import generate_trajectory

        e = self._env_ref
        self._trajectory = generate_trajectory(
            *e.cursor_pos,
            e.shapes,
            e.zones,
            rng=np.random.default_rng(),
        )
        self._step = 0

    def act(self, obs: np.ndarray, proprio: np.ndarray) -> dict:
        if self._step < len(self._trajectory):
            a = self._trajectory[self._step]
            self._step += 1
            return a
        return {"dx": 0.0, "dy": 0.0, "click": 0, "key": 0}


class RandomAgent:
    """Random baseline: random deltas, random click, random key."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng or np.random.default_rng()

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, proprio: np.ndarray) -> dict:
        return {
            "dx": float(self._rng.uniform(-10, 10)),
            "dy": float(self._rng.uniform(-10, 10)),
            "click": int(self._rng.integers(0, 2)),
            "key": int(self._rng.integers(0, 28)),
        }


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

def run_agent(
    agent,
    env: DragLabelEnv,
    num_episodes: int,
    max_steps: int,
) -> dict:
    """Run agent in environment and collect per-episode metrics.

    Returns dict with lists of per-episode metrics.
    """
    results = {
        "total_steps": [],
        "shapes_completed": [],
        "shapes_dropped": [],
        "success": [],
        "loop_times_ms": [],
    }

    from tqdm import tqdm
    for ep in tqdm(range(num_episodes), desc="  Episodes", leave=False):
        obs = env.reset(seed=ep + 10000)

        if hasattr(agent, "bind_env"):
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
                results["shapes_completed"].append(
                    info.get("shapes_completed", 0)
                )
                results["shapes_dropped"].append(
                    info.get("shapes_dropped", 0)
                )
                results["success"].append(info.get("success", False))
                results["loop_times_ms"].extend(step_times)
                break

        if not done:
            results["total_steps"].append(max_steps)
            results["shapes_completed"].append(0)
            results["shapes_dropped"].append(0)
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
    shapes_completed = np.array(results["shapes_completed"])
    shapes_dropped = np.array(results["shapes_dropped"])
    loop_ms = np.array(results["loop_times_ms"])

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Episodes:                {total_eps}")
    print(
        f"  Sequence completion:     {successes}/{total_eps}"
        f" ({completion_rate:.1f}%)"
    )
    print()
    print(
        f"  Shapes dropped (mean):   {shapes_dropped.mean():.2f}"
        f"  / completed: {shapes_completed.mean():.2f}"
    )
    print(
        f"  Mean steps:              {total_steps.mean():.1f}"
        f"  p50={np.median(total_steps):.0f}"
        f"  p95={np.percentile(total_steps, 95):.0f}"
    )

    if len(loop_ms) > 0:
        print()
        print(
            f"  Loop latency:            "
            f"p50={np.percentile(loop_ms, 50):.2f}ms  "
            f"p95={np.percentile(loop_ms, 95):.2f}ms  "
            f"p99={np.percentile(loop_ms, 99):.2f}ms"
        )
        p95 = np.percentile(loop_ms, 95)
        if p95 > 0:
            print(f"  Effective Hz (p95):      {1000.0 / p95:.0f} Hz")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_checkpoint(
    local_path: str,
    hf_repo: str | None,
    path_in_repo: str,
) -> str | None:
    """Return a local checkpoint path, downloading from HF Hub if needed.

    Downloads if: local file doesn't exist, or HF Hub has a newer version.
    Returns None if the checkpoint isn't available locally or on Hub.
    """
    if hf_repo is None:
        return local_path if os.path.exists(local_path) else None

    from huggingface_hub import hf_hub_download
    try:
        downloaded = hf_hub_download(
            hf_repo, path_in_repo, repo_type="model",
            local_dir=os.path.dirname(local_path),
        )
        print(f"  Checkpoint synced from {hf_repo}/{path_in_repo}", flush=True)
        return downloaded
    except Exception:
        return local_path if os.path.exists(local_path) else None


def main(
    backbone: str = "resnet18",
    chunk_size: int = CHUNK.default_chunk_size,
    checkpoint: str | None = None,
    num_shapes: int = 1,
    num_episodes: int = EVAL_CFG.num_episodes,
    visual: bool = False,
    device: str = "cpu",
    hf_checkpoint_repo: str | None = None,
) -> None:
    base = os.path.dirname(__file__)
    max_steps = (
        EVAL_CFG.max_steps_multi if num_shapes > 1
        else EVAL_CFG.max_steps_per_episode
    )

    env = DragLabelEnv(
        visual=visual,
        fps=ENV.control_hz if visual else 0,
        num_shapes=num_shapes,
    )

    # --- Expert ---
    print("Running expert assessment...")
    expert = ExpertAgent()
    expert_results = run_agent(expert, env, num_episodes, max_steps)
    print_metrics(expert_results, "Expert (Fitts's Law)")

    # --- Baseline CNN ---
    baseline_default = os.path.join(base, "checkpoints", "baseline", "best.pt")
    baseline_ckpt = checkpoint or resolve_checkpoint(
        baseline_default,
        hf_checkpoint_repo,
        "baseline/best.pt",
    )
    if baseline_ckpt and os.path.exists(baseline_ckpt):
        print(f"\nRunning BaselineCNN assessment ({baseline_ckpt})...")
        baseline_agent = BaselineCNNAgent(baseline_ckpt, device=device)
        baseline_results = run_agent(baseline_agent, env, num_episodes, max_steps)
        print_metrics(baseline_results, "BaselineCNN")
    else:
        print(f"\nNo BaselineCNN checkpoint found. Skipping.")

    # --- ACT ---
    act_default = os.path.join(
        base, "checkpoints", f"{backbone}_chunk{chunk_size}", "best.pt"
    )
    act_ckpt = checkpoint or resolve_checkpoint(
        act_default,
        hf_checkpoint_repo,
        f"{backbone}_chunk{chunk_size}/best.pt",
    )
    if act_ckpt and os.path.exists(act_ckpt):
        print(f"\nRunning ACT assessment ({act_ckpt})...")
        act_agent = ACTAgent(
            act_ckpt,
            backbone=backbone,
            chunk_size=chunk_size,
            device=device,
        )
        act_results = run_agent(act_agent, env, num_episodes, max_steps)
        print_metrics(act_results, f"ACT (backbone={backbone}, chunk={chunk_size})")

        # --- Comparison summary ---
        expert_success = (
            sum(expert_results["success"])
            / len(expert_results["success"])
            * 100
        )
        act_success = (
            sum(act_results["success"])
            / len(act_results["success"])
            * 100
        )

        print(f"\n{'='*60}")
        print(f"  SUMMARY: ACT vs Expert")
        print(f"{'='*60}")
        print(f"  Expert completion:  {expert_success:.1f}%")
        print(f"  ACT completion:     {act_success:.1f}%")
        print(
            f"  Expert mean steps:  {np.mean(expert_results['total_steps']):.1f}"
        )
        print(
            f"  ACT mean steps:     {np.mean(act_results['total_steps']):.1f}"
        )

        act_loop_p95 = np.percentile(act_results["loop_times_ms"], 95)
        hz_pass = act_loop_p95 < 33.3
        print()
        print(
            f"  30Hz target:        "
            f"{'PASS' if hz_pass else 'FAIL'} "
            f"(p95={act_loop_p95:.2f}ms < 33.3ms)"
        )
    else:
        print(f"\nNo ACT checkpoint found. Skipping.")

    # --- Random ---
    print("\nRunning random baseline assessment...")
    random_agent = RandomAgent()
    random_results = run_agent(random_agent, env, num_episodes, max_steps)
    print_metrics(random_results, "Random")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assess agents on drag-and-label task"
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
        help="Path to model checkpoint (auto-detects baseline vs ACT)"
    )
    parser.add_argument(
        "--num-shapes", type=int, default=1,
        help="Number of shapes per episode"
    )
    parser.add_argument(
        "-n", "--num-episodes", type=int, default=EVAL_CFG.num_episodes,
        help="Number of assessment episodes"
    )
    parser.add_argument(
        "--visual", action="store_true",
        help="Show Pygame window during assessment"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device (cpu, cuda, mps)"
    )
    parser.add_argument(
        "--hf-checkpoint-repo", type=str, default="PenTest-duck/cu-vla-checkpoints",
        help="HF model repo to auto-download checkpoints from"
    )
    args = parser.parse_args()

    main(
        backbone=args.backbone,
        chunk_size=args.chunk_size,
        checkpoint=args.checkpoint,
        num_shapes=args.num_shapes,
        num_episodes=args.num_episodes,
        visual=args.visual,
        device=args.device,
        hf_checkpoint_repo=args.hf_checkpoint_repo,
    )
