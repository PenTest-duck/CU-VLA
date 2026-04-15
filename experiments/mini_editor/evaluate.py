"""Evaluation pipeline for Experiment 5: Mini Text Editor.

Compares: expert, ACT (with probability-ensemble temporal smoothing),
and random agent. Reports overall and per-operation success rates,
step counts, and loop latency.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.mini_editor.config import (
    ACTION,
    BIN_CENTERS,
    CHUNK,
    ENV,
    EVAL_CFG,
    MODEL,
    NUM_BINS,
    NUM_KEYS,
    PROPRIO_DIM,
)
from experiments.mini_editor.corpus import extract_words, load_corpus, make_passage
from experiments.mini_editor.env import MiniEditorEnv
from experiments.mini_editor.expert import generate_episode_trajectory
from experiments.mini_editor.instructions import EditInstruction, generate_instruction
from experiments.mini_editor.model import ACT
from experiments.mini_editor.text_encoder import (
    build_text_encoder,
    tokenize_instruction,
)


# ---------------------------------------------------------------------------
# Proprioception helper
# ---------------------------------------------------------------------------


def build_proprio(env: MiniEditorEnv) -> np.ndarray:
    """Construct 56-dim proprio from env state.

    Layout: [cursor_x_norm, cursor_y_norm, mouse_left, keys_held_0..52]
    """
    cx, cy = env.cursor_pos
    proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
    proprio[0] = cx / max(ENV.window_w - 1, 1)
    proprio[1] = cy / max(ENV.window_h - 1, 1)
    proprio[2] = float(env._prev_mouse_left)
    for i in range(NUM_KEYS):
        proprio[3 + i] = float(env._prev_keys_held[i])
    return proprio


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------


def resolve_checkpoint(
    local_path: str,
    hf_repo: str | None,
    repo_prefix: str,
) -> str | None:
    """Return a local checkpoint path, downloading from HF Hub if needed.

    The training script uploads checkpoints to:
        {repo_prefix}/{run_id}/best.pt
    where run_id is a timestamp like 20260415-0830.  This function finds
    the latest run directory under repo_prefix and downloads best.pt.
    """
    if hf_repo is None:
        return local_path if os.path.exists(local_path) else None

    from huggingface_hub import HfApi, hf_hub_download

    try:
        api = HfApi()
        files = api.list_repo_files(hf_repo, repo_type="model")
        # Find all best.pt files under the prefix
        candidates = sorted(
            f for f in files
            if f.startswith(repo_prefix + "/") and f.endswith("/best.pt")
        )
        if not candidates:
            print(f"  No checkpoints found in {hf_repo}/{repo_prefix}/")
            return local_path if os.path.exists(local_path) else None

        path_in_repo = candidates[-1]  # latest run (sorted by timestamp)
        downloaded = hf_hub_download(
            hf_repo,
            path_in_repo,
            repo_type="model",
            local_dir=os.path.dirname(local_path),
        )
        print(f"  Checkpoint synced from {hf_repo}/{path_in_repo}", flush=True)
        return downloaded
    except Exception as e:
        print(f"  HF download failed: {e}")
        return local_path if os.path.exists(local_path) else None


# ---------------------------------------------------------------------------
# Agent wrappers
# ---------------------------------------------------------------------------


class ACTAgent:
    """Wraps a trained ACT model with probability-ensemble temporal smoothing.

    Handles text instruction tokenization and image resizing from the env's
    512x384 observation to the model's 384x288 input.
    """

    def __init__(
        self,
        checkpoint: str,
        chunk_size: int = CHUNK.default_chunk_size,
        device: str = "cpu",
        corpus_sentences: list[str] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.chunk_size = chunk_size

        # Load checkpoint first to get token_id_map if saved
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        saved_token_map = state.pop("__token_id_map__", None)

        # Build text encoder — use saved token_id_map for exact vocab match.
        # If not in checkpoint, rebuild from the training dataset's initial_text
        # (NOT the full corpus, which has different vocab coverage).
        print("  Building text encoder...", flush=True)
        if saved_token_map is not None:
            text_encoder, self.tokenizer, _ = build_text_encoder()
            self.token_map = saved_token_map
            vocab_size = max(saved_token_map.values()) + 1
            text_encoder.resize_vocab(vocab_size)
        else:
            # Fallback: stream training dataset to get same corpus as
            # train.py without loading the full ~36GB Arrow table.
            print("  No token_id_map in checkpoint, streaming dataset for vocab...")
            from datasets import load_dataset as _load_dataset
            ds = _load_dataset(
                "PenTest-duck/cu-vla-exp5-data", split="train", streaming=True
            )
            train_corpus: set[str] = set()
            n_rows = 0
            for row in ds:
                train_corpus.add(row["initial_text"])
                n_rows += 1
                if n_rows % 50000 == 0:
                    print(
                        f"    {n_rows} rows streamed, "
                        f"{len(train_corpus)} unique passages so far...",
                        flush=True,
                    )
            print(f"  {len(train_corpus)} unique passages from {n_rows} rows")
            text_encoder, self.tokenizer, self.token_map = build_text_encoder(
                list(train_corpus)
            )
            del train_corpus

        # Build model with text encoder
        self.model = ACT(
            chunk_size=chunk_size,
            text_encoder=text_encoder,
        ).to(self.device)

        self.model.load_state_dict(state)
        self.model.train(False)

        # Temporal ensemble state
        self.active_chunks: list[dict] = []

        # Cached instruction tokens (set via set_instruction)
        self._cached_ids: torch.Tensor | None = None
        self._cached_mask: torch.Tensor | None = None

    def set_instruction(self, instruction_text: str) -> None:
        """Tokenize and cache instruction for current episode."""
        ids, mask = tokenize_instruction(
            instruction_text, self.tokenizer, self.token_map
        )
        self._cached_ids = ids.to(self.device)
        self._cached_mask = mask.to(self.device)

    def reset(self) -> None:
        self.active_chunks = []

    @torch.no_grad()
    def act(self, obs: dict, proprio: np.ndarray) -> dict:
        assert self._cached_ids is not None, (
            "Call set_instruction() before act()"
        )

        # Resize screenshot from 512x384 (obs) to 384x288 (model input)
        # Match train.py: PIL resize in uint8, then float32 / 255 on device
        screenshot = obs["screenshot"]  # (384, 512, 3) uint8
        from PIL import Image

        img_pil = Image.fromarray(screenshot)
        if img_pil.size != (MODEL.obs_w, MODEL.obs_h):
            img_pil = img_pil.resize(
                (MODEL.obs_w, MODEL.obs_h), Image.BILINEAR
            )
        img_np = np.array(img_pil)
        img_t = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
            .float()
            .div_(255.0)
        )

        p = torch.from_numpy(proprio).unsqueeze(0).float().to(self.device)

        # Forward pass with cached text tokens
        out = self.model(
            img_t,
            p,
            self._cached_ids,
            self._cached_mask,
        )

        # Store softmax/sigmoid distributions
        dx_probs = torch.softmax(out["dx_logits"][0], dim=-1).cpu().numpy()
        dy_probs = torch.softmax(out["dy_logits"][0], dim=-1).cpu().numpy()
        mouse_chunk = torch.sigmoid(out["mouse_left"][0]).cpu().numpy()
        keys_chunk = torch.sigmoid(out["keys_held"][0]).cpu().numpy()

        self.active_chunks.append({
            "dx_probs": dx_probs,
            "dy_probs": dy_probs,
            "mouse": mouse_chunk,
            "keys": keys_chunk,
            "age": 0,
        })

        # Probability ensemble: blend all active chunk distributions
        total_w = 0.0
        blended_dx = np.zeros(NUM_BINS, dtype=np.float32)
        blended_dy = np.zeros(NUM_BINS, dtype=np.float32)
        blended_mouse = 0.0
        blended_keys = np.zeros(ACTION.num_keys, dtype=np.float32)

        # Separate faster decay for keys to prevent stale keypresses
        total_w_keys = 0.0

        for chunk in self.active_chunks:
            age = chunk["age"]
            if age >= self.chunk_size:
                continue
            w = np.exp(-CHUNK.ensemble_decay * age)
            w_keys = np.exp(-CHUNK.key_decay * age)
            total_w += w
            total_w_keys += w_keys
            blended_dx += w * chunk["dx_probs"][age]
            blended_dy += w * chunk["dy_probs"][age]
            blended_mouse += w * chunk["mouse"][age]
            blended_keys += w_keys * chunk["keys"][age]

        if total_w > 0:
            blended_dx /= total_w
            blended_dy /= total_w
            blended_mouse /= total_w
        if total_w_keys > 0:
            blended_keys /= total_w_keys

        # Expected value from blended distributions
        dx_out = float(np.dot(blended_dx, BIN_CENTERS))
        dy_out = float(np.dot(blended_dy, BIN_CENTERS))
        mouse_out = 1 if blended_mouse > 0.5 else 0
        keys_out = [
            1 if blended_keys[i] > 0.5 else 0 for i in range(ACTION.num_keys)
        ]

        # Increment ages, prune expired chunks
        for chunk in self.active_chunks:
            chunk["age"] += 1
        self.active_chunks = [
            c for c in self.active_chunks if c["age"] < self.chunk_size
        ]

        return {
            "dx": dx_out,
            "dy": dy_out,
            "mouse_left": mouse_out,
            "keys_held": keys_out,
        }


class ExpertAgent:
    """Wraps the expert policy for use in the evaluation loop.

    The expert pre-computes a full trajectory via generate_episode_trajectory(),
    then replays the recorded actions step by step.
    """

    def __init__(self) -> None:
        self._trajectory: list[dict] = []
        self._step: int = 0
        self._env_ref: MiniEditorEnv | None = None

    def bind_env(self, env: MiniEditorEnv) -> None:
        self._env_ref = env

    def set_instruction(self, instruction_text: str) -> None:
        # Expert doesn't need text — it uses the EditInstruction directly
        pass

    def generate_trajectory(
        self, instruction: EditInstruction, rng: np.random.Generator
    ) -> None:
        """Pre-compute the full expert trajectory for this episode."""
        assert self._env_ref is not None
        self._trajectory = generate_episode_trajectory(
            self._env_ref, instruction, rng
        )
        self._step = 0

    def reset(self) -> None:
        self._trajectory = []
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

    def set_instruction(self, instruction_text: str) -> None:
        pass

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
    env: MiniEditorEnv,
    corpus_sentences: list[str],
    num_episodes: int,
    max_steps: int,
) -> dict:
    """Run agent in environment and collect per-episode metrics.

    Each episode:
    1. Sample a passage and instruction from the corpus.
    2. Reset env with passage text and set expected text.
    3. Set instruction on agent.
    4. Run the agent step-by-step until done or max_steps.
    5. Record: success, steps, operation type, latency.
    """
    results: dict = {
        "total_steps": [],
        "success": [],
        "operation": [],
        "loop_times_ms": [],
    }

    is_expert = isinstance(agent, ExpertAgent)

    from tqdm import tqdm

    for ep in tqdm(range(num_episodes), desc="  Episodes", leave=False):
        seed = ep + 10000
        rng = np.random.default_rng(seed)

        # Sample a passage
        passage = None
        for _ in range(100):
            passage = make_passage(corpus_sentences, rng)
            if passage is not None:
                break
        if passage is None:
            # Fallback: use a simple sentence
            passage = "The quick brown fox jumps over the lazy dog near a river."

        words = extract_words(passage)
        vocab = [w["word"] for w in words]
        instruction = generate_instruction(passage, words, rng, vocab)

        # Reset env with the passage text
        obs = env.reset(text=passage, seed=seed)
        env.set_expected_text(instruction.expected_text)

        if is_expert:
            agent.bind_env(env)
            agent.generate_trajectory(instruction, rng)

        agent.set_instruction(instruction.instruction_text)
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
                results["operation"].append(instruction.operation)
                results["loop_times_ms"].extend(step_times)
                break

        if not done:
            results["total_steps"].append(max_steps)
            results["success"].append(False)
            results["operation"].append(instruction.operation)
            results["loop_times_ms"].extend(step_times)

    return results


# ---------------------------------------------------------------------------
# Metrics display
# ---------------------------------------------------------------------------

_OPERATIONS = ["click", "click_type", "select_delete", "replace"]


def print_metrics(results: dict, name: str) -> None:
    """Print formatted metrics block for one agent, including per-op breakdown."""
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

    # Per-operation breakdown
    ops = results.get("operation", [])
    if ops:
        print()
        print("  Per-operation success:")
        for op in _OPERATIONS:
            op_mask = [i for i, o in enumerate(ops) if o == op]
            if not op_mask:
                continue
            op_success = sum(results["success"][i] for i in op_mask)
            op_total = len(op_mask)
            op_rate = op_success / op_total * 100
            print(
                f"    {op:20s}  {op_success}/{op_total}"
                f" ({op_rate:.1f}%)"
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    checkpoint: str | None = None,
    chunk_size: int = CHUNK.default_chunk_size,
    num_episodes: int = EVAL_CFG.num_episodes,
    visual: bool = False,
    device: str = "cpu",
    hf_checkpoint_repo: str | None = None,
    model_only: bool = False,
) -> None:
    base = os.path.dirname(__file__)
    max_steps = EVAL_CFG.max_steps_per_episode

    # Load corpus once
    print("Loading corpus...", flush=True)
    corpus_sentences = load_corpus()
    print(f"  {len(corpus_sentences)} sentences loaded.", flush=True)

    env = MiniEditorEnv(visual=visual, fps=ENV.control_hz if visual else 0)

    agent_names: list[str] = []
    all_results: dict[str, dict] = {}

    # --- Expert ---
    if not model_only:
        expert_name = "Expert"
        print(f"\nRunning expert...", flush=True)
        expert = ExpertAgent()
        expert_results = run_agent(
            expert, env, corpus_sentences, num_episodes, max_steps
        )
        print_metrics(expert_results, expert_name)
        all_results[expert_name] = expert_results
        agent_names.append(expert_name)

    # --- ACT ---
    act_default = os.path.join(
        base, "checkpoints", f"chunk{chunk_size}", "best.pt"
    )
    act_ckpt = checkpoint or resolve_checkpoint(
        act_default,
        hf_checkpoint_repo,
        f"mini_editor_chunk{chunk_size}",
    )
    act_name = f"ACT (chunk={chunk_size})"
    if act_ckpt and os.path.exists(act_ckpt):
        print(f"\nRunning ACT ({act_ckpt})...", flush=True)
        act_agent = ACTAgent(
            act_ckpt,
            chunk_size=chunk_size,
            device=device,
            corpus_sentences=corpus_sentences,
        )
        act_results = run_agent(
            act_agent, env, corpus_sentences, num_episodes, max_steps
        )
        print_metrics(act_results, act_name)
        all_results[act_name] = act_results
        agent_names.append(act_name)
    else:
        print(f"\nNo ACT checkpoint found. Skipping.", flush=True)

    # --- Random ---
    if not model_only:
        random_name = "Random"
        print(f"\nRunning random baseline...", flush=True)
        random_agent = RandomAgent()
        random_results = run_agent(
            random_agent, env, corpus_sentences, num_episodes, max_steps
        )
        print_metrics(random_results, random_name)
        all_results[random_name] = random_results
        agent_names.append(random_name)

    env.close()

    # Summary comparison
    if len(agent_names) > 1:
        print(f"\n{'#'*60}")
        print(f"  SUMMARY")
        print(f"{'#'*60}")
        for name in agent_names:
            r = all_results[name]
            total = len(r["success"])
            succ = sum(r["success"])
            rate = succ / total * 100 if total > 0 else 0
            steps = np.array(r["total_steps"])
            print(
                f"  {name:30s}  "
                f"{succ}/{total} ({rate:.1f}%)  "
                f"mean_steps={steps.mean():.0f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate agents on Mini Text Editor task"
    )
    parser.add_argument(
        "-n",
        "--num-episodes",
        type=int,
        default=EVAL_CFG.num_episodes,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to ACT checkpoint",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK.default_chunk_size,
        help="Action chunk size for ACT model",
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Show Pygame window during evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--hf-checkpoint-repo",
        type=str,
        default=None,
        help="HF model repo to auto-download checkpoints from",
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Only evaluate ACT model, skip expert and random",
    )
    args = parser.parse_args()

    main(
        checkpoint=args.checkpoint,
        chunk_size=args.chunk_size,
        num_episodes=args.num_episodes,
        visual=args.visual,
        device=args.device,
        hf_checkpoint_repo=args.hf_checkpoint_repo,
        model_only=args.model_only,
    )
