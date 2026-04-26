"""Phase A / B0 evaluation: offline per-head accuracy + closed-loop L-click success.

Phase A mode (``--phase a``): legacy single-button env, generic instruction.
Phase B0 mode (``--phase b0``, default): scene reconstructed from parquet
metadata via the same RNG seed used by the generator, with instruction-aware
rollout, slice filtering, and instruction probes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from experiments.action_primitives.config import (
    ENV, HEAD_LOGITS, MODEL, MOUSE_BIN_CENTERS, NUM_KEYS, PROPRIO_DIM,
)
from experiments.action_primitives.dataset import PhaseAEpisodeDataset, build_action_history_vector, decode_jpeg_bytes
from experiments.action_primitives.env import Action, LClickEnv
from experiments.action_primitives.history import HISTORY_INPUT_DIM
from experiments.action_primitives.model import ActionPrimitivesACT
from experiments.action_primitives.scene import generate_scene


# The generator consumes RNG calls before generate_scene to choose tempo.
# Eval must replay that exact sequence to reconstruct the same scene.
# See `generate_one_b0_episode` in generator.py.
_GENERATOR_TEMPO_CHOICES = ("slow", "normal", "fast", "superhuman")


def reconstruct_scene_for_episode(episode_id: int, base_seed: int = 0):
    """Reconstruct the Scene used by the generator for the given episode.

    The generator (``generator.generate_one_b0_episode``) does
    ``rng = np.random.default_rng(seed)``, then consumes RNG via
    ``rng.choice(TEMPO_CHOICES)`` before calling ``generate_scene(rng=rng)``.
    Eval must replay that exact sequence; otherwise the reconstructed scene
    diverges from what the model trained against.
    """
    rng = np.random.default_rng(base_seed + episode_id)
    _ = str(rng.choice(list(_GENERATOR_TEMPO_CHOICES)))  # mirrors generator
    return generate_scene(rng=rng)


def load_model(ckpt_path: str, device: str) -> ActionPrimitivesACT:
    model = ActionPrimitivesACT().to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model


def offline_eval(model: ActionPrimitivesACT, data_dir: Path, device: str) -> dict:
    """Per-head top-1 accuracy on val set.

    Returns the standard aggregate per-head accuracies plus a *click head
    decomposition*: click_idle_recall, click_press_recall, click_release_recall.
    The aggregate `click` metric is dominated by the 95% of idle frames, so a
    model that gets press/release 50% wrong can still show 99% aggregate.
    Decomposition exposes the real per-event reliability.
    """
    val_ds = PhaseAEpisodeDataset(data_dir, split="val")
    correct = {k: 0 for k in HEAD_LOGITS}
    total = 0
    # Click head per-class breakdown: class id → (correct, total).
    # Class ids per config: 0=idle, 1=L_press, 2=L_release, 3=R_press, 4=R_release.
    from experiments.action_primitives.config import (
        CLICK_IDLE, CLICK_L_PRESS, CLICK_L_RELEASE, CLICK_R_PRESS, CLICK_R_RELEASE,
    )
    click_class_names = {
        CLICK_IDLE: "idle",
        CLICK_L_PRESS: "l_press",
        CLICK_L_RELEASE: "l_release",
        CLICK_R_PRESS: "r_press",
        CLICK_R_RELEASE: "r_release",
    }
    click_correct_per_class = {c: 0 for c in click_class_names}
    click_total_per_class = {c: 0 for c in click_class_names}

    for idx in tqdm(range(len(val_ds)), desc="offline", unit="ep", dynamic_ncols=True):
        ep = val_ds[idx]
        # Dataset default (preprocess=True) returns preprocessed vision tensors,
        # not PIL images. Derive T from pixel_values and build the dict for
        # model.forward's fast dispatch path.
        T = ep["pixel_values"].shape[0]
        with torch.no_grad():
            text_tokens = model.backbone.encode_text([ep["instruction"]])  # (1, T_text, d)
            text_rep = text_tokens.expand(T, -1, -1)
            text_mask = torch.ones(T, text_tokens.size(1), device=device)
            vision_input = {
                "pixel_values": ep["pixel_values"],
                "pixel_attention_mask": ep["pixel_attention_mask"],
                "spatial_shapes": ep["spatial_shapes"],
            }
            out = model(vision_input, text_rep, text_mask,
                        ep["proprio"].to(device), ep["history"].to(device).float())
        for head in HEAD_LOGITS:
            if head == "keys":
                logits = out.head_logits[head].view(T, NUM_KEYS, 3)
                preds = logits.argmax(dim=-1)  # (T, 77)
                tgt = ep["key_events"].to(device)
                correct[head] += int((preds == tgt).sum())
                total_for_head = T * NUM_KEYS
            elif head == "done":
                preds = (torch.sigmoid(out.head_logits[head].squeeze(-1)) > 0.5).long()
                tgt = ep["dones"].to(device)
                correct[head] += int((preds == tgt).sum())
                total_for_head = T
            else:
                preds = out.head_logits[head].argmax(dim=-1)
                tgt = ep[f"{head}_bins" if head in ("dx", "dy", "scroll") else "clicks" if head == "click" else "dones"].to(device)
                correct[head] += int((preds == tgt).sum())
                total_for_head = T
                if head == "click":
                    # Per-class recall: for each click class, count how often
                    # the model predicts it correctly when that was the target.
                    for cls_id in click_class_names:
                        mask = (tgt == cls_id)
                        n_cls = int(mask.sum())
                        if n_cls > 0:
                            click_correct_per_class[cls_id] += int((preds[mask] == cls_id).sum())
                            click_total_per_class[cls_id] += n_cls
        total += T
    results = {k: correct[k] / max(1, total * (NUM_KEYS if k == "keys" else 1)) for k in correct}
    # Decomposed click metrics, reported as `click_<name>_recall`.
    for cls_id, name in click_class_names.items():
        n = click_total_per_class[cls_id]
        key = f"click_{name}_recall"
        results[key] = click_correct_per_class[cls_id] / n if n > 0 else float("nan")
        results[f"click_{name}_support"] = n
    return results


def _decode_mouse(
    logits: torch.Tensor, centers: np.ndarray, mode: str
) -> tuple[int, float]:
    """Decode a 21-bin mouse-delta softmax.

    Returns `(argmax_bin, continuous_value)`. The argmax bin is always returned
    for history-vector bookkeeping (training saw one-hot history). The
    continuous value is what actually gets applied via env.step.

    - ``mode="argmax"``: continuous_value = centers[argmax_bin]. This is the
      original Phase A decode — reproduces the existing 70.5% closed-loop number.
    - ``mode="expected"``: continuous_value = E[centers] under the softmax —
      uses the full head distribution rather than collapsing to one bin.
      Addresses the quantization-drift zig-zag: when the true continuous
      action is between two bin centers, argmax snaps to one and introduces
      up to ~1.5 px drift per step; expected-value returns the interpolated
      value and the cursor tracks the expert trajectory more tightly.
    """
    probs = torch.softmax(logits, dim=-1)  # (B=1, 21)
    bin_idx = int(probs.argmax(dim=-1))
    if mode == "argmax":
        value = float(centers[bin_idx])
    elif mode == "expected":
        centers_t = torch.as_tensor(centers, device=probs.device, dtype=probs.dtype)
        value = float((probs.squeeze(0) * centers_t).sum())
    else:
        raise ValueError(f"unknown decode mode: {mode!r} (expected 'argmax' or 'expected')")
    return bin_idx, value


def rollout_one_episode(
    model: ActionPrimitivesACT,
    env: LClickEnv,
    device: str,
    max_frames: int = ENV.max_frames_lclick,
    decode_mode: str = "argmax",
) -> dict:
    """Closed-loop rollout. Returns info dict with success flag.

    ``decode_mode`` controls how the dx/dy softmaxes are turned into continuous
    deltas — see `_decode_mouse`. Default matches prior Phase A behaviour.
    """
    obs, info = env.reset()
    with torch.no_grad():
        # Must match dataset.py's Phase A fixed instruction (no theme leakage).
        text_tokens = model.backbone.encode_text(["click the button"])
    K = MODEL.action_history_len
    history = np.zeros((K, HISTORY_INPUT_DIM), dtype=np.float32)
    for t in range(max_frames):
        prop = np.concatenate([
            [obs["proprio"].cursor_x, obs["proprio"].cursor_y],
            obs["proprio"].held_keys.astype(np.float32),
            obs["proprio"].held_mouse.astype(np.float32),
            [float(obs["proprio"].capslock)],
        ])
        proprio_t = torch.from_numpy(prop).float().unsqueeze(0).to(device)
        history_t = torch.from_numpy(history).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out = model([obs["image"]], text_tokens, torch.ones(1, text_tokens.size(1), device=device), proprio_t, history_t)
        # Decode mouse deltas (argmax or probabilistic). Bin index is still
        # tracked for the one-hot history vector — training never saw soft
        # history, and introducing it at inference would itself be a distribution
        # shift.
        dx_bin, dx = _decode_mouse(out.head_logits["dx"], MOUSE_BIN_CENTERS, decode_mode)
        dy_bin, dy = _decode_mouse(out.head_logits["dy"], MOUSE_BIN_CENTERS, decode_mode)
        # Click stays argmax (5-way categorical, not ordinal).
        click = int(out.head_logits["click"].argmax(dim=-1))
        # Apply
        action = Action(dx=dx, dy=dy, click=click)
        obs, done, info = env.step(action)
        # Update history (oldest-first)
        new_hist = build_action_history_vector([{
            "dx_bin": dx_bin, "dy_bin": dy_bin, "click": click, "scroll_bin": 10,
            "key_events": [2] * NUM_KEYS, "done_gt": int(done),
        }])
        history = np.concatenate([history[1:], new_hist], axis=0)
        if done:
            return {"success": True, "frames": t + 1}
    return {"success": False, "frames": max_frames}


def closed_loop_eval(
    model: ActionPrimitivesACT,
    device: str,
    n_episodes: int = 200,
    tolerances_px: list[int] | None = None,
    visual: bool = False,
    fps: int = 30,
    decode_mode: str = "argmax",
) -> dict:
    """Rollout n_episodes of L-click; report binary success + tolerance curves.

    When ``visual=True``, each env opens a pygame display window and the
    rollout runs at ``fps`` (or slower if the model is the bottleneck).
    Prints per-episode success as it goes so the user can track live.

    ``decode_mode`` is forwarded to ``rollout_one_episode``; see `_decode_mouse`.
    """
    if tolerances_px is None:
        tolerances_px = [0, 3, 5, 10]
    results = []
    # Visual mode prints per-episode lines; non-visual shows a tqdm bar with
    # running success rate in the postfix so you can track progress.
    iterator = range(n_episodes)
    pbar = None
    if not visual:
        pbar = tqdm(iterator, desc=f"closed-loop({decode_mode})", unit="ep", dynamic_ncols=True)
        iterator = pbar
    for i in iterator:
        env = LClickEnv(seed=10000 + i, visual=visual, fps=fps)
        res = rollout_one_episode(model, env, device, decode_mode=decode_mode)
        # For tolerance curves, we check cursor distance to target center at episode end
        x, y, w, h = env._info()["target_bbox"]
        tx, ty = x + w / 2, y + h / 2
        cx, cy = env.cursor_x, env.cursor_y
        dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
        res["dist_px"] = dist
        results.append(res)
        running = sum(r["success"] for r in results) / len(results)
        if visual:
            tag = "✓" if res["success"] else "✗"
            print(f"  [{i+1:>3}/{n_episodes}] {tag}  "
                  f"frames={res['frames']:>2}  dist={res['dist_px']:.1f}px  "
                  f"running_success={running:.3f}", flush=True)
        elif pbar is not None:
            pbar.set_postfix(success=f"{running:.3f}")
    out = {"n_episodes": n_episodes, "success_rate": sum(r["success"] for r in results) / n_episodes}
    for tol in tolerances_px:
        out[f"click_within_{tol}px"] = sum(1 for r in results if r["dist_px"] <= tol) / n_episodes
    return out


def _decode_b0_click(head_logits: dict[str, torch.Tensor]) -> int:
    """Map B0's two 3-way click heads back to the env's 5-way action.

    Env still consumes the legacy 5-way encoding {0=idle, 1=L_press, 2=L_release,
    3=R_press, 4=R_release}. B0 model emits ``click_left`` and ``click_right``
    each in {0=idle, 1=press, 2=release}. We pick whichever (L or R) is most
    confident at non-idle; fall back to idle.
    """
    left = head_logits["click_left"]   # (1, 3)
    right = head_logits["click_right"] # (1, 3)
    left_pred = int(left.argmax(dim=-1))
    right_pred = int(right.argmax(dim=-1))
    if left_pred == 0 and right_pred == 0:
        return 0  # idle
    if left_pred != 0 and right_pred == 0:
        return 1 if left_pred == 1 else 2  # L_press / L_release
    if left_pred == 0 and right_pred != 0:
        return 3 if right_pred == 1 else 4  # R_press / R_release
    left_conf = float(torch.softmax(left, dim=-1).max())
    right_conf = float(torch.softmax(right, dim=-1).max())
    if left_conf >= right_conf:
        return 1 if left_pred == 1 else 2
    return 3 if right_pred == 1 else 4


def _is_b0_model(model: ActionPrimitivesACT) -> bool:
    """Detect whether the model has the B0 dual-click heads.

    Note: heads.py prefixes ModuleDict keys with 'head_' to avoid colliding with
    nn.ModuleDict's built-in .keys() method (one head is literally named "keys").
    So we check for "head_click_left" / "head_click_right", not the bare names.
    """
    return "head_click_left" in model.heads.heads and "head_click_right" in model.heads.heads


def load_episode_metadata(data_dir: Path, split: str = "test") -> pd.DataFrame:
    """Load one row per episode with all metadata fields needed for B0 eval slicing.

    Reads all parquet shards in ``data_dir``, filters by episode_id-hash split
    (mirrors :class:`PhaseB0EpisodeDataset`'s split logic), and dedups to one row
    per episode.
    """
    shards = sorted(Path(data_dir).glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards in {data_dir}")
    parts = []
    for s in shards:
        df_full = pd.read_parquet(s)
        cols = [c for c in (
            "episode_id", "frame_idx", "instruction", "target_button_id", "n_buttons",
            "composite_tier", "is_adversarial", "is_scenario_error", "scenario_type",
            "k_wrong_frames",
        ) if c in df_full.columns]
        parts.append(df_full[cols])
    df = pd.concat(parts, ignore_index=True)
    if "frame_idx" in df.columns:
        df = df[df["frame_idx"] == 0].drop(columns=["frame_idx"])
    else:
        df = df.drop_duplicates(subset=["episode_id"], keep="first")
    if split == "all":
        return df.reset_index(drop=True)
    bucket_pred = {"train": lambda b: b < 8, "val": lambda b: b == 8, "test": lambda b: b == 9}
    pred = bucket_pred[split]
    df = df[df["episode_id"].apply(lambda eid: pred(eid % 10))]
    return df.reset_index(drop=True)


def _proprio_to_array(proprio) -> np.ndarray:
    """Concatenate Proprio dataclass into the 83-dim flat vector."""
    return np.concatenate([
        np.array([proprio.cursor_x, proprio.cursor_y], dtype=np.float32),
        proprio.held_keys.astype(np.float32),
        proprio.held_mouse.astype(np.float32),
        np.array([float(proprio.capslock)], dtype=np.float32),
    ])


def rollout_one_episode_b0(
    model: ActionPrimitivesACT,
    env: LClickEnv,
    instruction: str,
    device: str,
    max_frames: int = ENV.max_frames_lclick,
    decode_mode: str = "argmax",
) -> dict:
    """B0 closed-loop rollout: instruction-aware + cursor-track + dual click heads."""
    obs, info = env.reset()
    is_b0 = _is_b0_model(model)
    with torch.no_grad():
        text_tokens = model.backbone.encode_text([instruction])
    K = MODEL.action_history_len
    history = np.zeros((K, HISTORY_INPUT_DIM), dtype=np.float32)
    cursor_xys: list[tuple[float, float]] = [(env.cursor_x, env.cursor_y)]
    success = False
    frames_used = max_frames
    for t in range(max_frames):
        prop = _proprio_to_array(obs["proprio"])
        proprio_t = torch.from_numpy(prop).float().unsqueeze(0).to(device)
        history_t = torch.from_numpy(history).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(
                [obs["image"]], text_tokens,
                torch.ones(1, text_tokens.size(1), device=device),
                proprio_t, history_t,
            )
        dx_bin, dx = _decode_mouse(out.head_logits["dx"], MOUSE_BIN_CENTERS, decode_mode)
        dy_bin, dy = _decode_mouse(out.head_logits["dy"], MOUSE_BIN_CENTERS, decode_mode)
        if is_b0:
            click = _decode_b0_click(out.head_logits)
        else:
            click = int(out.head_logits["click"].argmax(dim=-1))
        action = Action(dx=dx, dy=dy, click=click)
        obs, done, info = env.step(action)
        cursor_xys.append((env.cursor_x, env.cursor_y))
        new_hist = build_action_history_vector([{
            "dx_bin": dx_bin, "dy_bin": dy_bin, "click": click, "scroll_bin": 10,
            "key_events": [2] * NUM_KEYS, "done_gt": int(done),
        }])
        history = np.concatenate([history[1:], new_hist], axis=0)
        if done:
            success = True
            frames_used = t + 1
            break
    return {
        "success": success,
        "frames": frames_used,
        "cursor_xys": cursor_xys,
        "final_cursor": (env.cursor_x, env.cursor_y),
    }


def run_closed_loop_eval_b0(
    model: ActionPrimitivesACT,
    device: str,
    data_dir: Path,
    slice_name: str | None = None,
    n_rollouts: int = 200,
    instruction_probe: str = "none",
    decode_mode: str = "argmax",
    visual: bool = False,
    fps: int = 30,
    base_seed: int = 0,
    tolerances_px: list[int] | None = None,
    probe_seed: int = 0,
) -> dict:
    """B0 closed-loop eval: filter slice, reconstruct scenes from parquet, optional probes.

    For each episode in the test split (filtered by ``slice_name`` if given):
    1. Read ``target_button_id`` + ``instruction`` + ``episode_id`` from parquet.
    2. Reconstruct ``scene = generate_scene(rng=np.random.default_rng(base_seed + episode_id))``.
    3. Optionally rewrite the instruction per ``instruction_probe``.
    4. Run rollout and collect cursor positions + success.
    """
    if tolerances_px is None:
        tolerances_px = [0, 3, 5, 10]
    meta_df = load_episode_metadata(data_dir, split="test")
    if slice_name is not None:
        meta_df = filter_eval_split(meta_df, slice_name)
    if len(meta_df) == 0:
        return {
            "slice": slice_name, "probe": instruction_probe,
            "n_rollouts": 0, "closed_loop_success_rate": float("nan"),
            "wrong_direction_first_3_frames": float("nan"),
            "tolerance_curve": {str(t): float("nan") for t in tolerances_px},
            "note": "no episodes matched filter",
        }
    meta_df = meta_df.sort_values("episode_id").reset_index(drop=True)
    if len(meta_df) > n_rollouts:
        meta_df = meta_df.iloc[:n_rollouts]
    actual_n = len(meta_df)

    val_instructions: list[str] = []
    if instruction_probe == "shuffled":
        val_meta = load_episode_metadata(data_dir, split="val")
        val_instructions = sorted(set(val_meta["instruction"].astype(str).tolist()))

    probe_rng = np.random.default_rng(probe_seed)
    successes: list[bool] = []
    wrong_dir_flags: list[bool] = []
    final_dists: list[float] = []

    iterator = range(actual_n)
    pbar = None
    if not visual:
        pbar = tqdm(
            iterator,
            desc=f"closed-loop-b0[{slice_name or 'all'}/{instruction_probe}]({decode_mode})",
            unit="ep", dynamic_ncols=True,
        )
        iterator = pbar
    for i in iterator:
        row = meta_df.iloc[i]
        eid = int(row["episode_id"])
        instruction = str(row["instruction"])
        target_id = int(row["target_button_id"])

        # Reconstruct scene from generator's deterministic seed (must replay
        # the tempo-choice RNG call the generator made before scene gen).
        scene = reconstruct_scene_for_episode(eid, base_seed=base_seed)
        if not (0 <= target_id < len(scene.buttons)):
            continue

        used_instruction = instruction
        if instruction_probe == "zero":
            used_instruction = ""
        elif instruction_probe == "shuffled":
            used_instruction = build_shuffled_instruction(probe_rng, val_instructions, exclude=instruction)
        elif instruction_probe == "wrong":
            used_instruction = build_wrong_instruction(scene, target_id, probe_rng)
            if used_instruction == "":
                used_instruction = instruction  # 1-button fallback

        env = LClickEnv(
            scene=scene, target_button_id=target_id,
            seed=10000 + i, visual=visual, fps=fps,
        )
        res = rollout_one_episode_b0(
            model, env, used_instruction, device, decode_mode=decode_mode,
        )

        target = scene.buttons[target_id]
        target_xy = (target.x + target.w / 2, target.y + target.h / 2)
        wrong_dir = compute_wrong_direction_first_3_frames(res["cursor_xys"], target_xy)
        cx, cy = res["final_cursor"]
        dist = ((cx - target_xy[0]) ** 2 + (cy - target_xy[1]) ** 2) ** 0.5
        successes.append(bool(res["success"]))
        wrong_dir_flags.append(bool(wrong_dir))
        final_dists.append(float(dist))

        running = sum(successes) / len(successes)
        if visual:
            tag = "OK" if res["success"] else "FAIL"
            print(f"  [{i+1:>3}/{actual_n}] {tag}  eid={eid}  "
                  f"frames={res['frames']:>2}  dist={dist:.1f}px  "
                  f"running_success={running:.3f}", flush=True)
        elif pbar is not None:
            pbar.set_postfix(success=f"{running:.3f}")

    n = len(successes)
    out = {
        "slice": slice_name,
        "probe": instruction_probe,
        "n_rollouts": n,
        "closed_loop_success_rate": (sum(successes) / n) if n > 0 else float("nan"),
        "wrong_direction_first_3_frames": (sum(wrong_dir_flags) / n) if n > 0 else float("nan"),
        "tolerance_curve": {
            str(tol): (sum(1 for d in final_dists if d <= tol) / n) if n > 0 else float("nan")
            for tol in tolerances_px
        },
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/phase-a-lclick")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    parser.add_argument("--n-rollouts", type=int, default=200)
    parser.add_argument("--skip-offline", action="store_true")
    parser.add_argument("--visual", action="store_true",
                        help="Open a pygame display window for closed-loop rollouts. "
                             "Prints per-episode success line as it goes. Implies a slower "
                             "run because each frame is flipped to the screen; use a small "
                             "--n-rollouts (e.g. 20) when watching live.")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frame rate cap when --visual is set (default 30).")
    parser.add_argument("--decode", choices=["argmax", "expected"], default="argmax",
                        help="Decode mode for mouse dx/dy bins. 'argmax' (default) "
                             "snaps to the top-1 bin center. 'expected' computes "
                             "E[center] under the softmax. Click stays argmax regardless.")
    parser.add_argument("--phase", type=str, default="b0", choices=["a", "b0"],
                        help="Eval mode. 'a' uses Phase A legacy rollout (no scene). "
                             "'b0' (default) reconstructs scene from parquet metadata "
                             "and threads the per-episode instruction into the model.")
    parser.add_argument("--slice", type=str, default=None,
                        choices=["phase_a_holdout", "multi_btn_generic", "multi_btn_composite",
                                 "scenario_recovery", "adversarial"],
                        help="(B0 only) Filter test episodes to a B0 eval slice. "
                             "None means all test episodes.")
    parser.add_argument("--instruction-probe", type=str, default="none",
                        choices=["none", "zero", "shuffled", "wrong"],
                        help="(B0 only) Apply an instruction probe at inference time. "
                             "'zero' = empty string. 'shuffled' = a random other val "
                             "instruction. 'wrong' = an instruction targeting a "
                             "different button.")
    parser.add_argument("--report-by-tier", action="store_true",
                        help="(B0 + --slice adversarial) Print per-tier success-rate "
                             "breakdown. Currently a TODO — see writeup task.")
    parser.add_argument("--base-seed", type=int, default=0,
                        help="(B0 only) Base seed used by the data generator.")
    parser.add_argument("--probe-seed", type=int, default=0,
                        help="(B0 only) RNG seed for the instruction-probe shuffler.")
    parser.add_argument("--json-out", type=str, default=None,
                        help="(B0 only) Path to write the closed-loop results JSON. "
                             "Default: eval_results_<slice>_<probe>.json under cwd.")
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)
    if not args.skip_offline:
        print("=== offline eval ===")
        off = offline_eval(model, Path(args.data_dir), args.device)
        for head in ("dx", "dy", "click", "scroll", "keys", "done"):
            if head in off:
                print(f"  {head}: {off[head]:.4f}")
        print("  --- click head decomposition (recall per class; support in brackets) ---")
        for cls in ("idle", "l_press", "l_release", "r_press", "r_release"):
            recall = off.get(f"click_{cls}_recall", float("nan"))
            support = off.get(f"click_{cls}_support", 0)
            recall_str = f"{recall:.4f}" if not np.isnan(recall) else "n/a"
            print(f"    click_{cls:10s} recall={recall_str}  [support={support}]")

    if args.phase == "a":
        print(f"=== closed-loop eval [phase=a] (decode={args.decode}) ===")
        cl = closed_loop_eval(
            model, args.device, n_episodes=args.n_rollouts,
            visual=args.visual, fps=args.fps,
            decode_mode=args.decode,
        )
        for k, v in cl.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        return

    slice_label = args.slice or "all"
    print(f"=== closed-loop eval [phase=b0 slice={slice_label} probe={args.instruction_probe}] "
          f"(decode={args.decode}) ===")
    cl = run_closed_loop_eval_b0(
        model=model, device=args.device, data_dir=Path(args.data_dir),
        slice_name=args.slice, n_rollouts=args.n_rollouts,
        instruction_probe=args.instruction_probe,
        decode_mode=args.decode, visual=args.visual, fps=args.fps,
        base_seed=args.base_seed, probe_seed=args.probe_seed,
    )
    print(f"  n_rollouts: {cl['n_rollouts']}")
    print(f"  closed_loop_success_rate: {cl['closed_loop_success_rate']:.4f}")
    print(f"  wrong_direction_first_3_frames: {cl['wrong_direction_first_3_frames']:.4f}")
    print("  tolerance_curve:")
    for tol_str, frac in cl["tolerance_curve"].items():
        print(f"    click_within_{tol_str}px: {frac:.4f}")

    if args.slice == "adversarial" and args.report_by_tier:
        # TODO(Task 30 / writeup): per-tier breakdown requires the
        # `used_attrs` field which isn't currently emitted to parquet (the
        # InstructionResult is consumed inside the generator and only its
        # composite_tier survives). For the writeup, either add `used_attrs`
        # to the parquet schema in a future generation, OR re-derive it
        # post-hoc from instruction text + scene attributes.
        print("  [report-by-tier] TODO -- used_attrs not in parquet; "
              "writeup must classify post-hoc. See Task 30.")

    out_path = Path(args.json_out) if args.json_out else Path(
        f"eval_results_{slice_label}_{args.instruction_probe}.json"
    )
    out_path.write_text(json.dumps(cl, indent=2))
    print(f"  wrote {out_path}")


def filter_eval_split(df, slice_name: str):
    """Filter a parquet DataFrame to a B0 eval slice.

    Slices (per the B0 design doc):
    - phase_a_holdout: n_buttons == 1 and not is_scenario_error (apples-to-apples Phase A baseline)
    - multi_btn_generic: n_buttons >= 2 and composite_tier == 1 and not is_adversarial
    - multi_btn_composite: n_buttons >= 2 and composite_tier >= 2 and not is_adversarial
    - scenario_recovery: is_scenario_error == 1
    - adversarial: is_adversarial == 1
    """
    if slice_name == "phase_a_holdout":
        return df[(df["n_buttons"] == 1) & (df["is_scenario_error"] == 0)]
    elif slice_name == "multi_btn_generic":
        return df[(df["n_buttons"] >= 2) & (df["composite_tier"] == 1) & (df["is_adversarial"] == 0)]
    elif slice_name == "multi_btn_composite":
        return df[(df["n_buttons"] >= 2) & (df["composite_tier"] >= 2) & (df["is_adversarial"] == 0)]
    elif slice_name == "scenario_recovery":
        return df[df["is_scenario_error"] == 1]
    elif slice_name == "adversarial":
        return df[df["is_adversarial"] == 1]
    else:
        raise ValueError(f"Unknown slice_name: {slice_name!r}. Valid: phase_a_holdout, multi_btn_generic, multi_btn_composite, scenario_recovery, adversarial")


def classify_adversarial_tier(scene, target_id, used_attrs):
    """Identify which attribute is ambiguous (forcing the composite).

    Returns:
        "color-ambiguous": single-attr instruction, used_attrs is exactly the one that
            disambiguates while another attribute (color/shape/size/position) is shared
        "shape-ambiguous", "size-ambiguous", "position-ambiguous": likewise
        "2-attr-needed": used_attrs has length 2 (composite)
        "3-attr-needed": used_attrs has length 3 (full composite)
        "single-unique": single-attr but no other attr shared (no ambiguity to flag)
    """
    target = scene.buttons[target_id]
    ambiguous_attrs: list[str] = []
    for attr in ("color", "shape", "size", "position"):
        target_val = _attr_value_for_classify(target, attr)
        for j, other in enumerate(scene.buttons):
            if j == target_id:
                continue
            if _attr_value_for_classify(other, attr) == target_val:
                ambiguous_attrs.append(attr)
                break
    if len(used_attrs) >= 3:
        return "3-attr-needed"
    if len(used_attrs) == 2:
        return "2-attr-needed"
    # used_attrs == 1 (single-attr instruction)
    if len(ambiguous_attrs) == 0:
        return "single-unique"
    # Pick the first ambiguous attr name as the tier label.
    return f"{ambiguous_attrs[0]}-ambiguous"


def _attr_value_for_classify(b, attr):
    if attr == "position":
        return b.pos_zone
    if attr == "color":
        return b.color
    if attr == "shape":
        return b.shape
    if attr == "size":
        return b.size
    raise ValueError(f"Unknown attribute: {attr}")


def compute_wrong_direction_first_3_frames(
    cursor_xys: list[tuple[float, float]],
    target_xy: tuple[float, float],
) -> bool:
    """Returns True if cursor net displacement over the first 3 frames is AWAY from target.

    Compares distance(start → target) vs distance(end_of_3rd_frame → target).
    If the cursor got farther from target, the rollout went wrong-direction.

    Returns False if fewer than 3 frames are available (insufficient data).
    """
    if len(cursor_xys) < 3:
        return False
    start = cursor_xys[0]
    end = cursor_xys[2]
    start_dist = ((target_xy[0] - start[0])**2 + (target_xy[1] - start[1])**2)**0.5
    end_dist = ((target_xy[0] - end[0])**2 + (target_xy[1] - end[1])**2)**0.5
    return end_dist > start_dist  # got farther from target


def build_zero_instruction_embedding(emb_dim: int = 768) -> torch.Tensor:
    """Construct an all-zero text embedding for the zero-instruction probe.

    Returns a (1, emb_dim) tensor — typically used to replace the model's
    instruction-encoder output at inference time.
    """
    return torch.zeros(1, emb_dim)


def build_shuffled_instruction(
    rng: np.random.Generator,
    val_instructions: list[str],
    exclude: str | None = None,
) -> str:
    """Return a random instruction from val_instructions, optionally excluding `exclude`.

    Used for the shuffled-instruction probe — tests language reliance robustness
    (zero-embedding may be OOD; a shuffled real instruction is in-distribution).
    """
    if exclude is None:
        return val_instructions[int(rng.integers(0, len(val_instructions)))]
    candidates = [s for s in val_instructions if s != exclude]
    if not candidates:
        return val_instructions[0]  # only one unique instruction; return it
    return candidates[int(rng.integers(0, len(candidates)))]


def build_wrong_instruction(scene, target_id: int, rng: np.random.Generator) -> str:
    """Construct an instruction targeting a DIFFERENT button than target_id.

    Used for the wrong-instruction probe — strongest grounding test: the model
    must follow the instruction even when the visually-salient or scene-prior
    target is the "wrong" answer.

    For 1-button scenes returns "" (no alternative target available).
    """
    other_buttons = [i for i in range(len(scene.buttons)) if i != target_id]
    if not other_buttons:
        return ""
    fake_target_id = int(rng.choice(other_buttons))
    fake_target = scene.buttons[fake_target_id]
    # Generate a simple single-attribute instruction targeting the fake button.
    # We pick an attribute that uniquely identifies the fake from the (real)
    # target. If multiple attrs work, prefer color (most frequent in templates).
    return f"click the {fake_target.color} {fake_target.shape}"


if __name__ == "__main__":
    main()
