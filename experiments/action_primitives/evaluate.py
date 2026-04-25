"""Phase A evaluation: offline per-head accuracy + closed-loop L-click success."""
from __future__ import annotations

import argparse
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
                             "snaps to the top-1 bin center — reproduces the original "
                             "Phase A numbers. 'expected' computes E[center] under the "
                             "softmax, giving continuous output that tracks the expert "
                             "trajectory more tightly and reduces quantization-induced "
                             "zig-zag. Click stays argmax regardless (5-way categorical, "
                             "not ordinal).")
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)
    if not args.skip_offline:
        print("=== offline eval ===")
        off = offline_eval(model, Path(args.data_dir), args.device)
        # Print aggregates first, then the click decomposition separately.
        for head in ("dx", "dy", "click", "scroll", "keys", "done"):
            print(f"  {head}: {off[head]:.4f}")
        print("  --- click head decomposition (recall per class; support in brackets) ---")
        for cls in ("idle", "l_press", "l_release", "r_press", "r_release"):
            recall = off.get(f"click_{cls}_recall", float("nan"))
            support = off.get(f"click_{cls}_support", 0)
            recall_str = f"{recall:.4f}" if not np.isnan(recall) else "n/a"
            print(f"    click_{cls:10s} recall={recall_str}  [support={support}]")
    print(f"=== closed-loop eval (decode={args.decode}) ===")
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


if __name__ == "__main__":
    main()
