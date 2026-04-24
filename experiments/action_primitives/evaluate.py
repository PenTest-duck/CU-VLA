"""Phase A evaluation: offline per-head accuracy + closed-loop L-click success."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
    """Per-head top-1 accuracy on val set."""
    val_ds = PhaseAEpisodeDataset(data_dir, split="val")
    correct = {k: 0 for k in HEAD_LOGITS}
    total = 0
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
        total += T
    return {k: correct[k] / max(1, total * (NUM_KEYS if k == "keys" else 1)) for k in correct}


def rollout_one_episode(model: ActionPrimitivesACT, env: LClickEnv, device: str, max_frames: int = ENV.max_frames_lclick) -> dict:
    """Closed-loop rollout. Returns info dict with success flag."""
    obs, info = env.reset()
    with torch.no_grad():
        text_tokens = model.backbone.encode_text([f"click the {env.theme} button" if False else "click the button"])
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
        # Argmax decode
        dx_bin = int(out.head_logits["dx"].argmax(dim=-1))
        dy_bin = int(out.head_logits["dy"].argmax(dim=-1))
        dx = float(MOUSE_BIN_CENTERS[dx_bin])
        dy = float(MOUSE_BIN_CENTERS[dy_bin])
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
    tolerances_px: list[int] = [0, 3, 5, 10],
    visual: bool = False,
    fps: int = 30,
) -> dict:
    """Rollout n_episodes of L-click; report binary success + tolerance curves.

    When ``visual=True``, each env opens a pygame display window and the
    rollout runs at ``fps`` (or slower if the model is the bottleneck).
    Prints per-episode success as it goes so the user can track live.
    """
    results = []
    # Visual mode prints per-episode lines; non-visual shows a tqdm bar with
    # running success rate in the postfix so you can track progress.
    iterator = range(n_episodes)
    pbar = None
    if not visual:
        pbar = tqdm(iterator, desc="closed-loop", unit="ep", dynamic_ncols=True)
        iterator = pbar
    for i in iterator:
        env = LClickEnv(seed=10000 + i, visual=visual, fps=fps)
        res = rollout_one_episode(model, env, device)
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
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)
    if not args.skip_offline:
        print("=== offline eval ===")
        off = offline_eval(model, Path(args.data_dir), args.device)
        for k, v in off.items():
            print(f"  {k}: {v:.4f}")
    print("=== closed-loop eval ===")
    cl = closed_loop_eval(
        model, args.device, n_episodes=args.n_rollouts,
        visual=args.visual, fps=args.fps,
    )
    for k, v in cl.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
