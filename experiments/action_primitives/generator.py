"""Single-episode generator for L-click primitive.

Runs LClickEnv + LClickExpert in lockstep, emits one dict per frame including
rendered frame, proprio, expert action. Frame padding up to max_frames_lclick
uses no-op actions.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from PIL import Image

from experiments.action_primitives.config import ENV, NUM_KEYS
from experiments.action_primitives.env import Action, LClickEnv
from experiments.action_primitives.expert import LClickExpert, LClickExpertConfig


TEMPO_CHOICES = ("slow", "normal", "fast", "superhuman")
THEME_CHOICES = ("flat-modern", "flat-minimal", "dark-mode")


def _noop_action() -> Action:
    return Action()


def _action_to_row(a: Action) -> dict:
    return {
        "action_dx": float(a.dx),
        "action_dy": float(a.dy),
        "action_click": int(a.click),
        "action_scroll": float(a.scroll),
        "action_key_events": a.key_events.astype(np.int8).tolist(),
    }


def _proprio_to_row(p) -> dict:
    return {
        "cursor_x": float(p.cursor_x),
        "cursor_y": float(p.cursor_y),
        "held_keys": p.held_keys.astype(np.int8).tolist(),
        "held_mouse": p.held_mouse.astype(np.int8).tolist(),
        "capslock": int(p.capslock),
    }


def _image_to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def generate_one_episode(
    episode_id: int,
    seed: int,
    primitive: str = "lclick",
    theme: str | None = None,
    tempo: str | None = None,
    max_frames: int = ENV.max_frames_lclick,
) -> list[dict]:
    """Generate one episode; return list of per-frame rows."""
    rng = np.random.default_rng(seed)
    theme = theme if theme is not None else rng.choice(list(THEME_CHOICES))
    tempo = tempo if tempo is not None else rng.choice(list(TEMPO_CHOICES))

    env = LClickEnv(theme=theme, seed=seed)
    obs, info = env.reset(seed=seed)
    x, y, w, h = info["target_bbox"]
    target_center = (x + w / 2, y + h / 2)
    cursor_xy = info["cursor_xy"]

    expert_cfg = LClickExpertConfig(tempo=tempo, seed=seed + 1)
    expert = LClickExpert(expert_cfg, cursor_xy, target_center)

    rows: list[dict] = []
    done_frame = None

    # Drive expert until done, padding with no-ops up to max_frames
    frame_idx = 0
    expert_iter = iter(expert)
    while frame_idx < max_frames:
        if done_frame is None:
            try:
                action = next(expert_iter)
            except StopIteration:
                done_frame = frame_idx
                action = _noop_action()
        else:
            action = _noop_action()

        row = {
            "episode_id": int(episode_id),
            "frame_idx": int(frame_idx),
            "image_bytes": _image_to_jpeg_bytes(obs["image"]),
            "primitive_type": primitive,
            "theme": theme,
            "tempo": tempo,
            "target_bbox_x": int(x),
            "target_bbox_y": int(y),
            "target_bbox_w": int(w),
            "target_bbox_h": int(h),
            "done_gt": 1 if (done_frame is not None and frame_idx >= done_frame) else 0,
        }
        row.update(_action_to_row(action))
        row.update(_proprio_to_row(obs["proprio"]))
        rows.append(row)
        obs, env_done, info = env.step(action)
        # Sanity: env.done should align with expert-done within a 2-frame window
        frame_idx += 1

    return rows
