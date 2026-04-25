"""Single-episode generator for L-click primitive.

Runs LClickEnv + LClickExpert in lockstep, emits one dict per frame including
rendered frame, proprio, expert action. Frame padding up to max_frames_lclick
uses no-op actions.
"""
from __future__ import annotations

import io
import warnings

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
    done_frame: int | None = None
    env_done_frame: int | None = None

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
        if env_done and env_done_frame is None:
            env_done_frame = frame_idx
        frame_idx += 1

    # Drift check: compare env-reported done vs expert-reported done.
    # Expected offset is 1 frame (env.step flips done_flag on the release
    # frame, but the row for that frame was already appended above).
    if env_done_frame is None:
        warnings.warn(
            f"episode_id={episode_id} seed={seed}: env never signalled success "
            f"(expert done_frame={done_frame}); possible label noise.",
            stacklevel=2,
        )
    elif done_frame is not None and abs(env_done_frame - done_frame) > 2:
        warnings.warn(
            f"episode_id={episode_id} seed={seed}: env_done_frame={env_done_frame} "
            f"drifts from expert done_frame={done_frame} by "
            f"{abs(env_done_frame - done_frame)} frames; possible label noise.",
            stacklevel=2,
        )

    # Episode-level metadata stamped onto every row (-1 sentinel for "env never signalled")
    env_done_sentinel = env_done_frame if env_done_frame is not None else -1
    for row in rows:
        row["env_done_frame"] = int(env_done_sentinel)

    return rows


# ---------------------------------------------------------------------------
# Phase B0 generator: scene + instruction + (optional) wrong segment + DART
# ---------------------------------------------------------------------------
from experiments.action_primitives.scene import Scene, generate_scene, is_adversarial
from experiments.action_primitives.instructions import generate_instruction, maybe_typo
from experiments.action_primitives.expert import (
    InstructionAwareLClickExpert,
)
from experiments.action_primitives.recovery import (
    DART_PROB_DEFAULT,
    DART_SIGMA_DEFAULT,
    apply_dart_noise,
    generate_wrong_segment,
    sample_wrong_segment_type,
)


SCENARIO_ERROR_RATE_DEFAULT: float = 0.18  # ~15-20%


def _b0_action_to_row(a: Action) -> dict:
    return {
        "action_dx": float(a.dx),
        "action_dy": float(a.dy),
        "action_click": int(a.click),
        "action_scroll": float(a.scroll),
        "action_key_events": a.key_events.astype(np.int8).tolist(),
    }


def _b0_row(
    *,
    episode_id: int,
    frame_idx: int,
    obs: dict,
    action: Action,
    instruction: str,
    instr_result,
    scene: Scene,
    tempo: str,
    primitive: str,
    is_scenario_error: bool,
    scenario_type: str,
    k_wrong_frames: int,
    is_dart_noisy: bool,
    loss_mask: int,
    done_frame: int | None,
    clean_action: Action | None = None,
) -> dict:
    """Construct one parquet row with all B0 fields."""
    target = scene.buttons[instr_result.target_button_id]
    row = {
        "episode_id": int(episode_id),
        "frame_idx": int(frame_idx),
        "image_bytes": _image_to_jpeg_bytes(obs["image"]),
        "primitive_type": primitive,
        "tempo": tempo,
        "instruction": instruction,
        "target_button_id": int(instr_result.target_button_id),
        "target_bbox_x": int(target.x),
        "target_bbox_y": int(target.y),
        "target_bbox_w": int(target.w),
        "target_bbox_h": int(target.h),
        "n_buttons": int(len(scene.buttons)),
        "composite_tier": int(instr_result.composite_tier),
        "is_adversarial": int(is_adversarial(scene)),
        "is_scenario_error": int(is_scenario_error),
        "scenario_type": scenario_type,
        "k_wrong_frames": int(k_wrong_frames),
        "is_dart_noisy_frame": int(is_dart_noisy),
        "loss_mask": int(loss_mask),
        "done_gt": 1 if (done_frame is not None and frame_idx >= done_frame) else 0,
    }
    # action_applied (in env) — what the env saw (may be DART-noisy or wrong-segment).
    row.update(_b0_action_to_row(action))
    # action_label (training target — clean expert action). On wrong-segment
    # frames there is no clean label; we use action zeros + loss_mask=0.
    if clean_action is not None:
        row["action_label_dx"] = float(clean_action.dx)
        row["action_label_dy"] = float(clean_action.dy)
        row["action_label_click"] = int(clean_action.click)
        row["action_label_scroll"] = float(clean_action.scroll)
        row["action_label_key_events"] = clean_action.key_events.astype(np.int8).tolist()
    else:
        # Placeholder for wrong-segment frames (loss_mask=0 means these aren't trained on).
        row["action_label_dx"] = 0.0
        row["action_label_dy"] = 0.0
        row["action_label_click"] = 0
        row["action_label_scroll"] = 0.0
        row["action_label_key_events"] = np.full(NUM_KEYS, 2, dtype=np.int8).tolist()
    # Proprio
    row.update(_proprio_to_row(obs["proprio"]))
    return row


def generate_one_b0_episode(
    episode_id: int,
    seed: int,
    primitive: str = "lclick",
    tempo: str | None = None,
    max_frames: int = ENV.max_frames_lclick,
    scenario_error_rate: float = SCENARIO_ERROR_RATE_DEFAULT,
    force_scenario_error: bool | None = None,
    force_k_frames: int | None = None,
    dart_p: float = DART_PROB_DEFAULT,
    dart_sigma: float = DART_SIGMA_DEFAULT,
) -> list[dict]:
    """Generate one B0 episode: scene + instruction + (optional) wrong segment + DART + clean expert.

    Args:
        episode_id: integer episode id stamped on every row.
        seed: master RNG seed (also forwarded to env / expert).
        primitive: primitive label stamped on every row (default "lclick").
        tempo: optional tempo override; if None, sampled from TEMPO_CHOICES.
        max_frames: total frame budget (wrong-segment frames count toward this).
        scenario_error_rate: probability of running a wrong-segment recovery episode.
        force_scenario_error: if not None, override the random sampling.
        force_k_frames: if not None, override the wrong-segment length sampling.
        dart_p: per-frame DART noise probability.
        dart_sigma: DART noise stddev (in pixels) on dx/dy.
    """
    rng = np.random.default_rng(seed)
    tempo = tempo if tempo is not None else str(rng.choice(list(TEMPO_CHOICES)))

    # Generate scene + instruction
    scene = generate_scene(rng=rng)
    instr_result = generate_instruction(scene, rng=rng)
    instruction = maybe_typo(instr_result.instruction, rng=rng)

    # Decide scenario-error
    if force_scenario_error is None:
        is_scenario = bool(rng.random() < scenario_error_rate)
    else:
        is_scenario = bool(force_scenario_error)

    # Build environment grounded on the chosen target button
    env = LClickEnv(
        scene=scene,
        target_button_id=instr_result.target_button_id,
        seed=seed,
    )
    obs, info = env.reset(seed=seed)
    cursor_xy = info["cursor_xy"]

    rows: list[dict] = []
    frame_idx = 0
    done_frame: int | None = None
    env_done_frame: int | None = None

    # If scenario-error, run wrong segment first (loss_mask=0; no clean label)
    k_wrong = 0
    scenario_type = "none"
    if is_scenario:
        seg_type = sample_wrong_segment_type(rng=rng)
        if force_k_frames is not None:
            k_wrong = int(force_k_frames)
        else:
            k_wrong = int(rng.integers(5, 16))  # [5, 15] inclusive
        scenario_type = seg_type
        seg = generate_wrong_segment(
            scene=scene,
            target_button_id=instr_result.target_button_id,
            cursor_xy=cursor_xy,
            segment_type=seg_type,
            k_frames=k_wrong,
            rng=rng,
        )
        for action in seg.actions:
            if frame_idx >= max_frames:
                break
            row = _b0_row(
                episode_id=episode_id,
                frame_idx=frame_idx,
                obs=obs,
                action=action,
                instruction=instruction,
                instr_result=instr_result,
                scene=scene,
                tempo=tempo,
                primitive=primitive,
                is_scenario_error=True,
                scenario_type=scenario_type,
                k_wrong_frames=k_wrong,
                is_dart_noisy=False,
                loss_mask=0,
                done_frame=done_frame,
                clean_action=None,  # wrong-segment frames have no clean label
            )
            rows.append(row)
            obs, env_done, info = env.step(action)
            if env_done and env_done_frame is None:
                env_done_frame = frame_idx
            frame_idx += 1

    # Build instruction-aware expert from current cursor state (post-wrong-segment if any).
    cursor_xy = info["cursor_xy"]
    expert_cfg = LClickExpertConfig(tempo=tempo, seed=seed + 1)
    expert = InstructionAwareLClickExpert(
        cfg=expert_cfg,
        scene=scene,
        target_button_id=instr_result.target_button_id,
        cursor_xy=cursor_xy,
    )
    expert_iter = iter(expert)

    # Run clean expert with DART noise (loss_mask=1 throughout).
    # apply_dart_noise internally skips click frames (click != 0), so we can
    # call it unconditionally here.
    while frame_idx < max_frames:
        if done_frame is None:
            try:
                clean_action = next(expert_iter)
            except StopIteration:
                done_frame = frame_idx
                clean_action = _noop_action()
        else:
            clean_action = _noop_action()

        applied_action, was_dart_noisy = apply_dart_noise(
            clean_action, rng=rng, p=dart_p, sigma=dart_sigma,
        )

        row = _b0_row(
            episode_id=episode_id,
            frame_idx=frame_idx,
            obs=obs,
            action=applied_action,
            instruction=instruction,
            instr_result=instr_result,
            scene=scene,
            tempo=tempo,
            primitive=primitive,
            is_scenario_error=is_scenario,
            scenario_type=scenario_type,
            k_wrong_frames=k_wrong,
            is_dart_noisy=was_dart_noisy,
            loss_mask=1,
            done_frame=done_frame,
            clean_action=clean_action,
        )
        rows.append(row)
        obs, env_done, info = env.step(applied_action)
        if env_done and env_done_frame is None:
            env_done_frame = frame_idx
        frame_idx += 1

    # Stamp env_done_frame onto all rows (-1 sentinel for "never signalled")
    sentinel = env_done_frame if env_done_frame is not None else -1
    for row in rows:
        row["env_done_frame"] = int(sentinel)

    return rows
