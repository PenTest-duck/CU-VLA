"""Shared config for Experiment 6 Phase A.

Matches design decisions from docs/experiments/6-action-primitives.md.
Only L-click-related values populated; remaining primitives added in Phase B.
"""

from dataclasses import dataclass, field
import math

import numpy as np


# ---------- Environment ----------
@dataclass(frozen=True)
class EnvConfig:
    canvas_w: int = 720
    canvas_h: int = 450  # 16:10 full-screen task window (Q7)
    fps: int = 30        # logical 30Hz (Q5, Q16)
    max_frames_lclick: int = 45  # per-primitive window (Q8 amended by Spike E)


ENV = EnvConfig()


# ---------- Action space (Q1, Q3) ----------
# Mouse delta: 21 bins per axis, ±100px (widened 2026-04-23), 10+1+10 exponential
# Zero is bin 10; bins 0-9 are negative (large → small magnitude); bins 11-20 positive
NUM_BINS_MOUSE: int = 21
MOUSE_CAP_PX: float = 100.0  # Q1 amendment 2026-04-23
MOUSE_ALPHA: float = 2.5     # exponential growth factor (Q1)

def _build_exp_bin_centers(num_bins: int, cap: float, alpha: float) -> np.ndarray:
    """Symmetric exponential bin centers, zero at middle. Returns length-num_bins array."""
    half = num_bins // 2  # 10 for num_bins=21
    # geometric series i=1..half, scaled so largest = cap
    raw = alpha ** np.arange(1, half + 1)  # e.g. 2.5, 6.25, ..., 2.5^10
    scaled_pos = raw / raw[-1] * cap        # scale so last = cap
    centers = np.concatenate([-scaled_pos[::-1], np.array([0.0]), scaled_pos])
    return centers.astype(np.float32)

MOUSE_BIN_CENTERS: np.ndarray = _build_exp_bin_centers(NUM_BINS_MOUSE, MOUSE_CAP_PX, MOUSE_ALPHA)

# Click event: 5-way {idle, L_press, L_release, R_press, R_release}
NUM_CLICK_EVENTS: int = 5
CLICK_IDLE, CLICK_L_PRESS, CLICK_L_RELEASE, CLICK_R_PRESS, CLICK_R_RELEASE = range(5)

# Scroll: 21-bin signed, ±20 wheel ticks/frame (symmetric; Q1 notes unchanged in Phase A)
NUM_BINS_SCROLL: int = 21
SCROLL_CAP_TICKS: float = 20.0
SCROLL_BIN_CENTERS: np.ndarray = _build_exp_bin_centers(NUM_BINS_SCROLL, SCROLL_CAP_TICKS, MOUSE_ALPHA)

# Keys: 77 physical keys × 3-way {press, release, idle}
NUM_KEYS: int = 77
KEY_STATE_PRESS, KEY_STATE_RELEASE, KEY_STATE_IDLE = range(3)

# Head sizes
HEAD_LOGITS = {
    "dx": NUM_BINS_MOUSE,        # 21
    "dy": NUM_BINS_MOUSE,        # 21
    "click": NUM_CLICK_EVENTS,   # 5
    "scroll": NUM_BINS_SCROLL,   # 21
    "keys": NUM_KEYS * 3,        # 231
    "done": 1,                   # binary
}
TOTAL_LOGITS: int = sum(HEAD_LOGITS.values())  # 300


# ---------- Proprio (Q15) ----------
# Cursor xy (2) + held keys (77) + held mouse buttons (3) + CapsLock mode (1) = 83
PROPRIO_DIM: int = 83


# ---------- Model (Q15, Q16) ----------
@dataclass(frozen=True)
class ModelConfig:
    vision_model: str = "google/siglip2-base-patch16-naflex"
    max_num_patches: int = 256     # Q6, Q29
    d_model: int = 768             # SigLIP2-B output dim; trunk dim
    n_queries: int = 16            # Q15: learnable "read heads"
    n_blocks: int = 3              # Q15 Trunk A
    n_heads: int = 12
    ffn_mult: int = 4
    action_history_len: int = 8    # Q5, Q19
    action_vec_dim: int = 300      # Q19: per-frame composite action vector size
    lora_rank: int = 8             # Q15
    freeze_text_tower: bool = True # Q15


MODEL = ModelConfig()


# ---------- Loss (Q2, Q3) ----------
@dataclass(frozen=True)
class LossConfig:
    focal_gamma: float = 2.0
    label_smoothing_mouse: float = 0.05
    idle_smoothing_keys: float = 0.05
    # Initial per-head weights will be re-computed from L_i^init at train start (Q2)
    placeholder_weights: dict = field(default_factory=lambda: {
        "dx": 1.0, "dy": 1.0, "click": 1.0, "scroll": 1.0, "keys": 1.0, "done": 1.0,
    })


LOSS = LossConfig()


# ---------- Training (Q20, Q21) ----------
@dataclass(frozen=True)
class TrainConfig:
    macro_batch_episodes: int = 64
    micro_batch_episodes: int = 8        # 8 micro × 8 = 64 macro (Q2/Q8 clarification)
    phase_a_epochs: int = 5              # smaller than Phase B's 20
    lr_trunk: float = 3e-4
    lr_lora: float = 2e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 100              # smaller than Phase B's 500 for smaller run
    cosine_min_lr_frac: float = 0.1
    grad_clip_norm: float = 1.0
    eval_every_steps: int = 200
    ckpt_every_steps: int = 200


TRAIN = TrainConfig()


# ---------- Phase A data (L-click only) ----------
@dataclass(frozen=True)
class PhaseADataConfig:
    primitive: str = "lclick"
    n_episodes: int = 3000
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    # Minimal visual diversity: 3 themes only (full 20-theme system is Phase B)
    themes: tuple = ("flat-modern", "flat-minimal", "dark-mode")


PHASE_A_DATA = PhaseADataConfig()


# ---------- Phase B0 attribute palettes ----------
B0_COLORS: dict[str, tuple[int, int, int]] = {
    "red":    (220, 60,  60),
    "blue":   (60,  120, 230),
    "green":  (70,  180, 90),
    "yellow": (240, 220, 70),
    "orange": (240, 140, 50),
    "purple": (160, 80,  200),
    "pink":   (240, 130, 180),
    "cyan":   (80,  200, 220),
    "white":  (245, 245, 245),
    "black":  (30,  30,  30),
}

# rect: w != h (aspect ratio varies); square: w == h (enforced by generator)
B0_SHAPES: tuple[str, ...] = ("rect", "circle", "triangle", "square", "hexagon")

B0_SIZES: dict[str, tuple[int, int]] = {
    "small":  (30, 50),
    "medium": (60, 90),
    "large":  (100, 140),
}

B0_POSITION_GRID: tuple[int, int] = (3, 3)  # 3 cols × 3 rows of zones

# 6 pastel canvas backgrounds, sampled per episode
B0_BG_COLORS: tuple[tuple[int, int, int], ...] = (
    (245, 245, 248),
    (240, 246, 240),
    (245, 240, 240),
    (240, 245, 250),
    (250, 248, 235),
    (235, 240, 248),
)
