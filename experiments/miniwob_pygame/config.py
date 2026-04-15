"""All hyperparameters for Experiment 3: MiniWoB-Pygame multi-task suite."""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Key indices for the 43-key multi-binary held-state vector
# ---------------------------------------------------------------------------

# A-Z  ->  0..25
KEY_A, KEY_B, KEY_C, KEY_D, KEY_E = 0, 1, 2, 3, 4
KEY_F, KEY_G, KEY_H, KEY_I, KEY_J = 5, 6, 7, 8, 9
KEY_K, KEY_L, KEY_M, KEY_N, KEY_O = 10, 11, 12, 13, 14
KEY_P, KEY_Q, KEY_R, KEY_S, KEY_T = 15, 16, 17, 18, 19
KEY_U, KEY_V, KEY_W, KEY_X, KEY_Y = 20, 21, 22, 23, 24
KEY_Z = 25

# Special keys
KEY_SPACE = 26
KEY_ENTER = 27
KEY_BACKSPACE = 28
KEY_TAB = 29

# Digits 0-9  ->  30..39
KEY_0, KEY_1, KEY_2, KEY_3, KEY_4 = 30, 31, 32, 33, 34
KEY_5, KEY_6, KEY_7, KEY_8, KEY_9 = 35, 36, 37, 38, 39

# Modifiers
KEY_CTRL = 40
KEY_SHIFT = 41
KEY_ALT = 42

NUM_KEYS = 43

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def char_to_key_index(ch: str) -> int:
    """Convert a character to its key index.

    A-Z (case insensitive) -> 0-25, 0-9 -> 30-39, space -> 26.
    Raises ValueError for unsupported characters.
    """
    if len(ch) != 1:
        raise ValueError(f"Expected single character, got {ch!r}")
    upper = ch.upper()
    if "A" <= upper <= "Z":
        return ord(upper) - ord("A")  # 0..25
    if "0" <= ch <= "9":
        return 30 + int(ch)  # 30..39
    if ch == " ":
        return KEY_SPACE
    raise ValueError(f"Unsupported character: {ch!r}")


def key_index_to_char(idx: int) -> str | None:
    """Convert a key index back to its printable character, or None."""
    if 0 <= idx <= 25:
        return chr(ord("A") + idx)
    if idx == KEY_SPACE:
        return " "
    if 30 <= idx <= 39:
        return str(idx - 30)
    # ENTER, BACKSPACE, TAB, CTRL, SHIFT, ALT -> non-printable
    return None


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_NAMES: list[str] = [
    "click-target",
    "drag-to-zone",
    "use-slider",
    "type-field",
    "click-sequence",
    "draw-path",
    "highlight-text",
    "drag-sort",
    "form-fill",
    "drag-and-label",
    "scroll-and-click",
    "copy-paste",
]

# ---------------------------------------------------------------------------
# Vocabulary (shared across typing tasks)
# ---------------------------------------------------------------------------

VOCAB: list[str] = [
    "CAT", "DOG", "RED", "BOX", "SUN",
    "CUP", "HAT", "PEN", "MAP", "BUS",
    "FAN", "JAR", "KEY", "LOG", "NET",
    "OWL", "RUG", "TOP", "VAN", "WAX",
]

# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvConfig:
    window_size: int = 400
    obs_size: int = 224
    bg_color: tuple[int, int, int] = (30, 30, 30)
    cursor_color: tuple[int, int, int] = (255, 255, 255)
    cursor_radius: int = 3
    instruction_bar_height: int = 40
    instruction_bg_color: tuple[int, int, int] = (50, 50, 50)
    instruction_font_size: int = 24
    font_size: int = 28
    control_hz: int = 30
    shape_colors: tuple[tuple[int, int, int], ...] = (
        (220, 60, 60),    # red
        (60, 120, 220),   # blue
        (60, 180, 80),    # green
        (240, 180, 40),   # yellow
    )


@dataclass(frozen=True)
class ActionConfig:
    max_delta_px: float = 50.0
    num_keys: int = NUM_KEYS
    num_bins_per_side: int = 24  # 24 negative + 1 zero + 24 positive = 49 total
    bin_alpha: float = 3.0       # exponential curve parameter


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256
    encoder_layers: int = 4
    decoder_layers: int = 7
    nheads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    film_hidden_dim: int = 128
    num_vision_tokens: int = 49
    proprio_dim: int = 46  # 2 (mouse_xy) + 1 (mouse_btn) + 43 (keys_held)
    backbone_feature_dims: dict[str, int] = field(default_factory=lambda: {
        "resnet18": 512,
        "dinov2-vits14": 384,
        "siglip2-base": 768,
    })


@dataclass(frozen=True)
class ChunkConfig:
    default_chunk_size: int = 10
    query_frequency: int = 1
    ensemble_decay: float = 0.01
    key_decay: float = 0.5  # faster decay for key channels in temporal ensemble


@dataclass(frozen=True)
class TrainConfig:
    num_episodes_per_task: int = 5000
    batch_size: int = 256
    lr: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4
    epochs: int = 100
    early_stop_patience: int = 15
    val_fraction: float = 0.2
    loss_weight_mouse: float = 5.0
    loss_weight_keys: float = 5.0
    loss_weight_pad: float = 1.0
    bin_smooth_sigma: float = 1.5  # Gaussian label smoothing σ in bin units
    loss_weight_ev: float = 1.0    # weight for expected-value L1 loss on dx/dy
    grad_clip_norm: float = 100.0
    warmup_epochs: int = 5
    use_amp: bool = True


@dataclass(frozen=True)
class EvalConfig:
    num_episodes: int = 200
    max_steps_per_episode: int = 300
    max_steps_multi: int = 600
    max_steps_long: int = 900


@dataclass(frozen=True)
class ExpertConfig:
    fitts_a: float = 0.05
    fitts_b: float = 0.15
    noise_std: float = 2.0
    pause_min: int = 2
    pause_max: int = 5


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

ENV = EnvConfig()
ACTION = ActionConfig()
MODEL = ModelConfig()
CHUNK = ChunkConfig()
TRAIN = TrainConfig()
EVAL_CFG = EvalConfig()
EXPERT = ExpertConfig()

# ---------------------------------------------------------------------------
# Bin-related constants (computed once at import time)
# ---------------------------------------------------------------------------

import numpy as np


def build_bin_centers() -> np.ndarray:
    """Precompute 49 exponential bin centers in pixel space.

    Returns: (49,) array: [neg_24, ..., neg_1, 0, pos_1, ..., pos_24]
    """
    n = ACTION.num_bins_per_side
    alpha = ACTION.bin_alpha
    max_px = ACTION.max_delta_px
    i = np.arange(1, n + 1, dtype=np.float64)
    pos = (np.exp(alpha * i / n) - 1) / (np.exp(alpha) - 1) * max_px
    centers = np.concatenate([-pos[::-1], [0.0], pos]).astype(np.float32)
    return centers


BIN_CENTERS = build_bin_centers()  # (49,) — shared constant
NUM_BINS = 2 * ACTION.num_bins_per_side + 1  # 49


def build_soft_bin_targets(sigma: float) -> np.ndarray:
    """Precompute Gaussian-smoothed soft targets for each bin.

    Returns: (49, 49) array where row i is the soft target distribution
    when the true bin is i.
    """
    arange = np.arange(NUM_BINS, dtype=np.float64)
    diff = arange[None, :] - arange[:, None]
    soft = np.exp(-0.5 * (diff / sigma) ** 2)
    soft /= soft.sum(axis=1, keepdims=True)
    return soft.astype(np.float32)


SOFT_BIN_TARGETS = build_soft_bin_targets(TRAIN.bin_smooth_sigma)  # (49, 49)
