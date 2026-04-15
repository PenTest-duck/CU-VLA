"""All hyperparameters for Experiment 5: Mini Text Editor."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Root directory of this experiment (experiments/mini_editor/)
_EXP_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Key indices for the 53-key physical Mac keyboard held-state vector
# ---------------------------------------------------------------------------

# A-Z  ->  0..25
KEY_A, KEY_B, KEY_C, KEY_D, KEY_E = 0, 1, 2, 3, 4
KEY_F, KEY_G, KEY_H, KEY_I, KEY_J = 5, 6, 7, 8, 9
KEY_K, KEY_L, KEY_M, KEY_N, KEY_O = 10, 11, 12, 13, 14
KEY_P, KEY_Q, KEY_R, KEY_S, KEY_T = 15, 16, 17, 18, 19
KEY_U, KEY_V, KEY_W, KEY_X, KEY_Y = 20, 21, 22, 23, 24
KEY_Z = 25

# Digits 0-9  ->  26..35 (number row)
KEY_0, KEY_1, KEY_2, KEY_3, KEY_4 = 26, 27, 28, 29, 30
KEY_5, KEY_6, KEY_7, KEY_8, KEY_9 = 31, 32, 33, 34, 35

# Symbol keys  ->  36..46
KEY_BACKTICK = 36    # `  shifted: ~
KEY_MINUS = 37       # -  shifted: _
KEY_EQUALS = 38      # =  shifted: +
KEY_LBRACKET = 39    # [  shifted: {
KEY_RBRACKET = 40    # ]  shifted: }
KEY_BACKSLASH = 41   # \  shifted: |
KEY_SEMICOLON = 42   # ;  shifted: :
KEY_APOSTROPHE = 43  # '  shifted: "
KEY_COMMA = 44       # ,  shifted: <
KEY_PERIOD = 45      # .  shifted: >
KEY_SLASH = 46       # /  shifted: ?

# Modifiers
KEY_LSHIFT = 47
KEY_RSHIFT = 48

# Special keys
KEY_SPACE = 49
KEY_DELETE = 50      # Mac "Delete" (backspace function)
KEY_RETURN = 51
KEY_TAB = 52

NUM_KEYS = 53

# Proprio vector: cursor_xy (2) + mouse_left (1) + keys_held (53) = 56
PROPRIO_DIM = 56

# ---------------------------------------------------------------------------
# Shift mapping tables
# ---------------------------------------------------------------------------

# key_index → character produced WITHOUT shift held
UNSHIFTED_CHARS: dict[int, str] = {}
# key_index → character produced WITH shift held
SHIFTED_CHARS: dict[int, str] = {}

# Letters: unshifted → lowercase, shifted → uppercase
for _i in range(26):
    UNSHIFTED_CHARS[_i] = chr(ord("a") + _i)
    SHIFTED_CHARS[_i] = chr(ord("A") + _i)

# Digits: unshifted → digit, shifted → symbol
_DIGIT_SHIFTED = ")!@#$%^&*("  # shift+0 through shift+9
for _i in range(10):
    UNSHIFTED_CHARS[26 + _i] = str(_i)
    SHIFTED_CHARS[26 + _i] = _DIGIT_SHIFTED[_i]

# Symbol keys: unshifted → symbol, shifted → symbol
_SYMBOL_PAIRS = [
    (KEY_BACKTICK, "`", "~"),
    (KEY_MINUS, "-", "_"),
    (KEY_EQUALS, "=", "+"),
    (KEY_LBRACKET, "[", "{"),
    (KEY_RBRACKET, "]", "}"),
    (KEY_BACKSLASH, "\\", "|"),
    (KEY_SEMICOLON, ";", ":"),
    (KEY_APOSTROPHE, "'", '"'),
    (KEY_COMMA, ",", "<"),
    (KEY_PERIOD, ".", ">"),
    (KEY_SLASH, "/", "?"),
]
for _idx, _unsh, _sh in _SYMBOL_PAIRS:
    UNSHIFTED_CHARS[_idx] = _unsh
    SHIFTED_CHARS[_idx] = _sh

# Special keys produce characters too
UNSHIFTED_CHARS[KEY_SPACE] = " "
SHIFTED_CHARS[KEY_SPACE] = " "
UNSHIFTED_CHARS[KEY_RETURN] = "\n"
SHIFTED_CHARS[KEY_RETURN] = "\n"
UNSHIFTED_CHARS[KEY_TAB] = "\t"
SHIFTED_CHARS[KEY_TAB] = "\t"

# Reverse mapping: character → (key_index, needs_shift)
CHAR_TO_KEY: dict[str, tuple[int, bool]] = {}
for _idx, _ch in UNSHIFTED_CHARS.items():
    if _ch not in CHAR_TO_KEY:
        CHAR_TO_KEY[_ch] = (_idx, False)
for _idx, _ch in SHIFTED_CHARS.items():
    if _ch not in CHAR_TO_KEY:
        CHAR_TO_KEY[_ch] = (_idx, True)

# The set of all typeable characters (printable ASCII 0x20-0x7E)
TYPEABLE_CHARS: frozenset[str] = frozenset(CHAR_TO_KEY.keys())

# ---------------------------------------------------------------------------
# Opposite-hand shift convention for expert
# Left-hand keys use RShift, right-hand keys use LShift
# ---------------------------------------------------------------------------

# Left-hand keys (use RShift): A-G, 1-5, backtick, tab, and left-side symbols
_LEFT_HAND_KEYS = frozenset([
    KEY_A, KEY_B, KEY_C, KEY_D, KEY_E, KEY_F, KEY_G,
    KEY_Q, KEY_R, KEY_S, KEY_T, KEY_W, KEY_X, KEY_Z,
    KEY_1, KEY_2, KEY_3, KEY_4, KEY_5,
    KEY_BACKTICK, KEY_TAB,
])


def shift_key_for(key_index: int) -> int:
    """Return KEY_RSHIFT for left-hand keys, KEY_LSHIFT for right-hand keys."""
    return KEY_RSHIFT if key_index in _LEFT_HAND_KEYS else KEY_LSHIFT


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def char_to_key_action(ch: str) -> tuple[int, bool]:
    """Map a typeable ASCII character to (key_index, needs_shift).

    Raises ValueError for unsupported characters.
    """
    if len(ch) != 1:
        raise ValueError(f"Expected single character, got {ch!r}")
    result = CHAR_TO_KEY.get(ch)
    if result is None:
        raise ValueError(f"Unsupported character: {ch!r} (ord={ord(ch)})")
    return result


def key_index_to_char(idx: int, shifted: bool = False) -> str | None:
    """Convert a key index to the character it produces, or None."""
    table = SHIFTED_CHARS if shifted else UNSHIFTED_CHARS
    return table.get(idx)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvConfig:
    window_w: int = 640
    window_h: int = 480
    obs_w: int = 512
    obs_h: int = 384
    font_size: int = 20
    control_hz: int = 30
    margin: int = 20
    bg_color: tuple[int, int, int] = (255, 255, 255)
    border_color: tuple[int, int, int] = (200, 200, 200)
    border_width: int = 1
    selection_color: tuple[int, int, int] = (51, 153, 255)
    cursor_color: tuple[int, int, int] = (0, 0, 0)
    cursor_blink_hz: float = 2.0
    mouse_cursor_color: tuple[int, int, int] = (40, 40, 40)
    mouse_cursor_size: int = 12
    max_steps: int = 300


@dataclass(frozen=True)
class ActionConfig:
    max_delta_px: float = 50.0
    num_keys: int = NUM_KEYS
    num_bins_per_side: int = 24  # 24 negative + 1 zero + 24 positive = 49 total
    bin_alpha: float = 3.0       # exponential curve parameter


@dataclass(frozen=True)
class ExpertConfig:
    # Fitts's Law base parameters
    fitts_a: float = 0.05
    fitts_b: float = 0.15
    noise_std: float = 2.0
    pause_min: int = 2
    pause_max: int = 5
    # Mouse trajectory variance
    curvature_frac_lo: float = 0.05
    curvature_frac_hi: float = 0.15
    overshoot_prob: float = 0.3
    overshoot_px_lo: float = 5.0
    overshoot_px_hi: float = 20.0
    overshoot_pause_lo: int = 2
    overshoot_pause_hi: int = 5
    speed_noise_frac: float = 0.15   # ±15% per-frame velocity noise
    jitter_px: float = 1.5           # per-frame micro-jitter σ
    # Click timing variance
    click_dwell_lo: int = 2
    click_dwell_hi: int = 8
    click_duration_lo: int = 1
    click_duration_hi: int = 4
    click_scatter_px: float = 4.0
    # Typing rhythm variance
    typing_speed_lo: float = 0.7
    typing_speed_hi: float = 1.3
    iki_common_lo: int = 1           # inter-key interval for common bigrams
    iki_common_hi: int = 2
    iki_uncommon_lo: int = 3
    iki_uncommon_hi: int = 5
    iki_space_lo: int = 2
    iki_space_hi: int = 4
    shift_lead_lo: int = 1
    shift_lead_hi: int = 3
    shift_lag_lo: int = 0
    shift_lag_hi: int = 2
    micro_pause_prob: float = 0.05
    micro_pause_lo: int = 3
    micro_pause_hi: int = 8
    # Phase transition pauses
    post_click_pause_lo: int = 3
    post_click_pause_hi: int = 10
    post_select_pause_lo: int = 2
    post_select_pause_hi: int = 8
    reading_pause_lo: int = 0
    reading_pause_hi: int = 15
    post_arrive_pause_lo: int = 1
    post_arrive_pause_hi: int = 5


@dataclass(frozen=True)
class TrainConfig:
    num_episodes: int = 10000
    batch_size: int = 1024
    lr: float = 3e-4  # conservative 1.5x from 2e-4 at batch=512 (monitor grad norms)
    weight_decay: float = 1e-4
    epochs: int = 100
    early_stop_patience: int = 15
    val_fraction: float = 0.2
    bin_smooth_sigma: float = 1.5
    grad_clip_norm: float = 100.0
    warmup_epochs: int = 10  # doubled from 5 to match warmup steps at 2x batch
    use_amp: bool = True


@dataclass(frozen=True)
class DataConfig:
    num_episodes: int = 10000
    num_shards: int = 10
    jpeg_quality: int = 95
    output_dir: str = str(_EXP_DIR / "data")


@dataclass(frozen=True)
class EvalConfig:
    num_episodes: int = 200
    max_steps_per_episode: int = 300


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256
    nheads: int = 4
    encoder_layers: int = 4
    decoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    film_hidden_dim: int = 256
    proprio_dim: int = 56  # cursor_xy(2) + mouse_left(1) + keys_held(53)
    obs_h: int = 288       # model input height (resized from 384)
    obs_w: int = 384       # model input width (resized from 512)
    vision_grid_h: int = 9   # ResNet18 feature map height at 288px input
    vision_grid_w: int = 12  # ResNet18 feature map width at 384px input
    backbone_feature_dim: int = 512  # ResNet18 layer4 channels


@dataclass(frozen=True)
class ChunkConfig:
    default_chunk_size: int = 10
    ensemble_decay: float = 0.2
    key_decay: float = 0.8  # faster decay for keys to prevent stale presses


@dataclass(frozen=True)
class FocalLossConfig:
    gamma: float = 2.0
    alpha: float = 0.75  # positive class weight for key presses


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

ENV = EnvConfig()
ACTION = ActionConfig()
EXPERT = ExpertConfig()
TRAIN = TrainConfig()
DATA = DataConfig()
EVAL_CFG = EvalConfig()
MODEL = ModelConfig()
CHUNK = ChunkConfig()
FOCAL = FocalLossConfig()

# ---------------------------------------------------------------------------
# Bin-related constants (computed once at import time)
# ---------------------------------------------------------------------------


def build_bin_centers() -> np.ndarray:
    """Precompute 49 exponential bin centers in pixel space.

    Returns: (49,) array: [neg_24, ..., neg_1, 0, pos_1, ..., pos_24]
    """
    n = ACTION.num_bins_per_side
    alpha = ACTION.bin_alpha
    max_px = ACTION.max_delta_px
    i = np.arange(1, n + 1, dtype=np.float64)
    pos = (np.exp(alpha * i / n) - 1) / (np.exp(alpha) - 1) * max_px
    return np.concatenate([-pos[::-1], [0.0], pos]).astype(np.float32)


BIN_CENTERS = build_bin_centers()  # (49,)
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

# Common bigrams for typing rhythm variance
COMMON_BIGRAMS: frozenset[str] = frozenset([
    "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd",
    "ti", "es", "or", "te", "of", "ed", "is", "it", "al", "ar",
    "st", "to", "nt", "ng", "se", "ha", "as", "ou", "io", "le",
])
