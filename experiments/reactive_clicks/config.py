"""All hyperparameters for Experiment 1: Reactive Clicks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    window_size: int = 400
    obs_size: int = 128
    bg_color: tuple[int, int, int] = (30, 30, 30)
    circle_color: tuple[int, int, int] = (255, 0, 0)
    cursor_color: tuple[int, int, int] = (255, 255, 255)
    cursor_radius: int = 3
    circle_radius_min: int = 20
    circle_radius_max: int = 40
    delay_min: float = 0.5
    delay_max: float = 3.0
    control_hz: int = 30


@dataclass(frozen=True)
class ActionConfig:
    max_delta_px: float = 50.0  # max single-step displacement in native coords
    num_btn_classes: int = 3  # 0=no_change, 1=mouse_down, 2=mouse_up


@dataclass(frozen=True)
class ModelConfig:
    conv_channels: tuple[int, ...] = (32, 64, 64, 128)
    fc_dim: int = 256


@dataclass(frozen=True)
class TrainConfig:
    num_episodes: int = 10000
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    early_stop_patience: int = 10  # stop if val loss doesn't improve for N epochs
    val_fraction: float = 0.2
    btn_class_weights: tuple[float, float, float] = (0.1, 5.0, 5.0)


@dataclass(frozen=True)
class EvalConfig:
    num_episodes: int = 200
    max_steps_per_episode: int = 300  # safety cutoff (~10s at 30Hz)


@dataclass(frozen=True)
class ExpertConfig:
    fitts_a: float = 0.05  # intercept (seconds)
    fitts_b: float = 0.15  # slope (seconds per bit)
    noise_std: float = 2.0  # gaussian noise on path (pixels)


ENV = EnvConfig()
ACTION = ActionConfig()
MODEL = ModelConfig()
TRAIN = TrainConfig()
EVAL = EvalConfig()
EXPERT = ExpertConfig()
