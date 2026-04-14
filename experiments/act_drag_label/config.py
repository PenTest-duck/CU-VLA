"""All hyperparameters for Experiment 2: ACT Drag-and-Label."""

from dataclasses import dataclass, field

VOCAB = [
    "CAT", "DOG", "RED", "BOX", "SUN",
    "CUP", "HAT", "PEN", "MAP", "BUS",
    "FAN", "JAR", "KEY", "LOG", "NET",
    "OWL", "RUG", "TOP", "VAN", "WAX",
]

NUM_KEY_CLASSES = 28  # 0=no_key, 1-26=A-Z, 27=space

@dataclass(frozen=True)
class EnvConfig:
    window_size: int = 400
    obs_size: int = 224
    bg_color: tuple[int, int, int] = (30, 30, 30)
    cursor_color: tuple[int, int, int] = (255, 255, 255)
    cursor_radius: int = 10
    shape_width_min: int = 60
    shape_width_max: int = 80
    shape_height: int = 50
    zone_width: int = 90
    zone_height: int = 60
    zone_border_width: int = 3
    font_size: int = 28
    control_hz: int = 30
    shape_colors: tuple[tuple[int, int, int], ...] = (
        (220, 60, 60),    # red
        (60, 120, 220),   # blue
        (60, 180, 80),    # green
    )
    max_shapes: int = 3
    shape_x_min: int = 30
    shape_x_max: int = 170
    zone_x_min: int = 230
    zone_x_max: int = 340

@dataclass(frozen=True)
class ActionConfig:
    max_delta_px: float = 50.0
    num_key_classes: int = NUM_KEY_CLASSES

@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256
    encoder_layers: int = 4
    decoder_layers: int = 7
    nheads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    latent_dim: int = 32
    num_vision_tokens: int = 49
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

@dataclass(frozen=True)
class TrainConfig:
    num_episodes: int = 10000
    batch_size: int = 256
    lr: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4
    epochs: int = 100
    early_stop_patience: int = 15
    val_fraction: float = 0.2
    kl_weight_max: float = 0.1
    kl_anneal_fraction: float = 0.2
    loss_weight_click: float = 5.0
    loss_weight_key: float = 5.0
    loss_weight_pad: float = 1.0
    use_amp: bool = True

@dataclass(frozen=True)
class EvalConfig:
    num_episodes: int = 200
    max_steps_per_episode: int = 300
    max_steps_multi: int = 900

@dataclass(frozen=True)
class ExpertConfig:
    fitts_a: float = 0.05
    fitts_b: float = 0.15
    noise_std: float = 2.0
    pause_min: int = 2
    pause_max: int = 5
    inter_shape_pause_min: int = 1
    inter_shape_pause_max: int = 3

ENV = EnvConfig()
ACTION = ActionConfig()
MODEL = ModelConfig()
CHUNK = ChunkConfig()
TRAIN = TrainConfig()
EVAL_CFG = EvalConfig()
EXPERT = ExpertConfig()
