"""ACT (Action Chunking with Transformers) model for Experiment 2.

CVAE encoder produces a latent z during training; at inference z=0.
A transformer encoder fuses vision tokens, proprioception, and latent z.
A transformer decoder attends to the encoder memory and produces
chunk_size action predictions in parallel.
"""

import math

import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .backbones import build_backbone
from .config import ACTION, CHUNK, MODEL


class SinusoidalPositionEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500) -> None:
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) -> (B, L, d_model) with PE added."""
        return x + self.pe[:, : x.size(1), :]


class ACT(nn.Module):
    """Action Chunking with Transformers.

    Predicts chunk_size timesteps of (dx, dy, click, key, pad) actions
    from a single image observation and proprioceptive state.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        chunk_size: int = CHUNK.default_chunk_size,
        proprio_dim: int = 31,
    ) -> None:
        super().__init__()
        d_model = MODEL.d_model
        latent_dim = MODEL.latent_dim
        action_dim = 2 + 1 + ACTION.num_key_classes  # dx, dy, click, keys = 31

        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.d_model = d_model

        # 1. Vision backbone
        self.backbone = build_backbone(backbone_name)

        # 2. Proprioception projection
        self.proprio_proj = nn.Linear(proprio_dim, d_model)

        # 3. Special position embeddings for proprio and latent tokens
        self.special_pos_embed = nn.Embedding(2, d_model)

        # 4. CVAE encoder (used only during training)
        self.encoder_action_proj = nn.Linear(action_dim, d_model)
        self.encoder_proprio_proj = nn.Linear(proprio_dim, d_model)
        self.cls_embed = nn.Embedding(1, d_model)
        self.cvae_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=MODEL.nheads,
                dim_feedforward=MODEL.dim_feedforward,
                dropout=MODEL.dropout,
                activation="relu",
                batch_first=True,
            ),
            num_layers=MODEL.encoder_layers,
        )
        self.cvae_pos_enc = SinusoidalPositionEncoding(d_model)
        self.latent_proj = nn.Linear(d_model, latent_dim * 2)
        self.latent_out_proj = nn.Linear(latent_dim, d_model)

        # 5. Main encoder
        self.vision_pos_enc = SinusoidalPositionEncoding(d_model)
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=MODEL.nheads,
                dim_feedforward=MODEL.dim_feedforward,
                dropout=MODEL.dropout,
                activation="relu",
                batch_first=True,
            ),
            num_layers=MODEL.encoder_layers,
        )

        # 6. Decoder
        self.query_embed = nn.Embedding(chunk_size, d_model)
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=MODEL.nheads,
                dim_feedforward=MODEL.dim_feedforward,
                dropout=MODEL.dropout,
                activation="relu",
                batch_first=True,
            ),
            num_layers=MODEL.decoder_layers,
        )
        self.decoder_pos_enc = SinusoidalPositionEncoding(d_model)

        # 7. Action heads
        self.head_dx = nn.Linear(d_model, 1)
        self.head_dy = nn.Linear(d_model, 1)
        self.head_click = nn.Linear(d_model, 1)
        self.head_key = nn.Linear(d_model, ACTION.num_key_classes)
        self.head_pad = nn.Linear(d_model, 1)

    def _encode_cvae(
        self, proprio: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode actions + proprio into latent z via CVAE.

        Args:
            proprio: (B, proprio_dim)
            actions: (B, chunk_size, action_dim)

        Returns:
            z: (B, latent_dim)
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        B = proprio.size(0)

        cls_token = self.cls_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, 1, d)
        proprio_tok = self.encoder_proprio_proj(proprio).unsqueeze(1)  # (B, 1, d)
        action_toks = self.encoder_action_proj(actions)  # (B, chunk, d)

        sequence = torch.cat([cls_token, proprio_tok, action_toks], dim=1)
        sequence = self.cvae_pos_enc(sequence)

        encoded = self.cvae_encoder(sequence)

        cls_output = encoded[:, 0, :]  # (B, d)
        latent_params = self.latent_proj(cls_output)  # (B, latent_dim * 2)
        mu, logvar = latent_params.chunk(2, dim=-1)  # each (B, latent_dim)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        return z, mu, logvar

    def forward(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            images: (B, 3, 224, 224)
            proprio: (B, proprio_dim)
            actions: (B, chunk_size, action_dim) or None for inference

        Returns:
            dict with dx, dy, click, key_logits, pad_logits, mu, logvar.
        """
        B = images.size(0)

        # Vision tokens from backbone
        vision_tokens = self.backbone(images)  # (B, 49, d)
        vision_tokens = self.vision_pos_enc(vision_tokens)

        # Proprioception token
        proprio_tok = (
            self.proprio_proj(proprio).unsqueeze(1)
            + self.special_pos_embed.weight[0].unsqueeze(0).unsqueeze(0)
        )  # (B, 1, d)

        # Latent z
        if actions is not None:
            z, mu, logvar = self._encode_cvae(proprio, actions)
        else:
            z = torch.zeros(B, self.latent_dim, device=images.device)
            mu = z
            logvar = z

        latent_tok = (
            self.latent_out_proj(z).unsqueeze(1)
            + self.special_pos_embed.weight[1].unsqueeze(0).unsqueeze(0)
        )  # (B, 1, d)

        # Main encoder
        encoder_input = torch.cat(
            [vision_tokens, proprio_tok, latent_tok], dim=1
        )  # (B, 51, d)
        memory = self.encoder(encoder_input)

        # Decoder
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        queries = self.decoder_pos_enc(queries)  # (B, chunk, d)
        decoded = self.decoder(queries, memory)  # (B, chunk, d)

        # Action heads
        dx = (self.head_dx(decoded).squeeze(-1)).tanh() * ACTION.max_delta_px
        dy = (self.head_dy(decoded).squeeze(-1)).tanh() * ACTION.max_delta_px
        click = self.head_click(decoded).squeeze(-1)
        key_logits = self.head_key(decoded)
        pad_logits = self.head_pad(decoded).squeeze(-1)

        return {
            "dx": dx,  # (B, chunk)
            "dy": dy,  # (B, chunk)
            "click": click,  # (B, chunk)
            "key_logits": key_logits,  # (B, chunk, 28)
            "pad_logits": pad_logits,  # (B, chunk)
            "mu": mu,  # (B, 32)
            "logvar": logvar,  # (B, 32)
        }


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.

    Args:
        model: PyTorch module.
        trainable_only: If True, count only parameters with requires_grad.

    Returns:
        Total number of (trainable) parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
