"""ACT (Action Chunking with Transformers) model for Experiment 3.

Architecture:
- Vision backbone encodes the image into 49 spatial tokens.
- FiLM (Feature-wise Linear Modulation) conditions vision tokens on
  proprioception (cursor position + mouse state + keys held) via
  scale/shift before positional encoding.
- A transformer encoder fuses the FiLM-modulated vision tokens with a
  proprioception token (50 tokens total).
- A transformer decoder attends to encoder memory and produces chunk_size
  action predictions in parallel.
- dx/dy are predicted as logits over 49 discrete bins (no CVAE, no tanh
  regression). Multimodality is handled by the bin distribution.
- Mouse and keys are independent binary outputs (sigmoid/BCE).
"""

import math

import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .backbones import build_backbone
from .config import ACTION, CHUNK, MODEL, NUM_BINS


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
    """Action Chunking with Transformers for multi-task desktop interaction.

    Predicts chunk_size timesteps of (dx_logits, dy_logits, mouse, keys, pad)
    actions from a single image observation and proprioceptive state.

    Key design choices:
    - No CVAE: discrete bins replace the variational latent for multimodality.
    - FiLM conditioning: proprioception generates per-channel scale/shift
      applied to vision tokens before positional encoding.
    - 49-bin classification for dx/dy: replaces tanh regression.
    - Pre-LN (norm_first=True) on all transformer layers for stable training.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        chunk_size: int = CHUNK.default_chunk_size,
        proprio_dim: int = MODEL.proprio_dim,
    ) -> None:
        super().__init__()
        d_model = MODEL.d_model

        self.chunk_size = chunk_size
        self.d_model = d_model

        # 1. Vision backbone
        self.backbone = build_backbone(backbone_name)

        # 2. FiLM conditioning: proprio -> scale/shift for vision tokens
        self.film_net = nn.Sequential(
            nn.Linear(proprio_dim, MODEL.film_hidden_dim),
            nn.ReLU(),
            nn.Linear(MODEL.film_hidden_dim, d_model * 2),
        )

        # 3. Proprioception projection + learnable position embedding
        self.proprio_proj = nn.Linear(proprio_dim, d_model)
        self.proprio_pos_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 4. Main encoder: fuses vision (49 tokens) + proprio (1 token)
        self.vision_pos_enc = SinusoidalPositionEncoding(d_model)
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=MODEL.nheads,
                dim_feedforward=MODEL.dim_feedforward,
                dropout=MODEL.dropout,
                activation="relu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=MODEL.encoder_layers,
        )

        # 5. Decoder
        self.query_embed = nn.Embedding(chunk_size, d_model)
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=MODEL.nheads,
                dim_feedforward=MODEL.dim_feedforward,
                dropout=MODEL.dropout,
                activation="relu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=MODEL.decoder_layers,
        )
        self.decoder_pos_enc = SinusoidalPositionEncoding(d_model)

        # 6. Action heads
        # dx/dy: 49-bin classification (no tanh — logits fed to cross-entropy)
        self.head_dx = nn.Linear(d_model, NUM_BINS)
        self.head_dy = nn.Linear(d_model, NUM_BINS)
        self.head_mouse = nn.Linear(d_model, 1)                 # sigmoid (BCE)
        self.head_keys = nn.Linear(d_model, ACTION.num_keys)    # 43 independent sigmoids (BCE)
        self.head_pad = nn.Linear(d_model, 1)                   # sigmoid (BCE)

    def forward(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            images: (B, 3, 224, 224)
            proprio: (B, proprio_dim)  — 46-dim: 2 cursor + 1 mouse + 43 keys

        Returns:
            dict with:
                dx_logits:   (B, chunk, 49) — raw logits over displacement bins
                dy_logits:   (B, chunk, 49) — raw logits over displacement bins
                mouse_left:  (B, chunk)     — raw logit for left-click
                keys_held:   (B, chunk, 43) — raw logits per key (independent BCE)
                pad_logits:  (B, chunk)     — raw logit for padding mask
        """
        B = images.size(0)

        # Vision tokens from backbone
        vision_tokens = self.backbone(images)  # (B, 49, d)

        # FiLM: modulate vision tokens with proprio BEFORE positional encoding
        film_params = self.film_net(proprio)           # (B, d*2)
        gamma, beta = film_params.chunk(2, dim=-1)     # each (B, d)
        gamma = 1.0 + gamma.unsqueeze(1)               # (B, 1, d) — centered at 1
        beta = beta.unsqueeze(1)                       # (B, 1, d)
        vision_tokens = gamma * vision_tokens + beta

        # Positional encoding AFTER FiLM
        vision_tokens = self.vision_pos_enc(vision_tokens)

        # Proprioception token
        proprio_tok = (
            self.proprio_proj(proprio).unsqueeze(1) + self.proprio_pos_embed
        )  # (B, 1, d)

        # Main encoder: vision (49) + proprio (1) = 50 tokens
        encoder_input = torch.cat([vision_tokens, proprio_tok], dim=1)
        memory = self.encoder(encoder_input)

        # Decoder
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        queries = self.decoder_pos_enc(queries)
        decoded = self.decoder(queries, memory)  # (B, chunk, d)

        # Action heads — dx/dy are logits over 49 bins (no tanh!)
        dx_logits = self.head_dx(decoded)                    # (B, chunk, 49)
        dy_logits = self.head_dy(decoded)                    # (B, chunk, 49)
        mouse = self.head_mouse(decoded).squeeze(-1)         # (B, chunk)
        keys = self.head_keys(decoded)                       # (B, chunk, 43)
        pad = self.head_pad(decoded).squeeze(-1)             # (B, chunk)

        return {
            "dx_logits": dx_logits,
            "dy_logits": dy_logits,
            "mouse_left": mouse,
            "keys_held": keys,
            "pad_logits": pad,
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
