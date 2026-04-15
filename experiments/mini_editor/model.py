"""ACT (Action Chunking with Transformers) model for Experiment 5: Mini Text Editor.

Architecture:
- Vision backbone (ResNet18) encodes 288x384 image into 108 spatial tokens.
- FiLM (Feature-wise Linear Modulation) conditions vision tokens on
  proprioception (cursor + mouse + 53 keys) via scale/shift before PE.
- Text encoder produces per-token embeddings from NL instruction.
- A transformer encoder fuses vision + proprio + text tokens (~124 total).
- A transformer decoder attends to encoder memory and produces chunk_size
  action predictions in parallel.
- dx/dy are predicted as logits over 49 discrete exponential bins.
- Mouse, 53 keys, and pad are independent binary outputs (sigmoid/BCE).

First V+L+A model: vision (locate words), language (parse edit instruction),
action (multi-step motor sequences at 30 Hz).
"""

import math

import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .backbones import ResNet18Backbone, SinusoidalPositionEncoding2D
from .config import CHUNK, MODEL, NUM_BINS, NUM_KEYS


# ---------------------------------------------------------------------------
# 1-D sinusoidal positional encoding
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ACT model
# ---------------------------------------------------------------------------


class ACT(nn.Module):
    """Action Chunking with Transformers for V+L+A desktop interaction.

    Predicts chunk_size timesteps of (dx_logits, dy_logits, mouse, keys, pad)
    actions from a screenshot, proprioceptive state, and NL instruction.

    Key design choices:
    - No CVAE: discrete bins replace the variational latent for multimodality.
    - FiLM conditioning: proprioception generates per-channel scale/shift
      applied to vision tokens before positional encoding.
    - 49-bin classification for dx/dy: replaces tanh regression.
    - Pre-LN (norm_first=True) on all transformer layers for stable training.
    - Text encoder produces contextual embeddings from NL instructions.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK.default_chunk_size,
        d_model: int = MODEL.d_model,
        proprio_dim: int = MODEL.proprio_dim,
        text_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model

        # 1. Vision backbone: (B,3,288,384) -> (B,108,256)
        self.backbone = ResNet18Backbone()

        # 2. Text encoder
        if text_encoder is not None:
            self.text_encoder = text_encoder
        else:
            # Minimal default for smoke tests (no MobileBERT download)
            from .text_encoder import TextEncoder
            self.text_encoder = TextEncoder(vocab_size=100, d_model=d_model)

        # 3. FiLM conditioning: proprio -> scale/shift for vision tokens
        self.film_net = nn.Sequential(
            nn.Linear(proprio_dim, MODEL.film_hidden_dim),
            nn.ReLU(),
            nn.Linear(MODEL.film_hidden_dim, d_model * 2),
        )

        # 4. 2D PE for vision tokens (applied after FiLM)
        self.vision_pe = SinusoidalPositionEncoding2D(
            d_model, MODEL.vision_grid_h, MODEL.vision_grid_w
        )

        # 5. Proprio projection + learnable position embedding
        self.proprio_proj = nn.Linear(proprio_dim, d_model)
        self.proprio_pos_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 6. 1D sinusoidal PE for text tokens
        self.text_pos_enc = SinusoidalPositionEncoding(d_model)

        # 7. Encoder: fuses vision (108) + proprio (1) + text (L) tokens
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

        # 8. Decoder
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

        # 9. Action heads
        self.head_dx = nn.Linear(d_model, NUM_BINS)       # 49-bin dx
        self.head_dy = nn.Linear(d_model, NUM_BINS)       # 49-bin dy
        self.head_mouse = nn.Linear(d_model, 1)           # binary mouse_left
        self.head_keys = nn.Linear(d_model, NUM_KEYS)     # 53 independent keys
        self.head_pad = nn.Linear(d_model, 1)             # binary pad mask

        # Prior probability init for focal loss (RetinaNet, Lin et al. 2017).
        # Key presses are ~0.2% of all key-frames. Without this, sigmoid(0)=0.5
        # makes focal loss ineffective early on and the keys gradient dominates.
        import math
        _pi = 0.002  # prior positive rate for key presses
        nn.init.constant_(self.head_keys.bias, -math.log((1 - _pi) / _pi))

    def forward(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            images: (B, 3, 288, 384) RGB screenshots.
            proprio: (B, 56) proprioception — cursor_xy(2) + mouse_left(1) + keys(53).
            input_ids: (B, L) token IDs for the NL instruction.
            attention_mask: (B, L) optional, 1 = real token, 0 = padding.

        Returns:
            dict with:
                dx_logits:   (B, chunk, 49) — raw logits over displacement bins
                dy_logits:   (B, chunk, 49) — raw logits over displacement bins
                mouse_left:  (B, chunk)     — raw logit for left-click
                keys_held:   (B, chunk, 53) — raw logits per key (independent BCE)
                pad_logits:  (B, chunk)     — raw logit for padding mask
        """
        B = images.size(0)

        # --- Vision ---
        vision_tokens = self.backbone(images)  # (B, 108, d)

        # FiLM: modulate vision tokens with proprio BEFORE positional encoding
        film_params = self.film_net(proprio)            # (B, d*2)
        gamma, beta = film_params.chunk(2, dim=-1)      # each (B, d)
        gamma = 1.0 + gamma.unsqueeze(1)                # (B, 1, d) — centered at 1
        beta = beta.unsqueeze(1)                        # (B, 1, d)
        vision_tokens = gamma * vision_tokens + beta

        # 2D positional encoding AFTER FiLM
        vision_tokens = self.vision_pe(
            vision_tokens, MODEL.vision_grid_h, MODEL.vision_grid_w
        )

        # --- Text ---
        text_tokens = self.text_encoder(input_ids, attention_mask)  # (B, L, d)
        text_tokens = self.text_pos_enc(text_tokens)  # add 1D PE

        # --- Proprio token ---
        proprio_tok = (
            self.proprio_proj(proprio).unsqueeze(1) + self.proprio_pos_embed
        )  # (B, 1, d)

        # --- Encoder: vision (108) + proprio (1) + text (L) ---
        encoder_input = torch.cat(
            [vision_tokens, proprio_tok, text_tokens], dim=1
        )
        memory = self.encoder(encoder_input)

        # --- Decoder ---
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        queries = self.decoder_pos_enc(queries)
        decoded = self.decoder(queries, memory)  # (B, chunk, d)

        # --- Action heads ---
        dx_logits = self.head_dx(decoded)                     # (B, chunk, 49)
        dy_logits = self.head_dy(decoded)                     # (B, chunk, 49)
        mouse = self.head_mouse(decoded).squeeze(-1)          # (B, chunk)
        keys = self.head_keys(decoded)                        # (B, chunk, 53)
        pad = self.head_pad(decoded).squeeze(-1)              # (B, chunk)

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
