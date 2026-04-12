"""BaselineCNN — single-step, no-chunking baseline for Experiment 2.

Same task and data as the ACT model, but predicts one step at a time
with a simple 4-conv CNN (no transformer, no chunking, no temporal
ensemble).  If ACT doesn't beat this, chunking isn't helping.

Architecture:
  4 × Conv2d(3×3, stride=2, padding=1) + ReLU  →  channels 3→32→64→64→128
  224×224 input  →  14×14 spatial after 4 stride-2 layers
  Flatten  →  Linear(128*14*14, 256) → ReLU
  4 heads: dx, dy (tanh-scaled), click logit (BCE), key logits (28-class)

~6.5 M parameters.  <3 ms per forward pass on M1 CPU.
"""

import torch
import torch.nn as nn

from .config import ENV, ACTION


class BaselineCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        channels = (32, 64, 64, 128)
        in_ch = 3

        layers: list[nn.Module] = []
        for out_ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        # 224 / (2^4) = 14
        spatial = ENV.obs_size // 16  # 14
        flat_dim = channels[-1] * spatial * spatial  # 128 * 196 = 25088

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
        )

        # Output heads
        self.head_dx = nn.Linear(256, 1)
        self.head_dy = nn.Linear(256, 1)
        self.head_click = nn.Linear(256, 1)
        self.head_key = nn.Linear(256, ACTION.num_key_classes)  # 28

        self._max_delta = ACTION.max_delta_px

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, 224, 224) float32 tensor, normalized to [0, 1].

        Returns:
            dx:          (B, 1)  pixel delta, tanh-scaled to [-max_delta, +max_delta]
            dy:          (B, 1)  pixel delta, tanh-scaled to [-max_delta, +max_delta]
            click_logit: (B, 1)  raw logit for BCE loss
            key_logits:  (B, 28) logits for key classification
        """
        features = self.conv(x)
        features = features.flatten(1)
        features = self.fc(features)

        dx = torch.tanh(self.head_dx(features)) * self._max_delta
        dy = torch.tanh(self.head_dy(features)) * self._max_delta
        click_logit = self.head_click(features)
        key_logits = self.head_key(features)

        return dx, dy, click_logit, key_logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
