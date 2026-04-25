"""Proprio encoder: 83-dim state → single 768-dim token.

Per Q15: proprio is state-as-input integrated as a K/V token in the trunk.
Plain 2-layer MLP with GELU. Not FiLM, not AdaLN (Q15).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from experiments.action_primitives.config import MODEL, PROPRIO_DIM


class ProprioEncoder(nn.Module):
    def __init__(self, d_model: int = MODEL.d_model) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PROPRIO_DIM, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        """proprio: (B, PROPRIO_DIM) float → (B, 1, d_model)."""
        x = self.net(proprio)
        return x.unsqueeze(1)
