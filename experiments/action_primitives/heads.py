"""Output heads: unpooled flatten(16 queries × 768) → 6 action heads.

Per Q16: 21+21+5+21+231+1 = 300 logits/frame; unpooled trades ~3.7M params
for no information loss vs pooling.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from experiments.action_primitives.config import HEAD_LOGITS, MODEL


class ActionHeads(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        flat_dim = MODEL.n_queries * MODEL.d_model  # 16 * 768 = 12288
        # NB: prefix internal submodule names to avoid collision with nn.ModuleDict's
        # built-in .keys() method (one of the head names is "keys").
        self.heads = nn.ModuleDict({
            f"head_{name}": nn.Linear(flat_dim, n_logits)
            for name, n_logits in HEAD_LOGITS.items()
        })

    def forward(self, queries: torch.Tensor) -> dict[str, torch.Tensor]:
        """queries: (B, K, d) → {head_name: (B, n_logits)}."""
        flat = queries.flatten(start_dim=1)  # (B, K*d)
        return {name: self.heads[f"head_{name}"](flat) for name in HEAD_LOGITS}
