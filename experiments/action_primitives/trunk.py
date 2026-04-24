"""ACT trunk: learnable queries cross-attending to a K/V pool, then self-attending.

Per Q15: 16 queries, 3 alternating cross+self blocks, dim 768, 12 heads, 4x FFN.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from experiments.action_primitives.config import MODEL


class CrossSelfBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int) -> None:
        super().__init__()
        self.norm_q_cross = nn.LayerNorm(d_model)
        self.norm_kv_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        kv_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Cross-attn: queries attend to kv
        q_in = self.norm_q_cross(queries)
        kv_in = self.norm_kv_cross(kv)
        x, _ = self.cross_attn(q_in, kv_in, kv_in, key_padding_mask=kv_key_padding_mask)
        queries = queries + x
        # Self-attn among queries
        q_in = self.norm_self(queries)
        x, _ = self.self_attn(q_in, q_in, q_in)
        queries = queries + x
        # FFN
        x = self.ffn(self.norm_ffn(queries))
        queries = queries + x
        return queries


class Trunk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, MODEL.n_queries, MODEL.d_model))
        nn.init.normal_(self.query_tokens, std=0.02)
        self.blocks = nn.ModuleList([
            CrossSelfBlock(MODEL.d_model, MODEL.n_heads, MODEL.ffn_mult)
            for _ in range(MODEL.n_blocks)
        ])

    def forward(self, kv: torch.Tensor, kv_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """kv: (B, N, d), kv_key_padding_mask: (B, N) with True at pad positions (torch convention).
        Returns (B, K, d) final query states."""
        B = kv.size(0)
        queries = self.query_tokens.expand(B, -1, -1).contiguous()
        for block in self.blocks:
            queries = block(queries, kv, kv_key_padding_mask)
        return queries
