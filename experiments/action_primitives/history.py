"""Action-history encoder (Q19).

Input: per-timestep composite action vectors of shape (B, K=8, 300).
Composite: dx one-hot (21) + dy (21) + click (5) + scroll (21) + keys press-mask (77) + keys release-mask (77) + done (1) + 77 pad zeros.
Wait — the 300 in the design sums to 300. Let's be precise:
  21 + 21 + 5 + 21 + 77 + 77 + 1 = 223 ... +77 zero pad = 300? No; recount.
Design doc Q19 says: 21+21+5+21+154+1 = 223 core dims; we pad to 300 via MLP.
  (154 = 77 press bits + 77 release bits)
We take 223 as the true input width and project 223 → 256 → 768.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from experiments.action_primitives.config import MODEL, NUM_BINS_MOUSE, NUM_BINS_SCROLL, NUM_CLICK_EVENTS, NUM_KEYS


HISTORY_INPUT_DIM: int = (
    NUM_BINS_MOUSE + NUM_BINS_MOUSE      # dx, dy one-hot: 42
    + NUM_CLICK_EVENTS                   # click: 5
    + NUM_BINS_SCROLL                    # scroll one-hot: 21
    + 2 * NUM_KEYS                       # keys press + release mask: 154
    + 1                                  # done bit: 1
)  # = 223


class HistoryEncoder(nn.Module):
    def __init__(self, d_model: int = MODEL.d_model, history_len: int = MODEL.action_history_len) -> None:
        super().__init__()
        self.history_len = history_len
        self.proj = nn.Sequential(
            nn.Linear(HISTORY_INPUT_DIM, 256),
            nn.GELU(),
            nn.Linear(256, d_model),
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, history_len, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """history: (B, K, HISTORY_INPUT_DIM) → (B, K, d_model) with temporal PE."""
        x = self.proj(history)
        return x + self.pos_emb[:, : x.size(1), :]
