"""Output heads: unpooled flatten(16 queries × 768) → 7 action heads.

B0: 21+21+3+3+21+231+1 = 301 logits/frame (click split into click_left + click_right,
each 3-way {idle, press, release}); unpooled trades ~3.7M params for no information
loss vs pooling.

B0 attempt 2: AuxTargetHead added — predicts target's spatial grid cell from
the same flattened-query state, training-only (not part of action output).
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


class AuxTargetHead(nn.Module):
    """Auxiliary target-grid-cell classifier (B0 attempt 2 — A3 redesigned).

    Predicts which spatial grid cell (in B0_POSITION_GRID = 3×3 = 9 cells) the
    instruction-specified target lives in, from the same flattened query
    representation that the action heads read. Training-only — gradient-pump
    that forces the trunk to encode "which patch is the target" in its features
    (which the motor heads then benefit from).

    Why grid cell instead of `target_button_id` (the original A3 design):
    `target_button_id` is just the RNG-shuffled index in scene.buttons, not a
    semantic position slot. Predicting it teaches the head RNG-permutation
    priors. Predicting grid cell is position-grounded and consistent across
    episodes.

    Why flattened query state instead of pooled (mean over queries):
    Pooled discards per-query information that the action heads use; matching
    their flattened input keeps the auxiliary signal on the same feature
    surface as the motor heads, which is what we want to share gradient with.

    Param count: hidden_dim=256, flat=12288, n_cells=9 → 12288×256 + 256×9
    = ~3.15M (rounded). Modest fraction of the 33.6M trainable budget.
    """

    def __init__(self, n_queries: int, d_model: int, n_cells: int, hidden_dim: int = 256) -> None:
        super().__init__()
        flat_dim = n_queries * d_model
        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_cells),
        )

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """queries: (B, K, d) → (B, n_cells) logits."""
        flat = queries.flatten(start_dim=1)
        return self.net(flat)
