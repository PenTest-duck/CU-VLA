"""Full ACT model for Experiment 6 Phase A and B0.

Wires SigLIP2 naflex vision+text, proprio encoder, action-history encoder,
trunk, output heads, and (B0) the auxiliary target-grid-cell head.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from experiments.action_primitives.backbones import SigLIP2Naflex
from experiments.action_primitives.config import MODEL
from experiments.action_primitives.heads import ActionHeads, AuxTargetHead
from experiments.action_primitives.history import HISTORY_INPUT_DIM, HistoryEncoder
from experiments.action_primitives.proprio import ProprioEncoder
from experiments.action_primitives.trunk import Trunk


@dataclass
class ACTOutput:
    head_logits: dict[str, torch.Tensor]  # {head_name: (B, n_logits)}
    aux_target_logits: Optional[torch.Tensor] = None  # (B, n_cells) or None


class ActionPrimitivesACT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = SigLIP2Naflex()
        self.backbone.apply_lora(
            rank=MODEL.lora_rank,
            text_rank=getattr(MODEL, "text_lora_rank", 0),
            text_target_layers=getattr(MODEL, "text_lora_target_layers", 2),
        )
        self.proprio_enc = ProprioEncoder()
        self.history_enc = HistoryEncoder()
        self.trunk = Trunk()
        self.heads = ActionHeads()
        # Auxiliary target-grid-cell head (B0 attempt 2 — A3 redesigned).
        # Gated by config flag. When disabled, ACTOutput.aux_target_logits=None.
        if getattr(MODEL, "aux_target_enabled", False):
            self.aux_target_head = AuxTargetHead(
                n_queries=MODEL.n_queries,
                d_model=MODEL.d_model,
                n_cells=MODEL.aux_target_n_cells,
                hidden_dim=MODEL.aux_target_hidden_dim,
            )
        else:
            self.aux_target_head = None

    def forward(
        self,
        vision_input,                    # list[PIL.Image] OR dict[str, Tensor] of
                                         # preprocessed (pixel_values,
                                         # pixel_attention_mask, spatial_shapes).
        text_tokens: torch.Tensor,       # (B, T_text, d) — precomputed & cached
        text_mask: torch.Tensor,         # (B, T_text) — 1=real, 0=pad
        proprio: torch.Tensor,           # (B, PROPRIO_DIM)
        action_history: torch.Tensor,    # (B, K, HISTORY_INPUT_DIM)
    ) -> ACTOutput:
        # 1. Vision encoder (trainable via LoRA).
        #    Dispatch: pre-processed tensors (from DataLoader workers) take the
        #    fast path; raw PIL lists (eval, probes) take the convenience
        #    wrapper that runs the processor on the main process.
        if isinstance(vision_input, dict):
            vis = self.backbone.encode_preprocessed(vision_input)
        else:
            vis = self.backbone.encode_image(vision_input)
        # 2. Text tokens (frozen, passed in as cached argument)
        # 3. Proprio → single token
        proprio_tok = self.proprio_enc(proprio)             # (B, 1, d)
        # 4. Action history → K tokens
        history_toks = self.history_enc(action_history)     # (B, K, d)
        # 5. Concat K/V pool: vision + text + proprio + history
        kv = torch.cat([vis.patch_embeds, text_tokens, proprio_tok, history_toks], dim=1)
        # Build key-padding mask (True == pad; torch convention)
        vis_pad = ~vis.attention_mask.bool()
        text_pad = ~text_mask.bool()
        proprio_pad = torch.zeros(proprio.size(0), 1, dtype=torch.bool, device=kv.device)
        history_pad = torch.zeros(proprio.size(0), history_toks.size(1), dtype=torch.bool, device=kv.device)
        kv_mask = torch.cat([vis_pad, text_pad, proprio_pad, history_pad], dim=1)
        # 6. Trunk: queries cross/self-attend
        query_out = self.trunk(kv, kv_key_padding_mask=kv_mask)
        # 7. Heads
        head_logits = self.heads(query_out)
        # 8. Aux target head (B0 attempt 2; None when disabled)
        aux_target_logits = (
            self.aux_target_head(query_out) if self.aux_target_head is not None else None
        )
        return ACTOutput(head_logits=head_logits, aux_target_logits=aux_target_logits)

    def trainable_parameters_summary(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"total={total / 1e6:.1f}M  trainable={trainable / 1e6:.1f}M"
