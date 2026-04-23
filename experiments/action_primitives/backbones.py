"""SigLIP2 naflex vision + text tower loader for Experiment 6.

Variable-patch vision encoder (max_num_patches=256 default from Q6, Q29).
Text tower is frozen and used for instruction caching per Q15.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor

from experiments.action_primitives.config import MODEL


@dataclass
class NaflexOutput:
    """Output of a naflex vision forward pass."""

    patch_embeds: torch.Tensor       # (B, N_patches, d_model)
    attention_mask: torch.Tensor     # (B, N_patches), 1 = real, 0 = pad
    spatial_shapes: torch.Tensor     # (B, 2), (h_patches, w_patches) per sample


class SigLIP2Naflex(nn.Module):
    """Frozen SigLIP2-B-naflex vision + text towers (LoRA added later via peft)."""

    def __init__(self, max_num_patches: int = MODEL.max_num_patches) -> None:
        super().__init__()
        self.max_num_patches = max_num_patches
        self.processor = AutoProcessor.from_pretrained(MODEL.vision_model)
        self.model = AutoModel.from_pretrained(MODEL.vision_model)
        # Freeze text tower per Q15
        for p in self.model.text_model.parameters():
            p.requires_grad = False

    def encode_image(self, images: list) -> NaflexOutput:
        """Run vision tower on a list of PIL Images.

        Returns patch_embeds (B, N, d), attention_mask (B, N), spatial_shapes (B, 2).
        """
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            max_num_patches=self.max_num_patches,
        )
        # Move to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # vision_model expects 'attention_mask', processor emits 'pixel_attention_mask'
        vision_inputs = {
            "pixel_values": inputs["pixel_values"],
            "attention_mask": inputs["pixel_attention_mask"],
            "spatial_shapes": inputs["spatial_shapes"],
        }
        out = self.model.vision_model(**vision_inputs)
        # last_hidden_state: (B, N_patches, hidden); pixel_attention_mask: (B, N_patches)
        return NaflexOutput(
            patch_embeds=out.last_hidden_state,
            attention_mask=inputs["pixel_attention_mask"],
            spatial_shapes=inputs["spatial_shapes"],
        )

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Cache-friendly instruction encoder. Returns (B, T, d) text tokens."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = self.model.text_model(**inputs)
        return out.last_hidden_state  # (B, T, d)
