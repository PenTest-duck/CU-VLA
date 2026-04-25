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
        # Freeze BOTH towers by default. The vision tower is selectively
        # re-enabled via LoRA in apply_lora() (peft.get_peft_model handles
        # the requires_grad bookkeeping for the wrapped layers). If a caller
        # forgets to call apply_lora, this guarantees they don't silently
        # full-finetune the 93 M-param vision base when they build the
        # optimizer. Probes (typing_legibility*) deliberately skip apply_lora
        # and run inference-only — also consistent with this default.
        for p in self.model.vision_model.parameters():
            p.requires_grad = False
        # Freeze text tower per Q15
        for p in self.model.text_model.parameters():
            p.requires_grad = False
        # Detect which kwarg name the vision tower's forward expects for the
        # per-patch mask. transformers renamed `attention_mask` →
        # `pixel_attention_mask` at some point between the versions used locally
        # vs. inside HF Jobs. Pick the right one once to stay robust.
        import inspect
        params = inspect.signature(self.model.vision_model.forward).parameters
        if "pixel_attention_mask" in params:
            self._mask_kwarg = "pixel_attention_mask"
        elif "attention_mask" in params:
            self._mask_kwarg = "attention_mask"
        else:
            raise RuntimeError(
                f"vision_model.forward has neither `pixel_attention_mask` nor "
                f"`attention_mask` in its signature; got params: {list(params)}"
            )

    def apply_lora(self, rank: int = 8) -> None:
        """Apply LoRA adapters to vision tower attention projections (Q15).

        Idempotency-guarded: calling a second time raises RuntimeError rather
        than silently stacking adapters (PEFT emits warnings but continues,
        which is hard to debug). Safe to call exactly once, typically from
        ActionPrimitivesACT.__init__.
        """
        from peft import LoraConfig, PeftModel, get_peft_model

        if isinstance(self.model.vision_model, PeftModel):
            raise RuntimeError(
                "apply_lora() has already been called on this SigLIP2Naflex; "
                "vision_model is already a PeftModel. Refusing to stack adapters."
            )
        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            modules_to_save=[],
            bias="none",
            lora_dropout=0.0,
        )
        # Apply only to vision_model (text stays frozen)
        self.model.vision_model = get_peft_model(self.model.vision_model, lora_cfg)

    def preprocess(self, images: list) -> dict:
        """Run the naflex processor on PIL images; returns CPU tensors.

        Safe to call from a DataLoader worker process — no GPU ops, no
        `self.model` access. Returns a dict of tensors with keys matching
        what the processor emits: `pixel_values`, `pixel_attention_mask`,
        `spatial_shapes`.
        """
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            max_num_patches=self.max_num_patches,
        )
        return {
            "pixel_values": inputs["pixel_values"],
            "pixel_attention_mask": inputs["pixel_attention_mask"],
            "spatial_shapes": inputs["spatial_shapes"],
        }

    def encode_preprocessed(self, batch: dict) -> NaflexOutput:
        """Run the vision tower on an already-preprocessed batch.

        Moves tensors to the model's device (non_blocking for pinned tensors)
        and handles the transformers version-sensitive kwarg rename.
        """
        device = next(self.model.parameters()).device
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        pixel_attention_mask = batch["pixel_attention_mask"].to(device, non_blocking=True)
        spatial_shapes = batch["spatial_shapes"].to(device, non_blocking=True)
        vision_inputs = {
            "pixel_values": pixel_values,
            self._mask_kwarg: pixel_attention_mask,
            "spatial_shapes": spatial_shapes,
        }
        out = self.model.vision_model(**vision_inputs)
        return NaflexOutput(
            patch_embeds=out.last_hidden_state,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )

    def encode_image(self, images: list) -> NaflexOutput:
        """Convenience wrapper: preprocess + encode_preprocessed.

        Kept for one-off PIL→vision flows (probes, closed-loop eval). Training
        should call `preprocess` inside a DataLoader worker and
        `encode_preprocessed` on the GPU path, to hide the processor cost.
        """
        return self.encode_preprocessed(self.preprocess(images))

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Cache-friendly instruction encoder. Returns (B, T, d) text tokens."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = self.model.text_model(**inputs)
        return out.last_hidden_state  # (B, T, d)
