"""Smoke test for full ACTModel forward pass.

Marked `slow` — downloads SigLIP2 (~400MB) on first run.
"""
import pytest
import torch
from PIL import Image

from experiments.action_primitives.config import HEAD_LOGITS, MODEL, PROPRIO_DIM
from experiments.action_primitives.history import HISTORY_INPUT_DIM

pytestmark = pytest.mark.slow


def test_model_forward_pass_and_shapes():
    from experiments.action_primitives.model import ActionPrimitivesACT

    model = ActionPrimitivesACT().eval()
    images = [Image.new("RGB", (720, 450), color=(255, 255, 255)) for _ in range(2)]
    # Build dummy text cache: B=2, T=16 tokens, d=768
    text_tokens = torch.randn(2, 16, MODEL.d_model)
    text_mask = torch.ones(2, 16)
    proprio = torch.randn(2, PROPRIO_DIM)
    history = torch.randn(2, MODEL.action_history_len, HISTORY_INPUT_DIM)

    with torch.no_grad():
        out = model(images, text_tokens, text_mask, proprio, history)

    for name, n_logits in HEAD_LOGITS.items():
        assert out.head_logits[name].shape == (2, n_logits)


def test_model_parameter_counts_reasonable():
    from experiments.action_primitives.model import ActionPrimitivesACT

    model = ActionPrimitivesACT()
    total = sum(p.numel() for p in model.parameters())
    # Q16 estimated ~118M (42M vision + 44M text + ~32M rest). Actual SigLIP2-B
    # naflex ships 92.9M vision + 282.3M text = 375M backbone, so the true total
    # lands near ~408M. Assertion widened to reflect measured reality while still
    # guarding against silent architecture drift.
    assert 380e6 < total < 430e6, f"unexpected param count: {total / 1e6:.1f}M"


def test_lora_adapters_are_trainable():
    from experiments.action_primitives.model import ActionPrimitivesACT

    model = ActionPrimitivesACT()
    lora_params = [n for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    assert len(lora_params) > 0, "no trainable LoRA params found"


def test_text_tower_lora_state():
    """B0 attempt 2: with MODEL.text_lora_rank > 0, text tower has LoRA-only
    trainable params (just the lora_A / lora_B matrices on top text_lora_target_layers
    encoder layers — base text params remain frozen). Pre-attempt-2 behavior
    (text fully frozen) is exercised by the unit test in test_backbones.py:
    test_apply_lora_text_rank_zero_keeps_text_frozen.
    """
    from experiments.action_primitives.model import ActionPrimitivesACT
    from experiments.action_primitives.config import MODEL

    model = ActionPrimitivesACT()
    text_trainable = [
        n for n, p in model.backbone.model.text_model.named_parameters() if p.requires_grad
    ]
    if getattr(MODEL, "text_lora_rank", 0) > 0:
        # All trainable params should be LoRA adapters (not base weights)
        assert len(text_trainable) > 0, "text LoRA enabled but no trainable params found"
        for n in text_trainable:
            assert "lora_" in n, f"non-LoRA text param is trainable: {n}"
    else:
        assert len(text_trainable) == 0, f"text tower has trainable params: {text_trainable}"
