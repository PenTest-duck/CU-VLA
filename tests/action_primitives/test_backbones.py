"""Smoke tests for SigLIP2 naflex loader. Requires model download; marked slow."""

import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.slow  # run with `pytest -m slow`


def test_naflex_loads_and_forwards_image():
    from experiments.action_primitives.backbones import SigLIP2Naflex

    m = SigLIP2Naflex(max_num_patches=64)  # small for test speed
    img = Image.new("RGB", (720, 450), color="white")
    out = m.encode_image([img])
    assert out.patch_embeds.ndim == 3          # (1, N, d)
    assert out.patch_embeds.shape[0] == 1
    assert out.patch_embeds.shape[-1] == 768   # SigLIP2-B hidden dim
    # Attention mask: at least 1 real patch
    assert out.attention_mask.sum().item() > 0


def test_naflex_text_tower_frozen():
    from experiments.action_primitives.backbones import SigLIP2Naflex

    m = SigLIP2Naflex(max_num_patches=64)
    text_params = list(m.model.text_model.parameters())
    assert all(not p.requires_grad for p in text_params)


def test_naflex_encodes_text():
    from experiments.action_primitives.backbones import SigLIP2Naflex

    m = SigLIP2Naflex(max_num_patches=64)
    tokens, mask = m.encode_text(["click the red button"])
    assert tokens.ndim == 3  # (1, T, d)
    assert tokens.shape[-1] == 768
    assert mask.ndim == 2  # (1, T)
    assert mask.shape == tokens.shape[:2]
    # First token must be unmasked (real)
    assert int(mask[0, 0]) == 1


def test_apply_lora_text_rank_zero_keeps_text_frozen():
    """Backward-compat: text_rank=0 (default) keeps text tower fully frozen."""
    from experiments.action_primitives.backbones import SigLIP2Naflex

    m = SigLIP2Naflex(max_num_patches=64)
    m.apply_lora(rank=4, text_rank=0)
    text_trainable = sum(1 for p in m.model.text_model.parameters() if p.requires_grad)
    assert text_trainable == 0


def test_apply_lora_text_rank_positive_unfreezes_text_lora_params():
    """When text_rank > 0, top N text-encoder layers gain trainable LoRA params."""
    from experiments.action_primitives.backbones import SigLIP2Naflex

    m = SigLIP2Naflex(max_num_patches=64)
    # Before LoRA: text fully frozen
    assert all(not p.requires_grad for p in m.model.text_model.parameters())
    m.apply_lora(rank=4, text_rank=4, text_target_layers=2)
    # After LoRA on text: at least some text params trainable (the LoRA adapters)
    text_trainable = [p for p in m.model.text_model.parameters() if p.requires_grad]
    assert len(text_trainable) > 0
    # And param count should be small (rank-4 on 2 layers)
    n = sum(p.numel() for p in text_trainable)
    assert n < 1_000_000


def test_apply_lora_text_path_supports_gradients():
    """encode_text must be backward-friendly when text LoRA is enabled.
    Forwards through the text tower, sums and backprops, expects gradient
    on at least one text-LoRA parameter."""
    import torch
    from experiments.action_primitives.backbones import SigLIP2Naflex

    m = SigLIP2Naflex(max_num_patches=64)
    m.apply_lora(rank=4, text_rank=4, text_target_layers=2)
    tokens, _mask = m.encode_text(["click the red button"])
    loss = tokens.sum()
    loss.backward()
    text_lora_params_with_grad = [
        p for p in m.model.text_model.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert len(text_lora_params_with_grad) > 0, \
        "Expected non-zero gradient on at least one text-LoRA param after backward"
