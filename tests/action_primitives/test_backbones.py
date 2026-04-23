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
    tokens = m.encode_text(["click the red button"])
    assert tokens.ndim == 3  # (1, T, d)
    assert tokens.shape[-1] == 768
