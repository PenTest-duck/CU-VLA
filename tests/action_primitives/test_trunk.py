"""Unit tests for Trunk (cross+self attention)."""

import torch

from experiments.action_primitives.config import MODEL
from experiments.action_primitives.trunk import CrossSelfBlock, Trunk


def test_cross_self_block_forward_shape():
    block = CrossSelfBlock(d_model=MODEL.d_model, n_heads=MODEL.n_heads, ffn_mult=MODEL.ffn_mult)
    q = torch.randn(2, MODEL.n_queries, MODEL.d_model)
    kv = torch.randn(2, 100, MODEL.d_model)
    out = block(q, kv)
    assert out.shape == q.shape


def test_trunk_forward_shape():
    trunk = Trunk()
    kv = torch.randn(2, 265, MODEL.d_model)  # ~240 vision + 16 text + 1 proprio + 8 history
    out = trunk(kv)
    assert out.shape == (2, MODEL.n_queries, MODEL.d_model)


def test_trunk_respects_padding_mask():
    trunk = Trunk()
    kv = torch.randn(2, 300, MODEL.d_model)
    # Mask out last 50 tokens
    mask = torch.zeros(2, 300, dtype=torch.bool)
    mask[:, 250:] = True  # True = pad (torch convention for key_padding_mask)
    out = trunk(kv, kv_key_padding_mask=mask)
    assert out.shape == (2, MODEL.n_queries, MODEL.d_model)
    assert not torch.isnan(out).any()


def test_trunk_has_expected_n_blocks():
    trunk = Trunk()
    assert len(trunk.blocks) == MODEL.n_blocks
