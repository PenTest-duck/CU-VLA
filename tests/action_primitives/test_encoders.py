"""Unit tests for proprio and history encoders."""

import torch

from experiments.action_primitives.config import MODEL, PROPRIO_DIM
from experiments.action_primitives.history import HISTORY_INPUT_DIM, HistoryEncoder
from experiments.action_primitives.proprio import ProprioEncoder


def test_proprio_encoder_shape():
    enc = ProprioEncoder()
    x = torch.randn(4, PROPRIO_DIM)
    out = enc(x)
    assert out.shape == (4, 1, MODEL.d_model)


def test_history_encoder_shape():
    enc = HistoryEncoder()
    x = torch.randn(4, MODEL.action_history_len, HISTORY_INPUT_DIM)
    out = enc(x)
    assert out.shape == (4, MODEL.action_history_len, MODEL.d_model)


def test_history_input_dim_matches_heads():
    # 21 + 21 + 5 + 21 + 77 + 77 + 1 = 223
    assert HISTORY_INPUT_DIM == 223


def test_history_encoder_temporal_pe_nonzero_on_init():
    enc = HistoryEncoder()
    # PE should be non-trivially initialized (not all zeros)
    assert enc.pos_emb.abs().sum().item() > 0
