"""Tests for Experiment 3 ACT model and BaselineCNN."""

import torch
import pytest


def test_act_forward_pass_shapes():
    from experiments.miniwob_pygame.model import ACT
    from experiments.miniwob_pygame.config import MODEL, ACTION, NUM_BINS

    model = ACT(backbone_name="resnet18", chunk_size=10)
    B = 4
    images = torch.randn(B, 3, 224, 224)
    proprio = torch.randn(B, MODEL.proprio_dim)  # 46
    out = model(images, proprio)

    assert out["dx_logits"].shape == (B, 10, NUM_BINS)
    assert out["dy_logits"].shape == (B, 10, NUM_BINS)
    assert out["mouse_left"].shape == (B, 10)
    assert out["keys_held"].shape == (B, 10, ACTION.num_keys)
    assert out["pad_logits"].shape == (B, 10)


def test_act_forward_is_stable_for_same_input():
    from experiments.miniwob_pygame.model import ACT
    from experiments.miniwob_pygame.config import MODEL

    model = ACT(backbone_name="resnet18", chunk_size=10)
    model.train(False)
    B = 2
    images = torch.randn(B, 3, 224, 224)
    proprio = torch.randn(B, MODEL.proprio_dim)

    with torch.no_grad():
        out_a = model(images, proprio)
        out_b = model(images, proprio)

    assert torch.allclose(out_a["dx_logits"], out_b["dx_logits"])
    assert torch.allclose(out_a["dy_logits"], out_b["dy_logits"])
    assert torch.allclose(out_a["mouse_left"], out_b["mouse_left"])
    assert torch.allclose(out_a["keys_held"], out_b["keys_held"])


def test_baseline_cnn_forward():
    from experiments.miniwob_pygame.baseline_cnn import BaselineCNN

    model = BaselineCNN()
    x = torch.randn(4, 3, 224, 224)
    dx, dy, mouse_logit, key_logits = model(x)
    assert dx.shape == (4, 1)
    assert dy.shape == (4, 1)
    assert mouse_logit.shape == (4, 1)
    assert key_logits.shape == (4, 43)


def test_act_parameter_count():
    from experiments.miniwob_pygame.model import ACT, count_parameters

    model = ACT(backbone_name="resnet18", chunk_size=10)
    total = count_parameters(model, trainable_only=False)
    assert total > 1_000_000  # should be ~33M+
    assert total < 100_000_000  # but under 100M
