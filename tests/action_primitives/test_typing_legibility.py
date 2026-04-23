"""Unit tests for probe helpers. Does not require SigLIP2 download."""

import pytest
import torch
from PIL import Image


def test_render_text_frame_size_and_mode():
    from experiments.action_primitives.probes.typing_legibility import render_text_frame

    img = render_text_frame("hello", font_size=14)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (720, 450)


def test_render_text_frame_empty_string():
    from experiments.action_primitives.probes.typing_legibility import render_text_frame

    img = render_text_frame("", font_size=14)
    assert img.size == (720, 450)


def test_build_dataset_for_size_returns_matched_lengths():
    from experiments.action_primitives.probes.typing_legibility import build_dataset_for_size

    imgs, texts = build_dataset_for_size(font_size=14, n_strings=5)
    assert len(imgs) == len(texts) == 5
    for img in imgs:
        assert img.size == (720, 450)


# ---------- Pooling helper tests (no SigLIP2 needed, uses fake NaflexOutput) ----------


class _FakeNaflexOutput:
    """Minimal stand-in for backbones.NaflexOutput in pooling unit tests."""

    def __init__(self, patch_embeds: torch.Tensor, attention_mask: torch.Tensor):
        self.patch_embeds = patch_embeds
        self.attention_mask = attention_mask
        self.spatial_shapes = None


def test_attention_pool_shape():
    from experiments.action_primitives.probes.typing_legibility import attention_pool

    torch.manual_seed(0)
    B, N, d = 2, 10, 8
    patch_embeds = torch.randn(B, N, d)
    mask = torch.ones(B, N)
    out = _FakeNaflexOutput(patch_embeds, mask)
    q = torch.nn.Parameter(torch.randn(d) * 0.02)

    pooled = attention_pool(out, q)

    assert pooled.shape == (B, d)


def test_attention_pool_respects_mask():
    """Patches with mask=0 must not contribute to the pooled output. We verify this
    by showing that masking trailing patches gives the same pooled output as dropping
    those patches entirely.
    """
    from experiments.action_primitives.probes.typing_legibility import attention_pool

    torch.manual_seed(1)
    d = 6
    # Full: 5 real patches with random embeds
    full_embeds = torch.randn(1, 5, d)
    # Masked version: 5 patches, but last 2 are masked out and given a deliberately
    # large value to prove that high q-similarity alone does NOT leak in when mask=0.
    masked_embeds = full_embeds.clone()
    masked_embeds[0, 3, :] = 1000.0  # would dominate softmax if not masked
    masked_embeds[0, 4, :] = -1000.0

    # Same learnable query for both
    q = torch.nn.Parameter(torch.randn(d) * 0.02, requires_grad=False)

    full_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float32)
    out_masked = _FakeNaflexOutput(masked_embeds, full_mask)

    # Reference: the first 3 patches only, all unmasked.
    ref_embeds = full_embeds[:, :3, :].clone()
    ref_mask = torch.ones(1, 3)
    out_ref = _FakeNaflexOutput(ref_embeds, ref_mask)

    pooled_masked = attention_pool(out_masked, q)
    pooled_ref = attention_pool(out_ref, q)

    assert torch.allclose(pooled_masked, pooled_ref, atol=1e-5), (
        f"Masked patches leaked in. masked={pooled_masked}, ref={pooled_ref}"
    )


def test_max_pool_respects_mask():
    """A large value hidden under mask=0 must NOT appear in max-pool output."""
    from experiments.action_primitives.probes.typing_legibility import max_pool

    d = 4
    # 4 patches; patch 2 has a very large value but is masked out.
    embeds = torch.tensor([
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.1, 0.9, 0.2],
            [999.0, 999.0, 999.0, 999.0],  # masked out
            [0.3, 0.8, 0.1, 0.7],
        ]
    ])
    mask = torch.tensor([[1, 1, 0, 1]], dtype=torch.float32)

    out = _FakeNaflexOutput(embeds, mask)
    pooled = max_pool(out)

    # Expected: max over the 3 unmasked patches (rows 0, 1, 3)
    expected = torch.tensor([[0.5, 0.8, 0.9, 0.7]])
    assert torch.allclose(pooled, expected), f"max_pool={pooled}, expected={expected}"


def test_probe_result_schema_has_mean_f1_fields():
    from experiments.action_primitives.probes.typing_legibility import ProbeResult

    r = ProbeResult(font_size=14, mean_f1=0.5, mean_f1_train=0.6, mean_f1_test=0.5, n_samples=10)
    assert r.font_size == 14
    assert r.mean_f1 == 0.5
    assert r.mean_f1_train == 0.6
    assert r.mean_f1_test == 0.5
    assert r.n_samples == 10
