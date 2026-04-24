"""Unit tests for the stricter per-patch typing-legibility probe.

These tests cover the pure helpers — patch-index math, render, dataset builder,
and the result dataclass schema. They do NOT exercise the SigLIP2 vision tower.
"""

from PIL import Image


# ---------- compute_patch_index ----------


def test_compute_patch_index_center():
    from experiments.action_primitives.probes.typing_legibility_per_patch import (
        compute_patch_index,
    )

    # canvas 720x450, grid 20x14 -> patch size = 36x~32.142857 px
    # pixel (360, 225):
    #   col = int(360 / (720/20)) = int(360 / 36) = 10
    #   row = int(225 / (450/14)) = int(225 / 32.142857) = int(6.9999...) = 6
    #     (floor of 6.999 is 6 — floating-point: 225 is just under 7*32.142857)
    #   idx = 6 * 20 + 10 = 130
    idx = compute_patch_index(
        char_x=360.0,
        char_y=225.0,
        canvas_w=720,
        canvas_h=450,
        h_patches=14,
        w_patches=20,
    )
    assert idx == 130


def test_compute_patch_index_top_left():
    from experiments.action_primitives.probes.typing_legibility_per_patch import (
        compute_patch_index,
    )

    # pixel (5, 5) -> col=0, row=0 -> idx=0
    idx = compute_patch_index(
        char_x=5.0,
        char_y=5.0,
        canvas_w=720,
        canvas_h=450,
        h_patches=14,
        w_patches=20,
    )
    assert idx == 0


def test_compute_patch_index_bottom_right_clamped():
    from experiments.action_primitives.probes.typing_legibility_per_patch import (
        compute_patch_index,
    )

    # Max valid index for 20x14 grid is 14*20 - 1 = 279
    # pixel (719, 449) — exactly canvas_w-1, canvas_h-1 — should land on last patch
    # col = int(719 / 36) = int(19.97) = 19
    # row = int(449 / 32.14) = int(13.97) = 13
    # idx = 13*20 + 19 = 279
    idx = compute_patch_index(
        char_x=719.0,
        char_y=449.0,
        canvas_w=720,
        canvas_h=450,
        h_patches=14,
        w_patches=20,
    )
    assert idx == 279

    # Even a very-slightly-out-of-bounds pixel should clamp, not overflow
    idx_over = compute_patch_index(
        char_x=720.0,
        char_y=450.0,
        canvas_w=720,
        canvas_h=450,
        h_patches=14,
        w_patches=20,
    )
    assert idx_over == 279  # clamped to last valid patch


def test_compute_patch_index_known_values():
    """Two more concrete cases, manually computed."""
    from experiments.action_primitives.probes.typing_legibility_per_patch import (
        compute_patch_index,
    )

    # Case 1: 16x10 grid on 720x450, pixel (100, 100).
    # patch_w_px = 720/16 = 45 ; patch_h_px = 450/10 = 45
    # col = int(100/45) = 2 ; row = int(100/45) = 2
    # idx = 2*16 + 2 = 34
    idx1 = compute_patch_index(
        char_x=100.0,
        char_y=100.0,
        canvas_w=720,
        canvas_h=450,
        h_patches=10,
        w_patches=16,
    )
    assert idx1 == 34

    # Case 2: 20x14 grid, pixel (540, 360).
    # patch_w_px = 720/20 = 36 ; patch_h_px = 450/14 ≈ 32.142857
    # col = int(540/36) = 15 ; row = int(360/32.142857) = int(11.2) = 11
    # idx = 11*20 + 15 = 235
    idx2 = compute_patch_index(
        char_x=540.0,
        char_y=360.0,
        canvas_w=720,
        canvas_h=450,
        h_patches=14,
        w_patches=20,
    )
    assert idx2 == 235


# ---------- render_single_char ----------


def test_render_single_char_returns_image_with_correct_size():
    from experiments.action_primitives.probes.typing_legibility_per_patch import (
        render_single_char,
    )

    img = render_single_char("A", font_size=14, char_x=200, char_y=300)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (720, 450)


# ---------- build_dataset_for_size ----------


def test_build_dataset_for_size_returns_valid_records():
    from experiments.action_primitives.probes.typing_legibility_per_patch import (
        build_dataset_for_size,
        CHARSET,
    )

    records = build_dataset_for_size(font_size=14, n_chars=5)
    assert len(records) == 5

    margin = 2 * 14
    for rec in records:
        # Expect dict-like with char_id, char_x, char_y, image
        assert rec["char_id"] in range(len(CHARSET))
        assert 0 <= rec["char_id"] <= 62
        assert margin <= rec["char_x"] <= 720 - margin
        assert margin <= rec["char_y"] <= 450 - margin
        assert isinstance(rec["image"], Image.Image)
        assert rec["image"].size == (720, 450)


# ---------- PerPatchProbeResult dataclass ----------


def test_per_patch_probe_result_schema():
    from experiments.action_primitives.probes.typing_legibility_per_patch import (
        PerPatchProbeResult,
    )

    r = PerPatchProbeResult(
        font_size=14,
        top1_test=0.5,
        top1_train=0.6,
        n_samples=10,
        probe_type="per_patch_single_char",
    )
    assert r.font_size == 14
    assert r.top1_test == 0.5
    assert r.top1_train == 0.6
    assert r.n_samples == 10
    assert r.probe_type == "per_patch_single_char"
