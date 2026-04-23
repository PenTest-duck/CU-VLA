"""Unit tests for probe helpers. Does not require SigLIP2 download."""

import pytest
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
