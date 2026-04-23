"""Spike A — Typing legibility probe.

Renders pygame text at varied font sizes, encodes through SigLIP2 naflex,
trains a linear probe from patch features to per-patch character identity.
Reports top-1 accuracy vs font size.

Success signal: accuracy curve shows text becoming discriminable above a
threshold font size. If threshold >> 14pt, Q5's "typing handled via visual
feedback" assumption is questionable and Q6/max_num_patches should be revisited.
"""
from __future__ import annotations

import argparse
import json
import string
from dataclasses import dataclass, asdict
from pathlib import Path

import glob
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from experiments.action_primitives.backbones import SigLIP2Naflex
from experiments.action_primitives.config import ENV


CHARSET = string.ascii_letters + string.digits + " "  # 63 classes incl. space
CHAR_TO_ID = {c: i for i, c in enumerate(CHARSET)}
ID_TO_CHAR = {i: c for c, i in CHAR_TO_ID.items()}


@dataclass
class ProbeResult:
    font_size: int
    top1_accuracy: float
    n_samples: int


_FONT_SEARCH_PATHS = [
    "/Library/Fonts/Arial.ttf",
    "/Library/Fonts/DejaVuSans.ttf",
    "/Library/Fonts/Helvetica.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]


def _find_truetype_font() -> str | None:
    """Return the path of a usable TrueType font on this system, or None."""
    for fp in _FONT_SEARCH_PATHS:
        if os.path.exists(fp):
            return fp
    # Broader fallback: first .ttf under /Library/Fonts or /usr/share/fonts
    for pattern in ("/Library/Fonts/*.ttf", "/System/Library/Fonts/Supplemental/*.ttf",
                    "/usr/share/fonts/truetype/*/*.ttf"):
        hits = glob.glob(pattern)
        if hits:
            return hits[0]
    return None


def render_text_frame(text: str, font_size: int, font_name: str = "arial") -> Image.Image:
    """Render `text` centered on a 720x450 white background PIL image.

    Uses PIL ImageFont.truetype with a system font (font_name hint is ignored
    in favour of a reliable path lookup). Falls back to PIL's built-in bitmap
    font if no TrueType font is found — legibility at small sizes will suffer
    but the probe can still run.
    """
    img = Image.new("RGB", (ENV.canvas_w, ENV.canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_path = _find_truetype_font()
    if font_path is not None:
        pil_font = ImageFont.truetype(font_path, font_size)
    else:
        pil_font = ImageFont.load_default()
    # Render line-by-line, centered
    lines = text.split("\n") if text else [""]
    total_h = len(lines) * font_size
    y = ENV.canvas_h // 2 - total_h // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=pil_font)
        tw = bbox[2] - bbox[0]
        x = ENV.canvas_w // 2 - tw // 2
        draw.text((x, y), line, fill=(0, 0, 0), font=pil_font)
        y += font_size
    return img


def build_dataset_for_size(font_size: int, n_strings: int = 50) -> tuple[list[Image.Image], list[str]]:
    """Generate n_strings random short strings at `font_size` + their ground-truth text."""
    rng = np.random.default_rng(42 + font_size)
    images: list[Image.Image] = []
    texts: list[str] = []
    for _ in range(n_strings):
        length = rng.integers(3, 12)
        chars = rng.choice(list(CHARSET), size=length)
        text = "".join(chars)
        images.append(render_text_frame(text, font_size))
        texts.append(text)
    return images, texts


def global_avg_patch_repr(out, target_mask_is_text: torch.Tensor) -> torch.Tensor:
    """Average patch embeddings over all real (non-padded) patches per sample.
    Returns (B, d). We deliberately do NOT try to localize text — this tests
    whether SigLIP2 representation at max_patches=256 *anywhere* encodes char identity.
    """
    mask = out.attention_mask.unsqueeze(-1).float()  # (B, N, 1)
    summed = (out.patch_embeds * mask).sum(dim=1)    # (B, d)
    counts = mask.sum(dim=1).clamp(min=1)            # (B, 1)
    return summed / counts


def train_string_presence_probe(
    model: SigLIP2Naflex,
    images: list[Image.Image],
    texts: list[str],
    charset_size: int = 63,
    device: str = "cpu",
) -> float:
    """Train a linear probe: globally-avg SigLIP2 features → multi-label presence-of-char (63 binary).
    Returns mean per-character F1 on the held-out split.
    """
    model = model.to(device)
    model.train(False)
    X_list: list[torch.Tensor] = []
    with torch.no_grad():
        for img in images:
            out = model.encode_image([img])
            X_list.append(global_avg_patch_repr(out, None).cpu())
    X = torch.cat(X_list, dim=0)  # (N, d)
    # Multi-label target: y[i, c] = 1 if char c present in text[i]
    Y = torch.zeros(len(texts), charset_size)
    for i, t in enumerate(texts):
        for c in set(t):
            Y[i, CHAR_TO_ID[c]] = 1.0

    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, X_te = X[:split], X[split:]
    Y_tr, Y_te = Y[:split], Y[split:]

    # Linear probe: d -> charset_size
    probe = nn.Linear(X.shape[-1], charset_size)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(200):
        opt.zero_grad()
        loss = loss_fn(probe(X_tr), Y_tr)
        loss.backward()
        opt.step()

    with torch.no_grad():
        preds = (torch.sigmoid(probe(X_te)) > 0.5).float()
        # Per-char F1 (macro)
        tp = (preds * Y_te).sum(dim=0)
        fp = (preds * (1 - Y_te)).sum(dim=0)
        fn = ((1 - preds) * Y_te).sum(dim=0)
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        # Only average over chars that appear in the test set
        appearing = (Y_te.sum(dim=0) > 0).float()
        mean_f1 = (f1 * appearing).sum() / appearing.sum().clamp(min=1)

    return mean_f1.item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 10, 12, 14, 16, 20, 24, 32])
    parser.add_argument("--n-strings-per-size", type=int, default=80)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="docs/experiments/6-action-primitives-phase-a-results/spike-a-typing-legibility.json")
    args = parser.parse_args()

    m = SigLIP2Naflex(max_num_patches=256)

    results: list[ProbeResult] = []
    for size in args.sizes:
        images, texts = build_dataset_for_size(size, n_strings=args.n_strings_per_size)
        f1 = train_string_presence_probe(m, images, texts, device=args.device)
        print(f"font_size={size:3d}  mean_f1={f1:.3f}  n={len(images)}")
        results.append(ProbeResult(font_size=size, top1_accuracy=f1, n_samples=len(images)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump([asdict(r) for r in results], fh, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
