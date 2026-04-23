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
import math
import string
from dataclasses import dataclass, asdict, field
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
    mean_f1: float  # alias of mean_f1_test (kept as the rubric-compared value)
    mean_f1_train: float
    mean_f1_test: float
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


# ---------- Pooling ops over naflex patch embeddings ----------


def global_avg_patch_repr(out, target_mask_is_text: torch.Tensor | None = None) -> torch.Tensor:
    """Average patch embeddings over all real (non-padded) patches per sample.
    Returns (B, d). We deliberately do NOT try to localize text — this tests
    whether SigLIP2 representation at max_patches=256 *anywhere* encodes char identity.
    """
    mask = out.attention_mask.unsqueeze(-1).float()  # (B, N, 1)
    summed = (out.patch_embeds * mask).sum(dim=1)    # (B, d)
    counts = mask.sum(dim=1).clamp(min=1)            # (B, 1)
    return summed / counts


def max_pool(out) -> torch.Tensor:
    """Max-pool patch embeddings over real (unmasked) patches.

    Masked positions are set to -inf before reducing, so they never win the max.
    Returns (B, d). No trainable params.
    """
    mask = out.attention_mask.bool().unsqueeze(-1)  # (B, N, 1)
    masked_embeds = out.patch_embeds.masked_fill(~mask, float("-inf"))
    return masked_embeds.max(dim=1).values


def attention_pool(out, q: torch.Tensor) -> torch.Tensor:
    """Attention-pool patch embeddings using a single learnable query `q` (shape (d,)).

    logits = (patch_embeds @ q) / sqrt(d)   -> (B, N)
    Masked patches get -inf so softmax assigns them zero weight.
    weights = softmax over patches          -> (B, N)
    pooled = sum(weights * patch_embeds)    -> (B, d)
    """
    patch_embeds = out.patch_embeds  # (B, N, d)
    d = patch_embeds.shape[-1]
    logits = (patch_embeds @ q) / math.sqrt(d)  # (B, N)
    mask = out.attention_mask.bool()            # (B, N)
    logits = logits.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(logits, dim=1)      # (B, N)
    pooled = (weights.unsqueeze(-1) * patch_embeds).sum(dim=1)  # (B, d)
    return pooled


# ---------- Probe ----------


def train_string_presence_probe(
    model: SigLIP2Naflex,
    images: list[Image.Image],
    texts: list[str],
    charset_size: int = 63,
    device: str = "cpu",
    pool: str = "attention",
) -> tuple[float, float]:
    """Train a linear probe: pooled SigLIP2 features → multi-label presence-of-char (63 binary).

    `pool` selects the pooling strategy over naflex patch embeds:
        - "mean": global mean over unmasked patches (no params)          [original methodology]
        - "max":  max over unmasked patches (no params)
        - "attention": single learnable query trained jointly with the linear head

    Returns (mean_f1_train, mean_f1_test).
    """
    if pool not in {"mean", "max", "attention"}:
        raise ValueError(f"Unknown pool={pool!r}; expected mean|max|attention")

    model = model.to(device)
    model.train(False)

    # Cache raw (patch_embeds, attention_mask) so training can re-pool across epochs
    # without re-running the vision tower every step. This is essential for attention
    # pooling where q is updated every step.
    patch_list: list[torch.Tensor] = []
    mask_list: list[torch.Tensor] = []
    with torch.no_grad():
        for img in images:
            out = model.encode_image([img])
            patch_list.append(out.patch_embeds.cpu())
            mask_list.append(out.attention_mask.cpu())

    # Pad to a common N so we can batch — different images may yield different patch counts.
    max_n = max(p.shape[1] for p in patch_list)
    d = patch_list[0].shape[-1]
    N = len(patch_list)
    patch_tensor = torch.zeros(N, max_n, d)
    mask_tensor = torch.zeros(N, max_n)
    for i, (p, m) in enumerate(zip(patch_list, mask_list)):
        n_i = p.shape[1]
        patch_tensor[i, :n_i] = p[0]
        mask_tensor[i, :n_i] = m[0]

    # Multi-label target: y[i, c] = 1 if char c present in text[i]
    Y = torch.zeros(len(texts), charset_size)
    for i, t in enumerate(texts):
        for c in set(t):
            Y[i, CHAR_TO_ID[c]] = 1.0

    n = patch_tensor.shape[0]
    split = int(0.8 * n)
    P_tr, P_te = patch_tensor[:split], patch_tensor[split:]
    M_tr, M_te = mask_tensor[:split], mask_tensor[split:]
    Y_tr, Y_te = Y[:split], Y[split:]

    class _PseudoOut:
        def __init__(self, pe, am):
            self.patch_embeds = pe
            self.attention_mask = am

    # Probe head + (optional) learnable query
    probe = nn.Linear(d, charset_size)
    params = list(probe.parameters())
    q: torch.Tensor | None = None
    if pool == "attention":
        q = nn.Parameter(torch.randn(d) * 0.02)
        params.append(q)

    opt = torch.optim.Adam(params, lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    def _pool(pe: torch.Tensor, am: torch.Tensor) -> torch.Tensor:
        out = _PseudoOut(pe, am)
        if pool == "mean":
            return global_avg_patch_repr(out)
        if pool == "max":
            return max_pool(out)
        # attention
        return attention_pool(out, q)

    for _ in range(200):
        opt.zero_grad()
        feats_tr = _pool(P_tr, M_tr)
        loss = loss_fn(probe(feats_tr), Y_tr)
        loss.backward()
        opt.step()

    def _macro_f1(preds: torch.Tensor, Y_ref: torch.Tensor) -> float:
        tp = (preds * Y_ref).sum(dim=0)
        fp = (preds * (1 - Y_ref)).sum(dim=0)
        fn = ((1 - preds) * Y_ref).sum(dim=0)
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        appearing = (Y_ref.sum(dim=0) > 0).float()
        return (f1 * appearing).sum().item() / max(appearing.sum().item(), 1.0)

    with torch.no_grad():
        feats_tr = _pool(P_tr, M_tr)
        feats_te = _pool(P_te, M_te)
        preds_tr = (torch.sigmoid(probe(feats_tr)) > 0.5).float()
        preds_te = (torch.sigmoid(probe(feats_te)) > 0.5).float()
        f1_tr = _macro_f1(preds_tr, Y_tr)
        f1_te = _macro_f1(preds_te, Y_te)

    return f1_tr, f1_te


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 10, 12, 14, 16, 20, 24, 32])
    parser.add_argument("--n-strings-per-size", type=int, default=500)
    parser.add_argument(
        "--pool",
        type=str,
        choices=["mean", "max", "attention"],
        default="attention",
        help="Patch pooling strategy for the probe. 'attention' adds a single learnable query.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="docs/experiments/6-action-primitives-phase-a-results/spike-a-typing-legibility.json")
    args = parser.parse_args()

    m = SigLIP2Naflex(max_num_patches=256)

    results: list[ProbeResult] = []
    for size in args.sizes:
        images, texts = build_dataset_for_size(size, n_strings=args.n_strings_per_size)
        f1_tr, f1_te = train_string_presence_probe(m, images, texts, device=args.device, pool=args.pool)
        print(
            f"font_size={size:3d}  mean_f1_train={f1_tr:.3f}  mean_f1_test={f1_te:.3f}  "
            f"n={len(images)}  pool={args.pool}"
        )
        results.append(ProbeResult(
            font_size=size,
            mean_f1=f1_te,
            mean_f1_train=f1_tr,
            mean_f1_test=f1_te,
            n_samples=len(images),
        ))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump([asdict(r) for r in results], fh, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
