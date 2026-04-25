"""Spike A (stricter) — per-patch typing-legibility probe.

Renders ONE character at a known pixel position on a 720x450 white canvas,
encodes through SigLIP2 naflex, locates the patch that covers the pixel via
spatial_shapes, and trains a 62-way linear probe on that single patch's 768-d
embedding. Reports top-1 train/test accuracy vs font size.

Rationale vs the pooled string-level probe (typing_legibility.py): this
directly tests whether ONE patch preserves character identity, rather than
whether a pooled global feature encodes char presence. It's a closer analogue
to what a cross-attention query in the real trunk would need to read out, and
is a strictly harder test of spatial-plus-identity information in the feature
map.

Charset is A-Za-z0-9 (62 classes). We deliberately EXCLUDE space here: unlike
the pooled probe (which uses space to signal 'char absent' in the multi-label
presence task), a single all-white patch would give the linear probe a trivial
'is this patch white?' shortcut instead of forcing it to learn char identity.

DO NOT use pygame.font — circular-import bug on Python 3.14. PIL only.
"""
from __future__ import annotations

import argparse
import json
import string
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from experiments.action_primitives.backbones import SigLIP2Naflex
from experiments.action_primitives.config import ENV
from experiments.action_primitives.probes.typing_legibility import _find_truetype_font


CHARSET = string.ascii_letters + string.digits  # 62 classes, no space (see module docstring)
CHAR_TO_ID = {c: i for i, c in enumerate(CHARSET)}
ID_TO_CHAR = {i: c for c, i in CHAR_TO_ID.items()}


@dataclass
class PerPatchProbeResult:
    font_size: int
    top1_test: float
    top1_train: float
    n_samples: int
    probe_type: str  # "per_patch_single_char"


# ---------- Patch-index math ----------


def compute_patch_index(
    char_x: float,
    char_y: float,
    canvas_w: int,
    canvas_h: int,
    h_patches: int,
    w_patches: int,
) -> int:
    """Map a pixel coord (char_x, char_y) to a flat patch index in row-major order.

    Patches are assumed to tile the canvas uniformly. The flat index is
    `row * w_patches + col`, matching the flattening that naflex uses when it
    reshapes `(h, w)` spatial patches into `N = h*w` tokens.

    Edge clamping: rounding can push a pixel at `canvas_w - 1` slightly past
    `w_patches - 1` for non-integer patch sizes. We clamp to the valid range so
    the returned index is always in `[0, h_patches*w_patches)`.
    """
    patch_w_px = canvas_w / w_patches
    patch_h_px = canvas_h / h_patches
    col = int(char_x / patch_w_px)
    row = int(char_y / patch_h_px)
    col = max(0, min(col, w_patches - 1))
    row = max(0, min(row, h_patches - 1))
    return row * w_patches + col


# ---------- Rendering ----------


def render_single_char(char: str, font_size: int, char_x: int, char_y: int) -> Image.Image:
    """Render `char` centered at pixel (char_x, char_y) on a 720x450 white PIL image.

    Uses the TrueType font resolved by `_find_truetype_font` (cross-platform).
    Charset excludes space; see module docstring.
    """
    img = Image.new("RGB", (ENV.canvas_w, ENV.canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_path = _find_truetype_font()
    if font_path is not None:
        pil_font = ImageFont.truetype(font_path, font_size)
    else:
        pil_font = ImageFont.load_default()

    # Measure bbox to center the glyph at (char_x, char_y).
    bbox = draw.textbbox((0, 0), char, font=pil_font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    # PIL textbbox may have a non-zero (x0, y0) offset for certain fonts; subtract it
    # so the visible glyph truly centers on (char_x, char_y).
    x = int(char_x - tw / 2 - bbox[0])
    y = int(char_y - th / 2 - bbox[1])
    draw.text((x, y), char, fill=(0, 0, 0), font=pil_font)
    return img


# ---------- Dataset builder ----------


def build_dataset_for_size(font_size: int, n_chars: int = 500) -> list[dict]:
    """Generate n_chars single-char records at `font_size`.

    Each record is a dict with keys:
        - "char_id": int in [0, 61]
        - "char_x":  float pixel coordinate
        - "char_y":  float pixel coordinate
        - "image":   PIL.Image of size (720, 450)

    Char is chosen uniformly from CHARSET. Pixel position is uniform inside
    the canvas with a margin of 2 * font_size from every edge so the glyph
    bbox never clips.

    Reproducibility: seed = 42 + font_size + i per record, via
    `numpy.random.default_rng`.
    """
    margin = 2 * font_size
    records: list[dict] = []
    for i in range(n_chars):
        rng = np.random.default_rng(42 + font_size + i)
        char_id = int(rng.integers(0, len(CHARSET)))
        char = CHARSET[char_id]
        char_x = int(rng.integers(margin, ENV.canvas_w - margin))
        char_y = int(rng.integers(margin, ENV.canvas_h - margin))
        img = render_single_char(char, font_size, char_x, char_y)
        records.append({
            "char_id": char_id,
            "char_x": char_x,
            "char_y": char_y,
            "image": img,
        })
    return records


# ---------- Per-patch probe ----------


def train_per_patch_probe(
    model: SigLIP2Naflex,
    records: list[dict],
    charset_size: int = 62,
    device: str = "cpu",
    epochs: int = 200,
) -> tuple[float, float]:
    """Encode each single-char image, pick the covering patch's embedding, train a
    linear 62-way classifier. Returns (top1_train, top1_test) on an 80/20 split.

    Implementation notes:
    - We encode one image at a time so the naflex processor is free to pick the
      best-fit patch grid per image. In practice with fixed canvas dims it
      tends to pick the same grid — but we don't rely on it.
    - We cache the selected 768-d patch vector in CPU memory; training then runs
      200 epochs of full-batch Adam over these cached features.
    """
    model = model.to(device)
    model.train(False)

    # Aspect-ratio sanity check using the first record. If naflex departs
    # significantly from the canvas aspect ratio, the probe's patch-index
    # math breaks — abort with a clear message rather than silently reporting
    # nonsense numbers.
    with torch.no_grad():
        probe_out = model.encode_image([records[0]["image"]])
    h0, w0 = int(probe_out.spatial_shapes[0, 0]), int(probe_out.spatial_shapes[0, 1])
    expected_ratio = ENV.canvas_w / ENV.canvas_h  # 1.6 for 720x450
    actual_ratio = w0 / max(h0, 1)
    if abs(actual_ratio - expected_ratio) > 0.25:
        raise RuntimeError(
            f"naflex patch grid ({h0}x{w0}) aspect ratio {actual_ratio:.3f} departs "
            f"from canvas aspect ratio {expected_ratio:.3f}. Per-patch localization "
            "is not meaningful under this processor config; aborting."
        )

    feats: list[torch.Tensor] = []
    labels: list[int] = []
    d = int(probe_out.patch_embeds.shape[-1])

    with torch.no_grad():
        for rec in records:
            out = model.encode_image([rec["image"]])
            h, w = int(out.spatial_shapes[0, 0]), int(out.spatial_shapes[0, 1])
            idx = compute_patch_index(
                char_x=float(rec["char_x"]),
                char_y=float(rec["char_y"]),
                canvas_w=ENV.canvas_w,
                canvas_h=ENV.canvas_h,
                h_patches=h,
                w_patches=w,
            )
            # Guard against a degenerate naflex output where the flat index
            # happens to point beyond the unmasked region. This should not
            # occur given the aspect-ratio check, but we're defensive.
            n_unmasked = int(out.attention_mask[0].sum().item())
            if idx >= n_unmasked:
                idx = n_unmasked - 1
            feats.append(out.patch_embeds[0, idx].detach().cpu())
            labels.append(int(rec["char_id"]))

    X = torch.stack(feats)                     # (N, d)
    y = torch.tensor(labels, dtype=torch.long) # (N,)

    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    probe = nn.Linear(d, charset_size)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        opt.zero_grad()
        logits = probe(X_tr)
        loss = loss_fn(logits, y_tr)
        loss.backward()
        opt.step()

    with torch.no_grad():
        top1_tr = (probe(X_tr).argmax(dim=1) == y_tr).float().mean().item()
        top1_te = (probe(X_te).argmax(dim=1) == y_te).float().mean().item()

    return top1_tr, top1_te


# ---------- CLI ----------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-patch typing-legibility probe (stricter than pooled string probe)."
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 10, 12, 14, 16, 20, 24, 32])
    parser.add_argument("--n-chars-per-size", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--out",
        type=str,
        default="docs/experiments/6-action-primitives-phase-a-results/spike-a-typing-legibility-per-patch.json",
    )
    args = parser.parse_args()

    m = SigLIP2Naflex(max_num_patches=256)

    results: list[PerPatchProbeResult] = []
    for size in args.sizes:
        records = build_dataset_for_size(size, n_chars=args.n_chars_per_size)
        top1_tr, top1_te = train_per_patch_probe(m, records, device=args.device)
        print(
            f"font_size={size:3d}  top1_train={top1_tr:.3f}  top1_test={top1_te:.3f}  "
            f"n={len(records)}  pool=per_patch"
        )
        results.append(PerPatchProbeResult(
            font_size=size,
            top1_test=top1_te,
            top1_train=top1_tr,
            n_samples=len(records),
            probe_type="per_patch_single_char",
        ))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump([asdict(r) for r in results], fh, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
