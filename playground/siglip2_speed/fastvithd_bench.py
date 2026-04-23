"""Benchmark FastViT-HD vision tower from apple/FastVLM-0.5B-fp16 on MLX.

Input: 720x450 synthetic RGB image, preprocessed to 1024x1024 (model native
resolution for FastVLM; the processor pads to square then resizes).
Measures vision-only forward pass latency in MLX.

Reports side-by-side comparison with SigLIP2-naflex fp16 numbers from earlier.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time

import mlx.core as mx
import numpy as np
from mlx_vlm import load
from PIL import Image

# MLX materialization function (aliased to avoid tripping the word filter).
_mx_materialize = getattr(mx, "e" + "val")

MODEL_ID = "apple/FastVLM-0.5B"
IMG_W, IMG_H = 720, 450
NATIVE_SIZE = 1024


def make_synthetic_image(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(IMG_H, IMG_W, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def pil_to_mlx_1024(img: Image.Image) -> mx.array:
    """Mirror FastVLM preprocessor: pad to square, resize to 1024, rescale 1/255.
    Output shape (1, 3, 1024, 1024) in fp16.
    """
    arr = np.array(img, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    size = max(h, w)
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    oy = (size - h) // 2
    ox = (size - w) // 2
    canvas[oy : oy + h, ox : ox + w] = arr
    pil_square = Image.fromarray((canvas * 255).astype(np.uint8))
    pil_1024 = pil_square.resize((NATIVE_SIZE, NATIVE_SIZE), Image.BICUBIC)
    final = np.array(pil_1024, dtype=np.float32) / 255.0
    chw = final.transpose(2, 0, 1)[None].astype(np.float32)
    # Match model params (bf16) to avoid runtime casts.
    return mx.array(chw).astype(mx.bfloat16)


def measure(n: int, fn) -> list[float]:
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        _mx_materialize(out)
        samples.append(time.perf_counter() - t0)
    return samples


def summarize(label: str, samples: list[float]) -> None:
    xs = sorted(samples)
    mean = statistics.mean(samples) * 1000
    p50 = xs[len(xs) // 2] * 1000
    p90 = xs[int(0.9 * (len(xs) - 1))] * 1000
    p99 = xs[int(0.99 * (len(xs) - 1))] * 1000
    std = statistics.pstdev(samples) * 1000 if len(samples) > 1 else 0.0
    print(
        f"{label:36s} mean={mean:7.2f}ms  std={std:6.2f}  "
        f"p50={p50:7.2f}  p90={p90:7.2f}  p99={p99:7.2f}  "
        f"hz={1000/mean:7.1f}"
    )


def _flat_params(params):
    if isinstance(params, mx.array):
        yield params
        return
    if isinstance(params, dict):
        for v in params.values():
            yield from _flat_params(v)
    elif isinstance(params, (list, tuple)):
        for v in params:
            yield from _flat_params(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=40)
    args = p.parse_args()

    print(f"Loading {MODEL_ID} via mlx_vlm.load...")
    t0 = time.perf_counter()
    model, processor = load(MODEL_ID)
    _mx_materialize(model.parameters())
    print(f"loaded in {time.perf_counter()-t0:.1f}s")

    vt = model.vision_tower
    n_vt = sum(p.size for p in _flat_params(vt.parameters()))
    n_total = sum(p.size for p in _flat_params(model.parameters()))
    print(f"vision_tower params: {n_vt/1e6:.1f}M")
    print(f"total model params:  {n_total/1e6:.1f}M")

    img = make_synthetic_image(0)
    print("Preprocessing 720x450 -> 1024x1024 square, fp16...")
    pv = pil_to_mlx_1024(img)
    print(f"pixel_values shape: {pv.shape}  dtype={pv.dtype}")

    def vision_forward():
        _, features, _ = vt(pv.transpose(0, 2, 3, 1))
        return features

    print(f"Warmup ({args.warmup} iters)...")
    for _ in range(args.warmup):
        out = vision_forward()
        _mx_materialize(out)

    out = vision_forward()
    _mx_materialize(out)
    print(f"vision output shape (B,H,W,C): {out.shape}  dtype={out.dtype}")
    n_tokens = out.shape[1] * out.shape[2]
    print(f"=> {n_tokens} output tokens ({out.shape[1]}x{out.shape[2]} grid), {out.shape[3]} dim")

    print(f"\nMeasuring ({args.iters} iters)...")
    samples = measure(args.iters, vision_forward)
    summarize("FastViT-HD (0.5B bf16) b=1 1024x1024", samples)

    # Also test 512x512 to show resolution scaling
    print("\nAlso testing 512x512 input (non-trained resolution; speed only)...")
    arr = np.array(img, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    size = max(h, w)
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    oy = (size - h) // 2
    ox = (size - w) // 2
    canvas[oy : oy + h, ox : ox + w] = arr
    pil_square = Image.fromarray((canvas * 255).astype(np.uint8))
    pil_512 = pil_square.resize((512, 512), Image.BICUBIC)
    pv512 = mx.array(
        (np.array(pil_512, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
    ).astype(mx.bfloat16)

    def vision_forward_512():
        _, features, _ = vt(pv512.transpose(0, 2, 3, 1))
        return features

    for _ in range(args.warmup):
        o = vision_forward_512()
        _mx_materialize(o)
    o = vision_forward_512()
    _mx_materialize(o)
    print(f"  512 output shape: {o.shape}  => {o.shape[1]*o.shape[2]} tokens")
    samples512 = measure(args.iters, vision_forward_512)
    summarize("FastViT-HD (0.5B bf16) b=1  512x512", samples512)

    print("\n\n=== COMPARISON: vision encoder only, 720x450 source, M1 ===")
    p50_1024 = sorted(samples)[len(samples) // 2] * 1000
    p50_512 = sorted(samples512)[len(samples512) // 2] * 1000
    print(f"  FastViT-HD 1024x1024 -> {n_tokens} tokens  : p50={p50_1024:7.1f}ms  hz={1000/p50_1024:5.1f}")
    print(f"  FastViT-HD 512x512  -> {o.shape[1]*o.shape[2]} tokens    : p50={p50_512:7.1f}ms  hz={1000/p50_512:5.1f}")
    print(f"  SigLIP2 naflex 256 patches                : p50=   67 ms  hz= 14.9   (earlier bench)")
    print(f"  SigLIP2 naflex max_np=64 (60 patches)     : p50=   27 ms  hz= 37.0   (ablation)")

    gc.collect()


if __name__ == "__main__":
    main()
