"""Ablate `max_num_patches` on siglip2-base-patch16-naflex (MPS fp16) to find
the largest budget that still clears 30 Hz on full forward b=1.

Sweeps a range of max_num_patches values, reports the actual patch grid the
processor selects for a 720x450 input, and per-component latency.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL_ID = "google/siglip2-base-patch16-naflex"
IMG_W, IMG_H = 720, 450
TARGET_HZ = 30.0
TARGET_MS = 1000.0 / TARGET_HZ


def make_synthetic_image(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(IMG_H, IMG_W, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def sync():
    torch.mps.synchronize()


@dataclass
class Stat:
    samples: list[float]

    def mean_ms(self):
        return statistics.mean(self.samples) * 1000.0

    def p50_ms(self):
        return sorted(self.samples)[len(self.samples) // 2] * 1000.0

    def p90_ms(self):
        xs = sorted(self.samples)
        return xs[int(0.9 * (len(xs) - 1))] * 1000.0

    def std_ms(self):
        return statistics.pstdev(self.samples) * 1000.0 if len(self.samples) > 1 else 0.0


def measure(n, fn):
    samples = []
    for _ in range(n):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append(time.perf_counter() - t0)
    return Stat(samples)


def run_one(model, processor, max_num_patches: int, warmup: int, iters: int):
    img = make_synthetic_image(0)
    img_inputs = processor(
        images=[img],
        return_tensors="pt",
        max_num_patches=max_num_patches,
    ).to("mps")
    txt_inputs = processor(
        text=["a photo of a cat"],
        padding="max_length",
        max_length=64,
        return_tensors="pt",
    ).to("mps")
    combined = {**img_inputs, **txt_inputs}

    pv_shape = tuple(img_inputs["pixel_values"].shape)
    spatial = img_inputs["spatial_shapes"][0].tolist()
    valid = int(img_inputs["pixel_attention_mask"].sum())
    total = int(img_inputs["pixel_attention_mask"].numel())

    vision = model.vision_model
    v_kwargs = {
        "pixel_values": img_inputs["pixel_values"],
        "attention_mask": img_inputs["pixel_attention_mask"],
        "spatial_shapes": img_inputs["spatial_shapes"],
    }

    @torch.inference_mode()
    def enc_img():
        _ = vision(**v_kwargs)

    @torch.inference_mode()
    def full_forward():
        _ = model(**combined)

    for _ in range(warmup):
        enc_img()
        full_forward()
    sync()

    img_stat = measure(iters, enc_img)
    full_stat = measure(iters, full_forward)

    return {
        "max_num_patches": max_num_patches,
        "pv_shape": pv_shape,
        "spatial": spatial,
        "valid": valid,
        "total": total,
        "img_encode": img_stat,
        "full_forward": full_stat,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument(
        "--patches",
        type=int,
        nargs="+",
        default=[16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 384, 512],
    )
    args = p.parse_args()

    print("Loading model...")
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("mps")
    model.train(False)
    print(f"loaded in {time.perf_counter()-t0:.1f}s")

    rows = []
    for n in args.patches:
        print(f"\n-- max_num_patches={n} --")
        r = run_one(model, processor, n, args.warmup, args.iters)
        grid = r["spatial"]
        print(
            f"  grid={grid[0]}x{grid[1]} valid={r['valid']}/{r['total']}  "
            f"pv_shape={r['pv_shape']}"
        )
        imgs = r["img_encode"]
        full = r["full_forward"]
        print(
            f"  img_encode: mean={imgs.mean_ms():6.2f}  p50={imgs.p50_ms():6.2f}  "
            f"p90={imgs.p90_ms():6.2f}  std={imgs.std_ms():5.2f}  hz={1000/imgs.mean_ms():5.1f}"
        )
        print(
            f"  full_fwd  : mean={full.mean_ms():6.2f}  p50={full.p50_ms():6.2f}  "
            f"p90={full.p90_ms():6.2f}  std={full.std_ms():5.2f}  hz={1000/full.mean_ms():5.1f}"
        )
        rows.append(r)
        gc.collect()
        torch.mps.empty_cache()

    print("\n\n=== ABLATION SUMMARY (naflex MPS fp16, 720x450 input) ===")
    print(
        f"{'max_np':>7}  {'grid':>8}  {'valid':>5}  "
        f"{'img mean':>9}  {'img p50':>9}  {'img Hz':>7}  "
        f"{'full mean':>10}  {'full p50':>9}  {'full Hz':>8}  "
        f"{'hits 30Hz':>10}"
    )
    for r in rows:
        grid = r["spatial"]
        full = r["full_forward"]
        imgs = r["img_encode"]
        hits = "YES" if full.mean_ms() < TARGET_MS else "no"
        print(
            f"{r['max_num_patches']:>7}  {grid[0]:>3}x{grid[1]:<4d}  {r['valid']:>5}  "
            f"{imgs.mean_ms():>7.2f}ms  {imgs.p50_ms():>7.2f}ms  {1000/imgs.mean_ms():>5.1f} Hz  "
            f"{full.mean_ms():>8.2f}ms  {full.p50_ms():>7.2f}ms  {1000/full.mean_ms():>6.1f} Hz  "
            f"{hits:>10}"
        )

    passing = [r for r in rows if r["full_forward"].mean_ms() < TARGET_MS]
    if passing:
        best = max(passing, key=lambda r: r["max_num_patches"])
        g = best["spatial"]
        print(
            f"\n>> Largest max_num_patches hitting 30 Hz: {best['max_num_patches']} "
            f"(grid {g[0]}x{g[1]}={best['valid']} valid tokens, "
            f"full={best['full_forward'].mean_ms():.1f}ms = "
            f"{1000/best['full_forward'].mean_ms():.1f} Hz)"
        )
    else:
        fastest = min(rows, key=lambda r: r["full_forward"].mean_ms())
        print(
            f"\n>> No config hit 30 Hz. Fastest: max_num_patches={fastest['max_num_patches']} "
            f"at {fastest['full_forward'].mean_ms():.1f}ms "
            f"({1000/fastest['full_forward'].mean_ms():.1f} Hz)"
        )


if __name__ == "__main__":
    main()
