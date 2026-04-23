"""Single max_num_patches measurement in a clean process.
Prints one JSON line: {"max_np", "grid", "valid", "img_mean_ms", "img_p50_ms",
"full_mean_ms", "full_p50_ms", "img_std_ms", "full_std_ms"}
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL_ID = "google/siglip2-base-patch16-naflex"
IMG_W, IMG_H = 720, 450


def sync():
    torch.mps.synchronize()


def measure(n, fn):
    samples = []
    for _ in range(n):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append(time.perf_counter() - t0)
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-np", type=int, required=True)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--iters", type=int, default=40)
    args = p.parse_args()

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("mps")
    model.train(False)

    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (IMG_H, IMG_W, 3), dtype=np.uint8))

    img_inputs = processor(
        images=[img], return_tensors="pt", max_num_patches=args.max_np
    ).to("mps")
    txt_inputs = processor(
        text=["a photo of a cat"],
        padding="max_length",
        max_length=64,
        return_tensors="pt",
    ).to("mps")
    combined = {**img_inputs, **txt_inputs}

    grid = img_inputs["spatial_shapes"][0].tolist()
    valid = int(img_inputs["pixel_attention_mask"].sum())

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

    for _ in range(args.warmup):
        enc_img()
        full_forward()
    sync()

    img_samples = measure(args.iters, enc_img)
    full_samples = measure(args.iters, full_forward)

    def ms_stats(xs):
        xs_ms = [x * 1000.0 for x in xs]
        xs_sorted = sorted(xs_ms)
        return {
            "mean": statistics.mean(xs_ms),
            "p50": xs_sorted[len(xs_sorted) // 2],
            "p90": xs_sorted[int(0.9 * (len(xs_sorted) - 1))],
            "std": statistics.pstdev(xs_ms),
            "min": min(xs_ms),
        }

    result = {
        "max_np": args.max_np,
        "grid": grid,
        "valid": valid,
        "img": ms_stats(img_samples),
        "full": ms_stats(full_samples),
    }
    print("RESULT:" + json.dumps(result))


if __name__ == "__main__":
    main()
