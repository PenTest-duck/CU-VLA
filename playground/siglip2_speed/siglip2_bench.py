"""Benchmark SigLIP2 (google/siglip2-base-patch16-naflex) inference speed on M1 MPS.

Image input: 720x450 synthetic RGB. Text input: short sentences.
Measures per-component latency (image preprocess, image encode, text preprocess,
text encode, full forward) across fp32 and fp16 on MPS vs CPU.
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

DEFAULT_MODEL_ID = "google/siglip2-base-patch16-naflex"
IMG_W, IMG_H = 720, 450

SHORT_TEXTS = [
    "a photo of a cat",
    "a red button on a gray background",
    "a text field with blinking cursor",
    "a dropdown menu showing options",
    "a scrollbar on the right edge",
    "a dialog asking for confirmation",
    "an icon in the top left corner",
    "a slider partially moved to the right",
]


def make_synthetic_image(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(IMG_H, IMG_W, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@dataclass
class Stat:
    name: str
    samples: list[float]

    def ms(self, p: float) -> float:
        xs = sorted(self.samples)
        if not xs:
            return float("nan")
        k = max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))
        return xs[k] * 1000.0

    def summary(self) -> str:
        mean = statistics.mean(self.samples) * 1000.0
        std = statistics.pstdev(self.samples) * 1000.0 if len(self.samples) > 1 else 0.0
        return (
            f"{self.name:32s} "
            f"mean={mean:7.2f}ms  std={std:6.2f}  "
            f"p50={self.ms(0.5):7.2f}  p90={self.ms(0.9):7.2f}  "
            f"p99={self.ms(0.99):7.2f}  "
            f"hz={1.0/statistics.mean(self.samples):7.1f}"
        )


def measure(device: torch.device, n: int, fn) -> list[float]:
    samples = []
    for _ in range(n):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        samples.append(time.perf_counter() - t0)
    return samples


def bench(model_id: str, device_str: str, dtype: torch.dtype, warmup: int, iters: int) -> dict:
    device = torch.device(device_str)
    print(f"\n=== model={model_id}  device={device_str}  dtype={dtype} ===")
    t_load = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device).eval()
    load_s = time.perf_counter() - t_load
    n_params = sum(p.numel() for p in model.parameters())
    print(f"load: {load_s:.2f}s  params: {n_params/1e6:.1f}M")

    img = make_synthetic_image(seed=0)
    imgs_batch1 = [img]
    imgs_batch4 = [make_synthetic_image(i) for i in range(4)]
    texts_batch1 = [SHORT_TEXTS[0]]
    texts_batch4 = SHORT_TEXTS[:4]
    texts_batch8 = SHORT_TEXTS[:8]

    def prep_img_b1():
        _ = processor(images=imgs_batch1, return_tensors="pt")

    def prep_img_b4():
        _ = processor(images=imgs_batch4, return_tensors="pt")

    img_inputs_b1 = processor(images=imgs_batch1, return_tensors="pt").to(device)
    img_inputs_b4 = processor(images=imgs_batch4, return_tensors="pt").to(device)
    txt_inputs_b1 = processor(text=texts_batch1, padding="max_length", max_length=64, return_tensors="pt").to(device)
    txt_inputs_b4 = processor(text=texts_batch4, padding="max_length", max_length=64, return_tensors="pt").to(device)
    txt_inputs_b8 = processor(text=texts_batch8, padding="max_length", max_length=64, return_tensors="pt").to(device)

    is_naflex = "pixel_attention_mask" in img_inputs_b1
    if "pixel_values" in img_inputs_b1:
        pv = img_inputs_b1["pixel_values"]
        print(f"pixel_values shape (b1): {tuple(pv.shape)}  dtype={pv.dtype}")
    if "spatial_shapes" in img_inputs_b1:
        print(f"spatial_shapes (b1): {img_inputs_b1['spatial_shapes'].tolist()}")
    if is_naflex:
        mask = img_inputs_b1["pixel_attention_mask"]
        print(f"pixel_attention_mask shape (b1): {tuple(mask.shape)}  valid={int(mask.sum())}/{mask.numel()}")
    print(f"naflex mode: {is_naflex}")

    vision = model.vision_model
    text_model = model.text_model

    text_keys = {"input_ids", "attention_mask"}

    def vision_kwargs(d):
        if is_naflex:
            return {
                "pixel_values": d["pixel_values"],
                "attention_mask": d["pixel_attention_mask"],
                "spatial_shapes": d["spatial_shapes"],
            }
        return {"pixel_values": d["pixel_values"]}

    v_kwargs_b1 = vision_kwargs(img_inputs_b1)
    v_kwargs_b4 = vision_kwargs(img_inputs_b4)

    @torch.inference_mode()
    def enc_img_b1():
        _ = vision(**v_kwargs_b1)

    @torch.inference_mode()
    def enc_img_b4():
        _ = vision(**v_kwargs_b4)

    @torch.inference_mode()
    def enc_txt_b1():
        _ = text_model(**{k: v for k, v in txt_inputs_b1.items() if k in text_keys})

    @torch.inference_mode()
    def enc_txt_b4():
        _ = text_model(**{k: v for k, v in txt_inputs_b4.items() if k in text_keys})

    @torch.inference_mode()
    def enc_txt_b8():
        _ = text_model(**{k: v for k, v in txt_inputs_b8.items() if k in text_keys})

    combined_b1 = {**img_inputs_b1, **txt_inputs_b1}
    combined_b4 = {**img_inputs_b4, **txt_inputs_b4}

    @torch.inference_mode()
    def full_forward_b1():
        _ = model(**combined_b1)

    @torch.inference_mode()
    def full_forward_b4():
        _ = model(**combined_b4)

    jobs = [
        ("preprocess image b=1", prep_img_b1),
        ("preprocess image b=4", prep_img_b4),
        ("encode image b=1", enc_img_b1),
        ("encode image b=4", enc_img_b4),
        ("encode text b=1", enc_txt_b1),
        ("encode text b=4", enc_txt_b4),
        ("encode text b=8", enc_txt_b8),
        ("full forward b=1", full_forward_b1),
        ("full forward b=4", full_forward_b4),
    ]

    results = {}
    for name, fn in jobs:
        for _ in range(warmup):
            fn()
        sync(device)
        samples = measure(device, iters, fn)
        stat = Stat(name, samples)
        print(stat.summary())
        results[name] = stat

    del model, processor
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return {"load_s": load_s, "n_params": n_params, "stats": results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL_ID, help="HF model id")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument(
        "--configs",
        nargs="+",
        default=["mps:fp32", "mps:fp16", "cpu:fp32"],
        help="device:dtype combos to run",
    )
    args = p.parse_args()

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    all_results = {}
    for cfg in args.configs:
        dev, dt = cfg.split(":")
        all_results[cfg] = bench(args.model, dev, dtype_map[dt], args.warmup, args.iters)

    print("\n\n=== SUMMARY (mean ms) ===")
    print(f"model: {args.model}")
    names = sorted({n for r in all_results.values() for n in r["stats"]})
    header = f"{'job':32s} " + "  ".join(f"{c:>14s}" for c in all_results)
    print(header)
    for name in names:
        row = f"{name:32s} "
        for cfg, res in all_results.items():
            mean_ms = statistics.mean(res["stats"][name].samples) * 1000.0
            row += f"  {mean_ms:12.2f}ms"
        print(row)


if __name__ == "__main__":
    main()
