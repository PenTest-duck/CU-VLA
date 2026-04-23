"""Benchmark action-expert trunk + SigLIP2 encoder standalone latency on M1 MPS fp16.

Two action-expert configs (random init — weights don't matter, only compute time):
  A: 3 blocks  (~20 M target)
  B: 6 blocks  (~40 M target)

Each block: cross-attn (queries->encoder K/V) + self-attn (queries<->queries) + FFN 4x.
Inputs: vision (1,240,768), text (1,16,768), proprio (1,1,768), action-history (1,8,768),
all fp16 on MPS. Output heads produce logits of sizes [21, 21, 5, 21, 231, 1].

Also re-measures SigLIP2-base-patch16-naflex (fp16, max_num_patches=256, 720x450 image)
to confirm the earlier ~80 ms figure is reproducible on the current machine.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoProcessor

SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-naflex"
IMG_W, IMG_H = 720, 450

# Action-expert dims
DIM = 768
NUM_HEADS = 12
NUM_QUERIES = 16
VISION_LEN = 240
TEXT_LEN = 16
PROPRIO_LEN = 1
HIST_LEN = 8
ENC_LEN = VISION_LEN + TEXT_LEN + PROPRIO_LEN + HIST_LEN  # 265
OUTPUT_HEAD_SIZES = [21, 21, 5, 21, 231, 1]  # mouse_dx, mouse_dy, click, scroll, keys, done


class CrossAttnBlock(nn.Module):
    """Cross-attention: queries (Q) attend to concatenated encoder K/V (N=265 tokens)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        out, _ = self.attn(qn, kvn, kvn, need_weights=False)
        return q + out


class SelfAttnBlock(nn.Module):
    """Self-attention: queries attend to queries."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        qn = self.norm(q)
        out, _ = self.attn(qn, qn, qn, need_weights=False)
        return q + out


class FFN(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * mult)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * mult, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class ActionExpertBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.cross = CrossAttnBlock(dim, num_heads)
        self.self_attn = SelfAttnBlock(dim, num_heads)
        self.ffn = FFN(dim, mult=4)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q = self.cross(q, kv)
        q = self.self_attn(q)
        q = self.ffn(q)
        return q


class ActionExpert(nn.Module):
    def __init__(self, num_blocks: int, pool_heads: bool = False):
        super().__init__()
        self.pool_heads = pool_heads
        self.queries = nn.Parameter(torch.randn(1, NUM_QUERIES, DIM) * 0.02)
        self.blocks = nn.ModuleList([ActionExpertBlock(DIM, NUM_HEADS) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(DIM)
        # 6 output heads:
        #   pool_heads=False: flatten (NUM_QUERIES*DIM,) -> N
        #   pool_heads=True:  mean-pool queries to (DIM,)    -> N
        head_in = DIM if pool_heads else NUM_QUERIES * DIM
        self.heads = nn.ModuleList([nn.Linear(head_in, n) for n in OUTPUT_HEAD_SIZES])

    def forward(
        self,
        vision: torch.Tensor,  # (B, 240, 768)
        text: torch.Tensor,  # (B, 16, 768)
        proprio: torch.Tensor,  # (B, 1, 768)
        history: torch.Tensor,  # (B, 8, 768)
    ) -> list[torch.Tensor]:
        kv = torch.cat([vision, text, proprio, history], dim=1)  # (B, 265, 768)
        q = self.queries.expand(vision.size(0), -1, -1)
        for blk in self.blocks:
            q = blk(q, kv)
        q = self.norm(q)
        if self.pool_heads:
            feat = q.mean(dim=1)  # (B, 768)
        else:
            feat = q.reshape(q.size(0), -1)  # (B, 16*768)
        return [h(feat) for h in self.heads]


# ------------- measurement utilities -------------

def sync():
    torch.mps.synchronize()


def measure(n: int, fn) -> list[float]:
    out = []
    for _ in range(n):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        out.append(time.perf_counter() - t0)
    return out


def pct(xs: list[float], p: float) -> float:
    xs_s = sorted(xs)
    k = max(0, min(len(xs_s) - 1, int(round(p * (len(xs_s) - 1)))))
    return xs_s[k] * 1000.0


def stats_ms(xs: list[float]) -> dict:
    return {
        "mean": statistics.mean(xs) * 1000.0,
        "median": pct(xs, 0.5),
        "p5": pct(xs, 0.05),
        "p95": pct(xs, 0.95),
        "std": statistics.pstdev(xs) * 1000.0 if len(xs) > 1 else 0.0,
    }


# ------------- action expert bench -------------

def bench_action_expert(
    num_blocks: int, warmup: int, iters: int, device: torch.device, pool_heads: bool = False
) -> dict:
    torch.manual_seed(0)
    model = ActionExpert(num_blocks, pool_heads=pool_heads).to(device=device, dtype=torch.float16)
    model.train(False)
    n_params = sum(p.numel() for p in model.parameters())

    # Random fp16 inputs
    def rnd(*shape):
        return torch.randn(*shape, device=device, dtype=torch.float16)

    vision = rnd(1, VISION_LEN, DIM)
    text = rnd(1, TEXT_LEN, DIM)
    proprio = rnd(1, PROPRIO_LEN, DIM)
    history = rnd(1, HIST_LEN, DIM)

    @torch.inference_mode()
    def fwd():
        _ = model(vision, text, proprio, history)

    for _ in range(warmup):
        fwd()
    sync()
    samples = measure(iters, fwd)

    del model
    gc.collect()
    torch.mps.empty_cache()

    return {
        "n_params": n_params,
        "stats": stats_ms(samples),
        "samples": samples,
    }


# ------------- siglip2 encoder bench -------------

def bench_siglip2(warmup: int, iters: int, device: torch.device) -> dict:
    processor = AutoProcessor.from_pretrained(SIGLIP2_MODEL_ID)
    model = AutoModel.from_pretrained(SIGLIP2_MODEL_ID, torch_dtype=torch.float16).to(device)
    model.train(False)
    n_params = sum(p.numel() for p in model.parameters())

    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (IMG_H, IMG_W, 3), dtype=np.uint8))
    img_inputs = processor(
        images=[img], return_tensors="pt", max_num_patches=256
    ).to(device)
    txt_inputs = processor(
        text=["a photo of a cat"],
        padding="max_length",
        max_length=64,
        return_tensors="pt",
    ).to(device)
    combined = {**img_inputs, **txt_inputs}

    @torch.inference_mode()
    def fwd():
        _ = model(**combined)

    for _ in range(warmup):
        fwd()
    sync()
    samples = measure(iters, fwd)

    del model, processor
    gc.collect()
    torch.mps.empty_cache()

    return {
        "n_params": n_params,
        "stats": stats_ms(samples),
        "samples": samples,
    }


# ------------- main -------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument(
        "--only",
        choices=["A", "A_pooled", "B", "siglip2"],
        default=None,
        help="If set, run only this config and emit one RESULT:{json} line.",
    )
    args = ap.parse_args()

    device = torch.device("mps")

    def emit(key: str, result: dict) -> None:
        import json

        payload = {
            "key": key,
            "n_params": result["n_params"],
            "stats": result["stats"],
        }
        print("RESULT:" + json.dumps(payload))

    if args.only == "A":
        r = bench_action_expert(3, args.warmup, args.iters, device, pool_heads=False)
        emit("A_3blk", r)
        return
    if args.only == "A_pooled":
        r = bench_action_expert(3, args.warmup, args.iters, device, pool_heads=True)
        emit("A_pooled", r)
        return
    if args.only == "B":
        r = bench_action_expert(6, args.warmup, args.iters, device, pool_heads=False)
        emit("B_6blk", r)
        return
    if args.only == "siglip2":
        r = bench_siglip2(args.warmup, args.iters, device)
        emit("siglip2", r)
        return

    # Default: all in one process (noisy; prefer --only with subprocess driver)
    print(f"Device: {device}")
    print(f"Warmup: {args.warmup} | iters: {args.iters}")
    print(f"Action expert: {NUM_QUERIES} queries, {DIM}-dim, {NUM_HEADS} heads")
    print(f"Encoder K/V length: {ENC_LEN} tokens (vision={VISION_LEN}, text={TEXT_LEN}, proprio={PROPRIO_LEN}, history={HIST_LEN})")
    print(f"Output heads: {OUTPUT_HEAD_SIZES}\n")

    results = {}

    print("== Config A: 3 blocks ==")
    results["A_3blk"] = bench_action_expert(3, args.warmup, args.iters, device)
    print("\n== Config B: 6 blocks ==")
    results["B_6blk"] = bench_action_expert(6, args.warmup, args.iters, device)
    print("\n== SigLIP2 ==")
    results["siglip2"] = bench_siglip2(args.warmup, args.iters, device)

    for k, r in results.items():
        s = r["stats"]
        print(
            f"{k:12s} params={r['n_params']/1e6:.2f}M  "
            f"mean={s['mean']:.2f}  med={s['median']:.2f}  "
            f"p5={s['p5']:.2f}  p95={s['p95']:.2f}"
        )


if __name__ == "__main__":
    main()
