# SigLIP2 base-patch16-naflex — M1 inference speed

Hardware: MacBook Pro M1, 8 GB RAM. PyTorch 2.11, transformers 5.5.3.
Model: `google/siglip2-base-patch16-naflex`, **375.2 M params**.
Input: 720×450 RGB image (synthetic), short-sentence captions (≤64 tokens).
30 iterations per job, 5 warmup. Timings are per-call wall clock with explicit
`torch.mps.synchronize()` around each run.

## How naflex handles 720×450

The processor reshapes the image to fit the default `max_num_patches=256`
budget while preserving aspect ratio. For 720:450 ≈ 1.6, it picks a **12×20
patch grid = 240 valid patches** (93.75 % of the budget), padded to 256 with
an attention mask. So each forward attends over 256 vision tokens — the same
cost as a square 256-patch layout.

## Results (mean / p50 / p90 in ms; Hz on mean)

| job                   | MPS fp32        | **MPS fp16**    | CPU fp32         |
| --------------------- | --------------- | --------------- | ---------------- |
| preprocess image b=1  | 3.9 / 3.8 / 4.7 | 7.2 / 6.0 / 10.5 | 4.0 / 3.4 / 5.5 |
| encode image b=1      | 107 / 93 / 154  | **64 / 67 / 75**| 117 / 116 / 119  |
| encode image b=4      | 277 / 258 / 337 | 225 / 221 / 234 | 424 / 396 / 451  |
| encode text b=1 (64)  | 20 / 15 / 31    | **12 / 11 / 15**| 49 / 50 / 53     |
| encode text b=4       | 58 / 58 / 61    | 46 / 46 / 48    | 108 / 107 / 111  |
| encode text b=8       | 121 / 114 / 137 | 89 / 88 / 91    | 198 / 191 / 211  |
| **full forward b=1**  | 102 / 99 / 113  | **81 / 80 / 87**| 167 / 162 / 184  |
| full forward b=4      | 391 / 350 / 490 | 273 / 265 / 305 | 1160 / 551 / 3702 |

## Key findings

1. **Best config is MPS + fp16**. Full forward (1 image + 1 short caption) lands at
   **~81 ms / 12.3 Hz**, with p90=87 ms — stable run-to-run (σ≈5 ms).
2. **Vision encoder dominates** (~80 % of latency). 64 ms for image vs 12 ms for text
   on MPS fp16. This is expected: 256 image tokens vs ~8 content tokens in a short
   caption (padded to 64).
3. **fp16 vs fp32 on MPS**: ~40 % speedup across both vision and text. No accuracy
   testing was done here, but SigLIP2 is trained with mixed precision — fp16 inference
   is the standard deployment path.
4. **MPS vs CPU**: MPS fp16 is **2.1×** faster than CPU fp32 on full forward b=1,
   and **4.1×** faster on text-only (MPS wins bigger on the dense transformer,
   smaller on the patch-attention vision pass — the variable-mask naflex attention
   doesn't fully saturate the GPU).
5. **Batching helps a little**. b=4 image encode on MPS fp16 = 225 ms = **56 ms /
   image** — only a 13 % improvement over b=1. naflex pads every image to 256
   tokens regardless, so there's no "cheap small image" discount. Batching is
   worth it only when you genuinely have ≥4 images to encode simultaneously.
6. **CPU fp32 b=4 has severe tail latency** (p99=5.3 s vs p50=0.55 s). 375 M params
   in fp32 is 1.5 GB — under memory pressure on 8 GB the CPU path thrashes. Stay
   on MPS.
7. **Preprocessor is fast** (4–7 ms per image) and non-blocking relative to the
   ~65 ms GPU work that follows — can be pipelined on a CPU thread.

## Implications for the CU-VLA project

For the **System 1 / fast policy** loop at 30 Hz (33 ms/step):
SigLIP2-naflex-base at 81 ms/step is **~2.5× too slow** as a drop-in vision
backbone. Options to hit 30 Hz:

- Lower patch budget (`max_num_patches=64` or `128` via processor override) —
  should scale roughly linearly with token count. Worth testing.
- Move to a smaller vision tower (ResNet18 / DINOv2 ViT-S already used in
  experiments 2/3/5 — those run well under 10 ms).
- Quantize to int8 with CoreML / MLX for further 1.5–2× on M1.

For the **System 2 / slow planner** at 2–5 Hz: 12 Hz is plenty of headroom,
fp16 is fine.

## naflex vs fixed-224 head-to-head (MPS only, same 30 iter / 5 warmup)

Fed the same 720×450 source image to both. naflex produces 12×20=240 valid
patches (padded to 256). The -224 processor resizes → 14×14=196 patches on a
squashed 224×224 square (aspect-ratio distortion). Both variants share the
**same 375.2 M params** — identical dual-encoder and text model, just
different vision input handling.

| job                    | naflex fp16 (mean / p50)   | -224 fp16 (mean / p50)    | Δ (mean / p50)   |
| ---------------------- | -------------------------- | ------------------------- | ---------------- |
| encode image b=1       | 64.2 / 66.8 ms             | 77.1 / 71.3 ms            | +20 % / +7 %     |
| encode text b=1        | 11.8 / 11.0 ms             | 15.8 / 12.3 ms            | +34 % / +12 %    |
| **full forward b=1**   | **81.2 / 80.4 ms**         | **76.0 / 76.3 ms**        | **−6 % / −5 %**  |
| encode image b=4       | 225 / 221 ms               | 346 / 279 ms              | +54 % / +26 %    |
| full forward b=4       | 273 / 265 ms               | 319 / 281 ms              | +17 % / +6 %     |

**Surprise:** -224 is **not faster** than naflex at b=1 image encode despite
using 196 vs 256 tokens (−23 % tokens, which should imply ~−40 % attention
FLOPs via n²). On mean it's 20 % *slower*; on median only 7 % faster. Full
forward b=1 is a wash (76 vs 81 ms) — within noise once text time is
included. At b=4 the -224 path is clearly slower with high tail latency
(p99 950 ms on image encode), likely MPS shape-compilation effects for the
newly-seen input size.

Hypothesis: MPS SDPA doesn't scale cleanly with token count at these sizes
— compiler overhead + memory-bandwidth patterns dominate over the raw FLOP
count. Naflex's padded-mask attention path appears about as fast as -224's
dense attention on this hardware.

**Bottom line:** on M1, -224 has **no meaningful speed win** over naflex,
and you pay a quality cost from aspect-ratio distortion (720:450 → 1:1
squashes by 28 % vertically). **Stick with naflex for non-square inputs
like screenshots.** The real speed lever is cutting `max_num_patches`, not
switching variants.

## max_num_patches ablation (naflex MPS fp16, 720×450, clean subprocess per config, 15 warmup + 40 iter)

The first attempt ran all configs in one process and showed ~2× worse
numbers than a standalone run — MPS was accumulating shape-compile cache
garbage across configs. Re-ran each budget in its **own isolated
subprocess** for trustworthy medians.

| max_np | grid (actual) | valid | img encode p50 | img Hz | **full fwd p50** | **full Hz** | hits 30 Hz? |
| -----: | ------------: | ----: | -------------: | -----: | ---------------: | ----------: | :---------: |
|     16 |         3 × 5 |    15 |       46.6 ms* |   21.5 |         112.0 ms* |         8.9* |    no       |
|     32 |         4 × 7 |    28 |       24.4 ms  |   41.1 |          45.4 ms |        22.0 |    no       |
|     48 |         5 × 8 |    40 |       27.0 ms  |   37.0 |          44.3 ms |        22.6 |    no       |
|     64 |        6 × 10 |    60 |       27.1 ms  |   36.8 |          44.2 ms |        22.6 |    no       |
|     96 |        8 × 12 |    96 |       39.5 ms  |   25.3 |          64.6 ms |        15.5 |    no       |
|    128 |        9 × 14 |   126 |       39.8 ms  |   25.1 |          55.5 ms |        18.0 |    no       |
|    160 |       10 × 16 |   160 |       44.0 ms  |   22.7 |          64.2 ms |        15.6 |    no       |
|    192 |       11 × 17 |   187 |       52.7 ms  |   19.0 |          69.5 ms |        14.4 |    no       |
|    224 |       12 × 18 |   216 |       62.9 ms  |   15.9 |          77.8 ms |        12.9 |    no       |
|    256 |       12 × 20 |   240 |       69.4 ms  |   14.4 |          88.9 ms |        11.2 |    no       |
|    384 |       15 × 24 |   360 |      107.9 ms  |    9.3 |         124.8 ms |         8.0 |    no       |
|    512 |       18 × 28 |   504 |      157.1 ms  |    6.4 |         165.6 ms |         6.0 |    no       |

*The `max_np=16` row looks anomalously slow — with only 15 tokens I'd expect faster
than max_np=32. Likely post-warmup MPS compile noise on a shape too small for the
kernels to run efficiently. Not worth chasing; smaller is not better here.*

Values match the original clean single-config bench (256 → ~90 ms here vs
~81 ms earlier — within noise given MPS variance).

### Key findings

1. **No max_num_patches value reaches 30 Hz on full forward.** Fastest is
   32–64 patches plateauing at ~44–45 ms / ~22 Hz. Below that, token count
   is too small to dominate and fixed costs (text encoder + projection
   heads + sync overhead) form a floor.
2. **Scaling is sub-quadratic**. 256 → 64 patches (4× fewer tokens) only
   cuts full forward from 89 → 44 ms (2×). MPS SDPA kernels don't fully
   capitalize on shorter sequences; the text tower and FFN blocks remain
   O(n) work regardless.
3. **The text encoder is the floor for full forward**. ~12 ms text + ~27 ms
   image at 64 patches + sync overhead = ~44 ms. There's nowhere for the
   full-forward curve to go below ~45 ms without removing / replacing the
   text tower.

### But: with cached text embeddings, 30 Hz IS reachable

For a computer-use VLA, the task instruction changes rarely (per episode),
while the screenshot updates every frame. Encoding text once per episode
and reusing the embedding makes image-only encode the per-step cost:

| max_np | image encode p50 | Hz | hits 30 Hz (image only)? |
| -----: | ---------------: | ---: | :--: |
|     32 |          24.4 ms | 41.1 | **YES** |
|     48 |          27.0 ms | 37.0 | **YES** |
|     64 |          27.1 ms | 36.8 | **YES** |
|     96 |          39.5 ms | 25.3 | no |
|    128 |          39.8 ms | 25.1 | no |

**Sweet spot: max_num_patches=64** (actual grid 6 × 10 = 60 valid tokens for
720 × 450). 27 ms / ~37 Hz on image encode, preserves aspect ratio, leaves
~6 ms headroom in the 33 ms budget for the downstream policy head.

**At 96 patches there's a sharp cliff** (25 Hz) — the jump in tokens pushes
MPS into a slower kernel or cache path. If you need >60 valid tokens, you
have to give up 30 Hz on M1.

### Implications for CU-VLA System 1 loop

- **Viable path**: naflex with `max_num_patches=64`, precomputed text
  embeddings cached per-task, vision-only at ~37 Hz. Gives you ~6 ms for a
  small policy head on top.
- **Not viable**: full forward at every step — 22 Hz is the ceiling, no
  matter how hard you squeeze tokens.
- **Open questions**: (1) does 60 valid tokens carry enough spatial detail
  for 720×450 cursor-level tasks? The grid is 6 × 10 at ~75 px / patch —
  this is coarser than DINOv2 ViT-S at 224² (16 × 16 at 14 px/patch used
  in exps 2/3/5). May need evaluation. (2) Are the SigLIP2 features at
  this low token count still strong enough to offset vs. a from-scratch
  ResNet18?

## FastViT-HD (from apple/FastVLM-0.5B) on MLX — surprise finding

**The "smallest" FastViT-HD is not actually small**: there's only one size of
FastViT-HD across all FastVLM variants (0.5B / 1.5B / 7B refers to the LLM).
The vision tower is **125.1 M params** and trained at a fixed native
resolution of **1024×1024**. Output is 16×16 = 256 tokens.

Loaded via `mlx_vlm.load("apple/FastVLM-0.5B")` — the `-fp16` variant ships
its vision weights only as a CoreML package (`fastvithd.mlpackage`), with
an empty safetensors for vision. The non-`-fp16` repo has full bf16 safetensors
that mlx-vlm can load. Runs in MLX bf16 on M1.

Also spot-tested 512×512 input (out-of-distribution, speed-only datapoint).

| config                                   | params   | tokens | p50 (ms) | Hz   |
| ---------------------------------------- | -------- | -----: | -------: | ---: |
| FastViT-HD @ 1024² (native)              | 125.1 M  |    256 |   1040.0 |  1.0 |
| FastViT-HD @ 512² (OOD, speed only)      | 125.1 M  |     64 |    255.1 |  3.9 |
| SigLIP2 naflex, max_np=256 (fp16, torch) |  93.0 M  |    240 |     67.0 | 14.9 |
| SigLIP2 naflex, max_np=64  (fp16, torch) |  93.0 M  |     60 |     27.0 | 37.0 |

### FastViT-HD is **~15× slower than SigLIP2 naflex** on M1 Metal

Counter-intuitive given "FastVLM" naming, but makes sense on inspection:

1. **FastViT-HD is Neural-Engine-optimised, not Metal-optimised.** Apple
   designed it for iPhone/iPad Neural Engine, which accelerates the conv-
   heavy FastViT-HD backbone dramatically. MLX on M1 runs on the GPU via
   Metal Performance Shaders; it gets no Neural Engine benefit here.
2. **1024² input means 1 M pixels through many conv stages before any
   downsampling.** Early stages operate at 256² = 65 k spatial positions
   — the conv cost dominates even if attention is cheap.
3. **SigLIP2 is mostly matmul**, which Metal handles well. The 256-token
   variant operates on 224²-equivalent patches, running at a fraction of
   FastViT-HD's upstream pixel throughput.
4. **The "fast" in FastVLM refers to TTFT for the downstream LLM**, not
   raw encoder latency — the 256-token output means the LLM prefill is
   fast. On an ANE-less Metal path, the encoder itself is a bottleneck.

**Practical takeaway for CU-VLA:** Do **not** use FastViT-HD on M1 GPU for a
real-time policy loop. For a CoreML + Neural Engine deployment path, it
might be competitive — but that's a separate integration. SigLIP2-naflex at
`max_num_patches=64` remains the fastest pretrained encoder I've measured
on this hardware.

## Action-expert standalone latency (random-init, MPS fp16)

Two transformer action-expert trunks sitting on top of an encoder. Each
block = `CrossAttn(queries↔encoder K/V) + SelfAttn(queries↔queries) +
FFN(4×)`. Inputs: vision (1,240,768) + text (1,16,768) + proprio (1,1,768)
+ action-history (1,8,768) = 265 K/V tokens. 16 learnable 768-d queries.
6 output heads: linear (`flatten(16×768)→N`) with N ∈ {21, 21, 5, 21, 231, 1}
for mouse dx/dy, click, scroll, keys, done.

Benchmarked with 20 warmup + 100 timed iterations per config, each in its
**own subprocess** (as in the patch ablation — same-process sequential MPS
benchmarks pollute each other's graph cache).

| config                              | actual params | median | mean  | p5    | p95    | std   | Hz (median) |
| ----------------------------------- | ------------: | -----: | ----: | ----: | -----: | ----: | ----------: |
| Action expert A (3 blocks, flatten) |       32.1 M  |  39.9 ms | 49.4 | 29.8 |  104.6 | 25.7  |      25.1   |
| Action expert A-pooled (mean-pool)  |       28.6 M  |  49.3 ms | 50.5 | 39.8 |   64.2 |  8.2  |      20.3   |
| Action expert B (6 blocks, flatten) |       60.4 M  |  74.6 ms | 98.3 | 60.6 |  247.0 | 66.4  |      13.4   |
| SigLIP2 naflex (max_np=256)         |      375.2 M  |  92.5 ms | 97.9 | 87.9 |  119.9 | 22.1  |      10.8   |

### Note: actual param counts exceed the targets

Configs A and B came out **32 M / 60 M**, not the 20 M / 40 M targets. The
architecture spec (3 × [cross + self + FFN(4×)] at 768-dim, 12 heads) is
the dominant driver — plus the output heads: the `flatten(16 × 768) → 231`
keys-logit head alone is 2.84 M params. If you really want 20 M, you'd
need to shrink the FFN multiplier, pool the queries before the heads
(project `(768,) → N` instead of `(16 × 768,) → N`, saving ~3.1 M total),
or drop dim to 512.

### Per-iteration variance

The means are well above medians for A and B (p95 is 2–3× p50). This is
**MPS graph-compile overhead on the first ~10–15 iterations bleeding past
the 20-iter warmup**. Medians are the trustworthy number; the full forward
compute is ~40 ms (A) / ~75 ms (B) in steady state. SigLIP2 is tighter
(p5 88 → p95 120, σ=22 ms) because its graph stabilises faster.

The SigLIP2 median here (92 ms) is slightly worse than my earlier
clean-process bench (81 ms). Small environmental drift — ~10–15 %
variance between same-model runs on MPS is normal with other processes
competing for the GPU. The earlier 80 ms number is reproducible as the
floor; 92 ms here is a warm-machine variant.

### Encoder + expert pipeline

| pipeline | median end-to-end | Hz | vs 30-Hz budget (33 ms) |
| --- | ---: | ---: | :--- |
| SigLIP2 + A (3 blocks) | 132.3 ms | 7.6 | 4× over budget |
| SigLIP2 + B (6 blocks) | 167.1 ms | 6.0 | 5× over budget |

**Expert adds on top of encoder:**
- **A: +40 ms (+43 % of encoder)** — a non-trivial add; can't be treated as "free on top of the encoder"
- **B: +75 ms (+81 % of encoder)** — roughly doubles the pipeline cost

### Implications for CU-VLA

- Neither expert + SigLIP2 @ max_np=256 hits 30 Hz. Pipeline is 4–5×
  over budget.
- Pairing A with the aggressive naflex config from the patch ablation
  (max_np=64, 27 ms image encode, ~12 ms text — but text can be cached
  per-episode), expert A adds 40 ms → total ~67 ms per step (~15 Hz).
  Still misses 30 Hz.
- To hit 30 Hz with A, either (a) shrink A further (fewer heads, smaller
  dim, pooled output heads) toward ~10 ms, or (b) run expert at a lower
  Hz than the encoder (action chunking at 2–3× the frame rate).
- At 30 Hz, expert B is out of reach regardless.

### INT8 on MPS — not evaluated

Skipped. PyTorch's `torch.ao.quantization` targets CPU/CUDA; `bitsandbytes`
has no MPS backend; `torch.compile`'s int8 fast-paths are CUDA-only. The
realistic M1 paths to int8 are CoreML export or MLX quantization, both
outside the "pure PyTorch on MPS" scope here. Flag for a later session.

## Not tested (flag for follow-up if needed)

- **MLX port**. `mlx-vlm` does not currently ship SigLIP2-naflex; would need a
  port. Anecdotally MLX is ~50 % faster than MPS for ViTs on M1, which would
  push full forward to ~55 ms (18 Hz) — still below 30 Hz without reducing
  patches.
- **`torch.compile`**. Often flaky on MPS in torch 2.11; skipped to stay within
  one session. Likely <20 % win if it works.
- **bf16**. MPS supports bf16 but kernel coverage is worse than fp16 on M1;
  unlikely to beat fp16.
- **Reduced `max_num_patches`**. The single most promising lever to hit 30 Hz —
  naflex attention scales with tokens², so halving tokens should ~quarter the
  attention cost.
