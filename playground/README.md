# playground

Scratch directory for one-off experiments and benchmarks.

## siglip2_speed/

Benchmarks `google/siglip2-base-patch16-naflex` inference speed on M1.

```bash
uv run python playground/siglip2_speed/siglip2_bench.py --warmup 5 --iters 30
```

Raw numbers: `siglip2_speed/bench_results.txt`. Writeup: `siglip2_speed/SIGLIP2_ANALYSIS.md`.
