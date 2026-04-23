#!/bin/bash
# Run naflex max_num_patches ablation with each config in its own python process
# to avoid MPS shape-compile cache accumulation across configs.

set -u
cd "$(dirname "$0")/../.."
OUT=playground/siglip2_speed/patch_ablation_clean.jsonl
: > "$OUT"

for N in 16 32 48 64 96 128 160 192 224 256 384 512; do
  echo "=== max_num_patches=$N ==="
  uv run python playground/siglip2_speed/naflex_single.py --max-np "$N" --warmup 15 --iters 40 \
    2>&1 | tee /tmp/ablation_${N}.log | grep "^RESULT:" | sed 's/^RESULT://' >> "$OUT"
done

echo
echo "=== collected results ==="
python3 -c "
import json
rows = [json.loads(l) for l in open('$OUT')]
print(f\"{'max_np':>7}  {'grid':>8}  {'valid':>5}  \"
      f\"{'img mean':>10}  {'img p50':>10}  {'img Hz':>7}  \"
      f\"{'full mean':>10}  {'full p50':>10}  {'full Hz':>8}  {'<=33ms':>7}\")
for r in rows:
    g = r['grid']
    im = r['img']; fu = r['full']
    hits = 'YES' if fu['mean'] < 33.3 else 'no'
    print(f\"{r['max_np']:>7}  {g[0]:>3}x{g[1]:<4d}  {r['valid']:>5}  \"
          f\"{im['mean']:>8.2f}ms  {im['p50']:>8.2f}ms  {1000/im['mean']:>5.1f} Hz  \"
          f\"{fu['mean']:>8.2f}ms  {fu['p50']:>8.2f}ms  {1000/fu['mean']:>6.1f} Hz  {hits:>7}\")

passing = [r for r in rows if r['full']['mean'] < 33.3]
if passing:
    b = max(passing, key=lambda r: r['max_np'])
    g = b['grid']
    print(f\"\\n>> Largest max_num_patches hitting 30 Hz: {b['max_np']} \"
          f\"(grid {g[0]}x{g[1]}={b['valid']} valid), full={b['full']['mean']:.1f}ms = {1000/b['full']['mean']:.1f} Hz\")
else:
    f = min(rows, key=lambda r: r['full']['mean'])
    print(f\"\\n>> No config hit 30 Hz. Fastest: max_np={f['max_np']} at {f['full']['mean']:.1f}ms ({1000/f['full']['mean']:.1f} Hz)\")
"
