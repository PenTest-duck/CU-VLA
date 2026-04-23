#!/bin/bash
# Run action-expert + siglip2 benchmark with each config in its own Python
# process so MPS graph cache from one doesn't pollute another.

set -u
cd "$(dirname "$0")/../.."
OUT=playground/siglip2_speed/action_bench_clean.jsonl
: > "$OUT"

for CFG in A B siglip2; do
  echo "=== $CFG ==="
  uv run python playground/siglip2_speed/action_expert_bench.py \
    --only "$CFG" --warmup 20 --iters 100 \
    2>&1 | tee "/tmp/action_${CFG}.log" | grep "^RESULT:" | sed 's/^RESULT://' >> "$OUT"
done

echo
echo "=== collected ==="
python3 -c "
import json
rows = {json.loads(l)['key']: json.loads(l) for l in open('$OUT')}
order = ['A_3blk', 'B_6blk', 'siglip2']
labels = {
  'A_3blk':  'Action expert A (3 blocks)',
  'B_6blk':  'Action expert B (6 blocks)',
  'siglip2': 'SigLIP2 naflex (max_np=256)',
}
print(f\"{'config':<30} {'params':>9} {'mean':>9} {'median':>9} {'p5':>8} {'p95':>8} {'std':>8} {'Hz(med)':>8}\")
for k in order:
    if k not in rows: continue
    r = rows[k]
    s = r['stats']
    hz = 1000/s['median']
    print(f\"{labels[k]:<30} {r['n_params']/1e6:>7.2f}M {s['mean']:>7.2f}ms {s['median']:>7.2f}ms {s['p5']:>6.2f}ms {s['p95']:>6.2f}ms {s['std']:>6.2f}ms {hz:>7.1f}\")

if all(k in rows for k in order):
    med_sig = rows['siglip2']['stats']['median']
    med_a = rows['A_3blk']['stats']['median']
    med_b = rows['B_6blk']['stats']['median']
    print()
    print('--- Encoder + expert pipeline (median sums) ---')
    print(f'  SigLIP2 + A: {med_sig + med_a:6.2f} ms  ({1000/(med_sig+med_a):5.1f} Hz)')
    print(f'  SigLIP2 + B: {med_sig + med_b:6.2f} ms  ({1000/(med_sig+med_b):5.1f} Hz)')
    print(f'  30 Hz budget = 33.3 ms')
    print(f'  Expert overhead on top of encoder:')
    print(f'    A adds +{med_a:5.2f} ms ({med_a/med_sig*100:.1f}% of encoder)')
    print(f'    B adds +{med_b:5.2f} ms ({med_b/med_sig*100:.1f}% of encoder)')
"
