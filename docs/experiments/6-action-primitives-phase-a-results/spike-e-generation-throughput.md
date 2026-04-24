# Spike E — Generation throughput

**Ran:** 2026-04-24
**Script:** `experiments/action_primitives/measurements/gen_throughput.py`
**Command:** `uv run python -m experiments.action_primitives.measurements.gen_throughput -n 1000`
**Hardware:** MacBook Pro M1, 8 GB RAM. Single Python process.

## Results

```
Episodes:          1000
Total frames:      30000
Wall-clock:        276.85s
eps/s:             3.61  (target: ≥200)
frames/s:          108.4
Avg bytes/episode: 204.2 KB
Projected 24500-ep dataset size: 5.12 GB
```

### Primary finding: 55× below Q7 target

Q7 targeted ≥200 eps/s as the generator throughput floor. The measured single-process throughput is **3.61 eps/s** — a 55× gap. The target is unreachable with the current single-process pygame + PIL JPEG-encode pipeline on M1.

### Secondary finding: 5% label-noise rate

The T8 idempotency hardening added a drift-detection warning when the env never signals success. Of 1000 episodes run here with `seed = episode_id`, **50 emitted `env never signalled success` warnings (5.0%)**. Listed seeds span the full range, consistent with a distribution-level failure mode rather than a single seed outlier.

Root cause analysis:

The plan sets `max_frames_lclick = 30`. An L-click episode consists of:
- move phase (dist / peak_speed_px frames),
- settle phase (1–5 frames from tempo profile),
- press frame (1),
- release frame (1).

For `slow` tempo (`peak_speed_px = 18`, `settle_frames = (2, 5)` → effective 3–5 idle frames), starting from a corner with target on the opposite diagonal:
- distance ≈ 500 px,
- move frames ≈ 500 / 18 ≈ 28,
- settle ≈ 3–5,
- press + release = 2,
- **total ≈ 33–35 frames > 30**.

So the slow-tempo + long-distance combination cannot finish within the 30-frame window. These episodes hit `max_frames`, the expert never reaches `StopIteration`, `done_frame` stays `None`, `env_done_frame` stays `-1`, and `done_gt` is 0 for all 30 rows. The model would learn "never predict done" from these episodes.

The T8 follow-up already emits the warning; it does not prevent the episode from being written.

## Interpretation

**Throughput (primary):**

The 200 eps/s Q7 number was aspirational. The real bottleneck is pygame render + PIL JPEG encode per frame — ~9.2 ms/frame (108.4 fps). For reference:
- Phase A (3000 eps) single-process: ~14 minutes. Acceptable.
- Phase B (24,500 eps) single-process: ~115 minutes (~1.9 h). Acceptable as a one-time cost.
- Multi-process on 8-core M1 with linear scaling: ~30 eps/s expected → Phase B in ~14 min. Acceptable.

Optimizations ranked by leverage-to-risk:
1. **Multi-process generator** (expected ~8×). Low risk (episode-parallel, no shared state except output dir). High value.
2. **Reduce JPEG quality** 90 → 75 (expected ~1.5×). Low risk if training loss holds; test at training time.
3. **Reduce max_frames** 30 → 20 (not viable per label-noise finding — would worsen the 5%).
4. **Replace JPEG with raw numpy** (expected ~2× for generation, but ~6× storage blow-up). Not worth it.
5. **Reduce canvas resolution** 720×450 → 512×320 (expected ~2× render). Breaks SigLIP2 naflex patch math and parity with the design doc's Q7/Q29. Do not pursue.

**Label noise (secondary):**

5% is meaningful: 150 of 3000 Phase A episodes would be unusable as "success demonstrations" (done_gt all-zero, env never succeeded). Options:

1. **Filter at training time** via `env_done_frame != -1` column. Clean, no regeneration needed. The 2850 remaining successful episodes still meets Phase A's data-hungry-enough threshold.
2. **Increase `max_frames_lclick`** 30 → 45 (or 50). Eliminates the long-tail failures at a ~50–67% data-volume cost. Phase A goes from 5 GB → 7.5–8 GB projected. Probably worth it.
3. **Skip slow-tempo long-distance combinations** at generator time (re-roll seed or adjust). Complex, rejects legitimate tempo distribution. Not recommended.

## Recommendation

**For Phase A (landed):**
- `max_frames_lclick` raised 30 → 45 (commit `794f9be`). Residual label-noise rate: 0/100 at N=100; 1/400 (0.25%) at N=400. Below the <1% target.
- Generator `--workers N` flag added in the same commit; `workers=1` (default) is byte-identical to prior serial behaviour. See Phase B section for why multiproc didn't pay off as expected.
- Keep the `env_done_frame != -1` filter at training time as defence-in-depth against the remaining ~0.25% tail (expert itself runs out of frames on extreme slow-tempo + long-distance combinations).
- Ship T9 single-process. Wall-clock estimate at ~2.3 eps/s × 3000 eps ≈ 22 min.

**For Phase B:**

The naive `multiprocessing.Pool(...).imap(worker, episode_ids)` pattern we tried at `workers=4` is **slower than serial** on M1 Python 3.14:

| Config | N | eps/s |
|---|---|---|
| serial (workers=1) | 100 | 2.30 |
| multiproc (workers=4, chunksize=1) | 100 | 1.79 |
| multiproc (workers=4, chunksize=1) | 400 | 2.24 |

Two compounding causes:
1. **`spawn` start method + pygame-ce cold-start per worker.** On macOS Python 3.14, `fork` is forbidden for safety (Objective-C init); every worker does a full SDL init on startup. `pygame-ce 2.5.7 (SDL 2.32.10, ...)` prints once per worker.
2. **IPC payload dominates.** Default `imap` ships each episode's result (~14 MB of JPEG payload: 45 frames × ~307 KB JPEG @ q=90) back through the pipe synchronously. With `chunksize=1`, the parent process is IPC-bound before workers can saturate CPU.

Fix path for Phase B, in order of effort:
1. **`chunksize=8` or `chunksize=16`** on `imap`. Amortizes the pipe round trip. Expected modest improvement; untested in this spike.
2. **Worker-writes-shards redesign.** Each worker owns a contiguous episode range and writes its own parquet shard; parent only collects shard paths (a few bytes per worker). Removes the IPC payload entirely. Expected 4–8× real speedup, but requires restructuring `generate_all`.
3. **Async render + encode pipeline** (most ambitious). Render-to-numpy in the render worker; push raw frames to a separate JPEG-encoder pool. Decouples the two heavy CPU steps. Only worth it if (2) caps out below target.

**Revised Q7 target:** "≥20 eps/s single-process on consumer CPU; Phase B multiproc design is an open design question requiring the worker-writes-shards redesign (or equivalent)." The original 200 eps/s value was aspirational and does not hold for the current pygame + PIL pipeline at ANY parallelism level we've tested.

Consider lowering JPEG quality 90 → 75 after verifying training loss is unchanged on a small held-out set — it would cut per-frame encode time and per-episode bytes roughly in half.

## Next steps

- [x] Raise `max_frames_lclick` from 30 to 45 in `config.py` (commit `794f9be`).
- [x] Re-run N=100 smoke test; residual label-noise rate 0% (well under 1% target).
- [x] Add `--workers` flag to `generate_data.py` (commit `794f9be`; tested for correctness, but did not deliver the hoped-for speedup — see Phase B section).
- [ ] Update Q7 + Q8 in the design doc amendments changelog.
- [ ] T9 full 3000-episode Phase A data generation at `workers=1`.
- [ ] Phase B prep: worker-writes-shards redesign OR `chunksize` tuning of the existing multiproc path. Required before any Phase B-scale data generation (24,500 ep at 2.3 eps/s = ~3 h serial).
