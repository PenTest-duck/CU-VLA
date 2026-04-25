# Experiment 6: Action Primitives (Phase A)

Phase A = four feasibility spikes. See `docs/experiments/6-action-primitives.md`.

## Spike scripts
- `probes/typing_legibility.py` — Spike A (standalone, no training)
- `measurements/gen_throughput.py` — Spike E (measures generator eps/sec)
- `measurements/m1_eval_timing.py` — Spike C (uses Spike B checkpoint)

## Spike B flow
1. `python generate_data.py --primitive lclick -n 3000 -o data/phase-a-lclick/`
2. `python scripts/launch_hf_job_exp6.py -- --data-dir data/phase-a-lclick --epochs 5 --hf-upload-repo PenTest-duck/cu-vla-exp6-phasea-ckpt`
3. `python evaluate.py --checkpoint <ckpt> --primitive lclick --n 200 --visual` (on M1)

See the Phase A plan for full step-by-step.
