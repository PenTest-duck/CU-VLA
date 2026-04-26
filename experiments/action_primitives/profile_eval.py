"""Profile a small slice of closed-loop eval to find CPU hotspots.

CloudWatch on SageMaker showed GPU=22%, CPU=114% (1 saturated core out of 4)
for the eval workload. The model forward pass is fast on L40S; somewhere
in the per-rollout loop (pygame env step, image preprocess, action decode,
or text encode) is the bottleneck.

Usage (run locally on M1 mps; ~30 sec for 3 rollouts):
    uv run python -m experiments.action_primitives.profile_eval \
        --checkpoint /path/to/best.pt \
        --data-dir data/phase-b0-lclick \
        --slice multi_btn_generic \
        --device mps --n-rollouts 3

Then inspect:
- top-30 hotspots printed to stdout (sorted by cumulative + total time)
- profile.prof saved for further drill-down (snakeviz profile.prof, or
  python -m pstats profile.prof)
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
from pathlib import Path

from experiments.action_primitives.evaluate import (
    load_model,
    run_closed_loop_eval_b0,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--slice", default="multi_btn_generic",
                        help="Slice to profile (default: multi_btn_generic — has all 4 attrs)")
    parser.add_argument("--device", default="mps",
                        help="mps for M1 / cuda for SageMaker / cpu for fallback")
    parser.add_argument("--n-rollouts", type=int, default=3,
                        help="Few rollouts is enough — we want shape, not stats (default: 3)")
    parser.add_argument("--decode", default="expected", choices=["argmax", "expected"])
    parser.add_argument("--profile-out", type=Path, default=Path("profile.prof"),
                        help="Where to save the cProfile output for drill-down (default: ./profile.prof)")
    parser.add_argument("--top-n", type=int, default=30,
                        help="How many hotspots to print (default: 30)")
    args = parser.parse_args()

    print(f"[profile] loading checkpoint {args.checkpoint}")
    model = load_model(args.checkpoint, args.device)

    print(f"[profile] running {args.n_rollouts} rollouts under cProfile "
          f"(slice={args.slice}, device={args.device})")
    profiler = cProfile.Profile()
    profiler.enable()
    run_closed_loop_eval_b0(
        model=model,
        device=args.device,
        data_dir=args.data_dir,
        slice_name=args.slice,
        n_rollouts=args.n_rollouts,
        decode_mode=args.decode,
    )
    profiler.disable()

    profiler.dump_stats(str(args.profile_out))
    print(f"\n[profile] saved {args.profile_out} (use `snakeviz {args.profile_out}` "
          f"or `python -m pstats {args.profile_out}` to drill down)")

    stats = pstats.Stats(profiler)
    stats.strip_dirs()

    print(f"\n=== TOP {args.top_n} BY CUMULATIVE TIME (where total wall-time goes) ===")
    stats.sort_stats("cumulative").print_stats(args.top_n)

    print(f"\n=== TOP {args.top_n} BY TOTAL TIME (where actual CPU work happens) ===")
    stats.sort_stats("tottime").print_stats(args.top_n)


if __name__ == "__main__":
    main()
