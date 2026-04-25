"""Launch Experiment 6 training on SageMaker.

Translates exp6-specific shorthand (e.g. --phase b0) into the right combo
of --branch, --max-run, and forwarded train args, then calls into the
generic launcher.

Usage:
    uv run python scripts/launch_sm_job_exp6.py --phase b0 --spot \\
        -- --epochs 5 --hf-data-repo PenTest-duck/cu-vla-exp6-phaseb0 \\
           --hf-upload-repo PenTest-duck/cu-vla-exp6-phaseb0-ckpt \\
           --wandb-run-name phase-b0-sm-001
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.launch_sm_job import main as launch_main  # noqa: E402


# Per-phase defaults: branch, max_run (per-attempt seconds).
PHASE_DEFAULTS: dict[str, dict[str, object]] = {
    "a": {"branch": "feat/exp6-phase-a", "max_run": 4 * 3600},
    "b0": {"branch": "feat/exp6-phase-b0", "max_run": 4 * 3600},
    "b1": {"branch": "feat/exp6-phase-b1", "max_run": 8 * 3600},
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch Experiment 6 (action_primitives) training on SageMaker.",
        usage="%(prog)s [exp6-opts] -- [train.py opts]",
    )
    parser.add_argument("--phase", choices=PHASE_DEFAULTS.keys(), default="b0",
                        help="Phase shorthand: sets branch + max_run (default: b0)")
    parser.add_argument("--spot", dest="spot", action="store_true", default=True)
    parser.add_argument("--no-spot", dest="spot", action="store_false")
    parser.add_argument("--instance-type", default="ml.g6e.xlarge")
    parser.add_argument("--detach", action="store_true")
    # Allow overriding branch and max-run from the phase preset.
    parser.add_argument("--branch", default=None)
    parser.add_argument("--max-run", type=int, default=None)

    argv = sys.argv[1:]
    if "--" in argv:
        i = argv.index("--")
        own_argv, train_args = argv[:i], argv[i + 1:]
    else:
        own_argv, train_args = argv, []
    args = parser.parse_args(own_argv)

    presets = PHASE_DEFAULTS[args.phase]
    branch = args.branch or presets["branch"]
    max_run = args.max_run or presets["max_run"]

    # Build the generic launcher's argv.
    forwarded = [
        "--train-module", "experiments.action_primitives.train",
        "--experiment", "exp6",
        "--phase", args.phase,
        "--branch", branch,
        "--instance-type", args.instance_type,
        "--max-run", str(max_run),
    ]
    if args.spot:
        forwarded.append("--spot")
    else:
        forwarded.append("--no-spot")
    if args.detach:
        forwarded.append("--detach")
    forwarded.append("--")
    forwarded.extend(train_args)

    return launch_main(forwarded)


if __name__ == "__main__":
    sys.exit(main())
