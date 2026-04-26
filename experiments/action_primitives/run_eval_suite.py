"""Run the full Phase B0 eval suite against a checkpoint, upload JSONs to HF Hub.

Replaces the bash for-loop the user was running locally. Runs in one
SageMaker job (one container start, one dataset download) instead of N
jobs × N container starts. Each per-slice JSON is uploaded to HF Hub
immediately after that slice completes, so a mid-suite crash still
preserves partial results.

Usage (locally on a GPU box):
    uv run python -m experiments.action_primitives.run_eval_suite \
        --ckpt-repo PenTest-duck/cu-vla-exp6-b0-ckpt \
        --data-repo PenTest-duck/cu-vla-exp6-b0-lclick \
        --upload-repo PenTest-duck/cu-vla-exp6-b0-ckpt \
        --device cuda --phase b0 --n-rollouts 200

On SageMaker via the launcher (preferred for cost-efficient batch eval —
~$0.65 on ml.g6e.xlarge spot for 8 evals × 200 rollouts):
    uv run python scripts/launch_sm_job.py \
        --train-module experiments.action_primitives.run_eval_suite \
        --branch feat/exp6-phase-b0 \
        --instance-type ml.g6e.xlarge --spot --max-run 7200 \
        --experiment exp6 --phase b0-eval \
        -- --ckpt-repo ... --data-repo ... --upload-repo ... --device cuda --phase b0
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


# Slices and probes to run. The "none" probe is the standard eval; the
# zero/shuffled/wrong probes are only meaningful on the multi_btn_generic
# slice (per the user's existing local loop).
DEFAULT_SLICES: tuple[str, ...] = (
    "phase_a_holdout",
    "multi_btn_generic",
    "multi_btn_composite",
    "scenario_recovery",
    "adversarial",
)
DEFAULT_PROBES: tuple[str, ...] = ("zero", "shuffled", "wrong")
PROBE_SLICE = "multi_btn_generic"


@dataclass(frozen=True)
class EvalRun:
    """One slice/probe combination to evaluate."""
    slice_name: str
    probe: str  # "none" for the standard eval, else zero/shuffled/wrong
    json_filename: str  # bare filename, no directory


def build_runs(
    slices: tuple[str, ...] = DEFAULT_SLICES,
    probes: tuple[str, ...] = DEFAULT_PROBES,
    probe_slice: str = PROBE_SLICE,
) -> list[EvalRun]:
    """Generate the list of (slice, probe) pairs to run.

    Mirrors the user's bash loop:
      for s in slices: eval --slice $s --instruction-probe none
      for p in probes: eval --slice probe_slice --instruction-probe $p
    """
    runs: list[EvalRun] = []
    for s in slices:
        runs.append(EvalRun(s, "none", f"eval_{s}_none.json"))
    for p in probes:
        runs.append(EvalRun(probe_slice, p, f"eval_{probe_slice}_{p}.json"))
    return runs


def _build_eval_cmd(
    *,
    run: EvalRun,
    checkpoint_path: str,
    data_dir: str,
    json_out: Path,
    device: str,
    phase: str,
    n_rollouts: int,
    decode: str,
) -> list[str]:
    """Build the argv for a single evaluate.py invocation."""
    cmd = [
        sys.executable, "-u", "-m", "experiments.action_primitives.evaluate",
        "--checkpoint", checkpoint_path,
        "--data-dir", data_dir,
        "--n-rollouts", str(n_rollouts),
        "--device", device,
        "--decode", decode,
        "--skip-offline",
        "--phase", phase,
        "--slice", run.slice_name,
        "--report-by-tier",
        "--json-out", str(json_out),
    ]
    if run.probe != "none":
        cmd.extend(["--instruction-probe", run.probe])
    return cmd


def _default_upload_prefix() -> str:
    """Use the SageMaker job name when running in-container, else a UTC
    timestamp. Keeps eval JSONs from different runs separated under
    `evals/<key>/` in the upload repo."""
    sm_job = os.environ.get("SM_TRAINING_JOB_NAME")
    if sm_job:
        return f"evals/{sm_job}"
    return f"evals/local-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt-repo", required=True,
                        help="HF model repo holding the checkpoint to evaluate")
    parser.add_argument("--ckpt-file", default="best.pt",
                        help="Filename of the checkpoint within --ckpt-repo (default: best.pt)")
    parser.add_argument("--data-repo", required=True,
                        help="HF dataset repo to download for eval")
    parser.add_argument("--upload-repo", required=True,
                        help="HF model repo to upload eval JSONs to (typically same as --ckpt-repo)")
    parser.add_argument("--upload-prefix", default=None,
                        help="Path prefix inside --upload-repo (default: evals/<sm-job-name> or evals/local-<utc>)")
    parser.add_argument("--n-rollouts", type=int, default=200)
    parser.add_argument("--device", default="cuda",
                        help="cuda for SageMaker, mps for Apple Silicon, cpu for fallback")
    parser.add_argument("--phase", default="b0", choices=["a", "b0"])
    parser.add_argument("--decode", default="expected", choices=["argmax", "expected"])
    parser.add_argument("--out-dir", default="/tmp/eval-suite",
                        help="Local dir for per-slice JSON files before upload")
    parser.add_argument("--slices", nargs="*", default=list(DEFAULT_SLICES),
                        help="Override the default slice list")
    parser.add_argument("--probes", nargs="*", default=list(DEFAULT_PROBES),
                        help="Override the default probe list")
    parser.add_argument("--probe-slice", default=PROBE_SLICE,
                        help="Slice on which the probe variants run")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Run the suite but don't upload to HF Hub (debug mode)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    upload_prefix = args.upload_prefix or _default_upload_prefix()

    # Lazy imports — keep the module importable on machines without huggingface_hub
    # (e.g. when invoked just to run unit tests).
    from huggingface_hub import hf_hub_download, snapshot_download, upload_file

    print(f"[eval-suite] downloading checkpoint {args.ckpt_repo}/{args.ckpt_file}", flush=True)
    checkpoint_path = hf_hub_download(args.ckpt_repo, args.ckpt_file, repo_type="model")

    print(f"[eval-suite] downloading dataset {args.data_repo}", flush=True)
    data_dir = snapshot_download(args.data_repo, repo_type="dataset")

    runs = build_runs(
        slices=tuple(args.slices),
        probes=tuple(args.probes),
        probe_slice=args.probe_slice,
    )
    print(f"[eval-suite] running {len(runs)} evals → upload to "
          f"{args.upload_repo}/{upload_prefix}/", flush=True)

    n_failures = 0
    n_uploaded = 0
    for i, run in enumerate(runs, 1):
        json_out = out_dir / run.json_filename
        print(f"\n[eval-suite] === {i}/{len(runs)} slice={run.slice_name} probe={run.probe} ===",
              flush=True)
        cmd = _build_eval_cmd(
            run=run,
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            json_out=json_out,
            device=args.device,
            phase=args.phase,
            n_rollouts=args.n_rollouts,
            decode=args.decode,
        )
        result = subprocess.run(cmd)
        if result.returncode != 0:
            n_failures += 1
            print(f"[eval-suite] FAILED slice={run.slice_name} probe={run.probe} "
                  f"(exit {result.returncode}); continuing", file=sys.stderr, flush=True)
            continue

        # Upload per-slice immediately so a mid-suite crash preserves what
        # we already paid GPU time to compute.
        if not args.skip_upload and json_out.exists():
            print(f"[eval-suite] uploading {run.json_filename} → "
                  f"{args.upload_repo}/{upload_prefix}/", flush=True)
            upload_file(
                path_or_fileobj=str(json_out),
                path_in_repo=f"{upload_prefix}/{run.json_filename}",
                repo_id=args.upload_repo,
                repo_type="model",
            )
            n_uploaded += 1

    print(f"\n[eval-suite] done. {len(runs) - n_failures}/{len(runs)} evals succeeded; "
          f"{n_uploaded} uploaded.", flush=True)
    if not args.skip_upload:
        print(f"[eval-suite] results: https://huggingface.co/{args.upload_repo}/tree/main/{upload_prefix}",
              flush=True)
    return 1 if n_failures else 0


if __name__ == "__main__":
    sys.exit(main())
