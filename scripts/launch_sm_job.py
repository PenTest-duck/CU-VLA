"""Generic SageMaker launcher for CU-VLA training jobs.

Mirrors the CLI shape of scripts/launch_hf_job.py. Per-experiment defaults
go in scripts/launch_sm_job_<exp>.py wrappers that call into this module.

Usage:
    uv run python scripts/launch_sm_job.py \\
        --train-module experiments.action_primitives.train \\
        --branch feat/exp6-phase-b0 \\
        --instance-type ml.g6e.xlarge \\
        --spot --max-run 14400 \\
        -- --epochs 5 --hf-data-repo PenTest-duck/cu-vla-exp6-phaseb0
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Allow running as `python scripts/launch_sm_job.py` (direct file path) by
# putting the project root on sys.path. When run as `python -m scripts.launch_sm_job`
# or imported under pytest, this is a harmless no-op.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.sagemaker_trainer import make_trainer  # noqa: E402


def _load_dotenv_if_present() -> None:
    """Load CU_VLA_SM_* config from <repo-root>/.env if present.

    Project-scoped config vs polluting ~/.zshrc. Called from main() (not at
    import time) so unit tests don't inherit the user's local .env values.
    Existing env vars take precedence over .env (so explicit shell exports
    or CI overrides win).
    """
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        # python-dotenv is in deps but if a fresh checkout hasn't run `uv sync`,
        # silently skip rather than crash. The launcher's _resolve_*() helpers
        # will give a clear ERROR if the vars truly aren't set.
        return
    load_dotenv(env_path, override=False)


DEFAULT_REPO_URL = "https://github.com/PenTest-duck/CU-VLA.git"
DEFAULT_INSTANCE = "ml.g6e.xlarge"
FALLBACK_REGION = "us-west-2"


def _default_region() -> str:
    """Pin the SageMaker region. CU_VLA_SM_REGION (from .env or shell) wins;
    falls back to us-west-2. Note: we deliberately do NOT inherit the user's
    boto3 profile region — that often defaults to wherever the user's main
    AWS work happens (e.g. ap-southeast-2), causing CreateTrainingJob to
    fail with ResourceLimitExceeded for instance types that aren't
    available there. Pinning explicitly is safer.
    """
    return os.environ.get("CU_VLA_SM_REGION", FALLBACK_REGION)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a CU-VLA training job on SageMaker.",
        usage="%(prog)s [launcher-opts] -- [train.py opts]",
    )
    parser.add_argument("--train-module", required=True,
                        help="Dotted path of train module, e.g. experiments.action_primitives.train")
    parser.add_argument("--branch", default="main",
                        help="Git branch to clone in the container (default: main)")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL,
                        help=f"Source repo URL (default: {DEFAULT_REPO_URL})")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE,
                        help=f"SageMaker instance type (default: {DEFAULT_INSTANCE})")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--region", default=_default_region(),
                        help=f"AWS region (default: $CU_VLA_SM_REGION or {FALLBACK_REGION})")
    parser.add_argument("--spot", dest="spot", action="store_true", default=True,
                        help="Use spot instances (default; ~70%% cheaper)")
    parser.add_argument("--no-spot", dest="spot", action="store_false",
                        help="Use on-demand instead of spot")
    parser.add_argument("--max-run", type=int, default=None,
                        help="Max seconds per attempt (default: factory's 4h)")
    parser.add_argument("--max-wait", type=int, default=None,
                        help="Max total seconds incl. spot retries (default: 2*max_run)")
    parser.add_argument("--volume-size", type=int, default=None,
                        help="EBS volume GB (default: factory's 100)")
    parser.add_argument("--detach", action="store_true",
                        help="Submit and return immediately; don't stream logs")
    parser.add_argument("--experiment", default="cuvla",
                        help="Experiment name component for job name (e.g. exp6)")
    parser.add_argument("--phase", default=None,
                        help="Optional phase component for job name (e.g. b0, b1)")
    parser.add_argument("--role-arn", default=None,
                        help="IAM role ARN. If omitted, read from CU_VLA_SM_ROLE_ARN env var.")
    parser.add_argument("--s3-bucket", default=None,
                        help="S3 bucket name. If omitted, read from CU_VLA_SM_BUCKET env var.")

    # Anything after `--` is forwarded to the train module.
    if argv is None:
        argv = sys.argv[1:]
    if "--" in argv:
        i = argv.index("--")
        launcher_argv, train_args = argv[:i], argv[i + 1:]
    else:
        launcher_argv, train_args = argv, []
    args = parser.parse_args(launcher_argv)
    args.train_args = train_args
    return args


def build_job_name(*, experiment: str, phase: Optional[str], now_iso: Optional[str] = None) -> str:
    """cu-vla-<exp>[-<phase>]-YYYYMMDD-HHMMSS — sortable, parseable, <=63 chars."""
    if now_iso is None:
        now_iso = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    # Convert YYYYMMDDTHHMMSS -> YYYYMMDD-HHMMSS for readability.
    date, time = now_iso.split("T") if "T" in now_iso else (now_iso[:8], now_iso[8:])
    parts = ["cu-vla", experiment]
    if phase:
        parts.append(f"phase{phase}")
    parts.append(f"{date}-{time}")
    return "-".join(parts)


def validate_remote_branch(repo_url: str, branch: str) -> None:
    """Fail-fast if the branch doesn't exist on the remote — saves a spot
    allocation attempt on the inevitable git-clone failure."""
    result = subprocess.run(
        ["git", "ls-remote", "--exit-code", "--heads", repo_url, branch],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"ERROR: branch '{branch}' does not exist on {repo_url}", file=sys.stderr)
        print(f"git ls-remote stderr: {result.stderr.decode().strip()}", file=sys.stderr)
        sys.exit(1)


def _resolve_role_arn(args: argparse.Namespace) -> str:
    if args.role_arn:
        return args.role_arn
    role = os.environ.get("CU_VLA_SM_ROLE_ARN")
    if not role:
        sys.exit("ERROR: Pass --role-arn or set CU_VLA_SM_ROLE_ARN env var.")
    return role


def _resolve_s3_bucket(args: argparse.Namespace) -> str:
    if args.s3_bucket:
        return args.s3_bucket
    bucket = os.environ.get("CU_VLA_SM_BUCKET")
    if not bucket:
        sys.exit("ERROR: Pass --s3-bucket or set CU_VLA_SM_BUCKET env var.")
    return bucket


def main(argv: list[str] | None = None) -> int:
    _load_dotenv_if_present()
    args = parse_args(argv)
    validate_remote_branch(args.repo_url, args.branch)

    role_arn = _resolve_role_arn(args)
    s3_bucket = _resolve_s3_bucket(args)
    job_name = build_job_name(experiment=args.experiment, phase=args.phase)

    trainer = make_trainer(
        train_module=args.train_module,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        role_arn=role_arn,
        s3_bucket=s3_bucket,
        job_name=job_name,
        branch=args.branch,
        train_args=args.train_args,
        use_spot=args.spot,
        max_run=args.max_run,
        max_wait=args.max_wait,
        volume_size_in_gb=args.volume_size,
        region=args.region,
    )

    print(f"[launch_sm_job] launching job: {job_name}* (SDK appends a suffix)")
    print(f"[launch_sm_job]   instance_type: {args.instance_type}  spot: {args.spot}  branch: {args.branch}")
    print(f"[launch_sm_job]   train_module: {args.train_module}")
    print(f"[launch_sm_job]   train_args: {' '.join(args.train_args) if args.train_args else '(none)'}")

    # V3 ModelTrainer.train(wait=, logs=) — replaces V2 .fit(). logs=True
    # streams CloudWatch to the local terminal (same UX as V2's wait=True).
    trainer.train(wait=not args.detach, logs=not args.detach)

    # Surface the actual SageMaker job name (the SDK appends a -YYYYMMDDHHMMSS
    # suffix to the launcher's job_name; this is what the user / sm_jobs.py
    # needs to inspect the run).
    actual_name = None
    if hasattr(trainer, "_latest_training_job") and trainer._latest_training_job is not None:
        actual_name = getattr(trainer._latest_training_job, "training_job_name", None)
    name_display = actual_name or f"{job_name}* (SDK-suffixed; use 'sm_jobs.py latest' to resolve)"

    if args.detach:
        print(f"[launch_sm_job] detached. Job: {name_display}")
        print(f"[launch_sm_job] Watch logs: uv run python scripts/sm_jobs.py logs latest --follow")
    else:
        print(f"[launch_sm_job] training run finished. Job: {name_display}")
        print(f"[launch_sm_job] Inspect: uv run python scripts/sm_jobs.py status latest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
