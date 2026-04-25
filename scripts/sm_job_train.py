"""SageMaker training-job entrypoint for CU-VLA.

Runs INSIDE the SageMaker training container. The pre-built PyTorch DLC
already has torch + torchvision; SageMaker installs requirements-sagemaker.txt
before this script starts.

Sequence:
  1. Read TRAIN_MODULE, CU_VLA_BRANCH, TRAIN_ARGS from env (set by the
     factory).
  2. Pull WANDB_API_KEY + HF_TOKEN from SSM Parameter Store using the
     IAM execution role's credentials (already injected by SageMaker).
  3. Set WANDB_RUN_ID = SM_TRAINING_JOB_NAME and WANDB_RESUME=allow so
     a spot-reclaim re-launch continues the same wandb run.
  4. git-clone the configured branch into /workspace/CU-VLA, cd in,
     prepend to sys.path so `python -m experiments.foo.train` finds it.
  5. Scan /opt/ml/checkpoints/ for the latest step_*.pt; if found,
     append --resume <path> to the train args.
  6. Force --out-dir /opt/ml/checkpoints so step_*.pt + best.pt land on
     the S3-synced volume.
  7. exec `python -u -m <train_module> <train_args>`.

This script is self-contained (no `from scripts.something import ...`)
because SageMaker tars the source_dir and runs it standalone.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _latest_step_ckpt(ckpt_dir: Path) -> Optional[Path]:
    """Return the latest step_*.pt in ckpt_dir, or None if dir empty/missing.

    Step ckpts are written by train.py with 5-digit zero-padded names
    (step_00010.pt, step_00100.pt, ...) so lex sort == numeric sort.
    """
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        return None
    candidates = sorted(ckpt_dir.glob("step_*.pt"))
    return candidates[-1] if candidates else None


def _load_secrets_from_ssm() -> None:
    """Pull WANDB_API_KEY and HF_TOKEN from SSM Parameter Store."""
    import boto3

    region = os.environ.get("AWS_REGION", "us-west-2")
    ssm = boto3.client("ssm", region_name=region)
    for env_name, param_name in [
        ("WANDB_API_KEY", "/cu-vla/wandb-api-key"),
        ("HF_TOKEN", "/cu-vla/hf-token"),
    ]:
        if env_name in os.environ:
            continue  # already set (e.g. by an explicit override)
        try:
            resp = ssm.get_parameter(Name=param_name, WithDecryption=True)
            os.environ[env_name] = resp["Parameter"]["Value"]
        except Exception as e:
            # Don't crash if a single secret is missing — wandb fails gracefully.
            # train.py will catch HF upload errors. Log loudly so failures are
            # visible in CloudWatch.
            print(f"[sm_job_train] WARN: could not read SSM param {param_name}: {e}", flush=True)


def _configure_wandb_resume() -> None:
    """Set WANDB_RUN_ID + WANDB_RESUME so a spot-reclaim re-launch continues
    the same wandb run instead of starting a new one."""
    sm_job_name = os.environ.get("SM_TRAINING_JOB_NAME")
    if sm_job_name and "WANDB_RUN_ID" not in os.environ:
        os.environ["WANDB_RUN_ID"] = sm_job_name
    os.environ.setdefault("WANDB_RESUME", "allow")


def _git_clone_repo() -> Path:
    """Clone the configured branch into /workspace/CU-VLA and return that path."""
    repo_url = os.environ.get("CU_VLA_REPO_URL", "https://github.com/PenTest-duck/CU-VLA.git")
    branch = os.environ.get("CU_VLA_BRANCH", "main")
    workdir = Path("/workspace/CU-VLA")
    workdir.parent.mkdir(parents=True, exist_ok=True)
    # If running again after a spot reclaim, the previous clone is gone (fresh
    # container). Always do a fresh shallow clone.
    if workdir.exists():
        subprocess.run(["rm", "-rf", str(workdir)], check=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", branch, repo_url, str(workdir)],
        check=True,
    )
    return workdir


def main() -> int:
    train_module = os.environ.get("TRAIN_MODULE")
    if not train_module:
        print("[sm_job_train] FATAL: TRAIN_MODULE env var not set", flush=True)
        return 2

    _load_secrets_from_ssm()
    _configure_wandb_resume()
    workdir = _git_clone_repo()

    # Append the train_args env var (space-separated, shell-style) to the
    # invocation. shlex.split handles quoted values cleanly.
    raw_args = os.environ.get("TRAIN_ARGS", "")
    train_args = shlex.split(raw_args)

    # Force --out-dir to the S3-synced volume so step_*.pt + best.pt persist
    # across spot reclaims. If the user already passed --out-dir, leave it.
    if "--out-dir" not in train_args:
        train_args.extend(["--out-dir", "/opt/ml/checkpoints"])

    # Resume detection.
    ckpt_dir = Path("/opt/ml/checkpoints")
    latest = _latest_step_ckpt(ckpt_dir)
    if latest is not None and "--resume" not in train_args:
        train_args.extend(["--resume", str(latest)])
        print(f"[sm_job_train] resuming from {latest}", flush=True)
    else:
        print(f"[sm_job_train] starting fresh (no step_*.pt in {ckpt_dir})", flush=True)

    # Make the cloned repo importable.
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{workdir}:{env.get('PYTHONPATH', '')}"

    cmd = [sys.executable, "-u", "-m", train_module, *train_args]
    print(f"[sm_job_train] exec: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=workdir, env=env)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
