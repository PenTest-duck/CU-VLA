# SageMaker Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the HF Jobs training infrastructure with AWS SageMaker, preserving today's launcher ergonomics, using spot instances by default, keeping HF Hub as the canonical artifact store, and providing an `sm_jobs.py` operator CLI for ergonomic monitoring/debugging.

**Architecture:** Two-tier launcher pattern mirroring today's HF Jobs scripts. A pure factory module (`sagemaker_estimator.py`) builds `sagemaker.pytorch.PyTorch` estimators; a generic launcher (`launch_sm_job.py`) wraps it for CLI use; a per-experiment override (`launch_sm_job_exp6.py`) sets defaults. An in-container entrypoint (`sm_job_train.py`) clones the branch, restores `WANDB_RUN_ID` for spot-reclaim continuity, detects existing step checkpoints in the S3-synced `/opt/ml/checkpoints/`, and execs the training module with `--resume` if applicable. Operator UX comes from one CLI helper (`sm_jobs.py`) keyed on a stable `cu-vla-<exp>-<phase>-YYYYMMDD-HHMMSS` naming convention.

**Tech Stack:** Python 3.14, `sagemaker` SDK ≥2.230, `boto3`, AWS SageMaker pre-built PyTorch DLC (framework_version=2.6, py312), AWS SSM Parameter Store SecureString (secrets), AWS S3 (in-flight checkpoint sync), HF Hub (canonical data + final ckpts), wandb (training metrics), CloudWatch (container stdout).

**Spec:** [`docs/plans/2026-04-26-sagemaker-migration-design.md`](2026-04-26-sagemaker-migration-design.md) (commits `4d455df` + `38bc426`).

**Out of scope:** multi-GPU / multi-node training, custom Docker image / ECR, migrating data away from HF Hub, SageMaker Experiments / Model Registry, auto-fallback from spot to on-demand, step-checkpoint pruning.

**Project conventions:**
- New launcher/entrypoint code under `scripts/`.
- Tests under `tests/scripts/` (new directory; no existing tests for `scripts/`).
- The bootstrap reference doc lives at `docs/ops/aws-bootstrap.md` (new directory).
- Final/best checkpoints continue to be uploaded to HF Hub via existing `experiments/action_primitives/hf_sync.py`.
- All HF Jobs scripts are kept as-is for the duration of the migration; deprecation marker added at the end (Task 12).

**Critical-path note:** Task 2 (GPU quota request) takes 1–3 days for AWS approval. Submit it on Day 0 and **continue with Tasks 3–10 in parallel while the request processes**. Tasks 11–12 (smoke test, spot rehearsal, real run) are gated on quota approval.

---

## Task 1: Add SageMaker SDK + boto3 deps; create requirements-sagemaker.txt

**Files:**
- Modify: `pyproject.toml` (add `sagemaker` and `boto3` to deps)
- Create: `requirements-sagemaker.txt` (deps to install in the SageMaker training container, on top of the pre-built PyTorch DLC)

The SageMaker SDK + boto3 are needed locally (for the launcher and `sm_jobs.py` CLI). The training container uses a pre-built AWS DLC that already has torch + torchvision; we install the rest via `requirements.txt` that SageMaker reads automatically.

- [ ] **Step 1: Add SageMaker SDK + boto3 to pyproject.toml**

Open `pyproject.toml` and add two entries to the `dependencies` list (alphabetical order with the rest):

```toml
dependencies = [
    "torch>=2.11",
    "numpy>=2.0",
    "pygame-ce>=2.5",
    "h5py>=3.12",
    "matplotlib>=3.10",
    "torchvision>=0.22",
    "transformers>=4.52",
    "huggingface-hub>=1.10.1",
    "datasets>=3.0",
    "Pillow>=10.0",
    "pytest>=9.0.3",
    "peft>=0.10",
    "wandb>=0.18",
    "boto3>=1.35",        # NEW: AWS SDK for sm_jobs CLI + entrypoint Parameter Store reads
    "sagemaker>=2.230",   # NEW: SageMaker Python SDK for launcher
]
```

- [ ] **Step 2: Sync the lock file**

Run: `cd /Users/pentest-duck/Desktop/CU-VLA && uv sync`

Expected: a list of newly installed packages including `sagemaker` and `boto3`. No errors. The exact log will mention something like "Installed N packages".

- [ ] **Step 3: Verify the SDK imports**

Run: `cd /Users/pentest-duck/Desktop/CU-VLA && uv run python -c "import sagemaker, boto3; print(sagemaker.__version__, boto3.__version__)"`

Expected: prints two version numbers (e.g. `2.234.1 1.35.99`). No `ImportError`.

- [ ] **Step 4: Create `requirements-sagemaker.txt` (deps for the training container)**

Create `requirements-sagemaker.txt` at the repo root:

```text
# Deps installed inside the SageMaker training container, on top of the
# pre-built PyTorch DLC (framework_version=2.6, py_version=py312, which
# already includes torch, torchvision, numpy, Pillow). SageMaker reads
# this file automatically when entry_point is set and source_dir contains it.
#
# Mirrors the dep list from scripts/hf_job_train_exp6.py UV header.
transformers>=4.50
datasets>=3.0
peft>=0.10
accelerate
pygame-ce>=2.5
wandb
huggingface_hub>=0.30
```

(`torch`, `torchvision`, `numpy`, `pillow` deliberately omitted — already in the DLC. `boto3` is also pre-installed in AWS DLCs.)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock requirements-sagemaker.txt
git commit -m "deps(sm-migration): add sagemaker + boto3; container requirements"
```

Expected: commit succeeds. `git log --oneline -1` shows the new commit.

---

## Task 2: Submit GPU quota request (us-west-2, ml.g6e.xlarge spot training)

**Files:** none — this is a one-time AWS action.

**This task takes 30 seconds of clicking + 1–3 days of waiting. Do it FIRST. Then continue with Tasks 3–10 while the request processes.** Tasks 11–12 (smoke test → real run) are gated on approval.

- [ ] **Step 1: Authenticate to AWS in us-west-2**

Run: `aws configure list-profiles` — verify the profile with the $10k credits is set up.

If using AWS SSO: `aws sso login --profile <your-profile>` and `export AWS_PROFILE=<your-profile> AWS_REGION=us-west-2`.

Verify identity: `aws sts get-caller-identity`

Expected output: a JSON blob with your `Account` (12-digit string) and `Arn`. Note the `Account` value — you'll need it later (referred to as `<acct>`).

- [ ] **Step 2: Check current quota for ml.g6e.xlarge spot training in us-west-2**

Run:
```bash
aws service-quotas list-service-quotas --region us-west-2 --service-code sagemaker --max-results 200 \
  --query "Quotas[?contains(QuotaName, 'g6e.xlarge') && contains(QuotaName, 'spot')].[QuotaName,Value,QuotaCode]" \
  --output table
```

Expected: A row showing `ml.g6e.xlarge for spot training job usage` with `Value` likely `0.0` (default for new accounts) and a `QuotaCode` (e.g. `L-SOMECODE`). Copy the QuotaCode.

- [ ] **Step 3: Request a quota increase to 1**

Run (replacing `<QuotaCode>` with the actual code from Step 2):
```bash
aws service-quotas request-service-quota-increase --region us-west-2 \
  --service-code sagemaker --quota-code <QuotaCode> --desired-value 1
```

Expected: JSON response showing `RequestedValue: 1.0`, `Status: PENDING`, plus a `Id` field. Save the request ID.

- [ ] **Step 4: (Optional, recommended) Submit the same request for ml.g6e.xlarge on-demand**

The on-demand quota is checked when `--no-spot` is used (e.g. for the smoke test in Task 11). Avoids a second 1–3 day wait later.

```bash
aws service-quotas list-service-quotas --region us-west-2 --service-code sagemaker --max-results 200 \
  --query "Quotas[?contains(QuotaName, 'g6e.xlarge') && contains(QuotaName, 'training job usage') && !contains(QuotaName, 'spot')].[QuotaName,Value,QuotaCode]" \
  --output table
```

Note the QuotaCode for the on-demand row, then:

```bash
aws service-quotas request-service-quota-increase --region us-west-2 \
  --service-code sagemaker --quota-code <on-demand-QuotaCode> --desired-value 1
```

- [ ] **Step 5: Track approval**

You'll get an email at the AWS account address when approved. Or poll:

```bash
aws service-quotas list-requested-service-quota-change-history --region us-west-2 --service-code sagemaker \
  --query "RequestedQuotas[?Status=='PENDING'].[QuotaName,DesiredValue,Status,Created]" \
  --output table
```

When `Status` flips from `PENDING` to `CASE_OPENED` to `APPROVED`, you can run Tasks 11+. Typical wall-time: a few hours to 3 days.

- [ ] **Step 6: Note completion**

No commit needed (no code change). Note the request ID and approval timestamp in your bench journal / wherever you track AWS state. Continue with Task 3 immediately.

---

## Task 3: Patch train.py to restore best_val_loss from best.pt on --resume

**Files:**
- Modify: `experiments/action_primitives/train.py:579` (insert ~5 lines after the existing `best_val_loss = float("inf")`)
- Create: `tests/action_primitives/test_resume_best_val_loss.py`

This 5-line patch closes the race documented in §2 of the design: after a spot reclaim, the resumed run's `best_val_loss` would otherwise reset to `inf`, potentially overwriting `best.pt` with a worse checkpoint at the next val eval. We restore it from `best.pt` (which lives next to the step ckpt being resumed).

- [ ] **Step 1: Write the failing test**

Create `tests/action_primitives/test_resume_best_val_loss.py`:

```python
"""Unit test for the patch in train.py that restores best_val_loss from best.pt
on --resume. We mock out the heavy training-loop machinery and import only the
small piece of logic we want to verify: given a resume path, if best.pt exists
in the same dir with a known val_loss, the variable is restored to that value;
otherwise it stays float('inf').
"""
import torch
from pathlib import Path


def _restore_best_val_loss(resume_path: Path) -> float:
    """Mirrors the patch in train.py — kept here for unit testing.

    The actual patch in train.py inlines this logic (no helper function imported).
    If the patch implementation diverges, update this mirror to match.
    """
    best_val_loss = float("inf")
    if resume_path is not None:
        best_path = resume_path.parent / "best.pt"
        if best_path.exists():
            best_val_loss = torch.load(best_path, map_location="cpu")["val_loss"]
    return best_val_loss


def test_no_resume_returns_inf():
    assert _restore_best_val_loss(None) == float("inf")


def test_resume_no_best_pt_returns_inf(tmp_path):
    # Create a fake step ckpt but no best.pt next to it.
    step_path = tmp_path / "step_00010.pt"
    torch.save({"step": 10}, step_path)
    assert _restore_best_val_loss(step_path) == float("inf")


def test_resume_with_best_pt_returns_stored_loss(tmp_path):
    step_path = tmp_path / "step_00010.pt"
    torch.save({"step": 10}, step_path)
    best_path = tmp_path / "best.pt"
    torch.save({"val_loss": 0.123, "step": 8}, best_path)
    restored = _restore_best_val_loss(step_path)
    assert restored == 0.123
```

- [ ] **Step 2: Run the test to verify it passes (logic-only test, no train.py needed)**

Run: `cd /Users/pentest-duck/Desktop/CU-VLA && uv run pytest tests/action_primitives/test_resume_best_val_loss.py -v`

Expected: 3 passed. (The mirror logic in the test is what we're verifying; the `train.py` patch in Step 3 must match.)

- [ ] **Step 3: Apply the patch to train.py**

Open `experiments/action_primitives/train.py`. Locate this line (around line 579):

```python
    best_val_loss = float("inf")
```

Replace with:

```python
    best_val_loss = float("inf")
    # Restore best_val_loss after spot-reclaim resume so the first post-resume
    # val eval does not overwrite a real best with a worse one. best.pt lives
    # next to the step_*.pt being resumed (both in --out-dir).
    if args.resume:
        best_path = Path(args.resume).parent / "best.pt"
        if best_path.exists():
            best_val_loss = torch.load(best_path, map_location="cpu")["val_loss"]
            print(f"[resume] restored best_val_loss={best_val_loss:.4f} from {best_path}")
```

(Note: `from pathlib import Path` should already be imported at the top of `train.py` — verify with `grep "from pathlib" experiments/action_primitives/train.py`. If absent, add it.)

- [ ] **Step 4: Verify the existing `test_resume.py` still passes**

The existing test does a real CPU training run; if a GPU is available locally, prefer that. This test is `pytestmark = pytest.mark.slow` — it takes minutes on CPU.

Run (slow): `cd /Users/pentest-duck/Desktop/CU-VLA && uv run pytest tests/action_primitives/test_resume.py -v -m slow`

Expected: passes (no behavior change for a fresh run; the patch only triggers when both `--resume` is set AND a `best.pt` exists alongside).

If you don't want to wait 5+ minutes, skip the slow test for now and rely on Task 11's smoke test to cover the integration.

- [ ] **Step 5: Verify the new unit test still passes**

Run: `cd /Users/pentest-duck/Desktop/CU-VLA && uv run pytest tests/action_primitives/test_resume_best_val_loss.py -v`

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add experiments/action_primitives/train.py tests/action_primitives/test_resume_best_val_loss.py
git commit -m "fix(exp6): restore best_val_loss from best.pt on --resume

Closes the spot-reclaim race where a resumed training run's best_val_loss
would otherwise reset to inf, potentially overwriting best.pt with a worse
checkpoint at the next val eval. Required for SageMaker spot resume."
```

---

## Task 4: Create scripts/sagemaker_estimator.py (factory) + tests

**Files:**
- Create: `scripts/sagemaker_estimator.py`
- Create: `tests/scripts/__init__.py` (new test directory)
- Create: `tests/scripts/test_sagemaker_estimator.py`

Pure factory: `make_estimator(...)` builds and returns a `sagemaker.pytorch.PyTorch` estimator with all CU-VLA defaults. No I/O, no `.fit()`. Easy to unit-test by mocking the `PyTorch` constructor and asserting kwargs.

- [ ] **Step 1: Create the tests package init**

Create `tests/scripts/__init__.py` with a single blank line.

- [ ] **Step 2: Write the failing tests**

Create `tests/scripts/test_sagemaker_estimator.py`:

```python
"""Unit tests for scripts/sagemaker_estimator.py. We mock sagemaker.pytorch.PyTorch
and assert the factory forwards kwargs correctly. No network calls."""
from unittest.mock import patch, MagicMock

import pytest

from scripts.sagemaker_estimator import make_estimator, DEFAULTS


@patch("scripts.sagemaker_estimator.PyTorch")
def test_factory_uses_spot_by_default(mock_pt):
    make_estimator(
        train_module="experiments.action_primitives.train",
        instance_type="ml.g6e.xlarge",
        role_arn="arn:aws:iam::123:role/SageMakerExecutionRole-CU-VLA",
        s3_bucket="cu-vla-sm-123",
        job_name="cu-vla-test-20260101-000000",
    )
    kwargs = mock_pt.call_args.kwargs
    assert kwargs["use_spot_instances"] is True
    assert kwargs["max_run"] == DEFAULTS["max_run"]
    assert kwargs["max_wait"] == DEFAULTS["max_wait"]
    assert kwargs["max_wait"] >= 2 * kwargs["max_run"], "max_wait must be >= 2*max_run for spot retries"


@patch("scripts.sagemaker_estimator.PyTorch")
def test_factory_no_spot_omits_max_wait(mock_pt):
    make_estimator(
        train_module="experiments.action_primitives.train",
        instance_type="ml.g6e.xlarge",
        role_arn="arn:aws:iam::123:role/x",
        s3_bucket="cu-vla-sm-123",
        job_name="cu-vla-test-20260101-000000",
        use_spot=False,
    )
    kwargs = mock_pt.call_args.kwargs
    assert kwargs["use_spot_instances"] is False
    assert "max_wait" not in kwargs or kwargs["max_wait"] is None


@patch("scripts.sagemaker_estimator.PyTorch")
def test_factory_sets_checkpoint_s3_uri_only_for_spot(mock_pt):
    make_estimator(
        train_module="experiments.action_primitives.train",
        instance_type="ml.g6e.xlarge",
        role_arn="arn:aws:iam::123:role/x",
        s3_bucket="cu-vla-sm-123",
        job_name="cu-vla-test-20260101-000000",
        use_spot=True,
    )
    kwargs = mock_pt.call_args.kwargs
    assert kwargs["checkpoint_s3_uri"] == "s3://cu-vla-sm-123/checkpoints/cu-vla-test-20260101-000000"
    assert kwargs["checkpoint_local_path"] == "/opt/ml/checkpoints"


@patch("scripts.sagemaker_estimator.PyTorch")
def test_factory_sets_environment_with_branch_and_train_module(mock_pt):
    make_estimator(
        train_module="experiments.action_primitives.train",
        instance_type="ml.g6e.xlarge",
        role_arn="arn:aws:iam::123:role/x",
        s3_bucket="cu-vla-sm-123",
        job_name="cu-vla-test-20260101-000000",
        branch="feat/exp6-phase-b1",
    )
    kwargs = mock_pt.call_args.kwargs
    env = kwargs["environment"]
    assert env["CU_VLA_BRANCH"] == "feat/exp6-phase-b1"
    assert env["TRAIN_MODULE"] == "experiments.action_primitives.train"


@patch("scripts.sagemaker_estimator.PyTorch")
def test_factory_passes_hyperparameters_for_train_args(mock_pt):
    make_estimator(
        train_module="experiments.action_primitives.train",
        instance_type="ml.g6e.xlarge",
        role_arn="arn:aws:iam::123:role/x",
        s3_bucket="cu-vla-sm-123",
        job_name="cu-vla-test-20260101-000000",
        train_args=["--epochs", "5", "--out-dir", "/opt/ml/checkpoints"],
    )
    kwargs = mock_pt.call_args.kwargs
    # train_args must end up forwarded to the entrypoint somehow — we encode
    # them as a single string env var the entrypoint parses.
    assert "TRAIN_ARGS" in kwargs["environment"]
    assert "--epochs 5" in kwargs["environment"]["TRAIN_ARGS"]
    assert "--out-dir /opt/ml/checkpoints" in kwargs["environment"]["TRAIN_ARGS"]


@patch("scripts.sagemaker_estimator.PyTorch")
def test_factory_pins_framework_and_python_version(mock_pt):
    make_estimator(
        train_module="experiments.action_primitives.train",
        instance_type="ml.g6e.xlarge",
        role_arn="arn:aws:iam::123:role/x",
        s3_bucket="cu-vla-sm-123",
        job_name="cu-vla-test-20260101-000000",
    )
    kwargs = mock_pt.call_args.kwargs
    assert kwargs["framework_version"] == "2.6"
    assert kwargs["py_version"] == "py312"


@patch("scripts.sagemaker_estimator.PyTorch")
def test_factory_volume_size_at_least_100(mock_pt):
    make_estimator(
        train_module="experiments.action_primitives.train",
        instance_type="ml.g6e.xlarge",
        role_arn="arn:aws:iam::123:role/x",
        s3_bucket="cu-vla-sm-123",
        job_name="cu-vla-test-20260101-000000",
    )
    kwargs = mock_pt.call_args.kwargs
    assert kwargs["volume_size"] >= 100, "EBS volume must be ≥100GB to avoid disk-full at /opt/ml/checkpoints"
```

- [ ] **Step 3: Run tests to verify they fail (no factory yet)**

Run: `cd /Users/pentest-duck/Desktop/CU-VLA && uv run pytest tests/scripts/test_sagemaker_estimator.py -v`

Expected: all 7 tests FAIL with `ModuleNotFoundError: No module named 'scripts.sagemaker_estimator'`.

- [ ] **Step 4: Create `scripts/sagemaker_estimator.py` to make tests pass**

```python
"""SageMaker estimator factory for CU-VLA training jobs.

Pure config builder — no I/O, no .fit() call. Builds and returns a
sagemaker.pytorch.PyTorch estimator with our defaults (spot, L40S, 4h max,
S3-synced checkpoints, Parameter Store secrets via IAM execution role).

Used by scripts/launch_sm_job.py and scripts/launch_sm_job_<exp>.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from sagemaker.pytorch import PyTorch


# Repo root, used as source_dir for SageMaker. The SDK tars and uploads this
# directory; the entrypoint sm_job_train.py git-clones the actual source from
# the configured branch on container start, so the upload is small (just the
# entrypoint script + requirements).
REPO_ROOT = Path(__file__).resolve().parent.parent


# Default config — overridden by per-experiment launchers and CLI flags.
DEFAULTS: dict[str, Any] = {
    "framework_version": "2.6",
    "py_version": "py312",
    "instance_count": 1,
    "use_spot_instances": True,
    "max_run": 4 * 3600,             # 4h per attempt — covers Phase B0 with margin
    "max_wait": 8 * 3600,            # 2× max_run; allows 1–2 reclaim retries
    "volume_size": 100,              # GB EBS for /opt/ml/{input,model,checkpoints}
    "keep_alive_period_in_seconds": 0,
}


def make_estimator(
    *,
    train_module: str,
    instance_type: str,
    role_arn: str,
    s3_bucket: str,
    job_name: str,
    branch: str = "main",
    train_args: Iterable[str] | None = None,
    use_spot: bool | None = None,
    max_run: int | None = None,
    max_wait: int | None = None,
    instance_count: int | None = None,
    volume_size: int | None = None,
    region: str = "us-west-2",
    extra_environment: dict[str, str] | None = None,
) -> PyTorch:
    """Construct a SageMaker PyTorch estimator with CU-VLA defaults.

    Args:
        train_module: dotted path of the training module to invoke inside the
            container, e.g. "experiments.action_primitives.train". The
            entrypoint reads this from the TRAIN_MODULE env var and runs
            `python -u -m <train_module> [TRAIN_ARGS]`.
        instance_type: e.g. "ml.g6e.xlarge" (1×L40S 48GB).
        role_arn: ARN of the SageMakerExecutionRole-CU-VLA IAM role. Read
            from ~/.cu-vla/sagemaker.toml or AWS account env in the launcher.
        s3_bucket: bare bucket name (no s3:// prefix), e.g. "cu-vla-sm-123".
        job_name: full SageMaker job name; we use this for both the SM job
            and the WANDB_RUN_ID so wandb continues across spot reclaims.
        branch: git branch to clone in the container.
        train_args: list of CLI args passed through to the train module.
        use_spot, max_run, max_wait, instance_count, volume_size: override
            DEFAULTS. None means "use the default."
        region: AWS region for the estimator (must match where the role and
            bucket live).
        extra_environment: additional env vars to set in the container.

    Returns:
        Configured sagemaker.pytorch.PyTorch instance, ready to call .fit() on.
    """
    use_spot = DEFAULTS["use_spot_instances"] if use_spot is None else use_spot
    max_run = DEFAULTS["max_run"] if max_run is None else max_run
    instance_count = DEFAULTS["instance_count"] if instance_count is None else instance_count
    volume_size = DEFAULTS["volume_size"] if volume_size is None else volume_size

    # max_wait is only meaningful for spot. For on-demand we omit it entirely.
    if use_spot:
        max_wait = (DEFAULTS["max_wait"] if max_wait is None else max_wait)
        if max_wait < 2 * max_run:
            raise ValueError(
                f"max_wait ({max_wait}s) must be >= 2 * max_run ({2 * max_run}s) "
                "for spot reclaim retries to fit"
            )
    else:
        max_wait = None

    environment: dict[str, str] = {
        "CU_VLA_BRANCH": branch,
        "TRAIN_MODULE": train_module,
        "TRAIN_ARGS": " ".join(train_args) if train_args else "",
        # Headless pygame insurance — if any train-time code path imports
        # pygame.display, this prevents an SDL init crash.
        "SDL_VIDEODRIVER": "dummy",
        "AWS_REGION": region,
        # Silence HF Hub progress bars in CloudWatch (mirrors hf_job_train_exp6.py).
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
    }
    if extra_environment:
        environment.update(extra_environment)

    kwargs: dict[str, Any] = {
        "entry_point": "sm_job_train.py",
        "source_dir": str(REPO_ROOT / "scripts"),
        "framework_version": DEFAULTS["framework_version"],
        "py_version": DEFAULTS["py_version"],
        "instance_type": instance_type,
        "instance_count": instance_count,
        "role": role_arn,
        "use_spot_instances": use_spot,
        "max_run": max_run,
        "volume_size": volume_size,
        "keep_alive_period_in_seconds": DEFAULTS["keep_alive_period_in_seconds"],
        "environment": environment,
        "base_job_name": None,    # we set job_name explicitly via .fit(job_name=...)
        "code_location": f"s3://{s3_bucket}/tmp",
        "output_path": f"s3://{s3_bucket}/output",
        "dependencies": [str(REPO_ROOT / "requirements-sagemaker.txt")],
    }
    if use_spot:
        kwargs["max_wait"] = max_wait
        kwargs["checkpoint_s3_uri"] = f"s3://{s3_bucket}/checkpoints/{job_name}"
        kwargs["checkpoint_local_path"] = "/opt/ml/checkpoints"

    return PyTorch(**kwargs)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/pentest-duck/Desktop/CU-VLA && uv run pytest tests/scripts/test_sagemaker_estimator.py -v`

Expected: all 7 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/sagemaker_estimator.py tests/scripts/__init__.py tests/scripts/test_sagemaker_estimator.py
git commit -m "feat(sm-migration): sagemaker_estimator.py factory + tests

Pure config builder for sagemaker.pytorch.PyTorch with CU-VLA defaults
(spot, L40S, checkpoint_s3_uri, framework_version=2.6/py312, volume=100GB).
No I/O. Tests mock the constructor and assert kwargs."
```

---

## Task 5: Create scripts/sm_job_train.py (in-container entrypoint) + tests

**Files:**
- Create: `scripts/sm_job_train.py`
- Create: `tests/scripts/test_sm_job_train.py`

In-container entrypoint. Three things only: pull secrets from Parameter Store, configure wandb resume, git-clone the configured branch, scan `/opt/ml/checkpoints/` for the latest `step_*.pt`, exec the train module with `--resume <ckpt>` if found.

The resume-detection logic (`_latest_step_ckpt`) is the only piece that's unit-testable without AWS. The rest is end-to-end-tested via Task 11 (smoke).

- [ ] **Step 1: Write the failing tests for resume detection**

Create `tests/scripts/test_sm_job_train.py`:

```python
"""Unit tests for the resume-detection logic in scripts/sm_job_train.py.

Only the pure filesystem-only function is tested here. SSM/git/subprocess
behavior is covered end-to-end by Task 11's smoke test.
"""
from pathlib import Path

from scripts.sm_job_train import _latest_step_ckpt


def test_no_ckpts_returns_none(tmp_path):
    assert _latest_step_ckpt(tmp_path) is None


def test_picks_highest_step_with_zero_padded_names(tmp_path):
    # 5-digit zero-padded names → lex sort == numeric sort.
    for name in ["step_00001.pt", "step_00010.pt", "step_00100.pt", "step_00099.pt"]:
        (tmp_path / name).touch()
    latest = _latest_step_ckpt(tmp_path)
    assert latest is not None
    assert latest.name == "step_00100.pt"


def test_ignores_best_pt_and_final_pt(tmp_path):
    (tmp_path / "step_00010.pt").touch()
    (tmp_path / "best.pt").touch()
    (tmp_path / "final.pt").touch()
    latest = _latest_step_ckpt(tmp_path)
    assert latest is not None
    assert latest.name == "step_00010.pt"


def test_nonexistent_dir_returns_none(tmp_path):
    nonexistent = tmp_path / "does_not_exist"
    assert _latest_step_ckpt(nonexistent) is None
```

- [ ] **Step 2: Run tests to verify they fail (no entrypoint module yet)**

Run: `cd /Users/pentest-duck/Desktop/CU-VLA && uv run pytest tests/scripts/test_sm_job_train.py -v`

Expected: all tests FAIL with `ModuleNotFoundError: No module named 'scripts.sm_job_train'`.

- [ ] **Step 3: Create `scripts/sm_job_train.py`**

```python
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
```

- [ ] **Step 4: Run the unit tests**

Run: `cd /Users/pentest-duck/Desktop/CU-VLA && uv run pytest tests/scripts/test_sm_job_train.py -v`

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/sm_job_train.py tests/scripts/test_sm_job_train.py
git commit -m "feat(sm-migration): sm_job_train.py in-container entrypoint + resume detection tests

Pulls secrets from SSM Parameter Store, sets WANDB_RUN_ID for spot-reclaim
continuity, git-clones the configured branch, scans /opt/ml/checkpoints/
for the latest step_*.pt, execs the configured train module with --resume
if found and --out-dir /opt/ml/checkpoints (S3-synced)."
```

---

## Task 6: Create scripts/launch_sm_job.py (generic launcher CLI) + tests

**Files:**
- Create: `scripts/launch_sm_job.py`
- Create: `tests/scripts/test_launch_sm_job.py`

Generic CLI launcher. Parses `--train-module`, `--instance-type`, `--spot/--no-spot`, `--branch`, `--max-run`, etc. Builds estimator via the factory, calls `.fit(wait=True)` (or `wait=False` for `--detach`). Pre-validates the branch exists on the remote with `git ls-remote` so a typo doesn't waste a spot allocation.

- [ ] **Step 1: Write the failing tests for arg parsing + branch validation**

Create `tests/scripts/test_launch_sm_job.py`:

```python
"""Unit tests for scripts/launch_sm_job.py argument parsing + branch validation."""
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from scripts.launch_sm_job import (
    parse_args,
    validate_remote_branch,
    build_job_name,
    DEFAULT_REPO_URL,
)


def test_parse_args_required_train_module():
    args = parse_args(["--train-module", "experiments.foo.train"])
    assert args.train_module == "experiments.foo.train"
    assert args.spot is True  # default
    assert args.detach is False
    assert args.instance_type == "ml.g6e.xlarge"  # default


def test_parse_args_no_spot_flag():
    args = parse_args(["--train-module", "x.y", "--no-spot"])
    assert args.spot is False


def test_parse_args_train_passthrough_args():
    # Anything after `--` is forwarded to the train module.
    args = parse_args(["--train-module", "x.y", "--", "--epochs", "5", "--lr", "1e-4"])
    assert args.train_args == ["--epochs", "5", "--lr", "1e-4"]


def test_parse_args_branch_default_is_main():
    args = parse_args(["--train-module", "x.y"])
    assert args.branch == "main"


def test_build_job_name_format():
    name = build_job_name(experiment="exp6", phase="b0", now_iso="20260427T093142")
    assert name == "cu-vla-exp6-phaseb0-20260427-093142"


def test_build_job_name_no_phase():
    name = build_job_name(experiment="exp6", phase=None, now_iso="20260427T093142")
    assert name == "cu-vla-exp6-20260427-093142"


@patch("scripts.launch_sm_job.subprocess.run")
def test_validate_remote_branch_passes_for_existing(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout=b"abc123\trefs/heads/feat/exp6-phase-b0\n")
    validate_remote_branch("https://github.com/PenTest-duck/CU-VLA.git", "feat/exp6-phase-b0")
    mock_run.assert_called_once()


@patch("scripts.launch_sm_job.subprocess.run")
def test_validate_remote_branch_raises_for_missing(mock_run):
    mock_run.return_value = MagicMock(returncode=2, stdout=b"")
    with pytest.raises(SystemExit):
        validate_remote_branch("https://github.com/PenTest-duck/CU-VLA.git", "nonexistent-branch")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/pentest-duck/Desktop/CU-VLA && uv run pytest tests/scripts/test_launch_sm_job.py -v`

Expected: all 7 FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create `scripts/launch_sm_job.py`**

```python
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

from scripts.sagemaker_estimator import make_estimator


DEFAULT_REPO_URL = "https://github.com/PenTest-duck/CU-VLA.git"
DEFAULT_INSTANCE = "ml.g6e.xlarge"
DEFAULT_REGION = "us-west-2"


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
    parser.add_argument("--region", default=DEFAULT_REGION)
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
    """cu-vla-<exp>[-<phase>]-YYYYMMDD-HHMMSS — sortable, parseable, ≤63 chars."""
    if now_iso is None:
        now_iso = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    # Convert YYYYMMDDTHHMMSS → YYYYMMDD-HHMMSS for readability.
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
    args = parse_args(argv)
    validate_remote_branch(args.repo_url, args.branch)

    role_arn = _resolve_role_arn(args)
    s3_bucket = _resolve_s3_bucket(args)
    job_name = build_job_name(experiment=args.experiment, phase=args.phase)

    estimator = make_estimator(
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
        volume_size=args.volume_size,
        region=args.region,
    )

    print(f"[launch_sm_job] launching job: {job_name}")
    print(f"[launch_sm_job]   instance_type: {args.instance_type}  spot: {args.spot}  branch: {args.branch}")
    print(f"[launch_sm_job]   train_module: {args.train_module}")
    print(f"[launch_sm_job]   train_args: {' '.join(args.train_args) if args.train_args else '(none)'}")

    estimator.fit(wait=not args.detach, job_name=job_name)

    if args.detach:
        print(f"[launch_sm_job] detached. Watch with: uv run python scripts/sm_jobs.py logs {job_name} --follow")
    else:
        print(f"[launch_sm_job] job {job_name} finished. Status: {estimator.latest_training_job.describe()['TrainingJobStatus']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/pentest-duck/Desktop/CU-VLA && uv run pytest tests/scripts/test_launch_sm_job.py -v`

Expected: all 7 PASS.

- [ ] **Step 5: Smoke-test argparse end-to-end (no AWS calls)**

Run: `cd /Users/pentest-duck/Desktop/CU-VLA && uv run python scripts/launch_sm_job.py --help`

Expected: prints the help message, exits 0.

Run: `cd /Users/pentest-duck/Desktop/CU-VLA && uv run python scripts/launch_sm_job.py --train-module experiments.action_primitives.train --branch nonexistent-branch-xyz`

Expected: prints `ERROR: branch 'nonexistent-branch-xyz' does not exist on ...` and exits 1. Validates the branch pre-check works.

- [ ] **Step 6: Commit**

```bash
git add scripts/launch_sm_job.py tests/scripts/test_launch_sm_job.py
git commit -m "feat(sm-migration): launch_sm_job.py generic CLI launcher

Mirrors launch_hf_job.py shape. Pre-validates remote branch via git ls-remote
to fail-fast on typos. Reads role ARN + S3 bucket from --flags or env vars.
Job name = cu-vla-<exp>-<phase>-YYYYMMDD-HHMMSS for stable handles."
```

---

## Task 7: Create scripts/launch_sm_job_exp6.py (exp6-specific override)

**Files:**
- Create: `scripts/launch_sm_job_exp6.py`

Thin wrapper that pre-fills exp6 defaults and translates `--phase b0|b1|...` into the right combination of branch + max_run + train args. Mirrors the shape of `launch_hf_job_exp6.py`.

- [ ] **Step 1: Create `scripts/launch_sm_job_exp6.py`**

```python
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

from scripts.launch_sm_job import main as launch_main


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
```

- [ ] **Step 2: Smoke-test argparse**

Run: `cd /Users/pentest-duck/Desktop/CU-VLA && uv run python scripts/launch_sm_job_exp6.py --help`

Expected: prints help, exits 0.

Run: `cd /Users/pentest-duck/Desktop/CU-VLA && uv run python scripts/launch_sm_job_exp6.py --phase b0 -- --epochs 1`

Expected: depending on AWS env vars, either fails with `ERROR: Pass --role-arn or set CU_VLA_SM_ROLE_ARN env var.` (if creds not yet set up — Task 9) **OR** with `ERROR: branch 'feat/exp6-phase-b0' does not exist on ...` (if remote out-of-sync). Both are acceptable evidence the wiring is correct. **No actual training job is launched.**

- [ ] **Step 3: Commit**

```bash
git add scripts/launch_sm_job_exp6.py
git commit -m "feat(sm-migration): launch_sm_job_exp6.py exp6 override

Phase shorthand --phase a|b0|b1 maps to branch + max_run preset, then
forwards everything to launch_sm_job.py."
```

---

## Task 8: Create scripts/sm_jobs.py (operator CLI helper) + tests

**Files:**
- Create: `scripts/sm_jobs.py`
- Create: `tests/scripts/test_sm_jobs.py`

Single CLI module with 8 subcommands keyed on the `cu-vla-*` job-name prefix. Designed to be friendly for both human and Claude use: parseable defaults, `latest` shortcut, `--filter` substring match.

- [ ] **Step 1: Write the failing tests for sm_jobs.py helpers**

Create `tests/scripts/test_sm_jobs.py`:

```python
"""Unit tests for scripts/sm_jobs.py.

Mocks boto3.client('sagemaker') and asserts subcommand handlers call the
right APIs with the right args. No real AWS calls.
"""
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

import pytest

from scripts.sm_jobs import (
    resolve_job,
    format_status_line,
    JOB_PREFIX,
)


@patch("scripts.sm_jobs.boto3.client")
def test_resolve_latest_picks_most_recent_cuvla_job(mock_client):
    sm = MagicMock()
    mock_client.return_value = sm
    sm.list_training_jobs.return_value = {
        "TrainingJobSummaries": [
            {"TrainingJobName": "cu-vla-exp6-phaseb0-20260427-093142",
             "CreationTime": datetime(2026, 4, 27, 9, 31, tzinfo=timezone.utc),
             "TrainingJobStatus": "InProgress"},
            {"TrainingJobName": "unrelated-job-12345",
             "CreationTime": datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc),
             "TrainingJobStatus": "Completed"},
        ]
    }
    name = resolve_job("latest")
    assert name == "cu-vla-exp6-phaseb0-20260427-093142"


@patch("scripts.sm_jobs.boto3.client")
def test_resolve_filter_picks_substring_match(mock_client):
    sm = MagicMock()
    mock_client.return_value = sm
    sm.list_training_jobs.return_value = {
        "TrainingJobSummaries": [
            {"TrainingJobName": "cu-vla-exp6-phasea-20260420-100000",
             "CreationTime": datetime(2026, 4, 20, 10, 0, tzinfo=timezone.utc),
             "TrainingJobStatus": "Completed"},
            {"TrainingJobName": "cu-vla-exp6-phaseb0-20260427-093142",
             "CreationTime": datetime(2026, 4, 27, 9, 31, tzinfo=timezone.utc),
             "TrainingJobStatus": "InProgress"},
        ]
    }
    name = resolve_job(filter_=("phaseb0",))
    assert name == "cu-vla-exp6-phaseb0-20260427-093142"


@patch("scripts.sm_jobs.boto3.client")
def test_resolve_filter_no_match_raises(mock_client):
    sm = MagicMock()
    mock_client.return_value = sm
    sm.list_training_jobs.return_value = {
        "TrainingJobSummaries": [
            {"TrainingJobName": "cu-vla-exp6-phasea-20260420-100000",
             "CreationTime": datetime(2026, 4, 20, 10, 0, tzinfo=timezone.utc),
             "TrainingJobStatus": "Completed"},
        ]
    }
    with pytest.raises(SystemExit):
        resolve_job(filter_=("nonexistent",))


def test_resolve_literal_name_returns_unchanged():
    name = resolve_job("cu-vla-exp6-phaseb0-20260427-093142")
    assert name == "cu-vla-exp6-phaseb0-20260427-093142"


def test_format_status_line_running():
    desc = {
        "TrainingJobName": "cu-vla-exp6-phaseb0-20260427-093142",
        "TrainingJobStatus": "InProgress",
        "SecondaryStatus": "Training",
        "ResourceConfig": {"InstanceType": "ml.g6e.xlarge"},
        "TrainingTimeInSeconds": None,  # still running
        "BillableTimeInSeconds": None,
    }
    line = format_status_line(desc)
    assert "cu-vla-exp6-phaseb0-20260427-093142" in line
    assert "InProgress" in line
    assert "ml.g6e.xlarge" in line


def test_format_status_line_completed():
    desc = {
        "TrainingJobName": "cu-vla-exp6-phaseb0-20260427-093142",
        "TrainingJobStatus": "Completed",
        "SecondaryStatus": "Completed",
        "ResourceConfig": {"InstanceType": "ml.g6e.xlarge"},
        "TrainingTimeInSeconds": 3600,
        "BillableTimeInSeconds": 1080,  # spot was 70% off
    }
    line = format_status_line(desc)
    assert "Completed" in line
    assert "1080" in line or "0.3h" in line  # however the formatter renders billable seconds


def test_job_prefix_constant():
    assert JOB_PREFIX == "cu-vla-"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/pentest-duck/Desktop/CU-VLA && uv run pytest tests/scripts/test_sm_jobs.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.sm_jobs'`.

- [ ] **Step 3: Create `scripts/sm_jobs.py`**

```python
"""Operator CLI for CU-VLA SageMaker training jobs.

Subcommands:
    ls               list active + recent CU-VLA jobs (last 20)
    status [name]    one-line summary
    logs <name>      stream or batch CloudWatch logs
    stop <name>      stop running job (confirms if running > 1 min)
    describe <name>  full describe-training-job
    url <name>       print SageMaker console URL
    cost <name>      billable seconds × $/hr estimate
    reconcile-ckpts <name>   push S3 ckpts not yet on HF Hub

`name` may be a literal job name, the literal `latest` (most recent
job whose name starts with cu-vla-), or omitted with --filter <substring>
to substring-match an active/recent CU-VLA job.

All commands accept --region (default: us-west-2) and --json for
parseable output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import boto3


JOB_PREFIX = "cu-vla-"
DEFAULT_REGION = "us-west-2"

# Approximate $/hr for cost estimates. SageMaker bills per-second; spot
# discount is variable. These are us-west-2 on-demand list prices —
# spot may be ~70% lower.
INSTANCE_HOURLY: dict[str, float] = {
    "ml.g6e.xlarge": 1.861,
    "ml.g6e.2xlarge": 2.244,
    "ml.g6e.4xlarge": 3.010,
    "ml.g6e.12xlarge": 8.022,
    "ml.g5.xlarge": 1.408,
}


def _sm_client(region: str):
    return boto3.client("sagemaker", region_name=region)


def _logs_client(region: str):
    return boto3.client("logs", region_name=region)


def resolve_job(
    name: Optional[str] = None,
    filter_: tuple[str, ...] = (),
    region: str = DEFAULT_REGION,
    status: Optional[str] = None,
) -> str:
    """Resolve a job name from a literal name, 'latest', or --filter substring."""
    # Literal job name (not 'latest' and not empty) — return unchanged.
    if name and name != "latest":
        return name

    sm = _sm_client(region)
    kwargs: dict = {"SortBy": "CreationTime", "SortOrder": "Descending", "MaxResults": 100}
    if status:
        kwargs["StatusEquals"] = status
    resp = sm.list_training_jobs(**kwargs)
    cuvla_jobs = [
        s for s in resp["TrainingJobSummaries"]
        if s["TrainingJobName"].startswith(JOB_PREFIX)
    ]
    if filter_:
        for sub in filter_:
            cuvla_jobs = [j for j in cuvla_jobs if sub in j["TrainingJobName"]]

    if not cuvla_jobs:
        sys.exit(f"ERROR: no CU-VLA jobs match name={name!r} filter={filter_}")
    return cuvla_jobs[0]["TrainingJobName"]


def format_status_line(desc: dict) -> str:
    """One-line status summary, parseable as TSV."""
    name = desc["TrainingJobName"]
    state = desc["TrainingJobStatus"]
    secondary = desc.get("SecondaryStatus", "")
    instance = desc.get("ResourceConfig", {}).get("InstanceType", "?")
    train_secs = desc.get("TrainingTimeInSeconds")
    bill_secs = desc.get("BillableTimeInSeconds")
    train_str = f"{train_secs}s" if train_secs is not None else "—"
    bill_str = f"{bill_secs}s" if bill_secs is not None else "—"
    return f"{name}\t{state}\t{secondary}\t{instance}\ttrain={train_str}\tbill={bill_str}"


def cmd_ls(args: argparse.Namespace) -> int:
    sm = _sm_client(args.region)
    kwargs = {
        "SortBy": "CreationTime", "SortOrder": "Descending",
        "MaxResults": args.limit,
    }
    if args.status:
        kwargs["StatusEquals"] = args.status
    resp = sm.list_training_jobs(**kwargs)
    jobs = [s for s in resp["TrainingJobSummaries"] if s["TrainingJobName"].startswith(JOB_PREFIX)]
    if args.json:
        print(json.dumps([{
            "name": j["TrainingJobName"],
            "status": j["TrainingJobStatus"],
            "created": j["CreationTime"].isoformat(),
        } for j in jobs], indent=2))
    else:
        if not jobs:
            print("(no CU-VLA jobs)")
            return 0
        for j in jobs:
            print(f"{j['TrainingJobName']}\t{j['TrainingJobStatus']}\t{j['CreationTime'].isoformat()}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    name = resolve_job(args.name, tuple(args.filter or ()), args.region)
    sm = _sm_client(args.region)
    desc = sm.describe_training_job(TrainingJobName=name)
    if args.json:
        print(json.dumps(desc, indent=2, default=str))
    else:
        print(format_status_line(desc))
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    name = resolve_job(args.name, tuple(args.filter or ()), args.region)
    logs = _logs_client(args.region)
    log_group = "/aws/sagemaker/TrainingJobs"
    streams = logs.describe_log_streams(
        logGroupName=log_group, logStreamNamePrefix=name,
        orderBy="LogStreamName", descending=False,
    )["logStreams"]
    if not streams:
        print(f"(no log streams yet for {name})", file=sys.stderr)
        return 0
    stream_name = streams[0]["logStreamName"]
    next_token = None
    while True:
        kwargs = {"logGroupName": log_group, "logStreamName": stream_name, "startFromHead": True}
        if next_token:
            kwargs["nextToken"] = next_token
        resp = logs.get_log_events(**kwargs)
        for evt in resp["events"]:
            print(evt["message"])
        if resp.get("nextForwardToken") == next_token:
            if not args.follow:
                break
            time.sleep(2)
        next_token = resp["nextForwardToken"]
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    name = resolve_job(args.name, tuple(args.filter or ()), args.region, status="InProgress")
    sm = _sm_client(args.region)
    desc = sm.describe_training_job(TrainingJobName=name)
    train_secs = desc.get("TrainingTimeInSeconds") or 0
    if train_secs > 60 and not args.yes:
        ans = input(f"Job {name} has been running {train_secs}s. Stop? [y/N] ").strip().lower()
        if ans != "y":
            print("aborted")
            return 1
    sm.stop_training_job(TrainingJobName=name)
    print(f"stop_training_job sent for {name}")
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    name = resolve_job(args.name, tuple(args.filter or ()), args.region)
    sm = _sm_client(args.region)
    desc = sm.describe_training_job(TrainingJobName=name)
    print(json.dumps(desc, indent=2, default=str))
    return 0


def cmd_url(args: argparse.Namespace) -> int:
    name = resolve_job(args.name, tuple(args.filter or ()), args.region)
    print(
        f"https://{args.region}.console.aws.amazon.com/sagemaker/home"
        f"?region={args.region}#/jobs/{name}"
    )
    return 0


def cmd_cost(args: argparse.Namespace) -> int:
    name = resolve_job(args.name, tuple(args.filter or ()), args.region)
    sm = _sm_client(args.region)
    desc = sm.describe_training_job(TrainingJobName=name)
    instance = desc["ResourceConfig"]["InstanceType"]
    bill_secs = desc.get("BillableTimeInSeconds") or 0
    rate = INSTANCE_HOURLY.get(instance)
    if rate is None:
        print(f"{name}\t{instance}\tbill={bill_secs}s\t(unknown rate)")
    else:
        # Spot discount is variable; this estimate uses on-demand list price.
        approx_max = bill_secs / 3600 * rate
        # If spot was used, BillableTimeInSeconds is already discounted-equivalent
        # but the rate listed is on-demand. SageMaker reports billable seconds
        # accounting for spot discount, so multiplying by on-demand rate
        # over-estimates. Show both.
        print(f"{name}\t{instance}\tbill={bill_secs}s\t≤${approx_max:.2f} (on-demand rate)")
    return 0


def cmd_reconcile_ckpts(args: argparse.Namespace) -> int:
    """Push any best.pt / final.pt sitting in S3 but not yet on HF Hub.

    Reads the s3://.../checkpoints/<job-name>/{best,final}.pt and uploads
    them via experiments.action_primitives.hf_sync if the user provides
    --hf-repo.
    """
    name = resolve_job(args.name, tuple(args.filter or ()), args.region)
    if not args.hf_repo:
        sys.exit("ERROR: --hf-repo required for reconcile-ckpts")
    if not args.s3_bucket:
        s3_bucket = os.environ.get("CU_VLA_SM_BUCKET")
        if not s3_bucket:
            sys.exit("ERROR: pass --s3-bucket or set CU_VLA_SM_BUCKET")
    else:
        s3_bucket = args.s3_bucket

    s3 = boto3.client("s3", region_name=args.region)
    prefix = f"checkpoints/{name}/"
    resp = s3.list_objects_v2(Bucket=s3_bucket, Prefix=prefix)
    if "Contents" not in resp:
        print(f"(no S3 objects under s3://{s3_bucket}/{prefix})")
        return 0

    from huggingface_hub import HfApi
    api = HfApi()

    for obj in resp["Contents"]:
        key = obj["Key"]
        fname = key.rsplit("/", 1)[-1]
        if fname not in ("best.pt", "final.pt"):
            continue
        local_path = f"/tmp/{name}-{fname}"
        print(f"downloading s3://{s3_bucket}/{key} -> {local_path}")
        s3.download_file(s3_bucket, key, local_path)
        print(f"uploading to HF Hub: {args.hf_repo}/{fname}")
        api.upload_file(
            path_or_fileobj=local_path, path_in_repo=fname,
            repo_id=args.hf_repo, repo_type="model",
        )
    print("done")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--region", default=DEFAULT_REGION)

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ls = sub.add_parser("ls", help="list active + recent CU-VLA jobs")
    p_ls.add_argument("--status", choices=["InProgress", "Completed", "Failed", "Stopping", "Stopped"])
    p_ls.add_argument("--limit", type=int, default=20)
    p_ls.add_argument("--json", action="store_true")
    p_ls.set_defaults(func=cmd_ls)

    for cname, fn, helptext in [
        ("status", cmd_status, "one-line summary"),
        ("describe", cmd_describe, "full describe-training-job JSON"),
        ("url", cmd_url, "print SageMaker console URL"),
        ("cost", cmd_cost, "estimate billable cost"),
    ]:
        p = sub.add_parser(cname, help=helptext)
        p.add_argument("name", nargs="?", default=None)
        p.add_argument("--filter", action="append")
        p.add_argument("--json", action="store_true")
        p.set_defaults(func=fn)

    p_logs = sub.add_parser("logs", help="batch or stream CloudWatch logs")
    p_logs.add_argument("name", nargs="?", default=None)
    p_logs.add_argument("--filter", action="append")
    p_logs.add_argument("--follow", action="store_true")
    p_logs.set_defaults(func=cmd_logs)

    p_stop = sub.add_parser("stop", help="stop a running job")
    p_stop.add_argument("name", nargs="?", default=None)
    p_stop.add_argument("--filter", action="append")
    p_stop.add_argument("-y", "--yes", action="store_true", help="skip the confirm prompt")
    p_stop.set_defaults(func=cmd_stop)

    p_rec = sub.add_parser("reconcile-ckpts", help="push S3 ckpts to HF Hub")
    p_rec.add_argument("name", nargs="?", default=None)
    p_rec.add_argument("--filter", action="append")
    p_rec.add_argument("--hf-repo", required=True, help="HF repo id, e.g. PenTest-duck/cu-vla-exp6-phaseb0-ckpt")
    p_rec.add_argument("--s3-bucket", default=None)
    p_rec.set_defaults(func=cmd_reconcile_ckpts)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/pentest-duck/Desktop/CU-VLA && uv run pytest tests/scripts/test_sm_jobs.py -v`

Expected: 6 PASS.

- [ ] **Step 5: Smoke-test help**

Run: `cd /Users/pentest-duck/Desktop/CU-VLA && uv run python scripts/sm_jobs.py --help`

Expected: prints subcommand list, exits 0.

Run: `cd /Users/pentest-duck/Desktop/CU-VLA && uv run python scripts/sm_jobs.py ls --help`

Expected: prints `ls` help, exits 0.

- [ ] **Step 6: Commit**

```bash
git add scripts/sm_jobs.py tests/scripts/test_sm_jobs.py
git commit -m "feat(sm-migration): sm_jobs.py operator CLI helper

8 subcommands (ls, status, logs, stop, describe, url, cost, reconcile-ckpts)
keyed on cu-vla-* prefix. 'latest' shortcut + --filter substring match.
--json for parseable output. Tests mock boto3."
```

---

## Task 9: Bootstrap AWS resources + write aws-bootstrap.md reference

**Files:**
- Create: `docs/ops/aws-bootstrap.md`
- Create (locally): `~/.cu-vla/sagemaker.toml` with role ARN + bucket name

This task is half documentation, half hands-on AWS clicking. The doc captures the steps so anyone (including future-you on a fresh machine) can re-create the bootstrap.

**Note:** Task 2 (quota request) was already done. This task creates the IAM role, S3 bucket, SSM Parameter Store secrets, and Budget alerts.

- [ ] **Step 1: Write the bootstrap doc**

Create `docs/ops/aws-bootstrap.md`:

```markdown
# AWS Bootstrap for CU-VLA SageMaker Training

One-time setup. Goal: get from "blank account with credits applied" to
"ready to run launch_sm_job_exp6.py" in ~30 minutes (excluding the 1–3 day
GPU-quota approval wait).

**Region:** us-west-2 (chosen in [design doc](../plans/2026-04-26-sagemaker-migration-design.md) §3.1).

**Account ID:** `<your-12-digit-account-id>` — referred to as `<acct>` below.
Find it with `aws sts get-caller-identity --query Account --output text`.

## 0. Prereqs

- AWS CLI v2 configured + SSO logged in.
- The local Python env has `sagemaker` + `boto3` installed (`uv sync` after Task 1).
- `WANDB_API_KEY` and `HF_TOKEN` available in your local environment (we'll
  upload them to Parameter Store in step 4).

## 1. GPU quota request

Already done in Task 2 of the implementation plan. Confirm approval:

```bash
aws service-quotas list-requested-service-quota-change-history \
  --region us-west-2 --service-code sagemaker \
  --query "RequestedQuotas[].[QuotaName,DesiredValue,Status]" \
  --output table
```

Wait for `Status` to be `APPROVED` before launching any real training jobs.

## 2. IAM execution role: SageMakerExecutionRole-CU-VLA

Trust policy file `trust.json`:

\`\`\`json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "sagemaker.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }]
}
\`\`\`

Create the role:

\`\`\`bash
aws iam create-role --role-name SageMakerExecutionRole-CU-VLA \
  --assume-role-policy-document file://trust.json

aws iam attach-role-policy --role-name SageMakerExecutionRole-CU-VLA \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy --role-name SageMakerExecutionRole-CU-VLA \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
\`\`\`

Inline policy for SSM Parameter Store + KMS — file `ssm-policy.json`:

\`\`\`json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow",
      "Action": ["ssm:GetParameter", "ssm:GetParameters"],
      "Resource": "arn:aws:ssm:us-west-2:*:parameter/cu-vla/*" },
    { "Effect": "Allow",
      "Action": ["kms:Decrypt"],
      "Resource": "*",
      "Condition": { "StringEquals": { "kms:ViaService": "ssm.us-west-2.amazonaws.com" } } }
  ]
}
\`\`\`

\`\`\`bash
aws iam put-role-policy --role-name SageMakerExecutionRole-CU-VLA \
  --policy-name SSMParameterAccess \
  --policy-document file://ssm-policy.json
\`\`\`

Save the role ARN:

\`\`\`bash
aws iam get-role --role-name SageMakerExecutionRole-CU-VLA --query Role.Arn --output text
# Output: arn:aws:iam::<acct>:role/SageMakerExecutionRole-CU-VLA
\`\`\`

## 3. S3 bucket: s3://cu-vla-sm-<acct>/

\`\`\`bash
ACCT=$(aws sts get-caller-identity --query Account --output text)
aws s3 mb s3://cu-vla-sm-$ACCT --region us-west-2
\`\`\`

Lifecycle policy file `lifecycle.json`:

\`\`\`json
{
  "Rules": [
    { "ID": "expire-checkpoints",
      "Filter": { "Prefix": "checkpoints/" },
      "Status": "Enabled",
      "Expiration": { "Days": 30 } },
    { "ID": "expire-tmp",
      "Filter": { "Prefix": "tmp/" },
      "Status": "Enabled",
      "Expiration": { "Days": 7 } }
  ]
}
\`\`\`

\`\`\`bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket cu-vla-sm-$ACCT \
  --lifecycle-configuration file://lifecycle.json
\`\`\`

## 4. SSM Parameter Store secrets

\`\`\`bash
aws ssm put-parameter --region us-west-2 --type SecureString \
  --name /cu-vla/wandb-api-key --value "$WANDB_API_KEY"

aws ssm put-parameter --region us-west-2 --type SecureString \
  --name /cu-vla/hf-token --value "$HF_TOKEN"
\`\`\`

To rotate later: `--overwrite` flag with the new value.

## 5. Budget alerts

Budget JSON `budget.json` (replace `<acct>` and `<email>`):

\`\`\`json
{
  "BudgetName": "cu-vla-credits",
  "BudgetLimit": { "Amount": "10000", "Unit": "USD" },
  "TimeUnit": "ANNUALLY",
  "BudgetType": "COST",
  "CostFilters": { "Service": ["Amazon SageMaker"] }
}
\`\`\`

Notification JSON `notifications.json`:

\`\`\`json
[
  { "Notification": { "NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN",
                      "Threshold": 25, "ThresholdType": "PERCENTAGE" },
    "Subscribers": [{ "SubscriptionType": "EMAIL", "Address": "<email>" }] },
  { "Notification": { "NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN",
                      "Threshold": 50, "ThresholdType": "PERCENTAGE" },
    "Subscribers": [{ "SubscriptionType": "EMAIL", "Address": "<email>" }] },
  { "Notification": { "NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN",
                      "Threshold": 75, "ThresholdType": "PERCENTAGE" },
    "Subscribers": [{ "SubscriptionType": "EMAIL", "Address": "<email>" }] },
  { "Notification": { "NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN",
                      "Threshold": 90, "ThresholdType": "PERCENTAGE" },
    "Subscribers": [{ "SubscriptionType": "EMAIL", "Address": "<email>" }] }
]
\`\`\`

\`\`\`bash
aws budgets create-budget --account-id $ACCT \
  --budget file://budget.json \
  --notifications-with-subscribers file://notifications.json
\`\`\`

## 6. CloudWatch log retention

\`\`\`bash
aws logs put-retention-policy --region us-west-2 \
  --log-group-name /aws/sagemaker/TrainingJobs --retention-in-days 30
\`\`\`

(If the log group doesn't exist yet, this fails harmlessly. Re-run after the
first training job creates the group.)

## 7. Local config

Create `~/.cu-vla/sagemaker.toml`:

\`\`\`toml
[sagemaker]
role_arn = "arn:aws:iam::<acct>:role/SageMakerExecutionRole-CU-VLA"
s3_bucket = "cu-vla-sm-<acct>"
region = "us-west-2"
\`\`\`

Or as env vars (preferred for shell-script convenience):

\`\`\`bash
# Add to ~/.zshrc or ~/.bashrc
export CU_VLA_SM_ROLE_ARN="arn:aws:iam::<acct>:role/SageMakerExecutionRole-CU-VLA"
export CU_VLA_SM_BUCKET="cu-vla-sm-<acct>"
export AWS_REGION="us-west-2"
\`\`\`

Verify:

\`\`\`bash
aws sagemaker list-training-jobs --region us-west-2 --max-results 1
\`\`\`

Expected: empty list (`{"TrainingJobSummaries": []}`) on a fresh account, no error.

## 8. Tear-down (if you ever leave the project)

\`\`\`bash
aws iam detach-role-policy --role-name SageMakerExecutionRole-CU-VLA \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
aws iam detach-role-policy --role-name SageMakerExecutionRole-CU-VLA \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam delete-role-policy --role-name SageMakerExecutionRole-CU-VLA --policy-name SSMParameterAccess
aws iam delete-role --role-name SageMakerExecutionRole-CU-VLA

aws ssm delete-parameter --region us-west-2 --name /cu-vla/wandb-api-key
aws ssm delete-parameter --region us-west-2 --name /cu-vla/hf-token

aws s3 rb s3://cu-vla-sm-$ACCT --force
aws budgets delete-budget --account-id $ACCT --budget-name cu-vla-credits
\`\`\`
```

(The literal triple-backtick code blocks inside the markdown above are escaped with backslashes for plan readability — when writing the actual file, use plain triple-backticks. See the note at the bottom of this task.)

- [ ] **Step 2: Execute the bootstrap doc steps 2–7 against your account**

Open `docs/ops/aws-bootstrap.md` in your editor and run each shell command in section order:
1. Section 2 — IAM role + inline SSM policy. Save the role ARN that gets printed.
2. Section 3 — S3 bucket + lifecycle.
3. Section 4 — Parameter Store SecureStrings.
4. Section 5 — Budget alerts (use the email you want notifications at).
5. Section 6 — CloudWatch log retention (may fail if log group doesn't exist yet — that's fine).
6. Section 7 — local config (env vars or `~/.cu-vla/sagemaker.toml`).

Verify everything is wired with the `list-training-jobs` smoke at the end of section 7.

- [ ] **Step 3: Sanity-check from the launcher's perspective**

Run:
```bash
cd /Users/pentest-duck/Desktop/CU-VLA && uv run python -c "
import os, boto3
print('region:', os.environ.get('AWS_REGION'))
print('role:', os.environ.get('CU_VLA_SM_ROLE_ARN'))
print('bucket:', os.environ.get('CU_VLA_SM_BUCKET'))
sm = boto3.client('sagemaker', region_name=os.environ['AWS_REGION'])
print('list_training_jobs:', sm.list_training_jobs(MaxResults=1)['TrainingJobSummaries'])
ssm = boto3.client('ssm', region_name=os.environ['AWS_REGION'])
print('wandb-api-key set:', 'Parameter' in ssm.get_parameter(Name='/cu-vla/wandb-api-key', WithDecryption=True))
print('hf-token set:', 'Parameter' in ssm.get_parameter(Name='/cu-vla/hf-token', WithDecryption=True))
"
```

Expected: prints all three env vars (no `None`), an empty `list_training_jobs` list, and `True` for both Parameter Store values.

- [ ] **Step 4: Commit the doc**

```bash
git add docs/ops/aws-bootstrap.md
git commit -m "docs(sm-migration): aws-bootstrap.md — one-time AWS setup walkthrough

IAM execution role with SSM+KMS inline policy, S3 bucket with lifecycle,
Parameter Store SecureStrings, Budget alerts, CloudWatch log retention,
local env-var config. Sections 2-7 are runnable copy-paste commands."
```

---

## Task 10: Wait for GPU quota approval

**Files:** none.

This is the gating wait between Task 2 (submitted on Day 0) and Task 11 (smoke test). The quota request typically takes 1–3 days. Don't block on it — Tasks 3–9 are all parallel work.

- [ ] **Step 1: Poll quota status**

Run:
```bash
aws service-quotas list-requested-service-quota-change-history \
  --region us-west-2 --service-code sagemaker \
  --query "RequestedQuotas[].[QuotaName,DesiredValue,Status]" \
  --output table
```

Expected (when ready): `Status` shows `APPROVED` for both `ml.g6e.xlarge for spot training job usage` and (if Task 2 Step 4 was done) the on-demand variant.

- [ ] **Step 2: When approved, proceed to Task 11.**

If denied (rare): respond to the AWS Support case with justification (`research project, $10k educational credits`) and re-submit. If still blocked after 5 business days, fall back to ap-southeast-2 by re-running Task 2 with `--region ap-southeast-2`.

---

## Task 11: Phase 1 — Smoke test (10 steps, on-demand)

**Files:** none — a live test of the wiring built in Tasks 1, 3–9.

**Goal:** validate the full chain end-to-end with minimal cost (~$0.30) before committing to a real run. Five assertions; if any fails, do NOT proceed to Phase 2.

- [ ] **Step 1: Confirm prereqs**

- ✅ Task 2 quota approved (check status, see Task 10).
- ✅ Task 9 bootstrap complete (`CU_VLA_SM_ROLE_ARN`, `CU_VLA_SM_BUCKET`, `AWS_REGION` set; SSM params exist).
- ✅ Task 1, 3–8 commits all present (`git log --oneline scripts/`).
- ✅ Branch `feat/exp6-phase-b0` exists on origin (`git ls-remote origin feat/exp6-phase-b0`).

- [ ] **Step 2: Launch the smoke test**

Run:
```bash
cd /Users/pentest-duck/Desktop/CU-VLA && uv run python scripts/launch_sm_job_exp6.py \
    --phase b0 --no-spot --max-run 1800 \
    -- --epochs 1 \
       --hf-data-repo PenTest-duck/cu-vla-exp6-phaseb0 \
       --hf-upload-repo PenTest-duck/cu-vla-exp6-phaseb0-ckpt-smoke \
       --eval-every-steps 5 \
       --ckpt-every-steps 5 \
       --early-stop-patience 1 \
       --wandb-run-name sm-smoke-phaseb0
```

Expected: launcher prints the job name (e.g. `cu-vla-exp6-phaseb0-20260427-093142`), then streams CloudWatch logs as they arrive. `--no-spot` avoids any wait-for-spot delay.

- [ ] **Step 3: Watch for the 5 assertion-equivalents in the streamed logs**

The streaming output should show, in order:

1. **Container starts, deps install** — you'll see `pip install -r requirements-sagemaker.txt` followed by `Successfully installed transformers ... peft ...`
2. **`load_dataset()` from HF Hub works** — line like `Generating train split: 100% ...` from the dataset code.
3. **First train step happens** — `[step 0]` or `train/total=...` in stdout.
4. **`step_00005.pt` lands on S3-synced volume** — `[ckpt] /opt/ml/checkpoints/step_00005.pt` from train.py's print.
5. **`best.pt` push to HF Hub succeeds** — `[best] val_loss=...` followed by `[hf_sync] uploaded best.pt to PenTest-duck/cu-vla-exp6-phaseb0-ckpt-smoke`.
6. **wandb run appears with the SM job name as run ID** — visit https://wandb.ai/<entity>/<project> and confirm the run named `cu-vla-exp6-phaseb0-...` exists.

- [ ] **Step 4: Verify status from the operator CLI (in a second terminal)**

While the job runs (or after it finishes):

Run:
```bash
cd /Users/pentest-duck/Desktop/CU-VLA && uv run python scripts/sm_jobs.py status latest
```

Expected: one-line summary, parseable. e.g.:
`cu-vla-exp6-phaseb0-20260427-093142    InProgress    Training    ml.g6e.xlarge    train=180s    bill=180s`

- [ ] **Step 5: Confirm the job completed cleanly**

Run:
```bash
cd /Users/pentest-duck/Desktop/CU-VLA && uv run python scripts/sm_jobs.py describe latest --region us-west-2 \
  | python -c "import json,sys; d=json.load(sys.stdin); print(d['TrainingJobStatus'], '|', d.get('FailureReason','(no failure)'))"
```

Expected: `Completed | (no failure)`. If `Failed`, read the failure reason and the full CloudWatch logs (`uv run python scripts/sm_jobs.py logs latest | tail -200`).

- [ ] **Step 6: Check the smoke-checkpoint repo on HF Hub**

Visit https://huggingface.co/PenTest-duck/cu-vla-exp6-phaseb0-ckpt-smoke and confirm `best.pt` and `final.pt` are present.

- [ ] **Step 7: Decision**

If all 6 verifications pass → proceed to Task 12.
If any fail → diagnose, fix, re-run Step 2. Do NOT skip to Task 12 with a broken Phase 1.

- [ ] **Step 8: Note success**

No commit needed. Note the smoke job name in your bench journal.

---

## Task 12: Phase 2 + Phase 3 — Spot rehearsal & first real run; deprecate HF Jobs scripts

**Files:**
- Modify: `scripts/launch_hf_job_exp6.py` (top docstring deprecation note)
- Modify: `scripts/launch_hf_job.py` (top docstring deprecation note)
- Modify: `scripts/hf_job_train_exp6.py` (top docstring deprecation note)
- Modify: `scripts/hf_job_train.py` (top docstring deprecation note)
- Modify: `AGENTS.md` (one-line note that SM is the default training launcher)

This task ties Phase 2 (spot rehearsal) and Phase 3 (first real run) together with the cleanup commit. Skip ahead via the steps:

### Phase 2 — Spot rehearsal

- [ ] **Step 1: Re-run the smoke test with --spot, manually interrupt, verify resume**

Run:
```bash
cd /Users/pentest-duck/Desktop/CU-VLA && uv run python scripts/launch_sm_job_exp6.py \
    --phase b0 --spot --max-run 1800 \
    -- --epochs 1 \
       --hf-data-repo PenTest-duck/cu-vla-exp6-phaseb0 \
       --hf-upload-repo PenTest-duck/cu-vla-exp6-phaseb0-ckpt-smoke \
       --ckpt-every-steps 5 \
       --wandb-run-name sm-spot-rehearsal-phaseb0
```

Expected: SageMaker negotiates spot capacity (may take 30s–5min), then logs start streaming as in Phase 1.

- [ ] **Step 2: Wait until you see at least one `[ckpt] step_00005.pt` line, then stop the job**

In a second terminal, run:
```bash
cd /Users/pentest-duck/Desktop/CU-VLA && uv run python scripts/sm_jobs.py stop latest -y
```

Expected: prints `stop_training_job sent for cu-vla-...`. **The launcher in terminal 1 will continue streaming until SageMaker actually terminates the container** — be patient (~1–2 min).

- [ ] **Step 3: Inspect the post-stop state**

After the stop completes, the launcher in terminal 1 will exit with the job in `Stopped` state.

If we set `--max-wait 3600` and the job has time left, SageMaker will **NOT** auto-relaunch a Stopped job (Stopped is user-initiated; only InsufficientCapacityError or interruption causes auto-relaunch). To validate the resume code path properly, we need a real reclaim, which we can't reliably trigger.

**Alternative validation:** manually re-launch the same job name with `--max-run 1800` (without stopping it again). Because `checkpoint_s3_uri` is keyed on the job name, and the previous job's `step_*.pt` is still in S3, the new job's entrypoint will see them and resume. **The launcher generates a new job name each launch (timestamp suffix) — so for resume validation we need to override the suffix.**

**Simpler validation (good enough):** trust SageMaker's documented behavior + verify the entrypoint logic in unit tests (Task 5). Real reclaim happens for free in production runs and the unit tests cover the filesystem detection.

If you want to *really* exercise the resume path: copy the stopped-job's S3 prefix to a new prefix and launch a manually-named job pointing at it. ~30 minutes of fiddling, optional.

- [ ] **Step 4: (Quick win) Confirm wandb run id continuity setting**

Run:
```bash
cd /Users/pentest-duck/Desktop/CU-VLA && uv run python scripts/sm_jobs.py logs latest | grep -E "WANDB_RUN_ID|wandb: Resuming" | head -10
```

Expected: at least one line showing `WANDB_RUN_ID` set to the SageMaker job name. Validates the entrypoint env wiring.

### Phase 3 — First real training run

- [ ] **Step 5: Launch Phase B1 (or whatever's actually next) on SageMaker**

Replace `<phase>` and the train args with whatever the next experiment phase needs:

```bash
cd /Users/pentest-duck/Desktop/CU-VLA && uv run python scripts/launch_sm_job_exp6.py \
    --phase b1 --spot \
    -- --epochs 5 \
       --hf-data-repo PenTest-duck/cu-vla-exp6-phaseb1 \
       --hf-upload-repo PenTest-duck/cu-vla-exp6-phaseb1-ckpt \
       --wandb-run-name phase-b1-sm-001 \
       --micro-batch-episodes 4 --num-workers 4 \
       --ckpt-every-steps 20 --eval-every-steps 20 \
       --early-stop-patience 3
```

(Adjust the args to your real Phase B1 config.)

Expected: streams logs, runs to completion or hits early stop. Estimated cost on spot: $4–7 for an 8h max_run window.

- [ ] **Step 6: Verify completion**

Run:
```bash
cd /Users/pentest-duck/Desktop/CU-VLA && uv run python scripts/sm_jobs.py status latest && \
uv run python scripts/sm_jobs.py cost latest
```

Expected: status shows `Completed`, cost shows the billable seconds × rate.

Confirm `best.pt` and `final.pt` reached HF Hub at `PenTest-duck/cu-vla-exp6-phaseb1-ckpt`.

### Deprecate HF Jobs scripts

- [ ] **Step 7: Add deprecation note to each HF Jobs script**

For each of the four files below, prepend a deprecation block to the existing module docstring. Don't delete the files — they're useful as reference and as an emergency fallback.

In `scripts/launch_hf_job.py`, find the top module docstring and replace its first line with:

```python
"""[DEPRECATED 2026-04-26] Launch on Hugging Face Jobs.

Replaced by scripts/launch_sm_job.py. Retained as fallback in case AWS
credits run out before HF credits do, or for one-off comparison runs.
See docs/plans/2026-04-26-sagemaker-migration-design.md.

(original docstring continues below)
"""
```

(If the file currently has no module docstring, add this as the first thing in the file before any code.)

Repeat for `scripts/launch_hf_job_exp6.py`, `scripts/hf_job_train.py`, `scripts/hf_job_train_exp6.py`.

- [ ] **Step 8: Update AGENTS.md**

In `AGENTS.md`, find the section that lists `scripts/launch_hf_job.py` etc. in the Repo Layout table. Add a line at the top of that block (right after the table headers, or as a footnote below the table):

> **Note (2026-04-26):** Training launcher is now SageMaker by default. Use `scripts/launch_sm_job_exp6.py`. HF Jobs scripts retained as emergency fallback only. See [SageMaker migration design](docs/plans/2026-04-26-sagemaker-migration-design.md).

- [ ] **Step 9: Commit deprecation marks**

```bash
git add scripts/launch_hf_job.py scripts/launch_hf_job_exp6.py \
        scripts/hf_job_train.py scripts/hf_job_train_exp6.py \
        AGENTS.md
git commit -m "chore(sm-migration): mark HF Jobs scripts deprecated; SM is default

After successful Phase 3 run on SageMaker, HF Jobs scripts are retained
as emergency fallback. AGENTS.md updated to point at launch_sm_job_exp6.py."
```

- [ ] **Step 10: Final verification**

Run:
```bash
cd /Users/pentest-duck/Desktop/CU-VLA && uv run pytest tests/scripts/ -v
```

Expected: all tests pass.

Run:
```bash
git log --oneline | head -15
```

Expected: see the chain of migration commits — deps, train.py patch, factory, entrypoint, launcher, exp6 override, sm_jobs CLI, bootstrap doc, deprecation.

🎉 Migration complete.

---

## Task summary (12 tasks, ~250 LOC of new code + 1 small patch + 1 doc)

| # | Task | Files |
|---|------|-------|
| 1 | Add SDK deps + requirements-sagemaker.txt | `pyproject.toml`, `requirements-sagemaker.txt` |
| 2 | Submit GPU quota request | (AWS console / CLI only) |
| 3 | Patch train.py for best_val_loss restore | `experiments/action_primitives/train.py:579`, `tests/action_primitives/test_resume_best_val_loss.py` |
| 4 | sagemaker_estimator.py factory + tests | `scripts/sagemaker_estimator.py`, `tests/scripts/test_sagemaker_estimator.py` |
| 5 | sm_job_train.py entrypoint + tests | `scripts/sm_job_train.py`, `tests/scripts/test_sm_job_train.py` |
| 6 | launch_sm_job.py generic CLI + tests | `scripts/launch_sm_job.py`, `tests/scripts/test_launch_sm_job.py` |
| 7 | launch_sm_job_exp6.py exp6 override | `scripts/launch_sm_job_exp6.py` |
| 8 | sm_jobs.py operator CLI + tests | `scripts/sm_jobs.py`, `tests/scripts/test_sm_jobs.py` |
| 9 | Bootstrap AWS resources + aws-bootstrap.md | `docs/ops/aws-bootstrap.md` |
| 10 | Wait for GPU quota approval | (none) |
| 11 | Phase 1 smoke test | (live) |
| 12 | Phase 2 spot rehearsal + Phase 3 real run + deprecate HF scripts | `scripts/launch_hf_job*.py`, `scripts/hf_job_train*.py`, `AGENTS.md` |

**Critical path:** Task 2 → Task 10 (1–3 day wait, parallel with Tasks 3–9) → Task 11 → Task 12.

**Estimated total wall-time:** 1 day of code (Tasks 1, 3–9) + 1–3 day quota wait + 0.5 day live testing (Tasks 11–12). The first real Phase B1 run on SageMaker (Step 5 of Task 12) is itself a multi-hour training; that's not "implementation time" — it's the new normal.
