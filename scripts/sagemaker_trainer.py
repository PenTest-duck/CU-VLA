"""SageMaker V3 ModelTrainer factory for CU-VLA training jobs.

Pure config builder — no I/O, no .train() call. Builds and returns a
sagemaker.train.ModelTrainer with our defaults (managed spot, L40S, 4h max,
S3-synced checkpoints via CheckpointConfig, Parameter Store secrets via
IAM execution role).

V3 API note: V3 (Nov 2025) replaced the V2 PyTorch estimator class with the
unified ModelTrainer. Configuration is now spread across Compute,
SourceCode, StoppingCondition, CheckpointConfig, OutputDataConfig. The
PyTorch DLC image URI must be resolved explicitly via image_uris.retrieve.

Used by scripts/launch_sm_job.py and scripts/launch_sm_job_<exp>.py.
"""
from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Iterable

from sagemaker.core import image_uris
from sagemaker.core.shapes.shapes import (
    CheckpointConfig,
    OutputDataConfig,
    StoppingCondition,
)
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, SourceCode


# Repo root, used as source_dir for SageMaker. The SDK tars and uploads this
# directory; the entrypoint sm_job_train.py git-clones the actual source from
# the configured branch on container start, so the upload is small (just the
# entrypoint script + requirements file).
REPO_ROOT = Path(__file__).resolve().parent.parent


# Default config — overridden by per-experiment launchers and CLI flags.
DEFAULTS: dict[str, Any] = {
    "framework": "pytorch",
    "version": "2.6.0",              # PyTorch DLC version — matches train.py's torch>=2.11 (DLC has 2.6+)
    "py_version": "py312",
    "instance_count": 1,
    "use_spot": True,
    "max_run": 4 * 3600,             # 4h per attempt — covers Phase B0 with margin
    "max_wait": 8 * 3600,            # 2× max_run; allows 1–2 reclaim retries
    "volume_size_in_gb": 100,        # EBS for /opt/ml/{input,model,checkpoints}
    "keep_alive_period_in_seconds": 0,
}


def make_trainer(
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
    volume_size_in_gb: int | None = None,
    region: str = "us-west-2",
    extra_environment: dict[str, str] | None = None,
) -> ModelTrainer:
    """Construct a SageMaker V3 ModelTrainer with CU-VLA defaults.

    Args:
        train_module: dotted path of the training module to invoke inside the
            container, e.g. "experiments.action_primitives.train". The
            entrypoint reads this from the TRAIN_MODULE env var and runs
            `python -u -m <train_module> [TRAIN_ARGS]`.
        instance_type: e.g. "ml.g6e.xlarge" (1×L40S 48GB).
        role_arn: ARN of the SageMakerExecutionRole-CU-VLA IAM role.
        s3_bucket: bare bucket name (no s3:// prefix), e.g. "cu-vla-sm-123".
        job_name: full SageMaker job name; passed via base_job_name and used
            for the spot-resume CheckpointConfig prefix.
        branch: git branch to clone in the container.
        train_args: list of CLI args passed through to the train module.
        use_spot, max_run, max_wait, instance_count, volume_size_in_gb:
            override DEFAULTS. None means "use the default."
        region: AWS region (must match where role and bucket live).
        extra_environment: additional env vars to set in the container.

    Returns:
        Configured sagemaker.train.ModelTrainer, ready to call .train() on.
    """
    use_spot = DEFAULTS["use_spot"] if use_spot is None else use_spot
    max_run = DEFAULTS["max_run"] if max_run is None else max_run
    instance_count = DEFAULTS["instance_count"] if instance_count is None else instance_count
    volume_size_in_gb = (
        DEFAULTS["volume_size_in_gb"] if volume_size_in_gb is None else volume_size_in_gb
    )

    # max_wait is only meaningful for spot. For on-demand we set it to None.
    if use_spot:
        max_wait = (DEFAULTS["max_wait"] if max_wait is None else max_wait)
        if max_wait < 2 * max_run:
            raise ValueError(
                f"max_wait ({max_wait}s) must be >= 2 * max_run ({2 * max_run}s) "
                "for spot reclaim retries to fit"
            )
    else:
        max_wait = None

    # Resolve the pre-built PyTorch DLC image URI for this region + framework.
    training_image = image_uris.retrieve(
        framework=DEFAULTS["framework"],
        region=region,
        version=DEFAULTS["version"],
        py_version=DEFAULTS["py_version"],
        instance_type=instance_type,
        image_scope="training",
    )

    environment: dict[str, str] = {
        "CU_VLA_BRANCH": branch,
        "TRAIN_MODULE": train_module,
        "TRAIN_ARGS": shlex.join(train_args) if train_args else "",
        # Headless pygame insurance — if any train-time code path imports
        # pygame.display, this prevents an SDL init crash.
        "SDL_VIDEODRIVER": "dummy",
        "AWS_REGION": region,
        # Silence HF Hub progress bars in CloudWatch.
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
    }
    if extra_environment:
        environment.update(extra_environment)

    compute = Compute(
        instance_type=instance_type,
        instance_count=instance_count,
        volume_size_in_gb=volume_size_in_gb,
        keep_alive_period_in_seconds=DEFAULTS["keep_alive_period_in_seconds"],
        enable_managed_spot_training=use_spot,
    )

    # NOTE: V3 Pydantic validates that ``requirements`` is a path *within*
    # ``source_dir``. We keep the requirements file under ``scripts/`` so the
    # SourceCode upload self-contains everything the container needs.
    source_code = SourceCode(
        source_dir=str(REPO_ROOT / "scripts"),
        entry_script="sm_job_train.py",
        requirements="requirements-sagemaker.txt",
    )

    stopping_condition = StoppingCondition(
        max_runtime_in_seconds=max_run,
        max_wait_time_in_seconds=max_wait,  # None for on-demand
    )

    checkpoint_config = None
    if use_spot:
        checkpoint_config = CheckpointConfig(
            s3_uri=f"s3://{s3_bucket}/checkpoints/{job_name}",
            local_path="/opt/ml/checkpoints",
        )

    output_data_config = OutputDataConfig(
        s3_output_path=f"s3://{s3_bucket}/output",
    )

    return ModelTrainer(
        training_image=training_image,
        source_code=source_code,
        compute=compute,
        stopping_condition=stopping_condition,
        checkpoint_config=checkpoint_config,
        output_data_config=output_data_config,
        environment=environment,
        role=role_arn,
        base_job_name=job_name,    # SDK appends a UUID/timestamp suffix automatically
    )
