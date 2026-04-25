"""Unit tests for scripts/sagemaker_trainer.py.

We mock both ModelTrainer (the V3 unified trainer) and image_uris.retrieve so
no network calls are made. Each test asserts the factory forwards the right
config to the right place — Compute for instance/spot, StoppingCondition for
runtime, CheckpointConfig for spot resume sync.
"""
from unittest.mock import patch, MagicMock

import pytest

from scripts.sagemaker_trainer import make_trainer, DEFAULTS


def _make_kwargs(**overrides):
    """Build a baseline kwargs dict for make_trainer, with overrides."""
    kwargs = dict(
        train_module="experiments.action_primitives.train",
        instance_type="ml.g6e.xlarge",
        role_arn="arn:aws:iam::123:role/SageMakerExecutionRole-CU-VLA",
        s3_bucket="cu-vla-sm-123",
        job_name="cu-vla-test-20260101-000000",
    )
    kwargs.update(overrides)
    return kwargs


@patch("scripts.sagemaker_trainer.image_uris")
@patch("scripts.sagemaker_trainer.ModelTrainer")
def test_factory_uses_spot_by_default(mock_mt, mock_image):
    mock_image.retrieve.return_value = "1234.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.6.0-gpu-py312"
    make_trainer(**_make_kwargs())

    kwargs = mock_mt.call_args.kwargs
    compute = kwargs["compute"]
    stopping = kwargs["stopping_condition"]

    assert compute.enable_managed_spot_training is True
    assert stopping.max_runtime_in_seconds == DEFAULTS["max_run"]
    assert stopping.max_wait_time_in_seconds == DEFAULTS["max_wait"]
    assert stopping.max_wait_time_in_seconds >= 2 * stopping.max_runtime_in_seconds, \
        "max_wait must be >= 2*max_run for spot retries"


@patch("scripts.sagemaker_trainer.image_uris")
@patch("scripts.sagemaker_trainer.ModelTrainer")
def test_factory_no_spot_omits_max_wait(mock_mt, mock_image):
    mock_image.retrieve.return_value = "img"
    make_trainer(**_make_kwargs(use_spot=False))

    kwargs = mock_mt.call_args.kwargs
    assert kwargs["compute"].enable_managed_spot_training is False
    assert kwargs["stopping_condition"].max_wait_time_in_seconds is None


@patch("scripts.sagemaker_trainer.image_uris")
@patch("scripts.sagemaker_trainer.ModelTrainer")
def test_factory_sets_checkpoint_config_only_for_spot(mock_mt, mock_image):
    mock_image.retrieve.return_value = "img"
    make_trainer(**_make_kwargs(use_spot=True))

    kwargs = mock_mt.call_args.kwargs
    cc = kwargs["checkpoint_config"]
    assert cc is not None
    assert cc.s3_uri == "s3://cu-vla-sm-123/checkpoints/cu-vla-test-20260101-000000"
    assert cc.local_path == "/opt/ml/checkpoints"


@patch("scripts.sagemaker_trainer.image_uris")
@patch("scripts.sagemaker_trainer.ModelTrainer")
def test_factory_no_checkpoint_config_when_no_spot(mock_mt, mock_image):
    mock_image.retrieve.return_value = "img"
    make_trainer(**_make_kwargs(use_spot=False))
    # On-demand training has no need for the S3 sync volume — final/best.pt go
    # to HF Hub directly.
    assert mock_mt.call_args.kwargs["checkpoint_config"] is None


@patch("scripts.sagemaker_trainer.image_uris")
@patch("scripts.sagemaker_trainer.ModelTrainer")
def test_factory_sets_environment_with_branch_and_train_module(mock_mt, mock_image):
    mock_image.retrieve.return_value = "img"
    make_trainer(**_make_kwargs(branch="feat/exp6-phase-b1"))

    env = mock_mt.call_args.kwargs["environment"]
    assert env["CU_VLA_BRANCH"] == "feat/exp6-phase-b1"
    assert env["TRAIN_MODULE"] == "experiments.action_primitives.train"
    assert env["SDL_VIDEODRIVER"] == "dummy"


@patch("scripts.sagemaker_trainer.image_uris")
@patch("scripts.sagemaker_trainer.ModelTrainer")
def test_factory_encodes_train_args_as_env_var(mock_mt, mock_image):
    mock_image.retrieve.return_value = "img"
    make_trainer(**_make_kwargs(
        train_args=["--epochs", "5", "--out-dir", "/opt/ml/checkpoints"],
    ))

    env = mock_mt.call_args.kwargs["environment"]
    assert "TRAIN_ARGS" in env
    assert "--epochs 5" in env["TRAIN_ARGS"]
    assert "--out-dir /opt/ml/checkpoints" in env["TRAIN_ARGS"]


@patch("scripts.sagemaker_trainer.image_uris")
@patch("scripts.sagemaker_trainer.ModelTrainer")
def test_factory_resolves_pytorch_dlc_image_uri(mock_mt, mock_image):
    mock_image.retrieve.return_value = "1234.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.6.0-gpu-py312"
    make_trainer(**_make_kwargs())

    # Verify the image_uris.retrieve call has the right framework + version.
    call_kwargs = mock_image.retrieve.call_args.kwargs
    assert call_kwargs["framework"] == "pytorch"
    assert call_kwargs["version"] == "2.6.0"
    assert call_kwargs["py_version"] == "py312"
    assert call_kwargs["instance_type"] == "ml.g6e.xlarge"
    assert call_kwargs["image_scope"] == "training"
    assert call_kwargs["region"] == "us-west-2"

    # And the URI is forwarded as ModelTrainer's training_image kwarg.
    assert mock_mt.call_args.kwargs["training_image"] == \
        "1234.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.6.0-gpu-py312"


@patch("scripts.sagemaker_trainer.image_uris")
@patch("scripts.sagemaker_trainer.ModelTrainer")
def test_factory_volume_size_at_least_100(mock_mt, mock_image):
    mock_image.retrieve.return_value = "img"
    make_trainer(**_make_kwargs())
    assert mock_mt.call_args.kwargs["compute"].volume_size_in_gb >= 100, \
        "EBS volume must be ≥100GB to avoid disk-full at /opt/ml/checkpoints"


@patch("scripts.sagemaker_trainer.image_uris")
@patch("scripts.sagemaker_trainer.ModelTrainer")
def test_factory_passes_source_code_with_requirements(mock_mt, mock_image):
    mock_image.retrieve.return_value = "img"
    make_trainer(**_make_kwargs())

    sc = mock_mt.call_args.kwargs["source_code"]
    assert sc.entry_script == "sm_job_train.py"
    assert sc.requirements is not None
    assert "requirements-sagemaker.txt" in sc.requirements
