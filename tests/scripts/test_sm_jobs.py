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
