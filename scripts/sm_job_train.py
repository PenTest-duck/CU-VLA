"""SageMaker entrypoint script (placeholder).

This file will be implemented in Task 5 of the SageMaker migration plan. It
runs *inside* the SageMaker training container: clones the configured branch
from git, then exec's ``python -u -m $TRAIN_MODULE $TRAIN_ARGS``.

For now, this stub exists so Pydantic validation in
``scripts/sagemaker_trainer.py`` (which checks that the entry_script is a
file within source_dir) passes during local smoke tests.
"""
from __future__ import annotations

import sys


def main() -> int:
    raise NotImplementedError(
        "sm_job_train.py is a Task 5 deliverable. Do not run this stub."
    )


if __name__ == "__main__":
    sys.exit(main())
