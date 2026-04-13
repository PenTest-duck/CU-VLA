"""Launch an HF Jobs training run with dataset volume mounting.

Usage:
    uv run python scripts/launch_hf_job.py \
      --flavor t4-medium --timeout 4h \
      -- --backbone resnet18 --chunk-size 10

    uv run python scripts/launch_hf_job.py \
      --flavor a10g-small --timeout 6h \
      -- --backbone resnet18 --chunk-size 10 \
         --hf-upload-repo PenTest-duck/cu-vla-checkpoints

The dataset (PenTest-duck/cu-vla-data) is volume-mounted at /data inside the
job container, so no snapshot_download is needed.
"""

import argparse
import sys

from huggingface_hub import run_uv_job, Volume


SCRIPT_URL = "https://raw.githubusercontent.com/PenTest-duck/CU-VLA/main/scripts/hf_job_train.py"
DEFAULT_DATA_REPO = "PenTest-duck/cu-vla-data"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch HF Jobs training with volume-mounted dataset",
        usage="%(prog)s [launcher-options] -- [train.py options]",
    )
    parser.add_argument("--flavor", type=str, default="t4-medium",
                        help="HF Jobs hardware flavor (default: t4-medium)")
    parser.add_argument("--timeout", type=str, default="4h",
                        help="Job timeout (default: 4h)")
    parser.add_argument("--data-repo", type=str, default=DEFAULT_DATA_REPO,
                        help="HF dataset repo to mount (default: %(default)s)")
    parser.add_argument("--namespace", type=str, default=None,
                        help="HF namespace to run the job in")
    parser.add_argument("--detach", action="store_true",
                        help="Print job ID and return immediately")

    # Split on "--" to separate launcher args from train.py args
    argv = sys.argv[1:]
    if "--" in argv:
        split_idx = argv.index("--")
        launcher_argv = argv[:split_idx]
        train_argv = argv[split_idx + 1:]
    else:
        launcher_argv = argv
        train_argv = []

    args = parser.parse_args(launcher_argv)

    # Mount the dataset repo at /data (read-only)
    volumes = [
        Volume(type="dataset", source=args.data_repo, mount_path="/data"),
    ]

    print(f"Launching HF Job:")
    print(f"  Flavor:  {args.flavor}")
    print(f"  Timeout: {args.timeout}")
    print(f"  Dataset: {args.data_repo} -> /data")
    print(f"  Script:  {SCRIPT_URL}")
    print(f"  Args:    {train_argv}")

    kwargs = {}
    if args.namespace:
        kwargs["namespace"] = args.namespace

    # Pass HF token so the job can upload checkpoints and access gated repos
    from huggingface_hub import get_token
    token = get_token()
    secrets = {}
    if token:
        secrets["HF_TOKEN"] = token
    else:
        print("  WARNING: No HF token found. Run `huggingface-cli login` first.")

    job = run_uv_job(
        SCRIPT_URL,
        script_args=train_argv,
        flavor=args.flavor,
        timeout=args.timeout,
        volumes=volumes,
        secrets=secrets,
        **kwargs,
    )

    print(f"\nJob launched: {job.url}")
    print(f"Job ID: {job.id}")

    if args.detach:
        return

    # Stream logs
    print("\n--- Streaming logs ---")
    from huggingface_hub import fetch_job_logs
    for log in fetch_job_logs(job_id=job.id):
        print(log)


if __name__ == "__main__":
    main()
