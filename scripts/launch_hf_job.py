"""Launch an HF Jobs training run.

Usage (Exp 3 — default, 1xL4):
    uv run python scripts/launch_hf_job.py \
      -- --backbone resnet18 --chunk-size 10 \
         --hf-upload-repo PenTest-duck/cu-vla-exp3-checkpoints

Usage (Exp 2):
    uv run python scripts/launch_hf_job.py \
      -- --experiment exp2 --backbone resnet18 --chunk-size 10

The dataset is auto-downloaded from HF Hub inside the training script.
Pass --experiment exp2|exp3 after -- to select which experiment to train.
Defaults to exp3 (MiniWoB-Pygame). Default GPU is 1xL4 (24GB, bf16).
"""

import argparse
import os
import sys

from huggingface_hub import run_uv_job


SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "hf_job_train.py")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch HF Jobs training",
        usage="%(prog)s [launcher-options] -- [train.py options]",
    )
    parser.add_argument("--flavor", type=str, default="1x-l4",
                        help="HF Jobs hardware flavor (default: 1x-l4)")
    parser.add_argument("--timeout", type=str, default="4h",
                        help="Job timeout (default: 4h)")
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

    print(f"Launching HF Job:")
    print(f"  Flavor:  {args.flavor}")
    print(f"  Timeout: {args.timeout}")
    print(f"  Script:  {SCRIPT_PATH}")
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
        SCRIPT_PATH,
        script_args=train_argv,
        flavor=args.flavor,
        timeout=args.timeout,
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
