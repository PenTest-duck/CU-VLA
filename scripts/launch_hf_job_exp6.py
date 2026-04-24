"""Launch Experiment 6 training on HF Jobs.

Usage (Phase A, L4x1, 4h):
    uv run python scripts/launch_hf_job_exp6.py \
        --flavor l4x1 --timeout 4h -- \
        --data-dir data/phase-a-lclick \
        --epochs 5 \
        --hf-upload-repo PenTest-duck/cu-vla-exp6-phasea-ckpt
"""
import argparse
import os
import sys

from huggingface_hub import run_uv_job, get_token


DEFAULT_SCRIPT = os.path.join(os.path.dirname(__file__), "hf_job_train_exp6.py")


def main() -> None:
    parser = argparse.ArgumentParser(usage="%(prog)s [launcher-opts] -- [train.py opts]")
    parser.add_argument("--flavor", default="l4x1")
    parser.add_argument("--timeout", default="4h")
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--script", default=DEFAULT_SCRIPT)
    parser.add_argument("--detach", action="store_true")

    argv = sys.argv[1:]
    if "--" in argv:
        i = argv.index("--")
        launcher_argv, train_argv = argv[:i], argv[i + 1:]
    else:
        launcher_argv, train_argv = argv, []
    args = parser.parse_args(launcher_argv)

    secrets = {}
    token = get_token()
    if token:
        secrets["HF_TOKEN"] = token

    kwargs = {}
    if args.namespace:
        kwargs["namespace"] = args.namespace

    job = run_uv_job(args.script, script_args=train_argv, flavor=args.flavor, timeout=args.timeout, secrets=secrets, **kwargs)
    print(f"Launched: {job.url}  (id={job.id})")
    if args.detach:
        return
    from huggingface_hub import fetch_job_logs
    for log in fetch_job_logs(job_id=job.id):
        print(log)


if __name__ == "__main__":
    main()
