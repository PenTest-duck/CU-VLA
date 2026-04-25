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
from pathlib import Path
from typing import Optional

# Allow running as `python scripts/sm_jobs.py` (direct file path) by
# putting the project root on sys.path. When run as `python -m scripts.sm_jobs`
# or imported under pytest, this is a harmless no-op.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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

    Reads the s3://.../checkpoints/<job-name>/{best,final}.pt — the actual
    S3 prefix is taken from the job's CheckpointConfig.S3Uri (the SDK
    appends a timestamp suffix to the base_job_name we passed in, so
    constructing the prefix from the resolved job name would be wrong).
    Then uploads to HF Hub if the user provides --hf-repo.
    """
    # NOTE: args.s3_bucket is intentionally unused after the fix; the bucket
    # is parsed from CheckpointConfig.S3Uri below. The CLI flag is kept for
    # backward-compatibility.
    name = resolve_job(args.name, tuple(args.filter or ()), args.region)
    if not args.hf_repo:
        sys.exit("ERROR: --hf-repo required for reconcile-ckpts")

    sm = _sm_client(args.region)
    desc = sm.describe_training_job(TrainingJobName=name)
    ckpt_uri = desc.get("CheckpointConfig", {}).get("S3Uri")
    if not ckpt_uri:
        sys.exit(f"ERROR: job {name} has no CheckpointConfig.S3Uri (was it spot?)")

    # Parse s3://bucket/prefix/path/ → (bucket, key_prefix).
    if not ckpt_uri.startswith("s3://"):
        sys.exit(f"ERROR: unexpected CheckpointConfig.S3Uri format: {ckpt_uri}")
    bucket_and_key = ckpt_uri[len("s3://"):]
    parts = bucket_and_key.split("/", 1)
    s3_bucket = parts[0]
    prefix = (parts[1] if len(parts) > 1 else "").rstrip("/") + "/"

    s3 = boto3.client("s3", region_name=args.region)
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
