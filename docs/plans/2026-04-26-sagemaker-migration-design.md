# SageMaker Migration — Design Document

## Goal

Migrate the GPU training infrastructure for CU-VLA experiments from Hugging Face Jobs to AWS SageMaker before HF credits run out. Preserve the existing two-tier launcher pattern (`launch_hf_job.py` + `launch_hf_job_<exp>.py`) and keep HF Hub as the canonical store for datasets and final/best checkpoints. Use spot instances by default to stretch the $10k AWS grant from ~5k to ~15k+ GPU-hours.

## Scope

**In scope:**
- Replacement launcher + entrypoint scripts for SageMaker (`scripts/launch_sm_job*.py`, `scripts/sm_job_train.py`, `scripts/sagemaker_estimator.py`).
- Operator CLI helper (`scripts/sm_jobs.py`) for ls / status / logs / stop / describe / url / cost / reconcile-ckpts.
- One-time AWS bootstrap (region pick, IAM execution role, S3 bucket, SSM Parameter Store secrets, GPU quota request, Budget alerts).
- Spot-instance auto-resume mechanics (`checkpoint_s3_uri` + entrypoint resume detection + wandb run continuation).
- ~3-line patch to `experiments/action_primitives/train.py` to restore `best_val_loss` from `best.pt` on `--resume`.
- Migration sequencing (smoke test → spot rehearsal → first real run) and rollback path.

**Out of scope (deferred):**
- Multi-GPU / multi-node training (designed-around but not built; trivial to add when needed).
- Custom Docker image / ECR (we use the pre-built SageMaker PyTorch DLC).
- Migrating data away from HF Hub (Hub remains source-of-truth).
- SageMaker Experiments / Model Registry (wandb already covers this).
- Auto-fallback from spot to on-demand (explicit user re-launch).
- Pruning of `step_*.pt` to bound disk (deferred until disk full encountered).

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Compute footprint | 1×L40S (`ml.g6e.xlarge`) now, leave room for 4×GPU later | Mirrors current HF Jobs setup; multi-GPU added when an experiment actually needs it |
| Spot vs on-demand | Spot-first with auto-resume; `--no-spot` escape hatch | ~70% spot discount stretches $10k from ~5k → ~15k+ GPU-hours. `train.py` already supports `--resume`. |
| Persistence layer | HF Hub canonical for data + final/best ckpts; S3 for in-flight spot ckpts | Zero code change in `hf_sync.py`/`load_dataset()`; S3 is forced by spot-resume mechanics anyway |
| Container image | Pre-built SageMaker PyTorch DLC (framework_version=2.6, py312) + `requirements-sagemaker.txt` | torch+torchvision baked in; ~1–2 min install for the small remaining list (transformers, peft, pygame-ce, wandb, datasets). Zero Docker maintenance. |
| Source delivery | Git-clone-on-startup inside container (mirrors today's `hf_job_train.py` flow) | Matches existing mental model; `CU_VLA_BRANCH` env var preserved; no risk of launching dirty local state |
| Region | us-west-2 | ~70% spot discount (vs ~45% in ap-southeast-2), broad L40S spot capacity, ~$2k more effective credits than Sydney. Latency irrelevant for async `.fit()`. |
| Secrets | AWS Systems Manager Parameter Store SecureString | $0/month (vs Secrets Manager $0.40/secret/month). KMS decrypt via `alias/aws/ssm` is free. Token rotation = `put-parameter --overwrite`, no relaunch needed. |
| Launcher pattern | Generic launcher + factory + per-experiment override (mirrors today's HF Jobs two-tier) | New experiments add ~30-LOC override; production paths come from a branch (no dirty launches) |
| File split | `sagemaker_estimator.py` (factory, no I/O) + `sm_job_train.py` (entrypoint) + `launch_sm_job.py` (generic CLI) + `launch_sm_job_exp6.py` (override) | Each file single-purpose, testable in isolation |
| Observability | wandb (training metrics) + CloudWatch (stdout) + SageMaker console + Budget alerts (cost) | One canonical tool per layer, no duplicates. wandb continues across spot reclaims via `WANDB_RUN_ID=$SM_TRAINING_JOB_NAME`. |

## File Layout

```
scripts/
├── sagemaker_estimator.py      [NEW] ~80 LOC — pure factory: make_estimator(...)
├── sm_job_train.py             [NEW] ~60 LOC — in-container entrypoint
├── launch_sm_job.py            [NEW] ~80 LOC — generic launcher CLI
├── launch_sm_job_exp6.py       [NEW] ~30 LOC — exp6 defaults (Phase B0/B1, branch, instance)
├── sm_jobs.py                  [NEW] ~200 LOC — operator CLI: ls, status, logs, stop, describe, url, cost, reconcile-ckpts
├── launch_hf_job.py            [unchanged, deprecated] retained as fallback
├── launch_hf_job_exp6.py       [unchanged, deprecated] retained as fallback
├── hf_job_train.py             [unchanged, deprecated]
└── hf_job_train_exp6.py        [unchanged, deprecated]

requirements-sagemaker.txt      [NEW] — small list of deps not in the SageMaker PyTorch DLC

docs/
├── plans/2026-04-26-sagemaker-migration-design.md  [this doc]
└── ops/aws-bootstrap.md        [NEW] — one-time setup walkthrough

experiments/action_primitives/train.py   [MOD] +3 LOC — restore best_val_loss from best.pt on --resume
```

**Untouched:** every file under `experiments/action_primitives/` *except* the 3-line patch to `train.py` (`backbones.py`, `config.py`, `dataset.py`, `env.py`, `evaluate.py`, `expert.py`, `generate_data.py`, `generator.py`, `heads.py`, `hf_sync.py`, `history.py`, `instructions.py`, `losses.py`, `metrics.py`, `model.py`, `proprio.py`, `recovery.py`, `scene.py`, `trunk.py`). All other experiments (`act_drag_label`, `mini_editor`, `miniwob_pygame`, `reactive_clicks`) untouched. All dataset/checkpoint code paths unchanged.

## Architecture

```
                     ┌─ scripts/launch_sm_job_exp6.py  (exp6-specific defaults, ~30 LOC)
                     │      │
                     │      └─→ scripts/launch_sm_job.py  (generic launcher CLI, ~80 LOC)
                     │             │
                     │             └─→ scripts/sagemaker_estimator.py  (factory, ~80 LOC)
                     │                    │
                     │                    └─→ sagemaker.pytorch.PyTorch(...)
                     │                              │
                     │                              └─.fit({"checkpoints": s3://...})
                     │                                        │
   [your laptop]     │                                        │   [container in AWS]
   ──────────────────┼────────────────────────────────────────┼──────────────────────
                     ▼                                        ▼
                                                  scripts/sm_job_train.py  (entrypoint, ~60 LOC)
                                                        │
                                                        ├─ ssm GetParameter → set HF_TOKEN, WANDB_API_KEY env
                                                        ├─ set WANDB_RUN_ID=$SM_TRAINING_JOB_NAME, WANDB_RESUME=allow
                                                        ├─ set SDL_VIDEODRIVER=dummy
                                                        ├─ git clone --branch $CU_VLA_BRANCH
                                                        ├─ scan /opt/ml/checkpoints/ for latest step_*.pt
                                                        └─ python -u -m $TRAIN_MODULE --resume <ckpt>?
                                                              --out-dir /opt/ml/checkpoints  ...passthrough
                                                                  │
                                                                  ├─ HF Hub: load_dataset(...), upload best.pt + final.pt
                                                                  └─ S3 (auto-sync): /opt/ml/checkpoints ↔ s3://.../checkpoints
```

## Spot Resume Mechanics

This is the critical and trickiest part. SageMaker spot reclaim is a mid-run kill, but with the right plumbing the user sees only a brief delay and a continuous wandb run.

**How SageMaker's checkpoint sync works:**
1. Estimator passes `checkpoint_s3_uri="s3://bucket/path"`.
2. SageMaker auto-mounts `/opt/ml/checkpoints/` inside the container as an **S3-backed sync volume** — anything written there is uploaded to S3 in the background, asynchronously.
3. On a spot reclaim, SageMaker re-launches a fresh instance, downloads `s3://bucket/path/* → /opt/ml/checkpoints/` *before* starting your entrypoint, then re-runs it.
4. Entrypoint detects existing files and resumes.

**Three wiring points:**

1. **Launcher sets:**
   ```python
   PyTorch(
       use_spot_instances=True,
       max_run=4 * 3600,              # per-attempt training cap (4h for B0)
       max_wait=8 * 3600,             # 2× max_run; allows 1–2 reclaim retries
       checkpoint_s3_uri=f"s3://cu-vla-sm-{ACCT}/checkpoints/{job_name}",
       volume_size=100,               # GB EBS for /opt/ml/{input,model,checkpoints}
       keep_alive_period_in_seconds=0,
       ...
   )
   ```

2. **Train args force the synced dir:** `--out-dir /opt/ml/checkpoints` so `step_*.pt`, `best.pt`, and `final.pt` all land on the S3-synced volume.

3. **Entrypoint resume detection** (in `sm_job_train.py`):
   ```python
   ckpt_dir = Path("/opt/ml/checkpoints")
   step_ckpts = sorted(ckpt_dir.glob("step_*.pt"))   # 5-digit zero-pad → lex == numeric
   resume_args = ["--resume", str(step_ckpts[-1])] if step_ckpts else []
   subprocess.run(["python", "-u", "-m", train_module, *resume_args, *passthrough_args])
   ```

**Wandb continuation across reclaims:** the entrypoint sets the following env vars **before** invoking the train module, so `wandb.init()` in `train.py` honors them:
```python
os.environ.setdefault("WANDB_RUN_ID", os.environ["SM_TRAINING_JOB_NAME"])
os.environ.setdefault("WANDB_RESUME", "allow")
```
Same job, same wandb run, same metric history, even after a reclaim.

**`best_val_loss` restoration patch (3 LOC in `train.py`):**
```python
if args.resume:
    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    start_step = ckpt["step"]
    # NEW: restore best_val_loss so post-resume evals don't overwrite a real best with a worse one
    best_path = Path(args.resume).parent / "best.pt"
    if best_path.exists():
        best_val_loss = torch.load(best_path, map_location="cpu")["val_loss"]
    print(f"resumed from step {start_step}")
```

**Worst-case data loss on reclaim:** between two `step_*.pt` saves (`ckpt_every_steps=20`, ~1 min on B0). Acceptable. S3 sync is best-effort — a step ckpt written ~10s before reclaim *may* not have hit S3, but the previous one will have.

## AWS Bootstrap (one-time)

Goal: get from "blank account with credits applied" to "ready to run `launch_sm_job_exp6.py`" in under 30 minutes. **Quota request goes first** because it has the longest lead time.

### Region: us-west-2

Largest L40S spot pool, lowest g6e pricing, irrelevant latency for async `.fit()`. Sydney (ap-southeast-2) loses ~$2k of credits to higher pricing + smaller spot discount; only chosen if us-west-2 quota approval stalls.

### IAM execution role: `SageMakerExecutionRole-CU-VLA`

- Trust policy: `sagemaker.amazonaws.com`.
- Managed: `AmazonSageMakerFullAccess`, `AmazonS3FullAccess` (scope down later if needed).
- Inline policy for SSM Parameter Store + KMS:
  ```json
  {
    "Version": "2012-10-17",
    "Statement": [
      { "Effect": "Allow",
        "Action": ["ssm:GetParameter"],
        "Resource": "arn:aws:ssm:us-west-2:*:parameter/cu-vla/*" },
      { "Effect": "Allow",
        "Action": ["kms:Decrypt"],
        "Resource": "*",
        "Condition": { "StringEquals": { "kms:ViaService": "ssm.us-west-2.amazonaws.com" } } }
    ]
  }
  ```
- ARN stored in `~/.cu-vla/sagemaker.toml` and read by `launch_sm_job.py`.

### S3 bucket: `s3://cu-vla-sm-<account-id>/`

Layout:
```
s3://cu-vla-sm-<acct>/
├─ checkpoints/<job-name>/   # spot-resume sync volume, auto-managed by SageMaker
├─ logs/                     # CloudWatch overflow / batch download
└─ tmp/                      # source.tar.gz upload zone (SageMaker SDK uses this)
```

Lifecycle policy:
- `checkpoints/*`: delete after 30 days (final/best are on HF Hub anyway; S3 is scratch)
- `tmp/*`: delete after 7 days

### Secrets: AWS Systems Manager Parameter Store (SecureString)

```bash
aws ssm put-parameter --region us-west-2 --type SecureString \
  --name /cu-vla/wandb-api-key --value "$WANDB_API_KEY"
aws ssm put-parameter --region us-west-2 --type SecureString \
  --name /cu-vla/hf-token --value "$HF_TOKEN"
```

Cost: ~$0/month (Standard tier params + AWS-managed KMS key for SSM is free; ~$0.0002/job for KMS Decrypt API calls).

Entrypoint pulls secrets via boto3 (uses the IAM execution role's credentials automatically):
```python
import boto3, os
ssm = boto3.client("ssm", region_name=os.environ.get("AWS_REGION", "us-west-2"))
for env_name, param_name in [("WANDB_API_KEY", "/cu-vla/wandb-api-key"),
                              ("HF_TOKEN", "/cu-vla/hf-token")]:
    if env_name not in os.environ:
        os.environ[env_name] = ssm.get_parameter(Name=param_name, WithDecryption=True)["Parameter"]["Value"]
```

### GPU quota request

Submit via Service Quotas console: **SageMaker → `ml.g6e.xlarge` for spot training job usage → 1** (or 4 if you want headroom for parallel ablation runs). Approval typically 1–3 days. **Submit on day 0** — blocks the smoke test.

### Budget alerts

`aws budgets create-budget` with alerts at 25%, 50%, 75%, 90% of $10k → email. ~5 min one-time setup.

### Local CLI prereq

`pip install sagemaker boto3` (added to `requirements-dev.txt`). `aws configure sso login` already done.

## Cost Model & Spot Policy

**Reference: `ml.g6e.xlarge` (1× L40S 48GB, 4 vCPU, 32 GB RAM) in us-west-2**

| Mode | $/hr | 5h B0-style run | $10k credits buy |
|---|---|---|---|
| On-demand | $1.861 | $9.30 | ~5,374 GPU-hours |
| Spot (avg 70% off) | ~$0.56 | ~$2.80 | ~17,857 GPU-hours |
| Spot (worst-case 50% off) | ~$0.93 | ~$4.65 | ~10,750 GPU-hours |

**Translation:** spot turns $10k into ~12k–18k GPU-hours = ~100–150 B0-class runs of 4 hours each, or ~25–35 multi-day full-pipeline runs.

**Defaults in `sagemaker_estimator.py`:**
```python
DEFAULTS = dict(
    use_spot_instances=True,
    max_run=4 * 3600,
    max_wait=8 * 3600,
    keep_alive_period_in_seconds=0,
    volume_size=100,
)
```

`launch_sm_job_exp6.py` overrides `max_run` per phase: B0=4h, B1=8h, etc. `--no-spot` flag in `launch_sm_job.py` for short smoke tests where waiting for spot capacity is wasteful.

## Observability

| Layer | Tool | Setup |
|---|---|---|
| Training metrics | wandb | unchanged; spot reclaim continuity via `WANDB_RUN_ID=$SM_TRAINING_JOB_NAME`, `WANDB_RESUME=allow` |
| Container stdout/stderr | CloudWatch Logs | auto; 30-day retention set in bootstrap via `aws logs put-retention-policy` |
| Live log streaming to laptop | `Estimator.logs()` | mirrors today's `fetch_job_logs()` UX |
| Job lifecycle | SageMaker console | auto |
| Cost | AWS Budget alerts at 25/50/75/90% | one-time bootstrap |

**Not building:** SageMaker Experiments / Model Registry (wandb), CloudWatch dashboards (overlap), SNS notifications (`--no-detach` blocks until done).

## Migration Sequencing

The "cold path" — current B0 finishes on HF Jobs, then migrate carefully.

**Phase 0 (in parallel with current HF Jobs B0 run):**
- Day 0: Submit `ml.g6e.xlarge` spot quota in us-west-2 (1–3 day approval, longest lead time, blocks Phase 2).
- Day 0–1: Run `aws-bootstrap.md` walkthrough (IAM, S3, lifecycle, Parameter Store, Budget alerts). ~30 min hands-on.
- Day 0–1: Author the four scripts + `requirements-sagemaker.txt`. Local-only; not run yet.

**Phase 1 — Smoke test (after current B0 finishes):**
- `launch_sm_job_exp6.py --max-run 600 --no-spot --branch feat/exp6-phase-b0` for 10 steps.
- Assert: container starts, deps install, repo clones, `load_dataset()` from HF Hub works, `step_00010.pt` lands on S3-synced volume, `best.pt` push to HF Hub succeeds, wandb run appears with the SageMaker job name as run ID.
- Cost: ~$0.30 (10 min on-demand).
- **Gate:** all 5 assertions pass before proceeding.

**Phase 2 — Spot rehearsal:**
- Same smoke test with `--spot --max-run 600 --max-wait 1200`.
- Assert: spot allocation succeeds within `max_wait`.
- Manually `aws sagemaker stop-training-job` mid-run; SageMaker auto-relaunches; second attempt resumes from `step_*.pt`; wandb run continues.
- Cost: ~$0.10.
- **Gate:** spot reclaim path validated end-to-end before any real run.

**Phase 3 — First real training run:**
- Phase B1 (or whatever's next) launches via `launch_sm_job_exp6.py --phase b1 --spot` against `feat/exp6-phase-b1`.
- HF Jobs scripts retained in repo as fallback.

## Rollback

At any point in Phases 1–3, the user can revert with zero data loss:
- All HF Jobs scripts (`launch_hf_job_exp6.py`, `hf_job_train_exp6.py`) untouched in the repo.
- HF Hub artifacts (data + ckpts) unchanged — they're the source of truth.
- One command swap: `uv run python scripts/launch_hf_job_exp6.py ...` works exactly as today.
- S3 contents auto-cleaned by lifecycle policy.

After a successful Phase 3 run, mark HF Jobs scripts deprecated in their docstrings (don't delete — useful reference, zero cost to keep) and add a one-line note to AGENTS.md.

## Edge Cases & Failure Modes

| # | Failure | Likelihood | Mitigation |
|---|---------|------------|------------|
| 1 | Spot allocation timeout (`MaxWaitTimeExceeded`) | Med | Launcher prints actionable error; no silent on-demand fallback |
| 2 | Source git-clone fails (branch typo / deleted) | Med | Launcher pre-validates: `git ls-remote --exit-code origin <branch>` before `.fit()` |
| 3 | HF Hub upload of best.pt times out / 503s | Low–Med | `train.py` already catches and logs; best.pt persisted to S3 anyway; `sm_jobs.py reconcile-ckpts <job>` reconciles after the fact |
| 4 | Disk full on `/opt/ml/checkpoints/` | Low (short runs); real (>24h runs) | `volume_size=100` GB at bootstrap; ckpt pruning patch deferred until first occurrence |
| 5 | Token rotation (HF or wandb) | Low | `aws ssm put-parameter --overwrite` — next job picks up; no relaunch |
| 6 | Pygame import in headless container tries to init display | Med | Entrypoint sets `SDL_VIDEODRIVER=dummy` unconditionally |
| 7 | PyTorch DLC version mismatch with `torch>=2.6,<2.11` | Low | Estimator pins `framework_version="2.6"` + `py_version="py312"`; `image_uris.retrieve()` resolves the matching DLC |
| 8 | wandb run continuation race (`WANDB_RUN_ID` set after `wandb.init`) | Low | Entrypoint sets env vars **before** invoking train module |
| 9 | Cost runaway from forgotten detached jobs | Low | `max_run` hard-cap per attempt; Budget alerts; `aws sagemaker list-training-jobs --status-equals InProgress` |
| 10 | boto3 inside container can't auth | Very low | SageMaker auto-injects creds via IAM execution role; boto3 picks up via metadata service |

**Not safeguarded against (acceptable risk):**
- Multi-region failover.
- Network partition retry — SageMaker handles internally.
- Custom step-checkpoint pruning — defer until disk fill encountered.

## Operator UX

Goal: parity with today's HF Jobs ergonomics for both the user and Claude. **One CLI helper module (`scripts/sm_jobs.py`, ~200 LOC) wraps the SageMaker SDK so neither party needs to remember `aws sagemaker ...` subcommand syntax.**

### Naming convention (job names = stable handles)

All training jobs get names of the form `cu-vla-<exp>-<phase>-YYYYMMDD-HHMMSS`. Predictable, sortable, lex-comparable, parseable. Example: `cu-vla-exp6-phaseb0-20260427-093142`. Set in `launch_sm_job_<exp>.py` at launch time and passed to the estimator.

### Initiate (unchanged mental model from HF Jobs)

```bash
uv run python scripts/launch_sm_job_exp6.py --phase b0 --spot \
    -- --epochs 5 --hf-data-repo PenTest-duck/cu-vla-exp6-phaseb0
```

Default behavior: blocks, streams CloudWatch logs to terminal, prints summary on completion. `--detach` returns immediately with the job name. Args mirror today's HF Jobs launcher.

### `sm_jobs.py` subcommand surface

All subcommands accept either a literal job name, the literal `latest` (most recent CU-VLA job), or `--filter <substring>` matching against job names.

```
sm_jobs.py ls [--status <state>] [--limit N]            # list active + recent jobs (default last 20, sorted by start)
sm_jobs.py status [name|latest|--filter ...]            # one-line: state, instance, runtime, billable seconds
sm_jobs.py logs <name|latest> [--follow]                # batch or stream CloudWatch logs (mirrors HF Jobs streaming UX)
sm_jobs.py stop <name|latest|--filter ...>              # stop running job(s); confirms before kill if running > 1 min
sm_jobs.py describe <name|latest>                       # full describe-training-job output (failure reason, secondary status)
sm_jobs.py url <name|latest>                            # print SageMaker console URL (clickable from terminal)
sm_jobs.py cost <name|latest>                           # estimated cost = BillableTimeInSeconds × $/hr for that job
sm_jobs.py reconcile-ckpts <name|latest>                # push any S3 ckpts not yet on HF Hub (failure mode #3)
```

Defaults compact + parseable; `--verbose` / `--json` for richer formats. All subcommands exit with non-zero if the operation fails or no matching job is found.

### Live monitoring while a job runs

| Latency / detail | Tool |
|---|---|
| Foreground stream | launcher's default (CloudWatch tailed inline) — same as today |
| Out-of-band tail | `sm_jobs.py logs <name|latest> --follow` from another terminal |
| Live training metrics | wandb web UI (unchanged) |

### Cancel

`Ctrl-C` on the foreground launcher does **not** stop the job — it only detaches the log stream. Same as HF Jobs. Real stop:

```bash
uv run python scripts/sm_jobs.py stop latest               # most recent running job
uv run python scripts/sm_jobs.py stop --filter phaseb0     # the running B0 job
uv run python scripts/sm_jobs.py stop cu-vla-exp6-phaseb0-20260427-093142
```

Calls `boto3.client("sagemaker").stop_training_job(TrainingJobName=name)`. Confirms before kill if the job has been running > 1 min (paranoia for paid runs).

### Debug ladder (when something breaks)

1. **Foreground terminal output** during streaming — first stop.
2. **`sm_jobs.py logs <name>`** — pulls CloudWatch without re-launching anything; works after job exited.
3. **`sm_jobs.py describe <name>`** — full job metadata: failure reason, secondary status (`Stopping`, `MaxRuntimeExceeded`, `InternalServerError`), output S3 paths.
4. **wandb dashboard** — training-time metrics.
5. **`sm_jobs.py url <name>`** — opens SageMaker console for the job (web UI shows everything).

### Claude-specific ergonomics

When helping debug a run via Bash, Claude needs commands that produce parseable output without flooding context:

- `uv run python scripts/sm_jobs.py status latest` → one-line summary
- `uv run python scripts/sm_jobs.py logs latest | tail -200` → recent slice
- `uv run python scripts/sm_jobs.py describe latest --json` → structured failure cause

`sm_jobs.py` is designed to fit cleanly inside tool-result blocks. The `latest` shortcut + predictable job names mean Claude doesn't need to bash up arrow or remember UUIDs.

### What we're NOT building

- No GUI / TUI dashboard. CloudWatch + SageMaker console + wandb cover all needs.
- No auto-retry on transient failure. Explicit re-run.
- No completion notification. Foreground launcher blocks; wandb's desktop notification covers detached runs.

## Open Questions

None. Region, secrets store, container strategy, persistence layer, spot policy, file split, and edge case mitigations all decided in §1–§7 of the brainstorming.

## Implementation Plan

To be authored next via the writing-plans skill. Expected breakdown: ~12 tasks covering bootstrap (3) + script authoring (5: factory, generic launcher, entrypoint, exp6 override, sm_jobs CLI) + train.py patch (1) + smoke + spot rehearsal + first real run (3).
