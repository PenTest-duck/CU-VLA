# AWS Bootstrap for CU-VLA SageMaker Training

One-time setup. Goal: get from "blank account with credits applied" to
"ready to run `launch_sm_job_exp6.py`" in ~30 minutes (excluding the 1–3 day
GPU-quota approval wait, which is the longest single step).

**Region:** us-west-2 (chosen in [SageMaker migration design doc](../plans/2026-04-26-sagemaker-migration-design.md) §3.1).

**Account ID:** `<your-12-digit-account-id>` — referred to as `<acct>` below.
Find it with `aws sts get-caller-identity --query Account --output text`.

## 0. Prereqs

- AWS CLI v2 configured + SSO logged in (`aws configure sso login`).
- Local Python env has `sagemaker>=3.0` + `boto3` installed (`uv sync` after Task 1).
- `WANDB_API_KEY` and `HF_TOKEN` available in your local environment (we upload them to Parameter Store in §4).

## 1. GPU quota request

Submit on day 0 — approval typically takes a few hours to 3 days, and **everything else is parallelizable**.

```bash
# Find the spot quota for ml.g6e.xlarge in us-west-2
aws service-quotas list-service-quotas --region us-west-2 --service-code sagemaker --max-results 200 \
  --query "Quotas[?contains(QuotaName, 'g6e.xlarge') && contains(QuotaName, 'spot')].[QuotaName,Value,QuotaCode]" \
  --output table

# Request increase (replace <QuotaCode> with the value from above)
aws service-quotas request-service-quota-increase --region us-west-2 \
  --service-code sagemaker --quota-code <QuotaCode> --desired-value 1

# Repeat for the on-demand quota (used by --no-spot smoke tests)
aws service-quotas list-service-quotas --region us-west-2 --service-code sagemaker --max-results 200 \
  --query "Quotas[?contains(QuotaName, 'g6e.xlarge') && contains(QuotaName, 'training job usage') && !contains(QuotaName, 'spot')].[QuotaName,Value,QuotaCode]" \
  --output table

aws service-quotas request-service-quota-increase --region us-west-2 \
  --service-code sagemaker --quota-code <on-demand-QuotaCode> --desired-value 1
```

Track approval:

```bash
aws service-quotas list-requested-service-quota-change-history --region us-west-2 --service-code sagemaker \
  --query "RequestedQuotas[].[QuotaName,DesiredValue,Status]" \
  --output table
```

Wait until both rows show `Status=APPROVED` before launching any training job (Tasks 11–12).

## 2. IAM execution role: SageMakerExecutionRole-CU-VLA

Trust policy file `trust.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "sagemaker.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }]
}
```

Create the role and attach managed policies:

```bash
aws iam create-role --role-name SageMakerExecutionRole-CU-VLA \
  --assume-role-policy-document file://trust.json

aws iam attach-role-policy --role-name SageMakerExecutionRole-CU-VLA \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy --role-name SageMakerExecutionRole-CU-VLA \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

Inline policy file `ssm-policy.json` (SSM Parameter Store + KMS decrypt, scoped to the `/cu-vla/*` prefix):

```json
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
```

Attach the inline policy:

```bash
aws iam put-role-policy --role-name SageMakerExecutionRole-CU-VLA \
  --policy-name SSMParameterAccess \
  --policy-document file://ssm-policy.json
```

Get the role ARN (you'll set it as `CU_VLA_SM_ROLE_ARN` in §7):

```bash
aws iam get-role --role-name SageMakerExecutionRole-CU-VLA --query Role.Arn --output text
# Output: arn:aws:iam::<acct>:role/SageMakerExecutionRole-CU-VLA
```

## 3. S3 bucket: `s3://cu-vla-sm-<acct>/`

```bash
ACCT=$(aws sts get-caller-identity --query Account --output text)
aws s3 mb s3://cu-vla-sm-$ACCT --region us-west-2
```

Lifecycle policy file `lifecycle.json` (auto-delete old checkpoints + tmp uploads):

```json
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
```

Apply it:

```bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket cu-vla-sm-$ACCT \
  --lifecycle-configuration file://lifecycle.json
```

## 4. SSM Parameter Store secrets

```bash
aws ssm put-parameter --region us-west-2 --type SecureString \
  --name /cu-vla/wandb-api-key --value "$WANDB_API_KEY"

aws ssm put-parameter --region us-west-2 --type SecureString \
  --name /cu-vla/hf-token --value "$HF_TOKEN"
```

Cost: ~$0/month (Standard tier params + AWS-managed KMS key for SSM is free; ~$0.0002/job for KMS Decrypt).

To rotate later: add `--overwrite` flag with the new value. The next training job picks up the new value automatically (no relaunch logic change).

## 5. Budget alerts

Budget JSON `budget.json` (replace `<acct>` with your account ID):

```json
{
  "BudgetName": "cu-vla-credits",
  "BudgetLimit": { "Amount": "10000", "Unit": "USD" },
  "TimeUnit": "ANNUALLY",
  "BudgetType": "COST",
  "CostFilters": { "Service": ["Amazon SageMaker"] }
}
```

Notification JSON `notifications.json` (replace `<email>`):

```json
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
```

Create the budget:

```bash
aws budgets create-budget --account-id $ACCT \
  --budget file://budget.json \
  --notifications-with-subscribers file://notifications.json
```

## 6. CloudWatch log retention

```bash
aws logs put-retention-policy --region us-west-2 \
  --log-group-name /aws/sagemaker/TrainingJobs --retention-in-days 30
```

If the log group doesn't exist yet, this fails harmlessly. Re-run after the first training job creates the group.

## 7. Local config

**Recommended: project-scoped `.env` file** (gitignored). The launcher
auto-loads it via `python-dotenv` at startup. Keeps CU-VLA config out of
your global shell — useful when you have other AWS work using a different
default region or different credentials.

```bash
cp .env.example .env
# Edit .env with your values:
#   CU_VLA_SM_ROLE_ARN=arn:aws:iam::<acct>:role/SageMakerExecutionRole-CU-VLA
#   CU_VLA_SM_BUCKET=cu-vla-sm-<acct>
```

The launcher (`scripts/launch_sm_job.py`) loads `<repo-root>/.env` at
startup, but **shell env vars take precedence** (so CI overrides still
win). `.env` is in `.gitignore` already; never commit it.

**`AWS_REGION` is intentionally not in `.env`** — the launchers default to
`us-west-2` internally. If you need to run ad-hoc `aws ...` commands for
CU-VLA stuff, pass `--region us-west-2` explicitly (matches the rest of
this doc).

**Alternative: shell env vars** in `~/.zshrc` or `~/.bashrc` (older pattern,
pollutes other AWS work):

```bash
export CU_VLA_SM_ROLE_ARN="arn:aws:iam::<acct>:role/SageMakerExecutionRole-CU-VLA"
export CU_VLA_SM_BUCKET="cu-vla-sm-<acct>"
```

Verify the bootstrap works:

```bash
aws sagemaker list-training-jobs --region us-west-2 --max-results 1
```

Expected: empty list (`{"TrainingJobSummaries": []}`) on a fresh account, no error.

## 8. Tear-down (when leaving the project)

```bash
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
```
