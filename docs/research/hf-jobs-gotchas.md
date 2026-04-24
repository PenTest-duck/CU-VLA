# HF Jobs gotchas — reference for exp7+

Everything we hit while getting Experiment 6 Phase A training to run on HF Jobs L4. Five separate failed iterations before the first successful run. Save yourself the journey.

**Source runs (all on `PenTest-duck/cu-vla-exp6-phasea-ckpt` repo):**

| Job ID | Failure | Root cause | Fix commit |
|---|---|---|---|
| `69eb0ac6e8e12c6f0a67577e` | `No module named pip` | `pip install -e .` in UV env that has no pip | `36fc5d8` |
| `69eb0cece8e12c6f0a675780` | `train.py: error: --data-dir required` | argparse required=True blocked `--hf-data-repo` standalone | `7f60ff1` |
| `69eb0e616bbd7bff45bfecca` | `Siglip2VisionModel.forward() missing pixel_attention_mask` | transformers renamed kwarg; hardcoded `attention_mask` | `2cfff6a` |
| `69eb10366bbd7bff45bfecdb` | `CUDA OOM — 21.7 GiB in use on 22 GiB L4` | `micro_batch_episodes=8` → 360 images/fwd → 43 GB activations | `ad2f8a2` |
| `69eb1a1fe8e12c6f0a6757a1` | **Success** (with `--micro-batch-episodes 4 --num-workers 4`) | — | — |

## Pitfall 1: UV-managed envs have no `pip`

**Symptom:**
```
/root/.cache/uv/environments-v2/<hash>/bin/python: No module named pip
subprocess.CalledProcessError: '[...] pip install -e .' returned non-zero exit status 1.
```

**Root cause:** HF Jobs runs `uv run` on your UV script. UV creates a minimal env with only the dependencies listed in the script header — no `pip` is installed. Any `python -m pip` call inside fails.

**Fix:** Don't install the project as a package. Clone + `sys.path.insert(0, workdir)` + run the training module directly.

```python
# Inside hf_job_train_<exp>.py
subprocess.run(["git", "clone", "--depth", "1", repo_url, workdir], check=True)
os.chdir(workdir)
sys.path.insert(0, workdir)
result = subprocess.run([sys.executable, "-u", "-m", "experiments.your_exp.train", *sys.argv[1:]], cwd=workdir)
sys.exit(result.returncode)
```

All project dependencies must be declared in the UV script header (not just transitively required by the project). See `scripts/hf_job_train_exp6.py` for the exp6 version.

## Pitfall 2: HF Jobs clones the default branch

**Symptom:** `ModuleNotFoundError: No module named 'experiments.action_primitives'` after successful clone.

**Root cause:** `git clone https://github.com/user/repo.git` gets the default branch (`main` in our case). Feature branches don't exist there. Our exp6 code was only on `feat/exp6-phase-a`.

**Fix:** Either merge to main (risky if feature not yet shipped) OR parameterize the branch in the job script:

```python
branch = os.environ.get("CU_VLA_BRANCH", "feat/<exp>-phase-<n>")
subprocess.run(
    ["git", "clone", "--depth", "1", "--branch", branch, repo_url, workdir],
    check=True,
)
```

Default to your feature branch; override via env var at launch time if needed.

## Pitfall 3: `argparse required=True` doesn't compose with mutually-exclusive data sources

**Symptom:**
```
train.py: error: the following arguments are required: --data-dir
```
...when launching with `--hf-data-repo` but no `--data-dir`.

**Root cause:** `parser.add_argument("--data-dir", required=True)` blocks any invocation that omits `--data-dir`, even when `--hf-data-repo` provides the data a different way.

**Fix:** Make both optional, add a runtime mutex check with a clear error:
```python
parser.add_argument("--data-dir", default=None)
parser.add_argument("--hf-data-repo", default=None)
...
if (args.data_dir is None) == (args.hf_data_repo is None):
    parser.error("exactly one of --data-dir or --hf-data-repo must be provided")
```

Don't use `argparse.MutuallyExclusiveGroup` with `required=True` — it doesn't enforce "exactly one non-None" when defaults exist.

## Pitfall 4: transformers library API churn between pinned versions

**Symptom:**
```
TypeError: Siglip2VisionModel.forward() missing 1 required positional argument: 'pixel_attention_mask'
```
...when local tests pass with `attention_mask` as the kwarg name.

**Root cause:** transformers 5.5.x and 5.6.x renamed the SigLIP2 vision tower's mask kwarg. Local env had one; HF Jobs pulls latest and had the other. Version drift between your local env and whatever PyPI serves to HF Jobs is a silent correctness risk.

**Fix:** Detect the right kwarg name at init time via signature inspection:

```python
import inspect
params = inspect.signature(self.model.vision_model.forward).parameters
if "pixel_attention_mask" in params:
    self._mask_kwarg = "pixel_attention_mask"
elif "attention_mask" in params:
    self._mask_kwarg = "attention_mask"
else:
    raise RuntimeError(f"neither found: {list(params)}")

# Then at call time:
self.model.vision_model(pixel_values=..., **{self._mask_kwarg: mask_tensor}, ...)
```

General principle: if your library has API churn across minor versions, and your local and HF Jobs envs can disagree, prefer runtime introspection over hardcoded call sites.

## Pitfall 5: Default micro-batch sizes OOM on L4

**Symptom:** Training starts normally (model loads, wandb logs in, step 0 begins forward), then crashes mid-forward:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 540.00 MiB.
GPU 0 has a total capacity of 22.03 GiB of which 325.12 MiB is free.
```

**Root cause:** The design doc's `micro_batch_episodes=8` was set without measuring activations. On our model (SigLIP2-B 12-layer vision tower), each episode has 45 frames × 256 patches × 768 dims per layer. At 8 episodes/forward = 360 images × 12 layers × 10 MB/layer = ~43 GB activations. L4 has 22 GB usable.

**Fix:** CLI override to shrink micro batch independent of macro batch:
```python
parser.add_argument("--micro-batch-episodes", type=int, default=TRAIN.micro_batch_episodes)
parser.add_argument("--macro-batch-episodes", type=int, default=TRAIN.macro_batch_episodes)
if args.macro_batch_episodes % args.micro_batch_episodes != 0:
    parser.error("macro must be multiple of micro")
```

On L4 24 GB: `--micro-batch-episodes 4` fits (~14.5 GB) with bf16 autocast. Same total gradient signal per optimizer step (unchanged macro=64), just 16 passes instead of 8. Step time approximately unchanged — fewer big passes vs more small passes trade evenly for GPU-bound work.

**General rule:** for any new model, measure peak GPU memory at default batch size on a test run before scheduling a long training job. A 2-minute smoke run saves a 4-hour OOM wall clock.

## Pitfall 6: Naive `multiprocessing.Pool.imap` is slower than serial

**Symptom:** You add `multiprocessing.Pool(workers=4)` expecting ~4× speedup. Instead the multi-worker run is *slower* than serial.

**Root cause:** Two compounding:
1. **macOS `spawn` start method + pygame cold-start per worker.** SDL init is not free; `pygame-ce 2.5.7 (SDL 2.32.10, ...)` prints once per spawned worker.
2. **IPC payload cost dominates.** `Pool.imap(...)` at default `chunksize=1` sends each task's return value back through the pipe one at a time. For our episodes (~14 MB of JPEG bytes per episode), the parent process is IPC-bound before workers can saturate CPU.

**Fix options (in increasing effort):**
- Try `chunksize=8` first — amortizes IPC across 8 results.
- Worker-writes-shards redesign: each worker owns a contiguous episode range and writes its own parquet shard directly. Parent only receives shard paths (a few bytes). Removes the IPC payload entirely. ~4-8× real speedup.
- For really large data: go distributed (multiple HF Jobs / SageMaker instances, each writing its shard range).

See Spike E write-up for our exp6 measurements.

## Minor quality-of-life fixes

### `hf jobs logs` shows output in end-of-run batches, not live

**Fix:** `-u` unbuffered mode:
```python
subprocess.run([sys.executable, "-u", "-m", "...", ...])
```

### HF hub + datasets progress bars flood the log

Hundreds of `Filter: 37%|... ` / `Fetching 7 files: 40%|... ` lines make it hard to scroll back to actual training output.

**Fix:** Set env vars in the subprocess env dict:
```python
env = os.environ.copy()
env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
env["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
subprocess.run([...], env=env)
```

Don't set them globally on your local machine — you want the bars when working locally. Only silence in the remote subprocess.

### `WANDB_API_KEY` isn't magically available inside HF Jobs

**Fix:** Forward from local shell into HF Jobs secrets:
```python
# In launcher
secrets = {}
if "WANDB_API_KEY" in os.environ:
    secrets["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
job = run_uv_job(..., secrets=secrets)
```

User must `export WANDB_API_KEY=<key>` locally before launching. Forward `HF_TOKEN` the same way.

## Sane defaults checklist for exp7+ HF Jobs launcher

Before launching any new experiment on HF Jobs, verify:

- [ ] Job script uses `sys.path.insert`, NOT `pip install -e .`
- [ ] Job script clones an explicit branch (your feature branch, not `main`)
- [ ] Training CLI accepts `--hf-data-repo` as a standalone (no `--data-dir` required)
- [ ] Vision-model kwargs are detected at runtime, not hardcoded
- [ ] `--micro-batch-episodes N` override exists; default smoke-tested for OOM on target GPU
- [ ] `-u` unbuffered Python for live log streaming
- [ ] `HF_HUB_DISABLE_PROGRESS_BARS=1` + `HF_DATASETS_DISABLE_PROGRESS_BARS=1` in subprocess env
- [ ] `WANDB_API_KEY` forwarded from local shell to HF Jobs secrets (or `--wandb-mode disabled` for auth-free runs)
- [ ] A 1-minute smoke run with `--n-episodes 2 --epochs 0` or equivalent before the real job

See `scripts/hf_job_train_exp6.py` and `scripts/launch_hf_job_exp6.py` for the exp6 reference implementation of all the above.
