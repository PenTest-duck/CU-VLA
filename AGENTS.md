# AGENTS.md

Research project: high-frequency VLA (Vision-Language-Action) models for computer use. Applying first-principles thinking, treats desktop interaction as a real-time visuomotor control problem at 30+ Hz using event-level primitives (mouse_down/up, key_down/up) rather than high-level actions (click, type). Targets consumer hardware (MacBook Pro M1, 8GB RAM).

## Persona

Be a thoughtful, innovative, expert researcher. Apply first-principles thinking.

Feel free to challenge/critique/push back on the user, e.g. if they have made false assumptions, sub-optimal suggestions, wrong decisions etc.. They are not perfect. But also you are also not perfect. Be cognizant of this.

Explain concepts so that it's easy to learn and understand.

When asking questions or brainstorming or making decisions, use AskUserQuestion tool and present multiple options and a recommendation.

Don't be lazy whenever you perform a task.

When you perform some major task, made some big decision, or wrote significant plan, get a comprehensive, critical review from a second AI model by running `agent --model=gpt-5.4-xhigh --output-format=text --mode=ask -p "<prompt>"`. Then look through the results and ask the user on flagged areas and whether/how to incorporate them. You must get explicit approval for every feedback item you wish to incorporate.

You are encouraged to perform web research to ground yourself in latest literature/information/documentation using tools e.g. WebFetch, WebSearch, Firecrawl, Exa, context7.

Do not run any training locally unless you have explicit approval - it is likely to cause OOM.

## Repo Layout

Serves as a high-level table of contents.

| Path                                                                               | Contents                                                                                                               |
| ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `docs/research/Brain Dump.md`                                                      | Original idea brainstorm — first-principles decomposition of computer input                                            |
| `docs/research/VLA Models Meet Computer Use.md`                                    | Literature survey: GUI-VLAs, high-freq control, world models, Tesla Digital Optimus, sub-1B models, PoC sketch         |
| `docs/research/High-Frequency Vision-Language-Action Models for Computer Use...md` | Formal feasibility report with references — action granularity analysis, layered architecture, toy PoC design, roadmap |
| `docs/experiments/1-reactive-clicks.md`                                            | Experiment 1 brief — click reaction time test                                                                          |
| `docs/plans/2026-04-11-reactive-clicks-design.md`                                  | Experiment 1 full design doc — architecture decisions, env, model, training, eval criteria                             |
| `experiments/reactive_clicks/`                                                     | Experiment 1 code (see below)                                                                                          |
| `docs/experiments/2-act.md`                                                        | Experiment 2 brief — ACT for drag-and-label                                                                            |
| `docs/plans/2026-04-13-act-drag-and-label-design.md`                               | Experiment 2 full design doc — ACT architecture, drag-and-label task, vision backbone + chunk size ablations           |
| `docs/plans/2026-04-13-act-drag-and-label-implementation.md`                       | Experiment 2 implementation plan — 12-task breakdown                                                                   |
| `experiments/act_drag_label/`                                                      | Experiment 2 code (see below)                                                                                          |
| `docs/experiments/3-miniwob-pygame.md`                                             | Experiment 3 design doc — MiniWoB-Pygame unified task suite                                                            |
| `docs/plans/2026-04-13-miniwob-pygame-implementation.md`                           | Experiment 3 implementation plan                                                                                       |
| `experiments/miniwob_pygame/`                                                      | Experiment 3 code (see below)                                                                                          |
| `docs/plans/2026-04-14-mini-editor-design.md`                                      | Experiment 5 full design doc — mini text editor task for V+L+A                                                         |
| `experiments/mini_editor/`                                                         | Experiment 5 code (see below)                                                                                          |
| `docs/experiments/6-action-primitives.md`                                          | Experiment 6 full design doc — event-level action primitives + Amendments section from Phase A                         |
| `docs/plans/2026-04-23-action-primitives-phase-a-implementation.md`                | Experiment 6 Phase A implementation plan — 25-task breakdown                                                           |
| `docs/experiments/6-action-primitives-phase-a-results/`                            | Experiment 6 Phase A spike results + PHASE-A-SUMMARY.md                                                                |
| `experiments/action_primitives/`                                                   | Experiment 6 code (see below)                                                                                          |
| `docs/research/hf-jobs-gotchas.md`                                                 | HF Jobs debugging reference — 5-iteration journey to first successful Phase A training run                             |
| `scripts/launch_hf_job.py`                                                         | Launcher for HF Jobs training (calls `run_uv_job()`) — exp2/3/5                                                        |
| `scripts/hf_job_train.py`                                                          | UV script that runs inside HF Jobs (clones repo, runs train.py) — exp2/3/5                                             |
| `scripts/launch_hf_job_exp6.py`                                                    | Launcher for exp6 HF Jobs training (forwards WANDB_API_KEY + HF_TOKEN)                                                 |
| `scripts/hf_job_train_exp6.py`                                                     | UV script for exp6 HF Jobs (clones `feat/exp6-phase-a` branch, runs training unbuffered)                               |
| `scripts/migrate_hdf5_to_parquet.py`                                               | One-shot migration from HDF5 episodes to parquet                                                                       |

## Experiment 1: Reactive Clicks

Validates the core visuomotor loop: CNN observes 128x128 frame, outputs discretized delta mouse + click at 30 Hz.

**Run sequence:**

```bash
uv run python experiments/reactive_clicks/generate_data.py   # generates 1000 expert episodes
uv run python experiments/reactive_clicks/train.py            # trains TinyCNN via behavior cloning
uv run python experiments/reactive_clicks/evaluate.py         # evaluates expert, baseline, and CNN
uv run python experiments/reactive_clicks/evaluate.py --visual  # with visible Pygame window
```

**Code layout:**
| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters (env, action, model, training, eval, expert) |
| `env.py` | `ReactiveClicksEnv` — Pygame-based Gym-style env (headless + visual modes) |
| `expert.py` | Fitts's Law scripted expert with online correction |
| `baseline.py` | Deterministic color-threshold + centroid beeline controller |
| `generate_data.py` | Runs expert, saves HDF5 episodes to `data/` |
| `model.py` | `TinyCNN` — 4-conv + FC + regression dx/dy + classification btn (~2.2M params, <2ms) |
| `train.py` | BC training loop: HDF5 loader, L1 loss (dx/dy) + CE (btn), AdamW + cosine LR |
| `evaluate.py` | Runs all agents, reports decomposed metrics (MAE, hit rate, RT, loop Hz) |

## Experiment 2: ACT Drag-and-Label

Validates ACT (Action Chunking with Transformers) for desktop interaction: drag colored shapes to matching zones, type 3-char labels. Tests whether action chunking provides genuine advantage over reactive single-step models.

**Run sequence:**

```bash
uv run python experiments/act_drag_label/generate_data.py -n 10000       # generate expert demos (parquet)
uv run python experiments/act_drag_label/train.py --backbone resnet18 --chunk-size 10 --device mps
uv run python experiments/act_drag_label/train.py --hf-data-repo PenTest-duck/cu-vla-data --device mps  # from HF Hub
uv run python experiments/act_drag_label/evaluate.py --backbone resnet18 --chunk-size 10
uv run python experiments/act_drag_label/evaluate.py --visual            # with Pygame window
uv run python experiments/act_drag_label/evaluate.py --model-only        # ACT only, skip expert/baseline/random
```

**Code layout:**
| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters (env, action, model, chunk, training, eval, expert) |
| `env.py` | `DragLabelEnv` — Pygame-based Gym-style env (headless + visual modes) |
| `expert.py` | Fitts's Law expert with drag + typing phases |
| `backbones.py` | ResNet18, DINOv2 ViT-S/14, SigLIP2 base — all output (B, 49, 256) |
| `model.py` | `ACT` — transformer encoder-decoder + FiLM proprio + 49-bin discrete dx/dy heads (~28M params) |
| `baseline_cnn.py` | `BaselineCNN` — extended TinyCNN, single-step no-chunking baseline |
| `generate_data.py` | Runs expert, saves parquet dataset via HF `datasets` library |
| `train.py` | BC training: chunk sampling, CE loss on bins + BCE/CE for click/key, AMP |
| `evaluate.py` | Runs all agents, probability-ensemble temporal smoothing, decomposed metrics |
| `hf_sync.py` | Upload/download data and checkpoints to/from HuggingFace Hub |

**Dataset format:** Parquet via HF `datasets` library (~10 shards). One row per timestep with columns: `episode_id`, `timestep`, `image` (PNG), `action_dx/dy/click/key`, episode metadata. Loaded via `load_dataset("PenTest-duck/cu-vla-data")` or `load_from_disk("data/")`.

**HF Jobs training** (dataset loaded via `load_dataset()` — no volume mounting needed):

```bash
uv run python scripts/launch_hf_job.py --flavor t4-medium --timeout 4h \
  -- --backbone resnet18 --chunk-size 10 --hf-upload-repo PenTest-duck/cu-vla-checkpoints
```

## Experiment 3: MiniWoB-Pygame

Unified multi-task Pygame environment suite with 12 computer use tasks. Tests whether a single ACT model can generalize across click, drag, type, scroll, and copy-paste primitives using a multi-binary held-state action space at 30Hz.

**Key design:** All inputs (mouse, keyboard) represented as binary held state — the physical ground truth of input devices. No high-level abstractions. The env detects transitions (press/release) from state changes. 43 independent key channels support simultaneous key combos (Ctrl+C, Shift+A).

**Run sequence:**

```bash
# Generate expert demos for Phase 1 (MVP)
uv run python experiments/miniwob_pygame/generate_data.py \
    --tasks click-target drag-to-zone use-slider type-field -n 5000

# Train multi-task ACT on Phase 1
uv run python experiments/miniwob_pygame/train.py \
    --backbone resnet18 --chunk-size 10 --device mps \
    --tasks click-target drag-to-zone use-slider type-field

# Evaluate
uv run python experiments/miniwob_pygame/evaluate.py \
    --backbone resnet18 --chunk-size 10 --device mps \
    --tasks click-target drag-to-zone use-slider type-field

# Visual evaluation
uv run python experiments/miniwob_pygame/evaluate.py --visual --tasks click-target

# Generate all 12 tasks
uv run python experiments/miniwob_pygame/generate_data.py --tasks all -n 5000
```

**Code layout:**
| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters, 43 key constants, action space definition |
| `base_env.py` | `BaseTaskEnv` — shared Pygame rendering, cursor, held-state edge detection |
| `widgets.py` | Reusable widgets: TextInput, Slider, ScrollableList, TextBlock |
| `task_registry.py` | Task name → env class + expert function mapping |
| `tasks/*.py` | 12 task environments (Phase 1-3) |
| `experts/common.py` | Fitts's Law trajectory, shared expert utilities |
| `experts/*.py` | 12 scripted expert policies |
| `model.py` | `ACT` — FiLM + transformer encoder-decoder, 49-bin discrete dx/dy + multi-binary key heads (~28M params) |
| `baseline_cnn.py` | `BaselineCNN` — 4-conv single-step baseline (~6.5M params) |
| `backbones.py` | ResNet18, DINOv2 ViT-S/14, SigLIP2 vision backbones |
| `generate_data.py` | Multi-task expert demonstration generation (parquet via HF datasets) |
| `train.py` | BC training: soft CE on bins + EV L1 + BCE for mouse/keys, warmup+cosine LR, grad clip, memmap image cache |
| `evaluate.py` | Per-task and aggregate metrics, probability-ensemble temporal smoothing, HF checkpoint download |
| `hf_sync.py` | Upload/download data and checkpoints to HuggingFace Hub |

**Action space (multi-binary held state):**

```python
action = {
    "dx": float,           # cursor delta, continuous
    "dy": float,           # cursor delta, continuous
    "mouse_left": int,     # 0=released, 1=pressed
    "keys_held": [43],     # binary per key: A-Z(0-25), space(26), enter(27),
                           # backspace(28), tab(29), 0-9(30-39), ctrl(40),
                           # shift(41), alt(42)
}
```

**12 Tasks (3 phases):**
| Phase | Tasks |
|-------|-------|
| 1: Core Primitives | click-target, drag-to-zone, use-slider, type-field |
| 2: Compositions | click-sequence, draw-path, highlight-text, drag-sort |
| 3: Multi-Primitive | form-fill, drag-and-label, scroll-and-click, copy-paste |

**Design doc:** `docs/experiments/3-miniwob-pygame.md`
**Implementation plan:** `docs/plans/2026-04-13-miniwob-pygame-implementation.md`

## Experiment 5: Mini Text Editor

First V+L+A task: a Pygame text editor where the model must execute natural language edit instructions using low-level mouse + keyboard primitives at 30 Hz. Tests vision (locate words), language (parse instruction), and action (multi-step motor sequences) simultaneously.

**Run sequence:**

```bash
uv run python -m experiments.mini_editor.generate_data -n 100 -o data/mini_editor_test  # small test
uv run python -m experiments.mini_editor.generate_data -n 10000 -o data/mini_editor       # full dataset
uv run python -m experiments.mini_editor.train --epochs 100 --device cuda --hf-data-repo PenTest-duck/cu-vla-mini-editor
uv run python -m experiments.mini_editor.evaluate --checkpoint path/to/best.pt
uv run python -m experiments.mini_editor.evaluate --visual  # with Pygame window
```

**Code layout:**
| File | Purpose |
|------|---------|
| `config.py` | 53-key Mac keyboard constants, shift mapping tables, env/expert/train configs |
| `corpus.py` | Load + filter `agentlans/high-quality-english-sentences`, word extraction, passage assembly |
| `instructions.py` | 4 edit operations (click, click+type, select+delete, replace), template phrasings |
| `env.py` | `MiniEditorEnv` — 640×480 Pygame editor, physical keyboard edge detection, 512×384 obs |
| `expert.py` | Fitts's Law + human variance (curvature, overshoot, typing rhythm), episode state machine |
| `generate_data.py` | Expert demo generation → HF Dataset (parquet), JPEG q=95 screenshots |
| `backbones.py` | ResNet18 backbone (384×288 input, 108 tokens, 2D sinusoidal PE) |
| `text_encoder.py` | Trainable 2L transformer text encoder, MobileBERT-init embeddings, ~4M params |
| `model.py` | `ACT` — V+L+A fusion, 108 vision + text + proprio tokens, 53 keys (~21M params) |
| `train.py` | BC training: focal BCE for keys, soft CE for bins, AMP, warmup+cosine LR |
| `evaluate.py` | Per-operation eval, temporal ensemble smoothing, held-out phrasing accuracy |
| `hf_sync.py` | Upload/download data and checkpoints to HuggingFace Hub |

**Action space (53-key physical Mac keyboard held-state):**

```python
action = {
    "dx": float,           # cursor delta, 49 discrete exponential bins
    "dy": float,           # cursor delta, 49 discrete exponential bins
    "mouse_left": int,     # 0=released, 1=held
    "keys_held": [53],     # binary per physical key:
                           #   0-25: A-Z, 26-35: 0-9, 36-46: symbol keys,
                           #   47: LShift, 48: RShift, 49: Space,
                           #   50: Delete, 51: Return, 52: Tab
}
```

**Observation space:**

```python
observation = {
    "screenshot": np.ndarray,  # (384, 512, 3) uint8 RGB
    "proprio": np.ndarray,     # (56,) float32: cursor_xy(2) + mouse_left(1) + keys_held(53)
    "instruction": str,        # NL instruction (NOT rendered on screen)
}
```

**4 operations:** click-to-position, click+type, select+delete (shift+click), replace

**Design doc:** `docs/plans/2026-04-14-mini-editor-design.md`

## Experiment 6: Action Primitives

Event-level primitives (mouse_down/up, key_down/up, scroll) instead of high-level actions. Phase A is a minimum-viable end-to-end slice (L-click only) to de-risk the full design via four feasibility spikes: A (typing legibility), B (L-click end-to-end), C (M1 eval timing), E (pygame gen throughput).

**Architecture (Phase A):**

SigLIP2-B-naflex @ `max_num_patches=256` vision (LoRA rank-8, trainable ~0.6M) + frozen text tower (instructions only) + 2-layer proprio MLP (83 → 768) + history encoder (223 → 768, 8-frame lookback) + 3-block cross+self attention trunk (16 learnable queries, d=768) + 6 unpooled factored heads (dx 21 + dy 21 + click 5 + scroll 21 + keys 231 + done 1). Total 408.8M params, 33.6M trainable.

**Run sequence (Phase A L-click):**

```bash
uv run python -m experiments.action_primitives.generate_data -n 3000 -o data/phase-a-lclick --shard-size 500
uv run python -m experiments.action_primitives.measurements.gen_throughput -n 1000  # Spike E
uv run python -m experiments.action_primitives.probes.typing_legibility_per_patch  # Spike A
uv run python -c "from huggingface_hub import create_repo; create_repo('PenTest-duck/cu-vla-exp6-phasea-lclick', repo_type='dataset', exist_ok=True); create_repo('PenTest-duck/cu-vla-exp6-phasea-ckpt', repo_type='model', exist_ok=True)"
uv run python -c "from experiments.action_primitives.hf_sync import upload_parquet_dir; from pathlib import Path; upload_parquet_dir(Path('data/phase-a-lclick'), 'PenTest-duck/cu-vla-exp6-phasea-lclick', 'dataset')"
export WANDB_API_KEY=<key>
uv run python scripts/launch_hf_job_exp6.py --flavor l4x1 --timeout 4h -- --hf-data-repo PenTest-duck/cu-vla-exp6-phasea-lclick --epochs 5 --hf-upload-repo PenTest-duck/cu-vla-exp6-phasea-ckpt --wandb-run-name phase-a-spike-b --micro-batch-episodes 4 --num-workers 4 --ckpt-every-steps 20 --eval-every-steps 20 --early-stop-patience 3
uv run python -m experiments.action_primitives.evaluate --checkpoint <ckpt> --data-dir data/phase-a-lclick --n-rollouts 200 --device mps                    # Spike B closed-loop
uv run python -m experiments.action_primitives.evaluate --checkpoint <ckpt> --data-dir data/phase-a-lclick --n-rollouts 20 --device mps --visual --skip-offline  # live pygame eval window
uv run python -m experiments.action_primitives.measurements.m1_eval_timing --checkpoint <ckpt> --n-rollouts 100  # Spike C
```

**Code layout:**
| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters (ENV, MODEL, LOSS, TRAIN, PHASE_A_DATA); bin-center math; 83-dim proprio + 223-dim history schema |
| `env.py` | `LClickEnv` — 720×450 pygame canvas with one colored button; headless or `visual=True` for live eval |
| `expert.py` | `LClickExpert` — Fitts-law trajectory + 4 tempo profiles + overshoot with press-on-target verification |
| `generator.py` | `generate_one_episode` — runs env + expert in lockstep, emits per-frame rows with env_done_frame drift detection |
| `generate_data.py` | Batched parquet writer with `--workers` + idempotency guard |
| `backbones.py` | `SigLIP2Naflex` with `preprocess` / `encode_preprocessed` / `encode_image` split; LoRA attached via `apply_lora` |
| `proprio.py` / `history.py` | 2-layer MLP encoders (83→768 and 223→768 respectively) |
| `trunk.py` | 3-block cross+self attention with 16 learnable queries |
| `heads.py` | Unpooled 6-head flatten(16 × 768)→logits |
| `model.py` | `ActionPrimitivesACT` — wires everything; `forward` is polymorphic over PIL list or preprocessed dict |
| `dataset.py` | `PhaseAEpisodeDataset` — parquet loader + on-the-fly quantization + history-vector construction + optional SigLIP2 preprocessing in worker |
| `losses.py` | Focal CE + idle-biased smoothing on keys + focal BCE for done + `total_loss` aggregator |
| `train.py` | AdamW two-param-group + cosine warmup + grad accum + DataLoader prefetch + val eval + best.pt tracking + early stopping |
| `evaluate.py` | Offline per-head accuracy + closed-loop rollouts with tolerance curves; `--visual` live mode, tqdm progress |
| `hf_sync.py` | HF Hub upload/download helpers |
| `probes/typing_legibility.py` | Spike A string-presence probe (mean + attention pool) |
| `probes/typing_legibility_per_patch.py` | Spike A per-patch char-identity probe (62-way, no space) |
| `measurements/gen_throughput.py` | Spike E |
| `measurements/m1_eval_timing.py` | Spike C |

**Action space (Phase A):**

```python
action = {
    "dx": float,            # 21-bin exponential, ±100 px/frame
    "dy": float,            # 21-bin exponential, ±100 px/frame
    "click": int,           # 5-way {idle, L_press, L_release, R_press, R_release}
    "scroll": float,        # 21-bin exponential, ±20 ticks/frame (always idle in L-click)
    "key_events": [77],     # 3-way per key {press=0, release=1, idle=2}
    "done": int,            # binary (training-only)
}
```

**Design doc:** `docs/experiments/6-action-primitives.md` (see Amendments section for Phase A findings)
**Phase A plan:** `docs/plans/2026-04-23-action-primitives-phase-a-implementation.md`
**Phase A results:** `docs/experiments/6-action-primitives-phase-a-results/`

## Experiment Results

Experiment 1: simple CNN got pretty perfect results
Experiment 2: ACT, 10 chunks, ResNet18 got 94% eval. No ablations tested.
Experiment 3: skipped
Experiment 4: skipped
Experiment 5: in progress
Experiment 6 Phase A: Spike A ✓ (typing legibility per-patch 74% @ 14pt), Spike E ✓ (3.6 eps/s single-proc; 200 eps/s target revised), Spike B in progress, Spike C pending

## Key Technical Decisions (from research)

Nothing is fully locked in. Open to changes, deletions, challenges/critiques, alternatives and improvements.

- **Action space:** event-level primitives (mouse_down/up, key_down/up, scroll tick), not semantic actions
- **Architecture:** dual-system — tiny fast "System 1" policy at high Hz + larger slow "System 2" VLM planner
- **Target models:** SmolVLA (450M), SmolVLM-256M, Moondream 0.5B, Florence-2-base — all fit in ≤1GB
- **Runtime:** MLX preferred (50% faster than Ollama on Apple Silicon); llama.cpp as cross-platform fallback
- **Screen capture:** MSS library (47–61 FPS on M1) — not PyAutoGUI (2.8 FPS)
- **Input control:** Quartz CoreGraphics for sub-ms mouse/key events on macOS
- **PoC plan:** reaction-time test (Pygame, red circle → click), escalating from tiny CNN to full VLA

Keep this file updated as the project evolves.
