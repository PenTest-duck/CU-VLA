# AGENTS.md

Research project: high-frequency VLA (Vision-Language-Action) models for computer use. Applying first-principles thinking, treats desktop interaction as a real-time visuomotor control problem at 30+ Hz using event-level primitives (mouse_down/up, key_down/up) rather than high-level actions (click, type). Targets consumer hardware (MacBook Pro M1, 8GB RAM).

## Persona

Be a thoughtful, innovative, expert researcher. Apply first-principles thinking.

Feel free to challenge/critique/push back on the user, e.g. if they have made false assumptions, sub-optimal suggestions, wrong decisions etc.. They are not perfect. But also you are also not perfect. Be cognizant of this.

Explain concepts so that it's easy to learn and understand.

When asking questions or brainstorming or making decisions, use AskUserQuestion tool and present multiple options and a recommendation.

Don't be lazy whenever you perform a task.

When you perform some major task, made some big decision, or wrote significant plan, get a comprehensive, critical review from a second AI model by running `agent --model=auto --output-format=text --mode=ask -p "<prompt>"`. Then look through the results and ask the user on flagged areas and whether/how to incorporate them. You must get explicit approval for every feedback item you wish to incorporate.

You are encouraged to perform web research to ground yourself in latest literature/information/documentation using tools e.g. WebFetch, WebSearch, Firecrawl, Exa, context7.

Do not run any training locally unless you have explicit approval - it is likely to cause OOM.

## Repo Layout

Serves as a high-level table of contents.

| Path | Contents |
|------|----------|
| `docs/research/Brain Dump.md` | Original idea brainstorm — first-principles decomposition of computer input |
| `docs/research/VLA Models Meet Computer Use.md` | Literature survey: GUI-VLAs, high-freq control, world models, Tesla Digital Optimus, sub-1B models, PoC sketch |
| `docs/research/High-Frequency Vision-Language-Action Models for Computer Use...md` | Formal feasibility report with references — action granularity analysis, layered architecture, toy PoC design, roadmap |
| `docs/experiments/1-reactive-clicks.md` | Experiment 1 brief — click reaction time test |
| `docs/plans/2026-04-11-reactive-clicks-design.md` | Experiment 1 full design doc — architecture decisions, env, model, training, eval criteria |
| `experiments/reactive_clicks/` | Experiment 1 code (see below) |
| `docs/experiments/2-act.md` | Experiment 2 brief — ACT for drag-and-label |
| `docs/plans/2026-04-13-act-drag-and-label-design.md` | Experiment 2 full design doc — ACT architecture, drag-and-label task, vision backbone + chunk size ablations |
| `docs/plans/2026-04-13-act-drag-and-label-implementation.md` | Experiment 2 implementation plan — 12-task breakdown |
| `experiments/act_drag_label/` | Experiment 2 code (see below) |
| `docs/experiments/3-miniwob-pygame.md` | Experiment 3 design doc — MiniWoB-Pygame unified task suite |
| `docs/plans/2026-04-13-miniwob-pygame-implementation.md` | Experiment 3 implementation plan |
| `experiments/miniwob_pygame/` | Experiment 3 code (see below) |
| `scripts/launch_hf_job.py` | Launcher for HF Jobs training (calls `run_uv_job()`) |
| `scripts/hf_job_train.py` | UV script that runs inside HF Jobs (clones repo, runs train.py) |
| `scripts/migrate_hdf5_to_parquet.py` | One-shot migration from HDF5 episodes to parquet |

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
| `model.py` | `ACT` — CVAE + transformer, multi-binary action heads (~33M params) |
| `baseline_cnn.py` | `BaselineCNN` — 4-conv single-step baseline (~6.5M params) |
| `backbones.py` | ResNet18, DINOv2 ViT-S/14, SigLIP2 vision backbones |
| `generate_data.py` | Multi-task HDF5 expert demonstration generation |
| `train.py` | Behavior cloning: chunk sampling, multi-head BCE loss, KL annealing |
| `evaluate.py` | Per-task and aggregate metrics, temporal ensemble inference |
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
