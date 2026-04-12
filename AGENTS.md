# AGENTS.md

Research project: high-frequency VLA (Vision-Language-Action) models for computer use. Applying first-principles thinking, treats desktop interaction as a real-time visuomotor control problem at 30+ Hz using event-level primitives (mouse_down/up, key_down/up) rather than high-level actions (click, type). Targets consumer hardware (MacBook Pro M1, 8GB RAM).

## Persona

Be a thoughtful, innovative researcher. Apply first-principles thinking.

Feel free to challenge/critique/push back on the user, e.g. if they have made false assumptions, sub-optimal suggestions, wrong decisions etc.. They are not perfect. But also you are also not perfect. Be cognizant of this.

Explain concepts so that it's easy to learn and understand.

When asking questions or brainstorming or making decisions, use AskUserQuestion tool and present multiple options and a recommendation.

Don't be lazy whenever you perform a task.

When you perform some major task, made some big decision, or wrote significant plan, get a comprehensive, critical review from a second AI model by running `agent --model=auto --output-format=text --mode=ask -p "<prompt>"`. Then look through the results and ask the user on flagged areas and whether/how to incorporate them.

You are encouraged to perform web research to ground yourself in latest literature/information/documentation using tools e.g. WebFetch, WebSearch, Firecrawl, Exa, context7.

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
uv run python experiments/act_drag_label/generate_data.py -n 10000       # generate expert demos
uv run python experiments/act_drag_label/train.py --backbone resnet18 --chunk-size 10 --device mps
uv run python experiments/act_drag_label/evaluate.py --backbone resnet18 --chunk-size 10
uv run python experiments/act_drag_label/evaluate.py --visual            # with Pygame window
```

**Code layout:**
| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters (env, action, model, chunk, training, eval, expert) |
| `env.py` | `DragLabelEnv` — Pygame-based Gym-style env (headless + visual modes) |
| `expert.py` | Fitts's Law expert with drag + typing phases |
| `backbones.py` | ResNet18, DINOv2 ViT-S/14, SigLIP2 base — all output (B, 49, 256) |
| `model.py` | `ACT` — CVAE + transformer encoder-decoder + action heads (~33M params) |
| `baseline_cnn.py` | `BaselineCNN` — extended TinyCNN, single-step no-chunking baseline |
| `generate_data.py` | Runs expert, saves HDF5 episodes to `data/` |
| `train.py` | BC training: chunk sampling, multi-head loss, KL annealing, AMP |
| `evaluate.py` | Runs all agents, temporal ensemble, decomposed metrics by phase |

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
