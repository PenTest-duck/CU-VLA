# Experiment 6 — Phase A Feasibility Spikes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run four cheap feasibility spikes that de-risk load-bearing Phase B assumptions (typing legibility at naflex@256, single-primitive end-to-end pipeline, M1 closed-loop eval timing, pygame generation throughput), producing a short write-up per spike that gates the Phase B implementation plan.

**Architecture:** Minimal-viable slice of the full v1 architecture (SigLIP2 naflex + LoRA vision / frozen text, 16-query cross/self-attention trunk, 6 factored action heads, proprio + 8-frame action history tokens) trained via behavior cloning on L-click-only synthetic data (~3K episodes). Closed-loop eval in the same pygame env. A standalone probe script answers the naflex legibility question without touching training code.

**Tech Stack:** Python 3.11, PyTorch 2.x, HuggingFace `transformers` (SigLIP2 naflex), `peft` (LoRA), `datasets` (parquet), `pygame`, `wandb`, HF Jobs (L4 / L40S) for training, M1 MacBook (PyTorch MPS) for eval timing.

**Out of scope for Phase A:** all primitives other than L-click; 20-theme generator; adversarial scenes; combinatorial instruction system; full 6 OOD slices; EMA; full 5 inference-time probes (basic eval only); SageMaker fallback (HF Jobs only); MLX / CoreML / INT8.

**Project convention:** save new experiment code under `experiments/action_primitives/`, tests under `tests/action_primitives/`, spike write-ups under `docs/experiments/6-action-primitives-phase-a-results/`.

---

## Sub-Phase 1: Scaffolding

### Task 1: Create experiment directory structure + dependencies

**Files:**
- Create: `experiments/action_primitives/__init__.py`
- Create: `experiments/action_primitives/README.md`
- Create: `tests/action_primitives/__init__.py`
- Create: `docs/experiments/6-action-primitives-phase-a-results/.gitkeep`
- Modify: `pyproject.toml` (add deps if missing)

- [ ] **Step 1: Verify pyproject.toml has required deps**

Run: `grep -E 'torch|transformers|peft|datasets|pygame|wandb' /Users/pentest-duck/Desktop/CU-VLA/pyproject.toml`

Expected: `torch`, `transformers`, `datasets`, `pygame`, `wandb` present. If `peft` missing, add it to deps.

- [ ] **Step 2: Create empty `__init__.py` for experiment package**

Create `experiments/action_primitives/__init__.py`:

```python
"""Experiment 6: Action Primitives.

Minimal Phase A implementation (L-click only) for feasibility spikes.
See docs/experiments/6-action-primitives.md for full design,
docs/plans/2026-04-23-action-primitives-phase-a-implementation.md for this plan.
"""
```

- [ ] **Step 3: Create tests package init**

Create `tests/action_primitives/__init__.py` with a single blank line.

- [ ] **Step 4: Create experiment README**

Create `experiments/action_primitives/README.md`:

```markdown
# Experiment 6: Action Primitives (Phase A)

Phase A = four feasibility spikes. See `docs/experiments/6-action-primitives.md`.

## Spike scripts
- `probes/typing_legibility.py` — Spike A (standalone, no training)
- `measurements/gen_throughput.py` — Spike E (measures generator eps/sec)
- `measurements/m1_eval_timing.py` — Spike C (uses Spike B checkpoint)

## Spike B flow
1. `python generate_data.py --primitive lclick -n 3000 -o data/phase-a-lclick/`
2. `python scripts/launch_hf_job_exp6.py -- --data-dir data/phase-a-lclick --epochs 5 --hf-upload-repo PenTest-duck/cu-vla-exp6-phasea-ckpt`
3. `python evaluate.py --checkpoint <ckpt> --primitive lclick --n 200 --visual` (on M1)

See the Phase A plan for full step-by-step.
```

- [ ] **Step 5: Create results directory placeholder**

Create `docs/experiments/6-action-primitives-phase-a-results/.gitkeep` (empty file).

- [ ] **Step 6: Commit**

```bash
git add experiments/action_primitives/ tests/action_primitives/ docs/experiments/6-action-primitives-phase-a-results/
git commit -m "scaffold(exp6): phase-a directory structure + README"
```

---

### Task 2: Create shared config module

**Files:**
- Create: `experiments/action_primitives/config.py`

Captures all hyperparameters/constants referenced by env, expert, generator, model, training, eval. Single source of truth. Matches the design doc's Q1–Q35 numbers.

- [ ] **Step 1: Write config.py with env, action space, and model constants**

Create `experiments/action_primitives/config.py`:

```python
"""Shared config for Experiment 6 Phase A.

Matches design decisions from docs/experiments/6-action-primitives.md.
Only L-click-related values populated; remaining primitives added in Phase B.
"""

from dataclasses import dataclass, field
import math

import numpy as np


# ---------- Environment ----------
@dataclass(frozen=True)
class EnvConfig:
    canvas_w: int = 720
    canvas_h: int = 450  # 16:10 full-screen task window (Q7)
    fps: int = 30        # logical 30Hz (Q5, Q16)
    max_frames_lclick: int = 30  # per-primitive window (Q8)


ENV = EnvConfig()


# ---------- Action space (Q1, Q3) ----------
# Mouse delta: 21 bins per axis, ±100px (widened 2026-04-23), 10+1+10 exponential
# Zero is bin 10; bins 0-9 are negative (large → small magnitude); bins 11-20 positive
NUM_BINS_MOUSE: int = 21
MOUSE_CAP_PX: float = 100.0  # Q1 amendment 2026-04-23
MOUSE_ALPHA: float = 2.5     # exponential growth factor (Q1)

def _build_exp_bin_centers(num_bins: int, cap: float, alpha: float) -> np.ndarray:
    """Symmetric exponential bin centers, zero at middle. Returns length-num_bins array."""
    half = num_bins // 2  # 10 for num_bins=21
    # geometric series i=1..half, scaled so largest = cap
    raw = alpha ** np.arange(1, half + 1)  # e.g. 2.5, 6.25, ..., 2.5^10
    scaled_pos = raw / raw[-1] * cap        # scale so last = cap
    centers = np.concatenate([-scaled_pos[::-1], np.array([0.0]), scaled_pos])
    return centers.astype(np.float32)

MOUSE_BIN_CENTERS: np.ndarray = _build_exp_bin_centers(NUM_BINS_MOUSE, MOUSE_CAP_PX, MOUSE_ALPHA)

# Click event: 5-way {idle, L_press, L_release, R_press, R_release}
NUM_CLICK_EVENTS: int = 5
CLICK_IDLE, CLICK_L_PRESS, CLICK_L_RELEASE, CLICK_R_PRESS, CLICK_R_RELEASE = range(5)

# Scroll: 21-bin signed, ±20 wheel ticks/frame (symmetric; Q1 notes unchanged in Phase A)
NUM_BINS_SCROLL: int = 21
SCROLL_CAP_TICKS: float = 20.0
SCROLL_BIN_CENTERS: np.ndarray = _build_exp_bin_centers(NUM_BINS_SCROLL, SCROLL_CAP_TICKS, MOUSE_ALPHA)

# Keys: 77 physical keys × 3-way {press, release, idle}
NUM_KEYS: int = 77
KEY_STATE_PRESS, KEY_STATE_RELEASE, KEY_STATE_IDLE = range(3)

# Head sizes
HEAD_LOGITS = {
    "dx": NUM_BINS_MOUSE,        # 21
    "dy": NUM_BINS_MOUSE,        # 21
    "click": NUM_CLICK_EVENTS,   # 5
    "scroll": NUM_BINS_SCROLL,   # 21
    "keys": NUM_KEYS * 3,        # 231
    "done": 1,                   # binary
}
TOTAL_LOGITS: int = sum(HEAD_LOGITS.values())  # 300


# ---------- Proprio (Q15) ----------
# Cursor xy (2) + held keys (77) + held mouse buttons (3) + CapsLock mode (1) = 83
PROPRIO_DIM: int = 83


# ---------- Model (Q15, Q16) ----------
@dataclass(frozen=True)
class ModelConfig:
    vision_model: str = "google/siglip2-base-patch16-naflex"
    max_num_patches: int = 256     # Q6, Q29
    d_model: int = 768             # SigLIP2-B output dim; trunk dim
    n_queries: int = 16            # Q15: learnable "read heads"
    n_blocks: int = 3              # Q15 Trunk A
    n_heads: int = 12
    ffn_mult: int = 4
    action_history_len: int = 8    # Q5, Q19
    action_vec_dim: int = 300      # Q19: per-frame composite action vector size
    lora_rank: int = 8             # Q15
    freeze_text_tower: bool = True # Q15


MODEL = ModelConfig()


# ---------- Loss (Q2, Q3) ----------
@dataclass(frozen=True)
class LossConfig:
    focal_gamma: float = 2.0
    label_smoothing_mouse: float = 0.05
    idle_smoothing_keys: float = 0.05
    # Initial per-head weights will be re-computed from L_i^init at train start (Q2)
    placeholder_weights: dict = field(default_factory=lambda: {
        "dx": 1.0, "dy": 1.0, "click": 1.0, "scroll": 1.0, "keys": 1.0, "done": 1.0,
    })


LOSS = LossConfig()


# ---------- Training (Q20, Q21) ----------
@dataclass(frozen=True)
class TrainConfig:
    macro_batch_episodes: int = 64
    micro_batch_episodes: int = 8        # 8 micro × 8 = 64 macro (Q2/Q8 clarification)
    phase_a_epochs: int = 5              # smaller than Phase B's 20
    lr_trunk: float = 3e-4
    lr_lora: float = 2e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 100              # smaller than Phase B's 500 for smaller run
    cosine_min_lr_frac: float = 0.1
    grad_clip_norm: float = 1.0
    eval_every_steps: int = 200
    ckpt_every_steps: int = 200


TRAIN = TrainConfig()


# ---------- Phase A data (L-click only) ----------
@dataclass(frozen=True)
class PhaseADataConfig:
    primitive: str = "lclick"
    n_episodes: int = 3000
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    # Minimal visual diversity: 3 themes only (full 20-theme system is Phase B)
    themes: tuple = ("flat-modern", "flat-minimal", "dark-mode")


PHASE_A_DATA = PhaseADataConfig()
```

- [ ] **Step 2: Write a smoke test for bin center math**

Create `tests/action_primitives/test_config.py`:

```python
"""Basic sanity checks on config constants."""

import numpy as np
import pytest

from experiments.action_primitives.config import (
    MOUSE_BIN_CENTERS,
    MOUSE_CAP_PX,
    NUM_BINS_MOUSE,
    SCROLL_BIN_CENTERS,
    NUM_BINS_SCROLL,
    TOTAL_LOGITS,
    HEAD_LOGITS,
)


def test_mouse_bin_centers_shape_and_symmetry():
    assert MOUSE_BIN_CENTERS.shape == (NUM_BINS_MOUSE,)
    # Symmetric around zero
    assert MOUSE_BIN_CENTERS[NUM_BINS_MOUSE // 2] == 0.0
    np.testing.assert_allclose(
        MOUSE_BIN_CENTERS[: NUM_BINS_MOUSE // 2],
        -MOUSE_BIN_CENTERS[NUM_BINS_MOUSE // 2 + 1 :][::-1],
        atol=1e-5,
    )


def test_mouse_bin_centers_cap():
    assert MOUSE_BIN_CENTERS[-1] == pytest.approx(MOUSE_CAP_PX)
    assert MOUSE_BIN_CENTERS[0] == pytest.approx(-MOUSE_CAP_PX)


def test_mouse_bin_centers_monotonic():
    assert np.all(np.diff(MOUSE_BIN_CENTERS) > 0)


def test_scroll_bin_centers_shape():
    assert SCROLL_BIN_CENTERS.shape == (NUM_BINS_SCROLL,)


def test_total_logits_matches_design():
    assert TOTAL_LOGITS == 300  # 21+21+5+21+231+1 per design Q1
    assert HEAD_LOGITS["keys"] == 231
```

- [ ] **Step 3: Run tests; expect PASS**

Run: `uv run pytest tests/action_primitives/test_config.py -v`

Expected: 4 tests pass.

- [ ] **Step 4: Commit**

```bash
git add experiments/action_primitives/config.py tests/action_primitives/test_config.py
git commit -m "feat(exp6): shared config + bin-center math with tests"
```

---

## Sub-Phase 2: Spike A — Typing legibility probe (standalone)

Spike A is independent of training code — only needs the SigLIP2 naflex encoder and a pygame text renderer. Run this **first** since it's the cheapest and informs whether we need to revisit Q5/Q6.

### Task 3: SigLIP2 naflex backbone loader

**Files:**
- Create: `experiments/action_primitives/backbones.py`
- Create: `tests/action_primitives/test_backbones.py`

- [ ] **Step 1: Write backbones.py skeleton with naflex loader**

Create `experiments/action_primitives/backbones.py`:

```python
"""SigLIP2 naflex vision + text tower loader for Experiment 6.

Variable-patch vision encoder (max_num_patches=256 default from Q6, Q29).
Text tower is frozen and used for instruction caching per Q15.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor

from experiments.action_primitives.config import MODEL


@dataclass
class NaflexOutput:
    """Output of a naflex vision forward pass."""

    patch_embeds: torch.Tensor       # (B, N_patches, d_model)
    attention_mask: torch.Tensor     # (B, N_patches), 1 = real, 0 = pad
    spatial_shapes: torch.Tensor     # (B, 2), (h_patches, w_patches) per sample


class SigLIP2Naflex(nn.Module):
    """Frozen SigLIP2-B-naflex vision + text towers (LoRA added later via peft)."""

    def __init__(self, max_num_patches: int = MODEL.max_num_patches) -> None:
        super().__init__()
        self.max_num_patches = max_num_patches
        self.processor = AutoProcessor.from_pretrained(MODEL.vision_model)
        self.model = AutoModel.from_pretrained(MODEL.vision_model)
        # Freeze text tower per Q15
        for p in self.model.text_model.parameters():
            p.requires_grad = False

    def encode_image(self, images: list) -> NaflexOutput:
        """Run vision tower on a list of PIL Images.

        Returns patch_embeds (B, N, d), attention_mask (B, N), spatial_shapes (B, 2).
        """
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            max_num_patches=self.max_num_patches,
        )
        # Move to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = self.model.vision_model(**inputs)
        # last_hidden_state: (B, N_patches, hidden); pixel_attention_mask: (B, N_patches)
        return NaflexOutput(
            patch_embeds=out.last_hidden_state,
            attention_mask=inputs["pixel_attention_mask"],
            spatial_shapes=inputs["spatial_shapes"],
        )

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Cache-friendly instruction encoder. Returns (B, T, d) text tokens."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = self.model.text_model(**inputs)
        return out.last_hidden_state  # (B, T, d)
```

- [ ] **Step 2: Write backbone smoke test**

Create `tests/action_primitives/test_backbones.py`:

```python
"""Smoke tests for SigLIP2 naflex loader. Requires model download; marked slow."""

import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.slow  # run with `pytest -m slow`


def test_naflex_loads_and_forwards_image():
    from experiments.action_primitives.backbones import SigLIP2Naflex

    m = SigLIP2Naflex(max_num_patches=64)  # small for test speed
    img = Image.new("RGB", (720, 450), color="white")
    out = m.encode_image([img])
    assert out.patch_embeds.ndim == 3          # (1, N, d)
    assert out.patch_embeds.shape[0] == 1
    assert out.patch_embeds.shape[-1] == 768   # SigLIP2-B hidden dim
    # Attention mask: at least 1 real patch
    assert out.attention_mask.sum().item() > 0


def test_naflex_text_tower_frozen():
    from experiments.action_primitives.backbones import SigLIP2Naflex

    m = SigLIP2Naflex(max_num_patches=64)
    text_params = list(m.model.text_model.parameters())
    assert all(not p.requires_grad for p in text_params)


def test_naflex_encodes_text():
    from experiments.action_primitives.backbones import SigLIP2Naflex

    m = SigLIP2Naflex(max_num_patches=64)
    tokens = m.encode_text(["click the red button"])
    assert tokens.ndim == 3  # (1, T, d)
    assert tokens.shape[-1] == 768
```

- [ ] **Step 3: Run smoke tests**

Run: `uv run pytest tests/action_primitives/test_backbones.py -v -m slow`

Expected: 3 tests pass (after model downloads on first run — may take several minutes).

- [ ] **Step 4: Commit**

```bash
git add experiments/action_primitives/backbones.py tests/action_primitives/test_backbones.py
git commit -m "feat(exp6): SigLIP2 naflex vision+text loader with smoke tests"
```

---

### Task 4: Typing legibility probe script

**Files:**
- Create: `experiments/action_primitives/probes/__init__.py`
- Create: `experiments/action_primitives/probes/typing_legibility.py`

Render varied text sizes onto 720×450 frames, encode through SigLIP2 naflex @ max_patches=256, run a linear probe from patch features → per-patch character-ID prediction. Output accuracy-vs-font-size curve.

- [ ] **Step 1: Create `probes/__init__.py`**

Empty file.

- [ ] **Step 2: Write the probe script**

Create `experiments/action_primitives/probes/typing_legibility.py`:

```python
"""Spike A — Typing legibility probe.

Renders pygame text at varied font sizes, encodes through SigLIP2 naflex,
trains a linear probe from patch features to per-patch character identity.
Reports top-1 accuracy vs font size.

Success signal: accuracy curve shows text becoming discriminable above a
threshold font size. If threshold >> 14pt, Q5's "typing handled via visual
feedback" assumption is questionable and Q6/max_num_patches should be revisited.
"""
from __future__ import annotations

import argparse
import json
import string
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pygame
import torch
import torch.nn as nn
from PIL import Image

from experiments.action_primitives.backbones import SigLIP2Naflex
from experiments.action_primitives.config import ENV


CHARSET = string.ascii_letters + string.digits + " "  # 63 classes incl. space
CHAR_TO_ID = {c: i for i, c in enumerate(CHARSET)}
ID_TO_CHAR = {i: c for c, i in CHAR_TO_ID.items()}


@dataclass
class ProbeResult:
    font_size: int
    top1_accuracy: float
    n_samples: int


def render_text_frame(text: str, font_size: int, font_name: str = "arial") -> Image.Image:
    """Render `text` centered on a 720x450 white background pygame surface.
    Returns the surface as a PIL.Image.
    """
    pygame.init()
    surf = pygame.Surface((ENV.canvas_w, ENV.canvas_h))
    surf.fill((255, 255, 255))
    font = pygame.font.SysFont(font_name, font_size)
    # Render line-by-line, centered
    lines = text.split("\n")
    y = ENV.canvas_h // 2 - (len(lines) * font_size) // 2
    for line in lines:
        txt_surf = font.render(line, True, (0, 0, 0))
        rect = txt_surf.get_rect(center=(ENV.canvas_w // 2, y + font_size // 2))
        surf.blit(txt_surf, rect)
        y += font_size
    arr = pygame.surfarray.array3d(surf).transpose(1, 0, 2)  # (H, W, 3)
    return Image.fromarray(arr, mode="RGB")


def build_dataset_for_size(font_size: int, n_strings: int = 50) -> tuple[list[Image.Image], list[str]]:
    """Generate n_strings random short strings at `font_size` + their ground-truth text."""
    rng = np.random.default_rng(42 + font_size)
    images: list[Image.Image] = []
    texts: list[str] = []
    for _ in range(n_strings):
        length = rng.integers(3, 12)
        chars = rng.choice(list(CHARSET), size=length)
        text = "".join(chars)
        images.append(render_text_frame(text, font_size))
        texts.append(text)
    return images, texts


def global_avg_patch_repr(out, target_mask_is_text: torch.Tensor) -> torch.Tensor:
    """Average patch embeddings over all real (non-padded) patches per sample.
    Returns (B, d). We deliberately do NOT try to localize text — this tests
    whether SigLIP2 representation at max_patches=256 *anywhere* encodes char identity.
    """
    mask = out.attention_mask.unsqueeze(-1).float()  # (B, N, 1)
    summed = (out.patch_embeds * mask).sum(dim=1)    # (B, d)
    counts = mask.sum(dim=1).clamp(min=1)            # (B, 1)
    return summed / counts


def train_string_presence_probe(
    model: SigLIP2Naflex,
    images: list[Image.Image],
    texts: list[str],
    charset_size: int = 63,
    device: str = "cpu",
) -> float:
    """Train a linear probe: globally-avg SigLIP2 features → multi-label presence-of-char (63 binary).
    Returns mean per-character F1 on the held-out split.
    """
    model = model.to(device).eval()
    X_list: list[torch.Tensor] = []
    with torch.no_grad():
        for img in images:
            out = model.encode_image([img])
            X_list.append(global_avg_patch_repr(out, None).cpu())
    X = torch.cat(X_list, dim=0)  # (N, d)
    # Multi-label target: y[i, c] = 1 if char c present in text[i]
    Y = torch.zeros(len(texts), charset_size)
    for i, t in enumerate(texts):
        for c in set(t):
            Y[i, CHAR_TO_ID[c]] = 1.0

    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, X_te = X[:split], X[split:]
    Y_tr, Y_te = Y[:split], Y[split:]

    # Linear probe: d -> charset_size
    probe = nn.Linear(X.shape[-1], charset_size)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(200):
        opt.zero_grad()
        loss = loss_fn(probe(X_tr), Y_tr)
        loss.backward()
        opt.step()

    with torch.no_grad():
        preds = (torch.sigmoid(probe(X_te)) > 0.5).float()
        # Per-char F1 (macro)
        tp = (preds * Y_te).sum(dim=0)
        fp = (preds * (1 - Y_te)).sum(dim=0)
        fn = ((1 - preds) * Y_te).sum(dim=0)
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        # Only average over chars that appear in the test set
        appearing = (Y_te.sum(dim=0) > 0).float()
        mean_f1 = (f1 * appearing).sum() / appearing.sum().clamp(min=1)

    return mean_f1.item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 10, 12, 14, 16, 20, 24, 32])
    parser.add_argument("--n-strings-per-size", type=int, default=80)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="docs/experiments/6-action-primitives-phase-a-results/spike-a-typing-legibility.json")
    args = parser.parse_args()

    m = SigLIP2Naflex(max_num_patches=256)

    results: list[ProbeResult] = []
    for size in args.sizes:
        images, texts = build_dataset_for_size(size, n_strings=args.n_strings_per_size)
        f1 = train_string_presence_probe(m, images, texts, device=args.device)
        print(f"font_size={size:3d}  mean_f1={f1:.3f}  n={len(images)}")
        results.append(ProbeResult(font_size=size, top1_accuracy=f1, n_samples=len(images)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump([asdict(r) for r in results], fh, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write a minimal unit test for `render_text_frame`**

Create `tests/action_primitives/test_typing_legibility.py`:

```python
"""Unit tests for probe helpers. Does not require SigLIP2 download."""

import pytest
from PIL import Image


def test_render_text_frame_size_and_mode():
    from experiments.action_primitives.probes.typing_legibility import render_text_frame

    img = render_text_frame("hello", font_size=14)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (720, 450)


def test_render_text_frame_empty_string():
    from experiments.action_primitives.probes.typing_legibility import render_text_frame

    img = render_text_frame("", font_size=14)
    assert img.size == (720, 450)


def test_build_dataset_for_size_returns_matched_lengths():
    from experiments.action_primitives.probes.typing_legibility import build_dataset_for_size

    imgs, texts = build_dataset_for_size(font_size=14, n_strings=5)
    assert len(imgs) == len(texts) == 5
    for img in imgs:
        assert img.size == (720, 450)
```

- [ ] **Step 4: Run unit tests**

Run: `uv run pytest tests/action_primitives/test_typing_legibility.py -v`

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/action_primitives/probes/ tests/action_primitives/test_typing_legibility.py
git commit -m "feat(exp6): Spike A typing legibility probe script + unit tests"
```

---

### Task 5: Execute Spike A and write up results

**Files:**
- Create: `docs/experiments/6-action-primitives-phase-a-results/spike-a-typing-legibility.json` (produced by script)
- Create: `docs/experiments/6-action-primitives-phase-a-results/spike-a-typing-legibility.md`

- [ ] **Step 1: Run the probe**

Run (from repo root):

```bash
uv run python -m experiments.action_primitives.probes.typing_legibility \
    --sizes 8 10 12 14 16 20 24 32 \
    --n-strings-per-size 80 \
    --device cpu
```

Expected: prints `font_size=XX  mean_f1=0.YY  n=80` for each size. Writes JSON to `docs/experiments/6-action-primitives-phase-a-results/spike-a-typing-legibility.json`. Runs in ~10–30 min on CPU (first run downloads SigLIP2 ~400MB).

- [ ] **Step 2: Write up the results**

Create `docs/experiments/6-action-primitives-phase-a-results/spike-a-typing-legibility.md` with this structure (fill in actual numbers from JSON):

```markdown
# Spike A — Typing legibility probe

**Ran:** 2026-04-XX
**Script:** `experiments/action_primitives/probes/typing_legibility.py`
**Data:** `spike-a-typing-legibility.json`

## Method
Rendered 80 random 3–12-char strings at each font size in {8, 10, 12, 14, 16, 20, 24, 32}pt on white 720×450 pygame surfaces. Encoded through SigLIP2-B-patch16-naflex @ max_num_patches=256. Trained a linear probe from globally-averaged patch features (d=768) to 63-class multi-label char-presence vector. Reports macro-F1 across 16-sample test split.

Note: this measures "does SigLIP2 see characters at all" via multi-label presence, not per-patch char identity. A stricter per-patch test is possible as future work.

## Results
| Font size (pt) | Macro-F1 |
|---|---|
| 8  | X.XX |
| 10 | X.XX |
| 12 | X.XX |
| 14 | X.XX |  ← Q6 design floor
| 16 | X.XX |
| 20 | X.XX |
| 24 | X.XX |
| 32 | X.XX |

## Interpretation
- **If F1 @ 14pt ≥ 0.7:** Q5/Q6 design stands — typing primitives can rely on visual feedback at the current `max_num_patches=256`. Proceed to Phase B as specified.
- **If F1 @ 14pt < 0.5:** typing visual-feedback assumption is broken. Phase B Decision: (a) bump `max_num_patches` to 576 for typing primitives specifically, or (b) shift typing progress signal to action history (revises Q5's "typing handled via visual feedback" call).
- **If 0.5 ≤ F1 @ 14pt < 0.7:** marginal. Recommend re-running with per-patch probe (stricter) before deciding.

## Recommendation
[To be filled in after reading the results]

## Next steps
- [ ] Per-patch probe version (optional, if marginal result above)
- [ ] Proceed to Spike B with current design, OR
- [ ] Revise Q5/Q6 per interpretation above
```

- [ ] **Step 3: Fill in the actual numbers and recommendation**

Open the JSON output; paste numbers into the Results table; write a recommendation in 2–3 sentences.

- [ ] **Step 4: Commit**

```bash
git add docs/experiments/6-action-primitives-phase-a-results/spike-a-*
git commit -m "results(exp6): Spike A typing legibility probe results + write-up"
```

- [ ] **Step 5: CHECKPOINT — user reviews Spike A write-up before continuing**

Spike A's outcome may change Phase A downstream tasks. Stop here and surface the write-up. Proceed only after user approval.

---

## Sub-Phase 3: Env, expert, generator (infra for Spikes B and E)

### Task 6: Minimal pygame canvas env

**Files:**
- Create: `experiments/action_primitives/env.py`
- Create: `tests/action_primitives/test_env.py`

Single-primitive L-click env. Canvas is 720×450 white, contains one colored target button at a randomized position. Accepts action dicts (dx/dy/click events) and returns next observation + reward-less done signal. Must run headless on a GPU node.

- [ ] **Step 1: Write env.py with a single-button L-click task**

Create `experiments/action_primitives/env.py`:

```python
"""Pygame L-click environment for Experiment 6 Phase A.

Canvas: 720x450 white with one colored button.
Goal: cursor over button, L_press → L_release → success.
Headless by default (SDL_VIDEODRIVER=dummy).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pygame
from PIL import Image

from experiments.action_primitives.config import ENV, NUM_KEYS


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Theme palettes (minimal Phase A set; full 20-theme system is Phase B)
THEMES = {
    "flat-modern":  {"bg": (245, 245, 248), "button": (80, 130, 230), "label": (255, 255, 255)},
    "flat-minimal": {"bg": (255, 255, 255), "button": (30, 30, 30),   "label": (255, 255, 255)},
    "dark-mode":    {"bg": (28, 30, 36),    "button": (100, 200, 140),"label": (10, 10, 10)},
}


@dataclass
class Action:
    """Per-frame action. Matches 6-head structure except done is training-only."""
    dx: float = 0.0
    dy: float = 0.0
    click: int = 0                          # 0=idle, 1=L_press, 2=L_release, 3=R_press, 4=R_release
    scroll: float = 0.0
    key_events: np.ndarray = field(         # (77,) int: 0=press, 1=release, 2=idle
        default_factory=lambda: np.full(NUM_KEYS, 2, dtype=np.int64)
    )


@dataclass
class Proprio:
    cursor_x: float
    cursor_y: float
    held_keys: np.ndarray                   # (77,) bool
    held_mouse: np.ndarray                  # (3,)  bool  [L, R, middle]
    capslock: bool


class LClickEnv:
    """L-click primitive environment.

    reset() returns initial observation + info.
    step(action) returns (obs, done, info).
    """

    def __init__(self, theme: str = "flat-modern", seed: int = 0) -> None:
        pygame.init()
        self.screen = pygame.Surface((ENV.canvas_w, ENV.canvas_h))
        self.rng = np.random.default_rng(seed)
        self.theme = THEMES[theme]
        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None) -> tuple[dict, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Randomize button position and size
        btn_w = int(self.rng.integers(40, 120))
        btn_h = int(self.rng.integers(30, 80))
        margin = 20
        btn_x = int(self.rng.integers(margin, ENV.canvas_w - btn_w - margin))
        btn_y = int(self.rng.integers(margin, ENV.canvas_h - btn_h - margin))
        self.target_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
        self.target_color = self.theme["button"]
        # Cursor starts at random position far from target
        while True:
            cx = int(self.rng.integers(10, ENV.canvas_w - 10))
            cy = int(self.rng.integers(10, ENV.canvas_h - 10))
            if not self.target_rect.collidepoint(cx, cy):
                break
        self.cursor_x = float(cx)
        self.cursor_y = float(cy)
        self.held_keys = np.zeros(NUM_KEYS, dtype=bool)
        self.held_mouse = np.zeros(3, dtype=bool)
        self.capslock = False
        self.done_flag = False
        self._press_frame: Optional[int] = None
        self._release_frame: Optional[int] = None
        self.frame_idx = 0

        obs = self._render_obs()
        info = self._info()
        return obs, info

    def step(self, action: Action) -> tuple[dict, bool, dict]:
        # Apply mouse delta (clipped to canvas)
        self.cursor_x = float(np.clip(self.cursor_x + action.dx, 0, ENV.canvas_w - 1))
        self.cursor_y = float(np.clip(self.cursor_y + action.dy, 0, ENV.canvas_h - 1))
        # Apply click event
        if action.click == 1:  # L_press
            self.held_mouse[0] = True
            if self._press_frame is None and self.target_rect.collidepoint(self.cursor_x, self.cursor_y):
                self._press_frame = self.frame_idx
        elif action.click == 2:  # L_release
            self.held_mouse[0] = False
            if self._release_frame is None and self._press_frame is not None:
                self._release_frame = self.frame_idx
                if self.target_rect.collidepoint(self.cursor_x, self.cursor_y):
                    self.done_flag = True
        # (R_press / R_release ignored for L-click primitive success)
        # Apply key events (press=0, release=1, idle=2)
        press_mask = action.key_events == 0
        release_mask = action.key_events == 1
        self.held_keys[press_mask] = True
        self.held_keys[release_mask] = False
        self.frame_idx += 1
        obs = self._render_obs()
        info = self._info()
        return obs, self.done_flag, info

    def _render_obs(self) -> dict:
        self.screen.fill(self.theme["bg"])
        pygame.draw.rect(self.screen, self.target_color, self.target_rect, border_radius=6)
        # Cursor: simple arrow (macOS default-ish ~32px sprite)
        cx, cy = int(self.cursor_x), int(self.cursor_y)
        pygame.draw.polygon(
            self.screen,
            (0, 0, 0),
            [(cx, cy), (cx + 14, cy + 10), (cx + 8, cy + 12), (cx + 12, cy + 20), (cx + 9, cy + 21), (cx + 5, cy + 13), (cx, cy + 16)],
        )
        # PIL Image for SigLIP2 naflex compatibility
        arr = pygame.surfarray.array3d(self.screen).transpose(1, 0, 2).copy()
        img = Image.fromarray(arr, mode="RGB")
        proprio = Proprio(
            cursor_x=self.cursor_x / ENV.canvas_w,
            cursor_y=self.cursor_y / ENV.canvas_h,
            held_keys=self.held_keys.copy(),
            held_mouse=self.held_mouse.copy(),
            capslock=self.capslock,
        )
        return {"image": img, "proprio": proprio}

    def _info(self) -> dict:
        return {
            "target_bbox": (self.target_rect.x, self.target_rect.y, self.target_rect.w, self.target_rect.h),
            "target_color": self.target_color,
            "cursor_xy": (self.cursor_x, self.cursor_y),
            "frame_idx": self.frame_idx,
            "success": self.done_flag,
        }
```

- [ ] **Step 2: Write env unit tests**

Create `tests/action_primitives/test_env.py`:

```python
"""Unit tests for LClickEnv."""

import numpy as np
import pytest

from experiments.action_primitives.env import Action, LClickEnv


def test_env_reset_returns_valid_obs_and_info():
    env = LClickEnv(seed=0)
    obs, info = env.reset(seed=0)
    assert "image" in obs and "proprio" in obs
    assert obs["image"].size == (720, 450)
    assert "target_bbox" in info
    x, y, w, h = info["target_bbox"]
    assert w > 0 and h > 0
    # Cursor should not start on target
    cx, cy = info["cursor_xy"]
    assert not (x <= cx <= x + w and y <= cy <= y + h)


def test_env_step_moves_cursor():
    env = LClickEnv(seed=0)
    obs, info = env.reset(seed=0)
    cx0, cy0 = info["cursor_xy"]
    action = Action(dx=10.0, dy=5.0)
    obs, done, info = env.step(action)
    cx1, cy1 = info["cursor_xy"]
    assert cx1 == cx0 + 10.0
    assert cy1 == cy0 + 5.0
    assert not done


def test_env_lclick_on_target_succeeds():
    env = LClickEnv(seed=7)
    obs, info = env.reset(seed=7)
    x, y, w, h = info["target_bbox"]
    tx, ty = x + w // 2, y + h // 2
    cx0, cy0 = info["cursor_xy"]
    # Move to target in one step (unrealistic but tests success detection)
    env.step(Action(dx=tx - cx0, dy=ty - cy0))
    env.step(Action(click=1))   # press
    obs, done, info = env.step(Action(click=2))  # release
    assert done is True
    assert info["success"] is True


def test_env_lclick_off_target_no_success():
    env = LClickEnv(seed=11)
    obs, info = env.reset(seed=11)
    x, y, w, h = info["target_bbox"]
    # Move far from target
    cx0, cy0 = info["cursor_xy"]
    env.step(Action(dx=0.0, dy=0.0))  # no-op, still off-target
    env.step(Action(click=1))
    obs, done, info = env.step(Action(click=2))
    assert done is False


def test_env_clamps_cursor_to_canvas():
    env = LClickEnv(seed=0)
    obs, info = env.reset(seed=0)
    # Try to move way off-screen
    obs, done, info = env.step(Action(dx=10000, dy=10000))
    cx, cy = info["cursor_xy"]
    assert 0 <= cx <= 720 - 1
    assert 0 <= cy <= 450 - 1
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/action_primitives/test_env.py -v`

Expected: 5 tests pass.

- [ ] **Step 4: Commit**

```bash
git add experiments/action_primitives/env.py tests/action_primitives/test_env.py
git commit -m "feat(exp6): LClickEnv pygame env + success detection + tests"
```

---

### Task 7: L-click expert (Fitts-law trajectory + tempo variability)

**Files:**
- Create: `experiments/action_primitives/expert.py`
- Create: `tests/action_primitives/test_expert.py`

Produces a stream of Action objects that drive the cursor from its current position to the target, settles for a few frames, then press+release. Tempo variability from Q5/Q9.

- [ ] **Step 1: Write expert.py**

Create `experiments/action_primitives/expert.py`:

```python
"""Fitts-law expert for L-click primitive.

Generates a trajectory of per-frame actions that drive the cursor to the target,
settles for a sampled number of frames, then emits L_press and L_release.
Includes tempo variability (slow / normal / fast / superhuman) per Q5/Q9.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from experiments.action_primitives.config import ENV, MOUSE_CAP_PX, NUM_KEYS
from experiments.action_primitives.env import Action


TEMPO_PROFILES = {
    "slow":       {"peak_speed_px": 18.0, "settle_frames": (2, 5)},
    "normal":     {"peak_speed_px": 35.0, "settle_frames": (1, 3)},
    "fast":       {"peak_speed_px": 60.0, "settle_frames": (0, 2)},
    "superhuman": {"peak_speed_px": 95.0, "settle_frames": (0, 1)},
}


@dataclass
class LClickExpertConfig:
    tempo: str = "normal"          # "slow" | "normal" | "fast" | "superhuman"
    overshoot_prob: float = 0.1     # probability of a human-like overshoot-correct
    seed: int = 0


def _idle_keys() -> np.ndarray:
    return np.full(NUM_KEYS, 2, dtype=np.int64)  # 2 == idle


class LClickExpert:
    """Iterator yielding per-frame Actions that drive L-click completion."""

    def __init__(
        self,
        cfg: LClickExpertConfig,
        cursor_xy: tuple[float, float],
        target_center: tuple[float, float],
    ) -> None:
        self.rng = np.random.default_rng(cfg.seed)
        self.cfg = cfg
        self.cursor = np.array(cursor_xy, dtype=np.float64)
        self.target = np.array(target_center, dtype=np.float64)
        self.profile = TEMPO_PROFILES[cfg.tempo]
        # State machine: move -> settle -> press -> release -> done
        self.state = "move"
        self.settle_remaining = int(self.rng.integers(*self.profile["settle_frames"]) + 1)
        self._overshoot_done = False

    def _move_step(self) -> Action:
        to_target = self.target - self.cursor
        dist = np.linalg.norm(to_target)
        if dist < 1.0:
            # Arrived; transition to settle
            self.state = "settle"
            return Action(dx=0.0, dy=0.0, key_events=_idle_keys())
        # Velocity profile: minimum-jerk-ish — peak in middle, taper ends
        peak = self.profile["peak_speed_px"]
        step_mag = min(peak, dist)
        direction = to_target / dist
        # Random overshoot near end
        if (not self._overshoot_done
            and dist < peak * 2
            and self.rng.random() < self.cfg.overshoot_prob):
            step_mag = min(dist * 1.4, MOUSE_CAP_PX)
            self._overshoot_done = True
        dx, dy = direction * step_mag
        # Clip to mouse cap
        dx = float(np.clip(dx, -MOUSE_CAP_PX, MOUSE_CAP_PX))
        dy = float(np.clip(dy, -MOUSE_CAP_PX, MOUSE_CAP_PX))
        self.cursor = self.cursor + np.array([dx, dy])
        return Action(dx=dx, dy=dy, key_events=_idle_keys())

    def __iter__(self) -> Iterator[Action]:
        return self

    def __next__(self) -> Action:
        if self.state == "move":
            return self._move_step()
        if self.state == "settle":
            if self.settle_remaining > 0:
                self.settle_remaining -= 1
                return Action(dx=0.0, dy=0.0, key_events=_idle_keys())
            self.state = "press"
            return Action(dx=0.0, dy=0.0, click=1, key_events=_idle_keys())  # L_press
        if self.state == "press":
            self.state = "release"
            return Action(dx=0.0, dy=0.0, click=2, key_events=_idle_keys())  # L_release
        # After release, stop iterating
        raise StopIteration
```

- [ ] **Step 2: Write expert tests**

Create `tests/action_primitives/test_expert.py`:

```python
"""Unit tests for LClickExpert."""

import numpy as np
import pytest

from experiments.action_primitives.expert import LClickExpert, LClickExpertConfig, TEMPO_PROFILES


def test_expert_reaches_target_and_clicks():
    expert = LClickExpert(
        cfg=LClickExpertConfig(tempo="normal", seed=0),
        cursor_xy=(100.0, 100.0),
        target_center=(400.0, 300.0),
    )
    actions = list(expert)
    # Last two should be press then release
    assert actions[-2].click == 1
    assert actions[-1].click == 2


def test_expert_respects_mouse_cap():
    expert = LClickExpert(
        cfg=LClickExpertConfig(tempo="superhuman", seed=0),
        cursor_xy=(10.0, 10.0),
        target_center=(700.0, 440.0),
    )
    for a in expert:
        assert abs(a.dx) <= 100.0 + 1e-5
        assert abs(a.dy) <= 100.0 + 1e-5


@pytest.mark.parametrize("tempo", list(TEMPO_PROFILES.keys()))
def test_expert_all_tempos_terminate(tempo):
    expert = LClickExpert(
        cfg=LClickExpertConfig(tempo=tempo, seed=0),
        cursor_xy=(0.0, 0.0),
        target_center=(360.0, 225.0),
    )
    actions = list(expert)
    assert len(actions) > 0
    assert actions[-1].click == 2


def test_expert_no_spurious_keys():
    expert = LClickExpert(
        cfg=LClickExpertConfig(tempo="normal", seed=0),
        cursor_xy=(100.0, 100.0),
        target_center=(400.0, 300.0),
    )
    for a in expert:
        # Idle == 2 for all 77 keys
        assert np.all(a.key_events == 2)
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/action_primitives/test_expert.py -v`

Expected: 7 tests pass (4 tempos × parametrize + 3 named).

- [ ] **Step 4: Commit**

```bash
git add experiments/action_primitives/expert.py tests/action_primitives/test_expert.py
git commit -m "feat(exp6): L-click Fitts-law expert with tempo variability + tests"
```

---

### Task 8: Episode generator + parquet shards

**Files:**
- Create: `experiments/action_primitives/generator.py`
- Create: `experiments/action_primitives/generate_data.py`
- Create: `tests/action_primitives/test_generator.py`

Generates N episodes, each stored as one row per frame: `(episode_id, frame_idx, image_jpeg, proprio, action, primitive_type)`. Writes parquet shards via HF datasets.

- [ ] **Step 1: Write generator.py (single-episode path)**

Create `experiments/action_primitives/generator.py`:

```python
"""Single-episode generator for L-click primitive.

Runs LClickEnv + LClickExpert in lockstep, emits one dict per frame including
rendered frame, proprio, expert action. Frame padding up to max_frames_lclick
uses no-op actions.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from PIL import Image

from experiments.action_primitives.config import ENV, NUM_KEYS
from experiments.action_primitives.env import Action, LClickEnv
from experiments.action_primitives.expert import LClickExpert, LClickExpertConfig


TEMPO_CHOICES = ("slow", "normal", "fast", "superhuman")
THEME_CHOICES = ("flat-modern", "flat-minimal", "dark-mode")


def _noop_action() -> Action:
    return Action()


def _action_to_row(a: Action) -> dict:
    return {
        "action_dx": float(a.dx),
        "action_dy": float(a.dy),
        "action_click": int(a.click),
        "action_scroll": float(a.scroll),
        "action_key_events": a.key_events.astype(np.int8).tolist(),
    }


def _proprio_to_row(p) -> dict:
    return {
        "cursor_x": float(p.cursor_x),
        "cursor_y": float(p.cursor_y),
        "held_keys": p.held_keys.astype(np.int8).tolist(),
        "held_mouse": p.held_mouse.astype(np.int8).tolist(),
        "capslock": int(p.capslock),
    }


def _image_to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def generate_one_episode(
    episode_id: int,
    seed: int,
    primitive: str = "lclick",
    theme: str | None = None,
    tempo: str | None = None,
    max_frames: int = ENV.max_frames_lclick,
) -> list[dict]:
    """Generate one episode; return list of per-frame rows."""
    rng = np.random.default_rng(seed)
    theme = theme if theme is not None else rng.choice(list(THEME_CHOICES))
    tempo = tempo if tempo is not None else rng.choice(list(TEMPO_CHOICES))

    env = LClickEnv(theme=theme, seed=seed)
    obs, info = env.reset(seed=seed)
    x, y, w, h = info["target_bbox"]
    target_center = (x + w / 2, y + h / 2)
    cursor_xy = info["cursor_xy"]

    expert_cfg = LClickExpertConfig(tempo=tempo, seed=seed + 1)
    expert = LClickExpert(expert_cfg, cursor_xy, target_center)

    rows: list[dict] = []
    done_frame = None

    # Drive expert until done, padding with no-ops up to max_frames
    frame_idx = 0
    expert_iter = iter(expert)
    while frame_idx < max_frames:
        if done_frame is None:
            try:
                action = next(expert_iter)
            except StopIteration:
                done_frame = frame_idx
                action = _noop_action()
        else:
            action = _noop_action()

        row = {
            "episode_id": int(episode_id),
            "frame_idx": int(frame_idx),
            "image_bytes": _image_to_jpeg_bytes(obs["image"]),
            "primitive_type": primitive,
            "theme": theme,
            "tempo": tempo,
            "target_bbox_x": int(x),
            "target_bbox_y": int(y),
            "target_bbox_w": int(w),
            "target_bbox_h": int(h),
            "done_gt": 1 if (done_frame is not None and frame_idx >= done_frame) else 0,
        }
        row.update(_action_to_row(action))
        row.update(_proprio_to_row(obs["proprio"]))
        rows.append(row)
        obs, env_done, info = env.step(action)
        # Sanity: env.done should align with expert-done within a 2-frame window
        frame_idx += 1

    return rows
```

- [ ] **Step 2: Write batched generate_data.py script**

Create `experiments/action_primitives/generate_data.py`:

```python
"""Phase A batched episode generator.

Generates N L-click episodes and writes a HuggingFace `datasets`-compatible
parquet shard set.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value

from experiments.action_primitives.config import PHASE_A_DATA
from experiments.action_primitives.generator import generate_one_episode


FEATURES = Features({
    "episode_id": Value("int64"),
    "frame_idx": Value("int64"),
    "image_bytes": Value("binary"),
    "primitive_type": Value("string"),
    "theme": Value("string"),
    "tempo": Value("string"),
    "target_bbox_x": Value("int64"),
    "target_bbox_y": Value("int64"),
    "target_bbox_w": Value("int64"),
    "target_bbox_h": Value("int64"),
    "done_gt": Value("int8"),
    "action_dx": Value("float32"),
    "action_dy": Value("float32"),
    "action_click": Value("int8"),
    "action_scroll": Value("float32"),
    "action_key_events": Sequence(Value("int8"), length=77),
    "cursor_x": Value("float32"),
    "cursor_y": Value("float32"),
    "held_keys": Sequence(Value("int8"), length=77),
    "held_mouse": Sequence(Value("int8"), length=3),
    "capslock": Value("int8"),
})


def generate_all(n_episodes: int, out_dir: Path, shard_size: int = 500) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_rows: list[dict] = []
    shard_idx = 0
    t0 = time.time()
    for i in range(n_episodes):
        rows = generate_one_episode(episode_id=i, seed=i)
        shard_rows.extend(rows)
        if (i + 1) % shard_size == 0 or (i + 1) == n_episodes:
            ds = Dataset.from_list(shard_rows, features=FEATURES)
            shard_path = out_dir / f"shard_{shard_idx:04d}.parquet"
            ds.to_parquet(shard_path)
            print(f"[shard {shard_idx}] episodes {i + 1 - len(shard_rows) // 30 + 1}..{i + 1}  frames={len(shard_rows)}  → {shard_path}")
            shard_rows = []
            shard_idx += 1
    elapsed = time.time() - t0
    print(f"Generated {n_episodes} episodes in {elapsed:.1f}s  ({n_episodes / elapsed:.2f} eps/s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-episodes", type=int, default=PHASE_A_DATA.n_episodes)
    parser.add_argument("-o", "--out-dir", type=str, default="data/phase-a-lclick")
    parser.add_argument("--shard-size", type=int, default=500, help="Episodes per parquet shard")
    args = parser.parse_args()
    generate_all(args.n_episodes, Path(args.out_dir), args.shard_size)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write generator tests**

Create `tests/action_primitives/test_generator.py`:

```python
"""Unit tests for episode generator."""

import pytest

from experiments.action_primitives.config import ENV, NUM_KEYS
from experiments.action_primitives.generator import generate_one_episode


def test_generate_one_episode_returns_fixed_window():
    rows = generate_one_episode(episode_id=0, seed=0)
    assert len(rows) == ENV.max_frames_lclick


def test_generate_one_episode_schema_consistency():
    rows = generate_one_episode(episode_id=0, seed=0)
    required = {"episode_id", "frame_idx", "image_bytes", "action_dx", "action_dy",
                "action_click", "action_scroll", "action_key_events", "cursor_x",
                "cursor_y", "held_keys", "held_mouse", "capslock", "done_gt",
                "target_bbox_x", "primitive_type", "theme", "tempo"}
    for row in rows:
        assert required.issubset(row.keys())
        assert len(row["action_key_events"]) == NUM_KEYS
        assert len(row["held_keys"]) == NUM_KEYS
        assert len(row["held_mouse"]) == 3


def test_generate_one_episode_done_monotonic():
    rows = generate_one_episode(episode_id=0, seed=0)
    seen_done = False
    for row in rows:
        if row["done_gt"] == 1:
            seen_done = True
        else:
            # Once done_gt flips to 1 it should stay 1 for the rest of the episode
            assert not seen_done, f"done_gt went 1→0 at frame {row['frame_idx']}"


def test_generate_one_episode_deterministic():
    r1 = generate_one_episode(episode_id=0, seed=42)
    r2 = generate_one_episode(episode_id=0, seed=42)
    assert len(r1) == len(r2)
    for a, b in zip(r1, r2):
        assert a["action_dx"] == b["action_dx"]
        assert a["action_click"] == b["action_click"]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/action_primitives/test_generator.py -v`

Expected: 4 tests pass.

- [ ] **Step 5: Smoke test the parquet writer with a tiny run**

Run: `uv run python -m experiments.action_primitives.generate_data -n 20 -o data/smoke-test --shard-size 10`

Expected: 2 parquet shards written; prints `Generated 20 episodes in X.Xs (Y.YY eps/s)`. Verify shards exist:

```bash
ls -lh data/smoke-test/
```

- [ ] **Step 6: Clean up smoke-test data**

Run: `rm -rf data/smoke-test`

- [ ] **Step 7: Commit**

```bash
git add experiments/action_primitives/generator.py experiments/action_primitives/generate_data.py tests/action_primitives/test_generator.py
git commit -m "feat(exp6): episode generator + parquet shards + tests"
```

---

### Task 9: Spike E — Generation throughput measurement

**Files:**
- Create: `experiments/action_primitives/measurements/__init__.py`
- Create: `experiments/action_primitives/measurements/gen_throughput.py`
- Create: `docs/experiments/6-action-primitives-phase-a-results/spike-e-generation-throughput.md`

- [ ] **Step 1: Create measurements package init**

Empty `experiments/action_primitives/measurements/__init__.py`.

- [ ] **Step 2: Write the throughput script**

Create `experiments/action_primitives/measurements/gen_throughput.py`:

```python
"""Spike E — Pygame generation throughput measurement.

Validates Q7's "≥200 eps/sec" claim. Generates 1K L-click episodes
and reports eps/s, frames/s, per-episode storage.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from experiments.action_primitives.generator import generate_one_episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-episodes", type=int, default=1000)
    args = parser.parse_args()

    t0 = time.time()
    total_frames = 0
    total_bytes = 0
    for i in range(args.n_episodes):
        rows = generate_one_episode(episode_id=i, seed=i)
        total_frames += len(rows)
        total_bytes += sum(len(r["image_bytes"]) for r in rows)
    elapsed = time.time() - t0
    eps_per_s = args.n_episodes / elapsed
    frames_per_s = total_frames / elapsed
    avg_bytes_per_ep = total_bytes / args.n_episodes
    print(f"Episodes:          {args.n_episodes}")
    print(f"Total frames:      {total_frames}")
    print(f"Wall-clock:        {elapsed:.2f}s")
    print(f"eps/s:             {eps_per_s:.2f}  (target: ≥200)")
    print(f"frames/s:          {frames_per_s:.1f}")
    print(f"Avg bytes/episode: {avg_bytes_per_ep / 1024:.1f} KB")
    print(f"Projected 24500-ep dataset size: {24500 * avg_bytes_per_ep / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the measurement**

Run: `uv run python -m experiments.action_primitives.measurements.gen_throughput -n 1000`

Expected output format:
```
Episodes:          1000
Total frames:      ~30000
Wall-clock:        X.Xs
eps/s:             YY.Y  (target: ≥200)
...
```

- [ ] **Step 4: Write up Spike E results**

Create `docs/experiments/6-action-primitives-phase-a-results/spike-e-generation-throughput.md`:

```markdown
# Spike E — Generation throughput

**Ran:** 2026-04-XX
**Script:** `experiments/action_primitives/measurements/gen_throughput.py`
**Command:** `uv run python -m experiments.action_primitives.measurements.gen_throughput -n 1000`

## Results
- Wall-clock for 1000 episodes: X.Xs
- **eps/s: YY.Y** (Q7 target ≥200)
- frames/s: ZZZ
- Avg bytes/episode: NN KB
- Projected 24,500-episode dataset: X.X GB on disk

## Interpretation
[Fill in — if eps/s ≥ 200, Q7 validated. If significantly slower, estimate Phase B generation timeline impact and consider optimizations: multi-process generator, or JPEG quality reduction.]

## Recommendation
[Proceed to Phase B as-is / add multi-process parallelization / revise Q7 estimate]
```

- [ ] **Step 5: Fill in actual numbers**

Paste the real throughput numbers into the write-up.

- [ ] **Step 6: Commit**

```bash
git add experiments/action_primitives/measurements/ docs/experiments/6-action-primitives-phase-a-results/spike-e-*
git commit -m "feat+results(exp6): Spike E generation throughput measurement + write-up"
```

- [ ] **Step 7: CHECKPOINT — user reviews Spike E write-up**

---

## Sub-Phase 4: Model (for Spike B)

### Task 10: Proprio + action-history encoders

**Files:**
- Create: `experiments/action_primitives/proprio.py`
- Create: `experiments/action_primitives/history.py`
- Create: `tests/action_primitives/test_encoders.py`

Small shared helpers that produce single 768-dim tokens from proprio (83 dim) and history (8 × 300 dim composite action vectors).

- [ ] **Step 1: Write proprio.py**

Create `experiments/action_primitives/proprio.py`:

```python
"""Proprio encoder: 83-dim state → single 768-dim token.

Per Q15: proprio is state-as-input integrated as a K/V token in the trunk.
Plain 2-layer MLP with GELU. Not FiLM, not AdaLN (Q15).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from experiments.action_primitives.config import MODEL, PROPRIO_DIM


class ProprioEncoder(nn.Module):
    def __init__(self, d_model: int = MODEL.d_model) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PROPRIO_DIM, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        """proprio: (B, PROPRIO_DIM) float → (B, 1, d_model)."""
        x = self.net(proprio)
        return x.unsqueeze(1)
```

- [ ] **Step 2: Write history.py**

Create `experiments/action_primitives/history.py`:

```python
"""Action-history encoder (Q19).

Input: per-timestep composite action vectors of shape (B, K=8, 300).
Composite: dx one-hot (21) + dy (21) + click (5) + scroll (21) + keys press-mask (77) + keys release-mask (77) + done (1) + 77 pad zeros.
Wait — the 300 in the design sums to 300. Let's be precise:
  21 + 21 + 5 + 21 + 77 + 77 + 1 = 223 ... +77 zero pad = 300? No; recount.
Design doc Q19 says: 21+21+5+21+154+1 = 223 core dims; we pad to 300 via MLP.
  (154 = 77 press bits + 77 release bits)
We take 223 as the true input width and project 223 → 256 → 768.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from experiments.action_primitives.config import MODEL, NUM_BINS_MOUSE, NUM_BINS_SCROLL, NUM_CLICK_EVENTS, NUM_KEYS


HISTORY_INPUT_DIM: int = (
    NUM_BINS_MOUSE + NUM_BINS_MOUSE      # dx, dy one-hot: 42
    + NUM_CLICK_EVENTS                   # click: 5
    + NUM_BINS_SCROLL                    # scroll one-hot: 21
    + 2 * NUM_KEYS                       # keys press + release mask: 154
    + 1                                  # done bit: 1
)  # = 223


class HistoryEncoder(nn.Module):
    def __init__(self, d_model: int = MODEL.d_model, history_len: int = MODEL.action_history_len) -> None:
        super().__init__()
        self.history_len = history_len
        self.proj = nn.Sequential(
            nn.Linear(HISTORY_INPUT_DIM, 256),
            nn.GELU(),
            nn.Linear(256, d_model),
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, history_len, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """history: (B, K, HISTORY_INPUT_DIM) → (B, K, d_model) with temporal PE."""
        x = self.proj(history)
        return x + self.pos_emb[:, : x.size(1), :]
```

- [ ] **Step 3: Write encoder tests**

Create `tests/action_primitives/test_encoders.py`:

```python
"""Unit tests for proprio and history encoders."""

import torch

from experiments.action_primitives.config import MODEL, PROPRIO_DIM
from experiments.action_primitives.history import HISTORY_INPUT_DIM, HistoryEncoder
from experiments.action_primitives.proprio import ProprioEncoder


def test_proprio_encoder_shape():
    enc = ProprioEncoder()
    x = torch.randn(4, PROPRIO_DIM)
    out = enc(x)
    assert out.shape == (4, 1, MODEL.d_model)


def test_history_encoder_shape():
    enc = HistoryEncoder()
    x = torch.randn(4, MODEL.action_history_len, HISTORY_INPUT_DIM)
    out = enc(x)
    assert out.shape == (4, MODEL.action_history_len, MODEL.d_model)


def test_history_input_dim_matches_heads():
    # 21 + 21 + 5 + 21 + 77 + 77 + 1 = 223
    assert HISTORY_INPUT_DIM == 223


def test_history_encoder_temporal_pe_nonzero_on_init():
    enc = HistoryEncoder()
    # PE should be non-trivially initialized (not all zeros)
    assert enc.pos_emb.abs().sum().item() > 0
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/action_primitives/test_encoders.py -v`

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/action_primitives/proprio.py experiments/action_primitives/history.py tests/action_primitives/test_encoders.py
git commit -m "feat(exp6): proprio + action-history encoders with shape tests"
```

---

### Task 11: ACT trunk (cross+self attention)

**Files:**
- Create: `experiments/action_primitives/trunk.py`
- Create: `tests/action_primitives/test_trunk.py`

3 alternating cross+self attention blocks; 16 learnable query tokens.

- [ ] **Step 1: Write trunk.py**

Create `experiments/action_primitives/trunk.py`:

```python
"""ACT trunk: learnable queries cross-attending to a K/V pool, then self-attending.

Per Q15: 16 queries, 3 alternating cross+self blocks, dim 768, 12 heads, 4× FFN.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from experiments.action_primitives.config import MODEL


class CrossSelfBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int) -> None:
        super().__init__()
        self.norm_q_cross = nn.LayerNorm(d_model)
        self.norm_kv_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        kv_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Cross-attn: queries attend to kv
        q_in = self.norm_q_cross(queries)
        kv_in = self.norm_kv_cross(kv)
        x, _ = self.cross_attn(q_in, kv_in, kv_in, key_padding_mask=kv_key_padding_mask)
        queries = queries + x
        # Self-attn among queries
        q_in = self.norm_self(queries)
        x, _ = self.self_attn(q_in, q_in, q_in)
        queries = queries + x
        # FFN
        x = self.ffn(self.norm_ffn(queries))
        queries = queries + x
        return queries


class Trunk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, MODEL.n_queries, MODEL.d_model))
        nn.init.normal_(self.query_tokens, std=0.02)
        self.blocks = nn.ModuleList([
            CrossSelfBlock(MODEL.d_model, MODEL.n_heads, MODEL.ffn_mult)
            for _ in range(MODEL.n_blocks)
        ])

    def forward(self, kv: torch.Tensor, kv_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """kv: (B, N, d), kv_key_padding_mask: (B, N) with True at pad positions (torch convention).
        Returns (B, K, d) final query states."""
        B = kv.size(0)
        queries = self.query_tokens.expand(B, -1, -1).contiguous()
        for block in self.blocks:
            queries = block(queries, kv, kv_key_padding_mask)
        return queries
```

- [ ] **Step 2: Write trunk tests**

Create `tests/action_primitives/test_trunk.py`:

```python
"""Unit tests for Trunk (cross+self attention)."""

import torch

from experiments.action_primitives.config import MODEL
from experiments.action_primitives.trunk import CrossSelfBlock, Trunk


def test_cross_self_block_forward_shape():
    block = CrossSelfBlock(d_model=MODEL.d_model, n_heads=MODEL.n_heads, ffn_mult=MODEL.ffn_mult)
    q = torch.randn(2, MODEL.n_queries, MODEL.d_model)
    kv = torch.randn(2, 100, MODEL.d_model)
    out = block(q, kv)
    assert out.shape == q.shape


def test_trunk_forward_shape():
    trunk = Trunk()
    kv = torch.randn(2, 265, MODEL.d_model)  # ~240 vision + 16 text + 1 proprio + 8 history
    out = trunk(kv)
    assert out.shape == (2, MODEL.n_queries, MODEL.d_model)


def test_trunk_respects_padding_mask():
    trunk = Trunk()
    kv = torch.randn(2, 300, MODEL.d_model)
    # Mask out last 50 tokens
    mask = torch.zeros(2, 300, dtype=torch.bool)
    mask[:, 250:] = True  # True = pad (torch convention for key_padding_mask)
    out = trunk(kv, kv_key_padding_mask=mask)
    assert out.shape == (2, MODEL.n_queries, MODEL.d_model)
    assert not torch.isnan(out).any()


def test_trunk_has_expected_n_blocks():
    trunk = Trunk()
    assert len(trunk.blocks) == MODEL.n_blocks
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/action_primitives/test_trunk.py -v`

Expected: 4 tests pass.

- [ ] **Step 4: Commit**

```bash
git add experiments/action_primitives/trunk.py tests/action_primitives/test_trunk.py
git commit -m "feat(exp6): 3-block cross+self attention trunk with 16 queries"
```

---

### Task 12: Output heads (6 factored heads, unpooled)

**Files:**
- Create: `experiments/action_primitives/heads.py`
- Create: `tests/action_primitives/test_heads.py`

Unpooled (flatten 16 × 768 → 12288) → 6 linear heads.

- [ ] **Step 1: Write heads.py**

Create `experiments/action_primitives/heads.py`:

```python
"""Output heads: unpooled flatten(16 queries × 768) → 6 action heads.

Per Q16: 21+21+5+21+231+1 = 300 logits/frame; unpooled trades ~3.7M params
for no information loss vs pooling.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from experiments.action_primitives.config import HEAD_LOGITS, MODEL


class ActionHeads(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        flat_dim = MODEL.n_queries * MODEL.d_model  # 16 * 768 = 12288
        self.heads = nn.ModuleDict({
            name: nn.Linear(flat_dim, n_logits)
            for name, n_logits in HEAD_LOGITS.items()
        })

    def forward(self, queries: torch.Tensor) -> dict[str, torch.Tensor]:
        """queries: (B, K, d) → {head_name: (B, n_logits)}."""
        flat = queries.flatten(start_dim=1)  # (B, K*d)
        return {name: head(flat) for name, head in self.heads.items()}
```

- [ ] **Step 2: Write head tests**

Create `tests/action_primitives/test_heads.py`:

```python
"""Unit tests for ActionHeads."""

import torch

from experiments.action_primitives.config import HEAD_LOGITS, MODEL
from experiments.action_primitives.heads import ActionHeads


def test_action_heads_output_shapes():
    heads = ActionHeads()
    q = torch.randn(3, MODEL.n_queries, MODEL.d_model)
    out = heads(q)
    for name, n in HEAD_LOGITS.items():
        assert out[name].shape == (3, n), f"{name} got {out[name].shape}"


def test_action_heads_produce_all_six_outputs():
    heads = ActionHeads()
    q = torch.randn(1, MODEL.n_queries, MODEL.d_model)
    out = heads(q)
    assert set(out.keys()) == {"dx", "dy", "click", "scroll", "keys", "done"}
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/action_primitives/test_heads.py -v`

Expected: 2 tests pass.

- [ ] **Step 4: Commit**

```bash
git add experiments/action_primitives/heads.py tests/action_primitives/test_heads.py
git commit -m "feat(exp6): unpooled 6-head action output module"
```

---

### Task 13: Compose full ACT model

**Files:**
- Create: `experiments/action_primitives/model.py`
- Create: `tests/action_primitives/test_model.py`

Wires SigLIP2 (vision + text) + proprio + history + trunk + heads. Accepts PIL images + tokenized text + proprio tensor + history tensor; returns 6 head logits.

- [ ] **Step 1: Write model.py**

Create `experiments/action_primitives/model.py`:

```python
"""Full ACT model for Experiment 6 Phase A.

Wires SigLIP2 naflex vision+text, proprio encoder, action-history encoder,
trunk, and output heads per the Q15 wiring spec.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from experiments.action_primitives.backbones import SigLIP2Naflex
from experiments.action_primitives.config import MODEL
from experiments.action_primitives.heads import ActionHeads
from experiments.action_primitives.history import HISTORY_INPUT_DIM, HistoryEncoder
from experiments.action_primitives.proprio import ProprioEncoder
from experiments.action_primitives.trunk import Trunk


@dataclass
class ACTOutput:
    head_logits: dict[str, torch.Tensor]  # {head_name: (B, n_logits)}


class ActionPrimitivesACT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = SigLIP2Naflex()
        self.proprio_enc = ProprioEncoder()
        self.history_enc = HistoryEncoder()
        self.trunk = Trunk()
        self.heads = ActionHeads()

    def forward(
        self,
        images: list,                    # list of PIL.Image, len B
        text_tokens: torch.Tensor,       # (B, T_text, d) — precomputed & cached
        text_mask: torch.Tensor,         # (B, T_text) — 1=real, 0=pad
        proprio: torch.Tensor,           # (B, PROPRIO_DIM)
        action_history: torch.Tensor,    # (B, K, HISTORY_INPUT_DIM)
    ) -> ACTOutput:
        # 1. Vision encoder (trainable via LoRA)
        vis = self.backbone.encode_image(images)
        # 2. Text tokens (frozen, passed in as cached argument)
        # 3. Proprio → single token
        proprio_tok = self.proprio_enc(proprio)             # (B, 1, d)
        # 4. Action history → K tokens
        history_toks = self.history_enc(action_history)     # (B, K, d)
        # 5. Concat K/V pool: vision + text + proprio + history
        kv = torch.cat([vis.patch_embeds, text_tokens, proprio_tok, history_toks], dim=1)
        # Build key-padding mask (True == pad; torch convention)
        vis_pad = ~vis.attention_mask.bool()
        text_pad = ~text_mask.bool()
        proprio_pad = torch.zeros(proprio.size(0), 1, dtype=torch.bool, device=kv.device)
        history_pad = torch.zeros(proprio.size(0), history_toks.size(1), dtype=torch.bool, device=kv.device)
        kv_mask = torch.cat([vis_pad, text_pad, proprio_pad, history_pad], dim=1)
        # 6. Trunk: queries cross/self-attend
        query_out = self.trunk(kv, kv_key_padding_mask=kv_mask)
        # 7. Heads
        head_logits = self.heads(query_out)
        return ACTOutput(head_logits=head_logits)

    def trainable_parameters_summary(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"total={total / 1e6:.1f}M  trainable={trainable / 1e6:.1f}M"
```

- [ ] **Step 2: Write model forward-pass smoke test**

Create `tests/action_primitives/test_model.py`:

```python
"""Smoke test for full ACTModel forward pass.

Marked `slow` — downloads SigLIP2 (~400MB) on first run.
"""
import pytest
import torch
from PIL import Image

from experiments.action_primitives.config import HEAD_LOGITS, MODEL, PROPRIO_DIM
from experiments.action_primitives.history import HISTORY_INPUT_DIM

pytestmark = pytest.mark.slow


def test_model_forward_pass_and_shapes():
    from experiments.action_primitives.model import ActionPrimitivesACT

    model = ActionPrimitivesACT().eval()
    images = [Image.new("RGB", (720, 450), color=(255, 255, 255)) for _ in range(2)]
    # Build dummy text cache: B=2, T=16 tokens, d=768
    text_tokens = torch.randn(2, 16, MODEL.d_model)
    text_mask = torch.ones(2, 16)
    proprio = torch.randn(2, PROPRIO_DIM)
    history = torch.randn(2, MODEL.action_history_len, HISTORY_INPUT_DIM)

    with torch.no_grad():
        out = model(images, text_tokens, text_mask, proprio, history)

    for name, n_logits in HEAD_LOGITS.items():
        assert out.head_logits[name].shape == (2, n_logits)


def test_model_parameter_counts_reasonable():
    from experiments.action_primitives.model import ActionPrimitivesACT

    model = ActionPrimitivesACT()
    total = sum(p.numel() for p in model.parameters())
    # Per Q16: ~118M total inference, ~86M from SigLIP2 (42M vision + 44M text)
    assert 100e6 < total < 150e6, f"unexpected param count: {total / 1e6:.1f}M"
```

- [ ] **Step 3: Run slow smoke test**

Run: `uv run pytest tests/action_primitives/test_model.py -v -m slow`

Expected: 2 tests pass. First run takes minutes (model download).

- [ ] **Step 4: Commit**

```bash
git add experiments/action_primitives/model.py tests/action_primitives/test_model.py
git commit -m "feat(exp6): compose full ACTModel (vision+text+proprio+history+trunk+heads)"
```

---

### Task 14: LoRA adapters on SigLIP2 vision

**Files:**
- Modify: `experiments/action_primitives/backbones.py`

Add `peft.LoraConfig` adapters to the vision tower attention Q/K/V/O projections per Q15.

- [ ] **Step 1: Modify SigLIP2Naflex to support LoRA wrapping**

Open `experiments/action_primitives/backbones.py`. Add an `apply_lora()` method after `__init__`:

```python
    def apply_lora(self, rank: int = 8) -> None:
        """Apply LoRA adapters to vision tower attention projections (Q15)."""
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            modules_to_save=[],
            bias="none",
            lora_dropout=0.0,
        )
        # Apply only to vision_model (text stays frozen)
        self.model.vision_model = get_peft_model(self.model.vision_model, lora_cfg)
```

And modify `ActionPrimitivesACT.__init__` in `model.py` to call `apply_lora` on load:

```python
    def __init__(self) -> None:
        super().__init__()
        self.backbone = SigLIP2Naflex()
        self.backbone.apply_lora(rank=MODEL.lora_rank)  # <-- new line
        self.proprio_enc = ProprioEncoder()
        self.history_enc = HistoryEncoder()
        self.trunk = Trunk()
        self.heads = ActionHeads()
```

- [ ] **Step 2: Add LoRA param count test**

Append to `tests/action_primitives/test_model.py`:

```python
def test_lora_adapters_are_trainable():
    from experiments.action_primitives.model import ActionPrimitivesACT

    model = ActionPrimitivesACT()
    lora_params = [n for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    assert len(lora_params) > 0, "no trainable LoRA params found"


def test_text_tower_still_frozen_after_lora():
    from experiments.action_primitives.model import ActionPrimitivesACT

    model = ActionPrimitivesACT()
    text_trainable = [
        n for n, p in model.backbone.model.text_model.named_parameters() if p.requires_grad
    ]
    assert len(text_trainable) == 0, f"text tower has trainable params: {text_trainable}"
```

- [ ] **Step 3: Run slow tests**

Run: `uv run pytest tests/action_primitives/test_model.py -v -m slow`

Expected: 4 tests pass.

- [ ] **Step 4: Commit**

```bash
git add experiments/action_primitives/backbones.py experiments/action_primitives/model.py tests/action_primitives/test_model.py
git commit -m "feat(exp6): LoRA rank-8 adapters on SigLIP2 vision tower"
```

---

## Sub-Phase 5: Training

### Task 15: Loss functions

**Files:**
- Create: `experiments/action_primitives/losses.py`
- Create: `tests/action_primitives/test_losses.py`

Focal CE per head + idle-biased label smoothing (for keys) + per-class inverse-frequency weighting (for keys).

- [ ] **Step 1: Write losses.py**

Create `experiments/action_primitives/losses.py`:

```python
"""Loss functions for Experiment 6 (Q2, Q3).

- Focal CE on all action heads (mouse bins, click, scroll, keys 3-way)
- Label smoothing on bin heads (smoothing across adjacent bins)
- Idle-biased smoothing on keys (concentrate smoothing mass on idle neighbor)
- Per-class inverse-frequency weighting on keys (boosts rare keys)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.action_primitives.config import (
    HEAD_LOGITS,
    KEY_STATE_IDLE,
    LOSS,
    NUM_BINS_MOUSE,
    NUM_KEYS,
)


def focal_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Focal cross-entropy (Lin et al. 2017 focal loss, softmax variant).

    logits: (N, C)    target: (N,) int64 class ids.
    """
    logp = F.log_softmax(logits, dim=-1)
    if label_smoothing > 0.0:
        C = logits.size(-1)
        smooth = torch.full_like(logp, label_smoothing / (C - 1))
        smooth.scatter_(-1, target.unsqueeze(-1), 1.0 - label_smoothing)
        ce = -(smooth * logp).sum(dim=-1)
        p_true = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1).exp()
    else:
        ce = F.nll_loss(logp, target, reduction="none")
        p_true = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1).exp()
    focal_weight = (1.0 - p_true) ** gamma
    return (focal_weight * ce).mean()


def keys_focal_loss(
    logits_keys: torch.Tensor,     # (B, NUM_KEYS * 3)
    target_keys: torch.Tensor,     # (B, NUM_KEYS) int64 with 0=press, 1=release, 2=idle
    gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,  # (NUM_KEYS, 3) or None
    idle_smoothing: float = 0.05,
) -> torch.Tensor:
    """Per-key 3-way focal CE; each key treated independently.

    Handles 77 independent softmaxes in parallel by reshaping to (B*77, 3).
    """
    B = logits_keys.size(0)
    logits = logits_keys.view(B, NUM_KEYS, 3)
    logp = F.log_softmax(logits, dim=-1)  # (B, NUM_KEYS, 3)

    # Idle-biased smoothing: smoothing mass on the idle neighbor, not uniform
    smooth = torch.full_like(logp, 0.0)
    # Main target
    smooth.scatter_(-1, target_keys.unsqueeze(-1), 1.0 - idle_smoothing)
    # Add smoothing mass on idle (even if idle IS the target — no-op since scatter overwrote)
    idle_slot = smooth[..., KEY_STATE_IDLE]
    mask_target_not_idle = (target_keys != KEY_STATE_IDLE).float()
    smooth[..., KEY_STATE_IDLE] = idle_slot + idle_smoothing * mask_target_not_idle
    # For targets that ARE idle, leave full mass on idle (no redistribution)

    ce = -(smooth * logp).sum(dim=-1)  # (B, NUM_KEYS)

    # Focal weight
    p_true = logp.gather(-1, target_keys.unsqueeze(-1)).squeeze(-1).exp()
    focal_weight = (1.0 - p_true) ** gamma

    # Per-class weighting (rare keys)
    if class_weights is not None:
        # class_weights: (NUM_KEYS, 3); pick based on target
        w = class_weights.gather(-1, target_keys.unsqueeze(-1)).squeeze(-1)  # (B, NUM_KEYS)
        focal_weight = focal_weight * w

    return (focal_weight * ce).mean()


def done_loss(logits_done: torch.Tensor, target_done: torch.Tensor) -> torch.Tensor:
    """Focal BCE for the 1-logit done head (Q8)."""
    # logits_done: (B, 1)  target_done: (B,) float
    logits = logits_done.squeeze(-1)
    bce = F.binary_cross_entropy_with_logits(logits, target_done.float(), reduction="none")
    p_true = torch.sigmoid(logits)
    # Focal: (1 - p_t)^gamma
    p_t = torch.where(target_done.bool(), p_true, 1 - p_true)
    focal_weight = (1.0 - p_t) ** LOSS.focal_gamma
    return (focal_weight * bce).mean()


def total_loss(
    head_logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    head_weights: dict[str, float],
    class_weights_keys: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute weighted sum of per-head losses; return (total, per-head dict)."""
    per_head = {
        "dx":     focal_ce_loss(head_logits["dx"],     targets["dx"],     LOSS.focal_gamma, LOSS.label_smoothing_mouse),
        "dy":     focal_ce_loss(head_logits["dy"],     targets["dy"],     LOSS.focal_gamma, LOSS.label_smoothing_mouse),
        "click":  focal_ce_loss(head_logits["click"],  targets["click"],  LOSS.focal_gamma),
        "scroll": focal_ce_loss(head_logits["scroll"], targets["scroll"], LOSS.focal_gamma, LOSS.label_smoothing_mouse),
        "keys":   keys_focal_loss(head_logits["keys"], targets["keys"], LOSS.focal_gamma, class_weights_keys, LOSS.idle_smoothing_keys),
        "done":   done_loss(head_logits["done"], targets["done"]),
    }
    total = sum(head_weights[n] * per_head[n] for n in per_head)
    return total, per_head
```

- [ ] **Step 2: Write loss tests**

Create `tests/action_primitives/test_losses.py`:

```python
"""Unit tests for loss functions."""

import torch

from experiments.action_primitives.config import HEAD_LOGITS, NUM_BINS_MOUSE, NUM_CLICK_EVENTS, NUM_KEYS
from experiments.action_primitives.losses import (
    focal_ce_loss,
    keys_focal_loss,
    done_loss,
    total_loss,
)


def test_focal_ce_loss_scalar_and_finite():
    logits = torch.randn(8, NUM_BINS_MOUSE)
    target = torch.randint(0, NUM_BINS_MOUSE, (8,))
    loss = focal_ce_loss(logits, target, gamma=2.0, label_smoothing=0.05)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_focal_ce_loss_decreases_when_confidence_on_target_increases():
    target = torch.tensor([0])
    logits_wrong = torch.zeros(1, NUM_BINS_MOUSE)
    logits_right = torch.zeros(1, NUM_BINS_MOUSE); logits_right[0, 0] = 10.0
    l_wrong = focal_ce_loss(logits_wrong, target)
    l_right = focal_ce_loss(logits_right, target)
    assert l_right < l_wrong


def test_keys_focal_loss_shape_handling():
    B = 4
    logits = torch.randn(B, NUM_KEYS * 3)
    target = torch.full((B, NUM_KEYS), 2, dtype=torch.long)  # all idle
    loss = keys_focal_loss(logits, target, gamma=2.0, idle_smoothing=0.05)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_done_loss_shape_handling():
    logits = torch.randn(8, 1)
    target = torch.randint(0, 2, (8,))
    loss = done_loss(logits, target)
    assert torch.isfinite(loss)


def test_total_loss_all_heads_contribute():
    B = 4
    head_logits = {
        "dx":     torch.randn(B, NUM_BINS_MOUSE),
        "dy":     torch.randn(B, NUM_BINS_MOUSE),
        "click":  torch.randn(B, NUM_CLICK_EVENTS),
        "scroll": torch.randn(B, NUM_BINS_MOUSE),
        "keys":   torch.randn(B, NUM_KEYS * 3),
        "done":   torch.randn(B, 1),
    }
    targets = {
        "dx":     torch.randint(0, NUM_BINS_MOUSE, (B,)),
        "dy":     torch.randint(0, NUM_BINS_MOUSE, (B,)),
        "click":  torch.randint(0, NUM_CLICK_EVENTS, (B,)),
        "scroll": torch.randint(0, NUM_BINS_MOUSE, (B,)),
        "keys":   torch.full((B, NUM_KEYS), 2, dtype=torch.long),
        "done":   torch.randint(0, 2, (B,)),
    }
    weights = {n: 1.0 for n in head_logits}
    total, per_head = total_loss(head_logits, targets, weights)
    assert torch.isfinite(total)
    assert set(per_head.keys()) == set(head_logits.keys())
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/action_primitives/test_losses.py -v`

Expected: 5 tests pass.

- [ ] **Step 4: Commit**

```bash
git add experiments/action_primitives/losses.py tests/action_primitives/test_losses.py
git commit -m "feat(exp6): focal CE / keys / done loss functions with tests"
```

---

### Task 16: Parquet dataloader

**Files:**
- Create: `experiments/action_primitives/dataset.py`
- Create: `tests/action_primitives/test_dataset.py`

Streams parquet shards; groups episodes into micro-batches of N same-primitive episodes; quantizes continuous deltas to bins; constructs composite action-history vectors on-the-fly.

- [ ] **Step 1: Write dataset.py**

Create `experiments/action_primitives/dataset.py`:

```python
"""Parquet dataloader for Experiment 6 Phase A.

Each parquet shard has rows = one frame. Episodes identified by `episode_id`.
Output samples are (B, T, ...) tensors where T = episode_window.
Micro-batch homogeneity: all episodes in a micro-batch are the same primitive.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from experiments.action_primitives.config import (
    ENV,
    NUM_BINS_MOUSE,
    NUM_CLICK_EVENTS,
    NUM_KEYS,
    MODEL,
    MOUSE_BIN_CENTERS,
    SCROLL_BIN_CENTERS,
)
from experiments.action_primitives.history import HISTORY_INPUT_DIM

assert HISTORY_INPUT_DIM == 223  # sanity: 21+21+5+21+77+77+1 = 223


def quantize_to_bin(value: float, centers: np.ndarray) -> int:
    """Nearest-bin by L1 distance."""
    return int(np.argmin(np.abs(centers - value)))


def build_action_history_vector(prev_actions: list[dict]) -> np.ndarray:
    """Given list of K prior action-dicts (oldest→newest), build (K, 223) composite.
    Each dict has dx_bin, dy_bin, click, scroll_bin, key_events, done_gt.
    """
    K = len(prev_actions)
    out = np.zeros((K, HISTORY_INPUT_DIM), dtype=np.float32)
    for k, a in enumerate(prev_actions):
        offset = 0
        out[k, offset + a["dx_bin"]] = 1.0; offset += NUM_BINS_MOUSE
        out[k, offset + a["dy_bin"]] = 1.0; offset += NUM_BINS_MOUSE
        out[k, offset + a["click"]] = 1.0; offset += NUM_CLICK_EVENTS
        out[k, offset + a["scroll_bin"]] = 1.0; offset += NUM_BINS_MOUSE  # scroll uses 21 bins too
        # press mask (77)
        press_mask = (np.array(a["key_events"]) == 0).astype(np.float32)
        out[k, offset:offset + NUM_KEYS] = press_mask; offset += NUM_KEYS
        # release mask (77)
        release_mask = (np.array(a["key_events"]) == 1).astype(np.float32)
        out[k, offset:offset + NUM_KEYS] = release_mask; offset += NUM_KEYS
        # done (1)
        out[k, offset] = float(a.get("done_gt", 0)); offset += 1
    return out


def decode_jpeg_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


class PhaseAEpisodeDataset(Dataset):
    """Groups parquet rows into per-episode lists; returns (T, ...) per episode."""

    def __init__(self, data_dir: Path, split: str = "train") -> None:
        ds = load_dataset("parquet", data_files=str(Path(data_dir) / "shard_*.parquet"))["train"]
        # Simple split by episode_id hash
        def split_fn(ex):
            eid = ex["episode_id"]
            bucket = eid % 10
            if split == "train":
                return bucket < 8
            if split == "val":
                return bucket == 8
            if split == "test":
                return bucket == 9
            return False
        self.ds = ds.filter(split_fn)
        # Build episode index: episode_id → list of row indices
        self.episode_index: dict[int, list[int]] = {}
        for i, ex in enumerate(self.ds):
            self.episode_index.setdefault(ex["episode_id"], []).append(i)
        self.episode_ids = sorted(self.episode_index.keys())

    def __len__(self) -> int:
        return len(self.episode_ids)

    def __getitem__(self, idx: int) -> dict:
        eid = self.episode_ids[idx]
        row_indices = self.episode_index[eid]
        frames = [self.ds[i] for i in row_indices]
        # Decode images
        images = [decode_jpeg_bytes(f["image_bytes"]) for f in frames]
        # Quantize continuous actions to bin indices
        dx_bins = [quantize_to_bin(f["action_dx"], MOUSE_BIN_CENTERS) for f in frames]
        dy_bins = [quantize_to_bin(f["action_dy"], MOUSE_BIN_CENTERS) for f in frames]
        scroll_bins = [quantize_to_bin(f["action_scroll"], SCROLL_BIN_CENTERS) for f in frames]
        clicks = [f["action_click"] for f in frames]
        key_events_per_frame = [np.asarray(f["action_key_events"], dtype=np.int64) for f in frames]
        dones = [f["done_gt"] for f in frames]
        # Proprio (83-dim) per frame
        proprio_per_frame = []
        for f in frames:
            proprio_per_frame.append(np.concatenate([
                np.array([f["cursor_x"], f["cursor_y"]], dtype=np.float32),
                np.asarray(f["held_keys"], dtype=np.float32),
                np.asarray(f["held_mouse"], dtype=np.float32),
                np.array([f["capslock"]], dtype=np.float32),
            ]))
        # Build action-history vectors for every frame (K=8 prior)
        K = MODEL.action_history_len
        history_per_frame = []
        zero_action = {"dx_bin": NUM_BINS_MOUSE // 2, "dy_bin": NUM_BINS_MOUSE // 2,
                       "click": 0, "scroll_bin": NUM_BINS_MOUSE // 2,
                       "key_events": [2] * NUM_KEYS, "done_gt": 0}
        for t in range(len(frames)):
            prev = []
            for k in range(K, 0, -1):
                j = t - k
                if j < 0:
                    prev.append(zero_action)
                else:
                    prev.append({
                        "dx_bin": dx_bins[j], "dy_bin": dy_bins[j],
                        "click": clicks[j], "scroll_bin": scroll_bins[j],
                        "key_events": key_events_per_frame[j].tolist(),
                        "done_gt": dones[j],
                    })
            history_per_frame.append(build_action_history_vector(prev))
        # Instruction: for L-click only, use a single fixed string in Phase A
        instruction = f"click the {frames[0].get('theme', 'flat-modern')} button"
        return {
            "images": images,                                                     # list[PIL] length T
            "proprio": torch.from_numpy(np.stack(proprio_per_frame)),             # (T, 83)
            "history": torch.from_numpy(np.stack(history_per_frame)).float(),     # (T, K, 223)
            "dx_bins": torch.tensor(dx_bins, dtype=torch.long),                   # (T,)
            "dy_bins": torch.tensor(dy_bins, dtype=torch.long),                   # (T,)
            "clicks": torch.tensor(clicks, dtype=torch.long),                     # (T,)
            "scroll_bins": torch.tensor(scroll_bins, dtype=torch.long),           # (T,)
            "key_events": torch.from_numpy(np.stack(key_events_per_frame)),       # (T, 77)
            "dones": torch.tensor(dones, dtype=torch.long),                       # (T,)
            "instruction": instruction,
        }
```

- [ ] **Step 2: Write dataset tests**

Create `tests/action_primitives/test_dataset.py`:

```python
"""Dataset unit tests. Requires generating a tiny dataset first."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    """Generate 10 episodes into a temp parquet directory."""
    from experiments.action_primitives.generate_data import generate_all

    d = tmp_path_factory.mktemp("phase-a-tiny")
    generate_all(n_episodes=10, out_dir=d, shard_size=5)
    return d


def test_dataset_returns_per_episode_dict(tiny_dataset):
    from experiments.action_primitives.dataset import PhaseAEpisodeDataset

    ds = PhaseAEpisodeDataset(tiny_dataset, split="train")
    # 80% of 10 = 8 episodes (bucket 0..7)
    assert len(ds) > 0
    ep = ds[0]
    assert "images" in ep
    assert "proprio" in ep
    assert "history" in ep
    assert ep["proprio"].shape[-1] == 83
    assert ep["history"].shape[-1] == 223
    T = len(ep["images"])
    assert ep["dx_bins"].shape == (T,)
    assert ep["key_events"].shape == (T, 77)


def test_dataset_splits_disjoint_by_episode_id(tiny_dataset):
    from experiments.action_primitives.dataset import PhaseAEpisodeDataset

    tr = PhaseAEpisodeDataset(tiny_dataset, split="train")
    val = PhaseAEpisodeDataset(tiny_dataset, split="val")
    tr_ids = set(tr.episode_ids)
    val_ids = set(val.episode_ids)
    assert not (tr_ids & val_ids)
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/action_primitives/test_dataset.py -v`

Expected: 2 tests pass.

- [ ] **Step 4: Commit**

```bash
git add experiments/action_primitives/dataset.py tests/action_primitives/test_dataset.py
git commit -m "feat(exp6): parquet dataset with action-history vector construction"
```

---

### Task 17: Training loop

**Files:**
- Create: `experiments/action_primitives/train.py`

AdamW, cosine LR with warmup, micro-batch grad accumulation (8 × 8 = 64), bf16 mixed precision, wandb logging, checkpointing every N steps.

- [ ] **Step 1: Write train.py**

Create `experiments/action_primitives/train.py`:

```python
"""Phase A training loop for the action-primitives ACT model.

Trains on L-click episodes only. Micro-batch grad accumulation: 8 micro-batches
of 8 episodes each = macro-batch 64 (Q2/Q8 reconciliation).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.action_primitives.config import MODEL, TRAIN
from experiments.action_primitives.dataset import PhaseAEpisodeDataset
from experiments.action_primitives.losses import total_loss
from experiments.action_primitives.model import ActionPrimitivesACT


def cosine_lr(step: int, max_steps: int, warmup_steps: int, base_lr: float, min_frac: float = 0.1) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_frac + (1 - min_frac) * cos)


def infinite_loader(ds: PhaseAEpisodeDataset, seed: int = 0) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(len(ds))
        for i in perm:
            yield ds[int(i)]


def flatten_episode_to_frames(ep: dict) -> dict:
    """Turn (T, ...) per-episode tensors into (T, ...) frame tensors for loss."""
    return {
        "images": ep["images"],                                       # list[PIL] len T
        "proprio": ep["proprio"].float(),                             # (T, 83)
        "history": ep["history"].float(),                             # (T, K, 223)
        "dx":     ep["dx_bins"], "dy":     ep["dy_bins"],
        "click":  ep["clicks"],  "scroll": ep["scroll_bins"],
        "keys":   ep["key_events"], "done":  ep["dones"],
        "instruction": ep["instruction"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=TRAIN.phase_a_epochs)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="checkpoints/phase-a")
    parser.add_argument("--wandb-project", type=str, default="cu-vla-exp6")
    parser.add_argument("--wandb-run-name", type=str, default="phase-a-lclick")
    parser.add_argument("--hf-upload-repo", type=str, default=None, help="HF repo for checkpoint upload")
    parser.add_argument("--hf-data-repo", type=str, default=None, help="HF dataset repo; overrides --data-dir")
    parser.add_argument("--resume", type=str, default=None, help="path to .pt checkpoint to resume from")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Data
    if args.hf_data_repo is not None:
        from experiments.action_primitives.hf_sync import download_hf_dataset
        data_dir = download_hf_dataset(args.hf_data_repo)
    else:
        data_dir = Path(args.data_dir)
    train_ds = PhaseAEpisodeDataset(data_dir, split="train")
    val_ds = PhaseAEpisodeDataset(data_dir, split="val")
    print(f"train episodes: {len(train_ds)}   val: {len(val_ds)}")

    # Model
    model = ActionPrimitivesACT().to(device)
    print(model.trainable_parameters_summary())

    # Optimizer: two param groups per Q20
    trunk_params = list(model.proprio_enc.parameters()) + list(model.history_enc.parameters()) \
                   + list(model.trunk.parameters()) + list(model.heads.parameters())
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    opt = AdamW(
        [
            {"params": trunk_params, "lr": TRAIN.lr_trunk},
            {"params": lora_params,  "lr": TRAIN.lr_lora},
        ],
        betas=(TRAIN.beta1, TRAIN.beta2),
        weight_decay=TRAIN.weight_decay,
    )
    # Estimate max_steps
    steps_per_epoch = math.ceil(len(train_ds) / TRAIN.macro_batch_episodes)
    max_steps = steps_per_epoch * args.epochs
    print(f"steps/epoch: {steps_per_epoch}  total: {max_steps}")

    # Init per-head weights as placeholder 1.0; re-measure after first batch
    head_weights = {"dx": 1.0, "dy": 1.0, "click": 1.0, "scroll": 1.0, "keys": 1.0, "done": 1.0}

    # bf16 amp
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    scaler = None  # bf16 doesn't need gradient scaler

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config={
        "epochs": args.epochs, "macro_batch": TRAIN.macro_batch_episodes, "lr_trunk": TRAIN.lr_trunk,
        "lr_lora": TRAIN.lr_lora, "model": MODEL.vision_model, "max_num_patches": MODEL.max_num_patches,
    })

    # Resume
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"resumed from step {start_step}")

    iterator = infinite_loader(train_ds, seed=42)
    model.train()
    step = start_step
    while step < max_steps:
        # One macro-batch = 8 micro-batches × 8 episodes
        opt.zero_grad(set_to_none=True)
        step_per_head_loss = {k: 0.0 for k in ("dx", "dy", "click", "scroll", "keys", "done")}
        for _micro in range(TRAIN.macro_batch_episodes // TRAIN.micro_batch_episodes):
            micro_eps = [next(iterator) for _ in range(TRAIN.micro_batch_episodes)]
            # Concatenate frames across the micro-batch
            with autocast:
                flat_images: list = []
                flat_proprio: list = []
                flat_history: list = []
                flat_targets = {k: [] for k in ("dx", "dy", "click", "scroll", "keys", "done")}
                # Pre-encode text (single string per episode in Phase A; cache across frames)
                instructions = [e["instruction"] for e in micro_eps]
                with torch.no_grad():
                    text_tokens = model.backbone.encode_text(instructions)    # (M, T_text, d)
                # For each episode, append all its frames
                for ep_idx, ep in enumerate(micro_eps):
                    flat = flatten_episode_to_frames(ep)
                    T = len(flat["images"])
                    flat_images.extend(flat["images"])
                    flat_proprio.append(flat["proprio"])
                    flat_history.append(flat["history"])
                    for k in flat_targets:
                        flat_targets[k].append(flat[k])
                    # Repeat text tokens per frame of this episode
                    if ep_idx == 0:
                        text_rep = text_tokens[ep_idx:ep_idx+1].expand(T, -1, -1)
                        text_mask_rep = torch.ones(T, text_tokens.size(1), device=device)
                    else:
                        text_rep = torch.cat([text_rep, text_tokens[ep_idx:ep_idx+1].expand(T, -1, -1)], dim=0)
                        text_mask_rep = torch.cat([text_mask_rep, torch.ones(T, text_tokens.size(1), device=device)], dim=0)
                proprio = torch.cat(flat_proprio, dim=0).to(device)
                history = torch.cat(flat_history, dim=0).to(device)
                targets = {k: torch.cat(flat_targets[k], dim=0).to(device) for k in flat_targets}
                targets["done"] = targets["done"].float()

                out = model(flat_images, text_rep, text_mask_rep, proprio, history)
                loss, per_head = total_loss(out.head_logits, targets, head_weights)

            (loss / TRAIN.macro_batch_episodes * TRAIN.micro_batch_episodes).backward()
            for k, v in per_head.items():
                step_per_head_loss[k] += float(v.detach()) / (TRAIN.macro_batch_episodes // TRAIN.micro_batch_episodes)

        # Apply LR schedule
        for pg, base in zip(opt.param_groups, [TRAIN.lr_trunk, TRAIN.lr_lora]):
            pg["lr"] = cosine_lr(step, max_steps, TRAIN.warmup_steps, base, TRAIN.cosine_min_lr_frac)

        # Gradient clip and step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN.grad_clip_norm)
        opt.step()

        # Log
        total_loss_avg = sum(step_per_head_loss.values())
        log = {"step": step, "loss/total": total_loss_avg, "grad_norm": float(grad_norm),
               "lr/trunk": opt.param_groups[0]["lr"], "lr/lora": opt.param_groups[1]["lr"]}
        for k, v in step_per_head_loss.items():
            log[f"loss/{k}"] = v
        wandb.log(log)
        if step % 20 == 0:
            print(f"[step {step}] loss={total_loss_avg:.4f}  grad_norm={grad_norm:.3f}")

        # Checkpoint
        if step % TRAIN.ckpt_every_steps == 0 and step > 0:
            ckpt_path = out_dir / f"step_{step:05d}.pt"
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step}, ckpt_path)
            print(f"[ckpt] {ckpt_path}")
            if args.hf_upload_repo:
                try:
                    from huggingface_hub import upload_file
                    upload_file(path_or_fileobj=str(ckpt_path), path_in_repo=ckpt_path.name, repo_id=args.hf_upload_repo, repo_type="model")
                except Exception as e:
                    print(f"[ckpt upload failed] {e}")

        step += 1

    # Final checkpoint
    final_path = out_dir / "final.pt"
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step}, final_path)
    print(f"done, wrote {final_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit training loop**

```bash
git add experiments/action_primitives/train.py
git commit -m "feat(exp6): training loop with AdamW + cosine + micro-batch grad accum"
```

---

### Task 18: HF Jobs launcher

**Files:**
- Create: `scripts/hf_job_train_exp6.py`
- Create: `scripts/launch_hf_job_exp6.py`
- Create: `experiments/action_primitives/hf_sync.py`

Adapts the existing `scripts/launch_hf_job.py` + `scripts/hf_job_train.py` pattern for exp6. The UV script inside HF Jobs clones the repo and runs `python -m experiments.action_primitives.train ...`.

- [ ] **Step 1: Write the HF Jobs UV script**

Create `scripts/hf_job_train_exp6.py` (this runs *inside* HF Jobs):

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "torchvision",
#   "transformers",
#   "datasets",
#   "peft",
#   "accelerate",
#   "pygame",
#   "wandb",
#   "huggingface_hub",
#   "pillow",
#   "numpy",
# ]
# ///
"""HF Jobs training entrypoint for Experiment 6 Phase A.

Clones the CU-VLA repo, downloads the dataset from HF Hub (if --hf-data-repo set),
and runs experiments.action_primitives.train.
"""
from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    # Clone repo
    repo_url = os.environ.get("CU_VLA_REPO_URL", "https://github.com/PenTest-duck/CU-VLA.git")
    subprocess.run(["git", "clone", "--depth", "1", repo_url, "/workspace/CU-VLA"], check=True)
    os.chdir("/workspace/CU-VLA")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
    # Forward all args to the training script
    subprocess.run([sys.executable, "-m", "experiments.action_primitives.train", *sys.argv[1:]], check=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the HF Jobs launcher**

Create `scripts/launch_hf_job_exp6.py`:

```python
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
```

- [ ] **Step 3: Write hf_sync.py for data upload/download**

Create `experiments/action_primitives/hf_sync.py`:

```python
"""HF Hub sync helpers for Experiment 6 Phase A."""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def upload_parquet_dir(local_dir: Path, repo_id: str, repo_type: str = "dataset") -> None:
    api = HfApi()
    api.upload_folder(folder_path=str(local_dir), repo_id=repo_id, repo_type=repo_type)


def download_hf_dataset(repo_id: str, local_dir: str = "data/hf-download") -> Path:
    path = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)
    return Path(path)
```

- [ ] **Step 4: Commit launcher**

```bash
git add scripts/hf_job_train_exp6.py scripts/launch_hf_job_exp6.py experiments/action_primitives/hf_sync.py
git commit -m "feat(exp6): HF Jobs launcher + training entry + hf_sync helpers"
```

---

### Task 19: Checkpoint resume smoke test

**Files:**
- Create: `tests/action_primitives/test_resume.py`

- [ ] **Step 1: Write resume test (integration-level, CPU-only, tiny)**

Create `tests/action_primitives/test_resume.py`:

```python
"""Integration smoke test: generate 20 episodes, train 4 steps, save, resume, train 4 more steps.
Verifies loss curves are continuous across the restart (no huge jump).
"""
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def test_resume_smoke(tmp_path):
    data_dir = tmp_path / "data"
    ckpt_dir = tmp_path / "ckpt"

    # 1. Generate 20 episodes
    subprocess.run([sys.executable, "-m", "experiments.action_primitives.generate_data",
                    "-n", "20", "-o", str(data_dir), "--shard-size", "10"], check=True)

    # 2. Train for 4 steps
    subprocess.run([sys.executable, "-m", "experiments.action_primitives.train",
                    "--data-dir", str(data_dir), "--epochs", "1",
                    "--device", "cpu", "--out-dir", str(ckpt_dir)], check=True)

    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    assert len(ckpts) > 0, "no checkpoint saved"

    # 3. Resume from checkpoint, train 1 more step
    subprocess.run([sys.executable, "-m", "experiments.action_primitives.train",
                    "--data-dir", str(data_dir), "--epochs", "1",
                    "--device", "cpu", "--out-dir", str(ckpt_dir),
                    "--resume", str(ckpts[-1])], check=True)

    assert (ckpt_dir / "final.pt").exists()
```

- [ ] **Step 2: Run slow resume test**

Run: `uv run pytest tests/action_primitives/test_resume.py -v -m slow`

Expected: 1 test passes (may take several minutes on CPU — first SigLIP2 download + slow forward passes).

- [ ] **Step 3: Commit**

```bash
git add tests/action_primitives/test_resume.py
git commit -m "test(exp6): checkpoint-resume smoke test"
```

---

## Sub-Phase 6: Eval

### Task 20: Offline eval (per-head accuracy on val)

**Files:**
- Create: `experiments/action_primitives/evaluate.py`

- [ ] **Step 1: Write evaluate.py offline-eval section**

Create `experiments/action_primitives/evaluate.py`:

```python
"""Phase A evaluation: offline per-head accuracy + closed-loop L-click success."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.action_primitives.config import (
    ENV, HEAD_LOGITS, MODEL, MOUSE_BIN_CENTERS, NUM_KEYS, PROPRIO_DIM,
)
from experiments.action_primitives.dataset import PhaseAEpisodeDataset, build_action_history_vector, decode_jpeg_bytes
from experiments.action_primitives.env import Action, LClickEnv
from experiments.action_primitives.history import HISTORY_INPUT_DIM
from experiments.action_primitives.model import ActionPrimitivesACT


def load_model(ckpt_path: str, device: str) -> ActionPrimitivesACT:
    model = ActionPrimitivesACT().to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model


def offline_eval(model: ActionPrimitivesACT, data_dir: Path, device: str) -> dict:
    """Per-head top-1 accuracy on val set."""
    val_ds = PhaseAEpisodeDataset(data_dir, split="val")
    correct = {k: 0 for k in HEAD_LOGITS}
    total = 0
    for idx in range(len(val_ds)):
        ep = val_ds[idx]
        T = len(ep["images"])
        with torch.no_grad():
            text_tokens = model.backbone.encode_text([ep["instruction"]])  # (1, T_text, d)
            text_rep = text_tokens.expand(T, -1, -1)
            text_mask = torch.ones(T, text_tokens.size(1), device=device)
            out = model(ep["images"], text_rep, text_mask,
                        ep["proprio"].to(device), ep["history"].to(device).float())
        for head in HEAD_LOGITS:
            if head == "keys":
                logits = out.head_logits[head].view(T, NUM_KEYS, 3)
                preds = logits.argmax(dim=-1)  # (T, 77)
                tgt = ep["key_events"].to(device)
                correct[head] += int((preds == tgt).sum())
                total_for_head = T * NUM_KEYS
            elif head == "done":
                preds = (torch.sigmoid(out.head_logits[head].squeeze(-1)) > 0.5).long()
                tgt = ep["dones"].to(device)
                correct[head] += int((preds == tgt).sum())
                total_for_head = T
            else:
                preds = out.head_logits[head].argmax(dim=-1)
                tgt = ep[f"{head}_bins" if head in ("dx", "dy", "scroll") else "clicks" if head == "click" else "dones"].to(device)
                correct[head] += int((preds == tgt).sum())
                total_for_head = T
        total += T
    return {k: correct[k] / max(1, total * (NUM_KEYS if k == "keys" else 1)) for k in correct}


def rollout_one_episode(model: ActionPrimitivesACT, env: LClickEnv, device: str, max_frames: int = ENV.max_frames_lclick) -> dict:
    """Closed-loop rollout. Returns info dict with success flag."""
    obs, info = env.reset()
    with torch.no_grad():
        text_tokens = model.backbone.encode_text([f"click the {env.theme} button" if False else "click the button"])
    K = MODEL.action_history_len
    history = np.zeros((K, HISTORY_INPUT_DIM), dtype=np.float32)
    for t in range(max_frames):
        prop = np.concatenate([
            [obs["proprio"].cursor_x, obs["proprio"].cursor_y],
            obs["proprio"].held_keys.astype(np.float32),
            obs["proprio"].held_mouse.astype(np.float32),
            [float(obs["proprio"].capslock)],
        ])
        proprio_t = torch.from_numpy(prop).float().unsqueeze(0).to(device)
        history_t = torch.from_numpy(history).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out = model([obs["image"]], text_tokens, torch.ones(1, text_tokens.size(1), device=device), proprio_t, history_t)
        # Argmax decode
        dx_bin = int(out.head_logits["dx"].argmax(dim=-1))
        dy_bin = int(out.head_logits["dy"].argmax(dim=-1))
        dx = float(MOUSE_BIN_CENTERS[dx_bin])
        dy = float(MOUSE_BIN_CENTERS[dy_bin])
        click = int(out.head_logits["click"].argmax(dim=-1))
        # Apply
        action = Action(dx=dx, dy=dy, click=click)
        obs, done, info = env.step(action)
        # Update history (oldest-first)
        new_hist = build_action_history_vector([{
            "dx_bin": dx_bin, "dy_bin": dy_bin, "click": click, "scroll_bin": 10,
            "key_events": [2] * NUM_KEYS, "done_gt": int(done),
        }])
        history = np.concatenate([history[1:], new_hist], axis=0)
        if done:
            return {"success": True, "frames": t + 1}
    return {"success": False, "frames": max_frames}


def closed_loop_eval(model: ActionPrimitivesACT, device: str, n_episodes: int = 200, tolerances_px: list[int] = [0, 3, 5, 10]) -> dict:
    """Rollout n_episodes of L-click; report binary success + tolerance curves."""
    results = []
    for i in range(n_episodes):
        env = LClickEnv(seed=10000 + i)
        res = rollout_one_episode(model, env, device)
        # For tolerance curves, we check cursor distance to target center at episode end
        x, y, w, h = env._info()["target_bbox"]
        tx, ty = x + w / 2, y + h / 2
        cx, cy = env.cursor_x, env.cursor_y
        dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
        res["dist_px"] = dist
        results.append(res)
    out = {"n_episodes": n_episodes, "success_rate": sum(r["success"] for r in results) / n_episodes}
    for tol in tolerances_px:
        out[f"click_within_{tol}px"] = sum(1 for r in results if r["dist_px"] <= tol) / n_episodes
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/phase-a-lclick")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    parser.add_argument("--n-rollouts", type=int, default=200)
    parser.add_argument("--skip-offline", action="store_true")
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)
    if not args.skip_offline:
        print("=== offline eval ===")
        off = offline_eval(model, Path(args.data_dir), args.device)
        for k, v in off.items():
            print(f"  {k}: {v:.4f}")
    print("=== closed-loop eval ===")
    cl = closed_loop_eval(model, args.device, n_episodes=args.n_rollouts)
    for k, v in cl.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit eval**

```bash
git add experiments/action_primitives/evaluate.py
git commit -m "feat(exp6): offline + closed-loop eval with tolerance curves"
```

---

## Sub-Phase 7: Spike B — L-click end-to-end

### Task 21: Generate Spike B data

**Files:**
- Create (via script): `data/phase-a-lclick/shard_*.parquet`

- [ ] **Step 1: Generate 3000 L-click episodes**

Run:

```bash
uv run python -m experiments.action_primitives.generate_data -n 3000 -o data/phase-a-lclick --shard-size 500
```

Expected output: 6 parquet shards (~500 ep each), ~90K frames total. Prints throughput.

- [ ] **Step 2: Upload to HF Hub**

Run (Python in repl or one-liner):

```bash
uv run python -c "
from experiments.action_primitives.hf_sync import upload_parquet_dir
from pathlib import Path
upload_parquet_dir(Path('data/phase-a-lclick'), 'PenTest-duck/cu-vla-exp6-phasea-lclick', repo_type='dataset')
"
```

Expected: upload progress bar; visible in HF Hub UI afterward.

- [ ] **Step 3: Commit a marker .gitkeep (don't commit the data)**

```bash
mkdir -p data/.keep && touch data/.keep/README.md
# Make sure data/*.parquet is already in .gitignore; if not, add it
grep -q "^data/" .gitignore || echo "data/" >> .gitignore
git add .gitignore
git commit -m "chore(exp6): gitignore data/ for generated parquet shards"
```

---

### Task 22: Launch Spike B training run

**Files:**
- Remote: HF Jobs run

- [ ] **Step 1: Launch training on HF Jobs**

Run (from repo root):

```bash
uv run python scripts/launch_hf_job_exp6.py \
    --flavor l4x1 --timeout 4h -- \
    --hf-data-repo PenTest-duck/cu-vla-exp6-phasea-lclick \
    --epochs 5 \
    --hf-upload-repo PenTest-duck/cu-vla-exp6-phasea-ckpt \
    --wandb-run-name phase-a-spike-b
```

Expected: job URL prints; logs stream. Wall-clock ~1.5–3h on L4 for 5 epochs × ~40 steps/epoch = ~200 steps.

- [ ] **Step 2: Monitor epoch-1 diagnostics (Q40)**

Watch the wandb run for:
- Loss decreasing monotonically, not diverging
- No NaN/Inf
- Grad norms stable, not hitting clip every step
- No single head dominating total loss
- Per-head sparsity stats look reasonable

Kill the run manually if any check fails; investigate before relaunching.

- [ ] **Step 3: Confirm training finished + final checkpoint uploaded**

Check that `PenTest-duck/cu-vla-exp6-phasea-ckpt` contains a `final.pt` or latest `step_*.pt`.

---

### Task 23: Evaluate Spike B checkpoint + write-up

**Files:**
- Create: `docs/experiments/6-action-primitives-phase-a-results/spike-b-lclick-end-to-end.md`

- [ ] **Step 1: Download checkpoint locally**

Run:

```bash
uv run python -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id='PenTest-duck/cu-vla-exp6-phasea-ckpt', filename='final.pt')
print(p)
"
```

- [ ] **Step 2: Run evaluation on the downloaded checkpoint**

Run (on M1):

```bash
uv run python -m experiments.action_primitives.evaluate \
    --checkpoint /path/to/final.pt \
    --data-dir data/phase-a-lclick \
    --n-rollouts 200 \
    --device mps
```

Expected output: offline per-head accuracies + closed-loop success rate + tolerance curves (0/3/5/10px).

- [ ] **Step 3: Write up Spike B results**

Create `docs/experiments/6-action-primitives-phase-a-results/spike-b-lclick-end-to-end.md`:

```markdown
# Spike B — L-click end-to-end

**Ran:** 2026-04-XX
**Checkpoint:** `PenTest-duck/cu-vla-exp6-phasea-ckpt:final.pt`
**Wandb:** <paste run URL>

## Training config
- Episodes: 3000 (80/10/10)
- Epochs: 5
- Hardware: L4 (1x)
- Wall-clock: X.Xh

## Offline (val, per-head top-1 accuracy)
| Head | Top-1 |
|---|---|
| dx | X.XX |
| dy | X.XX |
| click | X.XX |
| scroll | X.XX |
| keys | X.XX |
| done | X.XX |

## Closed-loop (200 rollouts)
- Success rate: **X.XX**
- Click within 0px: X.XX
- Click within 3px: X.XX
- Click within 5px: X.XX
- Click within 10px: X.XX

## Interpretation
- Spike B target: >50% closed-loop success after 5 epochs.
- Result: [pass / fail / borderline]

## Failure modes observed (if any)
- [Fill in]

## Recommendation
- [Proceed to Phase B as-is / adjust trunk capacity / adjust loss weighting / etc.]

## Next steps
- [ ] Pass → proceed to Spike C (M1 timing)
- [ ] Fail → diagnose; Q16 capacity question first
```

- [ ] **Step 4: Fill in actual numbers and recommendation**

- [ ] **Step 5: Commit**

```bash
git add docs/experiments/6-action-primitives-phase-a-results/spike-b-*
git commit -m "results(exp6): Spike B L-click end-to-end write-up"
```

- [ ] **Step 6: CHECKPOINT — user reviews Spike B write-up**

---

## Sub-Phase 8: Spike C — M1 eval timing

### Task 24: M1 wall-clock timing measurement

**Files:**
- Create: `experiments/action_primitives/measurements/m1_eval_timing.py`
- Create: `docs/experiments/6-action-primitives-phase-a-results/spike-c-m1-eval-timing.md`

- [ ] **Step 1: Write the timing script**

Create `experiments/action_primitives/measurements/m1_eval_timing.py`:

```python
"""Spike C — M1 closed-loop eval timing.

Runs N rollouts on M1 MPS with the Spike B checkpoint, reports per-frame
wall-clock breakdown (encoder, trunk, env, total) and aggregate eps/rollout.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.action_primitives.evaluate import load_model, rollout_one_episode
from experiments.action_primitives.env import LClickEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--out", type=str, default="docs/experiments/6-action-primitives-phase-a-results/spike-c-m1-eval-timing.json")
    args = parser.parse_args()

    assert torch.backends.mps.is_available(), "run on M1 with MPS available"
    device = "mps"
    model = load_model(args.checkpoint, device)

    # Warmup
    env = LClickEnv(seed=0)
    for _ in range(5):
        rollout_one_episode(model, env, device, max_frames=5)

    # Measure
    wall_times = []
    frame_counts = []
    t0_total = time.time()
    for i in range(args.n_rollouts):
        env = LClickEnv(seed=5000 + i)
        t0 = time.time()
        res = rollout_one_episode(model, env, device)
        wall_times.append(time.time() - t0)
        frame_counts.append(res["frames"])
    total_wall = time.time() - t0_total

    per_rollout_mean = float(np.mean(wall_times))
    per_rollout_median = float(np.median(wall_times))
    frames_total = int(np.sum(frame_counts))
    per_frame_mean_ms = total_wall / frames_total * 1000

    result = {
        "n_rollouts": args.n_rollouts,
        "total_wall_s": total_wall,
        "per_rollout_mean_s": per_rollout_mean,
        "per_rollout_median_s": per_rollout_median,
        "frames_total": frames_total,
        "per_frame_mean_ms": per_frame_mean_ms,
        "eff_hz": 1000.0 / per_frame_mean_ms,
    }
    print(json.dumps(result, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on M1**

Run (on M1 MacBook):

```bash
uv run python -m experiments.action_primitives.measurements.m1_eval_timing \
    --checkpoint /path/to/final.pt \
    --n-rollouts 100
```

Expected: prints JSON with per-frame ms + effective Hz.

- [ ] **Step 3: Write up Spike C**

Create `docs/experiments/6-action-primitives-phase-a-results/spike-c-m1-eval-timing.md`:

```markdown
# Spike C — M1 closed-loop eval timing

**Ran on:** M1 MacBook Pro, PyTorch MPS, 2026-04-XX
**Checkpoint:** Spike B final
**Raw JSON:** `spike-c-m1-eval-timing.json`

## Results
- n_rollouts: 100
- Total wall-clock: X.Xs
- Per-rollout mean: X.XXs
- Per-rollout median: X.XXs
- Per-frame mean: XXX ms
- **Effective Hz: X.X**
- Per Q16 projection: ~7.6 Hz; measured is [better/same/worse].

## Interpretation
- Q25 Tier-B budget (~400 rollouts every 2000 steps) needs ~X minutes at measured Hz.
- Phase B Tier-B cadence feasibility: [yes, as-designed / reduce to every 4000 steps / trim rollout count to 200].

## Recommendation
- [Phase B eval-cadence adjustment if needed]
```

- [ ] **Step 4: Commit**

```bash
git add experiments/action_primitives/measurements/m1_eval_timing.py docs/experiments/6-action-primitives-phase-a-results/spike-c-*
git commit -m "feat+results(exp6): Spike C M1 closed-loop eval timing measurement + write-up"
```

- [ ] **Step 5: CHECKPOINT — user reviews Spike C write-up**

---

## Sub-Phase 9: Phase A close-out

### Task 25: Phase A summary + Phase B handoff

**Files:**
- Create: `docs/experiments/6-action-primitives-phase-a-results/PHASE-A-SUMMARY.md`

- [ ] **Step 1: Write the summary document**

Create `docs/experiments/6-action-primitives-phase-a-results/PHASE-A-SUMMARY.md`:

```markdown
# Phase A — Summary & Phase B Handoff

Consolidates all four spikes. Drives the Phase B implementation plan.

## Results at a glance
| Spike | Result | Phase B impact |
|---|---|---|
| A (typing legibility) | F1 @ 14pt = X.XX | [none / bump max_patches / revise Q5 feedback mechanism] |
| B (L-click end-to-end) | X.XX closed-loop success | [pipeline validated / capacity up / diagnose] |
| C (M1 eval timing) | X.X Hz | [Tier-B as-designed / reduce cadence / ...] |
| E (gen throughput) | X.X eps/s | [as-designed / multi-process / smaller shard] |

## Design-doc amendments (Phase B)
- [Any Q# reopenings / parameter changes / scope tweaks triggered by results]

## Next step
Write Phase B implementation plan (`docs/plans/2026-04-XX-action-primitives-phase-b-implementation.md`), incorporating the amendments above.
```

- [ ] **Step 2: Fill in based on all four spike write-ups**

- [ ] **Step 3: Commit**

```bash
git add docs/experiments/6-action-primitives-phase-a-results/PHASE-A-SUMMARY.md
git commit -m "docs(exp6): Phase A summary + Phase B handoff"
```

- [ ] **Step 4: CHECKPOINT — user reviews Phase A as a whole; decide on Phase B scope/changes before writing Phase B plan.**

---

## Self-review (plan)

All spec sections covered at Phase A scope:

| Design-doc concern | Plan task |
|---|---|
| Phase A structure | sub-phases + check-points |
| Q1 bin math + ±100px cap | Task 2 `config.py` + tests; Task 7 expert clip |
| Q2/Q8 micro-batch grad accum | Task 17 training loop |
| Q3 keyboard 77×3 heads | Tasks 12, 15, 16 |
| Q5 single-frame, 8-frame history | Tasks 10, 13, 16 |
| Q6 SigLIP2 naflex @ 256 + text cache | Task 3 |
| Q7 pygame env | Task 6 |
| Q9 L-click generator (minimal) | Tasks 7, 8 |
| Q10 minimal diversity (3 themes) | Task 6 theme dict |
| Q11 cursor rendering | Task 6 `_render_obs` |
| Q15 trunk architecture | Tasks 10, 11, 12, 13 |
| Q15 LoRA rank-8 | Task 14 |
| Q19 history MLP 300→256→768 | Task 10 (223-dim precise input) |
| Q20 AdamW + cosine + warmup + bf16 | Task 17 |
| Q21 micro-batch 8×8 + ckpt + resume | Tasks 17, 19 |
| Q24 Tier-1 + Tier-1.5 tolerance curves | Task 20 |
| Q27 probes deferred for Phase A | (Phase B) |
| Q35 HF Jobs only | Task 18 |
| Q40 epoch-1 diagnostics | Task 22 step 2 |
| Phase A Spike A (typing probe) | Tasks 3, 4, 5 |
| Phase A Spike B (end-to-end) | Tasks 6–8, 10–14, 15–17, 20, 21–23 |
| Phase A Spike C (M1 timing) | Task 24 |
| Phase A Spike E (gen throughput) | Task 9 |

**Placeholder scan:** plan steps that need user-filled data (spike write-ups) are clearly marked "[Fill in …]" — these are *deliberate* data-dependent fills, not planning placeholders.

**Type consistency:** `HISTORY_INPUT_DIM = 223` is used in `history.py`, `dataset.py`, `evaluate.py`. `HEAD_LOGITS` keys used uniformly: `dx`, `dy`, `click`, `scroll`, `keys`, `done`.

---

## Execution handoff

Plan complete. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** — execute tasks in this session with `executing-plans`, batch execution with checkpoints

The user will choose which approach in the next message.

