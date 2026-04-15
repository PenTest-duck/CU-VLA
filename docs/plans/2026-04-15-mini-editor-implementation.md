# Experiment 5: Mini Text Editor — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the ACT + trainable text encoder model, training loop, and evaluation pipeline for the mini text editor VLA task.

**Architecture:** ACT with ResNet18 (384×288 input, 108 spatial tokens, 2D PE) + trainable 2-layer text encoder (~4M params, MobileBERT-init embeddings) + focal BCE for keys. ~21M all-trainable params. See `docs/plans/2026-04-15-mini-editor-model-architecture.md`.

**Tech Stack:** PyTorch, HuggingFace transformers (MobileBERT tokenizer/embeddings), HuggingFace datasets, torchvision ResNet18, numpy, tqdm.

---

### Task 1: Backbone — ResNet18 at 384×288 with 2D PE

**Files:**
- Create: `experiments/mini_editor/backbones.py`

**Step 1: Implement ResNet18Backbone**

Adapt from `experiments/miniwob_pygame/backbones.py:ResNet18Backbone`. Key changes:
- Input is (B, 3, 288, 384) not (B, 3, 224, 224)
- Output is (B, 108, 256) not (B, 49, 256) — no adaptive pooling, natural 9×12 grid
- Add `SinusoidalPositionEncoding2D` that splits d_model into y-half and x-half

```python
# The backbone should:
# 1. Run ResNet18 features (remove avgpool+fc) on (B, 3, 288, 384)
# 2. Get (B, 512, 9, 12) feature map
# 3. Reshape to (B, 108, 512)
# 4. Project via Linear(512, 256) to (B, 108, 256)

# SinusoidalPositionEncoding2D should:
# 1. Take d_model, grid height H, grid width W
# 2. Build PE buffer of shape (1, H*W, d_model)
# 3. First d_model//2 dims encode row index (sinusoidal)
# 4. Second d_model//2 dims encode col index (sinusoidal)
# 5. forward(x, h, w) adds PE to (B, H*W, d_model) tensor
```

**Step 2: Smoke test**

```bash
uv run python -c "
from experiments.mini_editor.backbones import ResNet18Backbone, SinusoidalPositionEncoding2D
import torch
backbone = ResNet18Backbone()
x = torch.randn(2, 3, 288, 384)
out = backbone(x)
assert out.shape == (2, 108, 256), f'Expected (2,108,256), got {out.shape}'
pe = SinusoidalPositionEncoding2D(256, 9, 12)
out_pe = pe(out, 9, 12)
assert out_pe.shape == (2, 108, 256)
print('PASS: backbone (2,3,288,384) -> (2,108,256)')
"
```

**Step 3: Commit**

```bash
git add experiments/mini_editor/backbones.py
git commit -m "exp5: add ResNet18 backbone with 2D sinusoidal PE for 384x288 input"
```

---

### Task 2: Text Encoder — Trainable mini-transformer with MobileBERT-init

**Files:**
- Create: `experiments/mini_editor/text_encoder.py`

**Step 1: Implement TextEncoder**

The text encoder needs:
1. Load MobileBERT tokenizer from HuggingFace (`google/mobilebert-uncased`)
2. Build vocabulary trimming: identify which token IDs appear in instruction templates + corpus words, keep only those + special tokens ([PAD], [UNK], [CLS], [SEP])
3. Initialize embedding table from MobileBERT's 512-dim embeddings, projected to 256-dim via PCA (sklearn or manual SVD)
4. 2-layer transformer encoder at d=256, 4 heads, dim_ff=512, Pre-LN
5. 1D sinusoidal PE for token positions
6. Forward: tokenize → embed → PE → transformer → output (B, L, 256)

```python
class TextEncoder(nn.Module):
    """Trainable text encoder with MobileBERT-initialized embeddings.

    Tokenizes with MobileBERT's WordPiece tokenizer (trimmed vocab),
    embeds, applies 2-layer transformer, outputs per-token features.
    """
    def __init__(self, d_model=256, nhead=4, num_layers=2, ...): ...
    def forward(self, input_ids, attention_mask=None) -> torch.Tensor:
        """(B, L) input_ids -> (B, L, d_model) token features."""
        ...
```

**Important:** Vocabulary trimming and PCA initialization should happen in a `build_text_encoder()` factory function that:
1. Loads MobileBERT model temporarily to extract embeddings
2. Computes PCA projection matrix (512→256)
3. Builds the token ID mapping (original → trimmed)
4. Returns the TextEncoder with initialized weights
5. Deletes the MobileBERT model to free memory

**Step 2: Smoke test**

```bash
uv run python -c "
from experiments.mini_editor.text_encoder import build_text_encoder
encoder, tokenizer, token_map = build_text_encoder()
tokens = tokenizer(\"Click after the word 'fox'\", return_tensors='pt')
input_ids = tokens['input_ids']
# Remap to trimmed vocab
import torch
out = encoder(input_ids)
print(f'Input shape: {input_ids.shape}')
print(f'Output shape: {out.shape}')
assert out.shape[0] == 1
assert out.shape[2] == 256
print(f'Vocab size: {encoder.vocab_size}')
print('PASS: text encoder works')
"
```

**Step 3: Commit**

```bash
git add experiments/mini_editor/text_encoder.py
git commit -m "exp5: add trainable text encoder with MobileBERT-init embeddings"
```

---

### Task 3: ACT Model — Full architecture with V+L+A fusion

**Files:**
- Create: `experiments/mini_editor/model.py`

**Step 1: Implement ACT model**

Adapt from `experiments/miniwob_pygame/model.py`. Key changes:
- Vision: ResNet18Backbone (108 tokens, 2D PE) instead of 49 tokens with 1D PE
- Text: TextEncoder tokens concatenated into encoder sequence
- Proprio: 56-dim (up from 46), FiLM + token (same pattern)
- Action heads: 53 keys (up from 43), same dx/dy/mouse/pad heads
- Forward signature: `forward(images, proprio, input_ids, attention_mask) -> dict`

```python
class ACT(nn.Module):
    def __init__(self, chunk_size=10, d_model=256, proprio_dim=56):
        # 1. Vision backbone (ResNet18, 384x288, 108 tokens)
        # 2. Text encoder (trainable, ~4M)
        # 3. FiLM net (proprio -> scale/shift for vision)
        # 4. Proprio projection + learned PE
        # 5. 1D sinusoidal PE for text tokens (separate from 2D vision PE)
        # 6. Encoder (4L, d=256, 4 heads, Pre-LN)
        # 7. Decoder (4L, d=256, 4 heads, Pre-LN)
        # 8. Action heads: dx(49), dy(49), mouse(1), keys(53), pad(1)

    def forward(self, images, proprio, input_ids, attention_mask=None):
        # 1. backbone(images) -> (B, 108, 256)
        # 2. FiLM: modulate vision with proprio
        # 3. Add 2D PE to vision tokens
        # 4. text_encoder(input_ids) -> (B, L, 256), add 1D PE
        # 5. proprio_proj(proprio) -> (B, 1, 256), add learned PE
        # 6. concat [vision, proprio, text] -> (~124, 256)
        # 7. encoder(concat) -> memory
        # 8. decoder(queries, memory) -> (B, chunk, 256)
        # 9. heads -> dx_logits, dy_logits, mouse, keys, pad
```

**Step 2: Smoke test — forward pass**

```bash
uv run python -c "
from experiments.mini_editor.model import ACT, count_parameters
import torch
model = ACT(chunk_size=10)
images = torch.randn(2, 3, 288, 384)
proprio = torch.randn(2, 56)
input_ids = torch.randint(0, 100, (2, 12))
out = model(images, proprio, input_ids)
print(f'dx_logits: {out[\"dx_logits\"].shape}')  # (2, 10, 49)
print(f'dy_logits: {out[\"dy_logits\"].shape}')  # (2, 10, 49)
print(f'mouse_left: {out[\"mouse_left\"].shape}')  # (2, 10)
print(f'keys_held: {out[\"keys_held\"].shape}')  # (2, 10, 53)
print(f'pad_logits: {out[\"pad_logits\"].shape}')  # (2, 10)
print(f'Total params: {count_parameters(model, trainable_only=False):,}')
print(f'Trainable params: {count_parameters(model, trainable_only=True):,}')
print('PASS')
"
```

**Step 3: Commit**

```bash
git add experiments/mini_editor/model.py
git commit -m "exp5: add ACT model with V+L+A fusion (ResNet18 + text encoder + 53 keys)"
```

---

### Task 4: Training Loop

**Files:**
- Create: `experiments/mini_editor/train.py`

**Step 1: Implement training loop**

Adapt from `experiments/miniwob_pygame/train.py`. Key changes:
- Dataset loads instructions (string column) and tokenizes them
- Image resize from 512×384 to 384×288 in `__getitem__`
- Proprio is 56-dim (from dataset `proprio` column)
- Loss includes focal BCE for keys and BCE for pad
- Forward pass includes `input_ids` and `attention_mask`

Key components (adapt from Exp 3's `train.py`):
1. `ChunkDataset.__getitem__` returns (image, proprio, input_ids, attn_mask, dx_bins, dy_bins, mouse, keys, pad_mask)
2. `FocalBCELoss` class: standard focal loss with α and γ parameters
3. Training loop: same structure as Exp 3 (warmup+cosine, AMP, grad clip, early stopping, per-head loss tracking)
4. `predecode_images` adapted for 384×288 resolution (not 224×224)

**Image resize:** In `__getitem__`, resize from whatever the dataset stores (512×384) to 384×288 using `torchvision.transforms.functional.resize` or PIL resize.

**Instruction tokenization:** Pre-tokenize all instructions at dataset init time (they're strings in the parquet). Cache the token IDs and attention masks as numpy arrays.

Run commands:
```bash
# Local (small test):
uv run python -m experiments.mini_editor.train --epochs 5 --num-episodes 100

# Full (HF Jobs, L40S):
uv run python -m experiments.mini_editor.train --epochs 100 --device cuda --hf-data-repo PenTest-duck/cu-vla-mini-editor
```

**Step 2: Verify training starts and loss decreases**

```bash
uv run python -m experiments.mini_editor.train --epochs 2 --num-episodes 50 --device cpu
```

Expected: training starts, loss printed per epoch, checkpoint saved.

**Step 3: Commit**

```bash
git add experiments/mini_editor/train.py
git commit -m "exp5: add training loop with focal BCE for keys, AMP, warmup+cosine LR"
```

---

### Task 5: Evaluation Pipeline

**Files:**
- Create: `experiments/mini_editor/evaluate.py`

**Step 1: Implement evaluation**

Adapt from `experiments/miniwob_pygame/evaluate.py`. Key changes:
- `ACTAgent` wraps the new model with text encoder — needs instruction text at init
- `ACTAgent.act()` forward pass includes pre-tokenized instruction (cached once per episode)
- Temporal ensemble smoothing with 53 keys (up from 43), separate key decay
- `ExpertAgent` wraps the existing expert policy
- Per-operation success rate tracking (click, click+type, select_delete, replace)
- Report held-out vs training phrasing accuracy

```python
class ACTAgent:
    def __init__(self, checkpoint, device="cpu"):
        self.model = ACT(...)
        # load weights
        self._cached_text_tokens = None

    def set_instruction(self, instruction_text: str):
        """Tokenize and cache instruction for current episode."""
        self._cached_text_tokens = tokenize(instruction_text)

    def act(self, obs, proprio):
        # Forward with cached text tokens
        # Temporal ensemble smoothing (same as Exp 3)
        ...
```

Run commands:
```bash
uv run python -m experiments.mini_editor.evaluate --checkpoint path/to/best.pt
uv run python -m experiments.mini_editor.evaluate --visual  # with Pygame window
uv run python -m experiments.mini_editor.evaluate --model-only  # skip expert/random
```

**Step 2: Verify evaluation runs with expert**

```bash
uv run python -m experiments.mini_editor.evaluate -n 10
```

Expected: expert success rate ~95-100%, per-operation breakdown printed.

**Step 3: Commit**

```bash
git add experiments/mini_editor/evaluate.py
git commit -m "exp5: add evaluation pipeline with per-operation metrics and temporal ensemble"
```

---

### Task 6: Config Updates + HF Sync

**Files:**
- Modify: `experiments/mini_editor/config.py`
- Create: `experiments/mini_editor/hf_sync.py`

**Step 1: Add MODEL and CHUNK configs to config.py**

Add model/chunk/training hyperparameters that the new model.py and train.py reference:

```python
@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256
    nheads: int = 4
    encoder_layers: int = 4
    decoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    film_hidden_dim: int = 256
    proprio_dim: int = PROPRIO_DIM  # 56
    obs_h: int = 288  # model input height (resized from 384)
    obs_w: int = 384  # model input width (resized from 512)
    vision_grid_h: int = 9   # ResNet18 output at 288 height
    vision_grid_w: int = 12  # ResNet18 output at 384 width

@dataclass(frozen=True)
class ChunkConfig:
    default_chunk_size: int = 10
    ensemble_decay: float = 0.2
    key_decay: float = 0.8  # faster decay for keys to prevent stale presses

# Focal loss for keys
@dataclass(frozen=True)
class FocalLossConfig:
    gamma: float = 2.0
    alpha: float = 0.75  # positive class weight
```

**Step 2: Add hf_sync.py**

Copy from `experiments/miniwob_pygame/hf_sync.py`, adapt paths. Handles upload/download of data and checkpoints to HuggingFace Hub.

**Step 3: Commit**

```bash
git add experiments/mini_editor/config.py experiments/mini_editor/hf_sync.py
git commit -m "exp5: add model/chunk/focal loss configs and HF sync utilities"
```

---

### Task 7: Update AGENTS.md

**Files:**
- Modify: `AGENTS.md`

**Step 1: Update Exp 5 section in AGENTS.md**

Add the new files to the code layout table and update run commands:

```markdown
| `backbones.py` | ResNet18 backbone (384×288 input, 108 tokens, 2D sin PE) |
| `text_encoder.py` | Trainable 2L transformer, MobileBERT-init embeddings, ~4M params |
| `model.py` | `ACT` — V+L+A fusion, 108 vis + text + proprio tokens, 53 keys (~21M params) |
| `train.py` | BC training: focal BCE for keys, soft CE for bins, AMP, warmup+cosine |
| `evaluate.py` | Per-operation eval, temporal ensemble, held-out phrasing accuracy |
| `hf_sync.py` | Upload/download data and checkpoints to HuggingFace Hub |
```

**Step 2: Commit**

```bash
git add AGENTS.md
git commit -m "docs: update AGENTS.md with exp5 model/train/eval files"
```

---

## Task Dependency Graph

```
Task 6 (config) ──► Task 1 (backbone) ──► Task 3 (model) ──► Task 4 (train)
                                              ▲                     │
                    Task 2 (text encoder) ────┘               Task 5 (eval)
                                                                    │
                                                              Task 7 (docs)
```

**Recommended execution order:** 6 → 1 → 2 → 3 → 4 → 5 → 7

Tasks 1 and 2 are independent of each other (both depend on 6). Tasks 4 and 5 both depend on 3.
