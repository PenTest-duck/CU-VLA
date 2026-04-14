# Exp 2: Discrete Exponential Bins + FiLM Conditioning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace tanh regression dx/dy heads with 49-bin exponential discretization, drop CVAE, add FiLM conditioning on proprioception — fixing tanh saturation (model can't produce small deltas) and proprio being ignored (1/51 tokens in attention).

**Architecture:** ACT model with discrete exponential bins (49 per axis, alpha=3) for cursor deltas. Cross-entropy loss replaces L1. CVAE removed (bins handle multimodality natively). FiLM conditioning applies proprio-derived scale/shift to every vision token. Temporal ensemble blends softmax distributions and reads out expected value.

**Tech Stack:** PyTorch, HF datasets, ResNet18 backbone, HF Jobs (L40S GPU)

**Key references:**
- FDM-1 (si.inc/posts/fdm1/) — 49 exponential bins for mouse deltas
- OpenVLA — 256 uniform bins, CE loss, no CVAE
- Current codebase: `experiments/act_drag_label/{config,model,train,evaluate}.py`

---

## Bin Layout

49 bins per axis: 24 negative + 1 zero center + 24 positive.
Formula: `bin_center[i] = sign * (exp(alpha * i/N) - 1) / (exp(alpha) - 1) * max_delta_px`
where alpha=3.0, N=24, max_delta_px=50.

Near-zero resolution: +/-0.35px. Max bin width: 6.2px. Full range: +/-50px.

---

### Task 1: Add bin configuration to config.py

**Files:**
- Modify: `experiments/act_drag_label/config.py`

**Step 1: Add bin constants to ActionConfig**

Add `num_bins_per_side`, `bin_alpha` to `ActionConfig`, and add bin computation. Also remove CVAE-related config from `TrainConfig` (`kl_weight_max`, `kl_anneal_fraction`) and update loss weights.

```python
# In config.py, update ActionConfig:
@dataclass(frozen=True)
class ActionConfig:
    max_delta_px: float = 50.0
    num_key_classes: int = NUM_KEY_CLASSES
    num_bins_per_side: int = 24       # NEW: 24 negative + 1 zero + 24 positive = 49
    bin_alpha: float = 3.0            # NEW: exponential curve parameter

# In config.py, update TrainConfig — remove kl_weight_max, kl_anneal_fraction,
# adjust loss weights (click/key from 5.0 to 2.0 per earlier decision):
@dataclass(frozen=True)
class TrainConfig:
    # ... existing fields ...
    # REMOVE: kl_weight_max: float = 0.1
    # REMOVE: kl_anneal_fraction: float = 0.2
    loss_weight_click: float = 2.0    # CHANGED from 5.0
    loss_weight_key: float = 2.0      # CHANGED from 5.0
    loss_weight_pad: float = 1.0

# In config.py, update ModelConfig — remove latent_dim:
@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256
    encoder_layers: int = 4
    decoder_layers: int = 7
    nheads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    # REMOVE: latent_dim: int = 32
    film_hidden_dim: int = 128        # NEW: FiLM MLP hidden dimension
    num_vision_tokens: int = 49
    backbone_feature_dims: dict[str, int] = field(default_factory=lambda: {
        "resnet18": 512,
        "dinov2-vits14": 384,
        "siglip2-base": 768,
    })
```

**Step 2: Add bin computation utility**

Add a standalone function and precomputed constants at the bottom of config.py:

```python
# At the bottom of config.py, after all dataclass instantiations:
import numpy as np

def build_bin_centers() -> np.ndarray:
    """Precompute 49 exponential bin centers in pixel space.

    Returns: (49,) array: [neg_24, ..., neg_1, 0, pos_1, ..., pos_24]
    """
    n = ACTION.num_bins_per_side
    alpha = ACTION.bin_alpha
    max_px = ACTION.max_delta_px
    i = np.arange(1, n + 1, dtype=np.float64)
    pos = (np.exp(alpha * i / n) - 1) / (np.exp(alpha) - 1) * max_px
    centers = np.concatenate([-pos[::-1], [0.0], pos]).astype(np.float32)
    return centers

BIN_CENTERS = build_bin_centers()  # (49,) — shared constant
NUM_BINS = 2 * ACTION.num_bins_per_side + 1  # 49
```

**Step 3: Verify**

Run: `uv run python -c "from experiments.act_drag_label.config import BIN_CENTERS, NUM_BINS; print(f'Bins: {NUM_BINS}, range: [{BIN_CENTERS[0]:.1f}, {BIN_CENTERS[-1]:.1f}], center: {BIN_CENTERS[24]:.1f}')"`

Expected: `Bins: 49, range: [-50.0, 50.0], center: 0.0`

**Step 4: Commit**

```
git add experiments/act_drag_label/config.py
git commit -m "exp2: add discrete exponential bin config, drop CVAE config, add FiLM config"
```

---

### Task 2: Rewrite model.py — drop CVAE, add FiLM, add bin heads

**Files:**
- Modify: `experiments/act_drag_label/model.py`

This is the largest change. The model drops:
- CVAE encoder (`_encode_cvae`, `cvae_encoder`, `cls_embed`, `encoder_action_proj`, `encoder_proprio_proj`, `latent_proj`, `latent_out_proj`, `cvae_pos_enc`)
- `special_pos_embed` (was for proprio + latent tokens, now only proprio)
- `head_dx` / `head_dy` as single-output linear layers

The model adds:
- FiLM network: `nn.Sequential(Linear(31, 128), ReLU, Linear(128, d_model*2))` producing scale+shift
- `head_dx` / `head_dy` as `nn.Linear(d_model, NUM_BINS)` each
- Proprio position embedding simplified to single learned parameter

**Step 1: Rewrite the ACT class**

Replace the entire `ACT.__init__` and `forward` methods. The new forward signature drops `actions` parameter (no CVAE):

```python
class ACT(nn.Module):
    """Action Chunking with Transformers — discrete bin variant.

    Predicts chunk_size timesteps of (dx_bins, dy_bins, click, key, pad)
    from a single image observation and proprioceptive state.
    dx/dy are 49-bin exponential classification heads.
    Proprioception modulates vision features via FiLM conditioning.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        chunk_size: int = CHUNK.default_chunk_size,
        proprio_dim: int = 31,
    ) -> None:
        super().__init__()
        d_model = MODEL.d_model

        self.chunk_size = chunk_size
        self.d_model = d_model

        # 1. Vision backbone
        self.backbone = build_backbone(backbone_name)

        # 2. FiLM conditioning: proprio -> scale/shift for vision tokens
        self.film_net = nn.Sequential(
            nn.Linear(proprio_dim, MODEL.film_hidden_dim),
            nn.ReLU(),
            nn.Linear(MODEL.film_hidden_dim, d_model * 2),
        )

        # 3. Proprioception token (still included in encoder sequence)
        self.proprio_proj = nn.Linear(proprio_dim, d_model)
        self.proprio_pos_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 4. Main encoder (vision + proprio, no latent)
        self.vision_pos_enc = SinusoidalPositionEncoding(d_model)
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=MODEL.nheads,
                dim_feedforward=MODEL.dim_feedforward,
                dropout=MODEL.dropout,
                activation="relu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=MODEL.encoder_layers,
        )

        # 5. Decoder
        self.query_embed = nn.Embedding(chunk_size, d_model)
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=MODEL.nheads,
                dim_feedforward=MODEL.dim_feedforward,
                dropout=MODEL.dropout,
                activation="relu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=MODEL.decoder_layers,
        )
        self.decoder_pos_enc = SinusoidalPositionEncoding(d_model)

        # 6. Action heads — dx/dy are NUM_BINS-class classification
        from .config import NUM_BINS
        self.head_dx = nn.Linear(d_model, NUM_BINS)
        self.head_dy = nn.Linear(d_model, NUM_BINS)
        self.head_click = nn.Linear(d_model, 1)
        self.head_key = nn.Linear(d_model, ACTION.num_key_classes)
        self.head_pad = nn.Linear(d_model, 1)

    def forward(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass (no CVAE — deterministic).

        Args:
            images: (B, 3, 224, 224)
            proprio: (B, proprio_dim)

        Returns:
            dict with dx_logits (B,chunk,49), dy_logits (B,chunk,49),
            click (B,chunk), key_logits (B,chunk,28), pad_logits (B,chunk).
        """
        B = images.size(0)

        # Vision tokens from backbone
        vision_tokens = self.backbone(images)  # (B, 49, d)

        # FiLM: modulate vision tokens with proprio BEFORE positional encoding
        # (preserves spatial position signals from PE)
        film_params = self.film_net(proprio)  # (B, d*2)
        gamma, beta = film_params.chunk(2, dim=-1)  # each (B, d)
        # gamma centered at 1 (multiplicative identity), beta at 0
        gamma = 1.0 + gamma.unsqueeze(1)  # (B, 1, d)
        beta = beta.unsqueeze(1)           # (B, 1, d)
        vision_tokens = gamma * vision_tokens + beta

        # Positional encoding added after FiLM
        vision_tokens = self.vision_pos_enc(vision_tokens)

        # Proprioception token
        proprio_tok = (
            self.proprio_proj(proprio).unsqueeze(1)
            + self.proprio_pos_embed
        )  # (B, 1, d)

        # Main encoder: vision (49) + proprio (1) = 50 tokens
        encoder_input = torch.cat([vision_tokens, proprio_tok], dim=1)
        memory = self.encoder(encoder_input)

        # Decoder
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        queries = self.decoder_pos_enc(queries)
        decoded = self.decoder(queries, memory)  # (B, chunk, d)

        # Action heads
        dx_logits = self.head_dx(decoded)             # (B, chunk, 49)
        dy_logits = self.head_dy(decoded)             # (B, chunk, 49)
        click = self.head_click(decoded).squeeze(-1)  # (B, chunk)
        key_logits = self.head_key(decoded)            # (B, chunk, 28)
        pad_logits = self.head_pad(decoded).squeeze(-1)  # (B, chunk)

        return {
            "dx_logits": dx_logits,
            "dy_logits": dy_logits,
            "click": click,
            "key_logits": key_logits,
            "pad_logits": pad_logits,
        }
```

Note key changes from current model:
- `forward()` takes `(images, proprio)` — no `actions` parameter
- Returns `dx_logits` / `dy_logits` (B, chunk, 49) instead of `dx` / `dy` (B, chunk)
- No `mu` / `logvar` in output
- FiLM applied BEFORE positional encoding (preserves PE spatial signals), with `gamma = 1 + film_output` so initial gamma is approx 1 (identity)
- 50 encoder tokens (49 vision + 1 proprio) instead of 51 (49 + 1 proprio + 1 latent)
- Delete `_encode_cvae` method entirely

**Step 2: Verify model instantiation and parameter count**

Run: `uv run python -c "from experiments.act_drag_label.model import ACT, count_parameters; m = ACT(); print(f'Params: {count_parameters(m, False):,}')"`

Expected: parameter count should be LOWER than the current 32.9M (removed CVAE components), with slight increase from FiLM MLP and wider dx/dy heads.

**Step 3: Commit**

```
git add experiments/act_drag_label/model.py
git commit -m "exp2: discrete bins + FiLM model — drop CVAE, 49-bin dx/dy heads, FiLM proprio"
```

---

### Task 3: Update train.py — new loss, no CVAE, bin targets

**Files:**
- Modify: `experiments/act_drag_label/train.py`

Three areas of change:
1. **ChunkDataset.__getitem__**: dx/dy targets become bin indices (int64) instead of normalized floats. Remove `actions_cvae` from return tuple.
2. **compute_losses**: dx/dy use cross-entropy on bin logits. Remove KL loss. Remove `kl_weight` param.
3. **Training loop**: remove KL annealing, remove `actions_cvae` from model forward call. Update head_names (remove "kl"). Update epoch log format.

**Step 1: Add bin import and delta-to-bin helper**

```python
# At the top of train.py, add import:
from experiments.act_drag_label.config import BIN_CENTERS, NUM_BINS

# Before ChunkDataset class, add helper:
def _delta_to_bin(raw_delta: np.ndarray) -> np.ndarray:
    """Convert raw pixel deltas to closest bin indices."""
    idx = np.searchsorted(BIN_CENTERS, raw_delta, side='left')
    idx = np.clip(idx, 0, NUM_BINS - 1)
    left = np.clip(idx - 1, 0, NUM_BINS - 1)
    closer_left = np.abs(raw_delta - BIN_CENTERS[left]) < np.abs(raw_delta - BIN_CENTERS[idx])
    return np.where(closer_left, left, idx).astype(np.int64)
```

**Step 2: Modify ChunkDataset.__getitem__ return signature**

Remove `actions_cvae` from the return tuple. Change dx/dy chunks from float to int64 bin indices. Set padded positions to center bin (index 24).

The new return type:
```python
def __getitem__(self, idx: int) -> tuple[
    torch.Tensor,  # obs (3, 224, 224)
    torch.Tensor,  # proprio (31,)
    torch.Tensor,  # dx_chunk (chunk,) int64 — bin indices
    torch.Tensor,  # dy_chunk (chunk,) int64 — bin indices
    torch.Tensor,  # click_chunk (chunk,) float
    torch.Tensor,  # key_chunk (chunk,) int64
    torch.Tensor,  # pad_mask (chunk,) float
]:
```

Key changes in the body:
```python
# dx/dy as bin indices instead of normalized floats:
dx_chunk = np.full(C, ACTION.num_bins_per_side, dtype=np.int64)  # center bin = no movement
dy_chunk = np.full(C, ACTION.num_bins_per_side, dtype=np.int64)  # center bin = no movement
# ...
dx_chunk[:real_len] = _delta_to_bin(self.action_dx[sl])
dy_chunk[:real_len] = _delta_to_bin(self.action_dy[sl])

# Remove all actions_cvae construction (lines 159-165 in current code)
# Remove actions_cvae from the return tuple

return obs, proprio, dx_chunk, dy_chunk, click_chunk, key_chunk, pad_mask
```

**Step 3: Rewrite compute_losses**

Remove `kl_weight` parameter. dx/dy use cross-entropy on `dx_logits`/`dy_logits`. No KL loss.

```python
def compute_losses(
    out: dict[str, torch.Tensor],
    dx_gt: torch.Tensor,    # (B, chunk) int64 bin indices
    dy_gt: torch.Tensor,    # (B, chunk) int64 bin indices
    click_gt: torch.Tensor,
    key_gt: torch.Tensor,
    pad_gt: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute all loss components. dx/dy are cross-entropy on bins."""
    mask = 1.0 - pad_gt
    mask_sum = mask.sum().clamp(min=1)

    B, C_chunk = dx_gt.shape
    mask_flat = mask.reshape(B * C_chunk)

    # dx/dy: cross-entropy on bin logits
    loss_dx = (
        F.cross_entropy(
            out["dx_logits"].reshape(B * C_chunk, -1),
            dx_gt.reshape(B * C_chunk),
            reduction="none",
        ) * mask_flat
    ).sum() / mask_flat.sum().clamp(min=1)

    loss_dy = (
        F.cross_entropy(
            out["dy_logits"].reshape(B * C_chunk, -1),
            dy_gt.reshape(B * C_chunk),
            reduction="none",
        ) * mask_flat
    ).sum() / mask_flat.sum().clamp(min=1)

    # click: BCE (unchanged)
    loss_click = (
        F.binary_cross_entropy_with_logits(out["click"], click_gt, reduction="none") * mask
    ).sum() / mask_sum

    # key: CE (unchanged)
    loss_key = (
        F.cross_entropy(
            out["key_logits"].reshape(B * C_chunk, -1),
            key_gt.reshape(B * C_chunk),
            reduction="none",
        ) * mask_flat
    ).sum() / mask_flat.sum().clamp(min=1)

    # pad: BCE (unchanged)
    loss_pad = F.binary_cross_entropy_with_logits(out["pad_logits"], pad_gt)

    total_loss = (
        loss_dx + loss_dy
        + TRAIN.loss_weight_click * loss_click
        + TRAIN.loss_weight_key * loss_key
        + TRAIN.loss_weight_pad * loss_pad
    )

    head_losses = {
        "dx": loss_dx, "dy": loss_dy, "click": loss_click,
        "key": loss_key, "pad": loss_pad,
    }
    return total_loss, head_losses
```

**Step 4: Update _run_train_epoch and _run_val_epoch**

In both functions:
- Unpack batch as `(obs, proprio, dx_gt, dy_gt, click_gt, key_gt, pad_gt)` — no `actions_cvae`
- Remove `actions_cvae` device transfer
- Call `model(obs, proprio)` — no third argument (remove `actions_cvae` from forward call)
- Remove `kl_weight` parameter and pass from `compute_losses` calls
- Update function signatures to remove `kl_weight` param

**Step 5: Update train() main function**

- Remove `kl_anneal_epochs` and `kl_weight` computation from the epoch loop
- Change `head_names` from `["dx", "dy", "click", "key", "pad", "kl"]` to `["dx", "dy", "click", "key", "pad"]`
- Remove `"kl"` from `head_sums` initialisation in train/val epoch helpers
- Update epoch log format: remove `kl_w={kl_weight:.4f}` from print
- Remove `kl_weight` arg from `_run_train_epoch` and `_run_val_epoch` calls

**Step 6: Verify training starts locally**

Run: `uv run python experiments/act_drag_label/train.py --device mps --batch-size 4 --max-episodes 50 2>&1 | head -30`

(Ctrl+C after first epoch completes.)

Expected: training starts without errors, loss values are reasonable (CE on 49 classes: initial ~log(49) = approx 3.9 per head).

**Step 7: Commit**

```
git add experiments/act_drag_label/train.py
git commit -m "exp2: train.py — CE loss on bins, drop CVAE/KL, bin index targets"
```

---

### Task 4: Update evaluate.py — probability ensemble with expected value

**Files:**
- Modify: `experiments/act_drag_label/evaluate.py`

**Step 1: Add imports**

```python
from experiments.act_drag_label.config import BIN_CENTERS, NUM_BINS
```

**Step 2: Rewrite ACTAgent class**

The new ACTAgent:
- Stores full softmax distributions in the ensemble buffer (not scalar values)
- Blends distributions with exponential decay weights
- Reads out expected value via dot product with bin centers
- Memory: 10 chunks x 49 bins x 2 dims = 980 floats (trivial)

```python
class ACTAgent:
    """Wraps a trained ACT model with probability-ensemble temporal smoothing."""

    def __init__(
        self,
        checkpoint: str,
        backbone: str = "resnet18",
        chunk_size: int = CHUNK.default_chunk_size,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.chunk_size = chunk_size
        self.model = ACT(
            backbone_name=backbone,
            chunk_size=chunk_size,
        ).to(self.device)
        state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()
        self.active_chunks: list[dict] = []

    def reset(self) -> None:
        self.active_chunks = []

    @torch.no_grad()
    def act(self, obs: np.ndarray, proprio: np.ndarray) -> dict:
        x = (
            torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        ).to(self.device)
        p = torch.from_numpy(proprio).unsqueeze(0).float().to(self.device)

        out = self.model(x, p)

        # Store softmax/sigmoid distributions (not raw logits)
        dx_probs = torch.softmax(out["dx_logits"][0], dim=-1).cpu().numpy()  # (chunk, 49)
        dy_probs = torch.softmax(out["dy_logits"][0], dim=-1).cpu().numpy()  # (chunk, 49)
        click_chunk = torch.sigmoid(out["click"][0]).cpu().numpy()
        key_probs = torch.softmax(out["key_logits"][0], dim=-1).cpu().numpy()  # (chunk, 28)

        self.active_chunks.append({
            "dx_probs": dx_probs,
            "dy_probs": dy_probs,
            "click": click_chunk,
            "key_probs": key_probs,
            "age": 0,
        })

        # Probability ensemble: blend all distributions consistently
        total_w = 0.0
        blended_dx = np.zeros(NUM_BINS, dtype=np.float32)
        blended_dy = np.zeros(NUM_BINS, dtype=np.float32)
        blended_click = 0.0
        blended_key = np.zeros(ACTION.num_key_classes, dtype=np.float32)

        for chunk in self.active_chunks:
            age = chunk["age"]
            if age >= self.chunk_size:
                continue
            w = np.exp(-CHUNK.ensemble_decay * age)
            total_w += w
            blended_dx += w * chunk["dx_probs"][age]
            blended_dy += w * chunk["dy_probs"][age]
            blended_click += w * chunk["click"][age]
            blended_key += w * chunk["key_probs"][age]

        if total_w > 0:
            blended_dx /= total_w
            blended_dy /= total_w
            blended_click /= total_w
            blended_key /= total_w

        # Expected value: dot product of blended distribution with bin centers
        dx_out = float(np.dot(blended_dx, BIN_CENTERS))
        dy_out = float(np.dot(blended_dy, BIN_CENTERS))
        click_out = 1 if blended_click > 0.5 else 0
        key_out = int(np.argmax(blended_key))

        # Increment ages and prune
        for chunk in self.active_chunks:
            chunk["age"] += 1
        self.active_chunks = [
            c for c in self.active_chunks if c["age"] < self.chunk_size
        ]

        return {
            "dx": dx_out,
            "dy": dy_out,
            "click": click_out,
            "key": key_out,
        }
```

**Step 3: Verify structure**

The `build_proprio`, `BaselineCNNAgent`, `ExpertAgent`, `RandomAgent`, `run_agent`, `print_metrics`, and `main` functions should need NO changes (BaselineCNN is a separate model with its own architecture).

The only change in `main()` is that ACTAgent's constructor no longer needs changes since it matches the new model's `forward(images, proprio)` signature.

**Step 4: Commit**

```
git add experiments/act_drag_label/evaluate.py
git commit -m "exp2: probability ensemble with expected value readout for discrete bins"
```

---

### Task 5: Verify end-to-end locally with a tiny training run

**Files:** None (verification only)

**Step 1: Generate a small dataset (if not already present)**

Run: `uv run python experiments/act_drag_label/generate_data.py -n 100`

(Skip if `experiments/act_drag_label/data/` already exists with data.)

**Step 2: Run 2 epochs of training locally**

Note: there is no `--epochs` CLI arg — the epoch count is set via `TRAIN.epochs` in config.py. For a quick local test, temporarily set `epochs: int = 2` in config.py, or just let it run for 2 epochs and Ctrl+C.

Run: `uv run python experiments/act_drag_label/train.py --device mps --batch-size 32 --max-episodes 100`

(Ctrl+C after 2 epochs complete.)

Expected:
- Model instantiates without errors
- Loss starts at ~3.9 per dx/dy head (log(49) for uniform CE on 49 classes)
- Loss decreases after 1 epoch
- No CUDA/MPS errors
- Epoch log shows `val_heads: dx=X.XXX dy=X.XXX click=X.XXX key=X.XXX pad=X.XXX` (no kl)

**Step 3: Run model-only eval on the tiny checkpoint**

Run: `uv run python experiments/act_drag_label/evaluate.py --backbone resnet18 --chunk-size 10 --device mps --model-only -n 5`

Expected: ACT agent runs, produces some output. Won't be good with 2 epochs on 100 episodes, but should not crash. Verify that dx/dy outputs are in pixel range (not stuck at +/-50).

**Step 4: Commit any fixes**

If any issues found, fix and commit individually.

---

### Task 6: Update HF Jobs launch script and run full training

**Files:**
- Check: `scripts/hf_job_train.py` — grep for `kl`, `cvae`, `actions_cvae`, `latent`

**Step 1: Check hf_job_train.py for CVAE references**

Grep for CVAE-related code. Remove or update any references. The script likely just passes CLI args through to train.py, so it may need no changes.

**Step 2: Launch training on HF Jobs**

Run:
```bash
uv run python scripts/launch_hf_job.py --flavor l40s-x1 --timeout 8h \
  -- --backbone resnet18 --chunk-size 10 \
  --hf-data-repo PenTest-duck/cu-vla-data \
  --hf-upload-repo PenTest-duck/cu-vla-checkpoints
```

**Step 3: Monitor first few epochs**

Run: `hf jobs logs <job_id> 2>&1 | grep "Epoch"`

Expected: dx/dy CE losses start ~3.9 and decrease. No KL in output. Total loss is sum of CE(dx) + CE(dy) + 2*BCE(click) + 2*CE(key) + BCE(pad).

**Step 4: Commit any launch script changes**

```
git add scripts/hf_job_train.py
git commit -m "exp2: update HF Jobs script for discrete bins model"
```

---

### Task 7: Update docs

**Files:**
- Modify: `AGENTS.md` — update Experiment 2 section to reflect new architecture
- Modify: `docs/learnings/2026-04-14-exp2-training-optimisation.md` — add discrete bins decision

**Step 1: Update AGENTS.md**

In the Experiment 2 section:
- Update model description: mention 49 exponential bins for dx/dy, FiLM conditioning, no CVAE
- Update parameter count once known
- Add `--model-only` flag to evaluate examples
- Note the action space change: dx/dy are now discrete bin indices, not continuous

**Step 2: Update learnings doc**

Add a section documenting the discrete bins change:
- Why: tanh saturation (gradient vanishes at +/-1, model can't produce small deltas), proprio ignored (1/51 tokens)
- What: 49 exponential bins (alpha=3, FDM-1 inspired), FiLM conditioning, drop CVAE
- How: CE loss, probability ensemble with expected value readout
- Expected impact: sub-pixel precision near zero, full gradient signal, proprio modulates every vision token

**Step 3: Commit**

```
git add AGENTS.md docs/learnings/2026-04-14-exp2-training-optimisation.md
git commit -m "docs: update exp2 architecture description for discrete bins + FiLM"
```
