# Experiment 5: Mini Text Editor — Model Architecture

## Context

The environment, expert policy, and data generation are complete (see `2026-04-14-mini-editor-design.md`). This document specifies the model architecture for training a behavior-cloned policy on the expert demonstrations.

**What this experiment tests:** Can a single model ground natural language edit instructions (the "L" in VLA) in visual observations of a text editor, and execute the corresponding multi-step motor sequences? Same visual scene + different instruction = different action sequence.

## Architecture Overview

ACT (Action Chunking with Transformers) extended with a trainable text encoder. Three input modalities — vision, language, proprioception — fused via concatenation in the encoder. The decoder predicts `chunk_size` actions in parallel.

```
Instruction text ──► Trainable Text Encoder ──► text tokens (L, 256) + 1D sin PE
                                                        │
Screenshot (640×480 → 384×288) ──► ResNet18 ──► (108, 256) + 2D sin PE
                                                        │
Proprio (56-dim) ──► FiLM (modulates vision) ──►        │
                 └─► Proprio token (1, 256) + learned PE │
                                                        │
                              concat: (~124, 256) ──► ACT Encoder (4L) ──► ACT Decoder (4L) ──► Actions
```

**Parameter budget: ~21M, all trainable.**

## Vision Encoder: ResNet18 at 384×288

### Input Resolution

The environment renders at 640×480 and downsamples observations to 512×384. We further resize to **384×288** before feeding to ResNet18. This preserves the 4:3 aspect ratio and produces a natural feature grid.

**Resolution chain:**
```
640×480 (env window) → 512×384 (observation) → 384×288 (model input)
```

**Character legibility:** At 384×288, characters are ~7.5px wide — sufficient for visually distinct word patterns. The model does not need to OCR individual characters; it needs to distinguish spatial word patterns and locate them on screen.

### Feature Extraction

ResNet18 downsamples by 32× total. At 384×288 input:

```
Input: (B, 3, 288, 384)
  conv1 (stride 2) → (B, 64, 144, 192)
  maxpool (stride 2) → (B, 64, 72, 96)
  layer1            → (B, 64, 72, 96)
  layer2 (stride 2) → (B, 128, 36, 48)
  layer3 (stride 2) → (B, 256, 18, 24)
  layer4 (stride 2) → (B, 512, 9, 12)

Feature map: (B, 512, 9, 12) → reshape → (B, 108, 512) → Linear(512, 256) → (B, 108, 256)
```

**No adaptive pooling.** The 9×12 = 108 spatial tokens emerge naturally from the 4:3 input. Each token is **centered** on a 32×32 pixel region in the 384×288 image (output stride), but has a much larger effective receptive field (~435×435 pixels for ResNet18 layer-4). This means each token carries broad context while being spatially anchored to a ~3-character-wide region. The wide receptive field is beneficial — it lets each token incorporate surrounding word context for recognition.

ResNet18 is initialized from ImageNet pretrained weights and **fully fine-tuned** (all 11M params trainable). Proven on Pygame environments in Exp 2/3.

### 2D Sinusoidal Positional Encoding

Vision tokens have spatial structure (9 rows × 12 columns) that 1D positional encoding fails to capture. We use **2D sinusoidal PE**: d_model is split into two halves, one encoding the row (y-position) and one encoding the column (x-position).

```python
# For a token at grid position (row, col):
PE[0:128]   = sinusoidal_encoding(row)    # y-axis
PE[128:256] = sinusoidal_encoding(col)    # x-axis
```

2D PE is added to vision tokens **after** FiLM modulation and **before** concatenation with text and proprio tokens. This gives the encoder explicit spatial structure: "this vision token is at (row=2, col=8)" rather than "this is the 32nd token in a flattened sequence."

## Text Encoder: Trainable Mini-Transformer

### Why Not a Frozen Pretrained LM

The instruction space is small (~32 template structures across 4 operations). The critical capability is **vision-language grounding**: the embedding for the word "fox" must be optimized to match the visual feature pattern that "fox" produces when rendered in monospace font on screen. A frozen encoder produces general-purpose English embeddings with no relationship to visual appearance. The Linear(512→256) projection can only apply a global affine transform — it cannot reshape individual word embeddings for visual grounding.

A trainable encoder allows end-to-end gradient flow from the action loss, through the encoder self-attention (where vision-language grounding happens), all the way back to the word embeddings. The embedding for "fox" can learn to land near the ResNet18 feature for rendered "fox."

### Architecture

```
Instruction text
    │
[WordPiece tokenizer — 30K vocab from MobileBERT, trimmed to ~3-5K active tokens]
    │
[Embedding layer: active_vocab × 256]
  └─ initialized from MobileBERT's WordPiece embeddings (512-dim → 256-dim via PCA)
  └─ provides pretrained priors: "delete" ≈ "remove" ≈ "erase"
    │
[2-layer transformer encoder]
  d_model=256, 4 heads, dim_feedforward=512
  Pre-LN (norm_first=True), dropout=0.1
    │
text_tokens: (L, 256)  + 1D sinusoidal PE
```

**Vocabulary trimming:** The full MobileBERT WordPiece vocabulary has 30,522 tokens. Most are irrelevant (Chinese characters, rare subwords). We trim to only tokens that appear in our instruction templates and corpus — likely 3,000–5,000 active tokens. This reduces the embedding table from 7.8M to ~1–1.3M params.

**Initialization from MobileBERT embeddings:** The embedding for each retained token is initialized from MobileBERT's pretrained 512-dim embedding, projected to 256-dim via PCA (computed once on the full embedding matrix). This provides synonym priors ("delete" ≈ "remove") without requiring the model to learn them from scratch. The embeddings are then fine-tuned end-to-end.

**Parameter count:** ~1.3M (embeddings) + ~2.6M (2 transformer layers) ≈ **~4M trainable params.**

### Generalization to Held-Out Phrasings

The training set uses 80% of instruction templates; the eval set uses 20% held-out phrasings. The pretrained embedding initialization provides semantic similarity between synonymous phrasings ("Delete the word '{w}'" vs "Remove '{w}' from the text"). The 2-layer transformer learns instruction structure from 10K episodes — enough to generalize across templates that differ by one or two words.

## Proprioception

### FiLM Conditioning

Proprioception (56-dim: cursor_xy normalized [0,1], mouse_left, 53 key states) modulates vision tokens via Feature-wise Linear Modulation **before** positional encoding:

```python
film_params = film_net(proprio)           # (B, 512)
gamma, beta = film_params.chunk(2, -1)    # each (B, 256)
vision_tokens = (1 + gamma) * vision_tokens + beta
```

FiLM from proprio (not text) is intentional. Proprio conditioning is a low-level operation: "bias visual features based on current cursor state." Text-vision interaction is a high-level operation that happens in the encoder's self-attention. These are complementary mechanisms at different levels of abstraction.

### Proprio Token

In addition to FiLM, the full proprio vector is projected to a learnable token and concatenated into the encoder sequence:

```python
proprio_tok = proprio_proj(proprio).unsqueeze(1) + proprio_pos_embed  # (B, 1, 256)
```

This gives the encoder direct access to the proprio values for high-level reasoning (e.g., "shift is currently held" informs whether to extend a selection).

## Encoder: Multimodal Fusion via Concatenation

All three modalities are concatenated into a single token sequence:

```
encoder_input = concat([vision_tokens, proprio_token, text_tokens])
                        (108, 256)      (1, 256)       (L, 256)
              = (108 + 1 + L, 256)  ≈ (~124, 256) for typical L ≈ 15
```

Each modality's positional encoding is added to its tokens **before** concatenation:
- Vision tokens: 2D sinusoidal PE (row/col)
- Text tokens: 1D sinusoidal PE (sequential)
- Proprio token: learned embedding

The encoder is a 4-layer Pre-LN transformer:

```python
TransformerEncoder(
    TransformerEncoderLayer(
        d_model=256, nhead=4, dim_feedforward=512,
        dropout=0.1, activation="relu",
        batch_first=True, norm_first=True,
    ),
    num_layers=4,
)
```

**Why concatenation, not cross-attention:** At our scale (~124 tokens, 4 layers, d=256), concatenation is simpler and sufficient. The encoder's self-attention handles vision-language interaction, vision-proprio interaction, and spatial reasoning jointly. Cross-attention (separate text-to-vision attention layers) is more principled but only pays off at larger scales.

## Decoder: Action Chunking

```python
# Learnable query embeddings for chunk_size positions
queries = query_embed.weight  # (chunk_size, 256)
queries = decoder_pos_enc(queries)  # 1D sinusoidal for temporal ordering

decoded = decoder(queries, memory)  # cross-attention to encoder output
```

4-layer Pre-LN transformer decoder. Each query position cross-attends to the full encoder memory (~124 tokens) and produces one action prediction.

**chunk_size = 10** (same as Exp 2/3). **Inference uses every-frame replanning with temporal ensemble smoothing** — NOT open-loop 10-step execution. Each frame, the model predicts 10 future actions, but only one action is executed before re-planning. Active chunks overlap and are blended with exponential decay weighting (older predictions weighted less). This receding-horizon approach is critical for text editing: shift+click selection and typing require closed-loop correction that open-loop execution cannot provide. Same ACT-style inference as Exp 2/3 (see `evaluate.py` temporal smoothing implementation).

## Action Heads

Five independent heads on the decoded representations:

```python
head_dx:    Linear(256, 49)   # 49-bin discrete dx logits (soft CE loss)
head_dy:    Linear(256, 49)   # 49-bin discrete dy logits (soft CE loss)
head_mouse: Linear(256, 1)    # mouse_left binary (BCE loss)
head_keys:  Linear(256, 53)   # 53 independent key states (BCE loss)
head_pad:   Linear(256, 1)    # padding mask (BCE loss)
```

**dx/dy bins:** 49 exponential bins (24 negative + zero + 24 positive) mapping to pixel deltas in the 640×480 screen coordinate space. Same binning as Exp 2/3 — the bins are in action space, independent of the 384×288 observation resolution. Trained with Gaussian-smoothed soft cross-entropy (σ=1.5).

**Key heads:** 53 independent binary outputs (sigmoid + BCE), one per physical key. No mutual exclusion constraint — multiple keys can be held simultaneously (e.g., Shift + A).

**Padding head:** Predicts whether each chunk position is padding (beyond episode end). Used to mask training loss and to signal episode completion during inference.

## Within-Chunk Typing: A Known Risk

The decoder predicts chunk_size=10 actions from a single observation. For mouse movement, this is well-proven — trajectories are smooth. But typing is qualitatively different: predicting "b-r-o-w-n" requires the decoder to output 5 specific key presses in exact order, purely from positional query embeddings and cross-attention to the fused memory. No intermediate visual feedback (e.g., seeing 'b' appear on screen).

This should work — the decoder queries have position-specific embeddings that can specialize — but typing errors (transpositions, wrong characters) may appear. **Monitoring plan:**

- Per-character accuracy within typing chunks
- Character transposition rate
- Failure rate vs typed word length
- Chunk-boundary behavior for words > 10 characters

If within-chunk typing proves problematic, potential mitigations include reducing chunk_size for typing phases or adding an autoregressive refinement step.

## Training

### Loss Function

Weighted sum of per-head losses, computed only on non-padded positions:

```
L = w_dx · soft_CE(dx_logits, dx_bins)
  + w_dy · soft_CE(dy_logits, dy_bins)
  + w_dx_ev · L1(EV[dx_logits], dx_true)    # expected value L1 auxiliary
  + w_dy_ev · L1(EV[dy_logits], dy_true)
  + w_mouse · BCE(mouse_logit, mouse_true)
  + w_keys · focal_BCE(keys_logits, keys_true)
  + w_pad · BCE(pad_logits, pad_true)
```

Loss weights TBD via initial tuning. Start with equal weights and adjust based on gradient magnitude analysis.

### Key Loss: Asymmetric / Focal BCE

Keys present a severe class imbalance: in most frames, all 53 keys are released (0). Only during typing or shift+click does a key become held (1). Standard BCE is dominated by the easy negatives (predicting 0 for all keys), under-weighting the critical press events.

We use **focal BCE** for the key heads: `FL(p_t) = -α_t (1 - p_t)^γ · log(p_t)` with `γ=2.0` and `α=0.75` for the positive class. This down-weights easy negatives and focuses learning on the hard positives (key presses). Additionally, stale held-state errors (forgetting to release a key) are catastrophic in the env — a phantom Shift press corrupts all subsequent typing. The focal loss helps the model learn crisp press/release transitions rather than conservative always-zero predictions.

### Optimizer and Schedule

- AdamW, lr=1e-4, weight_decay=1e-4
- Warmup: 5 epochs linear warmup
- Cosine annealing after warmup
- Gradient clipping: max_norm=100
- AMP (mixed precision) for training speed

### Data Loading

- Dataset: HuggingFace Parquet (same format as Exp 2/3)
- Chunk sampling: random start within episode, extract chunk_size consecutive frames
- Image loading: JPEG decode → resize to 384×288 → normalize
- Instruction: tokenize with trimmed WordPiece vocabulary

### Evaluation

- Primary metric: task success rate (final editor text == expected text)
- Per-operation success rate (click, click+type, select+delete, replace)
- Steps to completion (efficiency)
- Held-out phrasing accuracy vs training phrasing accuracy (language generalization)
- Mouse trajectory MAE (diagnostic)

## Parameter Summary

| Component | Params | Trainable |
|---|---|---|
| ResNet18 vision backbone | 11.2M | Yes |
| Text encoder (embeddings + 2L transformer) | ~4M | Yes |
| Vision projection Linear(512, 256) | 131K | Yes |
| FiLM network (56 → 512) | 29K | Yes |
| Proprio projection (56 → 256) | 14K | Yes |
| ACT encoder (4 layers) | ~2.7M | Yes |
| ACT decoder (4 layers) | ~2.7M | Yes |
| Action heads (dx, dy, mouse, keys, pad) | ~40K | Yes |
| Positional encodings | ~0 (buffers) | — |
| **Total** | **~21M** | **All trainable** |

## Inference Budget (M1 MacBook Pro)

| Component | Estimated time |
|---|---|
| Image resize (512×384 → 384×288) | <0.5ms |
| ResNet18 forward (384×288) | ~4-5ms |
| Text encoder forward (~15 tokens) | ~1ms |
| FiLM + concatenation | <0.5ms |
| ACT encoder (124 tokens, 4 layers) | ~3-4ms |
| ACT decoder (10 queries, 4 layers) | ~2-3ms |
| **Total** | **~12-15ms** |

Comfortably within the 33ms budget for 30 Hz. With every-frame replanning, the text encoder output can be cached for the entire episode (the instruction doesn't change), reducing per-frame cost to ~11-14ms.

## Relation to Exp 2/3 ACT

Key differences from the Exp 3 ACT model:

| Aspect | Exp 3 | Exp 5 |
|---|---|---|
| Input resolution | 224×224 | 384×288 |
| Vision tokens | 49 (7×7, pooled) | 108 (12×9, natural) |
| Vision PE | 1D sinusoidal | 2D sinusoidal |
| Text encoder | None | Trainable 2L transformer, ~4M params |
| Proprio dim | 46 | 56 |
| Key outputs | 43 | 53 |
| Total params | ~28M | ~21M |
| Trainable params | ~28M | ~21M |

The model is actually smaller than Exp 3's because there is no frozen backbone overhead.

## Future: SmolVLA Follow-Up

After validating ACT + trainable text encoder, a second experiment will evaluate a SmolVLA-style architecture: frozen SigLIP2 vision + frozen small LLM + trainable action head. This tests whether pretrained VLM representations transfer to Pygame environments. The ACT result serves as the baseline.
