"""Trainable text encoder with MobileBERT-initialized embeddings.

Produces per-token embeddings (B, L, d_model) from natural language
instructions.  Word embeddings are initialized from MobileBERT's pretrained
WordPiece table, projected from 512-d to 256-d via PCA (SVD), then
fine-tuned end-to-end so that gradient flow can ground language in the
vision-action space.

The vocabulary is trimmed to only the tokens that actually appear in the
instruction templates and (optionally) a corpus sample, yielding ~3-5k
tokens instead of the full 30 522.  A ``token_id_map`` dict translates
original MobileBERT IDs to trimmed indices at inference time.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .config import MODEL

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding (1-D, standard)
# ---------------------------------------------------------------------------


class SinusoidalPositionEncoding1D(nn.Module):
    """Standard 1-D sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) -> (B, L, d_model) with PE added."""
        return x + self.pe[:, : x.size(1), :]


# ---------------------------------------------------------------------------
# Pre-LN Transformer encoder layer
# ---------------------------------------------------------------------------


class PreLNEncoderLayer(nn.Module):
    """Transformer encoder layer with Pre-LayerNorm (LN before attention/FFN)."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-LN self-attention
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, key_padding_mask=src_key_padding_mask)
        x = x + self.dropout(h)
        # Pre-LN FFN
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


# ---------------------------------------------------------------------------
# TextEncoder
# ---------------------------------------------------------------------------


class TextEncoder(nn.Module):
    """Trainable text encoder with MobileBERT-initialized embeddings.

    Architecture:
        nn.Embedding  ->  SinusoidalPE  ->  N x PreLN TransformerEncoder

    Produces (B, L, d_model) token features for downstream concatenation
    with vision tokens in the main ACT encoder.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = MODEL.d_model,
        nhead: int = MODEL.nheads,
        num_layers: int = 2,
        dim_feedforward: int = MODEL.dim_feedforward,
        dropout: float = MODEL.dropout,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionEncoding1D(d_model, max_len=max_seq_len)
        self.embed_scale = math.sqrt(d_model)

        self.layers = nn.ModuleList(
            [
                PreLNEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode token IDs to contextual embeddings.

        Parameters
        ----------
        input_ids : (B, L) long tensor of trimmed vocabulary IDs.
        attention_mask : (B, L) float/bool, 1 = real token, 0 = padding.
            Converted internally to the key_padding_mask convention
            (True = ignore).

        Returns
        -------
        (B, L, d_model) token features.
        """
        x = self.embedding(input_ids) * self.embed_scale
        x = self.pos_enc(x)
        x = self.dropout(x)

        # Convert attention_mask (1=keep, 0=pad) -> key_padding_mask (True=ignore)
        key_padding_mask: torch.Tensor | None = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # True where padded

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)

        x = self.final_norm(x)
        return x


# ---------------------------------------------------------------------------
# Factory: build_text_encoder
# ---------------------------------------------------------------------------


def build_text_encoder(
    corpus_sentences: list[str] | None = None,
    d_model: int = MODEL.d_model,
    num_layers: int = 2,
) -> tuple["TextEncoder", "PreTrainedTokenizerBase", dict[int, int]]:
    """Build a TextEncoder with MobileBERT-initialized embeddings.

    Steps:
        1. Load MobileBERT tokenizer + model.
        2. Collect token IDs from instruction templates (and optional corpus).
        3. Trim the vocabulary to only needed tokens.
        4. Project kept embeddings from 512-d to ``d_model`` via PCA (SVD).
        5. Initialize TextEncoder embedding layer with projected weights.
        6. Free MobileBERT model memory.

    Parameters
    ----------
    corpus_sentences : list[str] | None
        Optional corpus sentences to include in vocabulary coverage.
    d_model : int
        Output embedding dimension (default from MODEL config).
    num_layers : int
        Number of Pre-LN transformer encoder layers.

    Returns
    -------
    (encoder, tokenizer, token_id_map)
        - encoder: TextEncoder with PCA-projected MobileBERT embeddings.
        - tokenizer: The original MobileBERT tokenizer (for tokenizing new text).
        - token_id_map: dict mapping original MobileBERT token IDs -> trimmed IDs.
    """
    from transformers import AutoModel, AutoTokenizer

    # --- 1. Load MobileBERT tokenizer and model ---
    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
    model = AutoModel.from_pretrained("google/mobilebert-uncased")

    # --- 2. Extract full embedding matrix (30522, 512) ---
    full_embeddings: torch.Tensor = (
        model.embeddings.word_embeddings.weight.detach().clone()
    )
    mobilebert_dim = full_embeddings.shape[1]  # 512

    # --- 3. Collect all token IDs we need ---
    from .instructions import (
        CLICK_TEMPLATES,
        CLICK_TYPE_TEMPLATES,
        REPLACE_TEMPLATES,
        SELECT_DELETE_TEMPLATES,
    )

    # Always keep special tokens
    needed_ids: set[int] = {0, 100, 101, 102}  # [PAD], [UNK], [CLS], [SEP]

    # Tokenize all instruction template strings
    all_templates = (
        CLICK_TEMPLATES
        + CLICK_TYPE_TEMPLATES
        + SELECT_DELETE_TEMPLATES
        + REPLACE_TEMPLATES
    )
    for template in all_templates:
        # Replace format placeholders with dummy words so tokenizer sees real text
        text = template.replace("{word}", "example").replace(
            "{new_word}", "replacement"
        ).replace("{text}", "sample text")
        encoded = tokenizer.encode(text, add_special_tokens=True)
        needed_ids.update(encoded)

    # Also tokenize the placeholder words themselves and common edit vocabulary
    _extra_words = [
        "click", "type", "delete", "select", "replace", "remove",
        "insert", "erase", "highlight", "find", "change", "swap",
        "substitute", "overwrite", "position", "cursor", "navigate",
        "move", "place", "set", "after", "before", "the", "word",
        "and", "with", "for", "from", "to", "it", "a", "right",
        "just", "past", "end", "of", "then", "go", "put", "press",
        "enter", "at", "in", "text", "instead",
    ]
    for word in _extra_words:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        needed_ids.update(encoded)

    # If corpus provided, tokenize a sample
    if corpus_sentences is not None:
        for sentence in corpus_sentences:
            encoded = tokenizer.encode(sentence, add_special_tokens=True)
            needed_ids.update(encoded)

    # --- 4. Build mapping: original_id -> trimmed_id ---
    sorted_ids = sorted(needed_ids)
    token_id_map: dict[int, int] = {
        orig_id: trimmed_id for trimmed_id, orig_id in enumerate(sorted_ids)
    }
    trimmed_vocab_size = len(sorted_ids)

    print(f"Trimmed vocab size: {trimmed_vocab_size} "
          f"(from {full_embeddings.shape[0]} original MobileBERT tokens)")

    # --- 5. Extract embeddings for kept tokens ---
    kept_indices = torch.tensor(sorted_ids, dtype=torch.long)
    kept_embeddings = full_embeddings[kept_indices]  # (N_kept, 512)

    # --- 6. Project to d_model via PCA (SVD) ---
    if d_model < mobilebert_dim:
        # Center
        mean = kept_embeddings.mean(dim=0, keepdim=True)
        centered = kept_embeddings - mean

        # SVD: centered = U @ diag(S) @ Vh
        # We want the first d_model principal components
        _U, _S, Vh = torch.linalg.svd(centered, full_matrices=False)
        # Project: each row dotted with first d_model right singular vectors
        projection_matrix = Vh[:d_model, :].T  # (512, d_model)
        projected = centered @ projection_matrix  # (N_kept, d_model)
    elif d_model == mobilebert_dim:
        projected = kept_embeddings
    else:
        # d_model > mobilebert_dim: pad with zeros (unlikely but handle it)
        projected = torch.zeros(trimmed_vocab_size, d_model)
        projected[:, :mobilebert_dim] = kept_embeddings

    # --- 7. Create TextEncoder and initialize embeddings ---
    encoder = TextEncoder(
        vocab_size=trimmed_vocab_size,
        d_model=d_model,
        num_layers=num_layers,
    )
    with torch.no_grad():
        encoder.embedding.weight.copy_(projected)

    # --- 8. Free MobileBERT model ---
    del model
    del full_embeddings

    return encoder, tokenizer, token_id_map


# ---------------------------------------------------------------------------
# Tokenize helper
# ---------------------------------------------------------------------------


def tokenize_instruction(
    text: str,
    tokenizer: "PreTrainedTokenizerBase",
    token_id_map: dict[int, int],
    max_length: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize a string and remap IDs to the trimmed vocabulary.

    Parameters
    ----------
    text : str
        The instruction string to tokenize.
    tokenizer : PreTrainedTokenizerBase
        The original MobileBERT tokenizer.
    token_id_map : dict[int, int]
        Mapping from original MobileBERT token IDs to trimmed indices.
    max_length : int
        Maximum sequence length (padded/truncated).

    Returns
    -------
    (input_ids, attention_mask) each of shape (1, L).
        Unknown tokens (not in trimmed vocab) are mapped to [UNK]'s trimmed ID.
    """
    encoded = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    original_ids = encoded["input_ids"]  # (1, L)
    attention_mask = encoded["attention_mask"]  # (1, L)

    # Remap through trimmed vocabulary
    unk_trimmed_id = token_id_map[100]  # [UNK] original ID = 100
    remapped = torch.tensor(
        [
            [token_id_map.get(tok_id.item(), unk_trimmed_id) for tok_id in row]
            for row in original_ids
        ],
        dtype=torch.long,
    )

    return remapped, attention_mask
