"""Corpus loader and text utilities for Experiment 5: Mini Text Editor."""

import re

import numpy as np
from datasets import load_dataset


def load_corpus() -> list[str]:
    """Load and filter the sentence corpus from HuggingFace Hub.

    1. Load agentlans/high-quality-english-sentences
    2. Keep only sentences where every character is printable ASCII (0x20-0x7E)
    3. Keep sentences with length 20-120 characters
    """
    ds = load_dataset("agentlans/high-quality-english-sentences", split="train")
    sentences: list[str] = []
    ascii_re = re.compile(r"^[\x20-\x7e]+$")
    for row in ds:
        text = row["text"]
        if 20 <= len(text) <= 120 and ascii_re.match(text):
            sentences.append(text)
    return sentences


def extract_words(text: str) -> list[dict]:
    """Extract targetable words from text.

    A "word" is a maximal run of [a-zA-Z0-9]+ characters.
    Returns list of {"word": str, "start": int, "end": int}
    where start/end are character indices. Only words with len >= 3.
    """
    return [
        {"word": m.group(), "start": m.start(), "end": m.end()}
        for m in re.finditer(r"[a-zA-Z0-9]+", text)
        if len(m.group()) >= 3
    ]


def make_passage(
    sentences: list[str], rng: np.random.Generator, max_chars: int = 250
) -> str | None:
    """Combine 1-3 random sentences into a passage.

    Returns passage string, or None if constraints not met.
    Requires total length <= max_chars and >= 4 unique targetable words.
    """
    n = rng.integers(1, 4)  # 1, 2, or 3
    idxs = rng.choice(len(sentences), size=n, replace=False)
    passage = " ".join(sentences[i] for i in idxs)
    if len(passage) > max_chars:
        return None
    words = extract_words(passage)
    unique_words = {w["word"] for w in words}
    if len(unique_words) < 4:
        return None
    return passage


def wrap_text(text: str, chars_per_line: int = 32) -> list[str]:
    """Word-wrap text into lines. No mid-word breaks.

    Split on spaces, greedily fill lines up to chars_per_line.
    If a single word exceeds chars_per_line, put it on its own line.
    """
    tokens = text.split(" ")
    lines: list[str] = []
    current: list[str] = []
    current_len = 0

    for token in tokens:
        needed = len(token) if current_len == 0 else current_len + 1 + len(token)
        if current and needed > chars_per_line:
            lines.append(" ".join(current))
            current = [token]
            current_len = len(token)
        else:
            current.append(token)
            current_len = needed

    if current:
        lines.append(" ".join(current))

    return lines
