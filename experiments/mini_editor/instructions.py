"""Edit instruction generator for Experiment 5: Mini Text Editor."""

from dataclasses import dataclass

import numpy as np

from .corpus import extract_words

# ---------------------------------------------------------------------------
# Instruction dataclass
# ---------------------------------------------------------------------------


@dataclass
class EditInstruction:
    operation: str  # "click" | "click_type" | "select_delete" | "replace"
    instruction_text: str  # NL instruction string
    target_word: str  # the word being operated on
    target_word_start: int  # char index in text where word starts
    target_word_end: int  # char index in text where word ends
    new_text: str | None  # text to type (for click_type and replace ops)
    expected_text: str  # expected final editor text after operation


# ---------------------------------------------------------------------------
# Template phrasings (variety for data augmentation)
# ---------------------------------------------------------------------------

CLICK_TEMPLATES = [
    "Click after the word '{word}'",
    "Position the cursor after '{word}'",
    "Move to '{word}' and click after it",
    "Place your cursor right after '{word}'",
    "Set the cursor just past '{word}'",
    "Navigate to '{word}' and place the cursor after it",
    "Click at the end of '{word}'",
    "Put the cursor after '{word}'",
]

CLICK_TYPE_TEMPLATES = [
    "Click after '{word}' and type '{text}'",
    "Place cursor after '{word}' and insert '{text}'",
    "Position after '{word}', then type '{text}'",
    "Go to '{word}' and type '{text}' after it",
    "Move to '{word}', click after it, and type '{text}'",
    "Navigate after '{word}' and insert '{text}'",
    "Set cursor after '{word}' and enter '{text}'",
    "Click right after '{word}' then type '{text}'",
]

SELECT_DELETE_TEMPLATES = [
    "Select the word '{word}' and delete it",
    "Delete the word '{word}'",
    "Highlight '{word}' and remove it",
    "Select '{word}' and press delete",
    "Remove the word '{word}' from the text",
    "Find '{word}' and delete it",
    "Erase the word '{word}'",
    "Select and remove '{word}'",
]

REPLACE_TEMPLATES = [
    "Replace '{word}' with '{new_word}'",
    "Change '{word}' to '{new_word}'",
    "Swap '{word}' for '{new_word}'",
    "Substitute '{new_word}' for '{word}'",
    "Select '{word}' and type '{new_word}' instead",
    "Overwrite '{word}' with '{new_word}'",
    "Find '{word}' and replace it with '{new_word}'",
    "Change the word '{word}' to '{new_word}'",
]

_OPERATIONS = ["click", "click_type", "select_delete", "replace"]

_TEMPLATE_MAP = {
    "click": CLICK_TEMPLATES,
    "click_type": CLICK_TYPE_TEMPLATES,
    "select_delete": SELECT_DELETE_TEMPLATES,
    "replace": REPLACE_TEMPLATES,
}


# ---------------------------------------------------------------------------
# Instruction generation
# ---------------------------------------------------------------------------


def generate_instruction(
    text: str,
    words: list[dict],
    rng: np.random.Generator,
    vocab: list[str] | None = None,
    operation: str | None = None,
) -> EditInstruction:
    """Sample a random edit instruction for the given text.

    Parameters
    ----------
    text : str
        The current editor text.
    words : list[dict]
        Output of ``extract_words(text)`` — targetable words with positions.
    rng : numpy random Generator
        Source of randomness.
    vocab : list[str] | None
        Optional word list for sampling new_text.  Falls back to word texts
        from *words* if not provided.
    operation : str | None
        Force a specific operation (one of "click", "click_type",
        "select_delete", "replace").  If None, sample randomly.
    """
    op = operation if operation is not None else _OPERATIONS[rng.integers(len(_OPERATIONS))]
    target = words[rng.integers(len(words))]
    templates = _TEMPLATE_MAP[op]
    template = templates[rng.integers(len(templates))]

    word_text = target["word"]
    start = target["start"]
    end = target["end"]

    # Build the pool for sampling insertion / replacement text
    pool = vocab if vocab is not None else [w["word"] for w in words]

    new_text: str | None = None

    if op == "click":
        instruction_text = template.format(word=word_text)
        expected_text = text  # cursor moves, text unchanged

    elif op == "click_type":
        n_words = rng.integers(1, 3)  # 1 or 2
        sampled = [pool[rng.integers(len(pool))] for _ in range(n_words)]
        new_text = " ".join(sampled)
        instruction_text = template.format(word=word_text, text=new_text)
        # Insert new_text right after the target word
        expected_text = text[:end] + new_text + text[end:]

    elif op == "select_delete":
        instruction_text = template.format(word=word_text)
        new_text = None
        # Remove the word (characters at start:end)
        expected_text = text[:start] + text[end:]

    else:  # replace
        new_text = pool[rng.integers(len(pool))]
        instruction_text = template.format(word=word_text, new_word=new_text)
        expected_text = text[:start] + new_text + text[end:]

    return EditInstruction(
        operation=op,
        instruction_text=instruction_text,
        target_word=word_text,
        target_word_start=start,
        target_word_end=end,
        new_text=new_text,
        expected_text=expected_text,
    )
