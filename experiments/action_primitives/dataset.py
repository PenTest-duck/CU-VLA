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
    """Groups parquet rows into per-episode lists; returns (T, ...) per episode.

    If ``preprocess=True`` (default), the SigLIP2 naflex processor runs inside
    ``__getitem__`` and the returned dict contains preprocessed vision tensors
    (`pixel_values`, `pixel_attention_mask`, `spatial_shapes`) instead of PIL
    images. This lets a ``DataLoader(num_workers>0)`` hide the processor cost
    behind the GPU forward. Set ``preprocess=False`` to get raw PIL images
    back (used by tests that want to inspect image content).
    """

    def __init__(self, data_dir: Path, split: str = "train", preprocess: bool = True) -> None:
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
        self.preprocess = preprocess
        # Lazy-init the SigLIP2 processor; each DataLoader worker will create
        # its own instance on first __getitem__ call. Deferring to first access
        # keeps __init__ fast and avoids holding a large object across the
        # worker fork.
        self._processor = None

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
        # Instruction: Phase A uses a single fixed string. Theme metadata is
        # already available in the visual stream — putting it in language too
        # would leak scene supervision into the text channel and confound the
        # vision-vs-language attribution Phase A is supposed to hold constant.
        instruction = "click the button"
        out = {
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
        if self.preprocess:
            # Run the SigLIP2 naflex processor here so workers do the CPU-side
            # image → patch preprocessing. Lazy-create on first access.
            if self._processor is None:
                from transformers import AutoProcessor
                self._processor = AutoProcessor.from_pretrained(MODEL.vision_model)
            proc = self._processor(
                images=images,
                return_tensors="pt",
                max_num_patches=MODEL.max_num_patches,
            )
            out["pixel_values"] = proc["pixel_values"]                             # (T, N, P*P*C)
            out["pixel_attention_mask"] = proc["pixel_attention_mask"]             # (T, N)
            out["spatial_shapes"] = proc["spatial_shapes"]                         # (T, 2)
        else:
            out["images"] = images  # list[PIL] length T
        return out
