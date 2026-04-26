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


# ---------------------------------------------------------------------------
# Phase B0 dataset
# ---------------------------------------------------------------------------


def _split_click_5way(c: int) -> tuple[int, int]:
    """Map the legacy 5-way click event to two parallel 3-way labels.

    5-way: {0=idle, 1=L_press, 2=L_release, 3=R_press, 4=R_release}.
    Each 3-way head emits {0=idle, 1=press, 2=release}.
    """
    if c == 1:
        return (1, 0)  # L_press
    if c == 2:
        return (2, 0)  # L_release
    if c == 3:
        return (0, 1)  # R_press
    if c == 4:
        return (0, 2)  # R_release
    return (0, 0)      # idle


class PhaseB0EpisodeDataset(PhaseAEpisodeDataset):
    """B0 episode dataset: distractor scenes + grounded instructions + recovery + loss masking.

    Differences from Phase A:
    - Action history vector is built from `action_*` columns (applied actions —
      these may be DART-noisy or wrong-segment; matches inference distribution).
    - Training targets come from `action_label_*` columns (clean expert) and are
      returned under an ``action_label`` dict.
    - The legacy 5-way ``action_label_click`` is split on-the-fly into
      ``click_left`` / ``click_right`` 3-way labels (idle/press/release each).
    - ``loss_mask`` (T,) per-frame mask is returned as a float tensor.
    - ``instruction`` is the per-episode NL instruction string (pulled from row 0).
    - Episode-level metadata (composite_tier, is_adversarial, scenario_type, etc.)
      is returned for eval slicing.

    Splits use the same episode_id-hash pattern as Phase A. Adds ``"all"`` for
    tests/eval that want to iterate every episode.
    """

    def __init__(self, data_dir: Path, split: str = "train", preprocess: bool = True) -> None:
        # NOTE: bypass PhaseAEpisodeDataset.__init__ because B0 supports the
        # "all" split that Phase A does not. We re-implement the minimal init.
        ds = load_dataset("parquet", data_files=str(Path(data_dir) / "shard_*.parquet"))["train"]

        def split_fn(ex):
            eid = ex["episode_id"]
            bucket = eid % 10
            if split == "all":
                return True
            if split == "train":
                return bucket < 8
            if split == "val":
                return bucket == 8
            if split == "test":
                return bucket == 9
            return False

        self.ds = ds.filter(split_fn)
        self.episode_index: dict[int, list[int]] = {}
        for i, ex in enumerate(self.ds):
            self.episode_index.setdefault(ex["episode_id"], []).append(i)
        self.episode_ids = sorted(self.episode_index.keys())
        self.preprocess = preprocess
        self._processor = None

    def __getitem__(self, idx: int) -> dict:
        eid = self.episode_ids[idx]
        row_indices = self.episode_index[eid]
        # Sort rows by frame_idx to ensure temporal order (worker shards may
        # interleave but generator emits in order; sorting is cheap insurance).
        frames = sorted((self.ds[i] for i in row_indices), key=lambda f: f["frame_idx"])

        # ---- Vision -----------------------------------------------------
        images = [decode_jpeg_bytes(f["image_bytes"]) for f in frames]

        # ---- Applied actions (for history vector) -----------------------
        applied_dx_bins = [quantize_to_bin(f["action_dx"], MOUSE_BIN_CENTERS) for f in frames]
        applied_dy_bins = [quantize_to_bin(f["action_dy"], MOUSE_BIN_CENTERS) for f in frames]
        applied_scroll_bins = [
            quantize_to_bin(f["action_scroll"], SCROLL_BIN_CENTERS) for f in frames
        ]
        applied_clicks = [int(f["action_click"]) for f in frames]
        applied_keys = [
            np.asarray(f["action_key_events"], dtype=np.int64) for f in frames
        ]
        dones = [int(f["done_gt"]) for f in frames]

        # ---- Proprio ----------------------------------------------------
        proprio_per_frame = []
        for f in frames:
            proprio_per_frame.append(np.concatenate([
                np.array([f["cursor_x"], f["cursor_y"]], dtype=np.float32),
                np.asarray(f["held_keys"], dtype=np.float32),
                np.asarray(f["held_mouse"], dtype=np.float32),
                np.array([f["capslock"]], dtype=np.float32),
            ]))

        # ---- Action history (from APPLIED actions) ----------------------
        K = MODEL.action_history_len
        history_per_frame = []
        zero_action = {
            "dx_bin": NUM_BINS_MOUSE // 2,
            "dy_bin": NUM_BINS_MOUSE // 2,
            "click": 0,
            "scroll_bin": NUM_BINS_MOUSE // 2,
            "key_events": [2] * NUM_KEYS,
            "done_gt": 0,
        }
        for t in range(len(frames)):
            prev = []
            for k in range(K, 0, -1):
                j = t - k
                if j < 0:
                    prev.append(zero_action)
                else:
                    prev.append({
                        "dx_bin": applied_dx_bins[j],
                        "dy_bin": applied_dy_bins[j],
                        "click": applied_clicks[j],
                        "scroll_bin": applied_scroll_bins[j],
                        "key_events": applied_keys[j].tolist(),
                        "done_gt": dones[j],
                    })
            history_per_frame.append(build_action_history_vector(prev))

        # ---- Clean expert labels (training targets) ---------------------
        # Continuous values are kept around alongside the quantized bins
        # because B0 soft-CE losses (`total_loss_b0`) need the raw float
        # so they can place mass on the two bins bracketing the target.
        label_dx_continuous = [float(f["action_label_dx"]) for f in frames]
        label_dy_continuous = [float(f["action_label_dy"]) for f in frames]
        label_scroll_continuous = [float(f["action_label_scroll"]) for f in frames]
        label_dx_bins = [
            quantize_to_bin(v, MOUSE_BIN_CENTERS) for v in label_dx_continuous
        ]
        label_dy_bins = [
            quantize_to_bin(v, MOUSE_BIN_CENTERS) for v in label_dy_continuous
        ]
        label_scroll_bins = [
            quantize_to_bin(v, SCROLL_BIN_CENTERS) for v in label_scroll_continuous
        ]
        label_clicks_5way = [int(f["action_label_click"]) for f in frames]
        click_left, click_right = [], []
        for c in label_clicks_5way:
            cl, cr = _split_click_5way(c)
            click_left.append(cl)
            click_right.append(cr)
        label_key_events = [
            np.asarray(f["action_label_key_events"], dtype=np.int64) for f in frames
        ]

        # ---- Loss mask + instruction (per-frame mask, per-episode str) --
        loss_mask = [float(f["loss_mask"]) for f in frames]
        instruction = str(frames[0]["instruction"])

        # ---- Auxiliary target-grid-cell label (B0 attempt 2 — A3) -------
        # Compute grid cell from target_bbox center. The bbox can drift slightly
        # within an episode (re-rendered each frame), but per-cell category is
        # stable. We pick frame-0's bbox for the per-episode label.
        from experiments.action_primitives.config import ENV
        from experiments.action_primitives.config import B0_POSITION_GRID
        bbox_x = float(frames[0]["target_bbox_x"])
        bbox_y = float(frames[0]["target_bbox_y"])
        bbox_w = float(frames[0]["target_bbox_w"])
        bbox_h = float(frames[0]["target_bbox_h"])
        cx = bbox_x + bbox_w / 2.0
        cy = bbox_y + bbox_h / 2.0
        n_cols, n_rows = B0_POSITION_GRID
        col = min(int(cx / ENV.canvas_w * n_cols), n_cols - 1)
        row = min(int(cy / ENV.canvas_h * n_rows), n_rows - 1)
        target_cell = row * n_cols + col  # row-major index

        # ---- Episode-frame index (for aux loss masking + diagnostics) ---
        episode_frame_idx = [int(f["frame_idx"]) for f in frames]

        # ---- Episode metadata (constant across frames; pull from row 0) -
        meta = {
            "episode_id": int(frames[0]["episode_id"]),
            "primitive_type": str(frames[0]["primitive_type"]),
            "tempo": str(frames[0]["tempo"]),
            "n_buttons": int(frames[0]["n_buttons"]),
            "composite_tier": int(frames[0]["composite_tier"]),
            "is_adversarial": int(frames[0]["is_adversarial"]),
            "is_scenario_error": int(frames[0]["is_scenario_error"]),
            "scenario_type": str(frames[0]["scenario_type"]),
            "k_wrong_frames": int(frames[0]["k_wrong_frames"]),
            "target_button_id": int(frames[0]["target_button_id"]),
            "target_cell": int(target_cell),
        }

        out: dict = {
            "proprio": torch.from_numpy(np.stack(proprio_per_frame)),               # (T, 83)
            "action_history": torch.from_numpy(np.stack(history_per_frame)).float(), # (T, K, 223)
            "loss_mask": torch.tensor(loss_mask, dtype=torch.float32),               # (T,)
            "instruction": instruction,
            # B0 attempt 2: per-frame fields for aux target head.
            # target_cell is constant per episode but broadcast to (T,) for
            # micro-batch concat; episode_frame_idx is the per-frame position
            # within the episode (used for aux-loss masking to first_n frames).
            "target_cell": torch.tensor([target_cell] * len(frames), dtype=torch.long),  # (T,)
            "episode_frame_idx": torch.tensor(episode_frame_idx, dtype=torch.long),       # (T,)
            "action_label": {
                "dx_bins": torch.tensor(label_dx_bins, dtype=torch.long),            # (T,)
                "dy_bins": torch.tensor(label_dy_bins, dtype=torch.long),            # (T,)
                "dx_continuous": torch.tensor(label_dx_continuous, dtype=torch.float32),     # (T,)
                "dy_continuous": torch.tensor(label_dy_continuous, dtype=torch.float32),     # (T,)
                "scroll_continuous": torch.tensor(label_scroll_continuous, dtype=torch.float32),  # (T,)
                "click_left": torch.tensor(click_left, dtype=torch.long),            # (T,)
                "click_right": torch.tensor(click_right, dtype=torch.long),          # (T,)
                "scroll_bins": torch.tensor(label_scroll_bins, dtype=torch.long),    # (T,)
                "key_events": torch.from_numpy(np.stack(label_key_events)),          # (T, 77)
                "dones": torch.tensor(dones, dtype=torch.long),                      # (T,)
            },
            "metadata": meta,
        }
        if self.preprocess:
            if self._processor is None:
                from transformers import AutoProcessor
                self._processor = AutoProcessor.from_pretrained(MODEL.vision_model)
            proc = self._processor(
                images=images,
                return_tensors="pt",
                max_num_patches=MODEL.max_num_patches,
            )
            out["pixel_values"] = proc["pixel_values"]
            out["pixel_attention_mask"] = proc["pixel_attention_mask"]
            out["spatial_shapes"] = proc["spatial_shapes"]
            # Also expose the raw image list for callers that want both;
            # B0 keeps `images` for downstream eval/visualization. To keep
            # memory low when preprocess=True we omit `images` by default.
        else:
            out["images"] = images
        return out
