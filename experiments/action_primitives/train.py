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
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.action_primitives.config import MODEL, MOUSE_BIN_CENTERS, TRAIN
from experiments.action_primitives.dataset import (
    PhaseAEpisodeDataset,
    PhaseB0EpisodeDataset,
)
from experiments.action_primitives.losses import total_loss, total_loss_b0
from experiments.action_primitives.metrics import (
    bin_10_frequency,
    per_class_click_recall,
    soft_ce_diagnostics,
)
from experiments.action_primitives.model import ActionPrimitivesACT


def cosine_lr(step: int, max_steps: int, warmup_steps: int, base_lr: float, min_frac: float = 0.1) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_frac + (1 - min_frac) * cos)


def infinite_loader(ds: PhaseAEpisodeDataset | PhaseB0EpisodeDataset, seed: int = 0) -> Iterator[dict]:
    """Simple single-process infinite generator. No prefetching."""
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(len(ds))
        for i in perm:
            yield ds[int(i)]


def _unwrap_single(batch: list) -> dict:
    """Identity collate for batch_size=1 + non-tensor items (PIL lists etc.)."""
    return batch[0]


def prefetching_loader(
    ds: PhaseAEpisodeDataset | PhaseB0EpisodeDataset,
    num_workers: int,
    pin_memory: bool = False,
) -> Iterator[dict]:
    """Infinite loader backed by torch DataLoader with num_workers worker
    processes running `__getitem__` in parallel, prefetching 4× num_workers
    episodes ahead. Hides CPU-side JPEG decode + SigLIP2 processor +
    history-vector construction behind the GPU forward/backward.

    Follows the exp2/exp3 DataLoader pattern. Notes:
    - batch_size=1 + custom collate because episode dicts are keyed tensor
      bundles; we yield the raw dict per pull and batch inside train_one_step.
    - pin_memory defaults False but can be enabled when the dataset returns
      only tensors (no PIL) — call with pin_memory=True when training on GPU.
    - persistent_workers=True so each worker pays the __init__ cost once
      (HF `load_dataset` + filter is a few seconds per worker).
    """
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=_unwrap_single,
    )
    while True:
        for ep in loader:
            yield ep


def flatten_episode_to_frames(ep: dict) -> dict:
    """Turn (T, ...) per-episode tensors into (T, ...) frame tensors for loss.

    Dataset now preprocesses images in workers (`preprocess=True` path), so
    `ep` contains `pixel_values`/`pixel_attention_mask`/`spatial_shapes` instead
    of raw PIL. Returns a nested dict under `vision_preprocessed` for the
    model's fast dispatch path.
    """
    return {
        "vision_preprocessed": {
            "pixel_values": ep["pixel_values"],                       # (T, N, P*P*C)
            "pixel_attention_mask": ep["pixel_attention_mask"],       # (T, N)
            "spatial_shapes": ep["spatial_shapes"],                   # (T, 2)
        },
        "proprio": ep["proprio"].float(),                             # (T, 83)
        "history": ep["history"].float(),                             # (T, K, 223)
        "dx":     ep["dx_bins"], "dy":     ep["dy_bins"],
        "click":  ep["clicks"],  "scroll": ep["scroll_bins"],
        "keys":   ep["key_events"], "done":  ep["dones"],
        "instruction": ep["instruction"],
    }


def _assemble_micro_batch(
    model: ActionPrimitivesACT,
    micro_eps: list[dict],
    device: torch.device,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Concatenate episodes into one big (sum_T, ...) micro-batch (Phase A).

    Returns (vision_preprocessed, text_rep, text_mask_rep, proprio, history, targets).
    Used by both train_one_step (under grad) and run_validation (under no_grad).
    """
    flat_pv: list = []
    flat_pm: list = []
    flat_ss: list = []
    flat_proprio: list = []
    flat_history: list = []
    flat_targets: dict[str, list] = {k: [] for k in ("dx", "dy", "click", "scroll", "keys", "done")}
    text_rep_parts: list = []
    text_mask_parts: list = []
    instructions = [e["instruction"] for e in micro_eps]
    with torch.no_grad():
        text_tokens = model.backbone.encode_text(instructions)    # (M, T_text, d)
    for ep_idx, ep in enumerate(micro_eps):
        flat = flatten_episode_to_frames(ep)
        vp = flat["vision_preprocessed"]
        T = vp["pixel_values"].shape[0]
        flat_pv.append(vp["pixel_values"])
        flat_pm.append(vp["pixel_attention_mask"])
        flat_ss.append(vp["spatial_shapes"])
        flat_proprio.append(flat["proprio"])
        flat_history.append(flat["history"])
        for k in flat_targets:
            flat_targets[k].append(flat[k])
        text_rep_parts.append(text_tokens[ep_idx:ep_idx + 1].expand(T, -1, -1))
        text_mask_parts.append(torch.ones(T, text_tokens.size(1), device=device))

    vision_preprocessed = {
        "pixel_values": torch.cat(flat_pv, dim=0),
        "pixel_attention_mask": torch.cat(flat_pm, dim=0),
        "spatial_shapes": torch.cat(flat_ss, dim=0),
    }
    text_rep = torch.cat(text_rep_parts, dim=0)
    text_mask_rep = torch.cat(text_mask_parts, dim=0)
    proprio = torch.cat(flat_proprio, dim=0).to(device, non_blocking=True)
    history = torch.cat(flat_history, dim=0).to(device, non_blocking=True)
    targets = {k: torch.cat(flat_targets[k], dim=0).to(device, non_blocking=True) for k in flat_targets}
    targets["done"] = targets["done"].float()
    return vision_preprocessed, text_rep, text_mask_rep, proprio, history, targets


def _assemble_micro_batch_b0(
    model: ActionPrimitivesACT,
    micro_eps: list[dict],
    device: torch.device,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """B0 micro-batch assembler.

    Mirrors `_assemble_micro_batch` but reads from the B0 episode schema:
    - ``ep["action_history"]`` instead of ``ep["history"]``
    - training targets nested under ``ep["action_label"]`` (with both quantized
      bin labels for click_left/click_right/keys/done AND continuous floats for
      dx/dy/scroll soft-CE)
    - per-frame ``ep["loss_mask"]`` returned as a separate tensor

    Returns (vision_preprocessed, text_rep, text_mask_rep, proprio, history,
    targets, loss_mask).
    """
    flat_pv: list = []
    flat_pm: list = []
    flat_ss: list = []
    flat_proprio: list = []
    flat_history: list = []
    flat_loss_mask: list = []
    target_keys = (
        "dx", "dy", "dx_continuous", "dy_continuous",
        "click_left", "click_right",
        "scroll", "scroll_continuous",
        "keys", "done",
    )
    flat_targets: dict[str, list] = {k: [] for k in target_keys}
    text_rep_parts: list = []
    text_mask_parts: list = []
    instructions = [e["instruction"] for e in micro_eps]
    with torch.no_grad():
        text_tokens = model.backbone.encode_text(instructions)    # (M, T_text, d)
    for ep_idx, ep in enumerate(micro_eps):
        T = ep["proprio"].shape[0]
        # Vision (preprocessed in worker)
        flat_pv.append(ep["pixel_values"])
        flat_pm.append(ep["pixel_attention_mask"])
        flat_ss.append(ep["spatial_shapes"])
        flat_proprio.append(ep["proprio"].float())
        flat_history.append(ep["action_history"].float())
        flat_loss_mask.append(ep["loss_mask"].float())
        label = ep["action_label"]
        flat_targets["dx"].append(label["dx_bins"])
        flat_targets["dy"].append(label["dy_bins"])
        flat_targets["dx_continuous"].append(label["dx_continuous"])
        flat_targets["dy_continuous"].append(label["dy_continuous"])
        flat_targets["click_left"].append(label["click_left"])
        flat_targets["click_right"].append(label["click_right"])
        flat_targets["scroll"].append(label["scroll_bins"])
        flat_targets["scroll_continuous"].append(label["scroll_continuous"])
        flat_targets["keys"].append(label["key_events"])
        flat_targets["done"].append(label["dones"])
        text_rep_parts.append(text_tokens[ep_idx:ep_idx + 1].expand(T, -1, -1))
        text_mask_parts.append(torch.ones(T, text_tokens.size(1), device=device))

    vision_preprocessed = {
        "pixel_values": torch.cat(flat_pv, dim=0),
        "pixel_attention_mask": torch.cat(flat_pm, dim=0),
        "spatial_shapes": torch.cat(flat_ss, dim=0),
    }
    text_rep = torch.cat(text_rep_parts, dim=0)
    text_mask_rep = torch.cat(text_mask_parts, dim=0)
    proprio = torch.cat(flat_proprio, dim=0).to(device, non_blocking=True)
    history = torch.cat(flat_history, dim=0).to(device, non_blocking=True)
    loss_mask = torch.cat(flat_loss_mask, dim=0).to(device, non_blocking=True)
    targets = {
        k: torch.cat(flat_targets[k], dim=0).to(device, non_blocking=True)
        for k in flat_targets
    }
    targets["done"] = targets["done"].float()
    return vision_preprocessed, text_rep, text_mask_rep, proprio, history, targets, loss_mask


_PHASE_A_HEAD_NAMES = ("dx", "dy", "click", "scroll", "keys", "done")
_PHASE_B0_HEAD_NAMES = ("dx", "dy", "click_left", "click_right", "scroll", "keys", "done")


def train_one_step(
    model: ActionPrimitivesACT,
    opt: torch.optim.Optimizer,
    iterator: Iterator[dict],
    head_weights: dict[str, float],
    device: torch.device,
    autocast_ctx: torch.autocast,
    lrs: list[float] | None = None,
    macro_batch_episodes: int = TRAIN.macro_batch_episodes,
    micro_batch_episodes: int = TRAIN.micro_batch_episodes,
    phase: str = "a",
) -> dict[str, float]:
    """Execute one macro-batch step: gradient accumulation over N micro-batches,
    optimizer step, and return per-head losses. Caller manages LR schedule,
    grad clipping, logging, and checkpointing.

    If ``lrs`` is provided, each value is written to the corresponding param
    group's ``lr`` before the optimizer step (applied deterministically at the
    start, before any backward pass, so the scheduled LR is the one used for
    this step's update).

    ``macro_batch_episodes`` / ``micro_batch_episodes`` default to TRAIN config
    but can be overridden by the caller (e.g. to shrink micro batch for GPU
    memory pressure). The macro:micro ratio determines how many gradient
    accumulation passes happen per optimizer step.

    ``phase`` selects between Phase A (legacy `total_loss`) and Phase B0
    (`total_loss_b0` with loss_mask + soft-CE on dx/dy/scroll + dual click
    heads).

    Returns a dict of per-head float losses plus ``total`` and ``grad_norm``.
    """
    if lrs is not None:
        for pg, lr in zip(opt.param_groups, lrs, strict=True):
            pg["lr"] = lr

    opt.zero_grad(set_to_none=True)
    head_names = _PHASE_B0_HEAD_NAMES if phase == "b0" else _PHASE_A_HEAD_NAMES
    step_per_head_loss = {k: 0.0 for k in head_names}
    n_micros = macro_batch_episodes // micro_batch_episodes
    for _micro in range(n_micros):
        micro_eps = [next(iterator) for _ in range(micro_batch_episodes)]
        with autocast_ctx:
            if phase == "b0":
                vp, tr, tm, pr, hs, tgts, lm = _assemble_micro_batch_b0(model, micro_eps, device)
                out = model(vp, tr, tm, pr, hs)
                loss, per_head = total_loss_b0(out.head_logits, tgts, head_weights, lm)
            else:
                vp, tr, tm, pr, hs, tgts = _assemble_micro_batch(model, micro_eps, device)
                out = model(vp, tr, tm, pr, hs)
                loss, per_head = total_loss(out.head_logits, tgts, head_weights)

        (loss / macro_batch_episodes * micro_batch_episodes).backward()
        for k, v in per_head.items():
            step_per_head_loss[k] += float(v.detach()) / n_micros

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN.grad_clip_norm)
    opt.step()

    step_per_head_loss["total"] = sum(step_per_head_loss[k] for k in head_names)
    step_per_head_loss["grad_norm"] = float(grad_norm)
    return step_per_head_loss


def run_validation(
    model: ActionPrimitivesACT,
    val_ds: PhaseAEpisodeDataset | PhaseB0EpisodeDataset,
    device: torch.device,
    autocast_ctx: torch.autocast,
    head_weights: dict[str, float],
    micro_batch_episodes: int,
    max_episodes: int | None = None,
    phase: str = "a",
) -> dict[str, float]:
    """Mean per-head + total loss on the validation split.

    Iterates val_ds sequentially (no shuffle) in chunks of ``micro_batch_episodes``
    under torch.no_grad + autocast. If ``max_episodes`` is set, stops early after
    that many episodes (useful when the full val set is large).

    When ``phase == "b0"``, additional B0 diagnostics are accumulated (per-class
    click recall, soft-CE failure-mode summaries, bin-10 frequency on dx/dy)
    and returned alongside the loss values. The caller decides how to log them.
    """
    was_training = model.training
    model.train(False)
    head_names = _PHASE_B0_HEAD_NAMES if phase == "b0" else _PHASE_A_HEAD_NAMES
    total_losses = {k: 0.0 for k in head_names}
    total_sum = 0.0
    n = len(val_ds) if max_episodes is None else min(max_episodes, len(val_ds))
    idx = 0
    # B0 diagnostic accumulators: collect tensors across micro-batches (CPU)
    # and compute once at the end so per-class denominators are correct
    # (e.g. press-class recall must average over real press samples, not
    # batches that may contain zero of them).
    b0_buffers: dict[str, list[torch.Tensor]] = {
        "click_left_preds": [], "click_left_targets": [],
        "click_right_preds": [], "click_right_targets": [],
        "dx_logits": [], "dx_continuous": [],
        "dy_logits": [], "dy_continuous": [],
    }
    with torch.no_grad():
        while idx < n:
            batch_size = min(micro_batch_episodes, n - idx)
            micro_eps = [val_ds[i] for i in range(idx, idx + batch_size)]
            idx += batch_size
            with autocast_ctx:
                if phase == "b0":
                    vp, tr, tm, pr, hs, tgts, lm = _assemble_micro_batch_b0(
                        model, micro_eps, device,
                    )
                    out = model(vp, tr, tm, pr, hs)
                    loss, per_head = total_loss_b0(out.head_logits, tgts, head_weights, lm)
                else:
                    vp, tr, tm, pr, hs, tgts = _assemble_micro_batch(model, micro_eps, device)
                    out = model(vp, tr, tm, pr, hs)
                    loss, per_head = total_loss(out.head_logits, tgts, head_weights)
            for k, v in per_head.items():
                total_losses[k] += float(v.detach()) * batch_size
            total_sum += float(loss.detach()) * batch_size

            if phase == "b0":
                # Stash logits + targets on CPU for end-of-pass diagnostics.
                # Cast to float32 so soft_ce_diagnostics is numerically stable
                # under bf16 autocast.
                hl = out.head_logits
                b0_buffers["click_left_preds"].append(hl["click_left"].argmax(dim=-1).detach().cpu())
                b0_buffers["click_left_targets"].append(tgts["click_left"].detach().cpu())
                b0_buffers["click_right_preds"].append(hl["click_right"].argmax(dim=-1).detach().cpu())
                b0_buffers["click_right_targets"].append(tgts["click_right"].detach().cpu())
                b0_buffers["dx_logits"].append(hl["dx"].detach().float().cpu())
                b0_buffers["dx_continuous"].append(tgts["dx_continuous"].detach().float().cpu())
                b0_buffers["dy_logits"].append(hl["dy"].detach().float().cpu())
                b0_buffers["dy_continuous"].append(tgts["dy_continuous"].detach().float().cpu())
    if was_training:
        model.train(True)
    result = {k: v / max(1, n) for k, v in total_losses.items()}
    result["total"] = total_sum / max(1, n)

    if phase == "b0" and b0_buffers["click_left_preds"]:
        cl_p = torch.cat(b0_buffers["click_left_preds"]).flatten()
        cl_t = torch.cat(b0_buffers["click_left_targets"]).flatten()
        cr_p = torch.cat(b0_buffers["click_right_preds"]).flatten()
        cr_t = torch.cat(b0_buffers["click_right_targets"]).flatten()
        dx_logits = torch.cat(b0_buffers["dx_logits"], dim=0)         # (N, num_bins)
        dx_target = torch.cat(b0_buffers["dx_continuous"]).flatten()  # (N,)
        dy_logits = torch.cat(b0_buffers["dy_logits"], dim=0)
        dy_target = torch.cat(b0_buffers["dy_continuous"]).flatten()

        mouse_centers = torch.tensor(MOUSE_BIN_CENTERS, dtype=torch.float32)
        left_recall = per_class_click_recall(cl_p, cl_t)
        right_recall = per_class_click_recall(cr_p, cr_t)
        soft_dx = soft_ce_diagnostics(dx_logits, dx_target, mouse_centers)
        soft_dy = soft_ce_diagnostics(dy_logits, dy_target, mouse_centers)
        bin10_dx = bin_10_frequency(dx_logits.argmax(dim=-1), dx_target, mouse_centers)
        bin10_dy = bin_10_frequency(dy_logits.argmax(dim=-1), dy_target, mouse_centers)

        # Scoped under "diag/" to keep the loss namespace clean for the
        # caller's existing `f"val/{k}"` log shaping.
        result.update({
            "diag/click_left/recall_idle":     left_recall["recall_idle"],
            "diag/click_left/recall_press":    left_recall["recall_press"],
            "diag/click_left/recall_release":  left_recall["recall_release"],
            "diag/click_right/recall_idle":    right_recall["recall_idle"],
            "diag/click_right/recall_press":   right_recall["recall_press"],
            "diag/click_right/recall_release": right_recall["recall_release"],
            "diag/dx/sign_acc":          soft_dx["sign_acc"],
            "diag/dx/entropy":           soft_dx["entropy_mean"],
            "diag/dx/wrong_sign_mass":   soft_dx["wrong_sign_mass_mean"],
            "diag/dx/ev_l1":             soft_dx["ev_l1_mean"],
            "diag/dy/sign_acc":          soft_dy["sign_acc"],
            "diag/dy/entropy":           soft_dy["entropy_mean"],
            "diag/dy/wrong_sign_mass":   soft_dy["wrong_sign_mass_mean"],
            "diag/dy/ev_l1":             soft_dy["ev_l1_mean"],
            "diag/dx/bin_10_freq":       bin10_dx,
            "diag/dy/bin_10_freq":       bin10_dy,
        })
        # TODO(B0): instruction-zeroing probe — see Task 23.
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="a", choices=["a", "b0"],
                        help="Training phase. 'a' (default) uses PhaseAEpisodeDataset + "
                             "total_loss; 'b0' uses PhaseB0EpisodeDataset + total_loss_b0 "
                             "(loss_mask + soft-CE + dual click heads) and emits B0 val "
                             "diagnostics.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Local parquet dataset directory. Mutually exclusive with --hf-data-repo.")
    parser.add_argument("--epochs", type=int, default=TRAIN.phase_a_epochs)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="checkpoints/phase-a")
    parser.add_argument("--wandb-project", type=str, default="cu-vla-exp6")
    parser.add_argument("--wandb-run-name", type=str, default="phase-a-lclick")
    parser.add_argument("--wandb-mode", type=str, default=None,
                        choices=[None, "online", "offline", "disabled"],
                        help="Override WANDB_MODE env var. 'disabled' short-circuits wandb (use for smoke tests).")
    parser.add_argument("--hf-upload-repo", type=str, default=None, help="HF repo for checkpoint upload")
    parser.add_argument("--hf-data-repo", type=str, default=None, help="HF dataset repo; overrides --data-dir")
    parser.add_argument("--resume", type=str, default=None, help="path to .pt checkpoint to resume from")
    parser.add_argument("--macro-batch-episodes", type=int, default=TRAIN.macro_batch_episodes,
                        help=f"Episodes per optimizer step (default {TRAIN.macro_batch_episodes}).")
    parser.add_argument("--micro-batch-episodes", type=int, default=TRAIN.micro_batch_episodes,
                        help=f"Episodes per forward/backward pass (default {TRAIN.micro_batch_episodes}). "
                             "Drop to 1 or 2 on 24 GB GPUs to avoid OOM.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker processes for CPU-side prefetch. "
                             "0 = simple single-process generator (default, matches "
                             "behaviour for local tests). 4 is a good starting point "
                             "on HF Jobs L4 / L40S to hide JPEG decode + processor "
                             "cost behind the GPU forward.")
    parser.add_argument("--ckpt-every-steps", type=int, default=TRAIN.ckpt_every_steps,
                        help=f"Save a step_NNNNN.pt every N steps (default {TRAIN.ckpt_every_steps}). "
                             "Not uploaded to HF (only best.pt and final.pt are).")
    parser.add_argument("--eval-every-steps", type=int, default=0,
                        help="Run validation every N steps and log val/* to wandb. "
                             "0 = never (default). Setting this enables best.pt tracking.")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Stop training after N consecutive eval cycles without "
                             "improvement in val loss. 0 = disabled (default). Requires "
                             "--eval-every-steps > 0.")
    parser.add_argument("--val-episodes", type=int, default=None,
                        help="Cap val-eval episodes (default: full val set). Lower for "
                             "faster eval on large val sets.")
    args = parser.parse_args()
    if args.macro_batch_episodes % args.micro_batch_episodes != 0:
        parser.error(
            f"--macro-batch-episodes ({args.macro_batch_episodes}) must be a multiple of "
            f"--micro-batch-episodes ({args.micro_batch_episodes})."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Data — exactly one of --data-dir or --hf-data-repo must be provided
    if (args.data_dir is None) == (args.hf_data_repo is None):
        parser.error(
            "exactly one of --data-dir or --hf-data-repo must be provided "
            f"(got data_dir={args.data_dir!r}, hf_data_repo={args.hf_data_repo!r})"
        )
    if args.hf_data_repo is not None:
        from experiments.action_primitives.hf_sync import download_hf_dataset
        data_dir = download_hf_dataset(args.hf_data_repo)
    else:
        data_dir = Path(args.data_dir)
    if args.phase == "b0":
        train_ds = PhaseB0EpisodeDataset(data_dir, split="train")
        val_ds = PhaseB0EpisodeDataset(data_dir, split="val")
    else:
        train_ds = PhaseAEpisodeDataset(data_dir, split="train")
        val_ds = PhaseAEpisodeDataset(data_dir, split="val")
    print(f"[phase={args.phase}] train episodes: {len(train_ds)}   val: {len(val_ds)}")

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
    steps_per_epoch = math.ceil(len(train_ds) / args.macro_batch_episodes)
    max_steps = steps_per_epoch * args.epochs
    print(f"steps/epoch: {steps_per_epoch}  total: {max_steps}  "
          f"macro={args.macro_batch_episodes}  micro={args.micro_batch_episodes}")

    # Uniform per-head weights. Phase A uses the legacy 5-way "click" head;
    # Phase B0 splits it into "click_left" / "click_right" 3-way heads.
    if args.phase == "b0":
        head_weights = {
            "dx": 1.0, "dy": 1.0,
            "click_left": 1.0, "click_right": 1.0,
            "scroll": 1.0, "keys": 1.0, "done": 1.0,
        }
    else:
        head_weights = {"dx": 1.0, "dy": 1.0, "click": 1.0, "scroll": 1.0, "keys": 1.0, "done": 1.0}

    # bf16 amp
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, mode=args.wandb_mode, config={
        "phase": args.phase,
        "epochs": args.epochs,
        "macro_batch": args.macro_batch_episodes,
        "micro_batch": args.micro_batch_episodes,
        "num_workers": args.num_workers,
        "lr_trunk": TRAIN.lr_trunk,
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

    if args.num_workers > 0:
        print(f"Using prefetching loader with num_workers={args.num_workers}")
        iterator = prefetching_loader(
            train_ds, num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
    else:
        iterator = infinite_loader(train_ds, seed=42)

    def _upload_to_hf(path: Path) -> None:
        """Best-effort upload; log but don't crash the run if it fails."""
        if not args.hf_upload_repo:
            return
        try:
            from huggingface_hub import upload_file
            upload_file(path_or_fileobj=str(path), path_in_repo=path.name,
                        repo_id=args.hf_upload_repo, repo_type="model")
            print(f"[hf-upload] {path.name} → {args.hf_upload_repo}")
        except Exception as e:
            print(f"[hf-upload failed] {path.name}: {e}")

    model.train(True)
    step = start_step
    best_val_loss = float("inf")
    patience_counter = 0
    early_stopped = False
    head_loss_keys = (
        _PHASE_B0_HEAD_NAMES if args.phase == "b0" else _PHASE_A_HEAD_NAMES
    )
    while step < max_steps and not early_stopped:
        lrs = [cosine_lr(step, max_steps, TRAIN.warmup_steps, base, TRAIN.cosine_min_lr_frac)
               for base in (TRAIN.lr_trunk, TRAIN.lr_lora)]
        metrics = train_one_step(
            model, opt, iterator, head_weights, device, autocast,
            lrs=lrs,
            macro_batch_episodes=args.macro_batch_episodes,
            micro_batch_episodes=args.micro_batch_episodes,
            phase=args.phase,
        )
        total_loss_avg = metrics["total"]
        grad_norm = metrics["grad_norm"]

        # Log
        log = {"step": step, "loss/total": total_loss_avg, "grad_norm": grad_norm,
               "lr/trunk": opt.param_groups[0]["lr"], "lr/lora": opt.param_groups[1]["lr"]}
        for k in head_loss_keys:
            log[f"loss/{k}"] = metrics[k]
        wandb.log(log)
        if step % 20 == 0:
            print(f"[step {step}] loss={total_loss_avg:.4f}  grad_norm={grad_norm:.3f}")

        # Periodic local checkpoint (not uploaded; for in-VM crash recovery only).
        if step % args.ckpt_every_steps == 0 and step > 0:
            ckpt_path = out_dir / f"step_{step:05d}.pt"
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step}, ckpt_path)
            print(f"[ckpt] {ckpt_path}")

        # Validation + best.pt tracking + early stopping.
        if args.eval_every_steps > 0 and step > 0 and step % args.eval_every_steps == 0:
            val_metrics = run_validation(
                model, val_ds, device, autocast, head_weights,
                micro_batch_episodes=args.micro_batch_episodes,
                max_episodes=args.val_episodes,
                phase=args.phase,
            )
            # Loss metrics get logged as val/...; B0 diagnostics carry a
            # "diag/" prefix from run_validation, so we surface them under
            # val/diag/... to keep the wandb panel clean.
            val_log: dict[str, float] = {}
            for k, v in val_metrics.items():
                val_log[f"val/{k}"] = v
            val_log["step"] = step
            wandb.log(val_log)
            print(f"[val step {step}] total={val_metrics['total']:.4f}  "
                  + "  ".join(f"{k}={val_metrics[k]:.4f}" for k in head_loss_keys))

            if val_metrics["total"] < best_val_loss - 1e-6:
                best_val_loss = val_metrics["total"]
                patience_counter = 0
                best_path = out_dir / "best.pt"
                torch.save(
                    {"model": model.state_dict(), "optimizer": opt.state_dict(),
                     "step": step, "val_loss": best_val_loss},
                    best_path,
                )
                print(f"[best] val_loss={best_val_loss:.4f} at step {step}")
                _upload_to_hf(best_path)
            elif args.early_stop_patience > 0:
                patience_counter += 1
                print(f"[patience] {patience_counter}/{args.early_stop_patience} "
                      f"(best val_loss={best_val_loss:.4f})")
                if patience_counter >= args.early_stop_patience:
                    print(f"[early-stop] val loss has not improved for "
                          f"{args.early_stop_patience} consecutive evals; stopping at step {step}")
                    early_stopped = True

        step += 1

    # Final checkpoint — always uploaded to HF if configured.
    final_path = out_dir / "final.pt"
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step}, final_path)
    print(f"done, wrote {final_path} (early_stopped={early_stopped})")
    _upload_to_hf(final_path)


if __name__ == "__main__":
    main()
