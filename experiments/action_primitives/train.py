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


def train_one_step(
    model: ActionPrimitivesACT,
    opt: torch.optim.Optimizer,
    iterator: Iterator[dict],
    head_weights: dict[str, float],
    device: torch.device,
    autocast_ctx: torch.autocast,
    lrs: list[float] | None = None,
) -> dict[str, float]:
    """Execute one macro-batch step: gradient accumulation over N micro-batches,
    optimizer step, and return per-head losses. Caller manages LR schedule,
    grad clipping, logging, and checkpointing.

    If ``lrs`` is provided, each value is written to the corresponding param
    group's ``lr`` before the optimizer step (applied deterministically at the
    start, before any backward pass, so the scheduled LR is the one used for
    this step's update).

    Returns a dict of per-head float losses plus ``total`` and ``grad_norm``.
    """
    if lrs is not None:
        for pg, lr in zip(opt.param_groups, lrs):
            pg["lr"] = lr

    opt.zero_grad(set_to_none=True)
    step_per_head_loss = {k: 0.0 for k in ("dx", "dy", "click", "scroll", "keys", "done")}
    n_micros = TRAIN.macro_batch_episodes // TRAIN.micro_batch_episodes
    for _micro in range(n_micros):
        micro_eps = [next(iterator) for _ in range(TRAIN.micro_batch_episodes)]
        # Concatenate frames across the micro-batch
        with autocast_ctx:
            flat_images: list = []
            flat_proprio: list = []
            flat_history: list = []
            flat_targets = {k: [] for k in ("dx", "dy", "click", "scroll", "keys", "done")}
            text_rep_parts: list = []
            text_mask_parts: list = []
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
                text_rep_parts.append(text_tokens[ep_idx:ep_idx+1].expand(T, -1, -1))
                text_mask_parts.append(torch.ones(T, text_tokens.size(1), device=device))

            text_rep = torch.cat(text_rep_parts, dim=0)
            text_mask_rep = torch.cat(text_mask_parts, dim=0)
            proprio = torch.cat(flat_proprio, dim=0).to(device)
            history = torch.cat(flat_history, dim=0).to(device)
            targets = {k: torch.cat(flat_targets[k], dim=0).to(device) for k in flat_targets}
            targets["done"] = targets["done"].float()

            out = model(flat_images, text_rep, text_mask_rep, proprio, history)
            loss, per_head = total_loss(out.head_logits, targets, head_weights)

        (loss / TRAIN.macro_batch_episodes * TRAIN.micro_batch_episodes).backward()
        for k, v in per_head.items():
            step_per_head_loss[k] += float(v.detach()) / n_micros

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN.grad_clip_norm)
    opt.step()

    step_per_head_loss["total"] = sum(
        step_per_head_loss[k] for k in ("dx", "dy", "click", "scroll", "keys", "done")
    )
    step_per_head_loss["grad_norm"] = float(grad_norm)
    return step_per_head_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
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

    # Uniform per-head weights for Phase A; Q2's loss rebalancing is Phase B.
    head_weights = {"dx": 1.0, "dy": 1.0, "click": 1.0, "scroll": 1.0, "keys": 1.0, "done": 1.0}

    # bf16 amp
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, mode=args.wandb_mode, config={
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
        lrs = [cosine_lr(step, max_steps, TRAIN.warmup_steps, base, TRAIN.cosine_min_lr_frac)
               for base in (TRAIN.lr_trunk, TRAIN.lr_lora)]
        metrics = train_one_step(model, opt, iterator, head_weights, device, autocast, lrs=lrs)
        total_loss_avg = metrics["total"]
        grad_norm = metrics["grad_norm"]

        # Log
        log = {"step": step, "loss/total": total_loss_avg, "grad_norm": grad_norm,
               "lr/trunk": opt.param_groups[0]["lr"], "lr/lora": opt.param_groups[1]["lr"]}
        for k in ("dx", "dy", "click", "scroll", "keys", "done"):
            log[f"loss/{k}"] = metrics[k]
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
