#!/bin/bash
# RunPod pod bootstrap: clone repo, install deps, download data from HF.
#
# Prerequisites:
#   - RunPod PyTorch template (has CUDA + PyTorch pre-installed)
#   - HF_TOKEN env var set in pod config (or run `huggingface-cli login` manually)
#   - Set REPO_URL and HF_DATA_REPO below
#
# Usage:
#   bash scripts/setup_runpod.sh

set -e

REPO_URL="${REPO_URL:-https://github.com/PenTest-duck/CU-VLA.git}"
HF_DATA_REPO="${HF_DATA_REPO:-PenTest-duck/cu-vla-data}"
WORKDIR="/workspace/cu-vla"

echo "=== RunPod Setup ==="

# Clone repo
if [ ! -d "$WORKDIR" ]; then
    echo "Cloning repo..."
    git clone "$REPO_URL" "$WORKDIR"
else
    echo "Repo exists, pulling latest..."
    cd "$WORKDIR" && git pull && cd -
fi

cd "$WORKDIR"

# Install uv + deps
echo "Installing dependencies..."
pip install uv 2>/dev/null || true
uv sync

# HF login
if [ -n "$HF_TOKEN" ]; then
    echo "Logging in to HuggingFace..."
    huggingface-cli login --token "$HF_TOKEN"
else
    echo "WARNING: HF_TOKEN not set. Run 'huggingface-cli login' manually."
fi

# Download dataset
echo "Downloading dataset from HF..."
uv run python experiments/act_drag_label/hf_sync.py download-data --repo "$HF_DATA_REPO"

echo ""
echo "=== Ready ==="
echo "cd $WORKDIR"
echo ""
echo "Train (single run):"
echo "  uv run python experiments/act_drag_label/train.py --backbone resnet18 --chunk-size 10 --device cuda"
echo ""
echo "Full Phase 1 ablation:"
echo "  for bb in resnet18 dinov2-vits14 siglip2-base; do"
echo "    uv run python experiments/act_drag_label/train.py --backbone \$bb --chunk-size 10 --device cuda"
echo "  done"
echo ""
echo "Upload checkpoints when done:"
echo "  uv run python experiments/act_drag_label/hf_sync.py upload-checkpoints"
