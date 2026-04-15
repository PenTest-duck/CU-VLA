"""Vision backbone for Experiment 5: Mini Text Editor.

ResNet18 backbone outputs (batch, 108, 256) from 384x288 input —
a natural 9x12 spatial grid with NO adaptive pooling.
2D sinusoidal positional encoding applied separately by model.py
after FiLM modulation.
"""

import math

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from .config import MODEL


class SinusoidalPositionEncoding2D(nn.Module):
    """2D sinusoidal positional encoding for spatial feature grids.

    Encodes row (y) and column (x) positions independently using the
    standard sinusoidal formula, concatenated along the feature dimension.
    First d_model//2 dims encode the row index; second d_model//2 dims
    encode the column index.
    """

    def __init__(self, d_model: int, max_h: int = 20, max_w: int = 20) -> None:
        super().__init__()
        half_d = d_model // 2
        # Div term: exp(-2i/d * ln(10000)) for i in [0, half_d)
        div_term = torch.exp(
            torch.arange(0, half_d, 2, dtype=torch.float32)
            * -(math.log(10000.0) / half_d)
        )  # (half_d // 2,)

        pe = torch.zeros(1, max_h * max_w, d_model)

        for row in range(max_h):
            for col in range(max_w):
                pos_idx = row * max_w + col
                # Row (y) encoding -> first half_d dims
                pe[0, pos_idx, 0:half_d:2] = torch.sin(row * div_term)
                pe[0, pos_idx, 1:half_d:2] = torch.cos(row * div_term)
                # Col (x) encoding -> second half_d dims
                pe[0, pos_idx, half_d::2] = torch.sin(col * div_term)
                pe[0, pos_idx, half_d + 1::2] = torch.cos(col * div_term)

        self.register_buffer("pe", pe)  # (1, max_h*max_w, d_model)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Add 2D positional encoding for an h x w grid.

        Args:
            x: (B, h*w, d_model) token features.
            h: grid height.
            w: grid width.

        Returns:
            (B, h*w, d_model) with positional encoding added.
        """
        return x + self.pe[:, : h * w, :]


class ResNet18Backbone(nn.Module):
    """Fine-tunable ResNet18 backbone for 384x288 input.

    Produces (B, 108, 256) — a 9x12 spatial grid of 256-dim tokens.
    No adaptive pooling; the 9x12 grid arises naturally from ResNet18's
    stride-32 downsampling of 288x384 input.
    """

    def __init__(self) -> None:
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove avgpool and fc layers — keep conv backbone only
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.proj = nn.Linear(MODEL.backbone_feature_dim, MODEL.d_model)
        self.pos_enc = SinusoidalPositionEncoding2D(
            MODEL.d_model,
            max_h=MODEL.vision_grid_h,
            max_w=MODEL.vision_grid_w,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 288, 384) -> (B, 108, 256).

        NOTE: Positional encoding is NOT added here. The model.py
        adds PE after FiLM modulation of the visual tokens.
        """
        feat = self.features(x)              # (B, 512, 9, 12)
        B, C, H, W = feat.shape
        feat = feat.reshape(B, C, H * W)     # (B, 512, 108)
        feat = feat.permute(0, 2, 1)         # (B, 108, 512)
        return self.proj(feat)               # (B, 108, 256)
