"""Vision backbone wrappers for ACT Drag-and-Label experiment.

All backbones output (batch, 49, d_model=256) so the downstream
transformer is unchanged across backbone ablations.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from .config import MODEL


class ResNet18Backbone(nn.Module):
    """Fine-tunable ResNet18 backbone.

    At 224x224 input produces (B, 512, 7, 7) = 49 spatial tokens,
    projected to (B, 49, 256).
    """

    def __init__(self) -> None:
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove avgpool and fc layers
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.proj = nn.Linear(
            MODEL.backbone_feature_dims["resnet18"], MODEL.d_model
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) -> (B, 49, 256)."""
        feat = self.features(x)                  # (B, 512, 7, 7)
        B, C, H, W = feat.shape
        feat = feat.reshape(B, C, H * W)         # (B, 512, 49)
        feat = feat.permute(0, 2, 1)             # (B, 49, 512)
        return self.proj(feat)                    # (B, 49, 256)


class DINOv2Backbone(nn.Module):
    """Frozen DINOv2 ViT-S/14 backbone.

    Extracts patch tokens, pools to 7x7 spatial grid, projects to 256.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14"
        )
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.proj = nn.Linear(
            MODEL.backbone_feature_dims["dinov2-vits14"], MODEL.d_model
        )

    @torch.no_grad()
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from frozen DINOv2."""
        out = self.model.forward_features(x)
        return out["x_norm_patchtokens"]  # (B, 256, 384) for 224x224

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) -> (B, 49, 256)."""
        tokens = self._extract_features(x)       # (B, N, 384)
        B, N, C = tokens.shape
        H = W = int(N ** 0.5)                    # 16 for 224/14
        feat = tokens.permute(0, 2, 1)           # (B, 384, N)
        feat = feat.reshape(B, C, H, W)          # (B, 384, 16, 16)
        feat = self.pool(feat)                    # (B, 384, 7, 7)
        feat = feat.reshape(B, C, 49)            # (B, 384, 49)
        feat = feat.permute(0, 2, 1)             # (B, 49, 384)
        return self.proj(feat)                    # (B, 49, 256)


class SigLIP2Backbone(nn.Module):
    """Frozen SigLIP2 backbone.

    Extracts patch tokens (skip CLS), pools to 7x7 spatial grid,
    projects to 256.
    """

    def __init__(self) -> None:
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.proj = nn.Linear(
            MODEL.backbone_feature_dims["siglip2-base"], MODEL.d_model
        )

    @torch.no_grad()
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from frozen SigLIP2 (skip CLS)."""
        out = self.model.vision_model(pixel_values=x)
        return out.last_hidden_state[:, 1:, :]  # (B, 196, 768) for 14x14

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) -> (B, 49, 256)."""
        tokens = self._extract_features(x)       # (B, 196, 768)
        B, N, C = tokens.shape
        H = W = int(N ** 0.5)                    # 14 for 224/16
        feat = tokens.permute(0, 2, 1)           # (B, 768, 196)
        feat = feat.reshape(B, C, H, W)          # (B, 768, 14, 14)
        feat = self.pool(feat)                    # (B, 768, 7, 7)
        feat = feat.reshape(B, C, 49)            # (B, 768, 49)
        feat = feat.permute(0, 2, 1)             # (B, 49, 768)
        return self.proj(feat)                    # (B, 49, 256)


def build_backbone(name: str) -> nn.Module:
    """Factory function to create a vision backbone by name.

    Args:
        name: One of 'resnet18', 'dinov2-vits14', 'siglip2-base'.

    Returns:
        nn.Module outputting (batch, 49, 256).
    """
    if name == "resnet18":
        return ResNet18Backbone()
    elif name == "dinov2-vits14":
        return DINOv2Backbone()
    elif name == "siglip2-base":
        return SigLIP2Backbone()
    else:
        raise ValueError(f"Unknown backbone: {name}")
