"""TinyCNN policy network for reactive clicks.

4 conv layers with stride-2 downsampling, FC bottleneck, and 3 output heads:
- dx (1 value): horizontal delta in pixels, tanh-scaled to [-max_delta, +max_delta]
- dy (1 value): vertical delta in pixels, tanh-scaled to [-max_delta, +max_delta]
- btn_logits (3 classes): no_change / mouse_down / mouse_up

~2.2M parameters. Runs at <2ms per forward pass on M1 CPU.
"""

import torch
import torch.nn as nn

from .config import ENV, ACTION, MODEL


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        in_channels = 3
        channels = MODEL.conv_channels  # (32, 64, 64, 128)

        layers = []
        for out_channels in channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)

        # After 4x stride-2 on 128x128 input: 8x8 spatial
        flat_dim = channels[-1] * (ENV.obs_size // 16) ** 2  # 128 * 64 = 8192
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, MODEL.fc_dim),
            nn.ReLU(inplace=True),
        )

        # Regression heads for dx, dy (continuous pixel deltas)
        self.head_dx = nn.Linear(MODEL.fc_dim, 1)
        self.head_dy = nn.Linear(MODEL.fc_dim, 1)
        # Classification head for button
        self.head_btn = nn.Linear(MODEL.fc_dim, ACTION.num_btn_classes)

        self._max_delta = ACTION.max_delta_px

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) float32 tensor, normalized to [0, 1].

        Returns:
            dx: (B, 1) continuous pixel delta, tanh-scaled to [-max_delta, +max_delta]
            dy: (B, 1) continuous pixel delta, tanh-scaled to [-max_delta, +max_delta]
            btn_logits: (B, num_btn_classes)
        """
        features = self.conv(x)
        features = features.flatten(1)
        features = self.fc(features)

        dx = torch.tanh(self.head_dx(features)) * self._max_delta
        dy = torch.tanh(self.head_dy(features)) * self._max_delta
        btn_logits = self.head_btn(features)

        return dx, dy, btn_logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
