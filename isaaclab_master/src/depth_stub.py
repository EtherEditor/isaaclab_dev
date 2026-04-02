"""
Lightweight CNN depth-image embedding.
Filename: depth_stub.py

Converts a (4, 64, 64) RGBD depth image to a 64-dim feature vector.
Trained end-to-end with PPO; weights randomly initialised here.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DepthCNNEmbedder(nn.Module):
    """
    3-layer CNN that maps (B, 4, 64, 64) → (B, 64).

    Architecture
    ------------
    Conv(4→16, k=5, s=2)  → BN → ReLU  → 30×30
    Conv(16→32, k=3, s=2) → BN → ReLU  → 14×14
    Conv(32→64, k=3, s=2) → BN → ReLU  → 6×6
    AdaptiveAvgPool(1×1)   → Flatten    → (B, 64)
    """

    OUTPUT_DIM: int = 64

    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 4, 64, 64) RGBD image tensor, GPU-resident.
        Returns:
            feat: (B, 64) embedding vector.
        """
        return self.net(x).flatten(1)   # (B, 64)