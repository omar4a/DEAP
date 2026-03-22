"""Compact single-target EEGNet for 2 s DEAP windows."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from . import config
except ImportError:
    import config


class EEGNetBinary(nn.Module):
    """EEGNet classifier for one binary target."""

    def __init__(self):
        super().__init__()
        f1 = config.EEGNET_F1
        depth = config.EEGNET_D
        f2 = config.EEGNET_F2
        channels = config.N_CHANNELS
        dropout = config.DROPOUT

        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, config.TEMPORAL_KERNEL), padding="same", bias=False),
            nn.BatchNorm2d(f1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, depth * f1, kernel_size=(channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(depth * f1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(
                depth * f1,
                depth * f1,
                kernel_size=(1, 16),
                padding="same",
                groups=depth * f1,
                bias=False,
            ),
            nn.Conv2d(depth * f1, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        self.feature_dim = self._compute_feature_dim()
        self.classifier = nn.Linear(self.feature_dim, 2)

    def _compute_feature_dim(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, config.N_CHANNELS, config.WINDOW_SAMPLES)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
        return int(x.flatten(1).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_model() -> EEGNetBinary:
    return EEGNetBinary()
