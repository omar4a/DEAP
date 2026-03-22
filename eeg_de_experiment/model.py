"""EEGNet adapted for Differential Entropy input.

Input shape:  (B, 1, 32, 300)
  — 32 EEG channels, 300 = 5 bands × 60 time steps (bands vary slowest)
Output: (valence_logits, arousal_logits)  each (B, 2)

Differences from raw-EEG EEGNet:
  - Temporal conv kernel: (1, 50) instead of (1, 64)  — scaled for 300-pt input
  - DROPOUT = 0.5 (from config) instead of 0.25
  - Everything else identical: depthwise spatial, separable, dual heads
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import config


class EEGNetDE(nn.Module):
    """EEGNet for DE features. Input: (B, 1, 32, 300)."""

    def __init__(self):
        super().__init__()
        F1 = config.EEGNET_F1    # 8
        D  = config.EEGNET_D     # 2
        F2 = config.EEGNET_F2    # 16
        C  = config.N_CHANNELS   # 32
        p  = config.DROPOUT      # 0.5

        # Block 1 — Temporal convolution
        # kernel (1,50), padding='same' keeps T=300
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 50), padding='same', bias=False),
            nn.BatchNorm2d(F1),
        )

        # Block 2 — Depthwise spatial convolution (collapses channel dim)
        # After AvgPool(1,4): T = 300/4 = 75
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, D * F1, kernel_size=(C, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p),
        )

        # Block 3 — Separable convolution
        # DWConv padding='same' keeps T=75; AvgPool(1,8): T = floor(75/8) = 9
        self.block3 = nn.Sequential(
            nn.Conv2d(D * F1, D * F1, kernel_size=(1, 16), padding='same',
                      groups=D * F1, bias=False),
            nn.Conv2d(D * F1, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p),
        )

        # Infer flattened feature dim: 16 * 1 * floor(75/8) = 16*9 = 144
        self._feature_dim = self._compute_feature_dim()

        # Dual classification heads
        self.valence_head = nn.Linear(self._feature_dim, 2)
        self.arousal_head = nn.Linear(self._feature_dim, 2)

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[EEGNetDE] Total params: {total:,}  Trainable: {trainable:,}  "
              f"Feature dim: {self._feature_dim}")

    def _compute_feature_dim(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, config.N_CHANNELS, config.DE_SEQ_LEN)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
        return int(x.flatten(1).shape[1])

    def forward(self, x: torch.Tensor):
        """x: (B, 1, 32, 300) — already reshaped by caller."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(1)
        return self.valence_head(x), self.arousal_head(x)


def build_model() -> EEGNetDE:
    return EEGNetDE()
