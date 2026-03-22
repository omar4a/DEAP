"""EEGNet (Lawhern et al. 2018) — standard implementation.

Input:  (B, 32, 512)
Output: (valence_logits, arousal_logits) each (B, 2)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import config


class EEGNet(nn.Module):
    """Standard EEGNet with dual classification heads."""

    def __init__(self):
        super().__init__()
        F1 = config.EEGNET_F1          # 8
        D  = config.EEGNET_D           # 2
        F2 = config.EEGNET_F2          # 16
        C  = config.N_CHANNELS         # 32
        p  = config.DROPOUT            # 0.25

        # Block 1 — Temporal convolution
        # padding='same' gives output T == input T (512), avoiding the off-by-one
        # that (0,32) introduces: floor((512+64-64)/1)+1 = 513.
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding='same', bias=False),
            nn.BatchNorm2d(F1),
        )

        # Block 2 — Depthwise spatial convolution
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, D * F1, kernel_size=(C, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p),
        )

        # Block 3 — Separable convolution
        # padding='same' keeps T=128 so AvgPool(1,8) divides evenly → T'=16.
        self.block3 = nn.Sequential(
            nn.Conv2d(D * F1, D * F1, kernel_size=(1, 16), padding='same', groups=D * F1, bias=False),
            nn.Conv2d(D * F1, F2,      kernel_size=(1, 1),                               bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p),
        )

        # Compute flattened feature dimension via a dummy forward pass
        self._feature_dim = self._compute_feature_dim()

        # Dual classification heads
        self.valence_head = nn.Linear(self._feature_dim, 2)
        self.arousal_head = nn.Linear(self._feature_dim, 2)

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[EEGNet] Total params: {total:,}  Trainable: {trainable:,}")

    def _compute_feature_dim(self) -> int:
        """One dummy forward pass to get the flattened spatial size."""
        with torch.no_grad():
            x = torch.zeros(1, 1, config.N_CHANNELS, config.N_SAMPLES)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
        return int(x.flatten(1).shape[1])

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 32, 512) → (B, feature_dim)."""
        x = x.unsqueeze(1)   # (B, 1, 32, 512)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor):
        feat = self._extract_features(x)
        return self.valence_head(feat), self.arousal_head(feat)


def build_model() -> EEGNet:
    return EEGNet()
