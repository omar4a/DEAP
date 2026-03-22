"""ConvNeXt-EEG: 1D ConvNeXt adapted for EEG signals.

Treats time as the sequence dimension and EEG channels (32) as the feature
dimension. Deliberately small — do not increase CONVNEXT_DIMS.

Input:  (B, 32, 512)
Output: (valence_logits, arousal_logits) each (B, 2)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import config


class LayerNorm1d(nn.Module):
    """LayerNorm applied over the channel dimension of (B, C, T) tensors."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)  →  permute → LayerNorm over C → permute back
        x = x.permute(0, 2, 1)   # (B, T, C)
        x = self.norm(x)
        return x.permute(0, 2, 1)  # (B, C, T)


class ConvNeXt1DBlock(nn.Module):
    """Single ConvNeXt 1D residual block.

    DWConv → LayerNorm → Linear(dim→4dim) → GELU → Linear(4dim→dim) + residual
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dw_conv  = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm     = LayerNorm1d(dim)
        self.pw_up    = nn.Linear(dim, 4 * dim)
        self.act      = nn.GELU()
        self.pw_down  = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        residual = x
        x = self.dw_conv(x)          # (B, C, T)
        x = self.norm(x)             # (B, C, T)
        # Pointwise mixing over channel dim
        x = x.permute(0, 2, 1)      # (B, T, C)
        x = self.pw_up(x)           # (B, T, 4C)
        x = self.act(x)
        x = self.pw_down(x)         # (B, T, C)
        x = x.permute(0, 2, 1)     # (B, C, T)
        return x + residual


class ConvNeXtEEG(nn.Module):
    """1D ConvNeXt for EEG with three progressive stages."""

    def __init__(self):
        super().__init__()
        dims   = config.CONVNEXT_DIMS    # [32, 64, 128]
        blocks = config.CONVNEXT_BLOCKS  # [2, 2, 2]

        # Stem: Conv1d(32→32, k=15, s=2, p=7) — halves time to 256
        self.stem = nn.Sequential(
            nn.Conv1d(config.N_CHANNELS, dims[0], kernel_size=15, stride=2, padding=7),
            LayerNorm1d(dims[0]),
        )

        # Build stages and inter-stage downsamples
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(len(dims)):
            stage = nn.Sequential(*[ConvNeXt1DBlock(dims[i]) for _ in range(blocks[i])])
            self.stages.append(stage)

            if i < len(dims) - 1:
                # Downsample: LayerNorm → Conv1d(dims[i]→dims[i+1], k=2, s=2)
                down = nn.Sequential(
                    LayerNorm1d(dims[i]),
                    nn.Conv1d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
                self.downsamples.append(down)

        # Dual heads
        self.valence_head = nn.Linear(dims[-1], 2)
        self.arousal_head = nn.Linear(dims[-1], 2)

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[ConvNeXtEEG] Total params: {total:,}  Trainable: {trainable:,}")

    def forward(self, x: torch.Tensor):
        # x: (B, 32, 512)
        x = self.stem(x)                          # (B, 32, 256)

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)        # halve time, double channels

        # Global average pooling
        x = x.mean(dim=-1)                        # (B, 128)

        return self.valence_head(x), self.arousal_head(x)


def build_model() -> ConvNeXtEEG:
    return ConvNeXtEEG()
