"""EEGNet + Transformer encoder for EEG emotion recognition.

Architecture:
  EEGNet blocks (identical to eegnet.py, no heads) → reshape →
  Linear projection → Positional embedding → Transformer encoder →
  Global average pool → dual heads.

Input:  (B, 32, 512)
Output: (valence_logits, arousal_logits) each (B, 2)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import config


class _TransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer (MHSA + FFN with residuals)."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention + residual + LayerNorm
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN + residual + LayerNorm
        x = self.norm2(x + self.ffn(x))
        return x


class EEGNetTransformer(nn.Module):
    """EEGNet feature extractor followed by a Transformer encoder."""

    def __init__(self):
        super().__init__()
        F1 = config.EEGNET_F1          # 8
        D  = config.EEGNET_D           # 2
        F2 = config.EEGNET_F2          # 16
        C  = config.N_CHANNELS         # 32
        p  = config.DROPOUT            # 0.25
        d  = config.D_MODEL            # 64

        # ---- EEGNet blocks (identical architecture, independent weights) ----
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding='same', bias=False),
            nn.BatchNorm2d(F1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, D * F1, kernel_size=(C, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(D * F1, D * F1, kernel_size=(1, 16), padding='same', groups=D * F1, bias=False),
            nn.Conv2d(D * F1, F2,      kernel_size=(1, 1),                               bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p),
        )

        # Sequence length after EEGNet blocks  (T' = 16 for N_SAMPLES=512)
        self._seq_len = self._compute_seq_len()

        # ---- Transformer head ----
        self.projection    = nn.Linear(F2, d)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self._seq_len, d))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.transformer = nn.ModuleList([
            _TransformerEncoderLayer(d, config.N_HEADS,
                                     config.TRANSFORMER_FFN_DIM,
                                     config.TRANSFORMER_DROPOUT)
            for _ in range(config.N_TRANSFORMER_LAYERS)
        ])

        # Dual heads
        self.valence_head = nn.Linear(d, 2)
        self.arousal_head = nn.Linear(d, 2)

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[EEGNetTransformer] Total params: {total:,}  Trainable: {trainable:,}")

    def _compute_seq_len(self) -> int:
        """Determine T' (temporal steps after EEGNet blocks) via dummy pass."""
        with torch.no_grad():
            x = torch.zeros(1, 1, config.N_CHANNELS, config.N_SAMPLES)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)   # (1, F2, 1, T')
        return x.shape[-1]

    def forward(self, x: torch.Tensor):
        # EEGNet feature extraction
        x = x.unsqueeze(1)       # (B, 1, 32, 512)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)       # (B, F2, 1, T')

        # Reshape → (B, T', F2)
        B, F2, _, T = x.shape
        x = x.squeeze(2).permute(0, 2, 1)   # (B, T', F2)

        # Project to d_model + add positional embedding
        x = self.projection(x)              # (B, T', d_model)
        x = x + self.pos_embedding          # (B, T', d_model)

        # Transformer encoder
        for layer in self.transformer:
            x = layer(x)

        # Global average pool → (B, d_model)
        x = x.mean(dim=1)

        return self.valence_head(x), self.arousal_head(x)


def build_model() -> EEGNetTransformer:
    return EEGNetTransformer()
