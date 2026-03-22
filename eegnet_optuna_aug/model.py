import torch
import torch.nn as nn

class EEGNetBinary(nn.Module):
    """EEGNet classifier for one binary target."""

    def __init__(self, channels: int, samples: int, f1: int, depth: int, f2: int, dropout: float, temporal_kernel: int = 64):
        super().__init__()
        self.channels = channels
        self.samples = samples

        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, temporal_kernel), padding="same", bias=False),
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
            x = torch.zeros(1, 1, self.channels, self.samples)
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

class Tsception(nn.Module):
    """
    Tsception architecture for EEG emotion recognition.
    Uses multi-scale temporal 1D convolutions followed by spatial convolutions.
    """
    def __init__(self, num_classes: int = 2, input_size: tuple = (1, 32, 128*2), 
                 sampling_rate: int = 128, num_T: int = 15, num_S: int = 15, 
                 hidden: int = 32, dropout_rate: float = 0.5):
        super(Tsception, self).__init__()
        
        self.inception_window = [0.5, 0.25, 0.125, 0.0625, 0.03125]
        self.stride = int(sampling_rate * 0.25)
        self.split_sizes = [int(sampling_rate * w) for w in self.inception_window]
        
        # Temporal scales
        self.T1 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, self.split_sizes[0]), stride=(1, self.stride), padding=(0, self.split_sizes[0]//2)),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        )
        self.T2 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, self.split_sizes[1]), stride=(1, self.stride), padding=(0, self.split_sizes[1]//2)),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        )
        self.T3 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, self.split_sizes[2]), stride=(1, self.stride), padding=(0, self.split_sizes[2]//2)),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        )
        
        # Spatial Scales (Hemisphere asymmetry)
        channels = input_size[1]
        self.S1 = nn.Sequential(
            nn.Conv2d(num_T*3, num_S, kernel_size=(channels, 1), stride=(1, 1), padding=0),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        )
        self.S2 = nn.Sequential(
            nn.Conv2d(num_T*3, num_S, kernel_size=(int(channels/2), 1), stride=(int(channels/2), 1), padding=0),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(num_S*2, num_S, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.LeakyReLU()
        )
        
        self._feature_dim = self._compute_feature_dim(input_size)
        
        self.fc = nn.Sequential(
            nn.Linear(self._feature_dim, hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes)
        )
        
    def _compute_feature_dim(self, input_size):
        with torch.no_grad():
            x = torch.zeros(1, *input_size)
            y1 = self.T1(x)
            y2 = self.T2(x)
            y3 = self.T3(x)
            # Find min temporal dimension
            min_dim = min(y1.shape[3], y2.shape[3], y3.shape[3])
            y1, y2, y3 = y1[:, :, :, :min_dim], y2[:, :, :, :min_dim], y3[:, :, :, :min_dim]
            z = torch.cat((y1, y2, y3), dim=1)
            
            z1 = self.S1(z)
            z2 = self.S2(z)
            # Find min temporal dimension again
            min_dim_s = min(z1.shape[3], z2.shape[3])
            z1, z2 = z1[:, :, :, :min_dim_s], z2[:, :, :, :min_dim_s]
            
            # Upsample spatial dim of S1 to match S2 (S2 has 2 spatial blocks, S1 has 1)
            # For simplicity, flatten both and concat
            z_f1 = z1.flatten(1)
            z_f2 = z2.flatten(1)
            return z_f1.shape[1] + z_f2.shape[1]

    def forward(self, x):
        x = x.unsqueeze(1) # Add channel dim (B, 1, C, T)
        
        y1 = self.T1(x)
        y2 = self.T2(x)
        y3 = self.T3(x)
        
        min_t = min(y1.shape[3], y2.shape[3], y3.shape[3])
        y1, y2, y3 = y1[:, :, :, :min_t], y2[:, :, :, :min_t], y3[:, :, :, :min_t]
        
        z = torch.cat((y1, y2, y3), dim=1)
        z = self.dropout(z)
        
        z1 = self.S1(z)
        z2 = self.S2(z)
        
        z_f1 = z1.flatten(1)
        z_f2 = z2.flatten(1)
        
        out = torch.cat((z_f1, z_f2), dim=1)
        out = self.dropout(out)
        
        return self.fc(out)
