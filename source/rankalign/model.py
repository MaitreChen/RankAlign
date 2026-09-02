from __future__ import annotations

from torch import nn


class EEGNetLite(nn.Module):
    def __init__(self, n_channels: int = 30, dropout: float = 0.35):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, (1, 63), padding=(0, 31), bias=False),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, (1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 2))

    def forward(self, x):
        return self.classifier(self.features(x))
