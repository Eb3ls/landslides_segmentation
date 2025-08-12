import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from Super_Resolution.config import ConfigRCAN


class RCAB(nn.Module):
    """Residual Channel Attention Block"""

    def __init__(self, n_channels: int, reduction: int = 16):
        super(RCAB, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv3 = nn.Conv2d(n_channels, n_channels // reduction, kernel_size=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(n_channels // reduction, n_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Primo blocco
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)

        # Channel attention
        attention = self.global_avg_pool(out)
        attention = self.conv3(attention)
        attention = self.relu2(attention)
        attention = self.conv4(attention)
        attention = self.sigmoid(attention)

        # Applichiamo il peso
        out = out * attention

        # Residual connection
        out = out + residual

        return out


class ResidualGroup(nn.Module):
    """Gruppo di blocchi Residual Channel Attention Block"""

    def __init__(self, n_channels: int, reduction: int = 16, n_blocks: int = 4):
        super(ResidualGroup, self).__init__()
        self.blocks = nn.Sequential(
            *[RCAB(n_channels, reduction) for _ in range(n_blocks)]
        )
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.blocks(x)
        out = self.conv(out)
        # Residual connection
        out = out + residual
        return out


class RCAN(nn.Module):
    """Residual Channel Attention Network"""

    def __init__(self, config: ConfigRCAN):
        super(RCAN, self).__init__()
        self.in_channels = 4
        self.n_channels = config.model.feature_extraction_channels
        self.reduction = config.model.reduction_channels

        scale = config.model.scale

        self.conv1 = nn.Conv2d(
            self.in_channels, self.n_channels, kernel_size=3, padding=1
        )
        self.groups = nn.ModuleList(
            [
                ResidualGroup(self.n_channels, self.reduction)
                for _ in range(config.model.residual_groups)
            ]
        )
        self.conv2 = nn.Conv2d(
            self.n_channels, self.n_channels, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            self.n_channels, self.n_channels * (scale**2), kernel_size=3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.conv4 = nn.Conv2d(
            self.n_channels, self.in_channels, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Feature extraction
        out = self.conv1(x)
        feat = out

        # RIR (Residual In Residual)
        for group in self.groups:
            out = group(out)
        out = self.conv2(out)

        # Residual connection
        out = out + feat

        out = self.conv3(out)
        out = self.pixel_shuffle(out)
        out = self.conv4(out)

        return out
