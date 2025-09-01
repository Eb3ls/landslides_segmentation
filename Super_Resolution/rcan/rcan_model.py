import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from Super_Resolution.config import Config, RCANModelConfig
from torch.nn import functional as F


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

    def __init__(self, config: Config[RCANModelConfig]):
        super(RCAN, self).__init__()
        self.in_channels = 4
        self.n_channels = config.model.feature_extraction_channels
        self.reduction = config.model.reduction_channels

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(
            self.in_channels, self.n_channels, kernel_size=3, padding=1
        )
        self.groups = nn.ModuleList(
            [
                ResidualGroup(self.n_channels, self.reduction)
                for _ in range(config.model.residual_groups)
            ]
        )
        self.conv_after_body = nn.Conv2d(
            self.n_channels,
            self.n_channels,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(
                self.n_channels, self.n_channels, 3, 1, 1, padding_mode="reflect"
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv_up1 = nn.Conv2d(
            self.n_channels, self.n_channels, 3, 1, 1, padding_mode="reflect"
        )
        self.conv_up2 = nn.Conv2d(
            self.n_channels, self.n_channels, 3, 1, 1, padding_mode="reflect"
        )
        self.conv_hr = nn.Conv2d(
            self.n_channels, self.n_channels, 3, 1, 1, padding_mode="reflect"
        )
        self.conv_last = nn.Conv2d(self.n_channels, 4, 3, 1, 1, padding_mode="reflect")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Feature extraction
        x = self.conv1(x)
        res = x

        # RIR (Residual In Residual)
        for group in self.groups:
            x = group(x)

        x = self.conv_after_body(x) + res
        x = self.conv_before_upsample(x)
        x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode="nearest")))
        x = self.lrelu(self.conv_up2(F.interpolate(x, scale_factor=2, mode="nearest")))
        # Final 1.25x to reach 5x
        x = F.interpolate(x, scale_factor=5 / 4, mode="bicubic", align_corners=False)
        x = self.lrelu(self.conv_hr(x))
        x = self.conv_last(x)

        return x
