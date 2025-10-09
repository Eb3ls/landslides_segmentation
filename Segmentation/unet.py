import torch
import torch.nn as nn
from typing import cast


class DoubleConv(nn.Module):
    """Blocco di doppia convoluzione utilizzato in U-Net."""

    def __init__(self, in_channels: int, out_channels: int, groupnorm: bool = False):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            (
                nn.BatchNorm2d(out_channels)
                if not groupnorm
                else nn.GroupNorm(8, out_channels)
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1
            ),  # La seconda convoluzione mantiene la dimensione dei canali
            (
                nn.BatchNorm2d(out_channels)
                if not groupnorm
                else nn.GroupNorm(8, out_channels)
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class AttentionGate(nn.Module):
    """Gate di attenzione additiva per filtrare le skip connections."""

    def __init__(
        self, skip_channels: int, gate_channels: int, inter_channels: int
    ):  # inter_channels è il numero di canali in cui proietto skip e gate
        super().__init__()
        self.W_g = nn.Conv2d(
            gate_channels,
            inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.W_x = nn.Conv2d(  # ottengo i pesi processati della skip connection
            skip_channels,
            inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.psi = nn.Sequential(  # funzione psi dello scoring dell'attention gate
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

        self._init_open()

    def _init_open(self):
        # Piccole proiezioni e bias della psi alto → alpha≈1 all’inizio
        with torch.no_grad():  # per inizializzare i pesi si può disattivare il tracciamento dei gradienti
            nn.init.normal_(self.W_g.weight, std=1e-3)
            if self.W_g.bias is not None:
                nn.init.zeros_(self.W_g.bias)

            nn.init.normal_(self.W_x.weight, std=1e-3)
            if self.W_x.bias is not None:
                nn.init.zeros_(self.W_x.bias)

            # self.psi[1] è la Conv2d(1x1) -> esplicito il tipo per il type checker
            conv = cast(nn.Conv2d, self.psi[1])
            nn.init.zeros_(conv.weight)
            assert conv.bias is not None
            nn.init.constant_(conv.bias, 1.0)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # x: feature da encoder (skip), g: gating dal decoder (upsampled)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        alpha = self.psi(g1 + x1)  # output: coefficiente di attenzione in [0,1]
        with torch.no_grad():
            a = alpha.detach()
            self._alpha_stats = (
                float(a.mean().item()),
                float(a.min().item()),
                float(a.max().item()),
            )
        return x * alpha  # applica il gate


class Down(nn.Module):
    """Downscaling con maxpool seguito da doppia convoluzione."""

    def __init__(self, in_channels: int, out_channels: int, groupnorm: bool = False):
        super(Down, self).__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, groupnorm=groupnorm)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """Upscaling seguito da doppia convoluzione."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention: bool = False,
        groupnorm: bool = False,
    ):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,  # I canali nel decoder dimezzano a ogni up
        )

        self.attention = attention

        if self.attention:
            # Skip e gating hanno canali = in_channels//2 a questo livello di decoder
            inter = max(in_channels // 4, 1)
            self.att_gate = AttentionGate(
                skip_channels=in_channels // 2,
                gate_channels=in_channels // 2,
                inter_channels=inter,
            )

        self.conv = DoubleConv(in_channels, out_channels, groupnorm=groupnorm)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Calcola eventuale differenza tra upsampling e skip (tipicamente tra 0 e 1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Aggiunge padding per gestire eventuali differenze di dimensione
        x1 = nn.functional.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )

        if self.attention:
            x2 = self.att_gate(x2, x1)

        # Concatenazione della skip connection
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    """Layer di convoluzione di output."""

    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Modello U-Net per segmentazione."""

    def __init__(self, n_channels: int, n_classes: int):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.in_conv = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)  # Il logit è il valore grezzo di output al modello
        return logits


class AttentionUNet(nn.Module):
    """Modello Attention U-Net per segmentazione."""

    def __init__(self, n_channels: int, n_classes: int):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.in_conv = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, attention=True)
        self.up2 = Up(512, 256, attention=True)
        self.up3 = Up(256, 128, attention=True)
        self.up4 = Up(128, 64, attention=True)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)  # Il logit è il valore grezzo di output al modello
        return logits


def log_attention_stats(model: nn.Module, prefix: str = "") -> None:
    """Stampa statistiche delle mappe di attenzione (mean/min/max) per ogni gate."""
    idx = 0
    for m in model.modules():
        if isinstance(m, AttentionGate) and hasattr(m, "_alpha_stats"):
            mean_, min_, max_ = m._alpha_stats
            print(
                f"{prefix}AttnGate[{idx}]: mean={mean_:.3f} min={min_:.3f} max={max_:.3f}"
            )
            idx += 1
