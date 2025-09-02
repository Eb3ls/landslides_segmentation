import math
import numpy as np
import torch
import torch.nn as nn
from timm.layers.weight_init import trunc_normal_
from timm.layers.helpers import to_2tuple
import torch.nn.functional as F
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class ECA(nn.Module):
    """Efficient Channel Attention - più leggero di SE ma efficace"""

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # Calcola il kernel size in base ai canali
        t = int(abs((math.log(channels, 2) + b) / gamma))
        # Aumenta di 1 se pari
        k = t if t % 2 == 1 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Processa i canali come finestra di k vicini
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # [B, C, H, W] -> [B, C, 1, 1]
        # [B, C, 1, 1] -> [B, C, 1] -> [B, 1, C]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        # Ritorna a [B, C, 1, 1]
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class TextureEnhancementBlock(nn.Module):
    """Multi-scale texture enhancement per dettagli fini"""

    def __init__(self, channels):
        super().__init__()
        # Multi-scale feature extraction
        # Concatenando poi le 4 conv ritorniamo a channels
        self.conv1x1 = nn.Conv2d(channels, channels // 4, 1)
        self.conv3x3 = nn.Conv2d(channels, channels // 4, 3, padding=1)
        self.conv5x5 = nn.Conv2d(channels, channels // 4, 5, padding=2)
        self.conv7x7 = nn.Conv2d(channels, channels // 4, 7, padding=3)

        # Feature fusion with attention
        # Conv sulla concatenazione
        self.fusion = nn.Conv2d(channels, channels, 1)
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        f1 = self.conv1x1(x)
        f3 = self.conv3x3(x)
        f5 = self.conv5x5(x)
        f7 = self.conv7x7(x)

        fused = self.fusion(torch.cat([f1, f3, f5, f7], dim=1))
        attn = self.attention(fused)

        return x + fused * attn


class SpatialAttention(nn.Module):
    """Spatial attention per migliorare la fusione"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid()
        )

    def forward(self, x):
        # Pooling medio sui canali -> attivazione media in ogni pixel
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # Pooling max sui canali -> attivazione massima in ogni pixel
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatena e passa attraverso conv + sigmoid
        attn_input = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(attn_input)
        return x * attn


class Mlp(nn.Module):

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) con rpe e LePE,
    Supporta anche Shifted Window Attention (SW-MSA).

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads

        # Parametro moltiplicato prima della softmax con l'attn QxK
        # Requires grad fa si che qualsiasi operazione da questa venga tracciata per il backpropagation
        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        # get relative_coords_table
        relative_coords_h = torch.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32
        )
        relative_coords_w = torch.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32
        )
        relative_coords_table = (
            torch.stack(
                torch.meshgrid([relative_coords_h, relative_coords_w], indexing="ij")
            )
            .permute(1, 2, 0)
            .contiguous()
            .unsqueeze(0)
        )  # 1, 2*Wh-1, 2*Ww-1, 2
        relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
        relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
            torch.sign(relative_coords_table)
            * torch.log2(torch.abs(relative_coords_table) + 1.0)
            / np.log2(8)
        )

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # Query, Key, Value
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        # Se vero aggiunge un bias che il modello può imparare
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # Applicata ai vettori dell'ultima dimensione
        self.softmax = nn.Softmax(dim=-1)

        self.register_buffer("relative_coords_table", relative_coords_table)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # num_windows sono state estratte dalle immagini e quindi batch diventa num_windows*B
        # N é Wh * Ww, dimensione spaziale collassata in un vettore
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None and self.v_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )

        # Calcola qvk con shape (B_, N, 3 * C), espande quindi le dimensioni di 3 per ognuna
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        # -1 dice di calcolare la dimensione automaticamente per lasciare il numero di elementi invariato
        # Reshape per ottenere (B_, N, 3, num_heads, head_dim)
        # Permuta per ottenere (3, B_, num_heads, N, head_dim), il numero indica la dimensione da mettere in quel posto
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        # Prima normalizza Q e K, poi calcola l'attenzione (@ é il prodotto scalare)
        # É la similaritá coseno tra ogni coppia query-key, valori [-1, 1]
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(
            self.logit_scale,
            max=torch.log(torch.tensor(1.0 / 0.01)).to(self.logit_scale.device),
        ).exp()
        # Moltiplica per il logit scale prima della softmax, amplifica/riduce la temperatura dell'attenzione
        # Aiuta a stabilizzare l'addestramento
        attn = attn * logit_scale

        # relative position bias
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(
            -1, self.num_heads
        )
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)  # type: ignore
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # Numero di finestre per immagine
            nW = mask.shape[0]
            # Raggruppato in (batch_size, nW, nH, N, N)
            # Espansa mask per avere la forma (1, nW, 1, N, N), con il broadcasting é applicato a tutti i batch e alle finestre
            # Se due token sono in finestre diverse nel layer prima sono da annullare
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            # Tornato alla forma originale per la softmax
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # Prodotto scalare tra l'attenzione e V
        x = attn @ v

        # Ritorna alla forma originale (B_, N, C)
        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RDG(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size,
        mlp_ratio,
        qkv_bias,
        drop,
        attn_drop,
        gc,
        patch_size,
        img_size,
    ):
        super(RDG, self).__init__()

        self.swin1 = SwinTransformerBlock(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,  # For first block
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
        )
        self.adjust1 = nn.Conv2d(dim, gc, 1)

        self.swin2 = SwinTransformerBlock(
            dim + gc,
            input_resolution=input_resolution,
            num_heads=num_heads - ((dim + gc) % num_heads),
            window_size=window_size,
            shift_size=window_size // 2,  # For first block
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
        )
        self.adjust2 = nn.Conv2d(dim + gc, gc, 1)

        self.swin3 = SwinTransformerBlock(
            dim + 2 * gc,
            input_resolution=input_resolution,
            num_heads=num_heads - ((dim + 2 * gc) % num_heads),
            window_size=window_size,
            shift_size=0,  # For first block
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
        )
        self.adjust3 = nn.Conv2d(dim + gc * 2, gc, 1)

        self.swin4 = SwinTransformerBlock(
            dim + 3 * gc,
            input_resolution=input_resolution,
            num_heads=num_heads - ((dim + 3 * gc) % num_heads),
            window_size=window_size,
            shift_size=window_size // 2,  # For first block
            mlp_ratio=1,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
        )
        self.adjust4 = nn.Conv2d(dim + gc * 3, gc, 1)

        self.swin5 = SwinTransformerBlock(
            dim + 4 * gc,
            input_resolution=input_resolution,
            num_heads=num_heads - ((dim + 4 * gc) % num_heads),
            window_size=window_size,
            shift_size=0,  # For first block
            mlp_ratio=1,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
        )
        self.adjust5 = nn.Conv2d(dim + gc * 4, dim, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.pe = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

        self.pue = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

    def forward(self, x, xsize):
        x1 = self.pe(self.lrelu(self.adjust1(self.pue(self.swin1(x, xsize), xsize))))
        x2 = self.pe(
            self.lrelu(
                self.adjust2(self.pue(self.swin2(torch.cat((x, x1), -1), xsize), xsize))
            )
        )
        x3 = self.pe(
            self.lrelu(
                self.adjust3(
                    self.pue(self.swin3(torch.cat((x, x1, x2), -1), xsize), xsize)
                )
            )
        )
        x4 = self.pe(
            self.lrelu(
                self.adjust4(
                    self.pue(self.swin4(torch.cat((x, x1, x2, x3), -1), xsize), xsize)
                )
            )
        )
        x5 = self.pe(
            self.adjust5(
                self.pue(self.swin5(torch.cat((x, x1, x2, x3, x4), -1), xsize), xsize)
            )
        )

        return x5 * 0.2 + x


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, _, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(
                x_windows, mask=self.attn_mask
            )  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(
                x_windows, mask=self.calculate_mask(x_size).to(x.device)
            )

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # 结构为 [B, num_patches, C]
        if self.norm is not None:
            x = self.norm(x)  # 归一化
        return x


class PatchUnEmbed(nn.Module):
    r"""Image to Patch Unembedding

    输入:
        img_size (int): 图像的大小，默认为 224*224.
        patch_size (int): Patch token 的大小，默认为 4*4.
        in_chans (int): 输入图像的通道数，默认为 3.
        embed_dim (int): 线性 projection 输出的通道数，默认为 96.
        norm_layer (nn.Module, optional): 归一化层， 默认为N None.
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)  # 图像的大小，默认为 224*224
        patch_size = to_2tuple(patch_size)  # Patch token 的大小，默认为 4*4
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]  # patch 的分辨率
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = (
            patches_resolution[0] * patches_resolution[1]
        )  # patch 的个数，num_patches

        self.in_chans = in_chans  # 输入图像的通道数
        self.embed_dim = embed_dim  # 线性 projection 输出的通道数

    def forward(self, x, x_size):
        B, _, _ = x.shape  # 输入 x 的结构
        x = x.transpose(1, 2).view(B, -1, x_size[0], x_size[1])
        return x


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(
                f"scale {scale} is not supported. " "Supported scales: 2^n and 3."
            )
        super(Upsample, self).__init__(*m)


class DRCT(nn.Module):

    def __init__(self, config):
        # Estrai parametri dalla configurazione, con fallback per compatibilità
        img_size = config.model.img_size
        patch_size = config.model.patch_size
        in_chans = config.model.in_chans
        embed_dim = config.model.embed_dim
        depths = config.model.depths
        num_heads = config.model.num_heads
        window_size = config.model.window_size
        overlap_ratio = config.model.overlap_ratio
        mlp_ratio = config.model.mlp_ratio
        qkv_bias = config.model.qkv_bias
        drop_rate = config.model.drop_rate
        attn_drop_rate = config.model.attn_drop_rate
        upscale = config.model.scale  # Usa 'scale' dal config base
        upsampler = config.model.upsampler
        resi_connection = config.model.resi_connection
        gc = config.model.gc
        num_feat = config.model.num_feat

        # Parametri fissi non configurabili
        norm_layer = nn.LayerNorm
        patch_norm = True
        super(DRCT, self).__init__()

        print("Preparing model...")
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.overlap_ratio = overlap_ratio

        num_in_ch = in_chans
        num_out_ch = in_chans
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # build
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):

            layer = RDG(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                gc=gc,
                img_size=img_size,
                patch_size=patch_size,
            )

            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "identity":
            self.conv_after_body = nn.Identity()

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == "pixelshuffle":
            print("Using enhanced pixelshuffle upsampler with attention")
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )

            self.antialisiasing = nn.Conv2d(
                num_feat, num_feat, 3, 1, 1, padding_mode="reflect"
            )
            # Enhanced PixelShuffle blocks with ECA attention
            self.upsample1 = nn.Sequential(
                nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ECA(num_feat),  # ECA attention dopo 2x upsampling
            )
            self.upsample2 = nn.Sequential(
                nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ECA(num_feat),  # ECA attention dopo 4x upsampling
            )

            # Texture enhancement e spatial attention
            self.conv_hr = nn.Sequential(
                nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                TextureEnhancementBlock(num_feat),  # Multi-scale texture enhancement
            )
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

            # Residual path migliorato
            self.conv_before_upsample_res = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ECA(num_feat),
            )

            # Fusione migliorata con spatial attention
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(2 * num_feat, num_feat, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
            self.spatial_attention = SpatialAttention()
        elif self.upsampler == "nearest+conv":
            print("Using nearest+conv upsampler")
            # Two 2x nearest stages + final 1.25x bicubic with refinement
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1, padding_mode="reflect"),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
            self.conv_up1 = nn.Conv2d(
                num_feat, num_feat, 3, 1, 1, padding_mode="reflect"
            )
            self.conv_up2 = nn.Conv2d(
                num_feat, num_feat, 3, 1, 1, padding_mode="reflect"
            )
            self.conv_hr = nn.Conv2d(
                num_feat, num_feat, 3, 1, 1, padding_mode="reflect"
            )

            self.conv3_hr = nn.Conv2d(
                num_feat, num_feat, 3, 1, 1, padding_mode="reflect"
            )
            self.conv5_hr = nn.Conv2d(
                num_feat, num_feat, 5, 1, 2, padding_mode="reflect"
            )
            self.conv7_hr = nn.Conv2d(
                num_feat, num_feat, 7, 1, 3, padding_mode="reflect"
            )
            self.conv_combine = nn.Conv2d(
                4 * num_feat, num_feat, 1, 1, 0, padding_mode="reflect"
            )

            self.conv_last = nn.Conv2d(num_feat, 4, 3, 1, 1, padding_mode="reflect")
        elif self.upsampler == "test":
            self.tail = UpsampleTail5x(embed_dim, num_feat, num_out_ch)
        elif self.upsampler == "only_shuffle":
            print("Using only upsample")
            self.tail = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat * 5 * 5, 3, 1, 1, padding_mode="reflect"),
                nn.PixelShuffle(5),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(num_feat, num_out_ch, 3, 1, 1, padding_mode="reflect"),
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])

        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        if self.upsampler == "pixelshuffle":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x

            # Path 1: Residual path con ECA attention
            res = self.conv_before_upsample_res(
                x
            )  # (B, 128, 64, 64) → (B, 64, 64, 64) + ECA
            res = F.interpolate(
                res, scale_factor=5, mode="bicubic", align_corners=False
            )  # (B, 64, 64, 64) → (B, 64, 320, 320)
            res = self.antialisiasing(res)  # Antialiasing per ridurre aliasing

            # Path 2: Enhanced PixelShuffle path con ECA ad ogni step
            x_detail = self.conv_before_upsample(
                x
            )  # (B, 128, 64, 64) → (B, 64, 64, 64)
            x_detail = self.upsample1(
                x_detail
            )  # (B, 64, 64, 64) → (B, 64, 128, 128) + ECA
            x_detail = self.upsample2(
                x_detail
            )  # (B, 64, 128, 128) → (B, 64, 256, 256) + ECA
            x_detail = F.interpolate(
                x_detail, scale_factor=5 / 4, mode="bicubic", align_corners=False
            )  # (B, 64, 256, 256) → (B, 64, 320, 320)
            x_detail = self.conv_hr(x_detail)  # TextureEnhancement applicato

            # Fusione intelligente con spatial attention
            fusion_features = torch.cat([res, x_detail], dim=1)  # (B, 128, 320, 320)
            fusion_weight = torch.sigmoid(
                self.fusion_conv(fusion_features)
            )  # (B, 64, 320, 320)

            # Combinazione pesata
            x = (
                fusion_weight * x_detail + (1 - fusion_weight) * res
            )  # (B, 64, 320, 320)

            # Spatial attention finale per raffinare la fusione
            x = self.spatial_attention(x)  # (B, 64, 320, 320)

            # Conversione finale ai canali di output
            x = self.conv_last(x)  # (B, 64, 320, 320) → (B, 4, 320, 320)
        elif self.upsampler == "nearest+conv":
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(
                self.conv_up1(F.interpolate(x, scale_factor=2, mode="nearest"))
            )
            x = self.lrelu(
                self.conv_up2(F.interpolate(x, scale_factor=2, mode="nearest"))
            )
            # Final 1.25x to reach 5x
            x = F.interpolate(
                x, scale_factor=5 / 4, mode="bicubic", align_corners=False
            )
            x = self.lrelu(self.conv_hr(x))
            res = x
            x1 = self.conv3_hr(x)
            x2 = self.conv5_hr(x)
            x3 = self.conv7_hr(x)
            x = torch.cat([x, x1, x2, x3], dim=1)
            x = self.conv_combine(x) + res
            x = self.conv_last(x)
        elif self.upsampler == "test":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.tail(x)
        elif self.upsampler == "only_shuffle":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            print(x.shape)
            x = self.tail(x)

        return x


# --- piccolo blocco residuo (senza BatchNorm) ---
class ResBlock(nn.Module):
    def __init__(self, channels, kernel=3, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel, padding=kernel // 2, bias=bias
        )
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel, padding=kernel // 2, bias=bias
        )
        self.scale = 1.0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv2(self.act(self.conv1(x)))
        return x + out * 0.1  # small residual scale


# --- channel attention leggero (SE style) ---
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)
        self.act = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = self.fc2(self.act(self.fc1(s)))
        return x * self.sig(s)


# --- texture enhancement leggero (multi-scale ma controllato) ---
class TextureEnhancementLite(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # reduce channels for multi-kernel ops for stability
        mid = max(channels // 4, 8)
        self.conv1 = nn.Conv2d(channels, mid, 1, padding=0)
        self.conv3 = nn.Conv2d(channels, mid, 3, padding=1)
        self.conv5 = nn.Conv2d(channels, mid, 5, padding=2)
        self.fuse = nn.Conv2d(mid * 3, channels, 1)
        self.attn = ChannelAttention(channels, reduction=8)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        f1 = self.conv1(x)
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        fused = self.fuse(torch.cat([f1, f3, f5], dim=1))
        fused = self.attn(fused)
        return x + 0.2 * fused


# --- Tail per SR 5x: bicubic upsample + HR refinement ---
class UpsampleTail5x(nn.Module):
    def __init__(self, embed_dim, num_feat=64, num_out_ch=4, num_resblocks=8, scale=5):
        super().__init__()
        self.scale = scale
        self.conv_before_upsample = nn.Conv2d(embed_dim, num_feat, 3, 1, 1)
        self.antialias = nn.Conv2d(num_feat, num_feat, 3, 1, 1, padding_mode="reflect")
        # HR refinement: a stack of ResBlocks at HR
        self.hr_refine = nn.Sequential(
            *[ResBlock(num_feat) for _ in range(num_resblocks)]
        )
        self.texture = TextureEnhancementLite(num_feat)
        self.fusion_conv = nn.Conv2d(2 * num_feat, num_feat, 1)
        # learnable scalar blending (per-channel could be used too)
        self.alpha_param = nn.Parameter(torch.tensor(0.5))  # init 0.5
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, embed_dim, H, W)  LR features
        feat = self.conv_before_upsample(x)  # (B, num_feat, H, W)
        # Upsample both paths with the SAME bicubic grid -> no misalignment
        H_hr = feat.size(2) * self.scale
        W_hr = feat.size(3) * self.scale
        res = F.interpolate(
            feat, size=(H_hr, W_hr), mode="bicubic", align_corners=False
        )
        res = self.antialias(res)
        detail = res
        detail = self.hr_refine(detail)
        detail = self.texture(detail)
        fused = torch.cat([res, detail], dim=1)
        fused = self.fusion_conv(fused)
        alpha = torch.sigmoid(self.alpha_param)
        out_feat = alpha * fused + (1.0 - alpha) * res
        out = self.conv_last(out_feat)
        return out
