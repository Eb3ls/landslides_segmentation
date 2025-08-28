import math
import torch
import torch.nn as nn
from timm.layers.weight_init import trunc_normal_
from timm.layers.helpers import to_2tuple
import torch.nn.functional as F
import sys
import os

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
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)  # type: ignore
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class RDG(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        shift_size,
        mlp_ratio,
        qkv_bias,
        qk_scale,
        drop,
        attn_drop,
        drop_path,
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
            qk_scale=qk_scale,
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
            qk_scale=qk_scale,
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
            qk_scale=qk_scale,
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
            qk_scale=qk_scale,
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
            qk_scale=qk_scale,
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
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
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
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
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
        B, L, C = x.shape
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

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


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

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = (
            Ho
            * Wo
            * self.embed_dim
            * self.in_chans
            * (self.patch_size[0] * self.patch_size[1])
        )
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


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
        B, HW, C = x.shape  # 输入 x 的结构
        x = x.transpose(1, 2).view(
            B, -1, x_size[0], x_size[1]
        )  # 输出结构为 [B, Ph*Pw, C]
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
        drop_path_rate = config.model.drop_path_rate
        upscale = config.model.scale  # Usa 'scale' dal config base
        upsampler = config.model.upsampler
        resi_connection = config.model.resi_connection
        gc = config.model.gc

        # Parametri fissi non configurabili
        qk_scale = None
        norm_layer = nn.LayerNorm
        patch_norm = True
        img_range = 1.0
        super(DRCT, self).__init__()

        print("Preparing model...")
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.overlap_ratio = overlap_ratio

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
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

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):

            layer = RDG(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                num_heads=num_heads[i_layer],
                window_size=window_size,
                depth=0,
                shift_size=window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
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

            # Enhanced PixelShuffle blocks with ECA attention
            self.upsample1 = nn.Sequential(
                nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(inplace=True),
                ECA(num_feat),  # ECA attention dopo 2x upsampling
            )
            self.upsample2 = nn.Sequential(
                nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(inplace=True),
                ECA(num_feat),  # ECA attention dopo 4x upsampling
            )

            # Texture enhancement e spatial attention
            self.conv_hr = nn.Sequential(
                nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
                TextureEnhancementBlock(num_feat),  # Multi-scale texture enhancement
            )
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

            # Residual path migliorato
            self.conv_before_upsample_res = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
                ECA(num_feat),
            )

            # Fusione migliorata con spatial attention
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(2 * num_feat, num_feat, 1, 1, 0),
                nn.LeakyReLU(inplace=True),
            )
            self.spatial_attention = SpatialAttention()
        elif self.upsampler == "nearest+conv":
            print("Using nearest+conv upsampler")
            # Two 2x nearest stages + final 1.25x bicubic with refinement
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(128, num_feat, 3, 1, 1, padding_mode="reflect"),
                nn.LeakyReLU(inplace=True),
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
            self.conv_last = nn.Conv2d(num_feat, 4, 3, 1, 1, padding_mode="reflect")

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
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

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

            # Path 2: Enhanced PixelShuffle path con ECA ad ogni step
            x_detail = self.conv_before_upsample(
                x
            )  # (B, 128, 64, 64) → (B, 64, 64, 64)
            x_detail = self.upsample1(
                x_detail
            )  # (B, 64, 64, 64) → (B, 64, 128, 128) + ECA
            x_detail = self.upsample2(
                x_detail
            )  # (B, 64, 128, 128) → (B, 64, 256, 256) + SEBlockECA
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
            x = self.conv_last(x)

        x = x / self.img_range + self.mean

        return x
