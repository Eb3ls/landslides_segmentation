import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader

from typing import Literal, cast

from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_

from Super_Resolution.config import ConfigSwin2Mose
from Super_Resolution.model_utils import (
    evaluate_model,
    load_model,
    save_model,
    train_model,
    visualize_predictions,
    save_metrics,
)
from Super_Resolution.swin2mose.moe import MoE
from Super_Resolution.swin2mose.utils import (
    PatchEmbed,
    PatchUnEmbed,
    Upsample,
    Upsample_hf,
    UpsampleOneStep,
    get_resi_connection,
    window_partition,
    window_reverse,
)
from data_utils import SuperResolutionDataset


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
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
        pretrained_window_size: tuple[int, int] = (0, 0),
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        # Da capire
        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )

        ######## Relative Position Bias impostato a True seguendo l'articolo ########

        # Si crea una tabella 2*Wh-1 * 2*Ww-1, num_heads,
        # copre tutte le possibili differenze di posizioni relative tra due patch
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

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
        rpe_relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("rpe_relative_position_index", rpe_relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=0.02)

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

        ######## Relative Position Bias impostato a True seguendo l'articolo ########
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def lepe_pos(self, v: torch.Tensor) -> torch.Tensor:
        """Compute Local Enhancement Positional Encoding (LEPE).
        v shape: (B_, num_heads, N, head_dim), where N=window_size*window_size.
        Returns a tensor with same shape as v to be added after attention.
        """
        B_, nH, N, Hd = v.shape
        w = self.window_size[0]
        assert (
            N == w * w
        ), "WindowAttention expects N == window_size*window_size when using LEPE"
        C = nH * Hd
        # Merge heads -> (B_, N, C)
        x = v.permute(0, 2, 1, 3).contiguous().view(B_, N, C)
        # To (B_, C, w, w)
        x = x.transpose(1, 2).contiguous().view(B_, C, w, w)
        # Depth-wise conv
        x = self.get_v(x)
        # Back to (B_, nH, N, Hd)
        x = x.view(B_, C, N).transpose(1, 2).contiguous().view(B_, N, nH, Hd)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, x: Tensor, mask=None) -> Tensor:
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # num_windows sono state estratte dalle immagini e quindi batch diventa num_windows*B
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None and self.v_bias is not None:
            v_bias: torch.Tensor = self.v_bias
            qkv_bias = torch.cat(
                (self.q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias)
            )

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        # LePE
        lepe = self.lepe_pos(v)

        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(
            self.logit_scale,
            max=torch.log(torch.tensor(1.0 / 0.01)).to(self.logit_scale.device),
        ).exp()
        attn = attn * logit_scale

        # relative position bias
        rpe_idx: torch.Tensor = cast(torch.Tensor, self.rpe_relative_position_index)
        idx_flat_rpe = rpe_idx.reshape(-1)
        relative_position_bias = self.relative_position_bias_table[idx_flat_rpe].view(
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

        x = attn @ v
        x = x + lepe

        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""

    def __init__(
        self,
        dim: int,
        input_resolution: list[int],
        num_heads: int,
        MoE_config: dict,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer=nn.LayerNorm,
        pretrained_window_size: int = 0,
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
        ), "shift_size must in 0-window-size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=(pretrained_window_size, pretrained_window_size),
        )

        if isinstance(drop_path, list) or drop_path is None:
            print("Using list of drop_path values")
            raise TypeError("drop_path must be a single value, not a list")

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        # Se MoE é None allora usiamo Multi-Layer Perceptron
        self.mlp = MoE(
            input_size=dim,
            output_size=dim,
            hidden_size=mlp_hidden_dim,
            **MoE_config,
        )

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size: list[int]) -> Tensor:
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

    def forward(self, x: Tensor, x_size: list[int]) -> tuple[Tensor, Tensor | None]:
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
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
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN

        loss_moe = None
        res = self.mlp(x)
        if not torch.is_tensor(res):
            res, loss_moe = res

        x = x + self.drop_path(self.norm2(res))

        return x, loss_moe

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )


class BasicLayer(nn.Module):
    """
    Basic Layer for Swin Transformer.
    Contiene depth blocchi di SwinTransformerBlock
    """

    def __init__(
        self,
        dim: int,
        input_resolution: list[int],
        depth: int,
        num_heads: int,
        window_size: int,
        MoE_config: dict,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer=nn.LayerNorm,
        pretrained_window_size: int = 0,
    ):
        super(BasicLayer, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # Costruiamo i blocchi
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    norm_layer=norm_layer,
                    pretrained_window_size=pretrained_window_size,
                    MoE_config=MoE_config,
                )
                for i in range(depth)
            ]
        )

        # No downsampling in BasicLayer

    def forward(self, x: Tensor, x_size: list[int]) -> tuple[Tensor, float]:
        loss_moe_all: float = 0.0
        for block in self.blocks:
            x, loss_moe = block(x, x_size)
            if loss_moe is not None:
                loss_moe_all += (
                    loss_moe.item() if torch.is_tensor(loss_moe) else loss_moe
                )

        return x, loss_moe_all


class RSTB(nn.Module):
    """
    Residual Swin Transformer Block (RSTB)
    Applica un BasicLayer e una connessione residua
    """

    def __init__(
        self,
        dim: int,
        input_resolution: list[int],
        depth: int,
        num_heads: int,
        window_size: int,
        MoE_config: dict,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer=nn.LayerNorm,
        img_size: int = 128,
        patch_size: int = 4,
        resi_connection: Literal["1conv", "3conv"] = "1conv",
    ):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            MoE_config=MoE_config,
        )

        # Tipo di connessione residua da adottare
        self.conv = get_resi_connection(resi_connection, dim)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=dim,
            embed_dim=dim,
        )

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=dim,
            embed_dim=dim,
        )

    def forward(self, x: Tensor, x_size: list[int]) -> tuple[Tensor, Tensor | None]:
        res, loss_moe = self.residual_group(x, x_size)
        res = self.patch_embed(self.conv(self.patch_unembed(res, x_size)))

        return x + res, loss_moe


class Swin2MoSE(nn.Module):
    def __init__(
        self,
        cfg: ConfigSwin2Mose,
        patch_size: int = 1,
        norm_layer=nn.LayerNorm,
        dropout_rate: float = 0.0,
        # Parametri erediati da SwinTransformer, lascio default
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_dropout_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super(Swin2MoSE, self).__init__()
        num_in_ch = 4
        num_out_ch = 4
        num_feat = 64
        self.img_range = 1
        # Registriamo la mean con shape [1, C, 1, 1]
        # Il buffer serve per memorizzare un tensore che non è un parametro del modello ma una costante
        self.register_buffer("mean", torch.zeros(1, num_in_ch, 1, 1))
        self.upscale = cfg.model.scale
        self.upsampler = cfg.model.upsampler
        self.window_size = cfg.model.window_size

        ############ 1. Shallow feature extraction ############
        self.conv_first = nn.Conv2d(
            num_in_ch, cfg.model.embed_dim, kernel_size=3, stride=1, padding=1
        )

        ############ 2. Deep feature extraction ############
        self.num_layers = len(cfg.model.depths)
        self.embed_dim = cfg.model.embed_dim
        self.num_features = cfg.model.embed_dim
        self.mlp_ratio = mlp_ratio

        # Layer di embedding in non overlapping patch
        self.patch_embed = PatchEmbed(
            img_size=cfg.model.patch_size,
            patch_size=patch_size,
            in_chans=self.embed_dim,
            embed_dim=self.embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patch_resolution
        self.patches_resolution = patches_resolution

        # Layer di unembedding
        self.patch_unembed = PatchUnEmbed(
            img_size=cfg.model.patch_size,
            patch_size=patch_size,
            in_chans=self.embed_dim,
            embed_dim=self.embed_dim,
        )

        # Dropout genera una maschera con prob di dropout_rate di azzerare gli elementi, gli altri vengono
        # moltiplicati per 1/(1 - dropout_rate) per mantenere la media del tensore
        self.pos_drop = nn.Dropout(p=dropout_rate)

        # Stochastic Depth (Drop Path)
        dpr: list[float] = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, steps=sum(cfg.model.depths) + 1)
        ]

        # Costruiamo i Residual Swin Transformer Blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=self.embed_dim,
                input_resolution=self.patches_resolution,
                depth=cfg.model.depths[i_layer],
                num_heads=cfg.model.num_heads[i_layer],
                window_size=cfg.model.window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=dropout_rate,
                attn_drop=attn_dropout_rate,
                drop_path=dpr[
                    sum(cfg.model.depths[:i_layer]) : sum(
                        cfg.model.depths[: i_layer + 1]
                    )
                ],
                norm_layer=norm_layer,
                img_size=cfg.model.patch_size,
                patch_size=patch_size,
                resi_connection=cfg.model.resi_connection,
                MoE_config=cfg.model.MoE_config,
            )
            self.layers.append(layer)

        if self.upsampler == "pixelshuffle":
            self.layers_hf = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = RSTB(
                    dim=self.embed_dim,
                    input_resolution=patches_resolution,
                    depth=cfg.model.depths[i_layer],
                    num_heads=cfg.model.num_heads[i_layer],
                    window_size=cfg.model.window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=dropout_rate,
                    attn_drop=attn_dropout_rate,
                    drop_path=dpr[
                        sum(cfg.model.depths[:i_layer]) : sum(
                            cfg.model.depths[: i_layer + 1]
                        )
                    ],  # no impact on SR results
                    norm_layer=norm_layer,
                    img_size=cfg.model.patch_size,
                    patch_size=patch_size,
                    resi_connection=cfg.model.resi_connection,
                    MoE_config=cfg.model.MoE_config,
                )
                self.layers_hf.append(layer)

        self.norm = norm_layer(self.num_features)

        # Costruiamo l'ultimo layer di convoluzione
        self.conv_after_body = get_resi_connection(
            cfg.model.resi_connection, self.embed_dim
        )

        ############ 3. Ricostruzione dell'immagine ############
        if self.upsampler == "pixelshuffle":
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(self.embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(cfg.model.scale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == "pixelshuffle_aux":
            self.conv_bicubic = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(self.embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.conv_aux = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.conv_after_aux = nn.Sequential(
                nn.Conv2d(num_out_ch, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(cfg.model.scale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == "pixelshuffle_hf":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(self.embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(cfg.model.scale, num_feat)
            self.upsample_hf = Upsample_hf(cfg.model.scale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.conv_first_hf = nn.Sequential(
                nn.Conv2d(num_feat, self.embed_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.conv_after_body_hf = nn.Conv2d(self.embed_dim, self.embed_dim, 3, 1, 1)
            self.conv_before_upsample_hf = nn.Sequential(
                nn.Conv2d(self.embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.conv_last_hf = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == "pixelshuffledirect":
            # Semplice upsample con convoluzione 2D e pixel shuffle per upsampling
            self.upsample = UpsampleOneStep(
                cfg.model.scale,
                self.embed_dim,
                num_out_ch,
                (patches_resolution[0], patches_resolution[1]),
            )
        elif self.upsampler == "nearest+conv":
            # for real-world SR (less artifacts)
            assert self.upscale == 4, "only support x4 now."
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(self.embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(self.embed_dim, num_out_ch, 3, 1, 1)

        # Per applicare l'inizializzazione dei pesi su LayerNorm e Linear
        # self.apply(self._init_weights)

    # Da chiamare nel costruttore per inizializzare i pesi
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # Esclusi dal training optimizer dall'applicazione del weight decay
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    # Esclusi se contengono le keyword specificate
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return x

    def forward_features_hf(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        loss_moe_all = 0.0
        for layer in self.layers_hf:
            x = layer(x, x_size)

            if not torch.is_tensor(x):
                x, loss_moe = x
                if loss_moe is not None:
                    loss_moe_all += (
                        loss_moe.item() if torch.is_tensor(loss_moe) else loss_moe
                    )

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x, loss_moe_all

    # Forward principale che esegue tutti i layers
    def forward_features(self, x: Tensor) -> tuple[Tensor, float]:
        # Shape HxW
        x_size = (x.shape[2], x.shape[3])
        # Tokenizzazione in patch non sovrapposte
        x = self.patch_embed(x)

        # Dropout posizionale
        x = self.pos_drop(x)

        loss_moe_all = 0.0
        for layer in self.layers:
            x = layer(x, x_size)

            if not torch.is_tensor(x):
                x, loss_moe = x
                if loss_moe is not None:
                    loss_moe_all += (
                        loss_moe.item() if torch.is_tensor(loss_moe) else loss_moe
                    )

        # Normalizzazione finale
        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x, loss_moe_all

    # Metodo principale per il forward del modello
    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        # Normalizzazione
        mean_buf: torch.Tensor = cast(torch.Tensor, self.mean)
        mean_buf = mean_buf.type_as(x)
        x = (x - mean_buf) * self.img_range

        if self.upsampler == "pixelshuffle":
            # for classical SR
            x = self.conv_first(x)

            res = self.forward_features(x)
            if not torch.is_tensor(res):
                res, _ = res

            x = self.conv_after_body(res) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == "pixelshuffle_aux":
            bicubic = F.interpolate(
                x,
                size=(H * self.upscale, W * self.upscale),
                mode="bicubic",
                align_corners=False,
            )
            bicubic = self.conv_bicubic(bicubic)
            x = self.conv_first(x)

            res = self.forward_features(x)
            if not torch.is_tensor(res):
                res, _ = res

            x = self.conv_after_body(res) + x
            x = self.conv_before_upsample(x)
            aux = self.conv_aux(x)  # b, num_out_ch, LR_H, LR_W
            x = self.conv_after_aux(aux)
            x = (
                self.upsample(x)[:, :, : H * self.upscale, : W * self.upscale]
                + bicubic[:, :, : H * self.upscale, : W * self.upscale]
            )
            x = self.conv_last(x)
            # aux is not returned; keep only main SR output
        elif self.upsampler == "pixelshuffle_hf":
            # for classical SR with HF
            x = self.conv_first(x)

            res = self.forward_features(x)
            if not torch.is_tensor(res):
                res, _ = res

            x = self.conv_after_body(res) + x
            x_before = self.conv_before_upsample(x)
            x_out = self.conv_last(self.upsample(x_before))

            x_hf = self.conv_first_hf(x_before)

            res_hf = self.forward_features_hf(x_hf)
            if not torch.is_tensor(res_hf):
                res_hf, _ = res_hf

            x_hf = self.conv_after_body_hf(res_hf) + x_hf
            x_hf = self.conv_before_upsample_hf(x_hf)
            x_hf = self.conv_last_hf(self.upsample_hf(x_hf))
            x = x_out + x_hf
        elif self.upsampler == "pixelshuffledirect":
            # Shallow features
            x = self.conv_first(x)

            # Deep features con layer RSTB
            res, _ = self.forward_features(x)

            # Residual connection
            x = self.conv_after_body(res) + x
            x = self.upsample(x)
        elif self.upsampler == "nearest+conv":
            # for real-world SR
            x = self.conv_first(x)

            res = self.forward_features(x)
            if not torch.is_tensor(res):
                res, _ = res

            x = self.conv_after_body(res) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(
                self.conv_up1(
                    torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
                )
            )
            x = self.lrelu(
                self.conv_up2(
                    torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
                )
            )
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)

            res = self.forward_features(x_first)
            if not torch.is_tensor(res):
                res, _ = res

            res = self.conv_after_body(res) + x_first
            x = x + self.conv_last(res)

        # Denormalizzazione finale
        x = x / self.img_range + mean_buf
        # Crop del tensore alle dimensioni obiettivo nel caso in cui sia stato fatto padding
        return x[:, :, : H * self.upscale, : W * self.upscale]


def main():
    """Funzione principale per addestrare e valutare il modello di super risoluzione."""

    # Puliamo la memoria CUDA
    torch.cuda.empty_cache()

    config = ConfigSwin2Mose()

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Seed per riproducibilità
    torch.manual_seed(config.train.seed)
    np.random.seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.train.seed)

    try:

        # Dataset di valutazione
        test_dataset = SuperResolutionDataset(
            config.test.comune,
            config.model.scale,
            config.model.patch_size,
            config.test.dataset_size,
        )

        # Creazione del modello
        model = Swin2MoSE(config).to(device)

        print(f"Parametri: {sum(p.numel() for p in model.parameters())}")

        if config.test.load_model:
            # Caricamento del modello esistente
            model = load_model(config, model, device)
            visualize_predictions(model, test_dataset, device, config)
            return

        # Altrimenti alleniamo il modello

        # Dataset di addestramento
        train_dataset = SuperResolutionDataset(
            config.train.comune,
            config.model.scale,
            config.model.patch_size,
            config.train.dataset_size,
            config.train.augment_data,
        )

        train_loader = DataLoader(
            train_dataset,
            config.train.batch_size,
            num_workers=config.train.workers,
            persistent_workers=True,
        )

        # Allenamento
        print("Starting training...")
        losses = train_model(model, train_loader, device, config)

        # Plottiamo la loss del training logaritmica
        plt.figure(figsize=(12, 8))
        plt.plot(losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.grid(True)
        plt.savefig(f"{config.model.dir_path}{config.model.name}/loss.png")
        plt.show()

        print("Evaluating model...")
        metrics = evaluate_model(model, train_loader, device)
        save_metrics(metrics, config)

        visualize_predictions(model, test_dataset, device, config)

        # Salvataggio del modello
        save_model(model, config)

    except Exception as e:
        print(f"Error occurred during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
