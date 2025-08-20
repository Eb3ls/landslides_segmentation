from typing import Literal
import numpy as np
from torch import Tensor, nn
import torch
import torch.nn.functional as F

from timm.layers.drop import DropPath

from Super_Resolution.config import ConfigMyModel, MyModelConfig, Swin2MoseModelConfig
from Super_Resolution.swin2mose.utils import (
    PatchEmbed,
    PatchUnEmbed,
    get_resi_connection,
    window_partition,
    window_reverse,
)


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

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
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
        pretrained_window_size: tuple[int, int] = (0, 0),
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        # Parametro moltiplicato prima della softmax con l'attn QxK
        # Requires grad fa si che qualsiasi operazione da questa venga tracciata per il backpropagation
        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )

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
            torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))
            .permute(1, 2, 0)
            .contiguous()
            .unsqueeze(0)
        )  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
        else:
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

    def forward(self, x: Tensor, mask=None) -> Tensor:
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


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""

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
        drop_path: float = 0.0,
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
            # Se la dimensione della finestra é piú grande della risoluzione di input niente partizione
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

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

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        # Definiti quanti neuroni ci sono nella MLP rispetto a dim di input
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
        )

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size: tuple[int, int]) -> Tensor:
        # Calcoliamo l'attention mask per SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        # Intervallo di righe e colonne
        h_slices = (
            slice(0, -self.window_size),  # Parte alta
            slice(-self.window_size, -self.shift_size),  # Parte centrale
            slice(-self.shift_size, None),  # Parte bassa
        )
        w_slices = (
            slice(0, -self.window_size),  # Parte sinistra
            slice(-self.window_size, -self.shift_size),  # Parte centrale
            slice(-self.shift_size, None),  # Parte destra
        )

        # Per ogni combinazione di h_slices e w_slices, assegna un indice unico
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # Ha un solo canale perché fará broadcasting
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        # (nW, 1, N) - (nW, N, 1) -> (nW, N, N) matrice di differenza, 0 se i due token sono nella stessa regione
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # Trasforma le differenze in -100 che dopo la softmax diventeranno 0
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

        return attn_mask

    def forward(self, x: Tensor, x_size: tuple[int, int]) -> Tensor:
        H, W = x_size
        B, _, C = x.shape

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        # Collassa la dimensione spaziale delle finestre in un vettore
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            # Se la risoluzione di input non corrisponde a quella attesa, genera una nuova maschera
            attn_windows = self.attn(
                x_windows, mask=self.calculate_mask(x_size).to(x.device)
            )

        # Reshape in windows 2D
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # Ricompone le finestre in un tensore di dimensione originale (B, H, W, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Shift back
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x

        # Ritorna alla forma in sequenza di token
        x = x.view(B, H * W, C)
        # Attention + shortcut
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x


class RSTB(nn.Module):
    """
    Residual Swin Transformer Block (RSTB)
    Applica un BasicLayer e una connessione residua
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        img_size: int,
        patch_size: int,
        resi_connection: Literal["1conv", "3conv"],
    ):
        super(RSTB, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            layer = SwinTransformerBlock(
                dim=dim,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
            )
            self.layers.append(layer)

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

    def forward(self, x: Tensor, x_size: tuple[int]) -> Tensor:
        # TODO: non so se ha senso mettere il residuo qua, dopo faccio
        # unembed e lo passeró come residuo al finale

        res = x
        for layer in self.layers:
            x = layer(x, x_size)

        return self.patch_embed(self.conv(self.patch_unembed(x, x_size))) + res


class MyModel(nn.Module):
    def __init__(self, cfg: ConfigMyModel):
        super(MyModel, self).__init__()

        self.scale = cfg.model.scale
        self.img_size = cfg.model.img_size
        self.features = cfg.model.shallow_features
        self.num_in_channels = 4
        self.emb_patch_size = cfg.model.emb_patch_size
        self.emb_dim = cfg.model.embed_dim

        # Shallow feature extraction
        self.conv1 = nn.Conv2d(
            self.num_in_channels, self.features, kernel_size=3, stride=1, padding=1
        )

        # Base upsample branch (directly from input) to which residual refinements are added
        self.base_upsample = nn.Sequential(
            nn.Conv2d(
                self.num_in_channels,
                self.num_in_channels * self.scale * self.scale,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.PixelShuffle(self.scale),
        )

        # Embedding (tokenization) at shallow feature level
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.emb_patch_size,
            in_chans=self.features,
            embed_dim=self.emb_dim,
        )
        self.patch_unembed = PatchUnEmbed(
            img_size=self.img_size,
            patch_size=self.emb_patch_size,
            in_chans=self.emb_dim,
            embed_dim=self.emb_dim,
        )

        # Swin-based residual transformer layers (operate in token space)
        token_grid = (
            self.img_size // self.emb_patch_size,
            self.img_size // self.emb_patch_size,
        )

        self.num_layers = len(cfg.model.depths)
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = RSTB(
                dim=self.emb_dim,
                input_resolution=token_grid,
                depth=cfg.model.depths[i],
                num_heads=cfg.model.num_heads[i],
                window_size=cfg.model.window_size[i],
                img_size=self.img_size,
                patch_size=self.emb_patch_size,
                resi_connection=cfg.model.resi_connection,
            )
            self.layers.append(layer)

        # Convolution after body (on spatial features) before producing residuals
        self.conv_after_body = nn.Conv2d(
            self.emb_dim, self.emb_dim, kernel_size=3, stride=1, padding=1
        )

        # Per-layer residual refinement heads:
        # 1) upsample features to HR via PixelShuffle
        # 2) project to output channels (num_in_channels) as residual detail map
        self.residual_up_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    self.emb_dim,
                    self.emb_dim * self.scale * self.scale,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.residual_shuffle = nn.PixelShuffle(self.scale)
        self.residual_out_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    self.emb_dim,
                    self.num_in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Optional global refinement at HR (can be identity or small conv stack)
        self.hr_refine = nn.Sequential(
            nn.Conv2d(self.num_in_channels, self.num_in_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_in_channels, self.num_in_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Strategy:
        1. Base upsample of input (coarse SR baseline).
        2. Extract shallow features, tokenize and pass through transformer layers.
        3. After each layer: obtain spatial features, refine (conv), upsample to HR, project to residual output and accumulate.
        4. Final HR refinement.
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, "Input size mismatch"

        # Base HR prediction
        hr = self.base_upsample(x)  # (B, C, H*scale, W*scale)

        # Shallow features
        feat = self.conv1(x)  # (B, features, H, W)

        # Tokens
        tokens = self.patch_embed(feat)  # (B, N, emb_dim)
        token_grid = [
            self.img_size // self.emb_patch_size,
            self.img_size // self.emb_patch_size,
        ]

        for i, layer in enumerate(self.layers):
            # Transformer residual block
            tokens = layer(tokens, token_grid)
            # Back to spatial features
            spatial = self.patch_unembed(tokens, token_grid)  # (B, emb_dim, H, W)
            # Local conv refinement in LR space
            spatial = self.conv_after_body(spatial)
            # Upsample to HR and project to residual map
            up = self.residual_up_convs[i](spatial)
            up = self.residual_shuffle(up)  # (B, emb_dim, H*scale, W*scale)
            res = self.residual_out_convs[i](up)  # (B, C, H*scale, W*scale)
            hr = hr + res  # accumulate progressively finer details

        # Final refinement
        hr = self.hr_refine(hr) + hr
        return hr
