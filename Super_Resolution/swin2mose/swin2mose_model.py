import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from typing import Literal, cast

from timm.layers.weight_init import trunc_normal_

from Super_Resolution.config import Config, Swin2MoseModelConfig
from Super_Resolution.swin2mose.moe import MoE
from Super_Resolution.models_functions import (
    PatchEmbed,
    PatchUnEmbed,
    UpsampleOneStep,
    get_resi_connection,
    window_partition,
    window_reverse,
)


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

        ######## Relative Position Bias impostato a True seguendo l'articolo ########

        # Si crea una tabella 2*Wh-1 * 2*Ww-1, num_heads,
        # copre tutte le possibili differenze di posizioni relative tra due patch
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

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
        rpe_relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("rpe_relative_position_index", rpe_relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=0.02)

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

        # LePE
        lepe = self.lepe_pos(v)

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
        # Sommata Local Enhancement Positional Encoding finale
        x = x + lepe

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
        MoE_config: dict,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
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

        self.norm2 = norm_layer(dim)
        # Definiti quanti neuroni ci sono nella MLP rispetto a dim di input
        mlp_hidden_dim = int(dim * mlp_ratio)

        # Nell'articolo si usa MoE
        self.moe = MoE(
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

    def calculate_mask(self, x_size: tuple[int, int]) -> Tensor:
        # calculate attention mask for SW-MSA
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

    def forward(self, x: Tensor, x_size: tuple[int, int]) -> tuple[Tensor, Tensor]:
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
        x = shortcut + self.norm1(x)

        moe_out, loss_moe = self.moe(x)
        # MoE + shortcut
        x = x + self.norm2(moe_out)

        return x, loss_moe

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )


class RSTB(nn.Module):
    """
    Residual Swin Transformer Block (RSTB)
    Applica direttamente i SwinTransformerBlock senza BasicLayer intermedio
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        MoE_config: dict,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer=nn.LayerNorm,
        img_size: int = 128,
        patch_size: int = 4,
        resi_connection: Literal["1conv", "3conv"] = "1conv",
    ):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        # Gestione diretta dei blocchi come in MyModel
        self.layers = nn.ModuleList()
        for i in range(depth):
            layer = SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
                pretrained_window_size=0,
                MoE_config=MoE_config,
            )
            self.layers.append(layer)

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

    def forward(self, x: Tensor, x_size: tuple[int, int]) -> tuple[Tensor, Tensor]:
        """
        Gestisce direttamente i blocchi e accumula la loss MoE.
        Ritorna: (features, loss_moe_all)
        """
        # Accumula loss MoE come Tensor
        loss_moe_all: Tensor = torch.zeros((), device=x.device, dtype=x.dtype)

        # Applica tutti i layer come in MyModel
        res = x
        for layer in self.layers:
            x, loss_moe = layer(x, x_size)
            loss_moe_all += loss_moe.to(device=x.device, dtype=x.dtype)

        # Applica connessione residua con patch embed/unembed
        x = self.patch_embed(self.conv(self.patch_unembed(x, x_size))) + res

        return x, loss_moe_all


class Swin2MoSE(nn.Module):
    def __init__(
        self,
        cfg: Config[Swin2MoseModelConfig],
        norm_layer=nn.LayerNorm,
        dropout_rate: float = 0.0,
        # Parametri erediati da SwinTransformer, lascio default
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_dropout_rate: float = 0.0,
    ):
        super(Swin2MoSE, self).__init__()
        num_in_ch = 4
        num_feat = cfg.model.num_feat
        self.img_range = 1
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
            img_size=cfg.model.img_size,
            patch_size=cfg.model.patch_size,
            in_chans=self.embed_dim,
            embed_dim=self.embed_dim,
        )
        # Converti esplicitamente a tuple per consistenza con MyModel
        patches_resolution: tuple[int, int] = tuple(self.patch_embed.patch_resolution)  # type: ignore
        self.patches_resolution = patches_resolution

        # Layer di unembedding
        self.patch_unembed = PatchUnEmbed(
            img_size=cfg.model.img_size,
            patch_size=cfg.model.patch_size,
            in_chans=self.embed_dim,
            embed_dim=self.embed_dim,
        )

        # Costruiamo i Residual Swin Transformer Blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=self.embed_dim,
                input_resolution=self.patches_resolution,  # type: ignore
                depth=cfg.model.depths[i_layer],
                num_heads=cfg.model.num_heads[i_layer],
                window_size=cfg.model.window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=dropout_rate,
                attn_drop=attn_dropout_rate,
                norm_layer=norm_layer,
                img_size=cfg.model.img_size,
                patch_size=cfg.model.patch_size,
                resi_connection=cfg.model.resi_connection,
                MoE_config=cfg.model.MoE_config,
            )
            self.layers.append(layer)

        if self.upsampler in ("pixelshuffle", "pixelshuffle_hf"):
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
                    norm_layer=norm_layer,
                    img_size=cfg.model.img_size,
                    patch_size=cfg.model.patch_size,
                    resi_connection=cfg.model.resi_connection,
                    MoE_config=cfg.model.MoE_config,
                )
                self.layers_hf.append(layer)

        self.norm = norm_layer(self.num_features)

        # Costruiamo l'ultimo layer di convoluzione
        self.conv_after_body = get_resi_connection(
            cfg.model.resi_connection, self.embed_dim
        )

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        ############ 3. Ricostruzione dell'immagine ############
        if self.upsampler == "pixelshuffledirect":
            # Semplice upsample con convoluzione 2D e pixel shuffle per upsampling
            self.upsample = UpsampleOneStep(
                cfg.model.scale,
                self.embed_dim,
                num_in_ch,
                (patches_resolution[0], patches_resolution[1]),
            )
        elif self.upsampler == "nearest+conv":
            print("Using nearest+conv upsampler")
            # Two 2x nearest stages + final 1.25x bicubic with refinement
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(self.embed_dim, num_feat, 3, 1, 1, padding_mode="reflect"),
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
            self.conv_last = nn.Conv2d(
                num_feat, num_in_ch, 3, 1, 1, padding_mode="reflect"
            )

    # Esclusi se contengono le keyword specificate
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def check_image_size(self, x: Tensor) -> Tensor:
        """
        Controlla se l'immagine ha dimensioni multiple della finestra.
        Se no, applica un padding riflessivo per renderle tali.
        """
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        if mod_pad_h != 0 or mod_pad_w != 0:
            print(f"Padding applied: (0, {mod_pad_w}, 0, {mod_pad_h})")
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return x

    # Forward principale che esegue tutti i layers
    def forward_features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Shape HxW come list per compatibilità con RSTB
        x_size = [x.shape[2], x.shape[3]]
        # Tokenizzazione in patch non sovrapposte
        x = self.patch_embed(x)

        # Accumula loss MoE come Tensor
        loss_moe_all: Tensor = torch.zeros((), device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x, loss_moe = layer(x, x_size)
            loss_moe = loss_moe.to(device=x.device, dtype=x.dtype)
            loss_moe_all = loss_moe_all + loss_moe

        # Normalizzazione finale
        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x, loss_moe_all

    # Metodo principale per il forward del modello
    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        # MoE loss totale come Tensor
        total_moe_loss: Tensor = torch.zeros((), device=x.device, dtype=x.dtype)
        if self.upsampler == "pixelshuffle":
            # for classical SR
            x = self.conv_first(x)

            res, moe_loss = self.forward_features(x)
            total_moe_loss = total_moe_loss + moe_loss

            x = self.conv_after_body(res) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == "pixelshuffledirect":
            # Shallow features
            x = self.conv_first(x)

            # Deep features con layer RSTB
            res, moe_loss = self.forward_features(x)
            total_moe_loss = total_moe_loss + moe_loss

            # Residual connection
            x = self.conv_after_body(res) + x
            x = self.upsample(x)
        elif self.upsampler == "nearest+conv":
            # for real-world SR
            x = self.conv_first(x)

            res = x
            x, moe_loss = self.forward_features(x)
            total_moe_loss = total_moe_loss + moe_loss

            x = self.conv_after_body(x) + res
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

        # Crop del tensore alle dimensioni obiettivo nel caso in cui sia stato fatto padding
        out = x[:, :, : H * self.upscale, : W * self.upscale]
        return out, total_moe_loss
