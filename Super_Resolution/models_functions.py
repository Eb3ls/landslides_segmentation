import math
from typing import Literal
from torch import Tensor, nn


def window_reverse(windows: Tensor, window_size: int, H: int, W: int) -> Tensor:
    """
    Ricompone le finestre in un tensore di dimensione (B, H, W, C).

    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size
        H: Height of image
        W: Width of image

    Returns:
        x: (B, H, W, C)
    """
    # Ricalcola il batch size
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        # Ricostruisce la view intermedia B, num_windows_H, window_size, num_windows_W, window_size, C
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1,
    )
    # Ricompone nella posizione spaziale originale
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


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


def window_partition(x: Tensor, window_size: int) -> Tensor:
    """
    Scompone un tensore in finestre di dimensione window_size x window_size.

    Args:
        x: (B, H, W, C)
        window_size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # B, num_windows_H, window_size, num_windows_W, window_size, C
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous()
        # Collassa le dimensioni Bx num_windows_H num_windows_W sul primo asse
        .view(-1, window_size, window_size, C)
    )
    return windows


# =============== #


class PatchEmbed(nn.Module):
    """
    Converte un tensore di dimensione (B, C, H, W) in un tensore di dimensione (B, N, C)
    dove N = num_patches = (H // patch_size) * (W // patch_size)
    Tokenizza l'immagine in patch non sovrapposte.
    """

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        img_size_list = [img_size, img_size]
        patch_size_list = [patch_size, patch_size]
        patch_resolution = [img_size // patch_size, img_size // patch_size]
        self.img_size = img_size_list
        self.patch_size = patch_size_list
        self.patch_resolution = patch_resolution
        self.num_patches = patch_resolution[0] * patch_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Avendo stride uguale al kernel le patch sono non sovrapposte
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        # Appiattimento delle dimensioni spaziali partendo dalla seconda inclusa e porta i canali in coda
        x = self.proj(x).flatten(2).transpose(1, 2)
        # Qui si potrebbe normalizzare
        return x


class PatchUnEmbed(nn.Module):
    """
    Converte un tensore di dimensione (B, HW, C) in un tensore di dimensione (B, C, X, X)
    dove X = img_size // patch_size
    Ogni token 1D viene riconvertito in immagine 2D.
    """

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        img_size_list = [img_size, img_size]
        patch_size_list = [patch_size, patch_size]
        patch_resolution = [img_size // patch_size, img_size // patch_size]
        self.img_size = img_size_list
        self.patch_size = patch_size_list
        self.patch_resolution = patch_resolution
        self.num_patches = patch_resolution[0] * patch_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x: Tensor, x_size: list[int]) -> Tensor:
        B, _, _ = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
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


class Upsample_hf(nn.Sequential):
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
        super(Upsample_hf, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)


def get_resi_connection(
    resi_connection: Literal["1conv", "3conv"], dim: int
) -> nn.Module:
    if resi_connection == "1conv":
        return nn.Conv2d(dim, dim, 3, 1, 1)
    elif resi_connection == "3conv":
        # Meno parametri riducendo e alla fine espandendo i canali (~ 1/2)
        return nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim // 4, dim, 3, 1, 1),
        )
