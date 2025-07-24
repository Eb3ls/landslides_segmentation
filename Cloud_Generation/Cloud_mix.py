import torch
import torch.nn.functional as F
import math


def cloud_hue(input, cloud):
    """Genera la base di colore per la nuvola calcolando il colore medio del cielo chiaro."""
    mean_color = input.mean(dim=(-2, -1), keepdim=True)  # (B, C, 1, 1)
    return mean_color.expand_as(input)


def gaussian_kernel(kernel_size: int, sigma: float, device='cpu'):
    """Genera un kernel 2D gaussiano normalizzato."""
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device)
    xx, yy = torch.meshgrid([ax, ax], indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def apply_gaussian_blur(img, sigma: float):
    """Applica blur gaussiano a un'immagine (B, C, H, W) con sigma specificato."""
    B, C, H, W = img.shape
    kernel_size = int(2 * round(sigma * 3) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = gaussian_kernel(kernel_size, sigma, device=img.device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(C, 1, 1, 1)

    padding = kernel_size // 2
    img_padded = F.pad(img, (padding, padding, padding, padding), mode='reflect')
    blurred = F.conv2d(img_padded, kernel, groups=C)
    return blurred


def local_gaussian_blur(img, sigma_map, max_kernel_size=21):
    """Applica blur gaussiano localmente variabile, calcolando sigma medio per ogni immagine del batch."""
    B, C, H, W = img.shape
    blurred = torch.zeros_like(img)
    for b in range(B):
        sigma_val = sigma_map[b].mean().item()
        if sigma_val < 0.01:
            blurred[b] = img[b]
        else:
            blurred[b] = apply_gaussian_blur(img[b].unsqueeze(0), sigma_val).squeeze(0)
    return blurred


def mix(input, cloud, shadow=None, channel_magnitude=None, blur_scaling=2.0, cloud_color=True, invert=False):
    """
    Applica una maschera di nuvole a un'immagine satellitare.

    Args:
        input (Tensor): immagine originale (B, C, H, W)
        cloud (Tensor): maschera di nuvole (B, C, H, W)
        shadow (Tensor): maschera ombre (opzionale)
        channel_magnitude (Tensor): intensità per canale (opzionale)
        blur_scaling (float): se > 0, applica blur basato sulla densità nuvolosa
        cloud_color (bool): se True, la nuvola ha colore realistico
        invert (bool): se True, applica effetto opposto (uso termico, ecc.)

    Returns:
        Tensor: immagine con nuvole simulate (B, C, H, W)
    """
    if channel_magnitude is None:
        channel_magnitude = torch.ones(*input.shape[:-2], 1, 1, device=input.device)
    else:
        channel_magnitude = channel_magnitude.view(*input.shape[:-2], 1, 1)

    if shadow is not None:
        input = mix(input, shadow, blur_scaling=0.0, cloud_color=False, invert=not invert)

    if blur_scaling != 0.0:
        modulator = cloud.mean(1)  # media sul canale (B, H, W)
        input = local_gaussian_blur(input, blur_scaling * modulator)

    if invert:
        output = input * (1 - cloud.clamp(0, 1))
    else:
        max_lvl = cloud.max() if cloud.max() > 1.0 else 1.0
        cloud_base = torch.ones_like(input) if not cloud_color else cloud_hue(input, cloud)
        cloud_base = channel_magnitude * cloud_base
        output = input * (1 - cloud / max_lvl) + max_lvl * cloud_base * (cloud / max_lvl)

    return output