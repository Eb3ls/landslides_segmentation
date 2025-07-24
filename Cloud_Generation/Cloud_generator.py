import torch
import numpy as np
from Perlin_noise import perlin, output_transform


def generate_perlin(
    scales=None,
    shape=(256, 256),
    batch=1,
    device='cpu',
    weights=None,
    const_scale=True,
    decay_factor=1
):
    """
    Genera maschera di Perlin noise multi-scala per simulare nuvole.
    """
    # Scelte delle scale se non specificate
    if scales is None:
        up_lim = max([2, int(np.log2(min(shape))) - 1])
        scales = [2**i for i in range(2, up_lim)]

        if const_scale:
            f = int(2**np.floor(np.log2(0.25 * max(shape) / max(scales))))
            scales = [el * f for el in scales]

    # Pesi associati alle scale
    if weights is None:
        weights = [el**decay_factor for el in scales]

    # Pad della forma a potenza di due
    big_shape = [int(2**(np.ceil(np.log2(i)))) for i in shape]
    out = torch.zeros([batch, *shape], device=device)

    for scale, weight in zip(scales, weights):
        noise = perlin(
            int(big_shape[0] / scale),
            int(big_shape[1] / scale),
            scale=scale,
            batch=batch,
            device=device
        )[..., :shape[0], :shape[1]]
        out += weight * noise

    return output_transform(out)
