import torch
import numpy as np

# Questa funzione normalizza l'output del Perlin noise. 
# Prende un tensore x, lo divide per il suo valore assoluto massimo 
# (per garantire che i valori siano tra -1 e 1), e poi applica la funzione tanh(4x).
# La funzione tanh (tangente iperbolica) schiaccia i valori, 
# rendendo la distribuzione più contrastata e adatta a simulare la densità delle nuvole. y = tanh(4x)
def output_transform(x):
    x /= max([x.max(), -x.min()])
    return (4 * x).tanh()


# Implementa una funzione di interpolazione cubica smooth (3t^2−2t^3). 
# Questa funzione è cruciale per il Perlin noise perché garantisce transizioni fluide tra i valori del gradiente, 
# evitando artefatti a blocchi. f(t)=3t^2 −2t^3
def interp(t):
    return 3 * t**2 - 2 * t**3

# Questa è la funzione principale per generare il Perlin noise.
# Genera vettori di gradiente casuali (gx, gy) su una griglia. Questi vettori definiscono la "direzione" del rumore.
# Crea griglie di coordinate xs e ys normalizzate da 0 a 1.
# Calcola i "pesi" di interpolazione (wx, wy) usando la funzione interp per determinare l'influenza dei gradienti sui punti circostanti.
# Calcola il "dot product" (prodotto scalare) tra i vettori di distanza dai nodi della griglia e i vettori di gradiente. 
# Questo calcolo viene eseguito per i quattro angoli di ogni cella della griglia e interpolato per ottenere un valore continuo.
# Il risultato è un tensore di Perlin noise con la width, height e batch specificati.

def perlin(width, height, scale=10, batch=1, device=None):
    gx, gy = torch.randn(2, batch, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1, device=device)[:-1, None]
    ys = torch.linspace(0, 1, scale + 1, device=device)[None, :-1]

    wx = 1 - interp(xs)
    wy = 1 - interp(ys)

    dots = 0
    dots += wx * wy * (gx[:, :-1, :-1] * xs + gy[:, :-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[:, 1:, :-1] * (1 - xs) + gy[:, 1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:, :-1, 1:] * xs - gy[:, :-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[:, 1:, 1:] * (1 - xs) - gy[:, 1:, 1:] * (1 - ys))

    return dots.permute(0, 1, 3, 2, 4).contiguous().view(batch, width * scale, height * scale)


# Questa funzione genera una maschera di Perlin noise combinando diverse "ottava" (scale) di rumore. 
# Questo approccio multi-scala è fondamentale per la realisticità delle nuvole, poiché le nuvole reali presentano dettagli a diverse grandezze.
# scales: Se non specificato, le scale vengono determinate automaticamente in base alla dimensione dell'immagine, garantendo una copertura adeguata delle frequenze. 
# Vengono scelte potenze di 2.
# weights: A ogni scala viene assegnato un peso (el**decay_factor). 
# Tipicamente, le scale più grandi (bassa frequenza, dettagli grossolani) hanno un peso maggiore, 
# e le scale più piccole (alta frequenza, dettagli fini) hanno un peso decrescente, per simulare la morfologia tipica delle nuvole.
# Il processo itera su ogni scala, genera il Perlin noise per quella scala, lo scala per la dimensione finale desiderata, 
# e lo aggiunge al out complessivo moltiplicato per il suo weight.
# Infine, applica la output_transform per normalizzare e contrastare la maschera finale.
def generate_perlin(scales=None, shape=(256, 256), batch=1, device='cpu', weights=None, const_scale=True, decay_factor=1):
    if scales is None:
        up_lim = max([2, int(np.log2(min(shape))) - 1])
        scales = [2 ** i for i in range(2, up_lim)]
        if const_scale:
            f = int(2 ** np.floor(np.log2(0.25 * max(shape) / max(scales))))
            scales = [el * f for el in scales]

    if weights is None:
        weights = [el ** decay_factor for el in scales]

    big_shape = [int(2 ** (np.ceil(np.log2(i)))) for i in shape]
    out = torch.zeros([batch, *shape], device=device)
    for scale, weight in zip(scales, weights):
        out += weight * perlin(
            int(big_shape[0] / scale),
            int(big_shape[1] / scale),
            scale=scale,
            batch=batch,
            device=device
        )[..., :shape[0], :shape[1]]

    return output_transform(out)
