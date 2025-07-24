import os
import numpy as np
import torch
from PIL import Image
import rasterio
from Cloud_generator import generate_perlin
from Cloud_mix import mix
import matplotlib.pyplot as plt

# Impostazioni
output_dir = "cloud_animation_frames"
os.makedirs(output_dir, exist_ok=True)

H, W = 640, 640 # Output resolution (crop size)
n_frames = 20
step = 20
cloud_opacity = 1 # densità nuvole -> 0 (Non si vedono), 1 (tutte bianche, non si nota nulla sotto)
cloud_scale = 2

# Definizione della direzione del movimento:
# 'right', 'left', 'up', 'down', 'up_right', 'up_left', 'down_right', 'down_left'
movement_direction = 'up_right'
x_direction_factor = 0 
y_direction_factor = 0 

if movement_direction == 'right':
    x_direction_factor = -1
elif movement_direction == 'left':
    x_direction_factor = 1
elif movement_direction == 'up':
    y_direction_factor = 1
elif movement_direction == 'down':
    y_direction_factor = -1
elif movement_direction == 'up_right':
    x_direction_factor = -1
    y_direction_factor = 1
elif movement_direction == 'up_left':
    x_direction_factor = 1
    y_direction_factor = 1
elif movement_direction == 'down_right':
    x_direction_factor = -1
    y_direction_factor = -1
elif movement_direction == 'down_left':
    x_direction_factor = 1
    y_direction_factor = -1


tif_path = "../Comuni/PREDAPPIO/Agea_2020_2m.tif"
with rasterio.open(tif_path) as src:
    bands = src.read([1, 2, 3])
    bands = np.clip(bands, 0, np.percentile(bands, 99))
    bands = bands / bands.max()
    bands = bands.astype(np.float32)

    full_H, full_W = bands.shape[1:]
    top = (full_H - H) // 2
    left = (full_W - W) // 2
    bands_crop = bands[:, top:top+H, left:left+W]

img_tensor = torch.FloatTensor(bands_crop)   # (3, H, W)

#  Crea maschera di nuvole 
# Per semplicità, possiamo aumentarla sufficientemente grande.
# Calcoliamo la dimensione massima di spostamento in X e Y.
max_x_offset = abs(x_direction_factor * n_frames * step)
max_y_offset = abs(y_direction_factor * n_frames * step)

# La dimensione della maschera di Perlin deve essere almeno (H + max_y_offset) x (W + max_x_offset)
# per assicurare che ci sia sempre contenuto disponibile per il crop.
# Potresti voler aggiungere un piccolo buffer extra per maggiore sicurezza.
required_big_H = H + max_y_offset + step # Aggiungiamo un extra 'step' come buffer
required_big_W = W + max_x_offset + step # Aggiungiamo un extra 'step' come buffer

cloud_mask_big = generate_perlin(
    shape=(max(H * cloud_scale, required_big_H), max(W * cloud_scale, required_big_W)),
    batch=1,
    device='cpu'
).squeeze(0) # (Maschera_Grande_H, Maschera_Grande_W)

# Calcola gli offset iniziali per centrare la finestra di movimento all'interno della maschera grande
# Questo aiuta a prevenire che le nuvole appaiano da un bordo e scompaiano troppo presto dall'altro.
# Si posiziona l'inizio del percorso di movimento nel centro della maschera grande,
# in modo che il movimento possa avvenire sia in direzioni positive che negative.
initial_x_offset = (cloud_mask_big.shape[1] - W) // 2 - (n_frames * step * x_direction_factor) // 2
initial_y_offset = (cloud_mask_big.shape[0] - H) // 2 - (n_frames * step * y_direction_factor) // 2

# Assicurati che gli offset non siano negativi e non superino i limiti della maschera grande
initial_x_offset = max(0, initial_x_offset)
initial_y_offset = max(0, initial_y_offset)

frames = []

for i in range(n_frames):
    # Calcola lo spostamento per il frame corrente
    # Lo spostamento viene applicato alla maschera di nuvole "madre"
    # Ad esempio, per muoversi a destra, x_start aumenterà.
    # Per muoversi a sinistra, x_start diminuirà.
    # Per muoversi in alto, y_start diminuirà.
    # Per muoversi in basso, y_start aumenterà.
    x_current_offset = int(initial_x_offset + i * step * x_direction_factor)
    y_current_offset = int(initial_y_offset + i * step * y_direction_factor)

    # Assicurati che i crop rimangano all'interno dei limiti della maschera_big
    x_end = x_current_offset + W
    y_end = y_current_offset + H

    # Se il movimento porta il crop fuori dai limiti della maschera grande,
    # significa che la maschera grande non è sufficientemente ampia o gli offset iniziali non sono corretti.
    # Aggiungiamo un controllo per evitare errori di indicizzazione.
    if x_end > cloud_mask_big.shape[1] or y_end > cloud_mask_big.shape[0]:
        print(f"Warning: Crop out of bounds at frame {i}. Consider increasing cloud_scale or adjusting initial offsets.")
        # Potresti voler terminare il ciclo o gestire l'errore in altro modo.
        break

    cloud_crop = cloud_mask_big[y_current_offset:y_end, x_current_offset:x_end]

    # Normalizzazione e opacità (rimane come prima)
    cloud_crop = (cloud_crop - cloud_crop.min()) / (cloud_crop.max() - cloud_crop.min())
    cloud_crop = cloud_crop * cloud_opacity

    cloud_mask_3ch = cloud_crop.unsqueeze(0).expand(3, -1, -1)

    img_batch = img_tensor.unsqueeze(0)
    cloud_batch = cloud_mask_3ch.unsqueeze(0)

    # Aggiunge la nuvola appena creata all'immagine
    cloudy_frame_batch = mix(
        img_batch,
        cloud_batch,
        blur_scaling=0.0,
        cloud_color=False
    )

    cloudy_frame = cloudy_frame_batch.squeeze(0)
    frame_np = (cloudy_frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    frame_img = Image.fromarray(frame_np)
    frames.append(frame_img)

    frame_img.save(os.path.join(output_dir, f"frame_{i:03}.png"))

# Crea GIF 
gif_path = "cloud_animation.gif"
if frames:
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0
    )

    print(f"GIF salvata in: {gif_path}")
else:
    print("Nessun frame generato per la GIF.")



# Visualizza immagine originale
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 1. Immagine originale
axs[0].imshow(np.transpose(img_tensor.numpy(), (1, 2, 0)))
axs[0].set_title("Immagine originale")
axs[0].axis('off')

# 2. Maschera di nuvole (frame 0)
axs[1].imshow(cloud_mask_big[initial_y_offset:initial_y_offset+H, initial_x_offset:initial_x_offset+W], cmap='gray')
axs[1].set_title("Maschera nuvolosa (frame 0)")
axs[1].axis('off')

# 3. Primo frame con nuvola
axs[2].imshow(frames[0])
axs[2].set_title("Frame 0 con nuvola")
axs[2].axis('off')

plt.tight_layout()
plt.show()
