import matplotlib.pyplot as plt
import sys
import os
from huggingface_hub import hf_hub_download
from swin2_mose_files.utils import load_config, load_swin2_mose, run_swin2_mose

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import SuperResolutionDataset


model_weights = hf_hub_download(
    repo_id="isp-uv-es/superIX", filename="swin2_mose/weights/model-70.pt"
)
print(model_weights)


device = "cuda"
path = "swin2_mose/weights/config-70.yml"

# load config
cfg = load_config(path)

# load model
model = load_swin2_mose(model_weights, cfg)
model.to(device)
model.eval()

# load the dataset
eval_dataset = SuperResolutionDataset("Modigliana", 2, 64, 100)
lr_img, hr_img = eval_dataset[0]

# Convert tensors to numpy arrays (required by run_swin2_mose)
lr_img_np = lr_img.numpy()
hr_img_np = hr_img.numpy()

print(f"LR image shape: {lr_img_np.shape}")
print(f"HR image shape: {hr_img_np.shape}")

# Custom function to run with your data format

results = run_swin2_mose(model, lr_img_np, hr_img_np, device)


def plot_results(lr, sr, hr):
    lr_rgb = lr[0:3]  # Take first 3 channels for RGB
    lr_nir = lr[3]  # NIR channel
    sr_rgb = sr[0:3]  # Take first 3 channels for RGB
    sr_nir = sr[3]  # NIR channel
    hr_rgb = hr[0:3]  # Take first 3 channels for RGB
    hr_nir = hr[3]  # NIR channel

    # Plotting the results as lr, sr, and hr images
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0, 0].imshow(lr_rgb.transpose(1, 2, 0))
    axs[0, 0].set_title("Low Resolution (RGB)")
    axs[0, 1].imshow(sr_rgb.transpose(1, 2, 0))
    axs[0, 1].set_title("Super Resolution (RGB)")
    axs[0, 2].imshow(hr_rgb.transpose(1, 2, 0))
    axs[0, 2].set_title("High Resolution (RGB)")
    axs[1, 0].imshow(lr_nir, cmap="gray")
    axs[1, 0].set_title("Low Resolution (NIR)")
    axs[1, 1].imshow(sr_nir, cmap="gray")
    axs[1, 1].set_title("Super Resolution (NIR)")
    axs[1, 2].imshow(hr_nir, cmap="gray")
    axs[1, 2].set_title("High Resolution (NIR)")

    plt.tight_layout()
    plt.show()


plot_results(results["lr"], results["sr"], results["hr"])
