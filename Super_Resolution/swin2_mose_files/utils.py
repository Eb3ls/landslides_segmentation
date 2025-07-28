import torch
import yaml
import sys
import os

# Aggiungi la directory corrente al path per assicurarti che model.py sia trovato
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from swin2_mose_files.model import Swin2MoSE


def to_shape(t1, t2):
    t1 = t1[None].repeat(t2.shape[0], 1)
    t1 = t1.view((t2.shape[:2] + (1, 1)))
    return t1


def norm(tensor, mean, std):
    # get stats
    mean = torch.tensor(mean).to(tensor.device)
    std = torch.tensor(std).to(tensor.device)
    # denorm
    return (tensor - to_shape(mean, tensor)) / to_shape(std, tensor)


def denorm(tensor, mean, std):
    # get stats
    mean = torch.tensor(mean).to(tensor.device)
    std = torch.tensor(std).to(tensor.device)
    # denorm
    return (tensor * to_shape(std, tensor)) + to_shape(mean, tensor)


def load_config(path):
    # load config
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_swin2_mose(model_weights, cfg):
    # load checkpoint
    checkpoint = torch.load(model_weights)

    # build model
    sr_model = Swin2MoSE(**cfg["super_res"]["model"])
    sr_model.load_state_dict(checkpoint["model_state_dict"])

    sr_model.cfg = cfg

    return sr_model


def run_swin2_mose(model, lr, hr, device="cuda"):

    # select 10m lr bands: B02, B03, B04, B08 and hr bands (B G R NIR)
    lr_orig = lr[[2, 1, 0, 3]]  # B, G, R, NIR
    lr_orig = torch.from_numpy(lr_orig)[None].to(device)

    # predict a image
    with torch.no_grad():
        sr = model(lr_orig)
        if not torch.is_tensor(sr):
            sr, _ = sr

    # Printing shape, min, max
    print(f"sr shape: {sr.shape}, min: {sr.min()}, max: {sr.max()}")

    # Convert to numpy
    sr = sr.squeeze().cpu().numpy()
    sr = sr[[2, 1, 0, 3]]  # Reorder back to [R,G,B,NIR]

    # New shape after interpolation
    print(f"sr shape after interpolation: {sr.shape}, min: {sr.min()}, max: {sr.max()}")

    return {"lr": lr, "sr": sr, "hr": hr}
