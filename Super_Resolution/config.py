import os
import yaml
from dataclasses import dataclass
from data_utils import ComuneType


@dataclass
class ModelConfig:
    name: str
    dir_path: str
    scale: int
    patch_size: int
    residual_groups: int
    feature_extraction_channels: int
    reduction_channels: int


@dataclass
class TrainConfig:
    comune: ComuneType
    seed: int
    workers: int
    dataset_size: int
    augment_data: bool
    batch_size: int
    epochs: int
    show_progress: bool


@dataclass
class TestConfig:
    load_model: bool
    comune: ComuneType
    dataset_size: int
    image_samples: int
    run_napari: bool


class Config:
    def __init__(self, config_path: str = "Super_Resolution/rcan/config.yml"):
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        self.model = ModelConfig(**config_dict["model"])
        self.train = TrainConfig(**config_dict["train"])
        self.test = TestConfig(**config_dict["test"])

        # Controlliamo che la directory dove salvare i dati esista
        if not os.path.exists(os.path.join(self.model.dir_path, self.model.name)):
            os.makedirs(os.path.join(self.model.dir_path, self.model.name))

    def __repr__(self):
        return f"Config(model={self.model}, train={self.train}, test={self.test})"
