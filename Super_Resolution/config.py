import os
import yaml
from dataclasses import dataclass
from typing import Literal
from data_utils import ComuneType


@dataclass
class ModelConfig:
    name: str
    dir_path: str
    scale: int
    img_size: int


@dataclass
class RCANModelConfig(ModelConfig):
    residual_groups: int
    feature_extraction_channels: int
    reduction_channels: int


@dataclass
class Swin2MoseModelConfig(ModelConfig):
    patch_size: int
    num_feat: int
    embed_dim: int
    depths: list[int]
    num_heads: list[int]
    window_size: int
    mlp_ratio: float
    upsampler: str
    resi_connection: Literal["1conv", "3conv"]
    MoE_config: dict


@dataclass
class MyModelConfig(ModelConfig):
    num_feat: int
    emb_patch_size: int
    embed_dim: int
    depths: list[int]
    num_heads: list[int]
    window_size: int
    resi_connection: Literal["1conv", "3conv"]
    upsampler: str


@dataclass
class TrainConfig:
    seed: int
    workers: int
    dataset_size: int
    augment_data: bool
    syntetic_data: bool
    batch_size: int
    epochs: int
    show_progress: bool
    use_moe_loss: bool
    loss_weights: dict[str, float]


@dataclass
class TestConfig:
    load_model: bool
    comune: ComuneType
    dataset_size: int
    batch_size: int
    image_samples: int
    run_napari: bool


class Config:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        self.model = ModelConfig(**config_dict["model"])
        self.train = TrainConfig(**config_dict["train"])
        self.test = TestConfig(**config_dict["test"])

        if not os.path.exists(os.path.join(self.model.dir_path, self.model.name)):
            os.makedirs(os.path.join(self.model.dir_path, self.model.name))

    def __repr__(self):
        return f"Config(model={self.model}, train={self.train}, test={self.test})"


class ConfigRCAN(Config):
    def __init__(self, config_path="Super_Resolution/rcan/config.yml"):
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        self.model = RCANModelConfig(**config_dict["model"])
        self.train = TrainConfig(**config_dict["train"])
        self.test = TestConfig(**config_dict["test"])

        if not os.path.exists(os.path.join(self.model.dir_path, self.model.name)):
            os.makedirs(os.path.join(self.model.dir_path, self.model.name))


class ConfigSwin2Mose(Config):
    def __init__(self, config_path="Super_Resolution/swin2mose/config.yml"):
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        self.model = Swin2MoseModelConfig(**config_dict["model"])
        self.train = TrainConfig(**config_dict["train"])
        self.test = TestConfig(**config_dict["test"])

        # Controlliamo che embedded dim sia divisibile per ogni testa nella lista num_heads
        if not all(self.model.embed_dim % n == 0 for n in self.model.num_heads):
            raise ValueError(
                f"embed_dim {self.model.embed_dim} must be divisible by every num_heads in {self.model.num_heads}"
            )

        if self.model.img_size % self.model.window_size != 0:
            raise ValueError(
                f"patch_size {self.model.img_size} must be divisible by window_size {self.model.window_size}"
            )

        if not os.path.exists(os.path.join(self.model.dir_path, self.model.name)):
            os.makedirs(os.path.join(self.model.dir_path, self.model.name))


class ConfigMyModel(Config):
    def __init__(self, config_path="Super_Resolution/mymodel/config.yml"):
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        self.model = MyModelConfig(**config_dict["model"])
        self.train = TrainConfig(**config_dict["train"])
        self.test = TestConfig(**config_dict["test"])

        # Controlliamo che embedded dim sia divisibile per ogni testa nella lista num_heads
        if not all(self.model.embed_dim % n == 0 for n in self.model.num_heads):
            raise ValueError(
                f"embed_dim {self.model.embed_dim} must be divisible by every num_heads in {self.model.num_heads}"
            )

        if self.model.img_size % self.model.img_size != 0:
            raise ValueError(
                f"patch_size {self.model.img_size} must be divisible by window_size {self.model.window_size}"
            )

        if not os.path.exists(os.path.join(self.model.dir_path, self.model.name)):
            os.makedirs(os.path.join(self.model.dir_path, self.model.name))
