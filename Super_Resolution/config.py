import os
import yaml
from dataclasses import dataclass
from typing import Literal, TypeVar, Generic, Type, Union
from abc import ABC, abstractmethod
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
class PSWinModelConfig(ModelConfig):
    num_feat: int
    emb_patch_size: int
    embed_dim: int
    depths: list[int]
    num_heads: list[int]
    window_size: int
    resi_connection: Literal["1conv", "3conv"]
    upsampler: str
    multiscale_weights: list[float]


@dataclass
class DRCTModelConfig(ModelConfig):
    patch_size: int
    in_chans: int
    embed_dim: int
    depths: list[int]
    num_heads: list[int]
    window_size: int
    overlap_ratio: float
    mlp_ratio: float
    qkv_bias: bool
    drop_rate: float
    attn_drop_rate: float
    upsampler: str
    resi_connection: Literal["1conv", "3conv"]
    gc: int


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
    # opzionale
    info: str | None = None


@dataclass
class TestConfig:
    load_model: bool
    comune: ComuneType
    dataset_size: int
    batch_size: int
    image_samples: int
    run_napari: bool


# Generic Type Variable
M = TypeVar("M", bound=ModelConfig)


class ConfigValidator(ABC):
    """Validatore astratto per configurazioni"""

    @staticmethod
    @abstractmethod
    def validate_model(
        model_config,
    ) -> None:
        pass


class RCANConfigValidator(ConfigValidator):
    """Validatore per modelli RCAN"""

    @staticmethod
    def validate_model(model_config: RCANModelConfig) -> None:
        # Validazioni specifiche per RCAN se necessarie
        if model_config.residual_groups <= 0:
            raise ValueError("residual_groups must be positive")
        if model_config.feature_extraction_channels <= 0:
            raise ValueError("feature_extraction_channels must be positive")


class TransformerConfigValidator(ConfigValidator):
    """Validatore per modelli basati su Transformer"""

    @staticmethod
    def validate_model(
        model_config: Union[
            Swin2MoseModelConfig, MyModelConfig, PSWinModelConfig, DRCTModelConfig
        ],
    ) -> None:
        # Controllo embed_dim divisibile per num_heads
        if hasattr(model_config, "embed_dim") and hasattr(model_config, "num_heads"):
            if not all(model_config.embed_dim % n == 0 for n in model_config.num_heads):
                raise ValueError(
                    f"embed_dim {model_config.embed_dim} must be divisible by every num_heads in {model_config.num_heads}"
                )

        # Controllo window_size
        if hasattr(model_config, "window_size"):
            if model_config.img_size % model_config.window_size != 0:
                raise ValueError(
                    f"img_size {model_config.img_size} must be divisible by window_size {model_config.window_size}"
                )


class Config(Generic[M]):
    """Classe configurazione generica"""

    def __init__(
        self, config_path: str, model_class: Type[M], validator: Type[ConfigValidator]
    ):
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        self.model: M = model_class(**config_dict["model"])
        self.train = TrainConfig(**config_dict["train"])
        self.test = TestConfig(**config_dict["test"])

        # Validazione
        validator.validate_model(self.model)

        # Creazione directory se non esiste
        model_dir = os.path.join(self.model.dir_path, self.model.name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def __repr__(self):
        return f"Config(model={self.model}, train={self.train}, test={self.test})"


# Factory per creare configurazioni specifiche
class ConfigFactory:
    """Factory per creare configurazioni - Gestione semplificata"""

    # Mappatura modello -> (ConfigClass, ValidatorClass, DefaultPath)
    _MODEL_MAPPING = {
        "rcan": (
            RCANModelConfig,
            RCANConfigValidator,
            "Super_Resolution/rcan/config.yml",
        ),
        "swin2mose": (
            Swin2MoseModelConfig,
            TransformerConfigValidator,
            "Super_Resolution/swin2mose/config.yml",
        ),
        "mymodel": (
            MyModelConfig,
            TransformerConfigValidator,
            "Super_Resolution/mymodel/config.yml",
        ),
        "pswin": (
            PSWinModelConfig,
            TransformerConfigValidator,
            "Super_Resolution/pswin/config.yml",
        ),
        "drct": (
            DRCTModelConfig,
            TransformerConfigValidator,
            "Super_Resolution/myDRCT/config.yml",
        ),
    }

    @staticmethod
    def create_config(model_name: str, config_path: str | None = None):
        """
        Crea una configurazione per il modello specificato.

        Args:
            model_name: Nome del modello ("rcan", "swin2mose", "mymodel", "pswin", "drct")
            config_path: Percorso personalizzato del file config.yml (opzionale)

        Returns:
            Config object tipizzato per il modello specifico

        """
        if model_name not in ConfigFactory._MODEL_MAPPING:
            available = list(ConfigFactory._MODEL_MAPPING.keys())
            raise ValueError(
                f"Modello '{model_name}' non supportato. Disponibili: {available}"
            )

        model_class, validator_class, default_path = ConfigFactory._MODEL_MAPPING[
            model_name
        ]
        path = config_path if config_path else default_path

        return Config(path, model_class, validator_class)

    @staticmethod
    def create_rcan_config(
        config_path: str = "Super_Resolution/rcan/config.yml",
    ) -> Config[RCANModelConfig]:
        return ConfigFactory.create_config("rcan", config_path)

    @staticmethod
    def create_swin2mose_config(
        config_path: str = "Super_Resolution/swin2mose/config.yml",
    ) -> Config[Swin2MoseModelConfig]:
        return ConfigFactory.create_config("swin2mose", config_path)

    @staticmethod
    def create_mymodel_config(
        config_path: str = "Super_Resolution/mymodel/config.yml",
    ) -> Config[MyModelConfig]:
        return ConfigFactory.create_config("mymodel", config_path)

    @staticmethod
    def create_pswin_config(
        config_path: str = "Super_Resolution/pswin/config.yml",
    ) -> Config[PSWinModelConfig]:
        return ConfigFactory.create_config("pswin", config_path)

    @staticmethod
    def create_drct_config(
        config_path: str = "Super_Resolution/myDRCT/config.yml",
    ) -> Config[DRCTModelConfig]:
        return ConfigFactory.create_config("drct", config_path)


ConfigRCAN = ConfigFactory.create_rcan_config
ConfigSwin2Mose = ConfigFactory.create_swin2mose_config
ConfigMyModel = ConfigFactory.create_mymodel_config
ConfigPSWin = ConfigFactory.create_pswin_config
ConfigDRCT = ConfigFactory.create_drct_config


def load_config(model_name: str, config_path: str | None = None):
    """
    Funzione semplificata per caricare una configurazione.

    Args:
        model_name: Nome del modello ("rcan", "swin2mose", "mymodel", "pswin", "drct")
        config_path: Percorso personalizzato del file config.yml (opzionale)

    Returns:
        Config object configurato per il modello

    Examples:
        # Uso base con percorsi default
        config = load_config("drct")
        config = load_config("rcan")

        # Uso con percorso personalizzato
        config = load_config("drct", "my_custom_config.yml")
    """
    return ConfigFactory.create_config(model_name, config_path)
