import yaml
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeVar, Generic, Type, Union
from abc import ABC, abstractmethod
from data_utils import ComuneType


@dataclass
class ModelConfig:
    name: str
    dir_path: str
    scale: int
    img_size: int
    stats_file: str | None = field(default=None, kw_only=True)


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
    num_feat: int
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
    resi_connection: Literal["1conv", "identity"]
    gc: int
    MoE_config: dict


@dataclass
class TrainConfig:
    seed: int
    workers: int
    dataset_size: int
    augment_data: bool
    synthetic_data: bool
    batch_size: int
    epochs: int
    show_progress: bool
    use_moe_loss: bool
    loss_weights: dict[str, float]
    # opzionale
    info: str | None = None
    accumulation_steps: int = 1


@dataclass
class TestConfig:
    load_model: bool
    comune: ComuneType
    dataset_size: int
    batch_size: int
    image_samples: int
    run_napari: bool
    synthetic_data: bool = False


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
        _validate_common_model(model_config)
        if model_config.scale != 5:
            raise ValueError("RCAN currently implements only scale=5")
        if model_config.residual_groups <= 0:
            raise ValueError("residual_groups must be positive")
        if model_config.feature_extraction_channels <= 0:
            raise ValueError("feature_extraction_channels must be positive")
        if not 0 < model_config.reduction_channels <= model_config.feature_extraction_channels:
            raise ValueError(
                "reduction_channels must be positive and no greater than "
                "feature_extraction_channels"
            )


class TransformerConfigValidator(ConfigValidator):
    """Validatore per modelli basati su Transformer"""

    @staticmethod
    def validate_model(
        model_config: Union[
            Swin2MoseModelConfig, MyModelConfig, PSWinModelConfig, DRCTModelConfig
        ],
    ) -> None:
        _validate_common_model(model_config)
        if len(model_config.depths) != len(model_config.num_heads):
            raise ValueError("depths and num_heads must have the same length")
        if not model_config.depths or any(depth <= 0 for depth in model_config.depths):
            raise ValueError("depths must contain positive values")
        if any(heads <= 0 for heads in model_config.num_heads):
            raise ValueError("num_heads must contain positive values")
        if not all(model_config.embed_dim % heads == 0 for heads in model_config.num_heads):
            raise ValueError(
                f"embed_dim {model_config.embed_dim} must be divisible by every num_heads in {model_config.num_heads}"
            )
        if not isinstance(model_config, DRCTModelConfig) and model_config.resi_connection not in {"1conv", "3conv"}:
            raise ValueError("resi_connection must be '1conv' or '3conv'")

        if isinstance(model_config, Swin2MoseModelConfig):
            if model_config.scale != 5:
                raise ValueError("Swin2MoSE SR upsamplers currently support only scale=5")
            if model_config.patch_size != 1:
                raise ValueError(
                    "Swin2MoSE patch_size must be 1: its forward path unembeds at image resolution"
                )
            if model_config.window_size <= 0 or model_config.img_size % model_config.window_size != 0:
                raise ValueError("Swin2MoSE img_size must be divisible by window_size")
            if model_config.upsampler not in {"pixelshuffledirect", "nearest+conv"}:
                raise ValueError(
                    "Swin2MoSE upsampler must be 'pixelshuffledirect' or 'nearest+conv'"
                )
            if not isinstance(model_config.MoE_config, dict):
                raise ValueError("MoE_config must be a mapping")
            for key in ("num_experts", "k"):
                if key not in model_config.MoE_config:
                    raise ValueError(f"MoE_config must define '{key}'")
            num_experts = model_config.MoE_config["num_experts"]
            k = model_config.MoE_config["k"]
            if not isinstance(num_experts, int) or not isinstance(k, int) or not 0 < k <= num_experts:
                raise ValueError("MoE_config requires 0 < k <= num_experts")

        if isinstance(model_config, DRCTModelConfig):
            if model_config.scale != 5:
                raise ValueError("DRCT SR upsamplers currently support only scale=5")
            if model_config.patch_size != 1:
                raise ValueError(
                    "DRCT patch_size must be 1: its forward path unembeds at image resolution"
                )
            if model_config.window_size <= 0 or model_config.img_size % model_config.window_size != 0:
                raise ValueError("DRCT img_size must be divisible by window_size")
            if model_config.in_chans != 4:
                raise ValueError("DRCT currently supports exactly four input channels")
            if model_config.upsampler not in {
                "pixelshuffle",
                "nearest+conv",
                "test",
                "only_shuffle",
            }:
                raise ValueError("unsupported DRCT upsampler")
            if model_config.resi_connection not in {"1conv", "identity"}:
                raise ValueError("DRCT resi_connection must be '1conv' or 'identity'")
            if model_config.num_feat <= 0 or model_config.gc <= 0:
                raise ValueError("num_feat and gc must be positive")
            if any((model_config.embed_dim + step * model_config.gc) % heads != 0
                   for heads in model_config.num_heads for step in range(5)):
                raise ValueError(
                    "each num_heads value must divide embed_dim + n * gc for n=0..4"
                )


def _validate_common_model(model_config: ModelConfig) -> None:
    if not model_config.name:
        raise ValueError("model.name must not be empty")
    if not model_config.dir_path:
        raise ValueError("model.dir_path must not be empty")
    if model_config.scale <= 0:
        raise ValueError("scale must be positive")
    if model_config.img_size <= 0:
        raise ValueError("img_size must be positive")


class Config(Generic[M]):
    """Classe configurazione generica"""

    def __init__(
        self, config_path: str, model_class: Type[M], validator: Type[ConfigValidator]
    ):
        self.config_path = Path(config_path).resolve()
        with self.config_path.open("r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        if not isinstance(config_dict, dict):
            raise ValueError("configuration must be a mapping")

        self.model: M = model_class(**config_dict["model"])
        self.train = TrainConfig(**config_dict["train"])
        self.test = TestConfig(**config_dict["test"])

        # Validazione
        validator.validate_model(self.model)

        # Attach validated per-channel normalization stats used by DRCT.
        stats_file = self._resolve_stats_file(self.model.stats_file)
        if not stats_file.is_file():
            raise FileNotFoundError(
                f"Channel stats file not found at {stats_file}. Please run calc_mean.py first."
            )

        with stats_file.open("r", encoding="utf-8") as f:
            stats = json.load(f)
        mean = stats.get("mean")
        if mean is None:
            raise ValueError("mean not found in stats")
        std = stats.get("std")
        if std is None:
            raise ValueError("std not found in stats")

        self._validate_channel_stats(mean, std)
        self._validate_stats_provenance(stats, self.test.comune)
        setattr(self.model, "mean", [float(value) for value in mean])
        setattr(self.model, "std", [float(value) for value in std])

        print(f"Loaded channel stats from {stats_file}:\n mean={mean}\n std={std}\n")

    def _resolve_stats_file(self, stats_file: str | None) -> Path:
        if stats_file is None:
            return Path(__file__).with_name("channel_stats.json")
        path = Path(stats_file)
        return path if path.is_absolute() else self.config_path.parent / path

    @staticmethod
    def _validate_channel_stats(mean, std) -> None:
        if not isinstance(mean, list) or not isinstance(std, list):
            raise ValueError("mean and std must be lists")
        if len(mean) != 4 or len(std) != 4:
            raise ValueError("mean and std must each contain four channel values")
        try:
            numeric_mean = [float(value) for value in mean]
            numeric_std = [float(value) for value in std]
        except (TypeError, ValueError) as exc:
            raise ValueError("mean and std must contain numeric values") from exc
        if not all(value == value and abs(value) != float("inf") for value in numeric_mean):
            raise ValueError("mean must contain finite values")
        if not all(value == value and value != float("inf") and value != float("-inf") and value > 0 for value in numeric_std):
            raise ValueError("std must contain finite positive values")

    @staticmethod
    def _validate_stats_provenance(stats: dict, held_out_comune: ComuneType) -> None:
        details = stats.get("details")
        if details is None:
            return
        if not isinstance(details, dict):
            raise ValueError("stats details must be a mapping when present")
        training_comuni = details.get("training_comuni")
        if training_comuni is None:
            return
        if not isinstance(training_comuni, list) or not all(
            isinstance(comune, str) for comune in training_comuni
        ):
            raise ValueError("stats details.training_comuni must be a list of names")
        if held_out_comune in training_comuni:
            raise ValueError(
                "channel stats include the held-out test comune "
                f"'{held_out_comune}'"
            )

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
