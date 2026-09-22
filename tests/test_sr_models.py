import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F
import yaml

from Super_Resolution.config import load_config
from Super_Resolution.calc_mean import build_parser
from Super_Resolution.myDRCT.DRCT import DRCT
from Super_Resolution.models_functions import Upsample
from Super_Resolution.rcan.rcan_model import RCAN
from Super_Resolution.swin2mose.swin2mose_model import Swin2MoSE


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("model_name", "model_class", "config_relpath"),
    [
        ("rcan", RCAN, "Super_Resolution/rcan/config.yml"),
        ("swin2mose", Swin2MoSE, "Super_Resolution/swin2mose/config.yml"),
        ("drct", DRCT, "Super_Resolution/myDRCT/config.yml"),
    ],
)
def test_sr_models_support_four_channels_and_exact_five_x(
    tmp_path, model_name, model_class, config_relpath
):
    """Exercise real model imports and a backward pass without raster data."""
    with (ROOT / config_relpath).open(encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    # These are deliberately test-only inputs, not Sentinel normalization values.
    stats_path = tmp_path / "identity_stats.json"
    stats_path.write_text(json.dumps({"mean": [0.0] * 4, "std": [1.0] * 4}))
    raw_config["model"].update({"img_size": 8, "stats_file": str(stats_path)})
    if "window_size" in raw_config["model"]:
        raw_config["model"]["window_size"] = 4
    config_path = tmp_path / f"{model_name}.yml"
    config_path.write_text(yaml.safe_dump(raw_config))

    config = load_config(model_name, str(config_path))
    model = model_class(config)
    source = torch.randn(1, 4, 8, 8, requires_grad=True)
    output = model(source)
    if isinstance(output, tuple):
        output = output[0]

    assert output.shape == (1, 4, 40, 40)
    output.square().mean().backward()
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()


def test_swin2mose_rejects_an_unsupported_scale(tmp_path):
    with (ROOT / "Super_Resolution/swin2mose/config.yml").open(encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps({"mean": [0.0] * 4, "std": [1.0] * 4}))
    raw_config["model"].update({"scale": 4, "stats_file": str(stats_path)})
    config_path = tmp_path / "invalid.yml"
    config_path.write_text(yaml.safe_dump(raw_config))

    with pytest.raises(ValueError, match="scale=5"):
        load_config("swin2mose", str(config_path))


@pytest.mark.parametrize("height,width", [(7, 9), (8, 8)])
def test_rcan_upsample_is_exact_five_x_for_odd_and_non_square_inputs(height, width):
    upsample = Upsample(4)
    source = torch.randn(1, 4, height, width)
    output = upsample(source)

    assert output.shape == (1, 4, height * 5, width * 5)


def test_rcan_upsample_preserves_the_divisible_by_four_path():
    upsample = Upsample(4)
    source = torch.randn(1, 4, 8, 8)

    output = upsample(source)
    expected = upsample.conv_pre(source)
    expected = F.interpolate(
        expected, scale_factor=5 / 4, mode="bicubic", align_corners=False
    )
    expected = upsample.lrelu(upsample.conv_after_1(expected))
    expected = F.interpolate(expected, scale_factor=2, mode="nearest")
    expected = upsample.lrelu(upsample.conv_after_2(expected))
    expected = F.interpolate(expected, scale_factor=2, mode="nearest")
    expected = upsample.lrelu(upsample.conv_hr(expected))
    expected = upsample.conv_last(expected)

    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_config_resolves_relative_stats_path(tmp_path):
    with (ROOT / "Super_Resolution/rcan/config.yml").open(encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "mean": [0.1] * 4,
                "std": [0.5] * 4,
                "details": {
                    "training_comuni": ["Casola-Valsenio", "Modigliana", "Predappio"]
                },
            }
        )
    )
    raw_config["model"]["stats_file"] = "stats.json"
    config_path = tmp_path / "relative.yml"
    config_path.write_text(yaml.safe_dump(raw_config))

    config = load_config("rcan", str(config_path))
    assert config.model.mean == [0.1] * 4
    assert config.model.std == [0.5] * 4


@pytest.mark.parametrize(
    ("details", "message"),
    [
        ({"training_comuni": ["Brisighella"]}, "held-out test comune"),
        ({"training_comuni": "Brisighella"}, "must be a list"),
    ],
)
def test_config_rejects_invalid_stats_provenance(tmp_path, details, message):
    with (ROOT / "Super_Resolution/rcan/config.yml").open(encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(
        json.dumps({"mean": [0.0] * 4, "std": [1.0] * 4, "details": details})
    )
    raw_config["model"]["stats_file"] = str(stats_path)
    config_path = tmp_path / "invalid-provenance.yml"
    config_path.write_text(yaml.safe_dump(raw_config))

    with pytest.raises(ValueError, match=message):
        load_config("rcan", str(config_path))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"mean": [0.0] * 4, "std": [1.0, 1.0, 1.0, 0.0]}, "positive"),
        ({"mean": [float("inf")] * 4, "std": [1.0] * 4}, "finite"),
    ],
)
def test_config_rejects_invalid_channel_stats(tmp_path, payload, message):
    with (ROOT / "Super_Resolution/rcan/config.yml").open(encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(payload))
    raw_config["model"]["stats_file"] = str(stats_path)
    config_path = tmp_path / "invalid-stats.yml"
    config_path.write_text(yaml.safe_dump(raw_config))

    with pytest.raises(ValueError, match=message):
        load_config("rcan", str(config_path))


def test_calc_mean_synthetic_data_flag_is_boolean():
    parser = build_parser()
    assert parser.parse_args([]).synthetic_data is False
    assert parser.parse_args(["--synthetic_data"]).synthetic_data is True


def test_swin2mose_uses_configured_mlp_ratio_and_allows_an_override(tmp_path):
    with (ROOT / "Super_Resolution/swin2mose/config.yml").open(encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps({"mean": [0.0] * 4, "std": [1.0] * 4}))
    raw_config["model"].update(
        {"img_size": 8, "window_size": 4, "stats_file": str(stats_path)}
    )
    config_path = tmp_path / "swin.yml"
    config_path.write_text(yaml.safe_dump(raw_config))
    config = load_config("swin2mose", str(config_path))

    from_config = Swin2MoSE(config)
    overridden = Swin2MoSE(config, mlp_ratio=3.0)
    assert from_config.layers[0].layers[0].moe.hidden_size == 16
    assert overridden.layers[0].layers[0].moe.hidden_size == 24


def test_drct_pads_and_crops_an_odd_non_square_input(tmp_path):
    with (ROOT / "Super_Resolution/myDRCT/config.yml").open(encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps({"mean": [0.0] * 4, "std": [1.0] * 4}))
    raw_config["model"].update(
        {"img_size": 8, "window_size": 4, "stats_file": str(stats_path)}
    )
    config_path = tmp_path / "drct.yml"
    config_path.write_text(yaml.safe_dump(raw_config))
    model = DRCT(load_config("drct", str(config_path))).eval()

    source = torch.randn(1, 4, 7, 9)
    assert model(source).shape == (1, 4, 35, 45)


def test_drct_preserves_the_divisible_window_path(tmp_path):
    with (ROOT / "Super_Resolution/myDRCT/config.yml").open(encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps({"mean": [0.0] * 4, "std": [1.0] * 4}))
    raw_config["model"].update(
        {"img_size": 8, "window_size": 4, "stats_file": str(stats_path)}
    )
    config_path = tmp_path / "drct.yml"
    config_path.write_text(yaml.safe_dump(raw_config))
    model = DRCT(load_config("drct", str(config_path))).eval()
    source = torch.randn(1, 4, 8, 8)

    output = model(source)
    normalized = (source - model.mean.type_as(source)) / model.std.type_as(source)
    features = model.conv_first(normalized)
    expected = model.conv_after_body(model.forward_features(features)) + features
    expected = model.tail(expected)
    expected = expected * model.std.type_as(expected) + model.mean.type_as(expected)

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
