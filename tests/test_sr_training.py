import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from Super_Resolution import models_utils
from Super_Resolution.config import load_config
from Super_Resolution.rcan.rcan_model import RCAN


def _training_config(*, epochs=1, accumulation_steps=1):
    return SimpleNamespace(
        train=SimpleNamespace(
            accumulation_steps=accumulation_steps,
            epochs=epochs,
            loss_weights={"charb": 1.0},
            show_progress=False,
        )
    )


class _RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, x):
        self.calls.append((float(x[0, 0, 0, 0]), self.training))
        return x + self.bias


class _NoOpScheduler:
    instances = []

    def __init__(self, optimizer, *args, **kwargs):
        self.optimizer = optimizer
        self.step_calls = 0
        self.__class__.instances.append(self)

    def step(self):
        self.step_calls += 1


class _CountingSGD(torch.optim.SGD):
    instances = []

    def __init__(self, params, lr, **kwargs):
        super().__init__(params, lr=lr)
        self.step_calls = 0
        self.__class__.instances.append(self)

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


class _ZeroMetric:
    def __init__(self, *args, **kwargs):
        pass

    def to(self, device):
        return self

    def __call__(self, *args, **kwargs):
        return torch.zeros(())

    def reset(self):
        pass


class SuperResolutionTrainingTests(unittest.TestCase):
    def test_composite_loss_applies_charbonnier_weight_once(self):
        config = _training_config()
        config.train.loss_weights["charb"] = 0.25
        sr = torch.zeros(1, 4, 4, 4)
        hr = torch.ones_like(sr)

        total, components = models_utils.composite_loss(sr, hr, config)

        rgb = models_utils.charbonnier_loss(sr[:, :3], hr[:, :3])
        nir = models_utils.charbonnier_loss(sr[:, 3:], hr[:, 3:])
        expected = 0.25 * (rgb + 0.5 * nir)
        self.assertEqual(set(components), {"charb"})
        torch.testing.assert_close(components["charb"], expected)
        torch.testing.assert_close(total, expected)

    def test_composite_loss_rejects_an_empty_objective(self):
        config = _training_config()
        config.train.loss_weights = {"charb": 0.0}
        image = torch.zeros(1, 4, 4, 4)

        with self.assertRaisesRegex(ValueError, "At least one loss weight"):
            models_utils.composite_loss(image, image, config)

    def test_training_mode_is_restored_after_each_validation(self):
        model = _RecordingModel()
        train_low = torch.full((3, 4, 2, 2), 10.0)
        train_high = train_low.clone()
        eval_low = torch.full((1, 4, 2, 2), 20.0)
        eval_high = eval_low.clone()
        train_loader = DataLoader(TensorDataset(train_low, train_high), batch_size=2)
        eval_loader = DataLoader(TensorDataset(eval_low, eval_high), batch_size=1)
        config = _training_config(epochs=2)

        models_utils.train_model(
            model, train_loader, eval_loader, torch.device("cpu"), config
        )

        train_modes = [mode for marker, mode in model.calls if marker == 10.0]
        eval_modes = [mode for marker, mode in model.calls if marker == 20.0]
        self.assertEqual(train_modes, [True, True, True, True])
        self.assertEqual(eval_modes, [False, False])

    def test_single_update_schedule_does_not_decay_at_step_zero(self):
        model = _RecordingModel()
        image = torch.zeros(1, 4, 2, 2)
        loader = DataLoader(TensorDataset(image, image), batch_size=1)

        tracking = models_utils.train_model(
            model,
            loader,
            loader,
            torch.device("cpu"),
            _training_config(),
        )

        self.assertEqual(tracking["lr"], [2e-4])

    def test_remainder_accumulation_has_full_gradient_and_update(self):
        model = _RecordingModel()
        low = torch.tensor([0.0, 0.0, 1.0]).view(3, 1, 1, 1).expand(-1, 4, 2, 2)
        high = torch.zeros_like(low)
        train_loader = DataLoader(TensorDataset(low, high), batch_size=1)
        eval_loader = DataLoader(TensorDataset(high[:1], high[:1]), batch_size=1)
        config = _training_config(accumulation_steps=2)
        _CountingSGD.instances.clear()
        _NoOpScheduler.instances.clear()

        def output_mean_loss(outputs, hr, config, moe_loss=0):
            value = outputs.mean()
            return value, {"charb": value}

        with (
            mock.patch.object(models_utils, "composite_loss", output_mean_loss),
            mock.patch.object(models_utils.optim, "AdamW", _CountingSGD),
            mock.patch.object(
                models_utils.optim.lr_scheduler, "MultiStepLR", _NoOpScheduler
            ),
            mock.patch.object(torch.nn.utils, "clip_grad_norm_", return_value=None),
        ):
            models_utils.train_model(
                model, train_loader, eval_loader, torch.device("cpu"), config
            )

        self.assertEqual(_CountingSGD.instances[0].step_calls, 2)
        self.assertEqual(_NoOpScheduler.instances[0].step_calls, 2)
        # The first two batches contribute one full averaged gradient, and the
        # one-batch remainder contributes another full gradient.
        self.assertTrue(math.isclose(model.bias.item(), -4e-4, abs_tol=1e-8))

    def test_epoch_loss_is_weighted_by_sample_count(self):
        model = _RecordingModel()
        low = torch.zeros(3, 4, 2, 2)
        high = torch.tensor([1.0, 1.0, 3.0]).view(3, 1, 1, 1).expand_as(low)
        train_loader = DataLoader(TensorDataset(low, high), batch_size=2)
        eval_loader = DataLoader(TensorDataset(low[:1], low[:1]), batch_size=1)
        config = _training_config()

        def target_marker_loss(outputs, hr, config, moe_loss=0):
            value = hr[:, 0, 0, 0].mean() + outputs.sum() * 0
            return value, {"charb": value}

        with mock.patch.object(models_utils, "composite_loss", target_marker_loss):
            tracking = models_utils.train_model(
                model, train_loader, eval_loader, torch.device("cpu"), config
            )

        self.assertTrue(math.isclose(tracking["total"][0], 5 / 3, rel_tol=1e-6))
        self.assertTrue(math.isclose(tracking["charb"][0], 5 / 3, rel_tol=1e-6))

    def test_accumulated_gradient_is_weighted_by_sample_count(self):
        class ScaleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(()))

            def forward(self, x):
                return x * self.scale

        model = ScaleModel()
        low = torch.tensor([0.0, 0.0, 3.0]).view(3, 1, 1, 1).expand(-1, 4, 2, 2)
        high = torch.zeros_like(low)
        train_loader = DataLoader(TensorDataset(low, high), batch_size=2)
        eval_loader = DataLoader(TensorDataset(high[:1], high[:1]), batch_size=1)
        config = _training_config(accumulation_steps=2)
        _CountingSGD.instances.clear()

        def output_mean_loss(outputs, hr, config, moe_loss=0):
            value = outputs.mean()
            return value, {"charb": value}

        with (
            mock.patch.object(models_utils, "composite_loss", output_mean_loss),
            mock.patch.object(models_utils.optim, "AdamW", _CountingSGD),
            mock.patch.object(
                models_utils.optim.lr_scheduler, "MultiStepLR", _NoOpScheduler
            ),
            mock.patch.object(torch.nn.utils, "clip_grad_norm_", return_value=None),
        ):
            models_utils.train_model(
                model, train_loader, eval_loader, torch.device("cpu"), config
            )

        # Mean gradient over the three samples is (0 + 0 + 3) / 3 = 1.
        self.assertTrue(math.isclose(model.scale.item(), 1 - 2e-4, abs_tol=5e-8))

    def test_evaluation_psnr_is_averaged_per_sample(self):
        class UpscaleZeros(nn.Module):
            def forward(self, x):
                return torch.nn.functional.interpolate(x, scale_factor=5)

        low = torch.zeros(3, 4, 40, 40)
        high = torch.cat(
            (
                torch.ones(2, 4, 200, 200),
                torch.full((1, 4, 200, 200), 0.1),
            )
        )
        loader = DataLoader(TensorDataset(low, high), batch_size=2)

        with (
            mock.patch.object(
                models_utils, "LearnedPerceptualImagePatchSimilarity", _ZeroMetric
            ),
            mock.patch.object(models_utils, "ms_ssim", return_value=torch.zeros(())),
        ):
            metrics = models_utils.evaluate_model(
                UpscaleZeros(), loader, torch.device("cpu")
            )

        # The first two samples have PSNR 0 dB and the final sample 20 dB.
        self.assertTrue(math.isclose(metrics["psnr"], 20 / 3, rel_tol=1e-5))

    def test_last_validation_psnr_matches_final_evaluation(self):
        class OutOfRangeUpscaler(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.tensor(2.0))

            def forward(self, x):
                return torch.nn.functional.interpolate(x, scale_factor=5) + self.bias

        model = OutOfRangeUpscaler()
        low = torch.zeros(2, 4, 40, 40)
        high = torch.full((2, 4, 200, 200), 0.5)
        loader = DataLoader(TensorDataset(low, high), batch_size=2)
        tracking = models_utils.train_model(
            model, loader, loader, torch.device("cpu"), _training_config()
        )

        with (
            mock.patch.object(
                models_utils, "LearnedPerceptualImagePatchSimilarity", _ZeroMetric
            ),
            mock.patch.object(models_utils, "ms_ssim", return_value=torch.zeros(())),
        ):
            metrics = models_utils.evaluate_model(model, loader, torch.device("cpu"))

        self.assertTrue(
            math.isclose(tracking["psnr_total"][-1], metrics["psnr"], rel_tol=1e-6)
        )

    def test_rcan_head_only_finetuning_enables_upsampling_tail(self):
        config = SimpleNamespace(
            model=SimpleNamespace(
                feature_extraction_channels=4,
                reduction_channels=2,
                residual_groups=1,
            )
        )
        model = RCAN(config)

        trainable = models_utils._prepare_finetune_head_only(model)

        self.assertTrue(trainable)
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, name.startswith("upsample."), name)

    def test_example_models_support_head_only_backward(self):
        repository = Path(__file__).resolve().parents[1]
        examples = (
            ("rcan", repository / "Super_Resolution/rcan/config.yml", models_utils.RCAN),
            (
                "swin2mose",
                repository / "Super_Resolution/swin2mose/config.yml",
                models_utils.Swin2MoSE,
            ),
            ("drct", repository / "Super_Resolution/myDRCT/config.yml", models_utils.DRCT),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            stats_path = tmp_path / "channel_stats.json"
            stats_path.write_text(
                json.dumps({"mean": [0.0] * 4, "std": [1.0] * 4}),
                encoding="utf-8",
            )

            for model_name, source_path, model_class in examples:
                with self.subTest(model=model_name):
                    raw_config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
                    raw_config["model"]["stats_file"] = str(stats_path)
                    raw_config["model"]["dir_path"] = str(tmp_path / "outputs")
                    config_path = tmp_path / f"{model_name}.yml"
                    config_path.write_text(
                        yaml.safe_dump(raw_config), encoding="utf-8"
                    )
                    config = load_config(model_name, str(config_path))
                    model = model_class(config)

                    trainable = models_utils._prepare_finetune_head_only(model)
                    output = model(torch.rand(1, 4, 40, 40))
                    if isinstance(output, tuple):
                        output = output[0]
                    output.mean().backward()

                    self.assertEqual(tuple(output.shape), (1, 4, 200, 200))
                    self.assertTrue(trainable)
                    self.assertTrue(all(p.grad is not None for p in trainable))

    def test_missing_finetune_checkpoint_fails_before_training(self):
        model = nn.Conv2d(4, 4, 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.pth"
            with self.assertRaisesRegex(FileNotFoundError, "checkpoint not found"):
                models_utils._load_finetune_checkpoint(
                    model, str(missing), torch.device("cpu")
                )

    def test_incompatible_finetune_checkpoint_cannot_train_random_weights(self):
        model = nn.Conv2d(4, 4, 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "incompatible.pth"
            torch.save({"unrelated.weight": torch.ones(1)}, checkpoint)

            with self.assertRaisesRegex(ValueError, "no parameters matching"):
                models_utils._load_finetune_checkpoint(
                    model, str(checkpoint), torch.device("cpu")
                )

    def test_partial_finetune_checkpoint_cannot_freeze_random_backbone(self):
        model = nn.Conv2d(4, 4, 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "partial.pth"
            torch.save({"weight": model.weight.detach().clone()}, checkpoint)

            with self.assertRaisesRegex(ValueError, "not fully compatible"):
                models_utils._load_finetune_checkpoint(
                    model, str(checkpoint), torch.device("cpu")
                )

    def test_model_path_does_not_require_a_trailing_separator(self):
        model = nn.Conv2d(4, 4, 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SimpleNamespace(
                model=SimpleNamespace(dir_path=tmpdir, name="rcan_test")
            )
            output_dir = Path(tmpdir) / "rcan_test"
            output_dir.mkdir()
            expected_weight = model.weight.detach().clone()

            models_utils.save_model(model, config)
            model.weight.data.zero_()
            models_utils.load_model(config, model, torch.device("cpu"))

            self.assertTrue((output_dir / "model.pth").is_file())
            torch.testing.assert_close(model.weight, expected_weight)

    def test_two_scale_ms_ssim_accepts_production_patch_size(self):
        image = torch.rand(1, 3, 200, 200)
        score = models_utils.ms_ssim(
            image, image, data_range=1.0, weights=[0.5, 0.5]
        )
        torch.testing.assert_close(score, torch.ones_like(score), atol=1e-5, rtol=0)

    def test_perfect_psnr_is_preserved_in_standards_compliant_json(self):
        image = torch.zeros(1, 4, 4, 4)
        perfect_psnr = models_utils._psnr_sum(image, image).item()
        self.assertEqual(perfect_psnr, float("inf"))
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SimpleNamespace(model=SimpleNamespace(dir_path=tmpdir, name="run"))
            output = Path(tmpdir) / "run"
            output.mkdir()
            with mock.patch.object(models_utils, "_config_to_dict", return_value={}):
                models_utils.save_metrics(
                    {"psnr": perfect_psnr, "curves": [float("nan"), float("-inf")]},
                    config,
                )

            def reject_non_json_constant(value):
                raise AssertionError(f"Non-standard JSON constant: {value}")

            saved = json.loads(
                (output / "metrics.json").read_text(),
                parse_constant=reject_non_json_constant,
            )["metrics"]
            self.assertEqual(saved["psnr"], "Infinity")
            self.assertEqual(saved["curves"], ["NaN", "-Infinity"])


if __name__ == "__main__":
    unittest.main()
