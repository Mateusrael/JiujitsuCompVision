"""Image training modes, including real CPU updates and checkpoint restoration."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from src.Scripts.train import build_parser
from src.Train.config import resolve_settings


class ImageTrainingConfigurationTests(unittest.TestCase):
    def settings(self, *flags, previous=None):
        return resolve_settings(build_parser().parse_args(["--model", "image", *flags]), previous)

    def test_default_explicit_modes_and_legacy_aliases(self):
        cases = (([], "head"), (["--no-fine-tune"], "head"),
                 (["--fine-tune"], "full"),
                 (["--image-training", "head"], "head"),
                 (["--image-training", "input-head"], "input-head"),
                 (["--image-training", "layer4"], "layer4"),
                 (["--image-training", "full"], "full"))
        for flags, mode in cases:
            with self.subTest(flags=flags):
                settings = self.settings(*flags)
                self.assertEqual(settings["image_training"], mode)
                self.assertEqual(settings["fine_tune"], mode != "head")

    def test_cli_does_not_accept_ambiguous_modes(self):
        parser = build_parser()
        for mode in ("head", "input-head", "layer4", "full"):
            for alias in ("--fine-tune", "--no-fine-tune"):
                with self.subTest(mode=mode, alias=alias):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        parser.parse_args(["--model", "image", "--image-training", mode, alias])

    def test_pose_models_reject_image_training_even_for_head_mode(self):
        for model in ("pose", "pose-wide", "pose-attention"):
            with self.subTest(model=model):
                args = build_parser().parse_args(["--model", model, "--image-training", "head"])
                with self.assertRaisesRegex(ValueError, "image"):
                    resolve_settings(args)

    def test_resume_preserves_mode_and_rejects_changes_including_aliases(self):
        for mode in ("head", "input-head", "layer4", "full"):
            previous = self.settings("--image-training", mode)
            with self.subTest(saved=mode):
                self.assertEqual(self.settings(previous=previous)["image_training"], mode)
                self.assertEqual(self.settings("--image-training", mode,
                                               previous=previous)["image_training"], mode)
            for requested in ("head", "input-head", "layer4", "full"):
                if requested == mode:
                    continue
                with self.subTest(saved=mode, requested=requested):
                    with self.assertRaisesRegex(ValueError, "Cannot change"):
                        self.settings("--image-training", requested, previous=previous)
            for alias, effective in (("--fine-tune", "full"), ("--no-fine-tune", "head")):
                with self.subTest(saved=mode, alias=alias):
                    if effective == mode:
                        self.assertEqual(self.settings(alias, previous=previous)["image_training"], mode)
                    else:
                        with self.assertRaisesRegex(ValueError, "Cannot change"):
                            self.settings(alias, previous=previous)

    def test_legacy_run_settings_infer_head_or_full_and_cannot_become_partial(self):
        for fine_tune, mode in ((False, "head"), (True, "full")):
            previous = self.settings("--fine-tune" if fine_tune else "--no-fine-tune")
            previous.pop("image_training")
            with self.subTest(fine_tune=fine_tune):
                restored = self.settings(previous=previous)
                self.assertEqual(restored["image_training"], mode)
                self.assertEqual(restored["fine_tune"], fine_tune)
                self.assertEqual(self.settings("--image-training", mode,
                                               previous=previous)["image_training"], mode)
                with self.assertRaisesRegex(ValueError, "Cannot change"):
                    self.settings("--image-training", "layer4", previous=previous)

    def test_metrics_summary_retains_explicit_mode_without_guessing_legacy_modes(self):
        from src.Scripts.export_metrics import summarize
        from tests.test_export_metrics import config_bytes, metric, metrics_bytes
        for mode in ("head", "input-head", "layer4", "full", None):
            with self.subTest(mode=mode):
                settings = {} if mode is None else {"image_training": mode}
                files = {"config_start.json": config_bytes("image", fine_tune=True, **settings),
                         "diagnostics/metrics.jsonl": metrics_bytes(metric(1))}
                summary = summarize("example-image-run", files, [])
                self.assertEqual(summary["image_training"], mode)


@unittest.skipUnless(os.environ.get("JIUJITSU_RUN_TORCH_TESTS") == "1",
                     "Set JIUJITSU_RUN_TORCH_TESTS=1 for CPU image model checks")
class ImageTrainingModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from src.Modules.models import ImageClassifier, architecture_config, build_model
        cls.torch = torch
        cls.ImageClassifier = ImageClassifier
        cls.architecture_config = staticmethod(architecture_config)
        cls.build_model = staticmethod(build_model)
        torch.set_num_threads(1)

    def setUp(self):
        self.torch.manual_seed(27)

    @staticmethod
    def is_trainable(name, mode):
        return (name.startswith("classifier.") or mode == "full"
                or (mode == "layer4" and name.startswith("backbone.layer4."))
                or (mode == "input-head" and name.startswith("backbone.conv1.")))

    def test_optimizer_changes_only_selected_weights_and_batchnorm_buffers(self):
        torch = self.torch
        for mode in ("head", "input-head", "layer4", "full"):
            with self.subTest(mode=mode):
                model = self.ImageClassifier(pretrained=False, image_training=mode, dropout=0.2)
                model.train()
                before = {name: value.clone() for name, value in model.state_dict().items()}
                parameters = dict(model.named_parameters())
                for name, parameter in parameters.items():
                    self.assertEqual(parameter.requires_grad, self.is_trainable(name, mode), name)
                if mode == "input-head":
                    self.assertEqual(sum(p.numel() for p in parameters.values() if p.requires_grad), 14538)
                optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                              lr=0.001, weight_decay=0.01)
                features = torch.randn(4, 3, 64, 64)
                labels = torch.tensor([0, 3, 6, 9])
                loss = torch.nn.functional.cross_entropy(model(features), labels)
                loss.backward()
                for name, parameter in parameters.items():
                    if self.is_trainable(name, mode):
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    else:
                        self.assertIsNone(parameter.grad, name)
                if mode == "input-head":
                    self.assertGreater(model.backbone.conv1.weight.grad.abs().sum().item(), 0)
                optimizer.step()
                for name, value in model.state_dict().items():
                    if not self.is_trainable(name, mode):
                        self.assertTrue(torch.equal(before[name], value), name)
                self.assertFalse(torch.equal(before["classifier.weight"], model.classifier.weight))
                if mode in ("layer4", "full"):
                    self.assertFalse(torch.equal(before["backbone.layer4.0.conv1.weight"],
                                                 model.backbone.layer4[0].conv1.weight))
                if mode in ("input-head", "full"):
                    self.assertFalse(torch.equal(before["backbone.conv1.weight"], model.backbone.conv1.weight))
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.BatchNorm2d):
                        expected = 1 if self.is_trainable(name + ".weight", mode) else 0
                        self.assertEqual(module.num_batches_tracked.item(), expected, name)

    def test_repeated_train_eval_keeps_frozen_batchnorm_fixed_and_dropout_active(self):
        torch = self.torch
        for mode in ("head", "input-head", "layer4", "full"):
            with self.subTest(mode=mode):
                model = self.ImageClassifier(pretrained=False, image_training=mode, dropout=0.4)
                for _ in range(3):
                    self.assertIs(model.eval(), model)
                    self.assertTrue(all(not module.training for module in model.modules()))
                    self.assertIs(model.train(), model)
                    self.assertTrue(model.classifier.training)
                    self.assertTrue(model.dropout.training)
                    for name, module in model.named_modules():
                        if isinstance(module, torch.nn.BatchNorm2d):
                            self.assertEqual(module.training, self.is_trainable(name + ".weight", mode), name)

    def test_explicit_configuration_and_state_round_trip_preserves_mode(self):
        torch = self.torch
        for mode in ("head", "input-head", "layer4", "full"):
            with self.subTest(mode=mode):
                config = self.architecture_config("image", image_training=mode, dropout=0.2)
                self.assertEqual(config["image_training"], mode)
                self.assertEqual(config["fine_tune"], mode != "head")
                model = self.build_model(config, pretrained=False).eval()
                restored = self.build_model(json.loads(json.dumps(config)), pretrained=False).eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                self.assertEqual(restored.image_training, mode)
                self.assertEqual([name for name, p in restored.named_parameters() if p.requires_grad],
                                 [name for name, p in model.named_parameters() if p.requires_grad])
                features = torch.randn(2, 3, 32, 32)
                with torch.inference_mode():
                    torch.testing.assert_close(model(features), restored(features), rtol=0, atol=0)

    def test_legacy_model_definitions_still_restore_the_original_modes(self):
        for fine_tune, mode in ((False, "head"), (True, "full")):
            with self.subTest(fine_tune=fine_tune):
                legacy = {"type": "resnet18", "num_classes": 10, "fine_tune": fine_tune,
                          "image_size": 224, "image_preprocessing": "letterbox_imagenet_v1",
                          "dropout": 0.0}
                self.assertEqual(self.architecture_config("image", fine_tune=fine_tune), legacy)
                restored = self.build_model(legacy, pretrained=False)
                self.assertEqual(restored.image_training, mode)
                for name, parameter in restored.named_parameters():
                    self.assertEqual(parameter.requires_grad, self.is_trainable(name, mode), name)

    def test_invalid_or_inconsistent_image_definitions_are_rejected(self):
        with self.assertRaises(ValueError):
            self.architecture_config("image", image_training="layer3")
        config = self.architecture_config("image", image_training="layer4")
        config["fine_tune"] = False
        with self.assertRaises(ValueError):
            self.build_model(config, pretrained=False)

    def test_whole_model_compile_preserves_partial_training_and_eval_modes(self):
        from unittest.mock import patch
        from src.Modules.execution import compile_model
        torch = self.torch
        original_compile = torch.compile

        def cpu_graph_capture(model, **options):
            options.pop("mode", None)
            return original_compile(model, backend="eager", **options)

        model = self.ImageClassifier(pretrained=False, image_training="layer4").eval()
        frozen = {name: value.clone() for name, value in model.state_dict().items()
                  if not self.is_trainable(name, "layer4")}
        with patch("torch.compile", side_effect=cpu_graph_capture) as compiler:
            compiled = compile_model(model, enabled=True)
        compiler.assert_called_once_with(model, mode="default")
        features, labels = torch.randn(2, 3, 64, 64), torch.tensor([0, 3])
        for epoch in (1, 2):
            compiled.train()
            self.assertFalse(model.backbone.bn1.training)
            self.assertTrue(model.backbone.layer4[0].bn1.training)
            model.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(compiled(features), labels).backward()
            self.assertIsNone(model.backbone.conv1.weight.grad)
            self.assertGreater(model.backbone.layer4[0].conv1.weight.grad.abs().sum().item(), 0)
            self.assertEqual(model.backbone.layer4[0].bn1.num_batches_tracked.item(), epoch)
            compiled.eval()
            with torch.inference_mode():
                compiled(features)
            self.assertEqual(model.backbone.layer4[0].bn1.num_batches_tracked.item(), epoch)
            for name, value in frozen.items():
                self.assertTrue(torch.equal(model.state_dict()[name], value), name)

    def test_compiled_input_head_reaches_first_layer_through_frozen_stages(self):
        torch = self.torch
        model = self.ImageClassifier(pretrained=False, image_training="input-head", dropout=0.2)
        compiled = torch.compile(model, backend="eager")
        inputs = torch.randn(2, 3, 64, 64)
        for _ in range(2):
            compiled.train()
            model.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(compiled(inputs), torch.tensor([0, 3])).backward()
            self.assertGreater(model.backbone.conv1.weight.grad.abs().sum().item(), 0)
            self.assertGreater(model.classifier.weight.grad.abs().sum().item(), 0)
            for name, parameter in model.backbone.named_parameters():
                if not name.startswith("conv1."):
                    self.assertIsNone(parameter.grad, name)
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    self.assertFalse(module.training)
                    self.assertEqual(module.num_batches_tracked.item(), 0)
            compiled.eval()
            with torch.inference_mode():
                compiled(inputs)

    def test_layer4_train_resume_evaluate_matches_continuous_training(self):
        self._check_train_resume_evaluate("layer4")

    def test_input_head_train_resume_evaluate_matches_continuous_training(self):
        self._check_train_resume_evaluate("input-head")

    def _check_train_resume_evaluate(self, mode):
        from tests.test_training import synthetic_files
        from src.Eval.evaluate import evaluate
        from src.Scripts.evaluate import build_parser as evaluation_parser
        from src.Train.checkpoints import load_checkpoint
        from src.Train.engine import train
        torch = self.torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations, split, images, _ = synthetic_files(root, images=True)
            common = ["--model", "image", "--image-training", mode, "--no-pretrained",
                      "--annotations", str(annotations), "--split", str(split),
                      "--images-dir", str(images), "--device", "cpu", "--workers", "0",
                      "--batch-size", "10", "--dropout", "0.2", "--horizontal-flip-prob", "0.5",
                      "--weight-decay", "0.001", "--no-progress"]
            run = root / "partial"
            train(build_parser().parse_args(common + ["--run-dir", str(run), "--epochs", "1"]))
            last = run / "checkpoints" / "last.pt"
            first = load_checkpoint(last)
            self.assertEqual(first["model_state"]["backbone.bn1.num_batches_tracked"].item(), 0)
            self.assertEqual(first["model_state"]["backbone.layer4.0.bn1.num_batches_tracked"].item(),
                             1 if mode == "layer4" else 0)
            train(build_parser().parse_args(["--resume", str(last), "--epochs", "2"]))
            second = load_checkpoint(last)
            self.assertEqual(second["epoch"], 2)
            self.assertEqual(second["config"]["model_config"], first["config"]["model_config"])
            self.assertEqual(second["config"]["image_training"], mode)
            self.assertEqual(second["config"]["model_config"]["image_training"], mode)
            self.assertEqual(second["model_state"]["backbone.layer4.0.bn1.num_batches_tracked"].item(),
                             2 if mode == "layer4" else 0)
            for name, value in first["model_state"].items():
                if not self.is_trainable(name, mode):
                    self.assertTrue(torch.equal(value, second["model_state"][name]), name)
            for requested in ("head", "input-head", "layer4", "full"):
                if requested == mode:
                    continue
                with self.assertRaisesRegex(ValueError, "Cannot change"):
                    train(build_parser().parse_args(["--resume", str(last), "--epochs", "3",
                                                     "--image-training", requested]))
            result = evaluate(evaluation_parser().parse_args([
                "--checkpoint", str(run / "checkpoints" / "best.pt"), "--device", "cpu", "--no-progress"]))
            self.assertEqual(result["metrics"]["samples"], 10)
            self.assertEqual(result["model_config"], first["config"]["model_config"])
            continuous = root / "continuous"
            train(build_parser().parse_args(common + ["--run-dir", str(continuous), "--epochs", "2"]))
            uninterrupted = load_checkpoint(continuous / "checkpoints" / "last.pt")
            for name, value in second["model_state"].items():
                self.assertTrue(torch.equal(value, uninterrupted["model_state"][name]), name)

    def test_legacy_head_and_full_checkpoints_evaluate_and_resume_with_optimizer_state(self):
        from tests.test_training import synthetic_files
        from src.Eval.evaluate import evaluate
        from src.Scripts.evaluate import build_parser as evaluation_parser
        from src.Train.checkpoints import load_checkpoint, save_checkpoint
        from src.Train.engine import train
        torch = self.torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations, split, images, _ = synthetic_files(root, images=True)
            for fine_tune, mode in ((False, "head"), (True, "full")):
                with self.subTest(legacy_mode=mode):
                    run = root / mode
                    options = ["--model", "image", "--fine-tune" if fine_tune else "--no-fine-tune",
                               "--no-pretrained", "--annotations", str(annotations),
                               "--split", str(split), "--images-dir", str(images),
                               "--device", "cpu", "--workers", "0", "--batch-size", "10",
                               "--dropout", "0.1", "--lr", "0.0003", "--weight-decay", "0.001",
                               "--run-dir", str(run), "--epochs", "1", "--no-progress"]
                    train(build_parser().parse_args(options))
                    last = run / "checkpoints" / "last.pt"
                    legacy = load_checkpoint(last)
                    # Emulate the exact fine_tune-only settings/model definition
                    # persisted before explicit image training modes existed.
                    legacy["config"].pop("image_training")
                    legacy["config"].pop("parameter_counts", None)
                    legacy["config"]["model_config"].pop("image_training")
                    save_checkpoint(last, legacy)
                    evaluation = evaluate(evaluation_parser().parse_args([
                        "--checkpoint", str(last), "--device", "cpu", "--no-progress"]))
                    self.assertEqual(evaluation["metrics"]["samples"], 10)
                    self.assertEqual(evaluation["model_config"], legacy["config"]["model_config"])
                    self.assertNotIn("image_training", evaluation["model_config"])
                    train(build_parser().parse_args(["--resume", str(last), "--epochs", "2"]))
                    resumed = load_checkpoint(last)
                    self.assertEqual(resumed["epoch"], 2)
                    self.assertEqual(resumed["config"]["fine_tune"], fine_tune)
                    self.assertEqual(resumed["config"]["image_training"], mode)
                    self.assertEqual(resumed["config"]["model_config"]["image_training"], mode)
                    restored = self.build_model(resumed["config"]["model_config"], pretrained=False)
                    for name, parameter in restored.named_parameters():
                        self.assertEqual(parameter.requires_grad, self.is_trainable(name, mode), name)
                    previous_optimizer = legacy["optimizer_state"]
                    resumed_optimizer = resumed["optimizer_state"]
                    self.assertTrue(previous_optimizer["state"])
                    self.assertEqual(resumed_optimizer["param_groups"], previous_optimizer["param_groups"])
                    self.assertEqual(set(resumed_optimizer["state"]), set(previous_optimizer["state"]))
                    for index, state in previous_optimizer["state"].items():
                        self.assertEqual(state["step"].item(), 1)
                        self.assertEqual(resumed_optimizer["state"][index]["step"].item(), 2)
                        self.assertEqual(resumed_optimizer["state"][index]["exp_avg"].shape,
                                         state["exp_avg"].shape)
                        self.assertTrue(torch.isfinite(resumed_optimizer["state"][index]["exp_avg"]).all())
                    for name, value in legacy["model_state"].items():
                        if not self.is_trainable(name, mode):
                            self.assertTrue(torch.equal(value, resumed["model_state"][name]), name)
                    expected_batches = 2 if fine_tune else 0
                    self.assertEqual(resumed["model_state"]["backbone.bn1.num_batches_tracked"].item(),
                                     expected_batches)


if __name__ == "__main__":
    unittest.main()
