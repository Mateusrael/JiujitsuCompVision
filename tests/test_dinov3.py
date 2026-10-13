"""Frozen DINOv3 contracts, real Base-model gradients, and offline run restoration."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.Scripts.train import build_parser
from src.Train.config import resolve_settings


class DINOv3ConfigurationTests(unittest.TestCase):
    @staticmethod
    def settings(*flags, previous=None):
        return resolve_settings(
            build_parser().parse_args(["--model", "image-dinov3", *flags]), previous)

    def test_registered_image_model_defaults_to_pretrained_frozen_head(self):
        from src.Modules.registry import IMAGE_MODEL_NAMES, MODEL_NAMES, POSE_MODEL_NAMES
        self.assertIn("image-dinov3", MODEL_NAMES)
        self.assertIn("image-dinov3", IMAGE_MODEL_NAMES)
        self.assertNotIn("image-dinov3", POSE_MODEL_NAMES)
        for flags in ((), ("--image-training", "head"), ("--no-fine-tune",)):
            with self.subTest(flags=flags):
                settings = self.settings(*flags)
                self.assertEqual(settings["image_training"], "head")
                self.assertFalse(settings["fine_tune"])
                self.assertTrue(settings["pretrained"])
        self.assertFalse(self.settings("--no-pretrained")["pretrained"])

    def test_backbone_training_modes_and_attention_options_are_rejected(self):
        for flags in (("--fine-tune",), ("--image-training", "full"),
                      ("--image-training", "layer4"), ("--attention-dropout", "0.1")):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                self.settings(*flags)

    def test_resume_preserves_head_settings_and_rejects_training_changes(self):
        previous = self.settings("--dropout", "0.2", "--horizontal-flip-prob", "0.5")
        resumed = self.settings("--epochs", "30", previous=previous)
        self.assertEqual(resumed["image_training"], "head")
        self.assertFalse(resumed["fine_tune"])
        self.assertEqual(resumed["dropout"], 0.2)
        self.assertEqual(resumed["horizontal_flip_prob"], 0.5)
        for flags in (("--fine-tune",), ("--image-training", "full"),
                      ("--image-training", "layer4"), ("--dropout", "0.3")):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                self.settings(*flags, previous=previous)


@unittest.skipUnless(os.environ.get("JIUJITSU_RUN_TORCH_TESTS") == "1",
                     "Set JIUJITSU_RUN_TORCH_TESTS=1 for CPU DINOv3 checks")
class DINOv3ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
        from src.Modules.dinov3 import FrozenDINOv3Classifier
        from src.Modules.models import architecture_config, build_model
        cls.torch = torch
        cls.Backbone = DINOv3ViTModel
        cls.BackboneConfig = DINOv3ViTConfig
        cls.Classifier = FrozenDINOv3Classifier
        cls.architecture_config = staticmethod(architecture_config)
        cls.build_model = staticmethod(build_model)
        torch.set_num_threads(1)

    def setUp(self):
        self.torch.manual_seed(27)

    def tiny_backbone_class(self):
        """Retain the actual classifier and engine; replace only expensive features."""
        torch = self.torch

        class TinyBackbone(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.projection = torch.nn.Linear(3, config.hidden_size)

            def forward(self, pixel_values):
                features = self.projection(pixel_values.mean(dim=(-2, -1)))
                return SimpleNamespace(pooler_output=features,
                                       last_hidden_state=features[:, None, :])

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise AssertionError("Tests must not download pretrained weights")

        return TinyBackbone

    def test_architecture_is_explicit_base_config_and_rejects_backbone_training(self):
        from src.Modules.dinov3_config import DINOV3_BACKBONE_CONFIG, DINOV3_WEIGHTS_ID
        config = self.architecture_config("image-dinov3", dropout=0.2)
        self.assertEqual(config["backbone_config"], DINOV3_BACKBONE_CONFIG)
        self.assertEqual(config["weights_id"], DINOV3_WEIGHTS_ID)
        self.assertEqual(config["image_training"], "head")
        self.assertFalse(config["fine_tune"])
        self.assertEqual(config["num_classes"], 10)
        self.assertEqual(config["image_size"], 224)
        self.assertEqual(config["image_preprocessing"], "letterbox_imagenet_v1")
        self.assertEqual(config["dropout"], 0.2)
        for options in ({"fine_tune": True}, {"image_training": "layer4"},
                        {"image_training": "full"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.architecture_config("image-dinov3", **options)
        for change in ("fine_tune", "image_training", "weights_id", "backbone_config"):
            bad = copy.deepcopy(config)
            if change == "fine_tune":
                bad[change] = True
            elif change == "image_training":
                bad[change] = "full"
            elif change == "weights_id":
                bad[change] = "unsupported/checkpoint"
            else:
                bad[change]["hidden_size"] = 384
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.build_model(bad, pretrained=False)

    def test_actual_base_model_updates_only_7690_classifier_parameters(self):
        torch = self.torch
        with patch.object(self.Backbone, "from_pretrained",
                          side_effect=AssertionError("Unexpected download")) as remote:
            model = self.build_model(self.architecture_config("image-dinov3"), pretrained=False)
        remote.assert_not_called()
        self.assertIsInstance(model.backbone, self.Backbone)
        config = model.backbone.config
        self.assertEqual((config.hidden_size, config.num_hidden_layers,
                          config.num_attention_heads, config.patch_size,
                          config.num_register_tokens), (768, 12, 12, 16, 4))
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 7690)
        before = {key: value.clone() for key, value in model.backbone.state_dict().items()}
        head_before = model.classifier.weight.detach().clone()
        inputs = torch.randn(2, 3, 224, 224, requires_grad=True)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=0.01, weight_decay=0.01)
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (2, 10))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 9]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNone(inputs.grad)
        for name, parameter in model.named_parameters():
            if name.startswith("backbone."):
                self.assertFalse(parameter.requires_grad, name)
                self.assertIsNone(parameter.grad, name)
            else:
                self.assertTrue(parameter.requires_grad, name)
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        self.assertFalse(torch.equal(head_before, model.classifier.weight))
        for key, value in model.backbone.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_head_dropout_stays_active_while_backbone_stays_in_eval(self):
        torch = self.torch
        with patch("src.Modules.dinov3._transformers_classes",
                   return_value=(self.BackboneConfig, self.tiny_backbone_class())):
            model = self.Classifier(pretrained=False, dropout=0.5)
        inputs = torch.randn(16, 3, 32, 32)
        for _ in range(2):
            self.assertIs(model.train(), model)
            self.assertFalse(model.backbone.training)
            self.assertTrue(model.dropout.training)
            self.assertFalse(torch.equal(model(inputs), model(inputs)))
            self.assertIs(model.eval(), model)
            self.assertTrue(all(not module.training for module in model.modules()))
            with torch.inference_mode():
                torch.testing.assert_close(model(inputs), model(inputs), rtol=0, atol=0)

    def test_actual_backbone_and_head_restore_offline_with_identical_predictions(self):
        torch = self.torch
        config = json.loads(json.dumps(self.architecture_config("image-dinov3", dropout=0.3)))
        with patch.object(self.Backbone, "from_pretrained",
                          side_effect=AssertionError("Checkpoint restoration must be offline")) as remote:
            model = self.build_model(config, pretrained=False).eval()
            # Both models are real Base architectures; all weights are restored.
            restored = self.build_model(config, pretrained=False).eval()
            restored.load_state_dict(model.state_dict(), strict=True)
            pixels = torch.randn(1, 3, 224, 224)
            with torch.inference_mode():
                torch.testing.assert_close(model(pixels), restored(pixels), rtol=0, atol=0)
        remote.assert_not_called()
        self.assertTrue(all(not parameter.requires_grad
                            for parameter in restored.backbone.parameters()))

    def test_pretrained_source_errors_and_wrong_architecture_do_not_fall_back(self):
        from src.Modules.dinov3_config import DINOV3_BACKBONE_CONFIG, DINOV3_WEIGHTS_ID
        TinyBackbone = self.tiny_backbone_class()
        matching = TinyBackbone(self.BackboneConfig(**DINOV3_BACKBONE_CONFIG))
        with patch("src.Modules.dinov3._transformers_classes",
                   return_value=(self.BackboneConfig, TinyBackbone)):
            with patch.object(TinyBackbone, "from_pretrained", return_value=matching) as remote:
                model = self.Classifier(pretrained=True)
                self.assertIs(model.backbone, matching)
            remote.assert_called_once_with(DINOV3_WEIGHTS_ID, use_safetensors=True)
            denied = OSError("Access denied")
            with patch.object(TinyBackbone, "from_pretrained", side_effect=denied):
                with self.assertRaisesRegex(RuntimeError, "hf auth login") as failure:
                    self.Classifier(pretrained=True)
                self.assertIs(failure.exception.__cause__, denied)
            matching.config.hidden_size = 384
            with patch.object(TinyBackbone, "from_pretrained", return_value=matching):
                with self.assertRaisesRegex(ValueError, "configuration"):
                    self.Classifier(pretrained=True)

    def test_whole_base_model_compile_preserves_frozen_training_and_evaluation(self):
        from src.Modules.execution import compile_model
        torch = self.torch
        original_compile = torch.compile

        def cpu_graph_capture(model, **options):
            options.pop("mode", None)
            return original_compile(model, backend="eager", **options)

        model = self.Classifier(pretrained=False, dropout=0.2)
        with patch("torch.compile", side_effect=cpu_graph_capture) as compiler:
            compiled = compile_model(model, enabled=True)
        compiler.assert_called_once_with(model, mode="default")
        inputs = torch.randn(1, 3, 32, 32)
        compiled.train()
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.dropout.training)
        loss = torch.nn.functional.cross_entropy(compiled(inputs), torch.tensor([3]))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertTrue(torch.isfinite(model.classifier.weight.grad).all())
        self.assertGreater(model.classifier.weight.grad.abs().sum().item(), 0)
        compiled.eval()
        with torch.inference_mode():
            torch.testing.assert_close(compiled(inputs), model(inputs), rtol=0, atol=0)

    def test_image_cohort_and_letterboxing_match_resnet_including_missing_poses(self):
        from src.Dataset.annotations import CLASS_NAMES_10, LABEL_MAP_10
        from src.Train.engine import make_dataset, select_model_records
        from tests.test_training import synthetic_files
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, split, images, records = synthetic_files(root, images=True)
            manifest = json.loads(split.read_text())
            for record in records[::2]:
                record["pose2"] = None
            for partition in ("train", "val", "test"):
                dino, dino_counts = select_model_records("image-dinov3", records, manifest, partition)
                image, image_counts = select_model_records("image", records, manifest, partition)
                self.assertEqual(dino, image)
                self.assertEqual(dino_counts, image_counts)
                self.assertEqual(dino_counts["removed"], 0)
            config = {"class_names": CLASS_NAMES_10, "label_map": LABEL_MAP_10,
                      "images_dir": str(images), "horizontal_flip_prob": 0.5}
            dino_set = make_dataset("image-dinov3", records, config)
            image_set = make_dataset("image", records, config)
            self.assertEqual(len(dino_set), 30)
            self.assertEqual(dino_set.horizontal_flip_prob, 0.0)
            for index in (0, 15, 29):
                pixels, label = dino_set[index]
                self.assertEqual(tuple(pixels.shape), (3, 224, 224))
                self.torch.testing.assert_close(pixels, image_set[index][0], rtol=0, atol=0)
                self.assertEqual(label, image_set[index][1])
            self.assertEqual(make_dataset("image-dinov3", records, config,
                                          training=True).horizontal_flip_prob, 0.5)

    def test_train_resume_evaluate_persist_frozen_mode_without_downloads(self):
        from src.Eval.evaluate import evaluate
        from src.Modules.dinov3_config import DINOV3_BACKBONE_CONFIG
        from src.Scripts.evaluate import build_parser as evaluation_parser
        from src.Train.checkpoints import load_checkpoint
        from src.Train.engine import train
        from tests.test_training import synthetic_files
        torch = self.torch
        TinyBackbone = self.tiny_backbone_class()

        def fake_pretrained(*args, **kwargs):
            return TinyBackbone(self.BackboneConfig(**DINOV3_BACKBONE_CONFIG))

        with tempfile.TemporaryDirectory() as directory, patch(
                "src.Modules.dinov3._transformers_classes",
                return_value=(self.BackboneConfig, TinyBackbone)), patch.object(
                    TinyBackbone, "from_pretrained", side_effect=fake_pretrained) as remote:
            root = Path(directory)
            annotations, split, images, _ = synthetic_files(root, images=True)
            common = ["--model", "image-dinov3", "--pretrained", "--image-training", "head",
                      "--annotations", str(annotations), "--split", str(split),
                      "--images-dir", str(images), "--device", "cpu", "--workers", "0",
                      "--batch-size", "10", "--dropout", "0.2", "--horizontal-flip-prob", "0.5",
                      "--weight-decay", "0.001", "--no-progress"]
            run = root / "dinov3"
            train(build_parser().parse_args(common + ["--run-dir", str(run), "--epochs", "1"]))
            last = run / "checkpoints" / "last.pt"
            first = load_checkpoint(last)
            self.assertEqual(first["config"]["parameter_counts"]["trainable"], 7690)
            self.assertEqual(remote.call_count, 1)
            train(build_parser().parse_args(["--resume", str(last), "--epochs", "2"]))
            self.assertEqual(remote.call_count, 1, "Resume must use saved backbone weights")
            second = load_checkpoint(last)
            self.assertEqual(second["epoch"], 2)
            self.assertEqual(second["config"]["image_training"], "head")
            self.assertFalse(second["config"]["fine_tune"])
            self.assertTrue(second["config"]["pretrained"])
            self.assertEqual(second["config"]["pose_filter"], "not_applicable")
            self.assertEqual(second["config"]["model_config"], first["config"]["model_config"])
            for key, value in first["model_state"].items():
                if key.startswith("backbone."):
                    self.assertTrue(torch.equal(value, second["model_state"][key]), key)
            self.assertFalse(torch.equal(first["model_state"]["classifier.weight"],
                                         second["model_state"]["classifier.weight"]))
            metrics = [json.loads(line) for line in
                       (run / "diagnostics" / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["epoch"] for row in metrics], [1, 2])
            self.assertTrue(all(row["train"]["samples"] == 10 for row in metrics))
            result = evaluate(evaluation_parser().parse_args([
                "--checkpoint", str(last), "--device", "cpu", "--no-progress"]))
            self.assertEqual(remote.call_count, 1, "Evaluation must use saved backbone weights")
            self.assertEqual(result["metrics"]["samples"], 10)
            self.assertEqual(result["model_config"], first["config"]["model_config"])
            continuous = root / "continuous"
            train(build_parser().parse_args(common + ["--run-dir", str(continuous), "--epochs", "2"]))
            self.assertEqual(remote.call_count, 2)
            uninterrupted = load_checkpoint(continuous / "checkpoints" / "last.pt")
            for key, value in second["model_state"].items():
                self.assertTrue(torch.equal(value, uninterrupted["model_state"][key]), key)


if __name__ == "__main__":
    unittest.main()
