"""New image stages copy weights without inheriting optimizer state or history."""

import copy
import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.Dataset.annotations import sha256_file
from src.Scripts.train import build_parser
from src.Train.config import DEFAULTS
from tests.test_training import synthetic_files


@unittest.skipUnless(os.environ.get("JIUJITSU_RUN_TORCH_TESTS") == "1",
                     "Set JIUJITSU_RUN_TORCH_TESTS=1 for CPU warm-start checks")
class ImageWarmStartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from src.Train import engine
        from src.Train.checkpoints import load_checkpoint
        cls.torch = torch
        cls.engine = engine
        cls.load_checkpoint = staticmethod(load_checkpoint)
        torch.set_num_threads(1)

    def setUp(self):
        torch = self.torch

        class TinyResNet(torch.nn.Module):
            def __init__(self, *, weights=None):
                super().__init__()
                if weights is not None:
                    raise AssertionError("Warm starts must not download weights")
                self.conv1 = torch.nn.Linear(3, 8)
                self.bn1 = torch.nn.BatchNorm1d(8)
                self.layer4 = torch.nn.Sequential(torch.nn.Linear(8, 512), torch.nn.ReLU())
                self.fc = torch.nn.Identity()

            def forward(self, pixels):
                features = self.bn1(self.conv1(pixels.mean(dim=(-2, -1))))
                return self.fc(self.layer4(features))

        class TinyDINO(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.projection = torch.nn.Linear(3, config.hidden_size)
                self.middle_scale = torch.nn.Parameter(torch.ones(config.hidden_size))

            def get_input_embeddings(self):
                return self.projection

            def forward(self, pixel_values):
                features = self.projection(pixel_values.mean(dim=(-2, -1)))
                return SimpleNamespace(pooler_output=features.tanh() * self.middle_scale)

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise AssertionError("Warm starts must not download weights")

        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch("torchvision.models.resnet18", TinyResNet))
        self.stack.enter_context(patch("src.Modules.dinov3._transformers_classes",
                                      return_value=(SimpleNamespace, TinyDINO)))
        self.annotations, self.split, self.images, _ = synthetic_files(self.root, images=True)

    def make_source(self, model="image"):
        run = self.root / f"source-{model}"
        self.engine.train(build_parser().parse_args([
            "--model", model, "--annotations", str(self.annotations),
            "--split", str(self.split), "--images-dir", str(self.images),
            "--run-dir", str(run), "--no-pretrained", "--device", "cpu",
            "--epochs", "2", "--workers", "0", "--batch-size", "10",
            "--seed", "13", "--lr", "0.005", "--weight-decay", "0.01",
            "--dropout", "0.2", "--no-progress"]))
        return run / "checkpoints" / "last.pt"

    def warm_args(self, checkpoint, run="target", *flags):
        return build_parser().parse_args([
            "--init-from", str(checkpoint), "--run-dir", str(self.root / run),
            "--image-training", "input-head", "--epochs", "1", "--workers", "0",
            "--device", "cpu", "--no-progress", *flags])

    def test_new_stages_copy_all_weights_use_fresh_defaults_and_resume_independently(self):
        for model_name in ("image", "image-dinov3"):
            with self.subTest(model=model_name):
                checkpoint = self.make_source(model_name)
                source = self.load_checkpoint(checkpoint)
                source_run = checkpoint.parent.parent
                original_files = {str(path.relative_to(source_run)): sha256_file(path)
                                  for path in source_run.rglob("*") if path.is_file()}
                original_epoch = self.engine.train_epoch
                observed = []

                def inspect_first_epoch(model, loader, optimizer, device, **options):
                    self.assertFalse(optimizer.state, "Warm starts need a fresh optimizer")
                    self.assertEqual(optimizer.param_groups[0]["lr"], DEFAULTS["lr"])
                    self.assertEqual(optimizer.param_groups[0]["weight_decay"], DEFAULTS["weight_decay"])
                    self.assertEqual(loader.batch_size, DEFAULTS["batch_size"])
                    self.assertEqual(loader.generator.initial_seed(), DEFAULTS["seed"])
                    for key, value in model.state_dict().items():
                        self.assertTrue(self.torch.equal(value, source["model_state"][key]), key)
                    observed.append(True)
                    return original_epoch(model, loader, optimizer, device, **options)

                run = self.root / f"target-{model_name}"
                with patch.object(self.engine, "train_epoch", side_effect=inspect_first_epoch), patch.object(
                        self.engine, "restore_random_state",
                        side_effect=AssertionError("Initialization must not restore source RNG")):
                    self.engine.train(self.warm_args(checkpoint, run.name, "--dropout", "0.3"))
                self.assertEqual(observed, [True])
                last = run / "checkpoints" / "last.pt"
                target = self.load_checkpoint(last)
                self.assertEqual(target["epoch"], 1)
                self.assertEqual(target["config"]["dropout"], 0.3)
                self.assertEqual(target["config"]["seed"], DEFAULTS["seed"])
                self.assertEqual(target["config"]["image_training"], "input-head")
                self.assertFalse(target["config"]["pretrained"], "Copied weights retain their origin")
                metrics = [json.loads(line) for line in
                           (run / "diagnostics" / "metrics.jsonl").read_text().splitlines()]
                self.assertEqual([row["epoch"] for row in metrics], [1])
                self.assertEqual(target["best_macro_f1"], metrics[0]["val"]["macro_f1"])
                for state in target["optimizer_state"]["state"].values():
                    self.assertEqual(int(state["step"]), 1)
                event = json.loads((run / "config_start.json").read_text())
                self.assertIsNone(event["resumed_from"])
                lineage = {"checkpoint": str(checkpoint.resolve()),
                           "checkpoint_sha256": sha256_file(checkpoint), "source_epoch": 2}
                self.assertEqual(event["initialized_from"], lineage)
                self.assertEqual(target["config"]["initialized_from"], lineage)

                self.engine.train(build_parser().parse_args(["--resume", str(last), "--epochs", "2"]))
                resumed = self.load_checkpoint(last)
                self.assertEqual(resumed["config"]["initialized_from"], lineage)
                continuous = self.root / f"continuous-{model_name}"
                self.engine.train(self.warm_args(checkpoint, continuous.name,
                                                "--dropout", "0.3", "--epochs", "2"))
                uninterrupted = self.load_checkpoint(continuous / "checkpoints" / "last.pt")
                for key, value in resumed["model_state"].items():
                    self.assertTrue(self.torch.equal(value, uninterrupted["model_state"][key]), key)
                self.assertEqual(original_files,
                                 {str(path.relative_to(source_run)): sha256_file(path)
                                  for path in source_run.rglob("*") if path.is_file()})

    def test_best_checkpoint_and_explicit_new_hyperparameters_are_supported(self):
        checkpoint = self.make_source().with_name("best.pt")
        self.engine.train(self.warm_args(checkpoint, "best-target", "--lr", "0.00003",
                                        "--weight-decay", "0.002", "--seed", "91",
                                        "--image-training", "layer4"))
        saved = self.load_checkpoint(self.root / "best-target" / "checkpoints" / "last.pt")
        self.assertEqual(saved["epoch"], 1)
        self.assertEqual(saved["config"]["image_training"], "layer4")
        self.assertEqual(saved["config"]["seed"], 91)
        self.assertEqual(saved["optimizer_state"]["param_groups"][0]["lr"], 0.00003)
        self.assertEqual(saved["optimizer_state"]["param_groups"][0]["weight_decay"], 0.002)

    def test_source_replacement_during_load_keeps_weights_epoch_and_digest_consistent(self):
        checkpoint = self.make_source()
        original_bytes = checkpoint.read_bytes()
        original = self.load_checkpoint(checkpoint)
        replacement = copy.deepcopy(original)
        replacement["epoch"] = 17
        replacement["model_state"]["classifier.weight"].add_(1)
        replacement_path = checkpoint.with_name("replacement.pt")
        self.torch.save(replacement, replacement_path)
        path_open = Path.open
        train_epoch = self.engine.train_epoch

        def open_snapshot(path, mode="r", *args, **kwargs):
            if path == checkpoint and mode == "rb":
                # Model Unix open-file semantics on Windows too: an atomic
                # replacement changes the path but leaves this handle intact.
                return io.BytesIO(original_bytes)
            return path_open(path, mode, *args, **kwargs)

        def load_then_replace(handle):
            self.assertTrue(hasattr(handle, "read"), "Load and hash must share one handle")
            loaded = self.load_checkpoint(handle)
            replacement_path.replace(checkpoint)
            return loaded

        def inspect_loaded_weights(model, *args, **kwargs):
            for key, value in model.state_dict().items():
                self.assertTrue(self.torch.equal(value, original["model_state"][key]), key)
            return train_epoch(model, *args, **kwargs)

        with patch.object(Path, "open", open_snapshot), patch.object(
                self.engine, "load_checkpoint", side_effect=load_then_replace), patch.object(
                    self.engine, "train_epoch", side_effect=inspect_loaded_weights):
            self.engine.train(self.warm_args(checkpoint))
        saved = self.load_checkpoint(self.root / "target" / "checkpoints" / "last.pt")
        lineage = saved["config"]["initialized_from"]
        self.assertEqual(lineage["source_epoch"], original["epoch"])
        self.assertEqual(lineage["checkpoint_sha256"], hashlib.sha256(original_bytes).hexdigest())
        self.assertNotEqual(lineage["checkpoint_sha256"], sha256_file(checkpoint))
        self.assertEqual(self.load_checkpoint(checkpoint)["epoch"], 17)

    def test_rejects_changed_data_model_preprocessing_and_incomplete_weights(self):
        checkpoint = self.make_source()
        original = self.load_checkpoint(checkpoint)
        for key, replacement in (("annotation_sha256", "wrong"), ("split_sha256", "wrong"),
                                 ("class_names", ["wrong"]), ("label_map", {})):
            with self.subTest(key=key):
                bad = copy.deepcopy(original)
                bad["config"][key] = replacement
                self.torch.save(bad, checkpoint)
                with self.assertRaisesRegex(ValueError, key):
                    self.engine.train(self.warm_args(checkpoint))
                self.assertFalse((self.root / "target").exists())
        for key, replacement in (("image_preprocessing", "other"), ("image_size", 112),
                                 ("type", "other")):
            with self.subTest(architecture=key):
                bad = copy.deepcopy(original)
                bad["config"]["model_config"][key] = replacement
                self.torch.save(bad, checkpoint)
                with self.assertRaisesRegex(ValueError, "architecture/preprocessing"):
                    self.engine.train(self.warm_args(checkpoint))
        self.torch.save(original, checkpoint)
        with self.assertRaisesRegex(ValueError, "model"):
            self.engine.train(self.warm_args(checkpoint, "target", "--model", "image-dinov3"))
        with self.assertRaisesRegex(ValueError, "image models only"):
            self.engine.train(self.warm_args(checkpoint, "target", "--model", "pose"))
        with self.assertRaisesRegex(ValueError, "image directory"):
            self.engine.train(self.warm_args(checkpoint, "target", "--images-dir", str(self.root)))
        with self.assertRaisesRegex(ValueError, "Cannot change --pretrained"):
            self.engine.train(self.warm_args(checkpoint, "target", "--pretrained"))
        changed_split = self.root / "changed-split.json"
        changed_split.write_text(self.split.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "split_sha256"):
            self.engine.train(self.warm_args(checkpoint, "target", "--split", str(changed_split)))
        bad = copy.deepcopy(original)
        del bad["model_state"]["classifier.weight"]
        self.torch.save(bad, checkpoint)
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            self.engine.train(self.warm_args(checkpoint))
        self.assertFalse((self.root / "target").exists())

    def test_rejects_existing_run_invalid_paths_pose_sources_and_resume_combination(self):
        checkpoint = self.make_source()
        with self.assertRaises(FileExistsError):
            self.engine.train(self.warm_args(checkpoint, checkpoint.parent.parent.name))
        missing_run = self.warm_args(checkpoint)
        missing_run.run_dir = None
        with self.assertRaisesRegex(ValueError, "run-dir"):
            self.engine.train(missing_run)
        wrong_path = self.warm_args(checkpoint.with_name("weights.pt"))
        with self.assertRaisesRegex(ValueError, "best.pt or last.pt"):
            self.engine.train(wrong_path)
        both = self.warm_args(checkpoint)
        both.resume = str(checkpoint)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.engine.train(both)
        source = self.load_checkpoint(checkpoint)
        source["config"]["model"] = "pose"
        self.torch.save(source, checkpoint)
        with self.assertRaisesRegex(ValueError, "image checkpoints only"):
            self.engine.train(self.warm_args(checkpoint))


if __name__ == "__main__":
    unittest.main()
