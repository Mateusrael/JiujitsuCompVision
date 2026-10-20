"""Notebook exploration must respect checkpoint identity and training eligibility."""

import copy
import gc
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from src.Dataset.annotations import CLASS_NAMES_10, LABEL_MAP_10, sha256_file
from src.Dataset.splits import build_manifest
from src.Eval.exploration import ExplorationData, ModelExplorer, resolve_checkpoint


def fixture(root, *, images=False):
    reverse = {value: next(raw for raw, mapped in LABEL_MAP_10.items() if mapped == value)
               for value in CLASS_NAMES_10}
    records = []
    for source in range(3):
        for index, name in enumerate(CLASS_NAMES_10):
            records.append({
                "image": f"{source:02d}{index:05d}", "frame": index,
                "position": reverse[name],
                "pose1": [[float(j), float(j + index), 0.8] for j in range(17)],
                "pose2": None if index == 0 else [[float(j + 20), float(j), 1.1]
                                                 for j in range(17)],
            })
    annotations, split, images_dir = root / "annotations.json", root / "split.json", root / "images"
    annotations.write_text(json.dumps(records), encoding="utf-8")
    groups = {"description": "Independent synthetic recordings.",
              "sequences": {f"{index:02d}": {"recording": f"synthetic-{part}", "split": part}
                            for index, part in enumerate(("train", "val", "test"))}}
    manifest = build_manifest(records, groups, sha256_file(annotations))
    split.write_text(json.dumps(manifest), encoding="utf-8")
    if images:
        from PIL import Image
        images_dir.mkdir()
        for row in records:
            Image.new("RGB", (48, 32), (20, 80, row["frame"] * 20)).save(
                images_dir / f"{row['image']}.png")
    return ExplorationData(annotations, split, images_dir)


class ExplorationSelectionTests(unittest.TestCase):
    def test_filters_preserve_partition_and_pose_eligibility(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture(Path(directory))
            images = data.select("image", split="val")
            poses = data.select("pose-attention", split="val")
            self.assertEqual(len(images), 10)
            self.assertEqual(len(poses), 9)
            selected = data.select("pose", split="val", source=1, frame_range=(2, 5))
            self.assertEqual([row["frame"] for row in selected], [2, 3, 4, 5])
            selected = data.select("image", frame_range=(None, 2))
            self.assertEqual([row["frame"] for row in selected], [0, 1, 2])
            selected = data.select("image", frame_range=(8, None))
            self.assertEqual([row["frame"] for row in selected], [8, 9])
            selected = data.select("pose", split="val", class_name=CLASS_NAMES_10[2])
            self.assertEqual([row["image"] for row in selected], ["0100002"])
            self.assertTrue(all(data.describe(row)["split"] == "val" for row in poses))
            self.assertEqual(data.select("image", source=2, split="val"), [])
            self.assertEqual(data.features(poses[0]), data.features(poses[0]["image"]))
            self.assertEqual(len(data.features(poses[0])), 102)

    def test_paired_view_exclusions_and_evaluation_type_filters(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture(Path(directory))
            # Unit-test selection with an already verified manifest's derived fields.
            data.manifest["pose_assignments"] = dict(data.manifest["assignments"])
            data.manifest["pose_assignments"]["0100001"] = "excluded"
            data.manifest["evaluation_types"] = {
                row["image"]: ("held_out_view" if row["frame"] % 2 else "unseen_moment")
                for row in data.records if data.manifest["assignments"][row["image"]] != "train"}
            selected = data.select("pose", split="val", evaluation_type="held_out_view")
            self.assertEqual([row["frame"] for row in selected], [3, 5, 7, 9])
            selected = data.select("image", split="val", evaluation_type="held_out_view")
            self.assertEqual([row["frame"] for row in selected], [1, 3, 5, 7, 9])
            with self.assertRaisesRegex(ValueError, "Evaluation type"):
                data.select("pose", split="train", evaluation_type="unseen_moment")

    def test_sampling_is_reproducible_without_changing_global_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture(Path(directory))
            state = random.getstate()
            first = data.sample("pose", count=4, seed=23)
            second = data.sample("pose", count=4, seed=23)
            self.assertEqual(first, second)
            self.assertEqual(random.getstate(), state)
            self.assertEqual(len({row["image"] for row in first}), 4)
            self.assertEqual(len(data.sample("pose", count=100)), 9)
            with self.assertRaisesRegex(ValueError, "No eligible"):
                data.sample("pose", class_name=CLASS_NAMES_10[0])
            with self.assertRaisesRegex(ValueError, "positive integer"):
                data.sample("pose", count=0)

    def test_bad_filters_and_modified_annotations_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture(Path(directory))
            for kwargs in ({"source": 99}, {"class_name": "unknown"},
                           {"frame_range": (5, 2)}, {"split": "all"}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    data.select("image", **kwargs)
            altered = copy.deepcopy(data.records[0])
            altered["frame"] = 500
            with self.assertRaisesRegex(ValueError, "differs"):
                data.canonical(altered)
            data.annotations_path.write_text(data.annotations_path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                data.verify_unchanged()

    def test_checkpoint_resolution_never_falls_back_to_other_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoints").mkdir()
            last = root / "checkpoints" / "last.pt"
            last.write_bytes(b"fixture")
            self.assertEqual(resolve_checkpoint(root, "last"), last.resolve())
            self.assertEqual(resolve_checkpoint(last), last.resolve())
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint(root, "best")


@unittest.skipUnless(os.environ.get("JIUJITSU_RUN_TORCH_TESTS") == "1",
                     "Opt-in CPU checkpoint integration")
class ExplorationInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    @staticmethod
    def checkpoint(root, data, *, model_name="pose"):
        import torch
        from src.Modules.models import architecture_config, build_model
        from src.Train.config import ATTENTION_DEFAULTS, DEFAULTS
        config = {
            **DEFAULTS, **ATTENTION_DEFAULTS, "model": model_name,
            "class_names": data.class_names, "label_map": data.manifest["label_map"],
            "annotation_sha256": data.annotation_sha256, "split_sha256": data.split_sha256,
            "pose_filter": "both_present", "image_training": "head",
            "horizontal_flip_prob": 1.0,
            "annotations": "/old/dgx/annotations.json", "split": "/old/dgx/split.json",
            "images_dir": "/old/dgx/images",
        }
        config["dropout"] = 0.8
        options = {"image_training": "head"} if model_name in ("image", "image-dinov3") else {}
        if model_name == "pose-attention":
            config.update(attention_dim=32, attention_heads=2, attention_layers=2,
                          attention_mlp_dim=64, attention_dropout=0.2,
                          attention_mlp_dropout=0.8, attention_pooling="cls")
            options = {key: config[key] for key in ATTENTION_DEFAULTS}
        config["model_config"] = architecture_config(model_name, dropout=0.8, **options)
        model = build_model(config["model_config"], pretrained=False)
        model.eval()
        target = root / model_name / "checkpoints" / "best.pt"
        target.parent.mkdir(parents=True)
        torch.save({"schema_version": 1, "config": config, "model_state": model.state_dict(),
                    "optimizer_state": {}, "epoch": 12, "best_macro_f1": 0.5,
                    "random_state": {}}, target)
        return target, model

    def test_cpu_predictions_match_evaluator_inputs_and_ignore_training_augmentation(self):
        import torch
        from src.Loader.datasets import PoseDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture(root)
            target, original = self.checkpoint(root, data)
            explorer = ModelExplorer.from_run(target, data, device="cpu")
            rows = data.sample("pose", count=4, seed=11)
            expected_inputs = PoseDataset(rows, data.class_names, data.manifest["label_map"])
            with torch.inference_mode():
                expected = original(torch.stack([expected_inputs[i][0]
                                                 for i in range(len(rows))])).softmax(dim=1)
            first = explorer.predict(rows, top_k=4, batch_size=2)
            second = explorer.predict([row["image"] for row in rows], top_k=4, batch_size=2)
            self.assertEqual(first, second)
            self.assertEqual(explorer.checkpoint_epoch, 12)
            self.assertFalse(explorer.model.training)
            self.assertTrue(all(parameter.grad is None for parameter in explorer.model.parameters()))
            torch.testing.assert_close(torch.tensor([r["probabilities"] for r in first]), expected)
            for row, result in zip(rows, first):
                self.assertEqual(result["image"], row["image"])
                self.assertEqual(result["actual"], data.manifest["label_map"][row["position"]])
                self.assertEqual(result["predicted"], result["top_k"][0]["class_name"])
                self.assertAlmostEqual(sum(result["probabilities"]), 1, places=6)
                self.assertEqual(result["split"], "val")
            explorer.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                explorer.predict(rows)

    def test_checkpoint_split_mismatch_and_pose_ineligible_samples_fail(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture(root)
            target, _ = self.checkpoint(root, data)
            explorer = ModelExplorer.from_run(target, data)
            with self.assertRaisesRegex(ValueError, "pose-ineligible"):
                explorer.predict([data.by_id["0100000"]])
            payload = torch.load(target, weights_only=False)
            payload["config"]["split_sha256"] = "0" * 64
            torch.save(payload, target)
            with self.assertRaisesRegex(ValueError, "split_sha256"):
                ModelExplorer.from_run(target, data)
            explorer.close()

    def test_wide_and_attention_restore_saved_architecture_and_match_predictions(self):
        import torch
        from src.Loader.datasets import PoseDataset
        from src.Modules.models import build_model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture(root)
            rows = data.sample("pose", count=3, seed=13)
            dataset = PoseDataset(rows, data.class_names, data.manifest["label_map"])
            features = torch.stack([dataset[index][0] for index in range(len(rows))])
            for model_name in ("pose-wide", "pose-attention"):
                with self.subTest(model=model_name):
                    checkpoint, original = self.checkpoint(root, data, model_name=model_name)
                    with patch("src.Modules.models.build_model", wraps=build_model) as build:
                        explorer = ModelExplorer.from_run(checkpoint, data, device="cpu")
                    self.assertIs(build.call_args.kwargs["pretrained"], False)
                    self.assertEqual(explorer.model_name, model_name)
                    self.assertEqual(explorer.parameter_counts["total"],
                                     sum(parameter.numel() for parameter in original.parameters()))
                    if model_name == "pose-attention":
                        self.assertEqual(explorer.config["model_config"]["embedding_dim"], 32)
                        self.assertEqual(explorer.config["model_config"]["pooling"], "cls")
                        self.assertEqual(explorer.config["model_config"]["attention_dropout"], 0.2)
                        self.assertEqual(explorer.config["model_config"]["mlp_dropout"], 0.8)
                    with torch.inference_mode():
                        expected = original(features).softmax(dim=1)
                    first = explorer.predict(rows, top_k=10)
                    second = explorer.predict(rows, top_k=10)
                    self.assertEqual(first, second)
                    torch.testing.assert_close(torch.tensor([r["probabilities"] for r in first]),
                                               expected)
                    self.assertEqual([row["image"] for row in first],
                                     [row["image"] for row in rows])
                    self.assertTrue(all(parameter.grad is None
                                        for parameter in explorer.model.parameters()))
                    explorer.close()

    def test_resnet_restores_local_weights_without_downloading_and_matches_preprocessing(self):
        import torch
        from src.Loader.datasets import ImageDataset
        from src.Modules.models import build_model
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture(root, images=True)
            target, original = self.checkpoint(root, data, model_name="image")
            with patch("src.Modules.models.build_model", wraps=build_model) as build:
                explorer = ModelExplorer.from_run(target.parent.parent, data, checkpoint="best")
            self.assertIs(build.call_args.kwargs["pretrained"], False)
            rows = data.select("image", split="val")[:2]
            dataset = ImageDataset(rows, data.class_names, data.manifest["label_map"], data.images_dir)
            with torch.inference_mode():
                expected = original(torch.stack([dataset[0][0], dataset[1][0]])).softmax(dim=1)
            actual = explorer.predict(rows)
            torch.testing.assert_close(torch.tensor([row["probabilities"] for row in actual]), expected)
            image = data.open_image(rows[0])
            self.assertEqual(image.size, (48, 32))
            self.assertEqual(image.mode, "RGB")
            image.close()
            explorer.close()

    @unittest.skipUnless(importlib.util.find_spec("transformers") is not None,
                         "DINOv3 requires the optional transformers dependency")
    def test_dinov3_checkpoint_roundtrip_is_offline_and_matches_full_model_predictions(self):
        import torch
        from transformers import DINOv3ViTModel
        from src.Loader.datasets import ImageDataset
        from src.Modules.models import build_model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture(root, images=True)
            rows = data.select("image-dinov3", split="val")[:1]
            dataset = ImageDataset(rows, data.class_names, data.manifest["label_map"], data.images_dir)
            with patch.object(DINOv3ViTModel, "from_pretrained",
                              side_effect=AssertionError("Inference must never request weights")) as fetch:
                checkpoint, original = self.checkpoint(root, data, model_name="image-dinov3")
                with torch.inference_mode():
                    expected = original(dataset[0][0].unsqueeze(0)).softmax(dim=1)
                del original
                gc.collect()
                with patch("src.Modules.models.build_model", wraps=build_model) as build:
                    explorer = ModelExplorer.from_run(checkpoint, data, device="cpu")
                self.assertIs(build.call_args.kwargs["pretrained"], False)
                result = explorer.predict(rows)
                fetch.assert_not_called()
            self.assertEqual(explorer.model_name, "image-dinov3")
            self.assertEqual(explorer.checkpoint_epoch, 12)
            self.assertEqual(explorer.config["model_config"]["type"], "dinov3_vitb16")
            torch.testing.assert_close(torch.tensor([r["probabilities"] for r in result]), expected)
            self.assertEqual(result[0]["image"], rows[0]["image"])
            self.assertTrue(all(parameter.grad is None for parameter in explorer.model.parameters()))
            explorer.close()


if __name__ == "__main__":
    unittest.main()
