"""Pure-Python checks plus opt-in, tiny CPU train/resume/evaluate integration."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from src.Dataset.annotations import CLASS_NAMES_10, LABEL_MAP_10, sha256_file
from src.Dataset.splits import build_manifest
from src.Eval.metrics import classification_metrics
from src.Loader.pose import pose_features
from src.Loader.splits import read_manifest
from src.Scripts.train import build_parser
from src.Train.config import resolve_run_dir, resolve_settings


class PoseFeatureTests(unittest.TestCase):
    def test_shared_normalization_preserves_pair_geometry(self):
        pose1 = [[10, 20, 1.2]] * 17
        pose2 = [[30, 20, 0.5]] * 17
        features = pose_features(pose1, pose2)
        self.assertEqual(len(features), 102)
        self.assertEqual(features[:3], [-0.5, 0.0, 1.0])
        self.assertEqual(features[51:54], [0.5, 0.0, 0.5])
        translated = pose_features([[120, 140, 1.2]] * 17, [[160, 140, 0.5]] * 17)
        self.assertEqual(features, translated)
        self.assertEqual(pose_features(pose1, pose2, swap=True), features[51:] + features[:51])

    def test_missing_and_unobserved_joints_are_zero(self):
        pose = [[1, 2, 1]] + [[9999, 9999, 0]] * 16
        features = pose_features(pose, None)
        self.assertEqual(features[:3], [0.0, 0.0, 1.0])
        self.assertEqual(features[3:], [0.0] * 99)
        self.assertEqual(pose_features(None, None), [0.0] * 102)

    def test_metrics_include_all_classes_and_matrix_axes(self):
        metrics = classification_metrics([0, 0, 1], [0, 1, 1], ["a", "b"])
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], 2 / 3)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 1]])
        self.assertEqual(metrics["per_class"][0]["support"], 2)
        with self.assertRaises(ValueError):
            classification_metrics([], [], ["a"])


def synthetic_files(root, *, images=False):
    annotations = root / "annotations.json"
    split = root / "split.json"
    images_dir = root / "images"
    if images:
        images_dir.mkdir()
        from PIL import Image
    reverse = {name: next(raw for raw, mapped in LABEL_MAP_10.items() if mapped == name)
               for name in CLASS_NAMES_10}
    records, assignments = [], {}
    for part_index, part in enumerate(("train", "val", "test")):
        for index, name in enumerate(CLASS_NAMES_10):
            image_id = f"{part_index + 1:02d}{index:05d}"
            records.append({"image": image_id, "frame": index,
                            "position": reverse[name],
                            "pose1": [[float(j), float(j + index), 0.5 + index / 20]
                                      for j in range(17)],
                            "pose2": [[float(j + 20), float(j), 1.1] for j in range(17)]})
            assignments[image_id] = part
            if images:
                Image.new("RGB", (32, 24), (index * 20, part_index * 30, 100)).save(
                    images_dir / f"{image_id}.png")
    annotations.write_text(json.dumps(records), encoding="utf-8")
    groups = {"description": "Synthetic fixture: each prefix is a separately generated recording.",
              "sequences": {f"{index + 1:02d}": {"recording": f"synthetic-{part}", "split": part}
                            for index, part in enumerate(("train", "val", "test"))}}
    manifest = build_manifest(records, groups, sha256_file(annotations))
    split.write_text(json.dumps(manifest), encoding="utf-8")
    return annotations, split, images_dir, records


class TrainingConfigurationTests(unittest.TestCase):
    def test_resume_and_evaluation_cli_accept_checkpoint_only(self):
        from src.Scripts.evaluate import build_parser as evaluation_parser
        args = build_parser().parse_args(["--resume", "run/checkpoints/last.pt", "--epochs", "30"])
        self.assertIsNone(args.model)
        self.assertIsNone(args.annotations)
        evaluation = evaluation_parser().parse_args(["--checkpoint", "run/checkpoints/best.pt"])
        self.assertIsNone(evaluation.split)

    def test_resume_preserves_settings_and_rejects_seed_changes(self):
        parser = build_parser()
        initial = parser.parse_args(["--model", "pose", "--annotations", "a", "--split", "b",
                                     "--seed", "7", "--lr", "0.002"])
        settings = resolve_settings(initial)
        resume = parser.parse_args(["--model", "pose", "--annotations", "a", "--split", "b",
                                    "--epochs", "30"])
        resolved = resolve_settings(resume, settings)
        self.assertEqual(resolved["seed"], 7)
        self.assertEqual(resolved["lr"], 0.002)
        self.assertEqual(resolved["epochs"], 30)
        resume.seed = 8
        with self.assertRaisesRegex(ValueError, "seed"):
            resolve_settings(resume, settings)

    def test_new_runs_do_not_reuse_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                resolve_run_dir(directory)
            with self.assertRaisesRegex(ValueError, "last.pt"):
                resolve_run_dir(None, str(Path(directory) / "best.pt"))

    def test_manifest_checks_hash_and_every_split_class(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations, split, _, records = synthetic_files(root)
            manifest = read_manifest(split, annotations, records)
            manifest["assignments"][records[0]["image"]] = "test"
            split.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "recording groups"):
                read_manifest(split, annotations, records)
            annotations.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                read_manifest(split, annotations, records)

    def test_recording_group_evidence_cannot_cross_partitions(self):
        with tempfile.TemporaryDirectory() as directory:
            annotations, split, _, records = synthetic_files(Path(directory))
            manifest = json.loads(split.read_text())
            sequences = manifest["recording_groups"]["sequences"]
            sequences["02"]["recording"] = sequences["01"]["recording"]
            split.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                read_manifest(split, annotations, records)


@unittest.skipUnless(os.environ.get("JIUJITSU_RUN_TORCH_TESTS") == "1",
                     "Set JIUJITSU_RUN_TORCH_TESTS=1 for tiny CPU integration checks")
class TorchIntegrationTests(unittest.TestCase):
    def test_both_models_train_resume_and_evaluate(self):
        import torch
        from src.Eval.evaluate import evaluate
        from src.Scripts.evaluate import build_parser as evaluation_parser
        from src.Train.checkpoints import load_checkpoint
        from src.Train.engine import train
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations, split, images_dir, _ = synthetic_files(root, images=True)
            for name in ("pose", "image"):
                with self.subTest(model=name):
                    run = root / name
                    common = ["--model", name, "--annotations", str(annotations),
                              "--split", str(split), "--device", "cpu"]
                    if name == "image":
                        common += ["--images-dir", str(images_dir)]
                    train(build_parser().parse_args(common + ["--run-dir", str(run),
                        "--epochs", "1", "--workers", "0", "--batch-size", "10", "--no-pretrained"]))
                    last = run / "checkpoints" / "last.pt"
                    first = load_checkpoint(last)
                    self.assertEqual(first["epoch"], 1)
                    train(build_parser().parse_args(["--resume", str(last), "--epochs", "2"]))
                    second = load_checkpoint(last)
                    self.assertEqual(second["epoch"], 2)
                    self.assertEqual(second["config"]["workers"], 0)
                    metrics = [json.loads(line) for line in
                               (run / "diagnostics" / "metrics.jsonl").read_text().splitlines()]
                    self.assertEqual([row["epoch"] for row in metrics], [1, 2])
                    best = run / "checkpoints" / "best.pt"
                    chosen = load_checkpoint(best)
                    self.assertEqual(chosen["best_macro_f1"], max(row["val"]["macro_f1"] for row in metrics))
                    result = evaluate(evaluation_parser().parse_args([
                        "--checkpoint", str(best), "--device", "cpu"]))
                    self.assertEqual(result["split"], "test")
                    self.assertEqual(result["metrics"]["samples"], 10)
                    self.assertEqual(len(result["metrics"]["confusion_matrix"]), 10)
                    if name == "pose":
                        # Restore RNG+sampler state: a resumed run equals a continuous run.
                        continuous = root / "continuous"
                        train(build_parser().parse_args(common + ["--run-dir", str(continuous),
                            "--epochs", "2", "--workers", "0", "--batch-size", "10", "--no-pretrained"]))
                        uninterrupted = load_checkpoint(continuous / "checkpoints" / "last.pt")
                        for key, value in second["model_state"].items():
                            self.assertTrue(torch.equal(value, uninterrupted["model_state"][key]), key)


if __name__ == "__main__":
    unittest.main()
