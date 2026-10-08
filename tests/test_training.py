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
                image = Image.new("RGB", (32, 24), (index * 20, part_index * 30, 100))
                image.paste((index * 20, 255 - part_index * 30, 40), (0, 0, 8, 12))
                image.save(images_dir / f"{image_id}.png")
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

    def test_attention_settings_are_restored_and_cannot_change_on_resume(self):
        parser = build_parser()
        initial = parser.parse_args(["--model", "pose-attention", "--attention-dim", "48",
            "--attention-heads", "3", "--attention-layers", "2", "--attention-mlp-dim", "80",
            "--attention-dropout", "0.1", "--attention-mlp-dropout", "0.3",
            "--attention-pooling", "cls"])
        settings = resolve_settings(initial)
        resume = parser.parse_args(["--model", "pose-attention", "--epochs", "30"])
        restored = resolve_settings(resume, settings)
        for name in ("attention_dim", "attention_heads", "attention_layers",
                     "attention_mlp_dim", "attention_dropout", "attention_mlp_dropout",
                     "attention_pooling"):
            self.assertEqual(restored[name], settings[name])
        for flag, value in (("--attention-dim", "96"), ("--attention-heads", "6"),
                            ("--attention-layers", "3"), ("--attention-mlp-dim", "160"),
                            ("--attention-dropout", "0"), ("--attention-mlp-dropout", "0"),
                            ("--attention-pooling", "mean")):
            with self.subTest(flag=flag):
                changed = parser.parse_args(["--model", "pose-attention", flag, value])
                with self.assertRaisesRegex(ValueError, "Cannot change"):
                    resolve_settings(changed, settings)

    def test_attention_flags_and_fine_tuning_reject_inapplicable_models(self):
        parser = build_parser()
        for name in ("pose", "pose-wide", "image"):
            for flag, value in (("--attention-dim", "64"), ("--attention-dropout", "0.1"),
                                ("--attention-mlp-dropout", "0.2")):
                with self.subTest(model=name, flag=flag):
                    with self.assertRaisesRegex(ValueError, "pose-attention"):
                        resolve_settings(parser.parse_args(["--model", name, flag, value]))
        for name in ("pose", "pose-wide", "pose-attention"):
            with self.subTest(model=name):
                with self.assertRaisesRegex(ValueError, "only to the image"):
                    resolve_settings(parser.parse_args(["--model", name, "--fine-tune"]))

    def test_invalid_attention_cli_settings_fail_before_loading_torch(self):
        parser = build_parser()
        for flags in (["--attention-dim", "31", "--attention-heads", "4"],
                      ["--attention-heads", "0"], ["--attention-layers", "0"],
                      ["--attention-mlp-dim", "-1"], ["--attention-dropout", "nan"],
                      ["--attention-dropout", "1"], ["--attention-mlp-dropout", "-0.1"],
                      ["--attention-mlp-dropout", "nan"], ["--attention-mlp-dropout", "1"]):
            with self.subTest(flags=flags):
                with self.assertRaises(ValueError):
                    resolve_settings(parser.parse_args(["--model", "pose-attention"] + flags))

    def test_dropout_and_horizontal_mirror_defaults_are_off_and_resume_is_fixed(self):
        parser = build_parser()
        for name in ("pose", "pose-wide", "pose-attention", "image"):
            with self.subTest(model=name):
                fresh = resolve_settings(parser.parse_args(["--model", name]))
                self.assertEqual(fresh["dropout"], 0.0)
                self.assertEqual(fresh["horizontal_flip_prob"], 0.0)
                selected = resolve_settings(parser.parse_args([
                    "--model", name, "--dropout", "0.2", "--horizontal-flip-prob", "0.5"]))
                restored = resolve_settings(parser.parse_args(["--model", name]), selected)
                self.assertEqual(restored["dropout"], 0.2)
                self.assertEqual(restored["horizontal_flip_prob"], 0.5)
                for flag in ("--dropout", "--horizontal-flip-prob"):
                    with self.assertRaisesRegex(ValueError, "Cannot change"):
                        resolve_settings(parser.parse_args(["--model", name, flag, "0"]), selected)

    def test_attention_dropout_branches_use_independent_overrides_and_shared_fallback(self):
        parser = build_parser()
        cases = (([], (0.0, 0.0, 0.0)),
                 (["--attention-dropout", "0.2"], (0.0, 0.2, 0.0)),
                 (["--attention-mlp-dropout", "0.3"], (0.0, 0.0, 0.3)),
                 (["--dropout", "0.4"], (0.4, 0.4, 0.4)),
                 (["--dropout", "0.4", "--attention-dropout", "0"], (0.4, 0.0, 0.4)),
                 (["--dropout", "0.4", "--attention-mlp-dropout", "0"], (0.4, 0.4, 0.0)),
                 (["--dropout", "0.4", "--attention-dropout", "0.1",
                   "--attention-mlp-dropout", "0.2"], (0.4, 0.1, 0.2)))
        fields = ("dropout", "attention_dropout", "attention_mlp_dropout")
        for flags, expected in cases:
            with self.subTest(flags=flags):
                settings = resolve_settings(parser.parse_args(["--model", "pose-attention"] + flags))
                self.assertEqual(tuple(settings[field] for field in fields), expected)
                restored = resolve_settings(parser.parse_args(["--model", "pose-attention"]), settings)
                self.assertEqual(tuple(restored[field] for field in fields), expected)

    def test_dropout_and_flip_probability_reject_invalid_values(self):
        parser = build_parser()
        for flag, values in (("--dropout", ("-0.1", "1", "nan", "inf")),
                             ("--horizontal-flip-prob", ("-0.1", "1.1", "nan", "inf"))):
            for value in values:
                with self.subTest(flag=flag, value=value):
                    with self.assertRaises(ValueError):
                        resolve_settings(parser.parse_args(["--model", "pose", flag, value]))
        self.assertEqual(resolve_settings(parser.parse_args([
            "--model", "pose", "--horizontal-flip-prob", "1"]))["horizontal_flip_prob"], 1.0)

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
    def test_all_models_train_resume_and_evaluate(self):
        import torch
        from src.Eval.evaluate import evaluate
        from src.Scripts.evaluate import build_parser as evaluation_parser
        from src.Train.checkpoints import load_checkpoint
        from src.Train.engine import train
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations, split, images_dir, _ = synthetic_files(root, images=True)
            for name in ("pose", "pose-wide", "pose-attention", "image"):
                with self.subTest(model=name):
                    run = root / name
                    common = ["--model", name, "--annotations", str(annotations),
                              "--split", str(split), "--device", "cpu",
                              "--dropout", "0.1", "--horizontal-flip-prob", "0.5", "--no-progress"]
                    if name == "image":
                        common += ["--images-dir", str(images_dir)]
                    if name == "pose-attention":
                        common += ["--attention-dim", "32", "--attention-heads", "2",
                                   "--attention-layers", "2", "--attention-mlp-dim", "64",
                                   "--attention-dropout", "0.1", "--attention-mlp-dropout", "0.2",
                                   "--attention-pooling", "cls"]
                    train(build_parser().parse_args(common + ["--run-dir", str(run),
                        "--epochs", "1", "--workers", "0", "--batch-size", "10", "--no-pretrained"]))
                    last = run / "checkpoints" / "last.pt"
                    first = load_checkpoint(last)
                    self.assertEqual(first["epoch"], 1)
                    train(build_parser().parse_args(["--resume", str(last), "--epochs", "2"]))
                    second = load_checkpoint(last)
                    self.assertEqual(second["epoch"], 2)
                    self.assertEqual(second["config"]["workers"], 0)
                    self.assertEqual(second["config"]["dropout"], 0.1)
                    self.assertEqual(second["config"]["horizontal_flip_prob"], 0.5)
                    self.assertEqual(second["config"]["model_config"], first["config"]["model_config"])
                    if name == "pose-attention":
                        self.assertEqual(second["config"]["attention_dim"], 32)
                        self.assertEqual(second["config"]["attention_heads"], 2)
                        self.assertEqual(second["config"]["attention_layers"], 2)
                        self.assertEqual(second["config"]["attention_mlp_dim"], 64)
                        self.assertEqual(second["config"]["attention_dropout"], 0.1)
                        self.assertEqual(second["config"]["attention_mlp_dropout"], 0.2)
                        self.assertEqual(second["config"]["model_config"]["attention_dropout"], 0.1)
                        self.assertEqual(second["config"]["model_config"]["mlp_dropout"], 0.2)
                        self.assertNotIn("dropout", second["config"]["model_config"])
                        self.assertEqual(second["config"]["attention_pooling"], "cls")
                        for flag, value in (("--attention-dim", "64"), ("--attention-dropout", "0"),
                                            ("--attention-mlp-dropout", "0")):
                            with self.assertRaisesRegex(ValueError, f"Cannot change {flag}"):
                                train(build_parser().parse_args([
                                    "--resume", str(last), "--epochs", "3", flag, value]))
                    metrics = [json.loads(line) for line in
                               (run / "diagnostics" / "metrics.jsonl").read_text().splitlines()]
                    self.assertEqual([row["epoch"] for row in metrics], [1, 2])
                    best = run / "checkpoints" / "best.pt"
                    chosen = load_checkpoint(best)
                    self.assertEqual(chosen["best_macro_f1"], max(row["val"]["macro_f1"] for row in metrics))
                    result = evaluate(evaluation_parser().parse_args([
                        "--checkpoint", str(best), "--device", "cpu", "--no-progress"]))
                    self.assertEqual(result["split"], "test")
                    self.assertEqual(result["metrics"]["samples"], 10)
                    self.assertEqual(len(result["metrics"]["confusion_matrix"]), 10)
                    self.assertEqual(result["model_config"], first["config"]["model_config"])
                    # Restore RNG+sampler state, including dropout and mirroring:
                    # every resumed model must equal its continuous CPU run.
                    continuous = root / f"continuous-{name}"
                    train(build_parser().parse_args(common + ["--run-dir", str(continuous),
                        "--epochs", "2", "--workers", "0", "--batch-size", "10", "--no-pretrained"]))
                    uninterrupted = load_checkpoint(continuous / "checkpoints" / "last.pt")
                    for key, value in second["model_state"].items():
                        self.assertTrue(torch.equal(value, uninterrupted["model_state"][key]), key)

if __name__ == "__main__":
    unittest.main()
