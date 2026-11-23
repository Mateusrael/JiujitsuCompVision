"""Pose comparisons must verify each checkpoint's own split before inference."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from experiments import compare_pose_detectors as comparison
from src.Dataset.annotations import sha256_file
from src.Dataset.splits import build_manifest
from tests.test_training import synthetic_files


class _SplitFixture:
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.annotations, initial_split, _, self.records = synthetic_files(self.root)
        self.split_dir = self.root / "Data" / "splits"
        self.split_dir.mkdir(parents=True)
        self.base = self.split_dir / "multiview_sections.json"
        self.base.write_bytes(initial_split.read_bytes())
        self.shifted = self.split_dir / "multiview_sections_shifted.json"
        groups = {
            "description": "Synthetic second split with exchanged validation and test recordings.",
            "sequences": {
                "01": {"recording": "synthetic-train", "split": "train"},
                "02": {"recording": "synthetic-test", "split": "test"},
                "03": {"recording": "synthetic-val", "split": "val"},
            },
        }
        manifest = build_manifest(self.records, groups, sha256_file(self.annotations))
        self.shifted.write_text(json.dumps(manifest), encoding="utf-8")
        self.config = {"split": str(self.shifted), "split_sha256": sha256_file(self.shifted)}


class SplitResolutionTests(_SplitFixture, unittest.TestCase):
    def test_saved_shifted_split_takes_priority_over_old_default(self):
        with patch.object(comparison, "ROOT", self.root):
            selected = comparison.resolve_split_file(self.config)
        self.assertEqual(selected, self.shifted.resolve())
        self.assertNotEqual(sha256_file(selected), sha256_file(self.base))

    def test_relocated_split_is_found_by_exact_hash_not_filename(self):
        config = {**self.config, "split": str(self.root / "unavailable" / self.shifted.name)}
        relocated = self.split_dir / "relocated-copy.json"
        self.shifted.rename(relocated)
        with patch.object(comparison, "ROOT", self.root):
            self.assertEqual(comparison.resolve_split_file(config), relocated.resolve())
        self.assertEqual(comparison.resolve_split_file(config, split_dir=self.split_dir),
                         relocated.resolve())

    def test_missing_saved_split_does_not_fall_back_to_wrong_default(self):
        config = {**self.config, "split": str(self.root / "unavailable.json")}
        self.shifted.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "exact manifest used for training"):
            comparison.resolve_split_file(config, split_dir=self.split_dir)


@unittest.skipUnless(os.environ.get("JIUJITSU_RUN_TORCH_TESTS") == "1",
                     "Opt-in checkpoint data validation")
class CheckpointValidationTests(_SplitFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        from src.Train.config import resolve_settings
        from src.Train.engine import configured_architecture

        manifest = json.loads(self.shifted.read_text(encoding="utf-8"))
        self.config.update(resolve_settings(argparse.Namespace(model="pose")))
        self.config.update({
            "model": "pose", "annotations": str(self.annotations),
            "annotation_sha256": sha256_file(self.annotations), "pose_filter": "both_present",
            "class_names": manifest["class_names"], "label_map": manifest["label_map"],
        })
        self.config["model_config"] = configured_architecture("pose", len(manifest["class_names"]),
                                                              self.config)

    def test_verified_data_selects_saved_validation_population(self):
        with patch.object(comparison, "ROOT", self.root):
            data = comparison.verified_data(self.config)
        self.assertEqual(data.split_path, self.shifted.resolve())
        selected = data.select("pose", split="val")
        self.assertEqual(len(selected), 10)
        self.assertTrue(all(row["image"].startswith("03") for row in selected))

    def test_relocated_manifest_passes_real_checkpoint_validation(self):
        config = {**self.config, "split": str(self.root / "missing" / self.shifted.name)}
        relocated = self.split_dir / "local-snapshot.json"
        self.shifted.rename(relocated)
        with patch.object(comparison, "ROOT", self.root):
            data = comparison.verified_data(config)
        self.assertEqual(data.split_path, relocated.resolve())
        self.assertEqual(data.split_sha256, self.config["split_sha256"])

    def test_wrong_explicit_split_fails_before_classifier_or_detector(self):
        checkpoint = self.root / "best.pt"
        checkpoint.write_bytes(b"checkpoint deserialization is stubbed; metadata validation is real")
        external = types.ModuleType("src.Eval.external")
        external.ExternalClassifier = Mock(side_effect=AssertionError("model must not load"))
        external.compare_pose_sources = Mock(side_effect=AssertionError("detectors must not run"))
        external.score_predictions = Mock()
        with (patch.dict(sys.modules, {"src.Eval.external": external}),
              patch("src.Train.checkpoints.load_checkpoint", return_value={"config": self.config}),
              patch.object(comparison, "detect_records", side_effect=AssertionError("no detections")) as detect):
            with self.assertRaisesRegex(ValueError, "split_sha256 does not match"):
                comparison.main([
                    "--checkpoint", str(checkpoint), "--split-file", str(self.base),
                    "--device", "cpu", "--output-dir", str(self.root / "report"), "--no-progress",
                ])
        external.ExternalClassifier.assert_not_called()
        external.compare_pose_sources.assert_not_called()
        detect.assert_not_called()


class LightweightCLITests(unittest.TestCase):
    def invoke_without_site_packages(self, *flags):
        return subprocess.run([sys.executable, "-B", "-S", str(Path(comparison.__file__).resolve()),
                               *flags], capture_output=True, text=True, check=False)

    def test_help_works_without_torch_or_detector_packages(self):
        result = self.invoke_without_site_packages("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--split-file", result.stdout)
        self.assertIn("--detection-device", result.stdout)

    def test_zero_and_negative_sizes_fail_before_importing_dependencies(self):
        for flag in ("--per-class", "--batch-size", "--limit"):
            for value in ("0", "-1"):
                with self.subTest(flag=flag, value=value):
                    result = self.invoke_without_site_packages("--checkpoint", "unused.pt", flag, value)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("must be positive", result.stderr)
                    self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_conflicting_sampling_options_fail_without_dependencies(self):
        for flags in (("--all-frames", "--holdout"), ("--all-frames", "--limit", "1")):
            with self.subTest(flags=flags):
                result = self.invoke_without_site_packages("--checkpoint", "unused.pt", *flags)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("cannot be combined", result.stderr)


if __name__ == "__main__":
    unittest.main()
