"""Training pose models on poses from a detector's cache."""

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path

from src.Dataset.annotations import sha256_file
from src.Loader.pose import with_detected_poses
from src.Train.config import resolve_settings

POSE_A = [[1.0, 2.0, 0.9]] * 17
POSE_B = [[3.0, 4.0, 0.8]] * 17
ANNOTATED = [[5.0, 6.0, 1.0]] * 17


def record(image):
    return {"image": image, "position": "mount", "pose1": ANNOTATED, "pose2": ANNOTATED}


class DetectedPoseTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "cache.jsonl"
        rows = [{"image": "two", "poses": [POSE_A, POSE_B]}, {"image": "one", "poses": [POSE_A]},
                {"image": "none", "poses": []}, {"image": "other", "poses": [POSE_B]}]
        self.path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_replaces_poses_and_drops_frames_without_people(self):
        original = [record("two"), record("one"), record("none")]
        result = with_detected_poses(original, self.path)
        self.assertEqual([row["image"] for row in result], ["two", "one"])
        self.assertEqual((result[0]["pose1"], result[0]["pose2"]), (POSE_A, POSE_B))
        self.assertEqual((result[1]["pose1"], result[1]["pose2"]), (POSE_A, None))
        self.assertEqual(original[0]["pose1"], ANNOTATED)

    def test_keep_annotated_adds_detected_copies(self):
        result = with_detected_poses([record("two"), record("none")], self.path, keep_annotated=True)
        self.assertEqual([row["image"] for row in result], ["two", "none", "two"])
        self.assertEqual((result[0]["pose1"], result[2]["pose1"]), (ANNOTATED, POSE_A))

    def test_uncached_frame_is_an_error(self):
        with self.assertRaises(ValueError):
            with_detected_poses([record("absent")], self.path)

    def test_duplicate_ids_are_rejected_even_outside_requested_frames(self):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"image": "other", "poses": []}) + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate.*other"):
            with_detected_poses([record("two")], self.path)

    def test_malformed_rows_are_rejected(self):
        invalid_rows = [[], {"image": "", "poses": []}, {"image": "two"},
                        {"image": "two", "poses": [POSE_A] * 3},
                        {"image": "two", "poses": [POSE_A[:16]]},
                        {"image": "two", "poses": [[[1, 2]] * 17]},
                        {"image": "two", "poses": [[[True, 2, 1]] * 17]},
                        {"image": "two", "poses": [[[1, 2, float("nan")]] * 17]}]
        for row in invalid_rows:
            with self.subTest(row=row):
                self.path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Invalid detection cache row at line 1"):
                    with_detected_poses([record("two")], self.path)
        self.path.write_text("not json\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid detection cache JSON at line 1"):
            with_detected_poses([record("two")], self.path)

    def test_checksum_verifies_consumed_bytes(self):
        expected = sha256_file(self.path)
        self.assertEqual(len(with_detected_poses([record("two")], self.path,
                                               expected_sha256=expected)), 1)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"image": "new", "poses": []}) + "\n")
        with self.assertRaisesRegex(ValueError, "SHA-256 changed while loading"):
            with_detected_poses([record("two")], self.path, expected_sha256=expected)


class ConfigurationTests(unittest.TestCase):
    def test_settings(self):
        pose = lambda **values: argparse.Namespace(model="pose-wide", **values)
        self.assertIsNone(resolve_settings(pose())["pose_detections"])
        self.assertEqual(resolve_settings(pose(pose_detections="c.jsonl"))["pose_detections"],
                         str(Path("c.jsonl").resolve()))
        with self.assertRaises(ValueError):
            resolve_settings(pose(pose_keep_annotated=True))
        with self.assertRaises(ValueError):
            resolve_settings(argparse.Namespace(model="image", pose_detections="c.jsonl"))

    def test_resume_compares_resolved_cache_paths(self):
        previous = resolve_settings(argparse.Namespace(model="pose", pose_detections="c.jsonl"))
        for path in ("./c.jsonl", str(Path("c.jsonl").resolve())):
            restored = resolve_settings(argparse.Namespace(model="pose", pose_detections=path), previous)
            self.assertEqual(restored["pose_detections"], previous["pose_detections"])
        with self.assertRaisesRegex(ValueError, "Cannot change --pose-detections"):
            resolve_settings(argparse.Namespace(model="pose", pose_detections="other.jsonl"), previous)

    def test_annotation_checkpoint_without_detection_settings_stays_compatible(self):
        previous = resolve_settings(argparse.Namespace(model="pose"))
        for name in ("pose_detections", "pose_keep_annotated", "pose_joint_noise",
                     "pose_joint_dropout", "pose_athlete_dropout"):
            previous.pop(name)
        restored = resolve_settings(argparse.Namespace(model="pose"), previous)
        self.assertIsNone(restored["pose_detections"])
        self.assertFalse(restored["pose_keep_annotated"])
        self.assertEqual(restored["pose_joint_noise"], 0.0)


@unittest.skipUnless(os.environ.get("JIUJITSU_RUN_TORCH_TESTS") == "1", "opt-in tiny CPU training")
class DetectionResumeIntegrationTests(unittest.TestCase):
    def setUp(self):
        import torch
        from tests.test_training import synthetic_files

        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.annotations, self.split, _, records = synthetic_files(self.root)
        self.cache = self.root / "detections.jsonl"
        self.rows = [{"image": row["image"], "poses": [row["pose1"], row["pose2"]]}
                     for row in records if row["image"].startswith("01")]
        self.write_cache()
        self.run_dir = self.root / "run"
        self.last = self.run_dir / "checkpoints" / "last.pt"

    def write_cache(self):
        self.cache.write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")

    def train(self, *, resume=False, extra=()):
        from src.Scripts.train import build_parser
        from src.Train.engine import train

        flags = (["--resume", str(self.last), "--epochs", "2"] if resume else
                 ["--model", "pose", "--annotations", str(self.annotations), "--split", str(self.split),
                  "--run-dir", str(self.run_dir), "--pose-detections", str(self.cache),
                  "--epochs", "1", "--batch-size", "10", "--workers", "0"])
        return train(build_parser().parse_args([*flags, "--device", "cpu", "--no-progress", *extra]))

    def test_changed_cache_cannot_resume_or_modify_run(self):
        from src.Train.checkpoints import load_checkpoint

        self.train()
        before = load_checkpoint(self.last)
        metrics = self.run_dir / "diagnostics" / "metrics.jsonl"
        saved_metrics = metrics.read_bytes()
        self.rows[0]["poses"][0][0][0] += 100
        self.write_cache()
        with self.assertRaisesRegex(ValueError, "detection cache SHA-256 does not match"):
            self.train(resume=True)
        self.assertEqual(load_checkpoint(self.last)["epoch"], 1)
        self.assertEqual(load_checkpoint(self.last)["config"]["pose_detections_sha256"],
                         before["config"]["pose_detections_sha256"])
        self.assertEqual(metrics.read_bytes(), saved_metrics)
        self.assertFalse(list(self.run_dir.glob("config_resume*")))

    def test_same_cache_relative_path_resumes_with_preserved_hash(self):
        from src.Train.checkpoints import load_checkpoint

        self.train()
        expected = sha256_file(self.cache)
        self.train(resume=True, extra=["--pose-detections", os.path.relpath(self.cache)])
        resumed = load_checkpoint(self.last)
        self.assertEqual(resumed["epoch"], 2)
        self.assertEqual(resumed["config"]["pose_detections_sha256"], expected)
        self.assertEqual(resumed["config"]["pose_detections"], str(self.cache.resolve()))

    def test_cache_checkpoint_without_saved_checksum_cannot_resume(self):
        import torch
        from src.Train.checkpoints import load_checkpoint

        self.train()
        checkpoint = load_checkpoint(self.last)
        del checkpoint["config"]["pose_detections_sha256"]
        torch.save(checkpoint, self.last)
        with self.assertRaisesRegex(ValueError, "missing pose_detections_sha256"):
            self.train(resume=True)

    def test_mixed_training_counts_include_actual_loader_population(self):
        from src.Train.checkpoints import load_checkpoint

        self.rows[0]["poses"] = []
        self.write_cache()
        self.train(extra=["--pose-keep-annotated"])
        counts = load_checkpoint(self.last)["config"]["sample_counts"]
        self.assertEqual(counts["train"]["eligible"], 10)
        self.assertEqual(counts["train"]["samples"], 19)
        self.assertEqual(counts["train"]["annotated_samples"], 10)
        self.assertEqual(counts["train"]["detected_samples"], 9)
        self.assertEqual(counts["train"]["frames_without_detection"], 1)
        self.assertEqual(counts["val"]["samples"], 10)
        self.assertEqual(counts["test"]["samples"], 10)

    def test_empty_detection_training_is_rejected_before_run_creation(self):
        for row in self.rows:
            row["poses"] = []
        self.write_cache()
        with self.assertRaisesRegex(ValueError, "No training samples remain"):
            self.train()
        self.assertFalse(self.run_dir.exists())


if __name__ == "__main__":
    unittest.main()
