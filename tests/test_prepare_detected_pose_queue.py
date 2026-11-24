"""Verify the detector comparison recipe and fail closed on unfinished caches."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.Dataset.annotations import CLASS_NAMES_10, sha256_file
from src.Dataset.splits import build_manifest
from src.Eval.pose_detection import detector_config
from src.Scripts.prepare_detected_pose_queue import (
    _partition_counts, _split_balance, main, prepare_queue, verify_cache)
from src.Train.queue_plan import prepare_plan


class DetectedPoseQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache_dir = self.root / "Data/pose_detections/full_dataset"
        self.cache_dir.mkdir(parents=True)
        self.annotations = self.root / "Data/annotations.json"
        self.split = self.root / "Data/split.json"
        self.pose = [[float(i), float(i + 1), .9] for i in range(17)]
        self.records = []
        for sequence in (1, 2, 3):
            for index, label in enumerate(CLASS_NAMES_10):
                position = label if label in ("5050_guard", "standing") else label + "1"
                for repeat in (0, 1):
                    self.records.append({"image": f"{sequence:02}{2 * index + repeat:05}",
                                         "frame": 2 * index + repeat, "position": position,
                                         "pose1": self.pose, "pose2": self.pose})
        # Extraction is complete even for originally ineligible supplied poses.
        self.records[21]["pose2"] = None
        self.annotations.write_text(json.dumps(self.records), encoding="utf-8")
        self.annotation_sha = sha256_file(self.annotations)
        groups = {"description": "Fixture recording identities are explicitly independent.",
                  "sequences": {f"{i:02}": {"recording": f"recording-{i}", "split": partition}
                                for i, partition in enumerate(("train", "val", "test"), 1)}}
        self.manifest = build_manifest(self.records, groups, self.annotation_sha)
        self.split.write_text(json.dumps(self.manifest), encoding="utf-8")
        for name in ("yolo", "vitpose"):
            self.write_cache(name)

    def write_cache(self, name, *, missing=(), single=()):
        path = self.cache_dir / f"{name}.jsonl"
        rows = [{"image": row["image"],
                 "poses": [] if row["image"] in missing else
                          [self.pose] if row["image"] in single else [self.pose, self.pose],
                 "image_sha256": "a" * 64, "seconds": .01} for row in self.records]
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        metadata = {"complete": True, "configuration": detector_config(name),
                    "population": "all_original_annotations", "images": len(rows),
                    "annotation_sha256": self.annotation_sha, "cache_sha256": sha256_file(path),
                    "coverage": {str(n): sum(len(row["poses"]) == n for row in rows) for n in range(3)}}
        path.with_suffix(".meta.json").write_text(json.dumps(metadata), encoding="utf-8")
        return path

    def prepare(self):
        return prepare_queue(self.cache_dir, self.annotations, self.split, project_root=self.root)

    def change_metadata(self, name, **updates):
        path = self.cache_dir / f"{name}.meta.json"
        metadata = json.loads(path.read_text())
        metadata.update(updates)
        path.write_text(json.dumps(metadata), encoding="utf-8")

    def test_recipe_pairs_detectors_and_recovers_missing_supplied_poses(self):
        queue, report = self.prepare()
        self.assertEqual(len(queue["jobs"]), 24)
        self.assertEqual(queue["defaults"]["pose_detections_scope"], "all")
        self.assertFalse(queue["defaults"]["pose_keep_annotated"])
        self.assertEqual({job["train"]["model"] for job in queue["jobs"][:6]},
                         {"pose", "pose-wide", "pose-attention"})
        for left, right in zip(queue["jobs"][::2], queue["jobs"][1::2]):
            self.assertEqual({k: v for k, v in left["train"].items() if k != "pose_detections"},
                             {k: v for k, v in right["train"].items() if k != "pose_detections"})
            self.assertTrue(left["id"].startswith("yolo26-"))
            self.assertTrue(right["id"].startswith("vitpose-"))
        for name in ("yolo", "vitpose"):
            counts = report["detectors"][name]["partitions"]["val"]
            self.assertEqual(counts["original"]["samples"], 20)
            self.assertEqual(counts["annotated_pose_baseline"]["samples"], 19)
            self.assertEqual(counts["eligible"]["samples"], 20)
            self.assertEqual(counts["removed_from_annotated_baseline"], 0)
            self.assertEqual(counts["added_to_annotated_baseline"]["samples"], 1)
            recovered = counts["recovered_missing_supplied_poses"]
            self.assertEqual(recovered["samples"], 1)
            self.assertEqual(recovered["class_counts"]["5050_guard"], 1)
            full = report["detectors"][name]["full_dataset_coverage"]
            self.assertEqual(full["supplied_two_athletes"]["samples"], 59)
            self.assertEqual(full["detected_two_athletes"]["samples"], 60)
            self.assertEqual(full["recovered_missing_supplied_poses"]["samples"], 1)
            self.assertEqual(full["recovered_by_assignment"]["val"]["samples"], 1)
        self.assertEqual(report["common_cohort"]["val"]["samples"], 20)
        self.assertTrue(report["common_cohort"]["val"]["detector_cohorts_match"])
        plan_path = self.root / "queue.json"
        plan_path.write_text(json.dumps(queue), encoding="utf-8")
        resolved = prepare_plan(plan_path, self.root, "check")
        self.assertEqual(len(resolved["jobs"]), 24)
        for job in resolved["jobs"]:
            self.assertIn("--no-pose-keep-annotated", job["command"])
            index = job["command"].index("--pose-detections")
            self.assertTrue(Path(job["command"][index + 1]).is_absolute())

    def test_preflight_reports_detector_failures_and_common_cohort(self):
        self.write_cache("yolo", missing=("0200000",))
        self.write_cache("vitpose", single=("0200002",))
        _, report = self.prepare()
        for name in ("yolo", "vitpose"):
            counts = report["detectors"][name]["partitions"]["val"]
            self.assertEqual(counts["eligible"]["samples"], 19)
            self.assertEqual(counts["removed_from_annotated_baseline"], 1)
            self.assertEqual(counts["added_to_annotated_baseline"]["samples"], 1)
            self.assertEqual(counts["removed_from_annotated_baseline_by_reason"],
                             {"missing_detected_pose": 1})
            balance = report["detectors"][name]["split_balance"]
            self.assertEqual(balance["overall"]["total"], 59)
            self.assertAlmostEqual(balance["overall"]["proportions"]["train"], 20 / 59)
            self.assertAlmostEqual(balance["overall"]["deviation_percentage_points"]["val"],
                                   100 * (19 / 59 - .1))
            self.assertAlmostEqual(sum(balance["overall"]["proportions"].values()), 1)
        common = report["common_cohort"]["val"]
        self.assertEqual(common["samples"], 18)
        self.assertFalse(common["detector_cohorts_match"])

    def test_report_separates_recovered_missing_pose_from_old_unpaired_view(self):
        original = self.records[20:23]
        assignments = dict(self.manifest["assignments"])
        assignments["0200001"] = assignments["0200002"] = "excluded"
        manifest = {**self.manifest, "pose_assignments": assignments,
                    "pose_exclusion_reasons": {"0200001": "missing_pose",
                                               "0200002": "no_paired_training_view"}}
        effective = {"pose_exclusion_reasons": {"0200000": "missing_detected_pose"}}
        eligible = [{**original[1], "pose2": self.pose}, original[2]]
        counts = _partition_counts(original, eligible, manifest, effective)
        self.assertEqual(counts["retained_from_annotated_baseline"], 0)
        self.assertEqual(counts["removed_from_annotated_baseline"], 1)
        self.assertEqual(counts["added_to_annotated_baseline"]["samples"], 2)
        self.assertEqual(counts["recovered_missing_supplied_poses"]["samples"], 1)
        self.assertEqual(counts["recovered_unpaired_supplied_poses"]["samples"], 1)
        self.assertEqual(counts["other_additions"]["samples"], 0)
        self.assertEqual(counts["eligible"]["samples"],
                         counts["annotated_pose_baseline"]["samples"]
                         - counts["removed_from_annotated_baseline"]
                         + counts["added_to_annotated_baseline"]["samples"])

    def test_absent_class_and_split_imbalance_warn_without_reallocating(self):
        self.write_cache("yolo", missing=("0200000", "0200001"))
        queue, report = self.prepare()
        self.assertEqual(len(queue["jobs"]), 24)
        detector = report["detectors"]["yolo"]
        self.assertEqual(detector["partitions"]["val"]["eligible"]["samples"], 18)
        balance = detector["split_balance"]
        self.assertEqual(balance["by_class"]["5050_guard"]["counts"],
                         {"train": 2, "val": 0, "test": 2})
        self.assertEqual(balance["by_class"]["5050_guard"]["proportions"],
                         {"train": .5, "val": 0, "test": .5})
        flags = balance["quality_flags"]
        self.assertTrue(any(flag["code"] == "absent_class" and flag["class"] == "5050_guard"
                            and flag["partitions"] == ["val"] for flag in flags))
        self.assertTrue(any(flag["code"] == "overall_split_deviation" for flag in flags))
        self.assertTrue(any(flag["code"] == "class_split_deviation" for flag in flags))
        self.assertEqual(json.loads(self.split.read_text()), self.manifest)

    def test_balanced_proportions_have_no_flags_and_thresholds_are_strict(self):
        def partitions(train, val, test):
            return {part: {"eligible": {"samples": count, "class_counts": {"guard": count}}}
                    for part, count in zip(("train", "val", "test"), (train, val, test))}
        balanced = _split_balance(partitions(80, 10, 10), ["guard"])
        self.assertEqual(balanced["quality_flags"], [])
        self.assertEqual(balanced["overall"]["proportions"], {"train": .8, "val": .1, "test": .1})
        self.assertEqual(_split_balance(partitions(75, 15, 10), ["guard"])["quality_flags"], [])
        boundary = _split_balance(partitions(70, 20, 10), ["guard"])
        self.assertTrue(boundary["quality_flags"])
        self.assertFalse(any(flag["code"] == "class_split_deviation"
                             for flag in boundary["quality_flags"]))
        absent = _split_balance(partitions(0, 0, 0), ["guard"])
        self.assertIsNone(absent["by_class"]["guard"]["proportions"]["train"])
        self.assertEqual(absent["quality_flags"][0]["code"], "absent_class")

    def test_rejects_unfinished_wrong_identity_and_stale_metadata(self):
        cases = [({"complete": False}, "unfinished"),
                 ({"annotation_sha256": "b" * 64}, "annotation_sha256"),
                 ({"configuration": {**detector_config("yolo"), "model": "yolo11x-pose.pt"}}, "configuration"),
                 ({"population": "partial"}, "population"),
                 ({"images": 59}, "images"),
                 ({"cache_sha256": "b" * 64}, "SHA-256")]
        for updates, message in cases:
            with self.subTest(updates=updates):
                self.write_cache("yolo")
                self.change_metadata("yolo", **updates)
                with self.assertRaisesRegex(ValueError, message):
                    self.prepare()

    def test_rejects_partial_duplicate_or_absent_rows_without_repairing_cache(self):
        for damage, message in (("tail", "incomplete"), ("duplicate", "duplicate"), ("missing", "incomplete")):
            with self.subTest(damage=damage):
                path = self.write_cache("yolo")
                data = path.read_bytes()
                if damage == "tail":
                    data = data[:-1]
                elif damage == "duplicate":
                    data += data.splitlines(keepends=True)[0]
                else:
                    data = b"".join(data.splitlines(keepends=True)[:-1])
                path.write_bytes(data)
                self.change_metadata("yolo", cache_sha256=sha256_file(path))
                with self.assertRaisesRegex(ValueError, message):
                    verify_cache(self.cache_dir, "yolo", self.records, self.annotation_sha)
                self.assertEqual(path.read_bytes(), data)

    def test_rejects_split_drift_and_empty_detected_partition(self):
        self.manifest["assignments"]["0100000"] = "test"
        self.split.write_text(json.dumps(self.manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Split assignments"):
            self.prepare()
        self.manifest["assignments"]["0100000"] = "train"
        self.split.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.write_cache("yolo", missing=tuple(row["image"] for row in self.records
                                               if row["image"].startswith("01")))
        with self.assertRaisesRegex(ValueError, "train is empty"):
            self.prepare()

    def test_cli_dry_run_then_save_and_refusal_leave_no_training_artifacts(self):
        output = self.root / "launchers/detected-poses.local.json"
        args = ["--annotations", str(self.annotations), "--split", str(self.split),
                "--cache-dir", str(self.cache_dir), "--output", str(output)]
        with patch("src.Scripts.prepare_detected_pose_queue.PROJECT", self.root), \
                patch("sys.stdout", new=io.StringIO()) as stdout:
            self.assertEqual(main([*args, "--dry-run"]), 0)
            self.assertFalse(output.exists())
            self.assertEqual(main(args), 0)
            self.assertIn("Start explicitly when ready", stdout.getvalue())
        self.assertEqual(len(json.loads(output.read_text())["jobs"]), 24)
        self.assertTrue(output.with_name("detected-poses.preflight.local.json").is_file())
        self.assertFalse((self.root / "trainings").exists())
        previous = output.read_bytes()
        self.change_metadata("vitpose", complete=False)
        with patch("sys.stderr", new=io.StringIO()) as stderr:
            self.assertEqual(main(args), 1)
            self.assertIn("unfinished", stderr.getvalue())
        self.assertEqual(output.read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
