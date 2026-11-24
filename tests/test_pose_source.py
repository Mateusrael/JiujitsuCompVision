"""Detected inputs preserve split membership and held-out-view meaning."""

import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.Dataset.annotations import sha256_file
from src.Loader.pose_source import resolve_pose_data
from src.Loader.splits import filter_pose_records, split_records
from src.Train.config import resolve_settings


POSE = [[float(joint), 2.0, .8] for joint in range(17)]
ZERO = [[0., 0., 0.] for _ in range(17)]


class PoseSourceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cache = Path(temporary.name) / "detected.jsonl"
        # Source 01 has an offset and twice source 00's frame rate.
        ids = ["0000100", "0100210", "0000200", "0100410", "0000300", "0100610",
               "0000400", "0000500", "0000600", "0000700", "0000800"]
        self.records = [{"image": image, "frame": int(image[2:]), "position": "mount1",
                         "pose1": copy.deepcopy(POSE), "pose2": copy.deepcopy(POSE)}
                        for image in ids]
        self.records[-2]["pose2"] = None
        parts = ("train", "val", "train", "val", "train", "test", "val", "val",
                 "val", "train", "excluded")
        assignments = dict(zip(ids, parts))
        pose_assignments = dict(assignments)
        pose_assignments[ids[-2]] = "excluded"
        self.manifest = {
            "split_method": "multiview-sections", "assignments": assignments,
            "pose_assignments": pose_assignments, "pose_exclusion_reasons": {},
            "class_names": ["mount"], "label_map": {"mount1": "mount"},
            "evaluation_types": {image: "held_out_view" if image.startswith("01")
                                 else "unseen_moment" for image, part in assignments.items()
                                 if part in ("val", "test")},
            "section_plan": {"paired_view_tolerance_frames": 5,
                             "groups": [{"id": "g", "sources": ["00", "01"],
                                         "alignment": {"00": {"offset": 0, "scale": 1},
                                                       "01": {"offset": 10, "scale": 2}}}]}}
        self.rows = [{"image": image, "poses": [copy.deepcopy(POSE), copy.deepcopy(POSE)]}
                     for image in ids]
        # The second held-out view loses its only training witness.
        self.rows[2]["poses"] = [POSE]
        self.rows[7]["poses"] = []
        self.rows[8]["poses"] = [POSE, ZERO]
        self.config = {"model": "pose", "pose_detections_scope": "all",
                       "pose_detections": str(self.cache)}
        self.write_cache()

    def write_cache(self):
        self.cache.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        self.config["pose_detections_sha256"] = sha256_file(self.cache)

    def test_all_splits_use_detected_inputs_and_keep_original_boundaries(self):
        original_records, original_manifest = copy.deepcopy(self.records), copy.deepcopy(self.manifest)
        self.rows[0]["poses"][0][0] = [111., 222., .3]
        self.write_cache()
        records, manifest = resolve_pose_data(self.records, self.manifest, self.config)
        self.assertEqual(len(records), len(self.records))
        self.assertEqual(records[0]["pose1"][0], [111., 222., .3])
        self.assertEqual(manifest["assignments"], original_manifest["assignments"])
        actual = {part: [r["image"] for r in filter_pose_records(
                    split_records(records, manifest, part), manifest=manifest)]
                  for part in ("train", "val", "test")}
        self.assertEqual(actual, {"train": ["0000100", "0000300", "0000700"],
                                  "val": ["0100210", "0000400"], "test": ["0100610"]})
        self.assertEqual(manifest["pose_exclusion_reasons"]["0100410"],
                         "no_detected_paired_training_view")
        self.assertEqual(manifest["pose_exclusion_reasons"]["0000600"], "missing_detected_pose")
        self.assertEqual(self.records, original_records)
        self.assertEqual(self.manifest, original_manifest)

    def test_recovered_training_pose_restores_held_out_view_without_moving_partitions(self):
        # Both the training witness and validation pose were absent in supplied annotations.
        for index in (0, 1):
            self.records[index]["pose2"] = None
            image = self.records[index]["image"]
            self.manifest["pose_assignments"][image] = "excluded"
            self.manifest["pose_exclusion_reasons"][image] = "missing_pose"
        # The test pose was complete but lacked an eligible supplied training view.
        self.manifest["pose_assignments"]["0100610"] = "excluded"
        self.manifest["pose_exclusion_reasons"]["0100610"] = "no_paired_training_view"
        records, manifest = resolve_pose_data(self.records, self.manifest, self.config)
        self.assertEqual(manifest["pose_assignments"]["0000100"], "train")
        self.assertEqual(manifest["pose_assignments"]["0100210"], "val")
        self.assertEqual(manifest["pose_assignments"]["0100610"], "test")
        self.assertNotIn("0100210", manifest["pose_exclusion_reasons"])
        self.assertNotIn("0100610", manifest["pose_exclusion_reasons"])
        # Even two perfect detections cannot restore an image-level buffer/exclusion.
        self.assertEqual(manifest["pose_assignments"]["0000800"], "excluded")
        self.assertEqual(manifest["assignments"], self.manifest["assignments"])
        eligible = filter_pose_records(records, manifest=manifest)
        self.assertIn("0100210", [row["image"] for row in eligible])

    def test_different_class_does_not_supply_a_paired_witness(self):
        self.records[0]["position"] = "back1"
        self.manifest["label_map"]["back1"] = "back"
        _, manifest = resolve_pose_data(self.records, self.manifest, self.config)
        self.assertEqual(manifest["pose_assignments"]["0100210"], "excluded")

    def test_cache_identity_and_completeness_are_required(self):
        with self.cache.open("a") as handle:
            handle.write(json.dumps({"image": "9999999", "poses": []}) + "\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            resolve_pose_data(self.records, self.manifest, self.config)
        self.rows.pop()
        self.write_cache()
        with self.assertRaisesRegex(ValueError, "no cached detection"):
            resolve_pose_data(self.records, self.manifest, self.config)
        del self.config["pose_detections_sha256"]
        with self.assertRaisesRegex(ValueError, "pose_detections_sha256"):
            resolve_pose_data(self.records, self.manifest, self.config)

    def test_legacy_inputs_do_not_require_training_cache_at_evaluation(self):
        for scope in ({}, {"pose_detections_scope": "train"}):
            rows, manifest = resolve_pose_data(self.records, self.manifest,
                                               {"pose_detections": "missing.jsonl", **scope})
            self.assertIs(rows, self.records)
            self.assertIs(manifest, self.manifest)

    def test_scope_is_immutable_on_resume_and_does_not_mix_sources(self):
        initial = resolve_settings(argparse.Namespace(model="pose", pose_detections="cache.jsonl",
                                                       pose_detections_scope="all"))
        self.assertEqual(resolve_settings(argparse.Namespace(model="pose"), initial)
                         ["pose_detections_scope"], "all")
        with self.assertRaisesRegex(ValueError, "Cannot change"):
            resolve_settings(argparse.Namespace(model="pose", pose_detections_scope="train"), initial)
        for values in ({"pose_detections_scope": "all"},
                       {"pose_detections_scope": "all", "pose_detections": "x",
                        "pose_keep_annotated": True}):
            with self.assertRaises(ValueError):
                resolve_settings(argparse.Namespace(model="pose", **values))


@unittest.skipUnless(os.environ.get("JIUJITSU_RUN_TORCH_TESTS") == "1", "opt-in CPU training")
class DetectedSourceIntegrationTests(unittest.TestCase):
    def test_train_validation_and_test_consume_same_detected_source(self):
        import torch
        from src.Eval.evaluate import evaluate
        from src.Scripts.evaluate import build_parser as evaluate_parser
        from src.Scripts.train import build_parser
        from src.Train.checkpoints import load_checkpoint
        from src.Train.engine import make_dataset, train
        from tests.test_training import synthetic_files

        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations, split, _, original = synthetic_files(root)
            from src.Dataset.splits import build_manifest
            recording_groups = json.loads(split.read_text())["recording_groups"]
            # Recover one originally incomplete supplied pose in every partition.
            for index in (1, 11, 21):
                original[index]["pose2"] = None
            annotations.write_text(json.dumps(original))
            split.write_text(json.dumps(build_manifest(original, recording_groups,
                                                       sha256_file(annotations))))
            cache = root / "poses.jsonl"
            rows = [{"image": r["image"], "poses": [POSE, [[50., 90., .2]] * 17]}
                    for r in original]
            for index in (0, 10, 20):
                rows[index]["poses"] = [POSE]
            cache.write_text("".join(json.dumps(row) + "\n" for row in rows))
            run = root / "run"
            captured = []

            def capture(model, records, config, **kwargs):
                captured.append(copy.deepcopy(records))
                return make_dataset(model, records, config, **kwargs)

            with patch("src.Train.engine.make_dataset", side_effect=capture):
                train(build_parser().parse_args([
                    "--model", "pose", "--annotations", str(annotations), "--split", str(split),
                    "--pose-detections", str(cache), "--pose-detections-scope", "all",
                    "--run-dir", str(run), "--epochs", "1", "--workers", "0", "--batch-size", "9",
                    "--device", "cpu", "--no-progress"]))
            checkpoint = run / "checkpoints" / "best.pt"
            config = load_checkpoint(checkpoint)["config"]
            for part in ("train", "val", "test"):
                self.assertEqual(config["sample_counts"][part]["samples"], 9)
                self.assertEqual(config["sample_counts"][part]["removed_by_reason"],
                                 {"missing_detected_pose": 1})
            args = evaluate_parser().parse_args(["--checkpoint", str(checkpoint), "--device", "cpu",
                                                 "--no-progress"])
            with patch("src.Eval.evaluate.make_dataset", side_effect=capture):
                result = evaluate(args)
            self.assertEqual(result["metrics"]["samples"], 9)
            self.assertEqual(result["sample_counts"], config["sample_counts"]["test"])
            self.assertEqual(result["evaluation_pose_source"], "detected")
            for population in captured:
                self.assertEqual(len(population), 9)
                for row in population:
                    self.assertEqual(row["pose1"], POSE)
                    self.assertEqual(row["pose2"], [[50., 90., .2]] * 17)
            with cache.open("a") as handle:
                handle.write(json.dumps({"image": "9999999", "poses": []}) + "\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                evaluate(args)
