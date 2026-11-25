"""External image identity and paired detector evaluation, without pretrained downloads."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.Dataset.annotations import CLASS_NAMES_10, LABEL_MAP_10, load_annotations, sha256_file
from src.Eval.external import ExternalData, canonical_label, compare_pose_sources, score_predictions


def external_fixture(root, *, missing=True):
    from PIL import Image

    images = root / "images"
    (images / "half_guard").mkdir(parents=True)
    records = []
    for i in range(4):
        name = f"half_guard/sample_{i}.png"
        Image.new("RGBA", (31 + i, 18), (20 + i * 10, 80, 130, 255)).save(images / name)
        records.append({"image": name, "frame": i,
                        "position": "half_guard1" if i % 2 else "half_guard",
                        "pose1": [[float(j + i), float(j), 0.9] for j in range(17)],
                        "pose2": None if missing and i == 3 else
                        [[float(j + 30), float(j + 5), 0.8] for j in range(17)]})
    annotations = root / "sample_annotations.json"
    annotations.write_text(json.dumps(records), encoding="utf-8")
    metadata = {"samples": 4, "poses": {"origin": "detector_generated",
                 "independently_verified_keypoint_ground_truth": False}}
    (root / "sample_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return annotations, images, records


@unittest.skipUnless(importlib.util.find_spec("PIL"), "requires Pillow")
class ExternalDataTests(unittest.TestCase):
    def test_external_paths_and_labels_are_separate_from_training_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            annotations, images, records = external_fixture(Path(directory))
            data = ExternalData(annotations, images)
            self.assertEqual(data.records, records)
            self.assertEqual(data.image_path(records[0]), (images / records[0]["image"]).resolve())
            self.assertEqual(canonical_label(records[0]), canonical_label(records[1]))
            self.assertEqual(data.describe()["pose_eligible"], 3)
            self.assertEqual(data.annotation_sha256, sha256_file(annotations))
            self.assertIn("detector_generated", data.reference_kind)
            self.assertIsNotNone(data.metadata_sha256)
            with self.assertRaisesRegex(ValueError, "seven-digit"):
                load_annotations(annotations)

    def test_rejects_unsafe_paths_duplicate_ids_invalid_labels_and_nonfinite_poses(self):
        with tempfile.TemporaryDirectory() as directory:
            annotations, images, records = external_fixture(Path(directory))
            bad_rows = []
            for path in ("../outside.png", "/outside.png", "C:/outside.png", "folder\\a.png",
                         "half_guard/../sample_0.png", "./sample.png", "half_guard//sample_0.png"):
                bad_rows.append([{**records[0], "image": path}, *records[1:]])
            bad_rows.append([records[0], records[0], *records[2:]])
            bad_rows.append([{**records[0], "position": "unknown"}, *records[1:]])
            bad_rows.append([{**records[0], "pose1": [[0, 0, float("nan")]] * 17}, *records[1:]])
            bad_rows.append([{**records[0], "pose1": []}, *records[1:]])
            for rows in bad_rows:
                with self.subTest(rows=rows[0]["image"]):
                    annotations.write_text(json.dumps(rows), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        ExternalData(annotations, images)

    def test_hashes_missing_images_and_duplicate_pixels_are_checked(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            annotations, images, records = external_fixture(Path(directory))
            data = ExternalData(annotations, images)
            records[0]["image_sha256"] = "0" * 64
            annotations.write_text(json.dumps(records), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "image_sha256"):
                ExternalData(annotations, images)
            del records[0]["image_sha256"]
            annotations.write_text(json.dumps(records), encoding="utf-8")
            original = images / records[0]["image"]
            Image.new("RGB", (10, 20), (0, 0, 0)).save(original)
            with self.assertRaisesRegex(ValueError, "changed since validation"):
                data.image_path(records[0])
            with Image.open(original) as image:
                image.save(images / records[1]["image"], compress_level=0)
            with self.assertRaisesRegex(ValueError, "Duplicate image content"):
                ExternalData(annotations, images)
            original.unlink()
            with self.assertRaises(FileNotFoundError):
                ExternalData(annotations, images)


class StubClassifier:
    model_name = "pose"
    class_names = CLASS_NAMES_10

    def predict(self, records, image_path, **kwargs):
        rows = []
        for record in records:
            actual = canonical_label(record)
            predicted = actual if record.get("pose2") else "mount"
            probabilities = [float(label == predicted) for label in CLASS_NAMES_10]
            rows.append({"image": record["image"], "actual": actual,
                         "predicted": predicted, "probabilities": probabilities})
        return rows


@unittest.skipUnless(importlib.util.find_spec("PIL"), "requires Pillow")
class PairedPoseComparisonTests(unittest.TestCase):
    def test_filtered_cache_reports_coverage_and_paired_scores_without_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            annotations, images, _ = external_fixture(Path(directory))
            data = ExternalData(annotations, images)
            for accepted_count in (2, 0):
                found = {}
                for index, row in enumerate(data.records):
                    accepted = index < accepted_count
                    found[row["image"]] = {
                        "image": row["image"], "poses": [row["pose1"], row["pose2"]] if accepted else [],
                        "image_sha256": sha256_file(data.image_path(row)), "seconds": .1,
                        "quality_filter": {"accepted": accepted, "reason": "accepted" if accepted else "unreliable_pose"},
                    }
                with (patch("src.Eval.pose_cache_filter.read_verified_cache_subset",
                            return_value=(found, {"postprocessing": {"profile": "cached-pose-quality-v1"}})),
                      patch("src.Eval.pose_detection.detect_records", side_effect=AssertionError("no extraction"))):
                    report = compare_pose_sources(
                        StubClassifier(), data.records, data.image_path, detector_names=["yolo"],
                        cache_dir=directory, detection_device="cpu", progress=False, cache_only=True)
                detector = report["detectors"]["yolo"]
                self.assertTrue(report["cache_only"])
                self.assertEqual(detector["all_frames"]["metrics"]["samples"], 4)
                self.assertEqual(len(detector["accepted_frames"]["predictions"]), accepted_count)
                audit = detector["quality_filter"]
                self.assertEqual(audit["accepted_images"], accepted_count)
                self.assertEqual(audit["rejected_images"], 4 - accepted_count)
                self.assertEqual(audit["rejection_reasons"], {"unreliable_pose": 4 - accepted_count})
                paired = detector["same_two_detected"]
                self.assertEqual(paired["annotated"], paired["detected"])
                self.assertEqual(len(paired["image_ids"]), accepted_count)
                if not accepted_count:
                    self.assertIsNone(detector["accepted_frames"]["metrics"])

    def test_cache_only_refuses_changed_image_without_calling_detector(self):
        with tempfile.TemporaryDirectory() as directory:
            annotations, images, _ = external_fixture(Path(directory))
            data = ExternalData(annotations, images)
            found = {r["image"]: {"image_sha256": "0" * 64} for r in data.records}
            with (patch("src.Eval.pose_cache_filter.read_verified_cache_subset", return_value=(found, {})),
                  patch("src.Eval.pose_detection.detect_records", side_effect=AssertionError("no extraction")),
                  self.assertRaisesRegex(ValueError, "image content differs")):
                compare_pose_sources(StubClassifier(), data.records, data.image_path, detector_names=["yolo"],
                                     cache_dir=directory, detection_device="cpu", progress=False, cache_only=True)

    def test_missing_detections_are_scored_and_comparison_cohorts_are_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            annotations, images, _ = external_fixture(Path(directory))
            data = ExternalData(annotations, images)
            seen = []

            def detections(name, records, image_path, cache_dir, device, progress=True):
                seen.append([row["image"] for row in records])
                counts = [2, 1, 0, 0] if name == "yolo" else [2, 0, 2, 0]
                result = {}
                for row, count in zip(records, counts):
                    # Swapped athletes must have zero localization error, while inference
                    # still uses the detector's ordering rather than peeking at reference.
                    poses = [row["pose2"], row["pose1"]][:count]
                    result[row["image"]] = {"image": row["image"], "poses": poses,
                                            "seconds": 0.1, "image_sha256": "0" * 64}
                return result

            with patch("src.Eval.pose_detection.detect_records", side_effect=detections):
                report = compare_pose_sources(
                    StubClassifier(), data.records, data.image_path, detector_names=["yolo", "vitpose"],
                    cache_dir=Path(directory) / "cache", detection_device="cpu", progress=False)
            ids = [row["image"] for row in data.records]
            self.assertEqual(seen, [ids, ids])
            self.assertEqual(report["population"]["excluded_missing_reference_pose"], 1)
            self.assertEqual(report["baseline"]["metrics"]["samples"], 3)
            yolo = report["detectors"]["yolo"]
            self.assertEqual(yolo["coverage"], {"0": 2, "1": 1, "2": 1})
            self.assertEqual(yolo["all_frames"]["metrics"]["samples"], 4)
            self.assertAlmostEqual(yolo["all_frames"]["metrics"]["accuracy"], 1 / 4)
            self.assertEqual(yolo["same_reference_frames"]["metrics"]["samples"], 3)
            self.assertAlmostEqual(yolo["same_reference_frames"]["metrics"]["accuracy"], 1 / 3)
            self.assertEqual([p["image"] for p in yolo["same_reference_frames"]["predictions"]],
                             [p["image"] for p in report["baseline"]["predictions"]])
            self.assertEqual([p["image"] for p in yolo["all_frames"]["predictions"]], ids)
            self.assertEqual(yolo["same_two_detected"]["image_ids"], ids[:1])
            self.assertEqual(yolo["same_two_detected"]["annotated"]["metrics"]["samples"], 1)
            self.assertEqual(yolo["same_two_detected"]["detected"]["metrics"]["samples"], 1)
            self.assertEqual(yolo["localization"]["mean_normalized_joint_error"], 0)
            self.assertEqual(yolo["cache_provenance"]["selected_image_ids"], ids)
            self.assertEqual(len(yolo["cache_provenance"]["selected_rows_sha256"]), 64)
            common = report["common_two_detected"]
            self.assertEqual(common["image_ids"], ids[:1])
            for result in (common["annotated"], *common["detectors"].values()):
                self.assertEqual(result["metrics"]["samples"], 1)

    def test_zero_confidence_poses_do_not_count_as_success_and_empty_subsets_are_null(self):
        with tempfile.TemporaryDirectory() as directory:
            annotations, images, _ = external_fixture(Path(directory))
            data = ExternalData(annotations, images)
            empty = {r["image"]: {"poses": [[[0, 0, 0]] * 17, [[0, 0, 0]] * 17]}
                     for r in data.records}
            with patch("src.Eval.pose_detection.detect_records", return_value=empty):
                report = compare_pose_sources(
                    StubClassifier(), data.records, data.image_path, detector_names=["yolo"],
                    cache_dir=Path(directory), detection_device="cpu", progress=False)
            self.assertEqual(report["detectors"]["yolo"]["coverage"], {"0": 4, "1": 0, "2": 0})
            self.assertIsNone(report["common_two_detected"]["annotated"]["metrics"])
            self.assertEqual(report["detectors"]["yolo"]["all_frames"]["metrics"]["samples"], 4)
            predictions = report["baseline"]["predictions"]
            with self.assertRaisesRegex(ValueError, "IDs/order"):
                score_predictions(data.records[:3], predictions[::-1], CLASS_NAMES_10)

    def test_images_without_any_reference_poses_still_get_detector_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            annotations, images, _ = external_fixture(Path(directory))
            data = ExternalData(annotations, images)
            records = [{**row, "pose1": None, "pose2": None} for row in data.records]
            detections = {r["image"]: {"image": r["image"], "poses": [r["pose1"], r["pose1"]]}
                          for r in data.records}
            with patch("src.Eval.pose_detection.detect_records", return_value=detections):
                report = compare_pose_sources(
                    StubClassifier(), records, data.image_path, detector_names=["yolo"],
                    cache_dir=Path(directory), detection_device="cpu", progress=False)
            self.assertIsNone(report["baseline"]["metrics"])
            self.assertEqual(report["population"]["detector_all_frames"], 4)
            self.assertEqual(report["detectors"]["yolo"]["all_frames"]["metrics"]["samples"], 4)
            self.assertEqual(report["detectors"]["yolo"]["coverage"], {"0": 0, "1": 0, "2": 4})
            self.assertEqual(report["common_two_detected"]["image_ids"], [])


@unittest.skipUnless(os.environ.get("JIUJITSU_RUN_TORCH_TESTS") == "1", "opt-in torch integration")
class ExternalInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    @staticmethod
    def checkpoint(root, name, factory=None):
        import torch
        from src.Modules.models import build_model
        from src.Train.config import DEFAULTS, ATTENTION_DEFAULTS
        from src.Train.engine import configured_architecture

        config = {**DEFAULTS, **ATTENTION_DEFAULTS, "model": name, "dropout": 0.8,
                  "image_training": "head", "class_names": CLASS_NAMES_10, "label_map": LABEL_MAP_10,
                  "annotation_sha256": "a" * 64, "split_sha256": "b" * 64,
                  "pose_filter": "both_present", "horizontal_flip_prob": 1.0,
                  "attention_dim": 32, "attention_heads": 2, "attention_layers": 1,
                  "attention_mlp_dim": 64}
        config["model_config"] = configured_architecture(name, 10, config)
        model = (factory or build_model)(config["model_config"], pretrained=False).eval()
        target = root / (name + ".pt")
        torch.save({"schema_version": 1, "config": config, "model_state": model.state_dict(),
                    "optimizer_state": {}, "epoch": 12, "best_macro_f1": 0.5,
                    "random_state": {}}, target)
        return target, model, config

    def test_all_pose_models_restore_annotation_trained_weights_without_external_hash_matching(self):
        import torch
        from src.Eval.external import ExternalClassifier
        from src.Loader.pose import pose_features
        from src.Train.engine import validate_checkpoint_data

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations, images, _ = external_fixture(root)
            data = ExternalData(annotations, images)
            for name in ("pose", "pose-wide", "pose-attention"):
                with self.subTest(model=name):
                    path, model, config = self.checkpoint(root, name)
                    digest = sha256_file(path)
                    classifier = ExternalClassifier(path, device="cpu")
                    try:
                        predictions = classifier.predict(data.records, data.image_path, batch_size=2, progress=False)
                        inputs = torch.tensor([pose_features(r["pose1"], r["pose2"]) for r in data.records])
                        with torch.inference_mode():
                            expected = model(inputs).softmax(dim=1)
                        torch.testing.assert_close(torch.tensor([r["probabilities"] for r in predictions]), expected)
                        self.assertEqual(classifier.checkpoint_epoch, 12)
                        self.assertEqual(classifier.config, config)
                        self.assertFalse(classifier.model.training)
                        self.assertEqual(sha256_file(path), digest)
                        self.assertEqual(classifier.describe()["training_data_identity"]["annotation_sha256"], "a" * 64)
                    finally:
                        classifier.close()
                    self.assertIsNone(classifier.model)
                    # Explicit external loading does not disable the original split guard.
                    with self.assertRaisesRegex(ValueError, "annotation_sha256"):
                        validate_checkpoint_data(config, name, annotations, annotations,
                                                 {"class_names": CLASS_NAMES_10, "label_map": LABEL_MAP_10})

    def test_resnet_external_rgb_preprocessing_matches_original_image_dataset(self):
        import torch
        from src.Eval.external import ExternalClassifier, image_tensor
        from src.Loader.datasets import ImageDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations, images, records = external_fixture(root)
            data = ExternalData(annotations, images)
            path, model, _ = self.checkpoint(root, "image")
            run = root / "resnet-run"
            (run / "checkpoints").mkdir(parents=True)
            path.rename(run / "checkpoints/best.pt")
            classifier = ExternalClassifier(run)
            try:
                original_records = [{**r, "image": r["image"].removesuffix(".png"), "position": "half_guard1"}
                                    for r in records]
                dataset = ImageDataset(original_records, CLASS_NAMES_10, LABEL_MAP_10, images)
                inputs = torch.stack([dataset[i][0] for i in range(len(dataset))])
                torch.testing.assert_close(image_tensor(data.image_path(records[0])), inputs[0], rtol=0, atol=0)
                with torch.inference_mode():
                    expected = model(inputs).softmax(dim=1)
                actual = classifier.predict(data.records, data.image_path, batch_size=4, progress=False)
                torch.testing.assert_close(torch.tensor([r["probabilities"] for r in actual]), expected)
                self.assertEqual(len(actual), 4)  # Image model includes a missing-reference-pose frame.
            finally:
                classifier.close()

    def test_dinov3_uses_saved_architecture_without_pretrained_download_and_rejects_taxonomy_mismatch(self):
        import torch
        from src.Eval.external import ExternalClassifier

        def tiny_image(config, *, pretrained):
            self.assertFalse(pretrained)
            self.assertIn("dinov3", config["type"])
            return torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 10))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations, images, _ = external_fixture(root)
            data = ExternalData(annotations, images)
            path, _, _ = self.checkpoint(root, "image-dinov3", tiny_image)
            with patch("src.Modules.models.build_model", side_effect=tiny_image):
                classifier = ExternalClassifier(path)
                try:
                    self.assertEqual(len(classifier.predict(data.records, data.image_path, progress=False)), 4)
                finally:
                    classifier.close()
            checkpoint = torch.load(path, weights_only=False)
            checkpoint["config"]["class_names"] = list(reversed(CLASS_NAMES_10))
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "taxonomy/order"):
                ExternalClassifier(path)

    def test_multiple_checkpoint_cli_writes_reports_and_releases_models(self):
        from src.Eval.external import ExternalClassifier, evaluate_external
        from src.Scripts.evaluate_external import build_parser

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations, images, _ = external_fixture(root)
            first, _, _ = self.checkpoint(root, "pose")
            second, _, _ = self.checkpoint(root, "pose-wide")
            output = root / "results" / "external.json"
            args = build_parser().parse_args([
                "--checkpoint", str(first), "--checkpoint", str(second), "--annotations", str(annotations),
                "--images-dir", str(images), "--device", "cpu", "--no-progress", "--output", str(output)])
            closed = []
            original_close = ExternalClassifier.close

            def close(classifier):
                original_close(classifier)
                closed.append(classifier.model)

            with patch.object(ExternalClassifier, "close", close):
                result = evaluate_external(args)
            self.assertEqual(closed, [None, None])
            self.assertEqual(result["evaluation_kind"], "external")
            self.assertEqual(len(result["runs"]), 2)
            self.assertEqual(result["dataset"]["pose_eligible"], 3)
            self.assertTrue(output.is_file())
            for run in result["runs"]:
                self.assertTrue(Path(run["report_file"]).is_file())
                self.assertEqual(run["metrics"]["samples"], 3)
                self.assertIn("not_verified", run["pose_comparison"]["reference_kind"])
            args.checkpoint = [str(first)]
            args.pose_source = "yolo"
            args.output = root / "detected.json"
            data = ExternalData(annotations, images)
            detections = {row["image"]: {"image": row["image"], "poses": [], "seconds": 0.1}
                          for row in data.records}
            with patch("src.Eval.pose_detection.detect_records", return_value=detections):
                detected = evaluate_external(args)["runs"][0]
            self.assertEqual(detected["metrics_source"], "yolo_all_frames")
            self.assertEqual(detected["metrics"]["samples"], 4)
            self.assertEqual(detected["pose_comparison"]["baseline"]["metrics"]["samples"], 3)
            self.assertEqual(detected["pose_comparison"]["detectors"]["yolo"]["coverage"]["0"], 4)


if __name__ == "__main__":
    unittest.main()
