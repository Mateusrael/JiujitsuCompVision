"""Cache identity and interruption regressions; no detector downloads needed."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.Eval import pose_detection as detection


class FakeDetector:
    calls = 0
    provenance = {"fixture": "fixed"}

    def __init__(self, device):
        pass

    def __call__(self, path):
        type(self).calls += 1
        return [[[float(j), float(j + 1), 0.9] for j in range(17)]]


class DetectorCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image = self.root / "image.jpg"
        self.image.write_bytes(b"fake detector does not decode pixels")
        self.cache = self.root / "new" / "cache"
        self.records = [{"image": "sample"}]
        FakeDetector.calls = 0
        self.mock_detector = patch.object(detection, "YoloDetector", FakeDetector)
        self.mock_detector.start()
        self.addCleanup(self.mock_detector.stop)

    def detect(self):
        return detection.detect_records("yolo", self.records, lambda r: self.image,
                                        self.cache, "cpu", progress=False)

    def test_fresh_directory_and_reuse_without_detector(self):
        first = self.detect()
        self.assertEqual(FakeDetector.calls, 1)
        self.assertEqual(len(first["sample"]["poses"]), 1)
        self.assertTrue((self.cache / "yolo.meta.json").is_file())
        with patch.object(detection, "YoloDetector", side_effect=AssertionError("must reuse")):
            self.assertEqual(self.detect(), first)

    def test_changed_image_is_rejected(self):
        self.detect()
        self.image.write_bytes(b"a different frame with the same filename")
        with self.assertRaisesRegex(ValueError, "Image content changed"):
            self.detect()

    def test_filtered_cache_cannot_be_appended_or_reused_as_raw_detection(self):
        self.detect()
        path = self.cache / "yolo.meta.json"
        metadata = json.loads(path.read_text())
        metadata["postprocessing"] = {"profile": "cached-pose-quality-v1"}
        path.write_text(json.dumps(metadata))
        cache = self.cache / "yolo.jsonl"
        before = cache.read_bytes()
        with cache.open("ab") as handle:
            handle.write(b'{"interrupted":')
        # Failure happens before load_cache can repair/truncate this read-only cache.
        with self.assertRaisesRegex(ValueError, "cache-only"):
            self.detect()
        self.assertEqual(cache.read_bytes(), before + b'{"interrupted":')
        self.assertEqual(FakeDetector.calls, 1)

    def test_changed_configuration_is_rejected(self):
        self.detect()
        path = self.cache / "yolo.meta.json"
        metadata = json.loads(path.read_text())
        metadata["configuration"]["confidence"] = 0.9
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.detect()

    def test_interrupted_tail_is_removed_before_resume(self):
        expected = self.detect()
        cache = self.cache / "yolo.jsonl"
        content = cache.read_bytes()
        with cache.open("ab") as handle:
            handle.write(b'{"image": "incomplete')
        self.assertEqual(self.detect(), expected)
        self.assertEqual(cache.read_bytes(), content)

    def test_complete_invalid_row_and_duplicates_are_not_silently_removed(self):
        self.detect()
        cache = self.cache / "yolo.jsonl"
        original = cache.read_bytes()
        cache.write_bytes(original * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.detect()
        cache.write_bytes(original + b"broken\n")
        with self.assertRaises(ValueError):
            self.detect()

    def test_unverified_legacy_cache_is_rejected(self):
        self.cache.mkdir(parents=True)
        (self.cache / "yolo.jsonl").write_text('{"image":"sample","poses":[]}\n')
        with self.assertRaisesRegex(ValueError, "legacy cache"):
            self.detect()

    def test_zero_detections_are_retained(self):
        with patch.object(FakeDetector, "__call__", return_value=[]):
            rows = self.detect()
        self.assertEqual(rows["sample"]["poses"], [])


class YoloConfidenceTests(unittest.TestCase):
    def test_low_confidence_coordinates_are_not_treated_as_observed(self):
        # Avoid model construction/downloads; exercise the real result adapter.
        detector = detection.YoloDetector.__new__(detection.YoloDetector)
        detector.device = "cpu"
        detector.model = MagicMock()
        result = MagicMock()
        result.boxes.__len__.return_value = 2
        result.keypoints.data.__getitem__.return_value.cpu.return_value.tolist.return_value = [
            [[0, 0, 0.2], [35, 40, 0.4], [15, 20, 0.5]] + [[30, 40, 0.9]] * 14]
        detector.model.predict.return_value = [result]
        poses = detector("image.jpg")
        self.assertEqual(poses[0][:3], [[0, 0, 0], [0, 0, 0], [15, 20, 0.5]])
        self.assertEqual(detection.detector_config("yolo")["joint_confidence"], 0.5)


if __name__ == "__main__":
    unittest.main()
