import json
import tempfile
import unittest
from pathlib import Path

from src.Dataset.annotations import CLASS_NAMES_10, LABEL_MAP_10, load_annotations
from src.Dataset.audit import audit_annotations


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "annotations.json"

    def write(self, records):
        self.path.write_text(json.dumps(records), encoding="utf-8")

    def test_actual_schema_leading_zeros_missing_pose_and_confidence_above_one(self):
        self.write([{"image": "0000001", "frame": 1, "position": "standing",
                     "pose2": [[1.0, 2.0, 1.14]] * 17}])
        self.assertEqual(load_annotations(self.path)[0]["image"], "0000001")
        report = audit_annotations(self.path)
        self.assertEqual(report["missing_pose_counts"]["pose1"], 1)
        self.assertEqual(report["confidence"]["outside_zero_one"], 17)
        self.assertEqual(report["invalid_records"], 0)

    def test_rejects_duplicate_and_unknown_labels(self):
        record = {"image": "0000001", "frame": 1, "position": "mount1"}
        self.write([record, record])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_annotations(self.path)
        record["position"] = "mount3"
        self.write([record])
        with self.assertRaisesRegex(ValueError, "unknown position"):
            load_annotations(self.path)

    def test_invalid_numeric_pose_audited_without_silent_dropping(self):
        self.write([{"image": "0000001", "frame": 1, "position": "standing",
                     "pose1": [[float("nan"), 2, 1]] * 17}])
        self.assertEqual(audit_annotations(self.path)["invalid_records"], 1)
        with self.assertRaisesRegex(ValueError, "finite"):
            load_annotations(self.path)

    def test_explicit_role_mapping(self):
        self.assertEqual(len(CLASS_NAMES_10), 10)
        self.assertEqual(len(LABEL_MAP_10), 18)
        self.assertEqual(LABEL_MAP_10["mount1"], LABEL_MAP_10["mount2"])
        self.assertEqual(LABEL_MAP_10["5050_guard"], "5050_guard")

    def test_image_audit_detects_ambiguous_extensions(self):
        self.write([{"image": "0000001", "frame": 1, "position": "standing"}])
        self.path.with_name("0000001.jpg").write_bytes(b"fixture")
        self.path.with_name("0000001.png").write_bytes(b"fixture")
        report = audit_annotations(self.path, self.path.parent)
        self.assertEqual(report["missing_image_files"], 0)
        self.assertEqual(report["ambiguous_image_files"], 1)


if __name__ == "__main__":
    unittest.main()
