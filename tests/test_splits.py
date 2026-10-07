import copy
import json
from pathlib import Path
import unittest
import uuid

from src.Dataset.annotations import CLASS_NAMES_10, LABEL_MAP_10
from src.Dataset.splits import build_manifest, write_manifest


class RecordingSplitTests(unittest.TestCase):
    def setUp(self):
        self.groups = {"description": "Synthetic fixture: paired camera prefixes share a recording.",
                       "sequences": {}}
        self.records = []
        raw_labels = {normalized: raw for raw, normalized in LABEL_MAP_10.items()}
        for index in range(6):
            prefix = f"{index:02d}"
            self.groups["sequences"][prefix] = {
                "recording": f"fixture-{index // 2}",
                "split": ("train", "val", "test")[index // 2],
            }
            for frame, label in enumerate(CLASS_NAMES_10, start=1):
                self.records.append({"image": f"{prefix}{frame:05d}", "frame": frame,
                                     "position": raw_labels[label]})

    def build(self):
        return build_manifest(self.records, self.groups, "a" * 64)

    def test_shared_assignments_include_records_without_poses(self):
        manifest = self.build()
        self.assertEqual(manifest["counts"], {"train": 20, "val": 20, "test": 20})
        self.assertEqual(manifest["assignments"]["0000001"], "train")
        self.assertEqual(manifest["assignments"]["0100001"], "train")
        self.assertEqual(manifest["label_map"], LABEL_MAP_10)
        self.assertEqual(manifest["class_counts"]["test"], dict.fromkeys(CLASS_NAMES_10, 2))

    def test_recording_cannot_cross_splits(self):
        self.groups["sequences"]["01"]["split"] = "val"
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            self.build()

    def test_mapping_must_match_all_observed_prefixes(self):
        original = copy.deepcopy(self.groups)
        for change in ("missing", "extra"):
            with self.subTest(change=change):
                self.groups = copy.deepcopy(original)
                if change == "missing":
                    del self.groups["sequences"]["00"]
                else:
                    self.groups["sequences"]["99"] = {"recording": "extra", "split": "train"}
                with self.assertRaisesRegex(ValueError, "Sequence mapping mismatch"):
                    self.build()

    def test_class_coverage_failure_reports_counts(self):
        self.records = [r for r in self.records
                        if not (r["image"][:2] in ("04", "05") and r["frame"] == 1)]
        with self.assertRaisesRegex(ValueError, "Class counts:"):
            self.build()

    def test_duplicate_image_rejected(self):
        self.records.append(self.records[0])
        with self.assertRaisesRegex(ValueError, "Duplicate image"):
            self.build()

    def test_publish_refuses_overwrite_and_cleans_temporary_file(self):
        directory = Path(__file__).resolve().parent / (".split-test-" + uuid.uuid4().hex)
        directory.mkdir()
        output = directory / "manifest.json"
        try:
            manifest = self.build()
            write_manifest(manifest, output)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), manifest)
            with self.assertRaises(FileExistsError):
                write_manifest({"invalid": True}, output)
            self.assertEqual(list(output.parent.iterdir()), [output])
        finally:
            output.unlink(missing_ok=True)
            directory.rmdir()


if __name__ == "__main__":
    unittest.main()
