"""Verify shared temporal membership, integrity checks and direct CLI deployment."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.Dataset.annotations import CLASS_NAMES_10, sha256_file
from src.Dataset.temporal import build_temporal_manifest
from src.Loader.splits import read_manifest, split_records
from tests.test_temporal import fixture


class TemporalLoadingTests(unittest.TestCase):
    def test_loader_excludes_other_views_and_rejects_edited_manifest(self):
        records = fixture()
        records += [{**row, "image": "99" + row["image"][2:]} for row in records[:10]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations, split = root / "annotations.json", root / "split.json"
            annotations.write_text(json.dumps(records), encoding="utf-8")
            expected = build_temporal_manifest(records, sha256_file(annotations), gap_frames=10)
            split.write_text(json.dumps(expected), encoding="utf-8")
            manifest = read_manifest(split, annotations, records)
            retained = set()
            for part in ("train", "val", "test"):
                selected = split_records(records, manifest, part)
                self.assertEqual(len(selected), manifest["counts"][part])
                self.assertFalse(any(row["image"].startswith("99") for row in selected))
                ids = {row["image"] for row in selected}
                self.assertFalse(ids & retained)
                retained.update(ids)
            self.assertEqual(len(records) - len(retained), manifest["counts"]["excluded"])
            with self.assertRaisesRegex(ValueError, "partition"):
                split_records(records, manifest, "excluded")
            for field, replacement in (
                    ("counts", {}), ("class_counts", {}), ("class_names", []),
                    ("exclusion_reasons", {}), ("temporal_boundaries", {}),
                    ("source_selection", {}), ("separation_audit", {}),
                    ("evaluation_scope", "Unseen matches"),
                    ("assignments", dict.fromkeys(manifest["assignments"], "train")),
                    ("parameters", {**manifest["parameters"], "gap_frames": 0}),
                    ("class_sources", {**manifest["class_sources"], "back": "99"}),
                    ("parameters", {}), ("class_sources", None), ("schema_version", 1)):
                with self.subTest(field=field, replacement=replacement):
                    modified = copy.deepcopy(manifest)
                    modified[field] = replacement
                    split.write_text(json.dumps(modified), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        read_manifest(split, annotations, records)

    def test_direct_script_defaults_and_immutable_output(self):
        project = Path(__file__).resolve().parents[1]
        script = project / "src" / "Scripts" / "prepare_splits.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations, split = root / "annotations.json", root / "split.json"
            records = fixture(count=2000)
            annotations.write_text(json.dumps(records), encoding="utf-8")
            command = [sys.executable, "-B", str(script), "--annotations", str(annotations),
                       "--output", str(split)]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = read_manifest(split, annotations, records)
            self.assertEqual(manifest["split_method"], "single-view-temporal")
            self.assertEqual(manifest["parameters"]["gap_frames"], 150)
            self.assertIn("Excluded:", result.stdout)
            self.assertIn("Evaluation scope:", result.stdout)
            self.assertTrue(all(name in result.stdout for name in CLASS_NAMES_10))
            saved = split.read_bytes()
            rerun = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertIn("already exists", rerun.stderr)
            self.assertEqual(split.read_bytes(), saved)

    def test_recording_groups_require_explicit_method(self):
        script = Path(__file__).resolve().parents[1] / "src" / "Scripts" / "prepare_splits.py"
        command = [sys.executable, "-B", str(script), "--annotations", "unused", "--output", "unused"]
        for flags, message in ((["--groups", "groups.json"], "requires --method"),
                               (["--method", "recording-group"], "requires --groups")):
            with self.subTest(flags=flags):
                result = subprocess.run(command + flags, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()
