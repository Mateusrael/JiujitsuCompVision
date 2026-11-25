"""Offline cached-pair filtering, source integrity, and atomic publication."""

from copy import deepcopy
import builtins
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.Dataset.annotations import sha256_file
from src.Eval import pose_cache_filter as filtering
from src.Eval.pose_detection import detector_config
from tests.test_pose_quality import pose


class CacheFilterTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.source = self.root / "original"
        self.source.mkdir()
        self.output = self.root / "filtered"
        self.annotations = self.root / "annotations.json"
        self.records = [{"image": f"{index:07d}", "frame": index, "position": "standing"}
                        for index in range(4)]
        self.annotations.write_text(json.dumps(self.records), encoding="utf-8")
        self.rows = [{"image": record["image"], "poses": poses, "seconds": index+.25,
                      "image_sha256": str(index)*64}
                     for index, (record, poses) in enumerate(zip(self.records,
                         ([pose(), pose(100)], [pose()], [], [pose(), pose(1)])))]
        for name in filtering.DETECTORS:
            self.save_source(name, self.rows)

    def save_source(self, name, rows):
        path = self.source / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8", newline="\n")
        coverage = {"0": 0, "1": 0, "2": 0}
        for row in rows:
            coverage[str(sum(any(joint[2] > 0 for joint in person) for person in row["poses"]))] += 1
        metadata = {"configuration": detector_config(name), "complete": True,
                    "annotation_sha256": sha256_file(self.annotations),
                    "population": "all_original_annotations", "images": len(rows),
                    "coverage": coverage, "cache_sha256": sha256_file(path),
                    "weights": {"fixture": name}, "packages": {"fixture": "1"},
                    "device": "multiple", "gpu_uuids": ["original-gpu"],
                    "detection_seconds": sum(row["seconds"] for row in rows),
                    "merge_wall_seconds": .25, "cache_path": str(path)}
        path.with_suffix(".meta.json").write_text(json.dumps(metadata), encoding="utf-8")

    def snapshot(self):
        return {path.name: path.read_bytes() for path in self.source.iterdir()}

    def run_filter(self):
        return filtering.filter_caches(self.source, self.output, self.annotations)

    def test_filters_original_pairs_preserving_provenance_and_source_bytes(self):
        before = self.snapshot()
        summaries = self.run_filter()
        self.assertEqual(before, self.snapshot())
        for name in filtering.DETECTORS:
            rows, metadata = filtering.read_verified_cache_subset(name, self.records, self.output)
            self.assertEqual(summaries[name]["accepted_pairs"], 1)
            self.assertEqual(summaries[name]["rejected_frames"], 3)
            self.assertEqual(metadata["coverage"], {"0": 3, "1": 0, "2": 1})
            self.assertEqual(metadata["configuration"], detector_config(name))
            self.assertEqual(metadata["weights"], {"fixture": name})
            self.assertEqual(metadata["gpu_uuids"], ["original-gpu"])
            self.assertEqual(metadata["detection_seconds"], 7.)
            self.assertEqual(metadata["row_seconds_meaning"], "original_detector_inference")
            self.assertGreaterEqual(metadata["filtering_wall_seconds"], 0.)
            provenance = metadata["postprocessing"]
            source = self.source / f"{name}.jsonl"
            self.assertEqual(provenance["source_cache"], str(source.resolve()))
            self.assertEqual(provenance["source_cache_sha256"], sha256_file(source))
            self.assertEqual(provenance["source_metadata_sha256"], sha256_file(source.with_suffix(".meta.json")))
            for original in self.rows:
                filtered = rows[original["image"]]
                self.assertEqual(filtered["original_poses"], original["poses"])
                self.assertEqual(filtered["image_sha256"], original["image_sha256"])
                self.assertEqual(filtered["seconds"], original["seconds"])
                filtering.validate_filtered_row(filtered, name)

    def test_refuses_existing_output_and_filtering_an_already_filtered_cache(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("preserve")
        with self.assertRaises(FileExistsError):
            self.run_filter()
        self.assertEqual(sentinel.read_text(), "preserve")
        with self.assertRaises(FileExistsError):
            filtering.filter_caches(self.source, self.source, self.annotations)
        other = self.root / "first_filtered"
        filtering.filter_caches(self.source, other, self.annotations)
        with self.assertRaisesRegex(ValueError, "original, unfiltered"):
            filtering.filter_caches(other, self.root / "second_filtered", self.annotations)

    def test_rejects_malformed_incomplete_duplicate_missing_or_unknown_rows_without_publication(self):
        source = self.source / "vitpose.jsonl"
        originals = self.snapshot()
        changes = [
            ("unterminated", lambda: source.write_bytes(source.read_bytes().rstrip(b"\n"))),
            ("json", lambda: source.write_bytes(source.read_bytes()+b"broken\n")),
            ("duplicate", lambda: self.save_source("vitpose", self.rows[:-1]+[self.rows[0]])),
            ("missing", lambda: source.write_bytes(source.read_bytes().splitlines(keepends=True)[0])),
            ("unexpected", lambda: self.save_source("vitpose", [*self.rows[:-1], {**self.rows[-1], "image": "9999999"}])),
            ("invalid pose", lambda: self.save_source("vitpose", [{**self.rows[0], "poses": [[[0,1,2]]]}, *self.rows[1:]])),
        ]
        for label, change in changes:
            for name, content in originals.items():
                (self.source / name).write_bytes(content)
            change()
            broken_bytes = self.snapshot()
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.run_filter()
            self.assertFalse(self.output.exists())
            self.assertEqual(list(self.root.glob(".filtered.filter-*")), [])
            self.assertEqual(self.snapshot(), broken_bytes)

    def test_rejects_changed_configuration_coverage_annotation_hash_and_cache_hash(self):
        path = self.source / "vitpose.meta.json"
        original = json.loads(path.read_text())
        changes = [lambda m: m.update(complete=False),
                   lambda m: m["configuration"].update(dataset_index=1),
                   lambda m: m.update(coverage={"0":2,"1":1,"2":1}),
                   lambda m: m.update(annotation_sha256="f"*64),
                   lambda m: m.update(cache_sha256="f"*64),
                   lambda m: m.update(images=5)]
        for change in changes:
            metadata = deepcopy(original)
            change(metadata)
            path.write_text(json.dumps(metadata), encoding="utf-8")
            before = self.snapshot()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_filter()
            self.assertEqual(before, self.snapshot())
            self.assertFalse(self.output.exists())

    def test_source_change_during_filtering_blocks_both_final_caches(self):
        original_filter = filtering.filter_pose_pair
        changed = False
        def change_metadata(poses, **kwargs):
            nonlocal changed
            if not changed:
                changed = True
                path = self.source / "yolo.meta.json"
                path.write_text(path.read_text()+" ", encoding="utf-8")
            return original_filter(poses, **kwargs)
        with patch.object(filtering, "filter_pose_pair", side_effect=change_metadata):
            with self.assertRaisesRegex(ValueError, "metadata changed"):
                self.run_filter()
        self.assertFalse(self.output.exists())

    def test_subset_reads_all_rows_for_validation_but_retains_only_requested(self):
        before = self.snapshot()
        selected, metadata = filtering.read_verified_cache_subset("yolo", self.records[:1], self.source)
        self.assertEqual(list(selected), ["0000000"])
        self.assertIsNone(filtering.validate_postprocessing("yolo", metadata))
        self.assertEqual(self.snapshot(), before)
        with self.assertRaisesRegex(ValueError, "missing 1 requested"):
            filtering.read_verified_cache_subset("yolo", [{"image":"9999999"}], self.source)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            filtering.read_verified_cache_subset("yolo", [self.records[0]]*2, self.source)
        path = self.source / "yolo.jsonl"
        path.write_bytes(path.read_bytes()+b"broken\n")
        with self.assertRaises(ValueError):
            filtering.read_verified_cache_subset("yolo", self.records[:1], self.source)

    def test_filtered_validation_rejects_forged_audit_or_altered_selected_coordinates(self):
        self.run_filter()
        selected, metadata = filtering.read_verified_cache_subset("yolo", self.records, self.output)
        row = selected["0000000"]
        altered = deepcopy(row)
        altered["poses"][0][5][0] += 10
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            filtering.validate_filtered_row(altered, "yolo")
        altered = deepcopy(row)
        altered["quality_filter"]["accepted"] = False
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            filtering.validate_filtered_row(altered, "yolo")
        altered_metadata = deepcopy(metadata)
        altered_metadata["postprocessing"]["configuration"]["min_body_joints"] += 1
        with self.assertRaisesRegex(ValueError, "configuration"):
            filtering.validate_postprocessing("yolo", altered_metadata)

    def test_v1_archives_remain_readable_while_v2_recovers_partial_pairs(self):
        partial_torso, partial_legs = pose(), pose()
        for index in (5, 6):
            partial_torso[index] = [0., 0., 0.]
        for index in (13, 14, 15, 16):
            partial_legs[index] = [0., 0., 0.]
        rows = deepcopy(self.rows)
        rows[1]["poses"] = [partial_torso, pose(100)]
        rows[2]["poses"] = [partial_legs, pose(100)]
        for name in filtering.DETECTORS:
            self.save_source(name, rows)
        original_bytes = self.snapshot()
        v1_output = self.root / "archive_v1"
        filtering.filter_caches(self.source, v1_output, self.annotations, profile="cached-pose-quality-v1")
        self.run_filter()
        self.assertEqual(self.snapshot(), original_bytes)
        for name in filtering.DETECTORS:
            v1_rows, v1_meta = filtering.read_verified_cache_subset(name, self.records, v1_output)
            v2_rows, v2_meta = filtering.read_verified_cache_subset(name, self.records, self.output)
            self.assertEqual(filtering.validate_postprocessing(name, v1_meta), "cached-pose-quality-v1")
            self.assertEqual(filtering.validate_postprocessing(name, v2_meta), "cached-pose-quality-v2")
            self.assertEqual(v1_meta["coverage"], {"0":3, "1":0, "2":1})
            self.assertEqual(v2_meta["coverage"], {"0":1, "1":0, "2":3})
            for image in ("0000001", "0000002"):
                self.assertEqual(v1_rows[image]["poses"], [])
                self.assertEqual(v2_rows[image]["poses"], v2_rows[image]["original_poses"])
            for detected in (v1_rows, v2_rows):
                duplicate = detected["0000003"]
                self.assertEqual(duplicate["poses"], [])
                self.assertEqual(duplicate["quality_filter"]["reason"], "duplicate_skeletons")

    def test_rows_must_match_the_cache_profile_even_with_a_valid_checksum(self):
        filtering.filter_caches(self.source, self.output, self.annotations, profile="cached-pose-quality-v1")
        path = self.output / "yolo.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["quality_filter"]["profile"] = "cached-pose-quality-v2"
        # This fully observed pair passes either policy; its row is valid v2,
        # but the containing cache declares the immutable v1 policy.
        filtering.validate_filtered_row(rows[0], "yolo")
        with self.assertRaisesRegex(ValueError, "profile differs"):
            filtering.validate_filtered_row(rows[0], "yolo", profile="cached-pose-quality-v1")
        path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8", newline="\n")
        metadata_path = path.with_suffix(".meta.json")
        metadata = json.loads(metadata_path.read_text())
        metadata["cache_sha256"] = sha256_file(path)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "profile differs"):
            filtering.read_verified_cache_subset("yolo", self.records, self.output)

    def test_filtering_imports_no_gpu_image_or_detector_libraries(self):
        script = """
import builtins, sys
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch','torchvision','PIL','ultralytics','transformers'}:
        raise AssertionError('Forbidden import: '+name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
from src.Eval.pose_cache_filter import read_verified_cache_subset
read_verified_cache_subset('yolo', [{'image':'0000000'}], sys.argv[1])
"""
        result = subprocess.run([sys.executable, "-c", script, str(self.source)],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "torchvision", "PIL", "ultralytics", "transformers"}:
                raise AssertionError(f"Forbidden import: {name}")
            return original_import(name, *args, **kwargs)
        with patch.object(builtins, "__import__", side_effect=guarded):
            self.run_filter()

    def test_cli_defaults_are_cached_input_and_new_filtered_output(self):
        from src.Scripts.filter_pose_cache import build_parser, output_directory
        args = build_parser().parse_args([])
        self.assertEqual(args.input_dir.name, "full_dataset")
        self.assertEqual(args.profile, "cached-pose-quality-v2")
        self.assertEqual(output_directory(args).name, "full_dataset_filtered_v2")
        old = build_parser().parse_args(["--profile", "cached-pose-quality-v1"])
        self.assertEqual(output_directory(old).name, "full_dataset_filtered")
        custom = build_parser().parse_args(["--output-dir", str(self.output)])
        self.assertEqual(output_directory(custom), self.output)

    def test_cli_passes_selected_profile_and_dynamic_output_to_filter(self):
        from src.Scripts import filter_pose_cache as cli
        with patch.object(cli, "filter_caches", return_value={}) as run:
            cli.main(["--profile", "cached-pose-quality-v1"])
            self.assertEqual(run.call_args.kwargs["profile"], "cached-pose-quality-v1")
            self.assertEqual(run.call_args.args[1].name, "full_dataset_filtered")


if __name__ == "__main__":
    unittest.main()
