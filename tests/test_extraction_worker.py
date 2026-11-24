"""Chunk resume, provenance, complete publication, and persistent worker tests."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.Dataset.annotations import sha256_file
from src.Eval import extraction_worker as worker
from src.Train.run_metadata import write_json


class FakeDetector:
    name = "yolo"
    provenance = {"weights_sha256": "fixture-fixed"}
    extraction_packages = {"fixture": "1"}
    device = "cpu"

    def __init__(self, device="cpu", fail_at=None):
        self.device = device
        self.calls = 0
        self.fail_at = fail_at

    def __call__(self, path):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("CUDA error: out of memory")
        if path.stem.endswith("0"):
            return []
        return [[[float(i), float(i + 1), .9] for i in range(17)]]


class ChunkTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        images = self.root / "images"
        images.mkdir()
        ids = [f"{i:07d}" for i in range(4)]
        for image in ids:
            (images / f"{image}.jpg").write_bytes(f"fixture {image}".encode())
        annotations = self.root / "annotations.json"
        annotations.write_text("[]", encoding="utf-8")
        self.plan = {"directory": str(self.root), "images_dir": str(images),
                     "project_root": str(self.root), "annotations": str(annotations),
                     "annotation_sha256": sha256_file(annotations), "images": ids,
                     "chunk_size": 2, "detectors": {"yolo": worker.detector_config("yolo")},
                     "jobs": [{"id": f"yolo-{i:05d}", "detector": "yolo", "start": i * 2, "stop": i * 2 + 2}
                              for i in range(2)]}
        write_json(self.root / "plan.json", self.plan)

    def extract(self, index=0, detector=None):
        return worker.extract_chunk(self.plan, self.plan["jobs"][index],
                                    detector or FakeDetector(), progress=False)

    def complete(self):
        for i in range(2):
            self.extract(i)

    def test_resume_keeps_zero_detections_and_recovers_incomplete_tail(self):
        failing = FakeDetector(fail_at=2)
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            self.extract(detector=failing)
        cache, metadata = worker.chunk_paths(self.plan, self.plan["jobs"][0])
        self.assertFalse(json.loads(metadata.read_text())["complete"])
        with cache.open("ab") as handle:
            handle.write(b'{"image":"unterminated')
        resumed = FakeDetector()
        result = self.extract(detector=resumed)
        self.assertEqual((resumed.calls, result["reused_images"], result["images"]), (1, 1, 2))
        self.assertEqual(result["coverage"], {"0": 1, "1": 1, "2": 0})
        self.assertEqual([r["image"] for r in worker.load_cache(cache).values()], self.plan["images"][:2])
        self.assertTrue(json.loads(metadata.read_text())["complete"])

    def test_resume_rejects_changed_image_and_model_identity(self):
        self.extract()
        changed = FakeDetector()
        changed.provenance = {"weights_sha256": "different"}
        with self.assertRaisesRegex(ValueError, "weights changed"):
            self.extract(detector=changed)
        changed = FakeDetector()
        changed.extraction_packages = {"fixture": "2"}
        with self.assertRaisesRegex(ValueError, "packages changed"):
            self.extract(detector=changed)
        (Path(self.plan["images_dir"]) / "0000000.jpg").write_bytes(b"new image")
        with self.assertRaisesRegex(ValueError, "Image content changed"):
            self.extract()

    def test_resume_rejects_extra_or_reordered_rows(self):
        self.extract()
        cache, metadata = worker.chunk_paths(self.plan, self.plan["jobs"][0])
        # Exercise prefix validation on an interrupted, not completed, cache.
        value = json.loads(metadata.read_text())
        value["complete"] = False
        write_json(metadata, value)
        lines = cache.read_bytes().splitlines(keepends=True)
        cache.write_bytes(b"".join(reversed(lines)))
        with self.assertRaisesRegex(ValueError, "reordered or extra"):
            self.extract()

    def test_resume_rejects_modified_completed_pose_before_rewriting_checksum(self):
        self.extract()
        cache, metadata = worker.chunk_paths(self.plan, self.plan["jobs"][0])
        previous_metadata = metadata.read_bytes()
        rows = [json.loads(line) for line in cache.read_text().splitlines()]
        rows[1]["poses"][0][0][0] += 10
        cache.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "Completed chunk content changed"):
            self.extract()
        self.assertEqual(metadata.read_bytes(), previous_metadata)

    def test_chunks_from_different_gpus_merge_and_record_both_uuids(self):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "GPU-first"}):
            self.extract()
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "GPU-second"}):
            self.extract(1)
        worker.merge_detector(self.plan, "yolo")
        metadata = json.loads((self.root / "yolo.meta.json").read_text())
        self.assertEqual(metadata["gpu_uuids"], ["GPU-first", "GPU-second"])

    def test_yolo_asset_lock_wraps_construction_and_releases_on_error(self):
        fcntl = MagicMock()
        fcntl.LOCK_EX, fcntl.LOCK_UN = 2, 8
        operations = []
        fcntl.flock.side_effect = lambda handle, operation: operations.append(operation)
        def fail(device):
            self.assertEqual(operations, [fcntl.LOCK_EX])
            raise RuntimeError("download failed")
        with patch.dict(sys.modules, {"fcntl": fcntl}), patch.object(sys, "platform", "linux"), \
                patch.object(worker, "PROJECT_ROOT", self.root), \
                patch.object(worker, "YoloDetector", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "download failed"):
                worker._build_detector("yolo", "cuda:0")
        self.assertEqual(operations, [fcntl.LOCK_EX, fcntl.LOCK_UN])
        self.assertTrue((self.root / "Data/pose_detectors/yolo26x-pose.pt.lock").is_file())

    def test_merge_requires_every_chunk_and_preserves_previous_final_on_failure(self):
        self.extract()
        target = self.root / "yolo.jsonl"
        target.write_bytes(b"previous complete result")
        with self.assertRaises(FileNotFoundError):
            worker.merge_detector(self.plan, "yolo")
        self.assertEqual(target.read_bytes(), b"previous complete result")
        self.assertEqual(list(self.root.glob(".*.tmp")), [])
        self.extract(1)
        summary = worker.merge_detector(self.plan, "yolo")
        rows = worker.load_cache(target)
        self.assertEqual(list(rows), self.plan["images"])
        self.assertEqual(summary["images"], 4)
        self.assertEqual(summary["coverage"], {"0": 1, "1": 3, "2": 0})
        self.assertEqual(summary["cache_sha256"], sha256_file(target))
        metadata = json.loads((self.root / "yolo.meta.json").read_text())
        self.assertTrue(metadata["complete"])
        self.assertEqual(metadata["configuration"], worker.detector_config("yolo"))

    def test_merge_rejects_corruption_without_touching_published_cache(self):
        self.complete()
        worker.merge_detector(self.plan, "yolo")
        target = self.root / "yolo.jsonl"
        previous = target.read_bytes()
        cache, _ = worker.chunk_paths(self.plan, self.plan["jobs"][1])
        rows = [json.loads(line) for line in cache.read_text().splitlines()]
        rows[0]["poses"][0][0][0] += 1
        cache.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "Chunk content changed"):
            worker.merge_detector(self.plan, "yolo")
        self.assertEqual(target.read_bytes(), previous)

    def test_merge_rejects_mixed_provenance_and_changed_annotation(self):
        self.complete()
        _, metadata = worker.chunk_paths(self.plan, self.plan["jobs"][1])
        original = json.loads(metadata.read_text())
        changed = {**original, "weights": {"weights_sha256": "different"}}
        write_json(metadata, changed)
        with self.assertRaisesRegex(ValueError, "weights changed"):
            worker.merge_detector(self.plan, "yolo")
        write_json(metadata, original)
        Path(self.plan["annotations"]).write_text("[{}]")
        with self.assertRaisesRegex(ValueError, "Annotations changed"):
            worker.merge_detector(self.plan, "yolo")

    def test_merge_rejects_current_image_change(self):
        self.complete()
        (Path(self.plan["images_dir"]) / "0000003.jpg").write_bytes(b"different now")
        with self.assertRaisesRegex(ValueError, "Image content changed"):
            worker.merge_detector(self.plan, "yolo")

    def test_merge_rejects_overlap_and_missing_rows(self):
        self.complete()
        plan = {**self.plan, "jobs": [self.plan["jobs"][0], self.plan["jobs"][0], self.plan["jobs"][1]]}
        with self.assertRaisesRegex(ValueError, "Gap or overlap"):
            worker.merge_detector(plan, "yolo")
        cache, _ = worker.chunk_paths(self.plan, self.plan["jobs"][1])
        cache.write_bytes(cache.read_bytes().splitlines(keepends=True)[0])
        with self.assertRaisesRegex(ValueError, "Missing detections"):
            worker.merge_detector(self.plan, "yolo")

    def test_persistent_worker_constructs_detector_once_for_two_chunks(self):
        work = self.root / "workers" / "gpu0"
        work.mkdir(parents=True)
        write_json(work / "command.json", {"token": "first", "action": "run", "job_id": self.plan["jobs"][0]["id"]})
        detector = FakeDetector()
        actual_extract = worker.extract_chunk
        def extraction(plan, job, resident, progress=True):
            result = actual_extract(plan, job, resident, progress=False)
            if job["start"] == 0:
                command = {"token": "second", "action": "run", "job_id": plan["jobs"][1]["id"]}
            else:
                command = {"token": "end", "action": "stop"}
            write_json(work / "command.json", command)
            return result
        with patch.dict(sys.modules, {"torch": MagicMock()}), patch.object(sys, "platform", "win32"), \
                patch.object(worker, "YoloDetector", return_value=detector) as construct, \
                patch.object(worker, "_versions", return_value=FakeDetector.extraction_packages), \
                patch.object(worker, "extract_chunk", side_effect=extraction):
            self.assertEqual(worker.run_worker(self.root, "gpu0", "yolo"), 0)
        construct.assert_called_once_with("cuda:0")
        self.assertEqual(detector.calls, 4)
        final_reply = json.loads((work / "reply.json").read_text())
        self.assertEqual((final_reply["event"], final_reply["token"]), ("done", "second"))

    def test_worker_reports_oom_with_token_and_retains_partial_rows(self):
        work = self.root / "workers" / "gpu0"
        work.mkdir(parents=True)
        write_json(work / "command.json", {"token": "attempt1", "action": "run", "job_id": self.plan["jobs"][0]["id"]})
        with patch.dict(sys.modules, {"torch": MagicMock()}), patch.object(sys, "platform", "win32"), \
                patch.object(worker, "YoloDetector", return_value=FakeDetector(fail_at=2)), \
                patch.object(worker, "_versions", return_value=FakeDetector.extraction_packages):
            self.assertEqual(worker.run_worker(self.root, "gpu0", "yolo"), 2)
        reply = json.loads((work / "reply.json").read_text())
        self.assertEqual((reply["event"], reply["token"], reply["job_id"]), ("oom", "attempt1", self.plan["jobs"][0]["id"]))
        cache, _ = worker.chunk_paths(self.plan, self.plan["jobs"][0])
        self.assertEqual(len(worker.load_cache(cache)), 1)

    def test_worker_startup_oom_is_distinct_from_bad_input(self):
        with patch.dict(sys.modules, {"torch": MagicMock()}), patch.object(sys, "platform", "win32"), \
                patch.object(worker, "YoloDetector", side_effect=RuntimeError("CUDA out of memory")):
            self.assertEqual(worker.run_worker(self.root, "gpu0", "yolo"), 2)
        reply = json.loads((self.root / "workers/gpu0/reply.json").read_text())
        self.assertEqual((reply["event"], reply["token"]), ("oom", None))
        Path(self.plan["annotations"]).write_text("[{}]")
        self.assertEqual(worker.run_worker(self.root, "gpu0", "yolo"), 1)
        reply = json.loads((self.root / "workers/gpu0/reply.json").read_text())
        self.assertEqual(reply["event"], "error")


if __name__ == "__main__":
    unittest.main()
