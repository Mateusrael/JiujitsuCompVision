"""CLI checks that do not load detectors, initialize CUDA or launch real workers."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src.Scripts import extract_pose_queue as cli


def example_plan():
    return {"directory": str(Path("example-output").resolve()),
            "project_root": str(cli.PROJECT), "annotations": "annotations.json",
            "images_dir": "images", "annotation_sha256": "0" * 64,
            "images": ["0000001", "0000002"], "chunk_size": 256,
            "max_parallel": 6, "min_free_mib": {"vitpose": 8192, "yolo": 2048},
            "jobs": [{"id": "yolo-00000", "detector": "yolo", "start": 0, "stop": 2},
                     {"id": "vitpose-00000", "detector": "vitpose", "start": 0, "stop": 2}]}


class ExtractionCLI(unittest.TestCase):
    def test_dry_run_is_concise_and_does_not_probe_or_write(self):
        output = io.StringIO()
        with patch("src.Eval.extraction_queue.prepare_plan", return_value=example_plan()):
            with patch.object(cli, "_available_gpus", side_effect=AssertionError("GPU probe")):
                with patch.object(cli, "write_json", side_effect=AssertionError("write")):
                    with redirect_stdout(output):
                        result = cli.main(["start", "--dry-run"])
        self.assertEqual(result, 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value["image_count"], 2)
        self.assertEqual(value["chunks_per_detector"], {"yolo": 1, "vitpose": 1})
        self.assertNotIn("images", value)
        self.assertNotIn("jobs", value)

    def test_start_cannot_overwrite_existing_output(self):
        with patch("src.Eval.extraction_queue.prepare_plan", return_value=example_plan()):
            with patch.object(cli.sys, "platform", "linux"):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch.object(cli, "_available_gpus", side_effect=AssertionError("GPU probe")):
                        with redirect_stderr(io.StringIO()) as errors:
                            result = cli.main(["start"])
        self.assertEqual(result, 1)
        self.assertIn("use resume", errors.getvalue())

    def test_logs_rejects_paths_instead_of_worker_ids(self):
        with patch.object(cli, "read_state", side_effect=AssertionError("read")):
            with self.assertRaisesRegex(ValueError, "Invalid worker ID"):
                cli.show_logs(Path("queue"), "../manager", 1024)

    def test_stop_does_not_change_an_offline_queue(self):
        with patch.object(cli, "_manager_running", return_value=False):
            with patch.object(cli, "write_json", side_effect=AssertionError("write")):
                with redirect_stderr(io.StringIO()) as errors:
                    result = cli.main(["stop"])
        self.assertEqual(result, 1)
        self.assertIn("offline", errors.getvalue())

    def test_status_counts_images_without_counting_running_chunks_as_done(self):
        state = {"status": "running", "max_parallel": 6, "workers": [
            {"id": "w0001", "detector": "vitpose", "status": "busy", "gpu": {"index": 4},
             "pid": 123, "job_id": "vitpose-00001"}], "jobs": [
            {"detector": "vitpose", "status": "done", "start": 0, "stop": 256},
            {"detector": "vitpose", "status": "running", "start": 256, "stop": 300}]}
        with patch("src.Train.queue_manager.manager_alive", return_value=True):
            with redirect_stdout(io.StringIO()) as output:
                cli.print_status(state)
        self.assertIn("256/300 images", output.getvalue())
        self.assertIn("w0001", output.getvalue())
        self.assertIn("vitpose-00001", output.getvalue())


if __name__ == "__main__":
    unittest.main()
