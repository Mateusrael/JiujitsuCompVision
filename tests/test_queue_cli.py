"""Detached queue startup and terminal controls without launching CUDA processes."""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.Scripts.training_queue import main, read_state
from src.Train.queue_gpu import GPU


class QueueCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queues = self.root / "queues"
        self.directory = self.queues / "test"
        self.gpu = GPU(4, "GPU-00000000-0000-0000-0000-000000000004", "H100", 80000, 20000, 25)
        self.plan = {"version": 1, "name": "test", "project_root": str(self.root),
                     "max_parallel": 6, "poll_seconds": 10, "max_gpu_utilization": 100,
                     "jobs": [{"id": "attention", "min_free_mib": 4096,
                               "command": [sys.executable, "train"], "run_dir": "new-run"}]}
        self.stderr = io.StringIO()
        self.stdout = io.StringIO()

    def start(self, launch, *, allowed=None, telemetry=None):
        with patch("src.Scripts.training_queue.PROJECT", self.root), \
                patch("src.Scripts.training_queue.prepare_plan", return_value=self.plan), \
                patch("src.Scripts.training_queue.sys.platform", "linux"), \
                patch("src.Train.queue_gpu.discover_allowed_gpus", return_value=allowed if allowed is not None else {self.gpu.uuid}), \
                patch("src.Train.queue_gpu.query_gpus", return_value=telemetry if telemetry is not None else [self.gpu]), \
                patch("src.Scripts.training_queue.subprocess.Popen", side_effect=launch) as popen, \
                patch("src.Scripts.training_queue.time.sleep"), \
                patch("sys.stdout", self.stdout), patch("sys.stderr", self.stderr):
            code = main(["--queue-root", str(self.queues), "start", "plan.json", "--name", "test"])
        return code, popen

    def state(self, **updates):
        payload = {"name": "test", "status": "running", "max_parallel": 6,
                   "jobs": [{"id": "attention", "status": "running", "gpu": None,
                             "pid": 101, "message": "Training"}]}
        payload.update(updates)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "state.json").write_text(json.dumps(payload))
        return payload

    def test_success_detaches_manager_persists_visibility_and_selects_latest(self):
        def launch(command, **options):
            self.state()
            self.assertEqual(command[:5], [sys.executable, "-u", "-m", "src.Scripts.training_queue", "_manager"])
            self.assertEqual(command[-1], str(self.directory))
            self.assertTrue(options["start_new_session"])
            self.assertEqual(options["cwd"], self.root)
            self.assertEqual(options["stdin"], subprocess.DEVNULL)
            self.assertEqual(options["stderr"], subprocess.STDOUT)
            self.assertEqual(Path(options["stdout"].name), self.directory / "manager.log")
            self.assertNotIn("env", options)
            return Mock(pid=4321, poll=Mock(return_value=None))

        code, popen = self.start(launch)
        self.assertEqual(code, 0)
        popen.assert_called_once()
        saved = json.loads((self.directory / "plan.json").read_text())
        self.assertEqual(saved["allowed_gpu_uuids"], [self.gpu.uuid])
        self.assertEqual(json.loads((self.queues / "latest.json").read_text()), {"name": "test"})

    def test_failed_manager_does_not_replace_previous_latest(self):
        self.queues.mkdir()
        (self.queues / "latest.json").write_text(json.dumps({"name": "previous"}))

        def launch(*args, **kwargs):
            self.state(status="failed", error="Another queue is already active")
            return Mock(pid=4321, poll=Mock(return_value=1))

        code, _ = self.start(launch)
        self.assertEqual(code, 1)
        self.assertIn("Queue could not start", self.stderr.getvalue())
        self.assertEqual(json.loads((self.queues / "latest.json").read_text())["name"], "previous")

    def test_process_creation_failure_is_persisted_in_queue_state(self):
        code, _ = self.start(OSError("process creation unavailable"))
        self.assertEqual(code, 1)
        self.assertIn("process creation unavailable", self.stderr.getvalue())
        self.assertFalse((self.queues / "latest.json").exists())
        state = json.loads((self.directory / "state.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertIn("process creation unavailable", state["error"])

    def test_unacknowledged_start_is_reported_and_retains_manager_identity(self):
        with patch("src.Train.queue_manager.process_token", return_value="boot:start"):
            code, _ = self.start(lambda *args, **kwargs: Mock(pid=4321, poll=Mock(return_value=None)))
        self.assertEqual(code, 0)
        self.assertIn("still starting", self.stdout.getvalue().lower())
        state = read_state(self.directory)
        self.assertEqual(state["status"], "starting")
        self.assertEqual(state["manager_pid"], 4321)
        self.assertEqual(state["manager_token"], "boot:start")
        with patch("src.Train.queue_manager.manager_alive", return_value=True), patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(main(["--queue-root", str(self.queues), "pause", "test"]), 0)
        self.assertEqual(json.loads((self.directory / "control.json").read_text())["action"], "pause")

    def test_no_matching_visibility_and_impossible_memory_fail_before_process_start(self):
        code, popen = self.start(Mock(), allowed=set())
        self.assertEqual(code, 1)
        popen.assert_not_called()
        self.assertFalse(self.directory.exists())
        self.plan["jobs"][0]["min_free_mib"] = self.gpu.total_mib + 1
        code, popen = self.start(Mock())
        self.assertEqual(code, 1)
        popen.assert_not_called()
        self.assertFalse(self.directory.exists())
        self.assertIn("exceeds", self.stderr.getvalue())

    def test_existing_queue_is_never_overwritten(self):
        original = self.state(status="finished")
        code, popen = self.start(Mock())
        self.assertEqual(code, 1)
        popen.assert_not_called()
        self.assertEqual(json.loads((self.directory / "state.json").read_text()), original)

    def test_control_commands_pause_resume_stop_and_terminate(self):
        self.state()
        with patch("src.Train.queue_manager.manager_alive", return_value=True), \
                patch("sys.stdout", new=io.StringIO()), patch("sys.stderr", self.stderr):
            for command, action in ((["pause", "test"], "pause"), (["resume", "test"], "run"),
                                    (["stop", "test"], "stop")):
                self.assertEqual(main(["--queue-root", str(self.queues), *command]), 0)
                self.assertEqual(json.loads((self.directory / "control.json").read_text())["action"], action)
            self.assertEqual(main(["--queue-root", str(self.queues), "resume", "test"]), 1)
            self.assertEqual(json.loads((self.directory / "control.json").read_text())["action"], "stop")
            self.assertEqual(main(["--queue-root", str(self.queues), "stop", "test", "--terminate"]), 0)
            self.assertEqual(json.loads((self.directory / "control.json").read_text())["action"], "terminate")

    def test_offline_queue_control_fails_without_writing_request(self):
        self.state()
        with patch("src.Train.queue_manager.manager_alive", return_value=False), patch("sys.stderr", self.stderr):
            self.assertEqual(main(["--queue-root", str(self.queues), "stop", "test", "--terminate"]), 1)
        self.assertFalse((self.directory / "control.json").exists())

    def test_log_viewer_uses_selected_job_and_never_controls_training(self):
        self.state()

        def view(path, finished, **options):
            self.assertEqual(path, self.directory / "logs" / "attention.log")
            self.assertFalse(finished())
            self.state(jobs=[{"id": "attention", "status": "succeeded"}])
            self.assertTrue(finished())
            self.assertEqual(options["tail_bytes"], 500)

        with patch("src.Train.queue_process.follow_log", side_effect=view) as viewer, \
                patch("src.Train.queue_manager.manager_alive", return_value=True), \
                patch("sys.stdout", new=io.StringIO()):
            code = main(["--queue-root", str(self.queues), "logs", "attention", "--name", "test", "--tail-bytes", "500"])
        self.assertEqual(code, 0)
        viewer.assert_called_once()
        self.assertFalse((self.directory / "control.json").exists())


if __name__ == "__main__":
    unittest.main()
