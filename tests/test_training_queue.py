"""Queue configuration and scheduling without a GPU or dataset training."""

import json
import io
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src.Scripts.training_queue import build_parser, main, resolve_queue
from src.Train.queue_gpu import GPU
from src.Train.queue_manager import QueueManager, check_previous_manager, manager_alive, process_token
from src.Train.queue_plan import prepare_plan
from src.Train.queue_process import TrainingProcess


class QueuePlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("annotations.json", "split.json"):
            (self.root / name).write_text("{}")
        (self.root / "images").mkdir()
        self.source = {"version": 1, "defaults": {"annotations": "annotations.json", "split": "split.json"},
                       "jobs": [{"id": "pose", "train": {"model": "pose", "epochs": 10}}]}

    def prepare(self):
        path = self.root / "plan.json"
        path.write_text(json.dumps(self.source))
        return prepare_plan(path, self.root, "batch")

    def test_six_default_slots_absolute_paths_and_safe_argv(self):
        plan = self.prepare()
        self.assertEqual(plan["max_parallel"], 6)
        command = plan["jobs"][0]["command"]
        self.assertEqual(command[:4], [sys.executable, "-u", "-m", "src.Scripts.train"])
        self.assertIn(str(self.root / "annotations.json"), command)
        self.assertEqual(command[-3:], ["--device", "cuda", "--progress"])
        self.assertEqual(plan["jobs"][0]["min_free_mib"], 4096)
        self.assertEqual(plan["jobs"][0]["run_dir"], str(self.root / "trainings" / "batch-pose"))

    def test_job_overrides_defaults_and_image_memory_threshold(self):
        self.source["defaults"]["lr"] = .001
        self.source["jobs"][0]["train"].update(model="image-dinov3", images_dir="images", lr=.00001)
        job = self.prepare()["jobs"][0]
        self.assertEqual(job["command"].count("--lr"), 1)
        self.assertEqual(job["command"][job["command"].index("--lr") + 1], "1e-05")
        self.assertEqual(job["min_free_mib"], 16384)

    def test_resume_restores_own_run_and_cannot_be_duplicated(self):
        checkpoint = self.root / "previous" / "checkpoints" / "last.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        self.source["defaults"] = {"compile": True}
        self.source["jobs"] = [{"id": "continue", "train": {"resume": str(checkpoint), "epochs": 30}}]
        job = self.prepare()["jobs"][0]
        self.assertNotIn("--run-dir", job["command"])
        self.assertEqual(job["run_dir"], str(checkpoint.parent.parent))
        self.source["jobs"].append({"id": "duplicate", "train": {"resume": str(checkpoint)}})
        with self.assertRaisesRegex(ValueError, "same run"):
            self.prepare()

    def test_invalid_types_unknown_options_and_reserved_controls(self):
        for key, value in (("dropot", .1), ("dropout", True), ("compile", "yes"),
                           ("epochs", 1.5), ("run_dir", "elsewhere"), ("device", "cpu"),
                           ("progress", False), ("model", 9)):
            with self.subTest(key=key):
                self.source["jobs"][0]["train"] = {"model": "pose", key: value}
                with self.assertRaises(ValueError):
                    self.prepare()

    def test_invalid_scope_and_hyperparameters_fail_before_start(self):
        for values in ({"model": "pose", "lr": -1}, {"model": "pose", "image_training": "full"},
                       {"model": "pose-attention", "attention_dim": 33, "attention_heads": 4}):
            self.source["jobs"][0]["train"] = values
            with self.assertRaises(ValueError):
                self.prepare()

    def test_missing_paths_existing_runs_duplicate_ids_and_traversal(self):
        self.source["defaults"]["split"] = "missing.json"
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.source["defaults"]["split"] = "split.json"
        self.source["jobs"].append(self.source["jobs"][0].copy())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.prepare()
        self.source["jobs"].pop()
        self.source["jobs"][0]["id"] = "../../escape"
        with self.assertRaises(ValueError):
            self.prepare()
        self.source["jobs"][0]["id"] = "pose"
        (self.root / "trainings" / "batch-pose").mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_dry_run_does_not_probe_cuda_or_start_process(self):
        path = self.root / "plan.json"
        self.source["defaults"] = {key: str(self.root / value) for key, value in self.source["defaults"].items()}
        path.write_text(json.dumps(self.source))
        with patch("src.Scripts.training_queue.PROJECT", self.root), patch("builtins.print"), \
                patch("src.Train.queue_gpu.discover_allowed_gpus", side_effect=AssertionError("must not probe")), \
                patch("subprocess.Popen", side_effect=AssertionError("must not launch")):
            self.assertEqual(main(["start", str(path), "--dry-run", "--name", "test", "--trainings-dir", str(self.root / "trainings")]), 0)

    def test_shipped_example_resolves_both_model_options(self):
        example = Path(__file__).resolve().parents[1] / "docs" / "training_queue.example.json"
        (self.root / "Data" / "splits").mkdir(parents=True)
        (self.root / "Data" / "images").mkdir()
        (self.root / "Data" / "annotations.json").write_text("{}")
        (self.root / "Data" / "splits" / "multiview_sections_shifted.json").write_text("{}")
        plan = prepare_plan(example, self.root, "example")
        self.assertEqual([j["model"] for j in plan["jobs"]], ["pose-attention", "image"])
        self.assertEqual(plan["max_parallel"], 6)
        self.assertIn("input-head", plan["jobs"][1]["command"])


class FakeProcess:
    next_pid = 100

    def __init__(self, command, cwd, env, log_path):
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.command, self.env, self.log_path = command, env, log_path
        self.code = None
        self.terminated = False

    def poll(self):
        return self.code

    def wait(self):
        return self.code

    def terminate(self):
        self.terminated = True
        self.code = -15


class QueueLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.gpus = [GPU(i, f"GPU-{i}", "test", 80000, 40000, i * 5) for i in range(6)]
        self.created = []
        self.plan = {"name": "test", "project_root": str(self.directory), "max_parallel": 6,
                     "poll_seconds": 1, "allowed_gpu_uuids": [g.uuid for g in self.gpus],
                     "max_gpu_utilization": 100,
                     "jobs": [{"id": f"job{i}", "command": ["python", "training"], "run_dir": str(self.directory / f"run{i}"),
                               "min_free_mib": 4096, "model": "pose"} for i in range(8)]}

    def factory(self, *args):
        process = FakeProcess(*args)
        self.created.append(process)
        return process

    def manager(self, reader=None, factory=None):
        return QueueManager(self.directory, self.plan, reader or (lambda: self.gpus), factory or self.factory)

    def test_six_concurrent_jobs_distinct_gpus_then_refill_finished_slot(self):
        manager = self.manager()
        self.assertFalse(manager.step())
        self.assertEqual(len(manager.active), 6)
        self.assertEqual({p.env["CUDA_VISIBLE_DEVICES"] for p in self.created}, {g.uuid for g in self.gpus})
        self.assertEqual(manager.state["jobs"][6]["status"], "pending")
        self.created[2].code = 0
        manager.step()
        self.assertEqual(manager.state["jobs"][2]["status"], "succeeded")
        self.assertEqual(manager.state["jobs"][6]["gpu"]["uuid"], "GPU-2")
        self.assertEqual(len(manager.active), 6)
        persisted = json.loads((self.directory / "state.json").read_text())
        self.assertEqual(persisted["jobs"][6]["pid"], self.created[-1].pid)

    def test_waits_on_memory_can_skip_large_job_and_rechecks_telemetry(self):
        self.plan["jobs"] = self.plan["jobs"][:2]
        self.plan["jobs"][0]["min_free_mib"] = 50000
        manager = self.manager()
        manager.step()
        self.assertEqual([j["status"] for j in manager.state["jobs"]], ["pending", "running"])
        self.gpus[1] = GPU(1, "GPU-1", "test", 80000, 60000, 0)
        manager.step()
        self.assertEqual(manager.state["jobs"][0]["status"], "running")
        self.assertEqual(manager.state["jobs"][0]["gpu"]["uuid"], "GPU-1")

    def test_probe_failure_waits_without_any_launch(self):
        def reader():
            raise RuntimeError("nvidia-smi unavailable")
        manager = self.manager(reader)
        manager.step()
        self.assertEqual(len(self.created), 0)
        self.assertIn("unavailable", manager.state["gpu_error"])
        self.assertEqual(manager.state["jobs"][0]["status"], "pending")

    def test_child_failure_does_not_block_other_runs_or_retry_same_directory(self):
        self.plan["max_parallel"] = 1
        self.plan["jobs"] = self.plan["jobs"][:2]
        manager = self.manager()
        manager.step()
        self.created[0].code = 1
        manager.step()
        self.assertEqual(manager.state["jobs"][0]["status"], "failed")
        self.assertEqual(manager.state["jobs"][1]["status"], "running")
        self.created[1].code = 0
        self.assertTrue(manager.step())
        self.assertEqual(manager.state["status"], "finished_with_errors")
        self.assertEqual(len(self.created), 2)

    def test_pause_resume_and_graceful_stop(self):
        self.plan["max_parallel"] = 1
        manager = self.manager()
        manager.step({"action": "pause"})
        self.assertFalse(self.created)
        manager.step({"action": "run"})
        manager.step({"action": "pause"})
        self.assertFalse(self.created[0].terminated)
        manager.step({"action": "stop"})
        self.assertEqual(manager.state["status"], "draining")
        self.assertFalse(self.created[0].terminated)
        self.assertTrue(all(j["status"] == "cancelled" for j in manager.state["jobs"][1:]))
        self.created[0].code = 0
        self.assertTrue(manager.step({"action": "stop"}))
        self.assertEqual(manager.state["status"], "stopped")

    def test_terminate_stops_only_owned_children(self):
        manager = self.manager()
        manager.step()
        self.assertTrue(manager.step({"action": "terminate"}))
        self.assertTrue(all(p.terminated for p in self.created))
        self.assertTrue(all(j["status"] == "cancelled" for j in manager.state["jobs"]))

    def test_inherited_gpu_mask_cannot_expand(self):
        self.plan["allowed_gpu_uuids"] = ["GPU-4"]
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4", "NVIDIA_VISIBLE_DEVICES": "void"}):
            manager = self.manager()
            manager.step()
            self.assertEqual(len(manager.active), 1)
            self.assertEqual(self.created[0].env["CUDA_VISIBLE_DEVICES"], "GPU-4")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "4")

    def test_launch_failure_is_recorded_and_next_job_can_start(self):
        def launch(*args):
            if not self.created:
                self.created.append(None)
                raise OSError("unable to start")
            return self.factory(*args)
        manager = self.manager(factory=launch)
        manager.step()
        self.assertEqual(manager.state["jobs"][0]["status"], "failed")
        self.assertEqual(manager.state["jobs"][1]["status"], "running")

    def test_capture_and_cleanup_failure_does_not_abandon_other_children(self):
        self.plan["jobs"] = self.plan["jobs"][:2]
        manager = self.manager()
        manager.step()
        broken, healthy = self.created
        broken.poll = lambda: (_ for _ in ()).throw(RuntimeError("capture error"))
        broken.terminate = lambda: (_ for _ in ()).throw(RuntimeError("capture error again"))
        # Normal polling marks the broken job failed and keeps the other active.
        manager.step()
        self.assertEqual(manager.state["jobs"][0]["status"], "failed")
        self.assertEqual(manager.state["jobs"][1]["status"], "running")
        # An unexpected supervisor error still cleans all remaining children.
        manager.active["job0"] = broken
        with patch.object(manager, "step", side_effect=RuntimeError("supervisor error")):
            with self.assertRaisesRegex(RuntimeError, "supervisor error"):
                manager.run()
        self.assertTrue(healthy.terminated)
        state = json.loads((self.directory / "state.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["jobs"][1]["status"], "cancelled")

    def test_old_live_children_prevent_new_manager_but_reused_pid_does_not(self):
        state = {"jobs": [{"id": "old", "pid": 54, "process_token": "boot:old"}]}
        (self.directory / "state.json").write_text(json.dumps(state))
        lock = io.StringIO(json.dumps({"directory": str(self.directory)}))
        with patch("src.Train.queue_manager.process_token", return_value="boot:old"):
            with self.assertRaisesRegex(RuntimeError, "live job old"):
                check_previous_manager(lock)
        with patch("src.Train.queue_manager.process_token", return_value="boot:new"):
            check_previous_manager(lock)

    def test_real_children_log_separately_and_replenish_shared_queue(self):
        self.plan["max_parallel"] = 2
        self.plan["jobs"] = self.plan["jobs"][:3]
        for index, job in enumerate(self.plan["jobs"]):
            job["command"] = [sys.executable, "-u", "-c",
                              f"import time,sys; print('job{index}', flush=True); time.sleep(.4); sys.exit({int(index == 1)})"]
        manager = self.manager(factory=TrainingProcess)
        try:
            manager.step()
            self.assertEqual(len(manager.active), 2)
            self.assertEqual(manager.state["jobs"][2]["status"], "pending")
            deadline = time.monotonic() + 10
            while not manager.step():
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.05)
            self.assertEqual([j["status"] for j in manager.state["jobs"]], ["succeeded", "failed", "succeeded"])
            for index in range(3):
                log = (self.directory / "logs" / f"job{index}.log").read_text()
                self.assertEqual(log.strip(), f"job{index}")
        finally:
            for child in manager.active.values():
                child.terminate(grace_seconds=.1)


class QueueInterfaceTests(unittest.TestCase):
    def test_cli_subcommands_and_missing_latest(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args(["logs", "attention"]).job, "attention")
        self.assertTrue(parser.parse_args(["stop", "batch", "--terminate"]).terminate)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No queue"):
                resolve_queue(Path(directory), None)
            with self.assertRaises(ValueError):
                resolve_queue(Path(directory), "../../escape")

    def test_pid_identity_rejects_reused_pid(self):
        with patch("src.Train.queue_manager.process_token", return_value="boot:new"):
            self.assertFalse(manager_alive({"manager_pid": 20, "manager_token": "boot:old"}))
            self.assertTrue(manager_alive({"manager_pid": 20, "manager_token": "boot:new"}))
        self.assertFalse(manager_alive({}))

    def test_zombie_process_is_not_treated_as_live(self):
        for status in ("S", "Z", "X"):
            fields = [status] + ["0"] * 18 + ["123"]
            with patch.object(Path, "read_text", side_effect=["boot-id\n", "4 (name with spaces) " + " ".join(fields)]):
                self.assertEqual(process_token(4), "boot-id:123" if status == "S" else None)


if __name__ == "__main__":
    unittest.main()
