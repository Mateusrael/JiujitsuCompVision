"""Dynamic extraction scheduling without CUDA, downloads, or real children."""
import json
from pathlib import Path
import tempfile
import unittest

from src.Eval.extraction_queue import ExtractionManager, choose_detector, prepare_plan
from src.Train.queue_gpu import GPU
from src.Train.run_metadata import write_json


def gpu(index, free=12000):
    return GPU(index, f"GPU-00000000-0000-0000-0000-{index:012x}", "H100", 80000, free, 0)


class FakeProcess:
    def __init__(self, command, env):
        self.pid = 999999
        self.command, self.env = command, env
        self.directory, self.worker_id = Path(command[-3]), command[-2]
        self.code = None
        self.terminated = False

    def poll(self):
        path = self.directory / "workers" / self.worker_id / "command.json"
        if path.exists() and json.loads(path.read_text())["action"] == "stop":
            self.code = 0
        return self.code

    def wait(self):
        return self.code

    def terminate(self):
        self.terminated = True
        self.code = -15


class ExtractionSchedulingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        annotations = []
        for index in range(10):
            image = f"{index:07d}"
            (self.images / f"{image}.jpg").write_bytes(b"fixture")
            annotations.append({"image": image, "frame": index, "position": "standing",
                                "pose1": None, "pose2": None})
        self.annotations = self.root / "annotations.json"
        self.annotations.write_text(json.dumps(annotations))
        self.directory = self.root / "queue"
        self.directory.mkdir()
        self.plan = prepare_plan(self.directory, self.annotations, self.images, chunk_size=1)
        self.processes = []
        self.gpus = [gpu(0), gpu(1)]
        self.ticks = 1000
        self.merged = []
        def factory(command, cwd, env, log):
            process = FakeProcess(command, env)
            self.processes.append(process)
            return process
        def merge(plan, name):
            self.merged.append(name)
            return {"images": 10, "cache_path": f"{name}.jsonl"}
        self.manager = ExtractionManager(self.directory, self.plan, {g.uuid for g in self.gpus},
                                         gpu_reader=lambda: self.gpus, process_factory=factory,
                                         clock=lambda: self.ticks, merge=merge)

    def reply(self, worker, event, **extra):
        token = None if worker["status"] == "loading" else worker["command_token"]
        write_json(self.directory / "workers" / worker["id"] / "reply.json",
                   {"token": token, "event": event, "job_id": worker.get("job_id"),
                    "reserved_mib": 0, **extra})

    def ready(self):
        for worker in self.manager.state["workers"]:
            if worker["status"] == "loading":
                self.reply(worker, "ready")
        self.manager.step()

    def test_all_annotations_are_chunked_even_without_supplied_poses(self):
        self.assertEqual(len(self.plan["images"]), 10)
        self.assertEqual(len(self.plan["jobs"]), 20)
        for detector in ("yolo", "vitpose"):
            jobs = [j for j in self.plan["jobs"] if j["detector"] == detector]
            self.assertEqual([j["start"] for j in jobs], list(range(10)))
            self.assertEqual([j["stop"] for j in jobs], list(range(1, 11)))

    def test_fast_worker_takes_more_chunks_without_reloading(self):
        self.manager.step()
        self.ready()
        fast, slow = self.manager.state["workers"]
        slow_job = slow["job_id"]
        completed = []
        for _ in range(3):
            completed.append(fast["job_id"])
            self.reply(fast, "done", summary={"images": 1})
            self.manager.step()
        self.assertEqual(len(set(completed)), 3)
        self.assertEqual(slow["job_id"], slow_job)
        self.assertEqual(slow["status"], "busy")
        self.assertEqual(len(self.processes), 2)

    def test_low_memory_gpu_gets_yolo_and_children_are_pinned_by_uuid(self):
        self.gpus = [gpu(0, 3000), gpu(1, 12000)]
        self.manager.step()
        assignments = {w["gpu"]["index"]: w["detector"] for w in self.manager.state["workers"]}
        self.assertEqual(assignments, {0: "yolo", 1: "vitpose"})
        self.assertEqual({p.env["CUDA_VISIBLE_DEVICES"] for p in self.processes}, {g.uuid for g in self.gpus})

    def test_resident_model_memory_does_not_trigger_spurious_switch(self):
        self.gpus = [gpu(0)]
        self.manager.step()
        worker = self.manager.state["workers"][0]
        self.gpus = [gpu(0, 3000)]
        self.reply(worker, "ready", reserved_mib=6000)
        self.manager.step()
        self.assertEqual(worker["status"], "busy")
        self.assertEqual(worker["detector"], "vitpose")

    def test_oom_requeues_chunk_and_falls_back_to_yolo_on_same_gpu(self):
        self.gpus = [gpu(0)]
        self.manager.step()
        self.ready()
        worker = self.manager.state["workers"][0]
        job = self.manager.jobs[worker["job_id"]]
        self.reply(worker, "oom", error="CUDA out of memory")
        self.processes[0].code = 2
        self.manager.step()
        self.assertEqual(job["status"], "pending")
        self.assertEqual(self.manager.state["workers"][-1]["detector"], "yolo")
        self.assertNotIn("vitpose", self.manager.disabled)

    def test_repeated_startup_oom_does_not_disable_healthy_gpu_work(self):
        self.manager.step()
        worker = self.manager.state["workers"][0]
        for _ in range(15):
            self.manager.failure(worker, "CUDA out of memory", oom=True)
        self.assertNotIn("vitpose", self.manager.disabled)
        self.assertTrue(all(j["status"] == "pending" for j in self.manager.state["jobs"]))

    def test_startup_oom_published_during_poll_is_not_a_global_failure(self):
        self.gpus = [gpu(0)]
        self.manager.step()
        worker = self.manager.state["workers"][0]
        process = self.processes[0]
        def exit_with_reply():
            self.reply(worker, "oom", error="CUDA out of memory during model load")
            process.code = 2
            return 2
        process.poll = exit_with_reply
        self.manager.step()
        self.assertNotIn("vitpose", self.manager.disabled)
        self.assertEqual(self.manager.state["workers"][-1]["detector"], "yolo")
        self.assertTrue(all(j["status"] == "pending" for j in self.manager.state["jobs"]))

    def test_stop_finishes_active_chunk_without_assigning_another(self):
        self.gpus = [gpu(0)]
        self.manager.step()
        self.ready()
        worker = self.manager.state["workers"][0]
        self.assertFalse(self.manager.step({"action": "stop"}))
        self.assertEqual(worker["status"], "busy")
        self.reply(worker, "done", summary={"images": 1})
        self.assertFalse(self.manager.step({"action": "stop"}))
        self.assertTrue(self.manager.step({"action": "stop"}))
        self.assertEqual(self.manager.state["status"], "stopped")
        self.assertEqual(sum(j["status"] == "succeeded" for j in self.manager.state["jobs"]), 1)
        self.assertFalse(self.merged)

    def test_complete_queue_merges_each_detector_once_after_workers_exit(self):
        for job in self.manager.state["jobs"]:
            job["status"] = "succeeded"
        self.assertTrue(self.manager.step())
        self.assertEqual(self.manager.state["status"], "finished")
        self.assertEqual(self.merged, ["vitpose", "yolo"])

    def test_failed_chunk_does_not_publish_incomplete_detector(self):
        for job in self.manager.state["jobs"]:
            job["status"] = "succeeded"
        self.manager.state["jobs"][0]["status"] = "failed"
        self.assertTrue(self.manager.step())
        self.assertEqual(self.merged, ["yolo"])
        self.assertEqual(self.manager.state["status"], "finished_with_errors")

    def test_unknown_telemetry_never_launches_a_worker(self):
        def unavailable():
            raise RuntimeError("nvidia-smi unavailable")
        self.manager.gpu_reader = unavailable
        self.assertFalse(self.manager.step())
        self.assertFalse(self.processes)
        self.assertIn("unavailable", self.manager.state["gpu_error"])


class SelectionTests(unittest.TestCase):
    def test_thresholds_and_oom_block(self):
        pending = {"yolo": 5, "vitpose": 5}
        minimum = {"yolo": 2048, "vitpose": 8192}
        self.assertIsNone(choose_detector(pending, 1024, minimum))
        self.assertEqual(choose_detector(pending, 4096, minimum), "yolo")
        self.assertEqual(choose_detector(pending, 8192, minimum), "vitpose")
        self.assertEqual(choose_detector(pending, 20000, minimum, blocked={"vitpose"}), "yolo")
        self.assertEqual(choose_detector({"yolo": 1, "vitpose": 0}, 20000, minimum), "yolo")


if __name__ == "__main__":
    unittest.main()
