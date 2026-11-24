"""One dynamic queue for full-dataset YOLO/ViTPose extraction.

The supervisor never imports torch. One persistent child owns each selected GPU;
only the supervisor assigns chunks, and each chunk has its own resumable cache.
"""
import json
import os
from pathlib import Path
import signal
import sys
import time
import uuid

from src.Dataset.annotations import load_annotations, sha256_file
from src.Eval.pose_detection import detector_config
from src.Train.queue_gpu import query_gpus
from src.Train.queue_manager import now, process_token
from src.Train.queue_process import TrainingProcess
from src.Train.run_metadata import write_json

PROJECT = Path(__file__).resolve().parents[2]
DETECTORS = ("vitpose", "yolo")


def prepare_plan(directory, annotations, images_dir, *, chunk_size=256, max_parallel=6,
                 vitpose_min_free_mib=8192, yolo_min_free_mib=2048):
    for name, value in (("chunk_size", chunk_size), ("max_parallel", max_parallel),
                        ("vitpose_min_free_mib", vitpose_min_free_mib),
                        ("yolo_min_free_mib", yolo_min_free_mib)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    annotations, images_dir = Path(annotations).resolve(), Path(images_dir).resolve()
    digest = sha256_file(annotations)
    records = load_annotations(annotations)
    images = sorted(row["image"] for row in records)
    del records  # Do not retain the supplied skeletons in the supervisor.
    if sha256_file(annotations) != digest:
        raise ValueError("Annotations changed while preparing the extraction")
    missing = [image for image in images if not (images_dir / f"{image}.jpg").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing under {images_dir}; e.g. {missing[:3]}")
    jobs = [{"id": f"{name}-{number:05d}", "detector": name,
             "start": start, "stop": min(start + chunk_size, len(images))}
            for name in DETECTORS
            for number, start in enumerate(range(0, len(images), chunk_size))]
    return {"schema_version": 1, "created_utc": now(), "directory": str(Path(directory).resolve()),
            "project_root": str(PROJECT), "annotations": str(annotations),
            "annotation_sha256": digest, "images_dir": str(images_dir), "images": images,
            "population": "all_annotation_images", "chunk_size": chunk_size,
            "max_parallel": max_parallel, "min_free_mib": {"vitpose": vitpose_min_free_mib,
                                                          "yolo": yolo_min_free_mib},
            "detectors": {name: detector_config(name) for name in DETECTORS}, "jobs": jobs}


def choose_detector(pending, free_mib, minimums, *, resident=None, reserved_mib=0,
                    blocked=()):
    """Prefer ViTPose when it fits, otherwise run YOLO on that same GPU.

    Telemetry excludes the resident model's CUDA cache, so add only this worker's
    reported reservation back when comparing the requirements of a fresh load.
    """
    available = free_mib + max(0, reserved_mib if resident else 0)
    return next((name for name in DETECTORS if pending.get(name, 0) > 0
                 and name not in blocked and available >= minimums[name]), None)


class ExtractionManager:
    def __init__(self, directory, plan, allowed, *, gpu_reader=query_gpus,
                 process_factory=TrainingProcess, clock=time.monotonic, merge=None):
        self.directory, self.plan, self.allowed = Path(directory), plan, set(allowed)
        self.gpu_reader, self.process_factory, self.clock = gpu_reader, process_factory, clock
        if merge is None:
            from src.Eval.extraction_worker import merge_detector
            merge = merge_detector
        self.merge = merge
        self.active = {}
        self.blocked = {}
        self.oom_failures = {}
        self.start_failures = {name: 0 for name in DETECTORS}
        self.disabled = set()
        self.interrupted = False
        # Resume revalidates chunk caches; already saved images never need inference again.
        self.state = {"schema_version": 1, "status": "running", "created_utc": now(),
                      "manager_pid": os.getpid(), "manager_token": process_token(os.getpid()),
                      "max_parallel": plan["max_parallel"], "allowed_gpu_uuids": sorted(allowed),
                      "total_images": len(plan["images"]), "chunk_size": plan["chunk_size"],
                      "gpu_error": None, "workers": [], "outputs": {},
                      "jobs": [{**job, "status": "pending", "attempts": 0, "oom_count": 0,
                                "worker_id": None, "message": "Waiting for an eligible GPU"}
                               for job in plan["jobs"]]}
        self.jobs = {job["id"]: job for job in self.state["jobs"]}

    def save(self):
        self.state["updated_utc"] = now()
        self.state["counts"] = {
            name: {status: sum(j["stop"] - j["start"] for j in self.state["jobs"]
                               if j["detector"] == name and j["status"] == status)
                   for status in ("pending", "running", "succeeded", "failed")}
            for name in DETECTORS}
        write_json(self.directory / "state.json", self.state)

    def event(self, message):
        print(f"{now()} {message}", flush=True)

    def pending(self):
        return {name: sum(j["status"] == "pending" and j["detector"] == name
                          for j in self.state["jobs"]) for name in DETECTORS}

    def command(self, worker, action, job_id=None):
        token = uuid.uuid4().hex
        payload = {"token": token, "action": action}
        if job_id is not None:
            payload["job_id"] = job_id
        write_json(self.directory / "workers" / worker["id"] / "command.json", payload)
        worker["command_token"] = token
        worker["activity_time"] = self.clock()

    def stop_worker(self, worker):
        if worker["status"] not in {"stopping", "oom", "error", "exited"}:
            self.command(worker, "stop")
            worker.update(status="stopping", message="Unloading model")

    def launch(self, gpu, name):
        worker_id = "w-" + uuid.uuid4().hex[:10]
        (self.directory / "workers" / worker_id).mkdir(parents=True)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu.uuid, PYTHONUNBUFFERED="1", TERM="xterm-256color")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        command = [sys.executable, "-u", "-m", "src.Scripts.extract_pose_queue",
                   "_worker", str(self.directory), worker_id, name]
        process = self.process_factory(command, Path(self.plan["project_root"]), env,
                                       self.directory / "logs" / f"{worker_id}.log")
        worker = {"id": worker_id, "detector": name, "status": "loading",
                  "gpu": {"index": gpu.index, "uuid": gpu.uuid}, "pid": process.pid,
                  "process_token": process_token(process.pid), "job_id": None,
                  "command_token": None, "reserved_mib": 0, "activity_time": self.clock(),
                  "message": "Loading model"}
        self.active[worker_id] = process
        self.state["workers"].append(worker)
        self.event(f"{worker_id}: loading {name} on GPU {gpu.index}")
        self.save()  # Record ownership before the next potentially slow operation.

    def disable_detector(self, name, message):
        self.disabled.add(name)
        for job in self.state["jobs"]:
            if job["detector"] == name and job["status"] == "pending":
                job.update(status="failed", message=message)

    def failure(self, worker, message, *, oom=False, unexpected=False):
        name = worker["detector"]
        worker.update(status="oom" if oom else "error", message=message)
        key = (worker["gpu"]["uuid"], name)
        if oom:
            self.oom_failures[key] = self.oom_failures.get(key, 0) + 1
        cooldown = 300 * 2 ** min(self.oom_failures.get(key, 1) - 1, 3) if oom else 60
        self.blocked[key] = self.clock() + cooldown
        job = self.jobs.get(worker.get("job_id"))
        if job and job["status"] == "running":
            if oom:
                job["oom_count"] += 1
            retry = (oom and job["oom_count"] < 6) or (unexpected and job["attempts"] < 3)
            job.update(status="pending" if retry else "failed", worker_id=None,
                       message=message + ("; saved rows retained for retry" if retry else ""))
        elif not job and not oom and not unexpected:
            self.start_failures[name] += 1
            # OOM is GPU-specific and must not cancel work on healthy GPUs.
            # Missing dependencies or bad detector configuration cannot recover by retrying.
            self.disable_detector(name, message)
        worker["job_id"] = None
        self.event(f"{worker['id']}: {worker['status']}: {message}")

    def consume_reply(self, worker):
        path = self.directory / "workers" / worker["id"] / "reply.json"
        if not path.exists():
            return
        reply = json.loads(path.read_text(encoding="utf-8"))
        event, token = reply.get("event"), reply.get("token")
        startup = worker["status"] == "loading" and token is None
        assigned = worker["status"] == "busy" and token == worker["command_token"]
        if not (startup or assigned):
            return
        worker["reserved_mib"] = max(0, float(reply.get("reserved_mib", 0)))
        if event == "ready" and startup:
            worker.update(status="ready", message="Ready", activity_time=self.clock())
        elif event == "done" and assigned:
            job = self.jobs[worker["job_id"]]
            if reply.get("job_id") != job["id"]:
                raise ValueError("Worker replied for the wrong chunk")
            job.update(status="succeeded", summary=reply.get("summary", {}),
                       message="Complete and cached")
            self.event(f"{worker['id']}: completed {job['id']}")
            worker.update(status="ready", job_id=None, message="Ready", activity_time=self.clock())
        elif event in {"oom", "error"}:
            self.failure(worker, reply.get("error", event), oom=event == "oom")

    def replies_and_exits(self):
        for worker in self.state["workers"]:
            process = self.active.get(worker["id"])
            if process is None:
                continue
            self.consume_reply(worker)
            try:
                code = process.poll()
                if code is None and self.clock() - worker["activity_time"] > 1800:
                    process.terminate()
                    self.failure(worker, "No chunk completion within 30 minutes", unexpected=True)
                    code = process.poll()
                if code is None:
                    continue
                process.wait()
                # A worker may publish its final OOM/done reply between our first
                # read and poll(). Consume it before treating the exit as a crash.
                self.consume_reply(worker)
            except Exception as error:
                # The process wrapper may raise a log-capture error; stop its owned child.
                try:
                    process.terminate()
                finally:
                    self.failure(worker, f"Worker process error: {error}", unexpected=True)
                code = -1
            if worker["status"] not in {"stopping", "oom", "error"}:
                self.failure(worker, f"Worker exited unexpectedly ({code})", oom=code == 2, unexpected=True)
            worker.update(status="exited", exit_code=code, finished_utc=now())
            del self.active[worker["id"]]

    def step(self, control=None):
        self.replies_and_exits()
        stopping = self.interrupted or (control or {}).get("action") == "stop"
        if stopping:
            self.state["status"] = "stopping"
            for worker in self.state["workers"]:
                if worker["id"] in self.active and worker["status"] != "busy":
                    self.stop_worker(worker)
            if not self.active:
                self.state["status"] = "stopped"
                self.save()
                return True
            self.save()
            return False

        remaining = [j for j in self.state["jobs"] if j["status"] in {"pending", "running"}]
        if not remaining:
            for worker in self.state["workers"]:
                if worker["id"] in self.active:
                    self.stop_worker(worker)
            if self.active:
                self.save()
                return False
            self.state["status"] = "merging"
            self.save()
            for name in DETECTORS:
                if any(j["detector"] == name and j["status"] != "succeeded" for j in self.state["jobs"]):
                    continue
                self.event(f"Verifying and merging {name} cache")
                try:
                    self.state["outputs"][name] = self.merge(self.plan, name)
                except Exception as error:
                    self.state["outputs"][name] = {"error": str(error)}
                    self.event(f"{name} merge failed: {error}")
            good = len(self.state["outputs"]) == len(DETECTORS) and all(
                "error" not in value for value in self.state["outputs"].values())
            self.state["status"] = "finished" if good else "finished_with_errors"
            self.save()
            return True

        try:
            gpus = {g.uuid: g for g in self.gpu_reader() if g.uuid in self.allowed}
            self.state["gpu_error"] = None
        except Exception as error:
            self.state["gpu_error"] = str(error)
            self.save()
            return False  # Telemetry failure never means a GPU is free.
        pending = self.pending()
        for worker in self.state["workers"]:
            if worker["id"] not in self.active or worker["status"] != "ready":
                continue
            gpu = gpus.get(worker["gpu"]["uuid"])
            if gpu is None:
                self.stop_worker(worker)
                continue
            blocked = self.disabled | {name for name in DETECTORS
                        if self.blocked.get((gpu.uuid, name), 0) > self.clock()}
            name = choose_detector(pending, gpu.free_mib, self.plan["min_free_mib"],
                                   resident=worker["detector"], reserved_mib=worker["reserved_mib"],
                                   blocked=blocked)
            if name != worker["detector"]:
                self.stop_worker(worker)
                continue
            job = next(j for j in self.state["jobs"] if j["status"] == "pending" and j["detector"] == name)
            job.update(status="running", worker_id=worker["id"], attempts=job["attempts"] + 1,
                       message=f"GPU {gpu.index}: {worker['id']}")
            worker.update(status="busy", job_id=job["id"], message=f"Extracting {job['id']}")
            self.command(worker, "run", job["id"])
            pending[name] -= 1
        busy = {w["gpu"]["uuid"] for w in self.state["workers"] if w["id"] in self.active}
        # Reserve a pending chunk for each model still loading, preventing a tail
        # of unnecessary model launches when only one chunk remains.
        loading = _loading_counts(self.state["workers"], self.active)
        launchable = {name: max(0, pending[name] - loading[name]) for name in DETECTORS}
        for gpu in sorted(gpus.values(), key=lambda g: (g.utilization, -g.free_mib, g.index)):
            if len(self.active) >= self.plan["max_parallel"]:
                break
            if gpu.uuid in busy:
                continue
            blocked = self.disabled | {name for name in DETECTORS
                        if self.blocked.get((gpu.uuid, name), 0) > self.clock()}
            name = choose_detector(launchable, gpu.free_mib, self.plan["min_free_mib"], blocked=blocked)
            if name is None:
                continue
            self.launch(gpu, name)
            launchable[name] -= 1
            busy.add(gpu.uuid)
        self.save()
        return False

    def run(self):
        self.save()
        try:
            while True:
                path = self.directory / "control.json"
                control = json.loads(path.read_text()) if path.exists() else {}
                if self.step(control):
                    return 0 if self.state["status"] in {"finished", "stopped"} else 1
                time.sleep(.5)
        except BaseException as error:
            self.state.update(status="failed", error=str(error))
            raise
        finally:
            # Only owned process handles are terminated, never arbitrary GPU PIDs.
            for worker in self.state["workers"]:
                process = self.active.get(worker["id"])
                if process is not None:
                    try:
                        process.terminate()
                    except Exception as error:
                        worker["message"] = f"Cleanup error: {error}"
            self.save()


def _loading_counts(workers, active):
    return {name: sum(w["id"] in active and w["status"] == "loading" and w["detector"] == name
                      for w in workers) for name in DETECTORS}


def run_manager(directory):
    import fcntl
    directory = Path(directory).resolve()
    with (directory / ".manager.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This extraction queue already has an active manager") from error
        state_path = directory / "state.json"
        previous = json.loads(state_path.read_text()) if state_path.exists() else {}
        for worker in previous.get("workers", []):
            pid, token = worker.get("pid"), worker.get("process_token")
            if pid and token and process_token(pid) == token:
                raise RuntimeError(f"Previous worker {worker['id']} is still alive (PID {pid}); "
                                   "let it finish its chunk before resuming")
        plan = json.loads((directory / "plan.json").read_text())
        if Path(plan["directory"]).resolve() != directory:
            raise ValueError("Extraction directory moved; preserve the plan's original location")
        if sha256_file(plan["annotations"]) != plan["annotation_sha256"]:
            raise ValueError("Annotations changed since the extraction plan was created")
        if any(plan["detectors"][name] != detector_config(name) for name in DETECTORS):
            raise ValueError("Detector configuration changed; use a new output directory")
        runtime = json.loads((directory / "runtime.json").read_text())
        allowed = runtime["allowed_gpu_uuids"]
        if not allowed:
            raise ValueError("No GPUs allowed by this launch")
        identity = {"manager_pid": os.getpid(), "manager_token": process_token(os.getpid())}
        write_json(directory / "manager_identity.json", identity)
        manager = ExtractionManager(directory, plan, allowed)
        def interrupt(signum, frame):
            manager.interrupted = True
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        return manager.run()
