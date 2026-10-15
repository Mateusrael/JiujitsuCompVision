"""Concurrent single-host queue: one owned training per selected GPU."""

import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from src.Train.queue_gpu import choose_gpu, query_gpus
from src.Train.queue_process import TrainingProcess
from src.Train.run_metadata import write_json


FINAL = {"succeeded", "failed", "cancelled"}


def now():
    return datetime.now(timezone.utc).isoformat()


def process_token(pid):
    """Linux boot + process start ticks prevent confusing a reused PID with ours."""
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] in {"Z", "X"}:
            return None
        return boot + ":" + stat[19]
    except (OSError, IndexError):
        return None


def manager_alive(state):
    pid = state.get("manager_pid")
    token = state.get("manager_token")
    return bool(pid and token and process_token(pid) == token)


class QueueManager:
    def __init__(self, directory, plan, gpu_reader=query_gpus, process_factory=TrainingProcess):
        self.directory = Path(directory)
        self.plan = plan
        self.gpu_reader = gpu_reader
        self.process_factory = process_factory
        self.active = {}
        self.interrupted = False
        self.state = {"name": plan["name"], "created_utc": now(), "updated_utc": now(),
                      "manager_pid": os.getpid(), "manager_token": process_token(os.getpid()),
                      "status": "running", "max_parallel": plan["max_parallel"],
                      "allowed_gpu_uuids": plan["allowed_gpu_uuids"], "gpu_error": None,
                      "jobs": [{**job, "status": "pending", "gpu": None, "pid": None,
                                "exit_code": None, "message": "Waiting for a slot and eligible GPU"}
                               for job in plan["jobs"]]}

    def save(self):
        self.state["updated_utc"] = now()
        write_json(self.directory / "state.json", self.state)

    def event(self, message):
        print(f"{now()} {message}", flush=True)

    def step(self, control=None):
        """One scheduling iteration, injectable for CPU-only lifecycle tests."""
        control = control or {}
        action = control.get("action", "run")
        for job in self.state["jobs"]:
            if job["id"] not in self.active:
                continue
            process = self.active[job["id"]]
            try:
                if action == "terminate" or self.interrupted:
                    process.terminate()
                    job["status"] = "cancelled"
                code = process.poll()
                if code is None:
                    continue
                process.wait()
                job.update(exit_code=code, finished_utc=now())
                if job["status"] != "cancelled":
                    job["status"] = "succeeded" if code == 0 else "failed"
                job["message"] = f"Exited with code {code}"
            except Exception as error:
                # A capture error must not leave an untracked GPU process running.
                cleanup_error = self.stop_process(process)
                job.update(status="failed", message=str(error) + cleanup_error, finished_utc=now())
                token = job.get("process_token")
                if token and process_token(job["pid"]) == token:
                    # Do not refill this GPU if cleanup could not stop its training.
                    raise RuntimeError(f"Could not stop failed job {job['id']}; refusing further launches") from error
            del self.active[job["id"]]
            self.event(f"{job['id']}: {job['status']} ({job['message']})")
        stopping = action in {"stop", "terminate"} or self.interrupted
        if stopping:
            for job in self.state["jobs"]:
                if job["status"] == "pending":
                    job.update(status="cancelled", message="Cancelled before launch", finished_utc=now())
        elif action != "pause":
            # Scan in plan order, allowing a smaller job past one waiting for memory.
            for job in self.state["jobs"]:
                if len(self.active) >= self.plan["max_parallel"]:
                    break
                if job["status"] != "pending":
                    continue
                busy = {j["gpu"]["uuid"] for j in self.state["jobs"] if j["status"] == "running"}
                try:
                    gpu = choose_gpu(self.gpu_reader(), set(self.plan["allowed_gpu_uuids"]), busy,
                                     job["min_free_mib"], self.plan["max_gpu_utilization"])
                    self.state["gpu_error"] = None
                except Exception as error:
                    self.state["gpu_error"] = str(error)
                    job["message"] = "Waiting: GPU telemetry unavailable"
                    break
                if gpu is None:
                    job["message"] = f"Waiting for an unused queue GPU with >= {job['min_free_mib']} MiB free"
                    continue
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=gpu.uuid, PYTHONUNBUFFERED="1", TERM="xterm-256color")
                try:
                    process = self.process_factory(job["command"], Path(self.plan["project_root"]), env,
                                                   self.directory / "logs" / f"{job['id']}.log")
                except Exception as error:
                    job.update(status="failed", message=f"Launch failed: {error}", finished_utc=now())
                    self.event(f"{job['id']}: {job['message']}")
                    continue
                self.active[job["id"]] = process
                job.update(status="running", pid=process.pid, started_utc=now(),
                           process_token=process_token(process.pid),
                           gpu={"index": gpu.index, "uuid": gpu.uuid, "free_mib_at_start": gpu.free_mib,
                                "utilization_at_start": gpu.utilization}, message="Training")
                self.event(f"{job['id']}: started on GPU {gpu.index} ({gpu.uuid})")
                self.save()
        done = all(j["status"] in FINAL for j in self.state["jobs"])
        self.state["status"] = ("finished_with_errors" if any(j["status"] == "failed" for j in self.state["jobs"])
                                else "stopped" if any(j["status"] == "cancelled" for j in self.state["jobs"])
                                else "finished") if done else "draining" if stopping else "paused" if action == "pause" else "running"
        self.save()
        return done

    @staticmethod
    def stop_process(process):
        try:
            process.terminate()
            return ""
        except Exception as error:
            # A stopped child's capture failure may also be raised by terminate.
            return f"; cleanup reported: {error}"

    def run(self):
        self.save()
        try:
            while True:
                control_path = self.directory / "control.json"
                control = json.loads(control_path.read_text()) if control_path.exists() else {}
                if self.step(control):
                    return 1 if any(j["status"] == "failed" for j in self.state["jobs"]) else 0
                # Short sleeps allow a termination signal to be handled promptly.
                for _ in range(self.plan["poll_seconds"] * 5):
                    if self.interrupted or control_path.exists() and json.loads(control_path.read_text()) != control:
                        break
                    time.sleep(.2)
        finally:
            if self.active:
                # Every child must be attempted even if another one's log failed.
                for job in self.state["jobs"]:
                    process = self.active.get(job["id"])
                    if process is not None:
                        error = self.stop_process(process)
                        job.update(status="failed" if error else "cancelled",
                                   message="Manager exited; training terminated" + error,
                                   finished_utc=now())
                self.state["status"] = "failed"
                self.save()


def check_previous_manager(lock):
    """Refuse to overlap children surviving a previous supervisor crash."""
    lock.seek(0)
    raw = lock.read().strip()
    if not raw:
        return
    previous = Path(json.loads(raw)["directory"])
    state_path = previous / "state.json"
    if not state_path.exists():
        raise RuntimeError(f"Previous queue state is missing: {previous}; inspect it before clearing the queue lock record")
    state = json.loads(state_path.read_text())
    for job in state.get("jobs", []):
        pid, token = job.get("pid"), job.get("process_token")
        if pid and token and process_token(pid) == token:
            raise RuntimeError(f"Previous queue still owns live job {job['id']} (PID {pid}). "
                               f"Inspect {previous}; wait for or stop that training before starting another queue")


def run_manager(directory):
    import fcntl
    directory = Path(directory).resolve()
    plan = json.loads((directory / "plan.json").read_text())
    # All queues in this checkout share one supervisor, even with a custom output root.
    lock_path = Path(plan["project_root"]) / "diagnostics" / ".training-queue.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another queue is already active in this project") from error
        check_previous_manager(lock)
        lock.seek(0)
        lock.truncate()
        lock.write(json.dumps({"directory": str(directory)}))
        lock.flush()
        manager = QueueManager(directory, plan)
        def interrupt(signum, frame):
            manager.interrupted = True
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        return manager.run()
