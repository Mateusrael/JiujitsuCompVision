"""Run YOLO26 and ViTPose extraction through one resumable GPU chunk queue."""

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.Train.run_metadata import write_json

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT / "Data" / "pose_detections" / "full_dataset"
TERMINAL = {"finished", "finished_with_errors", "stopped", "failed"}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True,
                                    metavar="{start,status,logs,resume,stop}")
    start = commands.add_parser("start", help="Validate all annotations and start background extraction")
    start.add_argument("--annotations", type=Path, default=PROJECT / "Data" / "annotations.json")
    start.add_argument("--images-dir", type=Path, default=PROJECT / "Data" / "images")
    start.add_argument("--chunk-size", type=int, default=256)
    start.add_argument("--max-parallel", type=int, default=6)
    start.add_argument("--vitpose-min-free-mib", type=int, default=8192)
    start.add_argument("--yolo-min-free-mib", type=int, default=2048)
    start.add_argument("--dry-run", action="store_true", help="Validate and summarize without GPU probing or writing files")
    status = commands.add_parser("status", help="Show progress and current worker GPU assignments")
    status.add_argument("--watch", action="store_true", help="Refresh every two seconds; Ctrl+C detaches")
    logs = commands.add_parser("logs", help="Follow manager output or one worker's tqdm log")
    logs.add_argument("worker", nargs="?", default="manager", help="Worker ID from status, or manager")
    logs.add_argument("--tail-bytes", type=int, default=65536)
    resume = commands.add_parser("resume", help="Resume an offline or stopped queue, retaining cached detections")
    stop = commands.add_parser("stop", help="Stop assigning chunks and finish the active chunks")
    for sub in (start, status, logs, resume, stop):
        sub.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    manager = commands.add_parser("_manager")
    manager.add_argument("directory", type=Path)
    worker = commands.add_parser("_worker")
    worker.add_argument("directory", type=Path)
    worker.add_argument("worker_id")
    worker.add_argument("detector", choices=("yolo", "vitpose"))
    return parser


def read_state(directory):
    path = Path(directory) / "state.json"
    if not path.is_file():
        raise ValueError(f"No queue state at {path}; start a queue first")
    state = json.loads(path.read_text(encoding="utf-8"))
    identity = Path(directory) / "manager_identity.json"
    if state.get("status") == "starting" and identity.is_file():
        state.update(json.loads(identity.read_text(encoding="utf-8")))
    return state


def _manager_running(directory):
    from src.Train.queue_manager import manager_alive
    state = read_state(directory)
    if manager_alive(state):
        return True
    # The identity file also covers the interval before the new manager writes state.
    identity = Path(directory) / "manager_identity.json"
    return identity.is_file() and manager_alive(json.loads(identity.read_text(encoding="utf-8")))


@contextmanager
def _launch_lock(directory):
    """Serialize CLI launches so two resume commands cannot race."""
    import fcntl
    with (Path(directory) / ".launch.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another start/resume command is already launching this queue") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _available_gpus(plan):
    from src.Train.queue_gpu import discover_allowed_gpus, query_gpus
    allowed = discover_allowed_gpus(sys.executable, Path(plan["project_root"]))
    available = [gpu for gpu in query_gpus() if gpu.uuid in allowed]
    if not available:
        raise ValueError("No usable CUDA GPUs; check GPU access and CUDA_VISIBLE_DEVICES")
    maximum = max(gpu.total_mib for gpu in available)
    for detector, minimum in plan["min_free_mib"].items():
        if minimum > maximum:
            raise ValueError(f"{detector}: minimum free memory exceeds every accessible GPU's total memory")
    return sorted(gpu.uuid for gpu in available)


def _launch(directory, plan, allowed):
    from src.Train.queue_manager import process_token
    directory = Path(directory)
    if _manager_running(directory):
        raise ValueError("This extraction queue already has a live manager")
    write_json(directory / "runtime.json", {"allowed_gpu_uuids": allowed})
    write_json(directory / "control.json", {"action": "run"})
    # Keep previous jobs/workers so the manager can detect surviving workers.
    state = read_state(directory)
    state.update(status="starting", error=None)
    write_json(directory / "state.json", state)
    try:
        with (directory / "manager.log").open("ab", buffering=0) as log:
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", "src.Scripts.extract_pose_queue", "_manager", str(directory)],
                cwd=plan["project_root"], stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
    except OSError as error:
        state.update(status="failed", error=f"Could not start manager: {error}")
        write_json(directory / "state.json", state)
        raise
    # The manager also publishes its identity before starting workers. Use a
    # separate temporary name so concurrent atomic writes cannot share a .tmp.
    identity = directory / f"manager_identity-launch-{os.getpid()}.json"
    write_json(identity, {"manager_pid": process.pid,
                          "manager_token": process_token(process.pid)})
    identity.replace(directory / "manager_identity.json")
    for _ in range(50):
        state = read_state(directory)
        if state.get("status") != "starting" or process.poll() is not None:
            break
        time.sleep(.1)
    if process.poll() not in (None, 0) or state.get("status") == "failed":
        raise RuntimeError(f"Extraction manager could not start. See {directory / 'manager.log'}")
    print(f"Extraction queue: {directory}")
    print(f"{len(plan['images']):,} images per detector; {len(plan['jobs']):,} chunks; "
          f"up to {plan['max_parallel']} GPU workers.")
    if state.get("status") == "starting":
        print("Manager is still starting; check status or manager.log for readiness.")
    print("Use EXTRACT_POSES.sh status --watch, or EXTRACT_POSES.sh logs WORKER_ID.")
    print("The queue continues when this terminal or a log viewer closes.")


def start_queue(args):
    from src.Eval.extraction_queue import prepare_plan
    directory = args.output_dir.expanduser().resolve()
    plan = prepare_plan(directory, args.annotations, args.images_dir,
                        chunk_size=args.chunk_size, max_parallel=args.max_parallel,
                        vitpose_min_free_mib=args.vitpose_min_free_mib,
                        yolo_min_free_mib=args.yolo_min_free_mib)
    if args.dry_run:
        summary = {key: value for key, value in plan.items() if key not in {"images", "jobs"}}
        summary.update(image_count=len(plan["images"]),
                       chunks_per_detector=dict(Counter(job["detector"] for job in plan["jobs"])))
        print(json.dumps(summary, indent=2))
        return 0
    if sys.platform != "linux":
        raise ValueError("Background GPU extraction requires Linux; use --dry-run locally")
    if directory.exists():
        raise ValueError(f"Output directory already exists: {directory}; use resume or a new --output-dir")
    allowed = _available_gpus(plan)
    directory.mkdir(parents=True)
    (directory / "logs").mkdir()
    write_json(directory / "plan.json", plan)
    write_json(directory / "state.json", {"status": "starting", "jobs": [], "workers": [],
                                          "max_parallel": plan["max_parallel"]})
    with _launch_lock(directory):
        _launch(directory, plan, allowed)
    return 0


def resume_queue(directory):
    if sys.platform != "linux":
        raise ValueError("Background GPU extraction requires Linux")
    plan_path = directory / "plan.json"
    if not plan_path.is_file():
        raise ValueError(f"No extraction plan at {plan_path}")
    with _launch_lock(directory):
        if _manager_running(directory):
            raise ValueError("Queue manager is still running; wait for stop to finish before resuming")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        _launch(directory, plan, _available_gpus(plan))
    return 0


def _workers(state):
    workers = state.get("workers", [])
    return list(workers.values()) if isinstance(workers, dict) else workers


def print_status(state):
    from src.Train.queue_manager import manager_alive
    print(f"Pose extraction | {state.get('status', 'unknown')} | "
          f"maximum {state.get('max_parallel', '?')} GPU workers")
    if state.get("status") not in TERMINAL and not manager_alive(state):
        print("MANAGER OFFLINE: progress may be stale. Resume checks for surviving workers first.")
    for field in ("error", "gpu_error"):
        if state.get(field):
            print(f"{field}: {state[field]}")
    jobs = state.get("jobs", [])
    for detector in ("yolo", "vitpose"):
        selected = [job for job in jobs if job["detector"] == detector]
        counts = Counter(job["status"] for job in selected)
        done = [job for job in selected if job["status"] in {"done", "succeeded"}]
        images = sum(job.get("stop", 0) - job.get("start", 0) for job in done)
        total = sum(job.get("stop", 0) - job.get("start", 0) for job in selected)
        print(f"{detector}: {images:,}/{total:,} images in completed chunks; "
              f"{len(done)}/{len(selected)} chunks; " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"{'WORKER':12} {'DETECTOR':9} {'STATUS':11} {'GPU':5} {'PID':9} CHUNK / DETAILS")
    for worker in _workers(state):
        gpu = worker.get("gpu")
        index = gpu.get("index", "-") if isinstance(gpu, dict) else gpu if gpu is not None else "-"
        print(f"{worker['id']:12} {worker.get('detector', '-'):9} {worker.get('status', '-'):11} "
              f"{str(index):5} {str(worker.get('pid') or '-'):9} "
              f"{worker.get('job_id') or '-'} {worker.get('message') or ''}")


def show_logs(directory, worker_id, tail_bytes):
    from src.Train.queue_manager import manager_alive
    from src.Train.queue_process import follow_log
    if tail_bytes < 0:
        raise ValueError("tail-bytes must be nonnegative")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", worker_id):
        raise ValueError("Invalid worker ID; use status to list workers")
    state = read_state(directory)
    if worker_id != "manager" and not any(w["id"] == worker_id for w in _workers(state)):
        raise ValueError(f"Unknown worker {worker_id}; use status to list worker IDs")
    def finished():
        current = read_state(directory)
        if current.get("status") in TERMINAL or not manager_alive(current):
            return True
        if worker_id == "manager":
            return False
        worker = next((w for w in _workers(current) if w["id"] == worker_id), None)
        return worker is None or worker.get("status") in {"stopped", "exited", "failed", "oom", "finished"}
    path = directory / "manager.log" if worker_id == "manager" else directory / "logs" / f"{worker_id}.log"
    print(f"Following {worker_id}; Ctrl+C detaches without stopping extraction.", flush=True)
    follow_log(path, finished, tail_bytes=tail_bytes)
    print("\nViewer closed; extraction state unchanged.")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.action == "_manager":
        from src.Eval.extraction_queue import run_manager
        return run_manager(args.directory)
    if args.action == "_worker":
        from src.Eval.extraction_worker import run_worker
        return run_worker(args.directory, args.worker_id, args.detector)
    try:
        if args.action == "start":
            return start_queue(args)
        directory = args.output_dir.expanduser().resolve()
        if args.action == "resume":
            return resume_queue(directory)
        if args.action == "status":
            try:
                while True:
                    state = read_state(directory)
                    if args.watch and sys.stdout.isatty():
                        print("\033[2J\033[H", end="")
                    print_status(state)
                    if not args.watch or state.get("status") in TERMINAL:
                        break
                    time.sleep(2)
            except KeyboardInterrupt:
                print("\nDetached; extraction unchanged.")
            return 0
        if args.action == "logs":
            show_logs(directory, args.worker, args.tail_bytes)
            return 0
        if not _manager_running(directory):
            raise ValueError("Extraction manager is offline; there is no active manager to stop")
        temporary = directory / f"control-{os.getpid()}.json"
        write_json(temporary, {"action": "stop"})
        temporary.replace(directory / "control.json")
        print("Stop requested. Active chunks will finish; remaining work can be resumed.")
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Extraction queue error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
