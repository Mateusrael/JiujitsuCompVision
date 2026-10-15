"""Start a concurrent GPU queue and inspect individual live training terminals."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.Train.queue_plan import identifier, prepare_plan
from src.Train.run_metadata import write_json

PROJECT = Path(__file__).resolve().parents[2]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, default=PROJECT / "diagnostics" / "queues")
    commands = parser.add_subparsers(dest="action", required=True,
                                    metavar="{start,status,logs,pause,resume,stop}")
    start = commands.add_parser("start", help="Validate a plan and start the manager in the background")
    start.add_argument("plan", type=Path)
    start.add_argument("--name", help="Unique queue name; defaults to UTC timestamp")
    start.add_argument("--max-parallel", type=int, help="Override the plan's limit (default six)")
    start.add_argument("--trainings-dir", type=Path,
                       default=Path(os.environ.get("TRAININGS_DIR", PROJECT / "trainings")))
    start.add_argument("--dry-run", action="store_true", help="Validate and print commands without probing GPUs or starting anything")
    status = commands.add_parser("status", help="List jobs and assigned GPUs")
    status.add_argument("name", nargs="?", help="Queue name; defaults to latest")
    status.add_argument("--watch", action="store_true", help="Refresh every two seconds; Ctrl+C detaches")
    logs = commands.add_parser("logs", help="Follow one run's captured tqdm terminal; Ctrl+C detaches")
    logs.add_argument("job", help="Job id from the plan or status output")
    logs.add_argument("--name", help="Queue name; defaults to latest")
    logs.add_argument("--tail-bytes", type=int, default=65536)
    for action in ("pause", "resume", "stop"):
        sub = commands.add_parser(action, help={"pause": "Pause new launches; active runs continue",
                                               "resume": "Unpause a live queue",
                                               "stop": "Cancel pending jobs; let active runs finish"}[action])
        sub.add_argument("name", nargs="?")
        if action == "stop":
            sub.add_argument("--terminate", action="store_true", help="Also terminate this queue's active runs")
    manager = commands.add_parser("_manager")
    manager.add_argument("directory", type=Path)
    return parser


def resolve_queue(root, name):
    if name is None:
        latest = root / "latest.json"
        if not latest.is_file():
            raise ValueError("No queue has been started; specify a queue name or use start")
        name = json.loads(latest.read_text())["name"]
    return root / identifier(name)


def read_state(directory):
    path = directory / "state.json"
    if not path.is_file():
        raise ValueError(f"Queue has no state yet: {directory}. Check manager.log")
    state = json.loads(path.read_text())
    identity = directory / "manager_identity.json"
    if state["status"] == "starting" and identity.is_file():
        state.update(json.loads(identity.read_text()))
    return state


def start_queue(args):
    from src.Train.queue_gpu import discover_allowed_gpus, query_gpus
    name = identifier(args.name or datetime.now(timezone.utc).strftime("queue-%Y%m%dT%H%M%SZ-") + str(os.getpid()))
    plan = prepare_plan(args.plan, PROJECT, name, args.trainings_dir, args.max_parallel)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if sys.platform != "linux":
        raise ValueError("Starting a training queue requires Linux; use --dry-run locally")
    allowed = discover_allowed_gpus(sys.executable, PROJECT)
    available = [g for g in query_gpus() if g.uuid in allowed]
    if not available:
        raise ValueError("No usable GPUs match PyTorch visibility and nvidia-smi; check CUDA masks and GPU access")
    maximum = max(g.total_mib for g in available)
    for job in plan["jobs"]:
        if job["min_free_mib"] > maximum:
            raise ValueError(f"{job['id']}: min_free_mib exceeds all accessible GPUs' total memory")
    plan["allowed_gpu_uuids"] = sorted(allowed)
    root = args.queue_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    directory = root / name
    directory.mkdir()  # Never reuse a queue's state or overwrite its logs.
    (directory / "logs").mkdir()
    write_json(directory / "plan.json", plan)
    write_json(directory / "state.json", {"name": name, "status": "starting", "jobs": [],
                                          "max_parallel": plan["max_parallel"]})
    try:
        with (directory / "manager.log").open("ab", buffering=0) as log:
            process = subprocess.Popen([sys.executable, "-u", "-m", "src.Scripts.training_queue", "_manager", str(directory)],
                                       cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
    except OSError as error:
        state = read_state(directory)
        state.update(status="failed", error=f"Could not start manager: {error}")
        write_json(directory / "state.json", state)
        raise
    from src.Train.queue_manager import process_token
    write_json(directory / "manager_identity.json", {"manager_pid": process.pid,
                                                     "manager_token": process_token(process.pid)})
    # The detached manager initializes quickly; GPU discovery already happened above.
    for _ in range(50):
        state = read_state(directory)
        if state["status"] != "starting" or process.poll() is not None:
            break
        time.sleep(.1)
    if process.poll() not in (None, 0) or state["status"] == "failed":
        if state["status"] == "starting":
            state.update(status="failed", error="Manager exited before initialization")
            write_json(directory / "state.json", state)
        raise RuntimeError(f"Queue could not start. See {directory / 'manager.log'}")
    write_json(root / "latest.json", {"name": name})
    if state["status"] == "starting":
        print(f"Queue {name}: manager still starting; readiness is not yet confirmed. Check status or manager.log.")
    else:
        print(f"Queue {name}: {len(plan['jobs'])} jobs, up to {plan['max_parallel']} at once.")
    print(f"State and logs: {directory}")
    print("Use QUEUE.sh status --watch, or QUEUE.sh logs JOB_ID. Ctrl+C detaches either viewer.")
    return 0


def print_status(state):
    from src.Train.queue_manager import manager_alive
    status = state["status"]
    print(f"Queue {state['name']} | {status} | maximum {state['max_parallel']} concurrent")
    if status in {"starting", "running", "paused", "draining"} and not manager_alive(state):
        print("MANAGER OFFLINE: job states may be stale. Check existing processes before starting another queue.")
    if state.get("error"):
        print(state["error"])
    if state.get("gpu_error"):
        print(f"GPU telemetry: {state['gpu_error']}")
    print(f"{'JOB':30} {'STATUS':11} {'GPU':5} {'PID':9} DETAILS")
    for job in state["jobs"]:
        gpu = str(job["gpu"]["index"]) if job.get("gpu") else "-"
        print(f"{job['id']:30} {job['status']:11} {gpu:5} {str(job.get('pid') or '-'):9} {job['message']}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.action == "_manager":
        try:
            from src.Train.queue_manager import run_manager
            return run_manager(args.directory)
        except Exception as error:
            state = read_state(args.directory)
            state.update(status="failed", error=str(error))
            write_json(args.directory / "state.json", state)
            raise
    try:
        if args.action == "start":
            return start_queue(args)
        directory = resolve_queue(args.queue_root.resolve(), args.name)
        if args.action == "status":
            try:
                while True:
                    state = read_state(directory)
                    if args.watch and sys.stdout.isatty():
                        print("\033[2J\033[H", end="")
                    print_status(state)
                    if not args.watch or state["status"] in {"finished", "finished_with_errors", "stopped", "failed"}:
                        break
                    time.sleep(2)
            except KeyboardInterrupt:
                print("\nDetached; queue unchanged.")
            return 0
        if args.action == "logs":
            from src.Train.queue_manager import FINAL, manager_alive
            from src.Train.queue_process import follow_log
            state = read_state(directory)
            job = next((j for j in state["jobs"] if j["id"] == args.job), None)
            if job is None:
                raise ValueError(f"Unknown job {args.job}; use status to list job ids")
            if args.tail_bytes < 0:
                raise ValueError("tail-bytes must be nonnegative")
            def finished():
                current = read_state(directory)
                entry = next(j for j in current["jobs"] if j["id"] == args.job)
                return entry["status"] in FINAL or not manager_alive(current)
            print(f"Following {args.job}. Ctrl+C detaches. Captured terminal width: 120 columns.", flush=True)
            follow_log(directory / "logs" / f"{args.job}.log", finished, tail_bytes=args.tail_bytes)
            print("\nViewer closed; training state unchanged.")
            return 0
        from src.Train.queue_manager import manager_alive
        state = read_state(directory)
        if not manager_alive(state) or state["status"] in {"finished", "finished_with_errors", "stopped", "failed"}:
            raise ValueError("This queue is no longer active; completed queues cannot be restarted")
        existing = directory / "control.json"
        if existing.exists() and json.loads(existing.read_text()).get("action") in {"stop", "terminate"}:
            if args.action != "stop" or not args.terminate:
                raise ValueError("A stop is already pending; it cannot be undone")
        action = "terminate" if args.action == "stop" and args.terminate else "run" if args.action == "resume" else args.action
        # Unique writer temp avoids collisions between separate control terminals.
        temp = directory / f"control-{os.getpid()}.json"
        write_json(temp, {"action": action})
        temp.replace(existing)
        print(f"Requested {action} for {state['name']}.")
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Queue error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
