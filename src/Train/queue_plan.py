"""Validate a queue of training arguments without importing PyTorch."""

import argparse
import json
import re
import sys
from pathlib import Path

from src.Scripts.train import build_parser
from src.Train.config import resolve_settings


IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
PATH_ARGUMENTS = {"annotations", "split", "images_dir", "resume", "init_from"}


def identifier(value):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("Names must use 1-64 letters, digits, dots, underscores or hyphens")
    return value


def positive_int(value, name, minimum=1, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, int) or value < minimum
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{name} must be an integer from {minimum}"
                         + (f" to {maximum}" if maximum is not None else " upward"))
    return value


def arguments(values):
    """Turn structured options into exact argv, rejecting unknown keys and types."""
    parser = build_parser()
    actions = {a.dest: a for a in parser._actions if a.dest != "help"}
    argv = []
    for key, value in values.items():
        if key not in actions or key in {"device", "run_dir", "progress"}:
            raise ValueError(f"Unsupported queue training option: {key}")
        action = actions[key]
        option = "--" + key.replace("_", "-")
        if isinstance(action, argparse.BooleanOptionalAction):
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean")
            argv.append(option if value else "--no-" + key.replace("_", "-"))
        else:
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise ValueError(f"{key} must be a scalar argument")
            if action.type is int and not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
            if action.type is float and not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a number")
            if action.type is None and not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            argv += [option, str(value)]
    return argv


def prepare_plan(path, project_root, queue_name, trainings_dir=None, max_parallel=None):
    project_root = Path(project_root).resolve()
    trainings_dir = Path(trainings_dir or project_root / "trainings").resolve()
    identifier(queue_name)
    source = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("Queue plan must be a JSON object")
    unknown = set(source) - {"version", "max_parallel", "poll_seconds", "max_gpu_utilization", "defaults", "jobs"}
    if unknown or type(source.get("version")) is not int or source.get("version") != 1:
        raise ValueError(f"Expected queue version 1; unknown fields: {sorted(unknown)}")
    parallel = positive_int(max_parallel if max_parallel is not None else source.get("max_parallel", 6), "max_parallel")
    poll = positive_int(source.get("poll_seconds", 10), "poll_seconds")
    utilization = positive_int(source.get("max_gpu_utilization", 100), "max_gpu_utilization", 0, 100)
    defaults = source.get("defaults", {})
    jobs = source.get("jobs")
    if not isinstance(defaults, dict) or not isinstance(jobs, list) or not jobs:
        raise ValueError("defaults must be an object and jobs must be a nonempty list")
    arguments(defaults)
    result = []
    seen_ids, seen_outputs = set(), set()
    for job in jobs:
        if not isinstance(job, dict) or set(job) - {"id", "train", "min_free_mib"}:
            raise ValueError("Each job accepts id, train and optional min_free_mib")
        job_id = identifier(job.get("id"))
        if job_id in seen_ids:
            raise ValueError(f"Duplicate job id: {job_id}")
        seen_ids.add(job_id)
        if not isinstance(job.get("train"), dict):
            raise ValueError(f"{job_id}: train must be an object")
        values = {**defaults, **job["train"]}
        for key in PATH_ARGUMENTS & values.keys():
            if not isinstance(values[key], str) or not values[key]:
                raise ValueError(f"{job_id}: {key} requires a path")
            value = Path(values[key]).expanduser()
            value = (project_root / value).resolve() if not value.is_absolute() else value.resolve()
            if not (value.is_dir() if key == "images_dir" else value.is_file()):
                raise FileNotFoundError(value)
            values[key] = str(value)
        argv = arguments(values)
        try:
            args = build_parser().parse_args(argv)
        except SystemExit as error:
            raise ValueError(f"Invalid training arguments for {job_id}") from error
        if not args.resume and not args.init_from:
            if not args.model or not args.annotations or not args.split:
                raise ValueError(f"{job_id}: fresh training requires model, annotations and split")
            resolve_settings(args)
            if args.model.startswith("image") and not args.images_dir:
                raise ValueError(f"{job_id}: image training requires images_dir")
        if args.resume:
            checkpoint = Path(args.resume)
            if checkpoint.name != "last.pt" or checkpoint.parent.name != "checkpoints":
                raise ValueError("resume requires <run>/checkpoints/last.pt")
            run_dir = checkpoint.parent.parent
        else:
            run_dir = trainings_dir / f"{queue_name}-{job_id}"
            if run_dir.exists():
                raise FileExistsError(f"Fresh run directory already exists: {run_dir}")
            argv += ["--run-dir", str(run_dir)]
        if str(run_dir) in seen_outputs:
            raise ValueError(f"Two queue jobs would write the same run: {run_dir}")
        seen_outputs.add(str(run_dir))
        # Resume may omit model; choose the conservative image threshold then.
        default_memory = 16384 if args.model in (None, "image-dinov3") else 8192 if args.model == "image" else 4096
        minimum = positive_int(job.get("min_free_mib", default_memory), "min_free_mib")
        command = [sys.executable, "-u", "-m", "src.Scripts.train", *argv, "--device", "cuda", "--progress"]
        result.append({"id": job_id, "command": command, "run_dir": str(run_dir),
                       "min_free_mib": minimum, "model": args.model or "from-checkpoint"})
    return {"version": 1, "name": queue_name, "project_root": str(project_root),
            "max_parallel": parallel, "poll_seconds": poll,
            "max_gpu_utilization": utilization, "jobs": result}
