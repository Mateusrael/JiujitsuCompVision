"""Persistent detector workers and verified, resumable extraction chunks.

The manager assigns one chunk at a time to a worker. A worker owns its chunk
files exclusively, keeps its detector resident, and never writes a final cache.
Final caches are published only after a streaming verification of every chunk.
"""

import hashlib
import json
import os
import re
import sys
import time
import traceback
import uuid
from pathlib import Path

from src.Dataset.annotations import sha256_file
from src.Eval.pose_detection import (ROOT as PROJECT_ROOT, VitPoseDetector, YoloDetector, _versions,
                                     detector_config, load_cache, validate_detection)
from src.Train.run_metadata import write_json


_IDENTIFIER = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]*\Z")
_IDENTITY_FIELDS = ("configuration", "packages", "weights", "annotation_sha256")


def _identifier(value):
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"Invalid extraction identifier: {value!r}")
    return value


def _image_path(plan, image):
    # The original annotation schema uses exactly seven digits without suffix.
    if not isinstance(image, str) or re.fullmatch(r"[0-9]{7}", image) is None:
        raise ValueError(f"Invalid original-dataset image ID: {image!r}")
    return Path(plan["images_dir"]) / f"{image}.jpg"


def _chunk_ids(plan, job):
    _identifier(job["id"])
    start, stop = job["start"], job["stop"]
    if (isinstance(start, bool) or not isinstance(start, int)
            or isinstance(stop, bool) or not isinstance(stop, int)
            or not 0 <= start < stop <= len(plan["images"])):
        raise ValueError(f"Invalid image interval for {job['id']}")
    images = plan["images"][start:stop]
    if len(set(images)) != len(images):
        raise ValueError(f"Duplicate image IDs in {job['id']}")
    return images


def chunk_paths(plan, job):
    base = Path(plan["directory"]) / "chunks" / _identifier(job["id"])
    return base.with_suffix(".jsonl"), base.with_suffix(".meta.json")


def _chunk_identity(plan, job):
    name = job["detector"]
    config = detector_config(name)
    if config != plan["detectors"][name]:
        raise ValueError(f"Detector configuration changed for {name}; use a new extraction directory")
    images = _chunk_ids(plan, job)
    return {
        "configuration": config,
        "annotation_sha256": plan["annotation_sha256"],
        "job_id": job["id"],
        "start": job["start"], "stop": job["stop"],
        "image_ids_sha256": hashlib.sha256("\n".join(images).encode("utf-8")).hexdigest(),
    }


def _require_metadata(actual, expected, path):
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"Extraction cache {key} changed: {path}; use a new extraction directory")


def _coverage(rows):
    coverage = {"0": 0, "1": 0, "2": 0}
    for row in rows:
        count = sum(any(joint[2] > 0 for joint in pose) for pose in row["poses"])
        coverage[str(count)] += 1
    return coverage


def extract_chunk(plan, job, detector, progress=True):
    """Resume a chunk with a resident detector and retain every zero-pose row.

    Completed rows survive an exception. Reusing them checks image content and
    the exact detector configuration, package versions, and model provenance.
    """
    if detector.name != job["detector"]:
        raise ValueError("Worker detector does not match its assigned chunk")
    started = time.perf_counter()
    images = _chunk_ids(plan, job)
    cache, meta_path = chunk_paths(plan, job)
    cache.parent.mkdir(parents=True, exist_ok=True)
    expected = {**_chunk_identity(plan, job),
                "packages": getattr(detector, "extraction_packages", None) or _versions(),
                "weights": detector.provenance}
    if meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        _require_metadata(metadata, expected, meta_path)
    else:
        if cache.exists() and cache.stat().st_size:
            raise ValueError(f"Unverified chunk cache: {cache}")
        metadata = {**expected, "complete": False,
                    "device": str(getattr(detector, "device", "unknown"))}
        write_json(meta_path, metadata)
    if metadata.get("complete"):
        expected_digest = metadata.get("cache_sha256")
        if not expected_digest or not cache.is_file() or sha256_file(cache) != expected_digest:
            raise ValueError(f"Completed chunk content changed: {job['id']}")
    gpu_uuid = os.environ.get("CUDA_VISIBLE_DEVICES")
    seen_gpus = set(metadata.get("gpu_uuids", []))
    if gpu_uuid:
        seen_gpus.add(gpu_uuid)
    metadata.update(gpu_uuid=gpu_uuid, gpu_uuids=sorted(seen_gpus))
    rows = load_cache(cache)
    if list(rows) != images[:len(rows)]:
        raise ValueError(f"Unexpected, reordered or extra image IDs in chunk {job['id']}")
    for image, row in rows.items():
        if sha256_file(_image_path(plan, image)) != row["image_sha256"]:
            raise ValueError(f"Image content changed for {image}; use a new extraction directory")
    # Preserve all contributing GPU UUIDs even if the next inference raises OOM.
    write_json(meta_path, metadata)
    reused = len(rows)
    iterable = images[reused:]
    bar = None
    if progress:
        from tqdm.auto import tqdm
        bar = tqdm(iterable, total=len(images), initial=reused,
                   desc=job["id"], unit="image", dynamic_ncols=True)
        iterable = bar
    try:
        with cache.open("a", encoding="utf-8", newline="\n") as handle:
            for image in iterable:
                path = _image_path(plan, image)
                digest = sha256_file(path)
                before = time.perf_counter()
                poses = detector(path)
                row = {"image": image, "poses": poses,
                       "seconds": time.perf_counter() - before,
                       "image_sha256": digest}
                validate_detection(row)
                if sha256_file(path) != digest:
                    raise ValueError(f"Image changed during detection: {image}")
                handle.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                handle.flush()
                rows[image] = row
            os.fsync(handle.fileno())
    finally:
        if bar is not None:
            bar.close()
    summary = {"images": len(images), "reused_images": reused,
               "new_images": len(images) - reused,
               "detection_seconds": sum(row["seconds"] for row in rows.values()),
               "chunk_wall_seconds": time.perf_counter() - started,
               "coverage": _coverage(rows.values()), "cache_sha256": sha256_file(cache)}
    write_json(meta_path, {**metadata, **summary, "complete": True})
    return summary


def merge_detector(plan, name):
    """Verify all planned chunks and atomically publish one training JSONL.

    Only one chunk's poses are held in memory. Missing, partial, reordered,
    conflicting and stale-image chunks fail before the previous final file is
    replaced. Configuration and image hashes remain usable by detect_records.
    """
    expected_config = detector_config(name)
    if plan["detectors"][name] != expected_config:
        raise ValueError(f"Detector configuration changed for {name}")
    if sha256_file(plan["annotations"]) != plan["annotation_sha256"]:
        raise ValueError("Annotations changed since extraction was planned")
    jobs = sorted((job for job in plan["jobs"] if job["detector"] == name),
                  key=lambda job: job["start"])
    if not jobs:
        raise ValueError(f"No chunks planned for {name}")
    target = Path(plan["directory"]) / f"{name}.jsonl"
    meta_target = target.with_suffix(".meta.json")
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    started = time.perf_counter()
    reference_metadata = None
    digest = hashlib.sha256()
    count = 0
    seconds = 0.0
    coverage = {"0": 0, "1": 0, "2": 0}
    devices = set()
    gpu_uuids = set()
    try:
        with temporary.open("xb") as output:
            for job in jobs:
                if job["start"] != count:
                    raise ValueError(f"Gap or overlap in {name} chunk plan")
                cache, meta_path = chunk_paths(plan, job)
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                _require_metadata(metadata, _chunk_identity(plan, job), meta_path)
                if not metadata.get("complete") or not metadata.get("cache_sha256"):
                    raise ValueError(f"Incomplete chunk: {job['id']}")
                if reference_metadata is None:
                    reference_metadata = {key: metadata[key] for key in _IDENTITY_FIELDS}
                else:
                    _require_metadata(metadata, reference_metadata, meta_path)
                devices.add(metadata.get("device", "unknown"))
                gpu_uuids.update(metadata.get("gpu_uuids", []))
                if metadata.get("gpu_uuid"):
                    gpu_uuids.add(metadata["gpu_uuid"])
                chunk_digest = hashlib.sha256()
                expected_images = _chunk_ids(plan, job)
                chunk_count = 0
                with cache.open("rb") as handle:
                    for line in handle:
                        if not line.endswith(b"\n"):
                            raise ValueError(f"Interrupted chunk: {job['id']}")
                        row = json.loads(line)
                        validate_detection(row)
                        if chunk_count >= len(expected_images) or row["image"] != expected_images[chunk_count]:
                            raise ValueError(f"Unexpected image IDs in chunk {job['id']}")
                        if sha256_file(_image_path(plan, row["image"])) != row["image_sha256"]:
                            raise ValueError(f"Image content changed for {row['image']}")
                        output.write(line)
                        digest.update(line)
                        chunk_digest.update(line)
                        seconds += row["seconds"]
                        detected = sum(any(joint[2] > 0 for joint in pose) for pose in row["poses"])
                        coverage[str(detected)] += 1
                        chunk_count += 1
                if chunk_count != len(expected_images):
                    raise ValueError(f"Missing detections in chunk {job['id']}")
                if chunk_digest.hexdigest() != metadata["cache_sha256"]:
                    raise ValueError(f"Chunk content changed: {job['id']}")
                count += chunk_count
            if count != len(plan["images"]):
                raise ValueError(f"Incomplete {name} dataset: {count}/{len(plan['images'])}")
            output.flush()
            os.fsync(output.fileno())
        if meta_target.is_file():
            _require_metadata(json.loads(meta_target.read_text(encoding="utf-8")),
                              reference_metadata, meta_target)
        summary = {"images": count, "coverage": coverage, "detection_seconds": seconds,
                   "cache_sha256": digest.hexdigest(), "chunks": len(jobs),
                   "merge_wall_seconds": time.perf_counter() - started,
                   "cache_path": str(target.resolve())}
        # Publishing data then metadata leaves no partially written JSONL. A
        # manager restart can repeat this deterministic merge after either step.
        temporary.replace(target)
        write_json(meta_target, {**reference_metadata, **summary, "complete": True,
                                "device": "multiple", "devices": sorted(devices),
                                "gpu_uuids": sorted(gpu_uuids),
                                "population": "all_original_annotations"})
        return summary
    finally:
        temporary.unlink(missing_ok=True)


def _memory_stats():
    torch = sys.modules.get("torch")
    if torch is None:
        return {}
    try:
        return {"peak_allocated_mib": int(torch.cuda.max_memory_allocated() / 1024 ** 2),
                "reserved_mib": int(torch.cuda.memory_reserved() / 1024 ** 2),
                "peak_reserved_mib": int(torch.cuda.max_memory_reserved() / 1024 ** 2)}
    except Exception:
        return {}


def _is_oom(error):
    # torch.AcceleratorError can report CUDA OOM without OutOfMemoryError type.
    return "out of memory" in str(error).lower() or "cudaerrormemoryallocation" in str(error).lower()


def _build_detector(name, device):
    """Serialize YOLO's shared weight download/read across extraction workers."""
    if name != "yolo":
        if name != "vitpose":
            raise ValueError(f"Unknown pose detector: {name}")
        return VitPoseDetector(device)
    if sys.platform != "linux":
        return YoloDetector(device)
    import fcntl
    weights = PROJECT_ROOT / "Data" / "pose_detectors" / detector_config(name)["model"]
    weights.parent.mkdir(parents=True, exist_ok=True)
    # The lock is beside the globally shared asset, not inside one queue's output.
    with weights.with_name(weights.name + ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return YoloDetector(device)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run_worker(directory, worker_id, detector_name):
    """Serve chunk commands until stopped, keeping one CUDA detector resident."""
    directory = Path(directory).resolve()
    work = directory / "workers" / _identifier(worker_id)
    work.mkdir(parents=True, exist_ok=True)
    reply = work / "reply.json"
    token = None
    job_id = None
    try:
        plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
        if Path(plan["directory"]).resolve() != directory:
            raise ValueError("Worker directory differs from the immutable extraction plan")
        if sha256_file(plan["annotations"]) != plan["annotation_sha256"]:
            raise ValueError("Annotations changed since extraction was planned")
        if detector_config(detector_name) != plan["detectors"][detector_name]:
            raise ValueError("Detector configuration changed since extraction was planned")
        owner = None
        if sys.platform == "linux":
            from src.Train.queue_manager import process_token
            owner = json.loads((directory / "manager_identity.json").read_text(encoding="utf-8"))
            if (not owner.get("manager_pid") or not owner.get("manager_token")
                    or process_token(owner["manager_pid"]) != owner["manager_token"]):
                raise ValueError("Extraction manager is no longer alive")
        import torch
        torch.set_num_threads(int(os.environ.get("JIUJITSU_EXTRACTION_THREADS", "1")))
        print(f"Loading {detector_name} on cuda:0; visible GPU {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
        before = time.perf_counter()
        detector = _build_detector(detector_name, "cuda:0")
        detector.extraction_packages = _versions()
        write_json(reply, {"token": None, "event": "ready", "detector": detector_name,
                           "weights": detector.provenance, "packages": detector.extraction_packages,
                           "model_load_seconds": time.perf_counter() - before, **_memory_stats()})
        jobs = {job["id"]: job for job in plan["jobs"]}
        while True:
            if owner and process_token(owner["manager_pid"]) != owner["manager_token"]:
                print("Manager exited; worker releases its model after the current chunk.", flush=True)
                return 0
            command_path = work / "command.json"
            if not command_path.exists():
                time.sleep(.2)
                continue
            command = json.loads(command_path.read_text(encoding="utf-8"))
            next_token = command.get("token")
            if not isinstance(next_token, str) or not next_token:
                raise ValueError("Worker command needs a nonempty token")
            if next_token == token:
                time.sleep(.2)
                continue
            token = next_token
            if command.get("action") == "stop":
                print("Worker stopped after its current chunk.", flush=True)
                return 0
            if command.get("action") != "run":
                raise ValueError(f"Unknown worker command: {command.get('action')}")
            job_id = command["job_id"]
            job = jobs[job_id]
            print(f"Starting {job_id}: images {job['start']}..{job['stop'] - 1}", flush=True)
            summary = extract_chunk(plan, job, detector, progress=True)
            print(f"Completed {job_id}: {summary['images']} images ({summary['reused_images']} reused)", flush=True)
            write_json(reply, {"token": token, "event": "done", "job_id": job_id,
                               "summary": summary, **_memory_stats()})
    except Exception as error:
        event = "oom" if _is_oom(error) else "error"
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        write_json(reply, {"token": token, "event": event, "job_id": job_id,
                           "error": f"{type(error).__name__}: {error}"})
        return 2 if event == "oom" else 1
