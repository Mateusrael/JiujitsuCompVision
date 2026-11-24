"""COCO-17 pose detectors and resumable, image-verified detection caches.

Heavy dependencies are imported only when uncached images need detection.
Each cache belongs to one detector configuration. Use one writer per cache.
"""

import gc
import importlib.metadata
import json
import math
import time
from pathlib import Path

from src.Dataset.annotations import sha256_file
from src.Train.run_metadata import write_json


ROOT = Path(__file__).resolve().parents[2]
DEDUPE_IOU = 0.85


def detector_config(name):
    common = {"schema_version": 1, "keypoints": "COCO-17", "max_people": 2}
    if name == "yolo":
        return {**common, "detector": name, "model": "yolo26x-pose.pt",
                "confidence": 0.1, "image_size": 640,
                "joint_confidence": 0.5, "low_confidence_joints": "zero_xyz"}
    if name == "vitpose":
        return {**common, "detector": name,
                "person_model": "PekingU/rtdetr_r50vd_coco_o365",
                "pose_model": "usyd-community/vitpose-plus-huge",
                "person_confidence": 0.3, "dedupe_iou": DEDUPE_IOU,
                "dataset_index": 0}
    raise ValueError(f"Unknown pose detector: {name}")


def box_iou(a, b):
    width, height = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    overlap = width * height
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - overlap)
    return overlap / union if union > 0 else 0.0


class YoloDetector:
    name = "yolo"

    def __init__(self, device):
        from ultralytics import YOLO
        self.device = str(device)
        weights = ROOT / "Data" / "pose_detectors" / detector_config(self.name)["model"]
        weights.parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(weights))
        self.provenance = {"weights_sha256": sha256_file(weights)}

    def __call__(self, path):
        result = self.model.predict(str(path), device=self.device, conf=0.1,
                                    imgsz=640, verbose=False)[0]
        if result.keypoints is None or not len(result.boxes):
            return []
        order = result.boxes.conf.argsort(descending=True)[:2]
        poses = result.keypoints.data[order].cpu().tolist()
        # Older Ultralytics releases mask x/y below 0.5 but retain confidence.
        # Explicitly mask the whole triple, so pose_features never interprets
        # a discarded joint as an observed joint at pixel (0, 0).
        return [[joint if joint[2] >= 0.5 else [0.0, 0.0, 0.0]
                 for joint in pose] for pose in poses]


class VitPoseDetector:
    name = "vitpose"

    def __init__(self, device):
        import importlib.util
        if importlib.util.find_spec("scipy") is None:
            raise ImportError("ViTPose requires scipy; install install/requirements-pose-detectors.txt")
        from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation
        self.device = str(device)
        config = detector_config(self.name)
        self.person_processor = AutoProcessor.from_pretrained(config["person_model"])
        self.person_model = RTDetrForObjectDetection.from_pretrained(
            config["person_model"]).to(device).eval()
        self.pose_processor = AutoProcessor.from_pretrained(config["pose_model"])
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(
            config["pose_model"]).to(device).eval()
        self.provenance = {
            "person_revision": getattr(self.person_model.config, "_commit_hash", None),
            "pose_revision": getattr(self.pose_model.config, "_commit_hash", None),
        }

    def __call__(self, path):
        import torch
        from PIL import Image
        with Image.open(path) as source:
            image = source.convert("RGB")
        with torch.inference_mode():
            inputs = self.person_processor(images=image, return_tensors="pt").to(self.device)
            found = self.person_processor.post_process_object_detection(
                self.person_model(**inputs), threshold=0.3,
                target_sizes=torch.tensor([(image.height, image.width)]))[0]
            people = found["labels"] == 0
            ranked = sorted(zip(found["scores"][people].tolist(), found["boxes"][people].tolist()),
                            reverse=True)
            kept = []
            for _, box in ranked:
                if all(box_iou(box, other) < DEDUPE_IOU for other in kept):
                    kept.append(box)
                if len(kept) == 2:
                    break
            if not kept:
                return []
            boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in kept]
            inputs = self.pose_processor(image, boxes=[boxes], return_tensors="pt").to(self.device)
            outputs = self.pose_model(**inputs, dataset_index=torch.zeros(
                len(boxes), dtype=torch.long, device=self.device))
            results = self.pose_processor.post_process_pose_estimation(outputs, boxes=[boxes])[0]
            return [[[*xy, score] for xy, score in zip(
                r["keypoints"].cpu().tolist(), r["scores"].cpu().tolist())] for r in results]


def validate_detection(row):
    """Validate the serialized cache format without importing detector packages."""
    if not isinstance(row, dict) or not isinstance(row.get("image"), str) or not row["image"]:
        raise ValueError("Detection cache row needs a nonempty image ID")
    poses = row.get("poses")
    if not isinstance(poses, list) or len(poses) > 2:
        raise ValueError(f"{row['image']}: expected zero, one or two poses")
    for pose in poses:
        if not isinstance(pose, list) or len(pose) != 17:
            raise ValueError(f"{row['image']}: each pose must contain 17 joints")
        for joint in pose:
            if (not isinstance(joint, list) or len(joint) != 3
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) for v in joint)):
                raise ValueError(f"{row['image']}: joints must be finite [x, y, confidence] triples")
    digest = row.get("image_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{row['image']}: cache lacks a valid image_sha256; regenerate in a new cache directory")
    seconds = row.get("seconds")
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or seconds < 0):
        raise ValueError(f"{row['image']}: invalid detection time")


def load_cache(path):
    """Read complete rows; discard only an unterminated tail after interruption."""
    path = Path(path)
    rows = {}
    if not path.exists():
        return rows
    valid_bytes = 0
    incomplete = False
    with path.open("rb") as handle:
        for line in handle:
            if not line.endswith(b"\n"):
                incomplete = True
                break
            row = json.loads(line)
            validate_detection(row)
            if row["image"] in rows:
                raise ValueError(f"Duplicate detection cache image: {row['image']}")
            rows[row["image"]] = row
            valid_bytes += len(line)
    if incomplete:
        with path.open("r+b") as handle:
            handle.truncate(valid_bytes)
    return rows


def _versions():
    versions = {}
    for name in ("torch", "transformers", "ultralytics", "scipy", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def detect_records(name, records, image_path, cache_dir, device, progress=True):
    """Return detections keyed by image ID, reusing only identity-verified rows.

    A partially completed extraction resumes automatically. Existing legacy
    caches without metadata/image hashes must be regenerated in a new folder.
    Zero detections are retained, so downstream evaluation can report failures.
    """
    config = detector_config(name)
    records = list(records)
    if len({r["image"] for r in records}) != len(records):
        raise ValueError("Duplicate image IDs in detector input")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache, metadata_path = cache_dir / f"{name}.jsonl", cache_dir / f"{name}.meta.json"
    metadata = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("configuration") != config:
            raise ValueError(f"Detector configuration changed: {metadata_path}; choose a new cache directory")
    elif cache.exists() and cache.stat().st_size:
        raise ValueError(f"Unverified legacy cache: {cache}; choose a new cache directory")
    done = load_cache(cache)
    paths = {r["image"]: Path(image_path(r)) for r in records}
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    for key, digest in hashes.items():
        if key in done and done[key]["image_sha256"] != digest:
            raise ValueError(f"Image content changed for {key}; choose a new cache directory")
    missing = [r for r in records if r["image"] not in done]
    if not missing:
        return {r["image"]: done[r["image"]] for r in records}
    versions = _versions()
    if metadata:
        for package, version in metadata.get("packages", {}).items():
            if package in versions and versions[package] != version:
                raise ValueError(f"{package} changed since cached detection; choose a new cache directory")
    detector = None
    try:
        detector = (YoloDetector if name == "yolo" else VitPoseDetector)(device)
        provenance = getattr(detector, "provenance", {})
        if metadata and metadata.get("weights") != provenance:
            raise ValueError("Detector weights changed since cached detection; choose a new cache directory")
        if metadata is None:
            write_json(metadata_path, {"configuration": config, "packages": versions,
                                       "weights": provenance, "device": str(device)})
        iterable = missing
        if progress:
            from tqdm.auto import tqdm
            iterable = tqdm(missing, desc=f"{name} detections", unit="image")
        with cache.open("a", encoding="utf-8", newline="\n") as handle:
            for record in iterable:
                key = record["image"]
                before = time.perf_counter()
                poses = detector(paths[key])
                row = {"image": key, "poses": poses, "seconds": time.perf_counter() - before,
                       "image_sha256": hashes[key]}
                validate_detection(row)
                if sha256_file(paths[key]) != hashes[key]:
                    raise ValueError(f"Image changed during detection: {key}")
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
                done[key] = row
    finally:
        del detector
        gc.collect()
        # No detector package or torch import is required when all rows are cached.
        import sys
        torch = sys.modules.get("torch")
        if torch is not None:
            if str(device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif str(device) == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
    return {r["image"]: done[r["image"]] for r in records}
