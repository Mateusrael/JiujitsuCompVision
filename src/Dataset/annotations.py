"""Read the actual lowercase ViCoS annotation schema without losing image IDs."""

import hashlib
import json
import math
import re
from pathlib import Path


LABEL_MAP_10 = {
    "5050_guard": "5050_guard", "standing": "standing",
    "back1": "back", "back2": "back",
    "closed_guard1": "closed_guard", "closed_guard2": "closed_guard",
    "half_guard1": "half_guard", "half_guard2": "half_guard",
    "mount1": "mount", "mount2": "mount",
    "open_guard1": "open_guard", "open_guard2": "open_guard",
    "side_control1": "side_control", "side_control2": "side_control",
    "takedown1": "takedown", "takedown2": "takedown",
    "turtle1": "turtle", "turtle2": "turtle",
}
CLASS_NAMES_10 = sorted(set(LABEL_MAP_10.values()))


def both_poses_present(record):
    """A supplied pose is observed when at least one joint has positive confidence."""
    return all(any(point[2] > 0 for point in (record.get(name) or []))
               for name in ("pose1", "pose2"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_errors(record):
    """Missing poses are valid; malformed or non-finite present poses are not."""
    if not isinstance(record, dict):
        return ["record must be an object"]
    errors = []
    image = record.get("image")
    if not isinstance(image, str) or not re.fullmatch(r"[0-9]{7}", image):
        errors.append("image must be a seven-digit string, including leading zeros")
    frame = record.get("frame")
    if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
        errors.append("frame must be a non-negative integer")
    position = record.get("position")
    if not isinstance(position, str) or position not in LABEL_MAP_10:
        errors.append(f"unknown position: {position!r}")
    for key in ("pose1", "pose2"):
        pose = record.get(key)
        if pose is None:
            continue
        if not isinstance(pose, list) or len(pose) != 17:
            errors.append(f"{key} must contain 17 keypoints or be absent/null")
            continue
        for point in pose:
            if (not isinstance(point, list) or len(point) != 3
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) for v in point)):
                errors.append(f"{key} keypoints must be finite numeric [x, y, confidence]")
                break
    return errors


def read_records(path):
    with Path(path).open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError("Annotations must be a nonempty JSON array of objects.")
    return records


def load_annotations(path):
    """Load validated records. The complete annotation array is held in RAM."""
    records = read_records(path)
    seen = set()
    for index, record in enumerate(records):
        errors = record_errors(record)
        if errors:
            raise ValueError(f"Annotation {index}: {'; '.join(errors)}")
        image = record["image"]
        if image in seen:
            raise ValueError(f"Duplicate image ID {image!r} at annotation {index}")
        seen.add(image)
    return records
