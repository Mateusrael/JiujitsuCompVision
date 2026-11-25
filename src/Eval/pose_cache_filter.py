"""Filter completed pose caches on CPU without images or detector inference.

Source JSONL files are streamed read-only. New files retain the original pair
for audit and preserve image hashes and inference times. A complete directory
is published only after both caches and their source metadata are verified.
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import tempfile
import time

from src.Dataset.annotations import load_annotations, sha256_file
from src.Eval.pose_detection import detector_config, validate_detection
from src.Eval.pose_quality import DEFAULT_PROFILE, QUALITY_PROFILES, filter_pose_pair, quality_config


DETECTORS = ("yolo", "vitpose")


def _digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def validate_postprocessing(name, metadata):
    """Return the supported profile, or None for an unfiltered detector cache."""
    if not isinstance(metadata, dict):
        raise ValueError(f"{name}: metadata must be an object")
    if "postprocessing" not in metadata:
        return None
    details = metadata["postprocessing"]
    if (not isinstance(details, dict)
            or set(details) != {"profile", "configuration", "source_cache",
                                "source_cache_sha256", "source_metadata_sha256"}
            or details.get("profile") not in QUALITY_PROFILES
            or details.get("configuration") != quality_config(name, profile=details["profile"])):
        raise ValueError(f"{name}: unsupported postprocessing configuration")
    path = details["source_cache"]
    if (not isinstance(path, str) or not path
            or not (PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute())
            or not _digest(details["source_cache_sha256"])
            or not _digest(details["source_metadata_sha256"])):
        raise ValueError(f"{name}: invalid postprocessing source provenance")
    seconds = metadata.get("filtering_wall_seconds")
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or seconds < 0
            or metadata.get("row_seconds_meaning") != "original_detector_inference"):
        raise ValueError(f"{name}: invalid postprocessing timing metadata")
    return details["profile"]


def validate_filtered_row(row, name, profile=None):
    """Validate an output and replay filtering of its originally selected poses."""
    validate_detection(row)
    original = row.get("original_poses")
    validate_detection({**row, "poses": original})
    audit = row.get("quality_filter")
    if not isinstance(audit, dict) or audit.get("profile") not in QUALITY_PROFILES:
        raise ValueError(f"{name}: {row['image']} has an unsupported quality filter profile")
    if profile is not None and audit["profile"] != profile:
        raise ValueError(f"{name}: {row['image']} quality filter profile differs from cache metadata")
    expected = filter_pose_pair(original, detector=name, profile=audit["profile"])
    if (len(row["poses"]) not in (0, 2) or row["poses"] != expected["poses"]
            or row.get("quality_filter") != expected["quality_filter"]):
        raise ValueError(f"{name}: {row['image']} has inconsistent filtered poses or audit")


def _metadata(name, path):
    metadata_path = path.with_suffix(".meta.json")
    if not path.is_file() or not metadata_path.is_file():
        raise ValueError(f"{name}: completed cache and metadata are required")
    contents = metadata_path.read_bytes()
    metadata = json.loads(contents)
    if (not isinstance(metadata, dict) or metadata.get("complete") is not True
            or metadata.get("configuration") != detector_config(name)
            or metadata.get("population") != "all_original_annotations"
            or type(metadata.get("images")) is not int or metadata["images"] < 1
            or not _digest(metadata.get("annotation_sha256"))
            or not _digest(metadata.get("cache_sha256"))):
        raise ValueError(f"{name}: incomplete or incompatible full-dataset cache metadata")
    coverage = metadata.get("coverage")
    if (not isinstance(coverage, dict) or set(coverage) != {"0", "1", "2"}
            or any(type(value) is not int or value < 0 for value in coverage.values())
            or sum(coverage.values()) != metadata["images"]):
        raise ValueError(f"{name}: invalid cache coverage metadata")
    validate_postprocessing(name, metadata)
    return metadata, hashlib.sha256(contents).hexdigest()


def _verified_rows(name, path, metadata, metadata_sha256, *, expected_ids=None):
    """Yield rows, then verify completeness; callers must exhaust the iterator."""
    digest, seen = hashlib.sha256(), set()
    coverage = {"0": 0, "1": 0, "2": 0}
    profile = validate_postprocessing(name, metadata)
    with path.open("rb") as handle:
        for line in handle:
            if not line.endswith(b"\n"):
                raise ValueError(f"{name}: incomplete final cache row")
            digest.update(line)
            row = json.loads(line)
            validate_detection(row)
            if profile is not None:
                validate_filtered_row(row, name, profile=profile)
            elif any(key in row for key in ("quality_filter", "original_poses", "candidates", "selection")):
                raise ValueError(f"{name}: raw cache contains unexpected selection/filter metadata")
            image = row["image"]
            if image in seen or (expected_ids is not None and image not in expected_ids):
                raise ValueError(f"{name}: unexpected or duplicate cache image {image}")
            seen.add(image)
            coverage[str(sum(any(joint[2] > 0 for joint in pose) for pose in row["poses"]))] += 1
            yield row
    if len(seen) != metadata["images"] or (expected_ids is not None and seen != expected_ids):
        raise ValueError(f"{name}: incomplete full-dataset cache")
    if coverage != metadata["coverage"]:
        raise ValueError(f"{name}: cache coverage differs from completed metadata")
    if digest.hexdigest() != metadata["cache_sha256"]:
        raise ValueError(f"{name}: cache SHA-256 differs from completed metadata")
    if sha256_file(path.with_suffix(".meta.json")) != metadata_sha256:
        raise ValueError(f"{name}: source metadata changed while reading")


def read_verified_cache_subset(name, records, cache_dir):
    """Read only requested rows into RAM after verifying the entire saved cache.

    No image access or detector construction occurs. The caller can additionally
    compare image content hashes for its chosen frames. Source files stay intact,
    including an unterminated tail, which causes an error instead of truncation.
    """
    records = list(records)
    wanted = {record["image"] for record in records}
    if len(wanted) != len(records):
        raise ValueError("Requested cache subset contains duplicate image IDs")
    path = Path(cache_dir).resolve() / f"{name}.jsonl"
    metadata, metadata_sha256 = _metadata(name, path)
    selected = {}
    for row in _verified_rows(name, path, metadata, metadata_sha256):
        if row["image"] in wanted:
            selected[row["image"]] = row
    if set(selected) != wanted:
        raise ValueError(f"{name}: cache is missing {len(wanted - set(selected))} requested images")
    if sha256_file(path) != metadata["cache_sha256"]:
        raise ValueError(f"{name}: source cache changed while reading")
    return selected, metadata


def filter_caches(input_dir, output_dir, annotations, *, profile=DEFAULT_PROFILE):
    """Stream both original detector caches into a new, atomically published directory."""
    if profile not in QUALITY_PROFILES:
        raise ValueError(f"Unsupported quality filter profile: {profile}")
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    annotations = Path(annotations).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output_dir}")
    if input_dir == output_dir or input_dir in output_dir.parents or output_dir in input_dir.parents:
        raise ValueError("Input and output cache directories must be separate siblings or unrelated paths")
    annotation_sha256 = sha256_file(annotations)
    records = load_annotations(annotations)
    expected_ids = {record["image"] for record in records}
    del records
    sources = {}
    for name in DETECTORS:
        path = input_dir / f"{name}.jsonl"
        metadata, metadata_sha256 = _metadata(name, path)
        if validate_postprocessing(name, metadata) is not None:
            raise ValueError(f"{name}: input must be an original, unfiltered detector cache")
        if (metadata["annotation_sha256"] != annotation_sha256
                or metadata["images"] != len(expected_ids)):
            raise ValueError(f"{name}: source cache does not match the annotation dataset")
        sources[name] = (path, metadata, metadata_sha256)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.filter-", dir=output_dir.parent))
    summaries = {}
    try:
        for name, (source, metadata, metadata_sha256) in sources.items():
            started = time.perf_counter()
            digest, coverage, reasons, pose_reasons = hashlib.sha256(), {"0": 0, "1": 0, "2": 0}, Counter(), Counter()
            accepted, images = 0, 0
            target = staging / f"{name}.jsonl"
            with target.open("xb") as output:
                for row in _verified_rows(name, source, metadata, metadata_sha256, expected_ids=expected_ids):
                    result = filter_pose_pair(row["poses"], detector=name, profile=profile)
                    filtered = {**row, "original_poses": row["poses"], **result}
                    validate_detection(filtered)
                    audit = result["quality_filter"]
                    images += 1
                    accepted += int(audit["accepted"])
                    reasons[audit["reason"]] += 1
                    for assessed in audit["pose_quality"]:
                        pose_reasons.update(assessed["reasons"])
                    coverage[str(len(result["poses"]))] += 1
                    line = (json.dumps(filtered, allow_nan=False) + "\n").encode("utf-8")
                    digest.update(line)
                    output.write(line)
                output.flush()
                os.fsync(output.fileno())
            summary = {"images": images, "accepted_pairs": accepted, "rejected_frames": images-accepted,
                       "frame_reasons": dict(sorted(reasons.items())),
                       "pose_rejection_reasons": dict(sorted(pose_reasons.items()))}
            finished_metadata = {
                **metadata, "cache_path": str(output_dir / target.name),
                "cache_sha256": digest.hexdigest(), "coverage": coverage,
                "postprocessing": {"profile": profile, "configuration": quality_config(name, profile=profile),
                    "source_cache": str(source), "source_cache_sha256": metadata["cache_sha256"],
                    "source_metadata_sha256": metadata_sha256},
                "filtering_wall_seconds": time.perf_counter()-started,
                "filtered_utc": datetime.now(timezone.utc).isoformat(),
                "row_seconds_meaning": "original_detector_inference",
                "filtering_summary": summary,
            }
            validate_postprocessing(name, finished_metadata)
            (staging / f"{name}.meta.json").write_text(
                json.dumps(finished_metadata, indent=2, allow_nan=False)+"\n", encoding="utf-8")
            summaries[name] = {**summary, "coverage": coverage,
                               "filtering_wall_seconds": finished_metadata["filtering_wall_seconds"],
                               "path": str(output_dir / target.name)}
        for name, (source, metadata, metadata_sha256) in sources.items():
            if (sha256_file(source) != metadata["cache_sha256"]
                    or sha256_file(source.with_suffix(".meta.json")) != metadata_sha256):
                raise ValueError(f"{name}: source changed during filtering")
        if sha256_file(annotations) != annotation_sha256:
            raise ValueError("Annotations changed during filtering")
        if output_dir.exists():
            raise FileExistsError(f"Output appeared during filtering: {output_dir}")
        staging.rename(output_dir)
        return summaries
    finally:
        # This path was created by mkdtemp in the explicitly resolved output parent.
        if staging.exists() and staging.parent == output_dir.parent and staging != output_dir:
            shutil.rmtree(staging)
