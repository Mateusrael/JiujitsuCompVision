"""Validate the shared manifest before either classifier consumes it."""

import json
from pathlib import Path

from src.Dataset.annotations import sha256_file
from src.Dataset.splits import build_manifest


def read_manifest(path, annotations_path, records):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Split manifest must be a JSON object")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported split manifest schema_version")
    digest = sha256_file(annotations_path)
    if manifest.get("annotation_sha256") != digest:
        raise ValueError("Split manifest does not match this annotation file (SHA-256)")
    if manifest.get("split_method") != "recording-group":
        raise ValueError("A recording-group split manifest is required")
    # Rebuild from the saved evidence, including the cross-camera recording
    # constraint. A string claiming "recording-group" is insufficient by itself.
    expected = build_manifest(records, manifest.get("recording_groups"), digest)
    for key in ("assignments", "class_names", "label_map", "counts", "class_counts", "coverage"):
        if manifest.get(key) != expected[key]:
            raise ValueError(f"Split {key} does not match its recording groups and annotations")
    return manifest


def split_records(records, manifest, split):
    return [record for record in records
            if manifest["assignments"][record["image"]] == split]
