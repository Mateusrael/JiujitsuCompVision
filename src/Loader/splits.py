"""Validate the shared manifest before either classifier consumes it."""

import json
from pathlib import Path

from src.Dataset.annotations import both_poses_present, sha256_file
from src.Dataset.splits import build_manifest
from src.Dataset.temporal import build_temporal_manifest


def read_manifest(path, annotations_path, records):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Split manifest must be a JSON object")
    digest = sha256_file(annotations_path)
    if manifest.get("annotation_sha256") != digest:
        raise ValueError("Split manifest does not match this annotation file (SHA-256)")
    method = manifest.get("split_method")
    # Rebuild every derived field; a method name alone does not prove separation.
    if method == "recording-group" and manifest.get("schema_version") == 1:
        expected = build_manifest(records, manifest.get("recording_groups"), digest)
        evidence = "recording groups and annotations"
    elif method == "single-view-temporal" and manifest.get("schema_version") == 2:
        parameters = manifest.get("parameters")
        if not isinstance(parameters, dict) or set(parameters) != {
                "fractions", "gap_frames", "min_samples_per_class"}:
            raise ValueError("Temporal split requires complete, supported parameters")
        if not isinstance(manifest.get("class_sources"), dict):
            raise ValueError("Temporal split requires explicit class_sources")
        expected = build_temporal_manifest(
            records, digest, class_sources=manifest["class_sources"], **parameters)
        evidence = "temporal parameters, source selection and annotations"
    elif method == "multiview-sections" and manifest.get("schema_version") == 3:
        from src.Dataset.multiview import build_multiview_manifest
        if not isinstance(manifest.get("section_plan"), dict):
            raise ValueError("Multi-view split requires its complete section plan")
        expected = build_multiview_manifest(records, digest, section_plan=manifest["section_plan"])
        evidence = "section plan, pose eligibility and annotations"
    else:
        raise ValueError("Unsupported split manifest method/schema_version")
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Split {key} does not match its {evidence}")
    return manifest


def filter_pose_records(records, *, manifest=None):
    """Keep complete-pose inputs and, where present, the manifest's paired-view audit."""
    assignments = (manifest or {}).get("pose_assignments")
    return [record for record in records if both_poses_present(record)
            and (assignments is None or assignments[record["image"]] != "excluded")]


def split_records(records, manifest, split, evaluation_type=None):
    if split not in ("train", "val", "test"):
        raise ValueError("Requested partition must be train, val or test")
    if evaluation_type is not None and (split == "train" or evaluation_type not in
                                        ("held_out_view", "unseen_moment")):
        raise ValueError("Evaluation type requires val/test and a supported type")
    return [record for record in records
            if manifest["assignments"][record["image"]] == split
            and (evaluation_type is None or
                 manifest.get("evaluation_types", {}).get(record["image"]) == evaluation_type)]
