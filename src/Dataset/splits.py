"""Create shared evaluation splits from externally verified recording groups."""

import json
import os
from pathlib import Path
import tempfile

from src.Dataset.annotations import CLASS_NAMES_10, LABEL_MAP_10

SPLITS = ("train", "val", "test")


def build_manifest(records, recording_groups, annotation_sha256):
    """Require an evidence-backed map; never infer recording identity from IDs."""
    if not isinstance(recording_groups, dict):
        raise ValueError("Recording groups must be a JSON object.")
    description = recording_groups.get("description")
    if not isinstance(description, str) or len(description.strip()) < 20:
        raise ValueError("Groups description must explain the verified mapping evidence.")
    sequences = recording_groups.get("sequences")
    if not isinstance(sequences, dict):
        raise ValueError("Groups must contain a sequences object.")
    observed = {record["image"][:2] for record in records}
    missing, extra = observed - sequences.keys(), sequences.keys() - observed
    if missing or extra:
        raise ValueError(
            f"Sequence mapping mismatch: missing={sorted(missing)}, extra={sorted(extra)}. "
            "Map every observed prefix using verified recording/camera evidence."
        )
    recording_splits = {}
    for sequence, group in sequences.items():
        if not isinstance(group, dict):
            raise ValueError(f"Sequence {sequence}: expected recording and split fields.")
        recording, split = group.get("recording"), group.get("split")
        if not isinstance(recording, str) or not recording.strip():
            raise ValueError(f"Sequence {sequence}: recording must be a nonempty string.")
        if recording != recording.strip():
            raise ValueError(f"Sequence {sequence}: remove whitespace around recording ID.")
        if split not in SPLITS:
            raise ValueError(f"Sequence {sequence}: split must be train, val, or test.")
        previous = recording_splits.setdefault(recording, split)
        if previous != split:
            raise ValueError(f"Recording {recording!r} crosses splits: {previous} and {split}.")
    assignments = {}
    class_counts = {split: dict.fromkeys(CLASS_NAMES_10, 0) for split in SPLITS}
    for record in records:
        image = record["image"]
        if image in assignments:
            raise ValueError(f"Duplicate image ID: {image}.")
        split = sequences[image[:2]]["split"]
        try:
            label = LABEL_MAP_10[record["position"]]
        except KeyError as error:
            raise ValueError(f"Unknown raw position: {record['position']!r}.") from error
        assignments[image] = split
        class_counts[split][label] += 1
    missing_classes = {
        split: [label for label, count in counts.items() if count == 0]
        for split, counts in class_counts.items()
        if any(count == 0 for count in counts.values())
    }
    if missing_classes:
        raise ValueError(
            f"Every split must contain all 10 classes. Missing: {missing_classes}. "
            f"Class counts: {class_counts}. Revise verified recording assignments; "
            "if coverage is impossible, this dataset cannot support this evaluation "
            "protocol. Do not split neighboring frames to manufacture coverage."
        )
    return {
        "schema_version": 1,
        "annotation_sha256": annotation_sha256,
        "class_names": list(CLASS_NAMES_10),
        "label_map": dict(LABEL_MAP_10),
        "split_method": "recording-group",
        "recording_groups": recording_groups,
        "assignments": assignments,
        "counts": {split: sum(counts.values()) for split, counts in class_counts.items()},
        "class_counts": class_counts,
        "coverage": {split: list(CLASS_NAMES_10) for split in SPLITS},
    }


def write_manifest(manifest, output):
    """Publish complete JSON atomically, refusing to replace an existing manifest."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Manifest already exists: {output}. Choose a new path.")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Creating a hard link is atomic and fails if another writer won the race.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
