"""Resolve a checkpoint's detected inputs without reallocating temporal splits."""

from bisect import bisect_left
from collections import defaultdict

from src.Dataset.annotations import both_poses_present
from src.Loader.pose import with_detected_poses
from src.Modules.registry import POSE_MODEL_NAMES


def resolve_pose_data(records, manifest, config):
    """Return input records and eligibility for this run's evaluation pose source.

    Legacy runs replace only training inputs elsewhere. Scope ``all`` uses one
    immutable detector cache in every partition. Membership follows the shared
    image assignments, with two observed detector poses required regardless of
    supplied-pose availability. Held-out views also need a detected training-camera
    witness. Image buffers and excluded labels remain excluded.
    """
    scope = config.get("pose_detections_scope", "train")
    if scope not in ("train", "all"):
        raise ValueError(f"Unsupported pose detection scope: {scope!r}")
    if scope == "train":
        return records, manifest
    if config.get("model") not in POSE_MODEL_NAMES:
        raise ValueError("All-split detected inputs require a pose model")
    cache = config.get("pose_detections")
    digest = config.get("pose_detections_sha256")
    if not cache or not digest:
        raise ValueError("All-split detected inputs require a cache and pose_detections_sha256")
    if config.get("pose_keep_annotated", False):
        raise ValueError("All-split detected inputs cannot mix supplied poses into training")

    inputs = with_detected_poses(records, cache, expected_sha256=digest, include_empty=True)
    assignments, reasons = {}, dict(manifest.get("exclusion_reasons", {}))
    for row in inputs:
        image = row["image"]
        part = manifest["assignments"][image]
        assignments[image] = "excluded"
        if part == "excluded":
            continue
        if not both_poses_present(row):
            reasons[image] = "missing_detected_pose"
        else:
            assignments[image] = part

    if manifest.get("split_method") == "multiview-sections":
        _filter_unpaired_views(inputs, manifest, assignments, reasons)
    effective = {**manifest, "pose_assignments": assignments,
                 "pose_exclusion_reasons": reasons}
    return inputs, effective


def _filter_unpaired_views(records, manifest, assignments, reasons):
    """Removing a training detection must not turn a held-out view into an unseen moment."""
    plan = manifest["section_plan"]
    clocks = {source: (group["id"], group["alignment"][source])
              for group in plan["groups"] for source in group["sources"]}
    sources = {group["id"]: group["sources"] for group in plan["groups"]}
    indexed, witnesses = [], defaultdict(list)
    for row in records:
        image, source = row["image"], row["image"][:2]
        group, clock = clocks[source]
        time = (row["frame"] - clock["offset"]) / clock["scale"]
        label = manifest["label_map"][row["position"]]
        indexed.append((image, source, group, time, label))
        if assignments[image] == "train":
            witnesses[group, source, label].append(time)
    for times in witnesses.values():
        times.sort()
    tolerance = plan["paired_view_tolerance_frames"]
    for image, source, group, time, label in indexed:
        if (assignments[image] not in ("val", "test")
                or manifest["evaluation_types"][image] != "held_out_view"):
            continue
        paired = False
        for other in sources[group]:
            if other == source:
                continue
            times = witnesses[group, other, label]
            index = bisect_left(times, time)
            if any(abs(times[i] - time) <= tolerance + 1e-9
                   for i in (index - 1, index) if 0 <= i < len(times)):
                paired = True
                break
        if not paired:
            assignments[image] = "excluded"
            reasons[image] = "no_detected_paired_training_view"
