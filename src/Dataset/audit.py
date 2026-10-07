"""Describe annotation quality before choosing splits or training a model."""

from collections import Counter, defaultdict
from pathlib import Path

from .annotations import LABEL_MAP_10, read_records, record_errors, sha256_file


def audit_annotations(path, images_dir=None):
    records = read_records(path)
    labels, categories, prefixes = Counter(), Counter(), Counter()
    per_prefix = defaultdict(Counter)
    missing = Counter({"pose1": 0, "pose2": 0, "both": 0})
    seen, issues = set(), []
    invalid = duplicates = image_missing = image_ambiguous = frame_mismatch = 0
    conf_min = conf_max = None
    conf_outside_unit = joints = 0
    for index, record in enumerate(records):
        errors = record_errors(record)
        if errors:
            invalid += 1
            if len(issues) < 20:
                issues.append({"index": index, "errors": errors})
            continue
        image, position = record["image"], record["position"]
        if image in seen:
            duplicates += 1
        seen.add(image)
        labels[position] += 1
        categories[LABEL_MAP_10[position]] += 1
        prefixes[image[:2]] += 1
        per_prefix[image[:2]][LABEL_MAP_10[position]] += 1
        frame_mismatch += int(int(image[2:]) != record["frame"])
        both_missing = True
        for key in ("pose1", "pose2"):
            pose = record.get(key)
            if pose is None:
                missing[key] += 1
                continue
            both_missing = False
            for _, _, confidence in pose:
                joints += 1
                conf_min = confidence if conf_min is None else min(conf_min, confidence)
                conf_max = confidence if conf_max is None else max(conf_max, confidence)
                conf_outside_unit += int(not 0 <= confidence <= 1)
        missing["both"] += int(both_missing)
        if images_dir is not None:
            root = Path(images_dir)
            matches = sum((root / f"{image}{ext}").is_file()
                          for ext in (".jpg", ".jpeg", ".png"))
            image_missing += int(matches == 0)
            image_ambiguous += int(matches > 1)
    return {
        "schema_version": 1,
        "annotation_sha256": sha256_file(path),
        "record_count": len(records), "invalid_records": invalid,
        "first_errors": issues, "duplicate_image_ids": duplicates,
        "class_counts_18": dict(sorted(labels.items())),
        "class_counts_10": dict(sorted(categories.items())),
        "video_prefix_counts": dict(sorted(prefixes.items())),
        "class_counts_by_video_prefix": {k: dict(sorted(v.items()))
                                        for k, v in sorted(per_prefix.items())},
        "missing_pose_counts": dict(missing),
        "confidence": {"keypoint_count": joints, "min": conf_min, "max": conf_max,
                       "outside_zero_one": conf_outside_unit},
        "image_frame_mismatch_count": frame_mismatch,
        "images_checked": images_dir is not None,
        "missing_image_files": image_missing if images_dir is not None else None,
        "ambiguous_image_files": image_ambiguous if images_dir is not None else None,
        "split_status": "unassigned: video prefixes are not verified recording groups",
    }
