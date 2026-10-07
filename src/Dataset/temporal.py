"""Explicit single-view, class-wise temporal holdout for known videos."""

from bisect import bisect_left
import math
import re

from src.Dataset.annotations import CLASS_NAMES_10, LABEL_MAP_10, record_errors
from src.Dataset.splits import SPLITS


DEFAULT_CLASS_SOURCES = {
    "5050_guard": "03", "back": "14", "closed_guard": "00", "half_guard": "03",
    "mount": "06", "open_guard": "00", "side_control": "06", "standing": "11",
    "takedown": "11", "turtle": "09",
}


def _boundary(left, right, assignments):
    return {"left_frame": left["frame"], "right_frame": right["frame"],
            "midpoint_frame": (left["frame"] + right["frame"]) / 2,
            "from_split": assignments[left["image"]],
            "to_split": assignments[right["image"]]}


def build_temporal_manifest(records, annotation_sha256, *, class_sources=None,
                            fractions=(0.7, 0.15, 0.15), gap_frames=150,
                            min_samples_per_class=20):
    """Use one source for each class; exclude other views and boundary embargoes.

    Ratios describe provisional per-class counts, before removing temporal gaps.
    A gap is the minimum separation between retained different-split frames,
    not a claim that correlated activity ends after that interval.
    """
    sources = DEFAULT_CLASS_SOURCES if class_sources is None else class_sources
    if not isinstance(sources, dict) or set(sources) != set(CLASS_NAMES_10):
        raise ValueError("class_sources must map exactly the 10 normalized class names.")
    sources = {label: sources[label] for label in CLASS_NAMES_10}
    if any(not isinstance(v, str) or not re.fullmatch(r"[0-9]{2}", v)
           for v in sources.values()):
        raise ValueError("Every class_sources value must be a two-digit video prefix.")
    if (not isinstance(fractions, (tuple, list)) or len(fractions) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or v <= 0 for v in fractions)
            or not math.isclose(sum(fractions), 1, rel_tol=0, abs_tol=1e-9)):
        raise ValueError("fractions must be three finite positive numbers summing to 1.")
    for name, value, minimum in (("gap_frames", gap_frames, 0),
                                 ("min_samples_per_class", min_samples_per_class, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if not records:
        raise ValueError("Annotations must not be empty.")

    assignments, excluded, observed = {}, {}, set()
    by_class = {label: [] for label in CLASS_NAMES_10}
    by_prefix = {prefix: [] for prefix in sorted(set(sources.values()))}
    for index, record in enumerate(records):
        errors = record_errors(record)
        if errors:
            raise ValueError(f"Annotation {index}: {'; '.join(errors)}")
        image, frame = record["image"], record["frame"]
        if int(image[2:]) != frame:
            raise ValueError(f"Image/frame mismatch: {image}, frame={frame}.")
        if image in assignments:
            raise ValueError(f"Duplicate image ID: {image}.")
        prefix, label = image[:2], LABEL_MAP_10[record["position"]]
        observed.add(prefix)
        assignments[image] = "excluded"
        if prefix != sources[label]:
            excluded[image] = "unselected_source"
        else:
            by_class[label].append(record)
            by_prefix[prefix].append(record)
    missing = sorted(set(sources.values()) - observed)
    if missing:
        raise ValueError(f"Selected source prefixes are absent: {missing}.")

    class_boundaries = {}
    for label, rows in by_class.items():
        rows.sort(key=lambda row: row["frame"])
        count = len(rows)
        cut1 = math.floor(count * fractions[0])
        cut2 = math.floor(count * (fractions[0] + fractions[1]))
        if not (0 < cut1 < cut2 < count):
            raise ValueError(f"Class {label} in source {sources[label]} has only {count} "
                             "samples; cannot create three nonempty chronological partitions.")
        for index, row in enumerate(rows):
            assignments[row["image"]] = SPLITS[0 if index < cut1 else 1 if index < cut2 else 2]
        class_boundaries[label] = [_boundary(rows[c - 1], rows[c], assignments)
                                   for c in (cut1, cut2)]

    embargo_boundaries = {}
    for prefix, rows in by_prefix.items():
        rows.sort(key=lambda row: row["frame"])
        boundaries = [_boundary(left, right, assignments)
                      for left, right in zip(rows, rows[1:])
                      if assignments[left["image"]] != assignments[right["image"]]]
        embargo_boundaries[prefix] = boundaries
        # Doubled coordinates preserve exact integer/half-frame midpoint arithmetic.
        midpoints = [b["left_frame"] + b["right_frame"] for b in boundaries]
        for row in rows:
            frame2 = 2 * row["frame"]
            index = bisect_left(midpoints, frame2 - gap_frames)
            if index < len(midpoints) and midpoints[index] <= frame2 + gap_frames:
                assignments[row["image"]] = "excluded"
                excluded[row["image"]] = "temporal_gap"

    counts = dict.fromkeys((*SPLITS, "excluded"), 0)
    class_counts = {split: dict.fromkeys(CLASS_NAMES_10, 0) for split in SPLITS}
    for label, rows in by_class.items():
        for row in rows:
            part = assignments[row["image"]]
            if part != "excluded":
                class_counts[part][label] += 1
    for part in assignments.values():
        counts[part] += 1
    shortages = {part: {label: n for label, n in table.items()
                        if n < min_samples_per_class}
                 for part, table in class_counts.items()}
    shortages = {part: table for part, table in shortages.items() if table}
    if shortages:
        raise ValueError(f"Insufficient class coverage after temporal embargo: {shortages}. "
                         f"Require >= {min_samples_per_class} samples per class per split; "
                         f"gap_frames={gap_frames}. Review the source selection and temporal "
                         "protocol or collect more data; no automatic fallback was applied.")

    separation, all_gaps = {}, []
    for prefix, rows in by_prefix.items():
        retained = [row for row in rows if assignments[row["image"]] != "excluded"]
        gaps = [right["frame"] - left["frame"] for left, right in zip(retained, retained[1:])
                if assignments[left["image"]] != assignments[right["image"]]]
        if any(gap <= gap_frames for gap in gaps):
            raise ValueError(f"Temporal separation verification failed for source {prefix}.")
        all_gaps.extend(gaps)
        separation[prefix] = {"retained_records": len(retained),
                              "cross_split_transitions": len(gaps),
                              "minimum_cross_split_gap_frames": min(gaps, default=None)}
    return {
        "schema_version": 2, "split_method": "single-view-temporal",
        "evaluation_scope": "Class-wise chronological holdout within known videos; "
                            "not unseen matches, athletes, or independently verified events.",
        "annotation_sha256": annotation_sha256,
        "class_names": list(CLASS_NAMES_10), "label_map": dict(LABEL_MAP_10),
        "class_sources": sources,
        "parameters": {"fractions": list(fractions), "gap_frames": gap_frames,
                       "min_samples_per_class": min_samples_per_class},
        "assignments": dict(sorted(assignments.items())),
        "exclusion_reasons": dict(sorted(excluded.items())),
        "counts": counts, "class_counts": class_counts,
        "coverage": {part: list(CLASS_NAMES_10) for part in SPLITS},
        "source_selection": {"selected_prefixes": sorted(by_prefix),
                             "selected_before_gap": sum(map(len, by_class.values())),
                             "class_counts_before_gap": {k: len(v) for k, v in by_class.items()}},
        "temporal_boundaries": {"class_quantiles": class_boundaries,
                                "embargo_by_prefix": embargo_boundaries},
        "separation_audit": {"required_gap_frames": gap_frames,
                             "minimum_cross_split_gap_frames": min(all_gaps, default=None),
                             "by_prefix": separation},
    }
