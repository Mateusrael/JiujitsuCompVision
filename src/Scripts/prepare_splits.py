"""Prepare one immutable image assignment manifest shared by all classifiers."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.Dataset.annotations import load_annotations, sha256_file
from src.Dataset.multiview import build_multiview_manifest
from src.Dataset.splits import build_manifest, write_manifest
from src.Dataset.temporal import build_temporal_manifest


def print_population(manifest, *, prefix="", title="Image population"):
    """Show actual pooled and per-type coverage without assuming a split algorithm."""
    counts = manifest[f"{prefix}counts"]
    class_counts = manifest[f"{prefix}class_counts"]
    retained = sum(counts[part] for part in ("train", "val", "test"))
    print(f"{title}: {counts}")
    if retained:
        print("Retained shares: " + ", ".join(
            f"{part}={counts[part] / retained:.2%}" for part in ("train", "val", "test")))
    if f"{prefix}exclusion_reasons" in manifest:
        reasons = Counter(manifest[f"{prefix}exclusion_reasons"].values())
        print("Excluded: " + (", ".join(f"{reason}={count}" for reason, count in sorted(reasons.items()))
                              or "none"))
    print("Per-class shares use the retained examples of that class as the denominator:")
    print(f"{'Class':18} {'Train (share)':>17} {'Val (share)':>17} {'Test (share)':>17}")
    for label in manifest["class_names"]:
        total = sum(class_counts[part][label] for part in ("train", "val", "test"))
        print(f"{label:18}" + "".join(
            f" {class_counts[part][label]:8d} {class_counts[part][label] / total if total else 0:7.2%}"
            for part in ("train", "val", "test")))
    balance = manifest.get("balance_audit", {})
    population_balance = balance.get("populations", {}).get("pose" if prefix == "pose_" else "image")
    if population_balance:
        print("Per-class deviation from target (percentage points): "
              f"worst={population_balance['maximum_class_deviation_percentage_points']:.2f}, "
              f"RMS={population_balance['rms_class_deviation_percentage_points']:.2f}, "
              f"allowed={balance['maximum_allowed_class_deviation_percentage_points']:.2f}")
    if f"{prefix}evaluation_counts" in manifest:
        evaluation_counts = manifest[f"{prefix}evaluation_counts"]
        evaluation_class_counts = manifest[f"{prefix}evaluation_class_counts"]
        print("Evaluation subsets (shares of this population's retained images):")
        for part in ("val", "test"):
            for kind, count in evaluation_counts[part].items():
                covered = sum(value > 0 for value in evaluation_class_counts[part][kind].values())
                share = count / retained if retained else 0
                print(f"  {part}/{kind}: {count} ({share:.2%}), "
                      f"classes={covered}/{len(manifest['class_names'])}")
        print(f"{'Class':18} {'Val view':>10} {'Val moment':>11} {'Test view':>10} {'Test moment':>12}")
        for label in manifest["class_names"]:
            values = [evaluation_class_counts[part][kind][label]
                      for part in ("val", "test") for kind in ("held_out_view", "unseen_moment")]
            print(f"{label:18} {values[0]:10d} {values[1]:11d} {values[2]:10d} {values[3]:12d}")


def main():
    parser = argparse.ArgumentParser(
        description="Create the shared multiview section split, or an explicit legacy split."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--method", choices=("multiview-sections", "single-view-temporal", "recording-group"),
                        default="multiview-sections")
    parser.add_argument("--section-plan", type=Path,
                        help="Multiview split: JSON section plan (default: docs/multiview_sections.json)")
    parser.add_argument("--groups", type=Path,
                        help="JSON with mapping evidence and sequence recording/split assignments")
    parser.add_argument("--class-sources", type=Path,
                        help="Temporal split: JSON mapping all 10 class names to source prefixes")
    parser.add_argument("--fractions", type=float, nargs=3,
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Legacy temporal split: proportions before gaps (default: 0.7 0.15 0.15)")
    parser.add_argument("--gap-frames", type=int,
                        help="Temporal split: total embargo width around boundaries (default: 150)")
    parser.add_argument("--min-samples-per-class", type=int,
                        help="Legacy temporal split: minimum retained per class/partition (default: 20)")
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON manifest path; existing files are never overwritten")
    args = parser.parse_args()
    if args.method == "recording-group" and args.groups is None:
        parser.error("--method recording-group requires --groups")
    if args.method != "recording-group" and args.groups is not None:
        parser.error("--groups requires --method recording-group")
    if args.method != "multiview-sections" and args.section_plan is not None:
        parser.error("--section-plan requires --method multiview-sections")
    if args.method != "single-view-temporal" and any(value is not None for value in (
            args.class_sources, args.fractions, args.gap_frames, args.min_samples_per_class)):
        parser.error("--class-sources, --fractions, --gap-frames and --min-samples-per-class "
                     "require --method single-view-temporal; multiview settings belong in --section-plan")
    try:
        if args.output.exists():
            raise FileExistsError(f"Manifest already exists: {args.output}. Choose a new path.")
        digest = sha256_file(args.annotations)
        records = load_annotations(args.annotations)
        if sha256_file(args.annotations) != digest:
            raise ValueError("Annotations changed while reading; retry with an unchanged file.")
        if args.method == "multiview-sections":
            plan = (json.loads(args.section_plan.read_text(encoding="utf-8"))
                    if args.section_plan else None)
            manifest = build_multiview_manifest(records, digest, section_plan=plan)
        elif args.method == "recording-group":
            groups = json.loads(args.groups.read_text(encoding="utf-8"))
            manifest = build_manifest(records, groups, digest)
        else:
            sources = (json.loads(args.class_sources.read_text(encoding="utf-8"))
                       if args.class_sources else None)
            manifest = build_temporal_manifest(records, digest, class_sources=sources,
                fractions=args.fractions if args.fractions is not None else (0.7, 0.15, 0.15),
                gap_frames=args.gap_frames if args.gap_frames is not None else 150,
                min_samples_per_class=(args.min_samples_per_class
                                       if args.min_samples_per_class is not None else 20))
        write_manifest(manifest, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Saved {args.output}: {manifest['counts']}")
    print(f"Method: {manifest['split_method']}")
    if "evaluation_scope" in manifest:
        print(f"Evaluation scope: {manifest['evaluation_scope']}")
    minimum_gap = manifest.get("separation_audit", {}).get("minimum_cross_split_gap_frames")
    if minimum_gap is not None:
        print("Minimum retained cross-partition separation: "
              f"{minimum_gap} frames")
    print_population(manifest)
    if "pose_counts" in manifest:
        print_population(manifest, prefix="pose_", title="Pose population (both athletes required)")
    if "pose_buffer_audit" in manifest:
        buffers = manifest["pose_buffer_audit"]
        print("Temporal buffers: "
              f"{buffers['annotated_images_in_buffers']:,} annotated images; "
              f"{buffers['already_ineligible_missing_athlete']:,} already ineligible for pose models "
              "because an athlete is missing; "
              f"{buffers['otherwise_eligible_pose_samples']:,} with both poses also removed.")


if __name__ == "__main__":
    main()
