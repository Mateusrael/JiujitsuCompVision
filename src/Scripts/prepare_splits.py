"""Prepare one immutable image assignment manifest for both baselines."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.Dataset.annotations import load_annotations, sha256_file
from src.Dataset.splits import build_manifest, write_manifest
from src.Dataset.temporal import build_temporal_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Create the shared temporal baseline split, or an explicit recording split."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--method", choices=("single-view-temporal", "recording-group"),
                        default="single-view-temporal")
    parser.add_argument("--groups", type=Path,
                        help="JSON with mapping evidence and sequence recording/split assignments")
    parser.add_argument("--class-sources", type=Path,
                        help="Temporal split: JSON mapping all 10 class names to source prefixes")
    parser.add_argument("--fractions", type=float, nargs=3, default=(0.7, 0.15, 0.15),
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Temporal split: per-class proportions before removing boundary gaps")
    parser.add_argument("--gap-frames", type=int, default=150,
                        help="Temporal split: total embargo width around boundaries (default: 150)")
    parser.add_argument("--min-samples-per-class", type=int, default=20,
                        help="Temporal split: minimum retained samples per class per partition")
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON manifest path; existing files are never overwritten")
    args = parser.parse_args()
    if args.method == "recording-group" and args.groups is None:
        parser.error("--method recording-group requires --groups")
    if args.method == "single-view-temporal" and args.groups is not None:
        parser.error("--groups requires --method recording-group")
    if args.method == "recording-group" and (args.class_sources is not None
            or tuple(args.fractions) != (0.7, 0.15, 0.15)
            or args.gap_frames != 150 or args.min_samples_per_class != 20):
        parser.error("Temporal source/fraction/gap options cannot configure a recording-group split")
    try:
        if args.output.exists():
            raise FileExistsError(f"Manifest already exists: {args.output}. Choose a new path.")
        digest = sha256_file(args.annotations)
        records = load_annotations(args.annotations)
        if sha256_file(args.annotations) != digest:
            raise ValueError("Annotations changed while reading; retry with an unchanged file.")
        if args.method == "recording-group":
            groups = json.loads(args.groups.read_text(encoding="utf-8"))
            manifest = build_manifest(records, groups, digest)
        else:
            sources = (json.loads(args.class_sources.read_text(encoding="utf-8"))
                       if args.class_sources else None)
            manifest = build_temporal_manifest(records, digest, class_sources=sources,
                fractions=args.fractions, gap_frames=args.gap_frames,
                min_samples_per_class=args.min_samples_per_class)
        write_manifest(manifest, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Saved {args.output}: {manifest['counts']}")
    print(f"Method: {manifest['split_method']}")
    if "evaluation_scope" in manifest:
        print(f"Evaluation scope: {manifest['evaluation_scope']}")
        reasons = manifest["exclusion_reasons"]
        print("Excluded: " + ", ".join(f"{reason}={sum(v == reason for v in reasons.values())}"
              for reason in ("unselected_source", "temporal_gap")))
        print("Minimum retained cross-partition separation: "
              f"{manifest['separation_audit']['minimum_cross_split_gap_frames']} frames")
    print(f"{'Class':18} {'Train':>8} {'Val':>8} {'Test':>8}")
    for label in manifest["class_names"]:
        print(f"{label:18}" + "".join(f" {manifest['class_counts'][part][label]:8d}"
                                      for part in ("train", "val", "test")))


if __name__ == "__main__":
    main()
