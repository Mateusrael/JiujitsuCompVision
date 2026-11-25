"""Filter the already extracted YOLO26/ViTPose pairs on CPU; no new inference."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.Eval.pose_cache_filter import filter_caches
from src.Eval.pose_quality import DEFAULT_PROFILE, QUALITY_PROFILES


PROJECT = Path(__file__).resolve().parents[2]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=PROJECT / "Data" / "pose_detections" / "full_dataset")
    parser.add_argument("--profile", choices=QUALITY_PROFILES, default=DEFAULT_PROFILE)
    parser.add_argument("--output-dir", type=Path,
                        help="Default: full_dataset_filtered_v2 for v2; full_dataset_filtered for v1")
    parser.add_argument("--annotations", type=Path, default=PROJECT / "Data" / "annotations.json")
    return parser


def output_directory(args):
    if args.output_dir is not None:
        return args.output_dir
    name = "full_dataset_filtered_v2" if args.profile == "cached-pose-quality-v2" else "full_dataset_filtered"
    return PROJECT / "Data" / "pose_detections" / name


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = output_directory(args)
    summaries = filter_caches(args.input_dir, output, args.annotations, profile=args.profile)
    print(json.dumps(summaries, indent=2, allow_nan=False))
    print(f"Filtered caches ({args.profile}): {output.resolve()}")
    print("Original cached pairs reused; no images read and no detector inference performed.")


if __name__ == "__main__":
    main()
