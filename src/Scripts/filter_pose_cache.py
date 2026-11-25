"""Filter the already extracted YOLO26/ViTPose pairs on CPU; no new inference."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.Eval.pose_cache_filter import filter_caches


PROJECT = Path(__file__).resolve().parents[2]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=PROJECT / "Data" / "pose_detections" / "full_dataset")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT / "Data" / "pose_detections" / "full_dataset_filtered")
    parser.add_argument("--annotations", type=Path, default=PROJECT / "Data" / "annotations.json")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summaries = filter_caches(args.input_dir, args.output_dir, args.annotations)
    print(json.dumps(summaries, indent=2, allow_nan=False))
    print(f"Filtered caches: {args.output_dir.resolve()}")
    print("Original cached pairs reused; no images read and no detector inference performed.")


if __name__ == "__main__":
    main()
