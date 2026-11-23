"""Evaluate existing checkpoints on external labeled images, independently of training splits."""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="Trusted checkpoint or run directory (best.pt); repeat for multiple models")
    parser.add_argument("--annotations", type=Path,
                        default=ROOT / "Data/jiujitsu_images_out_of_sample_annotations.json")
    parser.add_argument("--images-dir", type=Path,
                        default=ROOT / "Data/jiujitsu_images_out_of_sample")
    parser.add_argument("--metadata", type=Path,
                        help="Dataset provenance JSON; automatically detects <dataset>_metadata.json")
    parser.add_argument("--pose-source", choices=("annotations", "yolo", "vitpose", "all"),
                        default="annotations", help="Pose models: supplied reference, or paired detector comparison")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "Data/pose_detections/external")
    parser.add_argument("--device", default="cuda", help="Classifier device: cuda, cuda:N, cpu, or mps")
    parser.add_argument("--detector-device", help="Detector device; defaults to --device")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    parser.add_argument("--output", type=Path, default=ROOT / "diagnostics" / f"external-{stamp}.json",
                        help="Aggregate JSON path; per-checkpoint reports are written beside it")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    from src.Eval.external import evaluate_external
    evaluate_external(args)


if __name__ == "__main__":
    main()
