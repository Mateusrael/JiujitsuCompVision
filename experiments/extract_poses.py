"""Cache detected poses for a verified manifest's two-pose population.

Frames are shuffled deterministically; rerunning resumes completed detections.
Supply --split-file explicitly to select the intended training split.
"""

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.Eval.exploration import ExplorationData
from src.Eval.pose_detection import detect_records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=str(ROOT / "Data/annotations.json"))
    parser.add_argument("--images-dir", default=str(ROOT / "Data/images"))
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--detector", choices=("yolo", "vitpose"), default="vitpose")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda", help="cpu, cuda[:index], or mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=str(ROOT / "diagnostics/pose_detectors/verified_cache"))
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    data = ExplorationData(args.annotations, args.split_file, args.images_dir)
    records = sorted(data.select("pose-wide", split=args.split), key=lambda record: record["image"])
    random.Random(args.seed).shuffle(records)
    if args.limit:
        records = records[:args.limit]
    detections = detect_records(args.detector, records, data.image_path, args.cache_dir,
                               args.device, progress=args.progress)
    print(f"Done: {len(detections)}/{len(records)} frames; cache: "
          f"{Path(args.cache_dir) / (args.detector + '.jsonl')}")


if __name__ == "__main__":
    main()
