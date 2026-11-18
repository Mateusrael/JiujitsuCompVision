"""Cache ViTPose detections for every frame of a split's pose population.

Frames run in a fixed shuffled order, so a partial run is still a usable sample of
every class. Rerun the same command to resume. Output: diagnostics/pose_detectors/.

    .venv/bin/python experiments/extract_poses.py --split train --limit 20
    .venv/bin/python experiments/extract_poses.py --split train
"""

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.compare_pose_detectors import detect_all
from src.Eval.exploration import ExplorationData


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--limit", type=int, help="Quick pilot: process only this many frames")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = ExplorationData(ROOT / "Data/annotations.json", ROOT / "Data/splits/multiview_sections.json")
    records = sorted(data.select("pose-wide", split=args.split), key=lambda record: record["image"])
    random.Random(args.seed).shuffle(records)
    absent = [record["image"] for record in records if not Path(data.image_path(record)).is_file()]
    if absent:
        raise SystemExit(f"{len(absent)} of {len(records)} images are missing, e.g. {absent[:3]}")
    if args.limit:
        records = records[:args.limit]
    detections = detect_all("vitpose", records, data, args.device)
    print(f"Done: {sum(record['image'] in detections for record in records)}/{len(records)} frames")


if __name__ == "__main__":
    main()
