"""Compare supplied poses, YOLO and ViTPose with unchanged pose checkpoints.

The checkpoint's own split is verified before any detection or classification.
Choose validation for development, or test for a final held-out comparison.
"""

import argparse
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.Dataset.annotations import sha256_file
from src.Eval.exploration import ExplorationData, resolve_checkpoint
# Existing notebook imports use these names; implementation is now shared.
from src.Eval.pose_detection import (DEDUPE_IOU, VitPoseDetector, YoloDetector,
                                     box_iou, detect_records, load_cache)
from src.Modules.registry import POSE_MODEL_NAMES
from src.Train.run_metadata import write_json

OUTPUT = ROOT / "diagnostics" / "pose_detectors"


def stratified_sample(records, label_map, per_class, seed, limit):
    groups = defaultdict(list)
    for record in records:
        groups[label_map[record["position"]]].append(record)
    rng = random.Random(seed)
    chosen = []
    for name in sorted(groups):
        members = sorted(groups[name], key=lambda record: record["image"])
        chosen.extend(rng.sample(members, min(per_class, len(members))))
    chosen.sort(key=lambda record: record["image"])
    if limit:
        chosen = rng.sample(chosen, min(limit, len(chosen)))
    return chosen


def resolve_split_file(config, explicit=None, split_dir=None):
    """Relocated checkpoints may use a local copy with the exact saved hash."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    saved = config.get("split")
    if saved and Path(saved).expanduser().is_file():
        return Path(saved).expanduser().resolve()
    folder = Path(split_dir) if split_dir else ROOT / "Data" / "splits"
    digest = config.get("split_sha256")
    matches = sorted(p for p in folder.glob("*.json") if sha256_file(p) == digest)
    if matches:
        return matches[0].resolve()
    raise FileNotFoundError("Checkpoint split is unavailable. Supply --split-file with the "
                            "exact manifest used for training; its checksum will be verified.")


def verified_data(config, annotations=None, split_file=None, images_dir=None):
    from src.Train.engine import validate_checkpoint_data
    if config.get("model") not in POSE_MODEL_NAMES:
        raise ValueError("Pose comparison requires a pose, pose-wide or pose-attention checkpoint")
    annotation_path = Path(annotations or config.get("annotations") or ROOT / "Data/annotations.json")
    if annotations is None and not annotation_path.is_file():
        annotation_path = ROOT / "Data/annotations.json"
    split_path = resolve_split_file(config, split_file)
    image_root = images_dir or config.get("images_dir")
    if images_dir is None and image_root and not Path(image_root).is_dir():
        image_root = None
    data = ExplorationData(annotation_path, split_path, image_root)
    validate_checkpoint_data(config, config["model"], data.annotations_path,
                             data.split_path, data.manifest)
    return data


def detect_all(name, records, data, device):
    """Compatibility wrapper; creates the output directory on first use."""
    return detect_records(name, records, data.image_path, OUTPUT / "verified_cache", device)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="Trusted checkpoint or run directory; repeat for more classifiers")
    parser.add_argument("--annotations")
    parser.add_argument("--images-dir")
    parser.add_argument("--split-file", help="Default: exact manifest recorded in each checkpoint")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--detector", choices=("yolo", "vitpose", "all"), default="all")
    parser.add_argument("--per-class", type=int, default=150)
    parser.add_argument("--all-frames", action="store_true", help="Use the complete eligible partition")
    parser.add_argument("--limit", type=int, help="Quick pilot: at most this many selected frames")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout", action="store_true", help="Use frames outside the stratified sample")
    parser.add_argument("--device", default="cuda", help="cpu, cuda[:index], or mps")
    parser.add_argument("--detection-device", help="Default: same as --device")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cache-dir", default=str(OUTPUT / "verified_cache"))
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--report", help="Optional report filename (one checkpoint only)")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.per_class < 1 or args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--per-class, --batch-size and --limit must be positive")
    if args.all_frames and (args.holdout or args.limit):
        parser.error("--all-frames cannot be combined with --holdout or --limit")
    if args.report and (len(args.checkpoint) != 1 or Path(args.report).name != args.report):
        parser.error("--report must be a filename and requires exactly one checkpoint")
    from src.Eval.external import ExternalClassifier, compare_pose_sources, score_predictions
    from src.Train.checkpoints import load_checkpoint
    from src.Train.engine import evaluation_types_for_records, validate_checkpoint_data

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = ("yolo", "vitpose") if args.detector == "all" else (args.detector,)
    for number, selected_checkpoint in enumerate(args.checkpoint, 1):
        path = resolve_checkpoint(selected_checkpoint)
        checkpoint = load_checkpoint(path)
        config = checkpoint["config"]
        data = verified_data(config, args.annotations, args.split_file, args.images_dir)
        del checkpoint
        population = data.select(config["model"], split=args.split)
        if args.all_frames:
            records = population
        else:
            records = stratified_sample(population, config["label_map"], args.per_class,
                                         args.seed, args.limit)
            if args.holdout:
                chosen = {r["image"] for r in records}
                records = [r for r in population if r["image"] not in chosen]
        if not records:
            raise ValueError("No frames selected; change sampling options")
        classifier = ExternalClassifier(path, args.device)
        try:
            # best.pt may be replaced atomically by a running trainer between
            # the initial split lookup and classifier loading.
            validate_checkpoint_data(classifier.config, classifier.model_name,
                                     data.annotations_path, data.split_path, data.manifest)
            result = compare_pose_sources(
                classifier, records, data.image_path, detector_names=names,
                cache_dir=args.cache_dir, detection_device=args.detection_device or args.device,
                batch_size=args.batch_size, progress=args.progress,
                reference_kind="original_dataset_supplied_poses")
            kinds = evaluation_types_for_records(records, data.manifest)
            if kinds:
                result["by_evaluation_type"] = {}
                for kind in sorted(set(kinds)):
                    subset = [r for r, value in zip(records, kinds) if value == kind]
                    ids = {r["image"] for r in subset}
                    def score_subset(predictions):
                        return score_predictions(subset, [p for p in predictions if p["image"] in ids],
                                                 classifier.class_names)
                    result["by_evaluation_type"][kind] = {
                        "annotated": score_subset(result["baseline"]["predictions"]),
                        "detectors": {name: score_subset(result["detectors"][name]["all_frames"]["predictions"])
                                      for name in names},
                    }
            report = {"evaluation": "original_split_pose_comparison", "checkpoint": str(path),
                      "checkpoint_sha256": classifier.checkpoint_sha256, "epoch": classifier.checkpoint_epoch,
                      "model": classifier.model_name, "annotations": str(data.annotations_path),
                      "annotation_sha256": data.annotation_sha256, "split_file": str(data.split_path),
                      "split_sha256": data.split_sha256, "partition": args.split,
                      "eligible_population": len(population), "selected_frames": len(records),
                      "sampling": {"seed": args.seed, "per_class": args.per_class,
                                   "all_frames": args.all_frames, "limit": args.limit, "holdout": args.holdout},
                      "comparison": result}
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            filename = args.report or f"report-{number}-{classifier.model_name}-{args.split}-{stamp}.json"
            target = output / filename
            write_json(target, report)
            print(f"Saved {target}: {len(records)} frames; supplied-pose accuracy "
                  f"{result['baseline']['metrics']['accuracy']:.4f}", flush=True)
            for name in names:
                entry = result["detectors"][name]
                metrics = entry["all_frames"]["metrics"]
                coverage = entry["coverage"]
                print(f"{name}: accuracy {metrics['accuracy']:.4f}, macro F1 {metrics['macro_f1']:.4f}; "
                      f"0/1/2 usable poses: {coverage['0']}/{coverage['1']}/{coverage['2']}", flush=True)
        finally:
            classifier.close()


if __name__ == "__main__":
    main()
