"""Replace annotated poses with detected ones and score a trained pose classifier.

Uses validation frames only. Detections are cached per detector, so an interrupted
run resumes where it stopped. Results go to diagnostics/pose_detectors/ (ignored).

    .venv/bin/python experiments/compare_pose_detectors.py --detector yolo --limit 20
    .venv/bin/python experiments/compare_pose_detectors.py --detector all
"""

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from src.Eval.exploration import ExplorationData
from src.Eval.metrics import classification_metrics
from src.Loader.pose import pose_features
from src.Modules.models import build_model
from src.Train.checkpoints import load_checkpoint
from src.Train.engine import evaluation_types_for_records

OUTPUT = ROOT / "diagnostics" / "pose_detectors"
DEDUPE_IOU = 0.85


def box_iou(a, b):
    """IoU of two (x1, y1, x2, y2) boxes."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    overlap = width * height
    return overlap / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap)


class YoloDetector:
    name = "yolo"

    def __init__(self, device):
        from ultralytics import YOLO
        self.device = device
        self.model = YOLO(str(OUTPUT / "yolo11x-pose.pt"))

    def __call__(self, path):
        result = self.model.predict(str(path), device=self.device, conf=0.1, imgsz=640,
                                    verbose=False)[0]
        if result.keypoints is None or not len(result.boxes):
            return []
        order = result.boxes.conf.argsort(descending=True)[:2]
        return result.keypoints.data[order].cpu().tolist()


class VitPoseDetector:
    name = "vitpose"

    def __init__(self, device):
        from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation
        self.device = device
        person = "PekingU/rtdetr_r50vd_coco_o365"
        pose = "usyd-community/vitpose-plus-huge"
        self.person_processor = AutoProcessor.from_pretrained(person)
        self.person_model = RTDetrForObjectDetection.from_pretrained(person).to(device).eval()
        self.pose_processor = AutoProcessor.from_pretrained(pose)
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(pose).to(device).eval()

    @torch.inference_mode()
    def __call__(self, path):
        with Image.open(path) as source:
            image = source.convert("RGB")
        inputs = self.person_processor(images=image, return_tensors="pt").to(self.device)
        found = self.person_processor.post_process_object_detection(
            self.person_model(**inputs), threshold=0.3,
            target_sizes=torch.tensor([(image.height, image.width)]))[0]
        people = found["labels"] == 0
        ranked = sorted(zip(found["scores"][people].tolist(), found["boxes"][people].tolist()),
                        reverse=True)
        # RT-DETR has no NMS step; keep overlapping athletes but drop near-identical boxes.
        kept = []
        for _, box in ranked:
            if all(box_iou(box, other) < DEDUPE_IOU for other in kept):
                kept.append(box)
            if len(kept) == 2:
                break
        if not kept:
            return []
        boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in kept]
        inputs = self.pose_processor(image, boxes=[boxes], return_tensors="pt").to(self.device)
        outputs = self.pose_model(**inputs, dataset_index=torch.zeros(len(boxes), dtype=torch.long,
                                                                      device=self.device))
        results = self.pose_processor.post_process_pose_estimation(outputs, boxes=[boxes])[0]
        return [[[*xy, score] for xy, score in zip(r["keypoints"].cpu().tolist(),
                                                   r["scores"].cpu().tolist())] for r in results]


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


def load_cache(cache):
    """Read cached detections, dropping a last line cut short by an interrupted run."""
    done = {}
    if not cache.exists():
        return done
    valid_bytes = 0
    with cache.open("rb") as handle:
        for line in handle:
            try:
                if not line.endswith(b"\n"):
                    raise ValueError("incomplete line")
                row = json.loads(line)
            except ValueError:
                if handle.read():
                    raise
                with cache.open("r+b") as writer:
                    writer.truncate(valid_bytes)
                break
            done[row["image"]] = row
            valid_bytes += len(line)
    return done


def detect_all(name, records, data, device):
    cache = OUTPUT / f"{name}.jsonl"
    done = load_cache(cache)
    missing = [record for record in records if record["image"] not in done]
    if not missing:
        return done
    detector = (YoloDetector if name == "yolo" else VitPoseDetector)(device)
    started = time.perf_counter()
    with cache.open("a") as handle:
        for index, record in enumerate(missing, 1):
            before = time.perf_counter()
            poses = detector(data.image_path(record))
            row = {"image": record["image"], "poses": poses,
                   "seconds": time.perf_counter() - before}
            done[record["image"]] = row
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if index % 100 == 0:
                rate = index / (time.perf_counter() - started)
                print(f"{name}: {index}/{len(missing)} ({rate:.1f} img/s)", flush=True)
    return done


def pose_error(detected, record):
    """Mean joint distance over the pair scale, using the better athlete assignment."""
    truth = [record["pose1"], record["pose2"]]
    joints = [joint for pose in truth for joint in pose if joint[2] > 0]
    xs, ys = [j[0] for j in joints], [j[1] for j in joints]
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-8)

    def distances(pose, reference):
        return [((p[0] - r[0]) ** 2 + (p[1] - r[1]) ** 2) ** 0.5 / scale
                for p, r in zip(pose, reference) if r[2] > 0]

    straight = distances(detected[0], truth[0]) + distances(detected[1], truth[1])
    crossed = distances(detected[0], truth[1]) + distances(detected[1], truth[0])
    return min(straight, crossed, key=lambda values: sum(values) / max(len(values), 1))


@torch.inference_mode()
def predict(model, features):
    return model(torch.tensor(features, dtype=torch.float32)).argmax(1).tolist()


def score(targets, predictions, class_names):
    metrics = classification_metrics(targets, predictions, class_names)
    return {"samples": metrics["samples"], "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "per_class_f1": {row["class_name"]: row["f1"] for row in metrics["per_class"]},
            "confusion_matrix": metrics["confusion_matrix"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", choices=("yolo", "vitpose", "all"), default="all")
    parser.add_argument("--checkpoint", default=str(ROOT / "trainings/pose-wide/checkpoints/best.pt"))
    parser.add_argument("--per-class", type=int, default=150)
    parser.add_argument("--limit", type=int, help="Quick pilot: score only this many frames")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout", action="store_true",
                        help="Score the validation frames outside the stratified tuning sample")
    parser.add_argument("--report", help="Report file name inside diagnostics/pose_detectors")
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = ExplorationData(ROOT / "Data/annotations.json", ROOT / "Data/splits/multiview_sections.json")
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    model = build_model(config["model_config"], pretrained=False).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    label_map, class_names = config["label_map"], config["class_names"]
    index = {name: position for position, name in enumerate(class_names)}

    population = data.select(config["model"], split="val")
    full_targets = [index[label_map[record["position"]]] for record in population]
    full_features = [pose_features(record["pose1"], record["pose2"]) for record in population]
    check = score(full_targets, predict(model, full_features), class_names)
    print(f"Annotated poses, all validation ({check['samples']}): "
          f"accuracy {check['accuracy']:.4f}, macro F1 {check['macro_f1']:.4f}", flush=True)

    records = stratified_sample(population, label_map, args.per_class, args.seed, args.limit)
    if args.holdout:
        tuning = {record["image"] for record in records}
        records = [record for record in population if record["image"] not in tuning]
    targets = [index[label_map[record["position"]]] for record in records]
    kinds = evaluation_types_for_records(records, data.manifest) if not args.limit else None
    report = {"checkpoint": args.checkpoint, "epoch": checkpoint["epoch"], "seed": args.seed,
              "per_class": args.per_class, "annotated_full_validation": check}
    baseline = predict(model, [pose_features(r["pose1"], r["pose2"]) for r in records])
    report["annotated"] = score(targets, baseline, class_names)
    print(f"Annotated poses, sample ({len(records)}): accuracy "
          f"{report['annotated']['accuracy']:.4f}, macro F1 {report['annotated']['macro_f1']:.4f}", flush=True)

    names = ("yolo", "vitpose") if args.detector == "all" else (args.detector,)
    for name in names:
        detections = detect_all(name, records, data, args.device)
        rows = [detections[record["image"]] for record in records]
        features = [pose_features(*(row["poses"] + [None, None])[:2]) for row in rows]
        predictions = predict(model, features)
        people = [len(row["poses"]) for row in rows]
        both = [position for position, count in enumerate(people) if count == 2]
        errors = []
        for position in both:
            errors.extend(pose_error(rows[position]["poses"], records[position]))
        entry = {"seconds_per_image": statistics.median(row["seconds"] for row in rows),
                 "detected_0": people.count(0), "detected_1": people.count(1),
                 "detected_2": people.count(2),
                 "median_joint_error": statistics.median(errors) if errors else None,
                 "joints_within_5pct": sum(e <= 0.05 for e in errors) / max(len(errors), 1),
                 "joints_within_10pct": sum(e <= 0.10 for e in errors) / max(len(errors), 1),
                 "all_frames": score(targets, predictions, class_names)}
        if both:
            entry["two_detected"] = score([targets[p] for p in both],
                                          [predictions[p] for p in both], class_names)
            entry["annotated_on_two_detected"] = score([targets[p] for p in both],
                                                       [baseline[p] for p in both], class_names)
        if kinds:
            entry["by_evaluation_type"] = {}
            for kind in sorted(set(kinds)):
                chosen = [p for p, value in enumerate(kinds) if value == kind]
                entry["by_evaluation_type"][kind] = {
                    "detected": score([targets[p] for p in chosen], [predictions[p] for p in chosen], class_names),
                    "annotated": score([targets[p] for p in chosen], [baseline[p] for p in chosen], class_names)}
        report[name] = entry
        print(f"{name}: all frames accuracy {entry['all_frames']['accuracy']:.4f}, macro F1 "
              f"{entry['all_frames']['macro_f1']:.4f}; detected 0/1/2 people "
              f"{entry['detected_0']}/{entry['detected_1']}/{entry['detected_2']}; "
              f"median joint error {entry['median_joint_error']}; "
              f"{entry['seconds_per_image']:.3f} s/image", flush=True)
    suffix = f"-limit{args.limit}" if args.limit else ""
    target = OUTPUT / (args.report or f"report-{args.detector}{suffix}.json")
    target.write_text(json.dumps(report, indent=1))
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
