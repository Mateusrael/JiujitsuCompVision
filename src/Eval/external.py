"""Explicit external evaluation without relaxing the original split identity checks.

Classifier weights and preprocessing are restored from trusted project checkpoints.
External records have relative image paths, and are never a training/test manifest.
"""

import copy
import gc
import hashlib
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath

from src.Dataset.annotations import CLASS_NAMES_10, LABEL_MAP_10, both_poses_present, sha256_file
from src.Eval.metrics import classification_metrics
from src.Loader.pose import pose_features
from src.Modules.registry import MODEL_NAMES, POSE_MODEL_NAMES


def canonical_label(record):
    value = record.get("position")
    if not isinstance(value, str):
        raise ValueError("position must be a known raw or canonical class label")
    if value in LABEL_MAP_10:
        return LABEL_MAP_10[value]
    if value in CLASS_NAMES_10:
        return value
    raise ValueError(f"Unknown position label: {value!r}")


def _validate_pose(pose, context):
    if pose is None:
        return
    if not isinstance(pose, list) or len(pose) != 17:
        raise ValueError(f"{context}: each pose must contain 17 joints or be null")
    for joint in pose:
        if not isinstance(joint, list) or len(joint) != 3:
            raise ValueError(f"{context}: joints must contain x, y, confidence")
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) for v in joint):
            raise ValueError(f"{context}: joint values must be finite numbers")


def _safe_image_path(root, name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or "\x00" in name or any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError(f"Unsafe relative image path: {name!r}")
    relative = PurePosixPath(name)
    if relative.is_absolute() or relative.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        raise ValueError(f"Expected a relative JPG/PNG image path: {name!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path escapes images directory: {name!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing external image: {path}")
    return path


class ExternalData:
    """Validated external images, labels and optional COCO-17 reference poses."""

    def __init__(self, annotations_path, images_dir, metadata_path=None):
        from PIL import Image

        self.annotations_path = Path(annotations_path).expanduser().resolve()
        self.images_dir = Path(images_dir).expanduser().resolve()
        content = self.annotations_path.read_bytes()
        self.annotation_sha256 = hashlib.sha256(content).hexdigest()
        rows = json.loads(content)
        if not isinstance(rows, list) or not rows:
            raise ValueError("External annotations must be a nonempty JSON list")
        self.records, self.image_sha256, self._paths = [], {}, {}
        byte_owners, pixel_owners = {}, {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("External annotation rows must be objects")
            name = row.get("image")
            path = _safe_image_path(self.images_dir, name)
            if name in self._paths:
                raise ValueError(f"Duplicate external image ID: {name}")
            canonical_label(row)
            for field in ("pose1", "pose2"):
                _validate_pose(row.get(field), f"{name}/{field}")
            frame = row.get("frame")
            if frame is not None and (isinstance(frame, bool) or not isinstance(frame, int)
                                      or frame < 0):
                raise ValueError(f"{name}: frame must be a nonnegative integer when supplied")
            digest = sha256_file(path)
            if row.get("image_sha256") is not None and row["image_sha256"] != digest:
                raise ValueError(f"{name}: image_sha256 does not match image bytes")
            with Image.open(path) as source:
                rgb = source.convert("RGB")
                pixel_digest = hashlib.sha256(
                    str(rgb.size).encode("ascii") + rgb.tobytes()).hexdigest()
            duplicate = byte_owners.get(digest) or pixel_owners.get(pixel_digest)
            if duplicate is not None:
                raise ValueError(f"Duplicate image content: {name} and {duplicate}; "
                                 "remove the duplicate from images and annotations before evaluation")
            byte_owners[digest], pixel_owners[pixel_digest] = name, name
            self._paths[name], self.image_sha256[name] = path, digest
            self.records.append(copy.deepcopy(row))

        self.metadata_path = None
        if metadata_path is not None:
            self.metadata_path = Path(metadata_path).expanduser().resolve()
        else:
            stem = self.annotations_path.stem.removesuffix("_annotations")
            candidate = self.annotations_path.with_name(stem + "_metadata.json")
            if candidate.is_file():
                self.metadata_path = candidate
        self.metadata = {}
        self.metadata_sha256 = None
        if self.metadata_path is not None:
            content = self.metadata_path.read_bytes()
            self.metadata_sha256 = hashlib.sha256(content).hexdigest()
            self.metadata = json.loads(content)
            if not isinstance(self.metadata, dict):
                raise ValueError("External metadata must be a JSON object")
            if self.metadata.get("samples", len(rows)) != len(rows):
                raise ValueError("External metadata sample count does not match annotations")

    def image_path(self, record):
        name = record["image"] if isinstance(record, dict) else record
        if name not in self._paths:
            raise ValueError(f"Unknown external image ID: {name!r}")
        path = _safe_image_path(self.images_dir, name)
        if sha256_file(path) != self.image_sha256[name]:
            raise ValueError(f"External image changed since validation: {name}")
        return path

    @property
    def reference_kind(self):
        poses = self.metadata.get("poses", {})
        if poses.get("origin") == "detector_generated":
            return "detector_generated_reference_not_verified_keypoint_ground_truth"
        return "supplied_reference_with_unspecified_keypoint_provenance"

    def describe(self):
        eligible = [r for r in self.records if both_poses_present(r)]
        return {
            "evaluation_scope": "external_images_not_original_held_out_test",
            "annotations": str(self.annotations_path), "annotation_sha256": self.annotation_sha256,
            "images_dir": str(self.images_dir), "image_sha256": self.image_sha256,
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
            "metadata_sha256": self.metadata_sha256, "metadata": self.metadata,
            "reference_kind": self.reference_kind,
            "samples": len(self.records), "pose_eligible": len(eligible),
            "missing_reference_pose": len(self.records) - len(eligible),
            "class_counts": dict(Counter(canonical_label(r) for r in self.records)),
            "pose_eligible_class_counts": dict(Counter(canonical_label(r) for r in eligible)),
            "duplicate_policy": "reject_duplicate_bytes_or_decoded_RGB_pixels",
        }


def image_tensor(path):
    """Use the original training image transform, without augmentation."""
    from PIL import Image
    from torchvision.transforms import Normalize, PILToTensor
    from src.Loader.datasets import letterbox

    with Image.open(path) as source:
        image = letterbox(source.convert("RGB"), 224)
    return Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(
        PILToTensor()(image).float().div_(255))


class ExternalClassifier:
    """Inference-only classifier; original training data identity is reported separately."""

    def __init__(self, checkpoint_path, device="cpu"):
        from src.Modules.models import build_model
        from src.Eval.exploration import resolve_checkpoint
        from src.Train.checkpoints import load_checkpoint
        from src.Train.engine import configured_architecture, resolve_device

        self.checkpoint_path = resolve_checkpoint(checkpoint_path)
        # Hash exactly the checkpoint file handle loaded, even if a trainer replaces best.pt.
        with self.checkpoint_path.open("rb") as handle:
            checkpoint = load_checkpoint(handle)
            handle.seek(0)
            self.checkpoint_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        self.config = copy.deepcopy(checkpoint["config"])
        self.model_name = self.config.get("model")
        self.class_names = self.config.get("class_names")
        if self.model_name not in MODEL_NAMES:
            raise ValueError("Checkpoint has an unsupported model")
        if self.class_names != CLASS_NAMES_10 or self.config.get("label_map") != LABEL_MAP_10:
            raise ValueError("Checkpoint class taxonomy/order does not match the ten original classes")
        architecture = configured_architecture(self.model_name, len(self.class_names), self.config)
        if self.config.get("model_config") != architecture:
            raise ValueError("Checkpoint architecture/preprocessing configuration is unsupported")
        self.checkpoint_epoch = checkpoint["epoch"]
        self.device = resolve_device(device)
        if self.device.type not in ("cpu", "cuda", "mps"):
            raise ValueError("Evaluation device must be cpu, cuda[:index], or mps")
        self.model = build_model(architecture, pretrained=False)
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        del checkpoint
        self.model = self.model.to(self.device).eval()
        self.model.requires_grad_(False)

    def describe(self):
        return {
            "path": str(self.checkpoint_path), "sha256": self.checkpoint_sha256,
            "epoch": self.checkpoint_epoch, "model": self.model_name,
            "model_config": self.config["model_config"], "device": str(self.device),
            "training_data_identity": {key: self.config.get(key) for key in (
                "annotations", "annotation_sha256", "split", "split_sha256", "pose_filter",
                "pose_detections", "pose_detections_sha256", "pose_detections_scope",
                "pose_keep_annotated")},
            "external_data_hash_matching": "not_applicable_to_explicit_external_evaluation",
            "class_names": self.class_names,
        }

    def predict(self, records, image_path, *, batch_size=32, progress=True):
        import torch
        from tqdm.auto import tqdm

        if self.model is None:
            raise RuntimeError("Classifier has been closed")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        records = list(records)
        if len({r["image"] for r in records}) != len(records):
            raise ValueError("Prediction records must have unique image IDs")
        predictions = []
        with torch.inference_mode():
            for start in tqdm(range(0, len(records), batch_size), disable=not progress,
                              desc=f"{self.model_name} inference", unit="batch"):
                batch = records[start:start + batch_size]
                if self.model_name in POSE_MODEL_NAMES:
                    inputs = torch.tensor([pose_features(r.get("pose1"), r.get("pose2"))
                                           for r in batch], dtype=torch.float32)
                else:
                    inputs = torch.stack([image_tensor(image_path(r)) for r in batch])
                logits = self.model(inputs.to(self.device))
                if logits.shape != (len(batch), len(self.class_names)) or not torch.isfinite(logits).all():
                    raise ValueError("Classifier returned invalid or nonfinite logits")
                log_probs = logits.log_softmax(dim=1).cpu()
                probabilities = log_probs.exp().tolist()
                for row, values, log_values in zip(batch, probabilities, log_probs):
                    actual = canonical_label(row)
                    predicted_index = max(range(len(values)), key=values.__getitem__)
                    predicted = self.class_names[predicted_index]
                    predictions.append({
                        "image": row["image"], "frame": row.get("frame"), "actual": actual,
                        "predicted": predicted, "correct": predicted == actual,
                        "probabilities": values,
                        "loss": -float(log_values[self.class_names.index(actual)]),
                    })
        return predictions

    def close(self):
        import torch
        self.model = None
        gc.collect()
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()


def score_predictions(records, predictions, class_names):
    """Score exactly the supplied cohort; empty subsets have null metrics."""
    records, predictions = list(records), list(predictions)
    if [r["image"] for r in records] != [p["image"] for p in predictions]:
        raise ValueError("Prediction image IDs/order do not match the scored cohort")
    if not records:
        return None
    actual = [canonical_label(r) for r in records]
    if actual != [p["actual"] for p in predictions]:
        raise ValueError("Prediction target labels do not match the scored cohort")
    metrics = classification_metrics(
        [class_names.index(label) for label in actual],
        [class_names.index(p["predicted"]) for p in predictions], class_names)
    if all("loss" in p for p in predictions):
        metrics["loss"] = sum(p["loss"] for p in predictions) / len(predictions)
    return metrics


def _result(records, predictions, class_names):
    return {"metrics": score_predictions(records, predictions, class_names),
            "predictions": predictions}


def _subset(records, predictions, ids, class_names):
    selected = [(r, p) for r, p in zip(records, predictions) if r["image"] in ids]
    return _result([r for r, _ in selected], [p for _, p in selected], class_names)


def _localization(records, detected_records, reference_kind):
    """Pair-extent-normalized joint error; resolve ambiguous athlete ordering per frame."""
    per_image = []
    for reference, detected in zip(records, detected_records):
        observed = [joint for field in ("pose1", "pose2") for joint in reference[field]
                    if joint[2] > 0]
        scale = max(max(j[0] for j in observed) - min(j[0] for j in observed),
                    max(j[1] for j in observed) - min(j[1] for j in observed))
        if scale <= 1e-8 or not both_poses_present(detected):
            continue
        candidates = []
        for swapped in (False, True):
            fields = ("pose2", "pose1") if swapped else ("pose1", "pose2")
            errors = [math.hypot(a[0] - b[0], a[1] - b[1]) / scale
                      for ref_field, det_field in zip(("pose1", "pose2"), fields)
                      for a, b in zip(reference[ref_field], detected[det_field])
                      if a[2] > 0 and b[2] > 0]
            if errors:
                candidates.append((sum(errors) / len(errors), swapped, errors))
        if candidates:
            error, swapped, errors = min(candidates, key=lambda item: item[0])
            per_image.append({"image": reference["image"], "mean_normalized_joint_error": error,
                              "athletes_swapped": swapped, "compared_joints": len(errors),
                              "reference_visible_joints": len(observed),
                              "joints_within_0_1_pair_extent": sum(e <= 0.1 for e in errors)})
    joints = sum(row["compared_joints"] for row in per_image)
    return {
        "reference_kind": reference_kind,
        "definition": "Euclidean pixel error / longest reference pair bounding-box extent; "
                      "best of two athlete assignments; only joints observed in both poses",
        "samples": len(per_image), "compared_joints": joints,
        "mean_normalized_joint_error": (sum(r["mean_normalized_joint_error"] * r["compared_joints"]
                                             for r in per_image) / joints if joints else None),
        "fraction_within_0_1_pair_extent": (sum(r["joints_within_0_1_pair_extent"]
                                               for r in per_image) / joints if joints else None),
        "per_image": per_image,
    }


def compare_pose_sources(classifier, records, image_path, *, detector_names, cache_dir,
                         detection_device, batch_size=32, progress=True,
                         reference_kind="supplied_annotations", cache_only=False):
    """Compare fixed classifier weights on identical reference and detected-pose frames.

    Detectors use every input image, including images without reference poses. The
    reference baseline and matched comparisons require two reference poses. Detector
    failures stay in all_frames as zero/one-pose inputs. Filtered-cache operating
    scores use accepted_frames with coverage. cache_only never runs a detector.
    """
    from src.Eval.pose_detection import detect_records, detector_config

    if classifier.model_name not in POSE_MODEL_NAMES:
        raise ValueError("Pose-source comparison requires a pose classifier")
    detector_names = tuple(dict.fromkeys(detector_names))
    for name in detector_names:
        detector_config(name)
    records = list(records)
    if len({r["image"] for r in records}) != len(records):
        raise ValueError("Pose comparison records must have unique image IDs")
    reference_records = [r for r in records if both_poses_present(r)]
    reference_ids = {r["image"] for r in reference_records}
    baseline_predictions = classifier.predict(reference_records, image_path,
                                              batch_size=batch_size, progress=progress)
    result = {
        "reference_kind": reference_kind,
        "population": {"input": len(records), "detector_all_frames": len(records),
                       "reference_two_present": len(reference_records),
                       "excluded_missing_reference_pose": len(records) - len(reference_records)},
        "baseline": _result(reference_records, baseline_predictions, classifier.class_names),
        "detectors": {},
        "cache_only": cache_only,
        "failure_policy": "all_frames retains zero/one detected poses; missing joints become zero features",
        "classifier_policy": "same checkpoint weights; no retraining or athlete-order alignment before inference",
    }
    common_ids = {r["image"] for r in reference_records}
    predicted_by_name = {}
    for name in detector_names:
        metadata = None
        if cache_only:
            from src.Eval.pose_cache_filter import read_verified_cache_subset
            detections, metadata = read_verified_cache_subset(name, records, cache_dir)
            for record in records:
                if detections[record["image"]]["image_sha256"] != sha256_file(image_path(record)):
                    raise ValueError(f"{name}/{record['image']}: image content differs from cached detection")
        else:
            detections = detect_records(name, records, image_path, cache_dir,
                                        detection_device, progress=progress)
        detected_records, coverage, coverage_ids, two_ids = [], Counter(), {str(i): [] for i in range(3)}, set()
        for row in records:
            found = detections[row["image"]]
            poses = found["poses"]
            if not isinstance(poses, list) or len(poses) > 2:
                raise ValueError(f"{name}/{row['image']}: invalid detector pose count")
            for pose in poses:
                if pose is None:
                    raise ValueError(f"{name}/{row['image']}: detected pose must contain 17 joints")
                _validate_pose(pose, f"{name}/{row['image']}")
            replacement = {**row, "pose1": poses[0] if poses else None,
                           "pose2": poses[1] if len(poses) > 1 else None}
            detected_records.append(replacement)
            count = sum(any(joint[2] > 0 for joint in pose) for pose in poses)
            coverage[count] += 1
            coverage_ids[str(count)].append(row["image"])
            if both_poses_present(replacement) and row["image"] in reference_ids:
                two_ids.add(row["image"])
        predictions = classifier.predict(detected_records, image_path,
                                          batch_size=batch_size, progress=progress)
        predicted_by_name[name] = predictions
        common_ids &= two_ids
        meta_path = Path(cache_dir) / (name + ".meta.json")
        cache_path = Path(cache_dir) / (name + ".jsonl")
        if metadata is None:
            metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else None
        selected_rows = [detections[row["image"]] for row in sorted(records, key=lambda row: row["image"])]
        selected_digest = hashlib.sha256(json.dumps(selected_rows, sort_keys=True, separators=(",", ":"),
                                                     allow_nan=False).encode("utf-8")).hexdigest()
        result["detectors"][name] = {
            "configuration": detector_config(name), "cache_metadata": metadata,
            "cache_provenance": {"cache_path": str(cache_path.resolve()),
                                 "cache_sha256": sha256_file(cache_path) if cache_path.is_file() else None,
                                 "selected_rows_sha256": selected_digest,
                                 "selected_image_ids": [r["image"] for r in records]},
            "coverage": {str(i): coverage[i] for i in range(3)}, "coverage_image_ids": coverage_ids,
            "detection_seconds": sum(row.get("seconds", 0) for row in selected_rows),
            "all_frames": _result(records, predictions, classifier.class_names),
            "accepted_frames": _subset(records, predictions, set(coverage_ids["2"]), classifier.class_names),
            "same_reference_frames": _subset(records, predictions, reference_ids, classifier.class_names),
            "same_two_detected": {
                "image_ids": [r["image"] for r in reference_records if r["image"] in two_ids],
                "annotated": _subset(reference_records, baseline_predictions, two_ids, classifier.class_names),
                "detected": _subset(records, predictions, two_ids, classifier.class_names)},
            "localization": _localization(reference_records,
                                           [r for r in detected_records if r["image"] in reference_ids],
                                           reference_kind),
        }
        if metadata and "postprocessing" in metadata:
            audits = [detections[r["image"]]["quality_filter"] for r in records]
            result["detectors"][name]["quality_filter"] = {
                "configuration": metadata["postprocessing"],
                "input_images": len(records), "accepted_images": coverage[2],
                "rejected_images": len(records) - coverage[2],
                "rejection_reasons": dict(Counter(a["reason"] for a in audits if not a["accepted"])),
                "by_class": {label: {
                    "input": sum(canonical_label(r) == label for r in records),
                    "accepted": sum(canonical_label(r) == label and a["accepted"]
                                    for r, a in zip(records, audits))}
                    for label in classifier.class_names},
                "per_image": [{"image": r["image"], **a} for r, a in zip(records, audits)],
                "operating_metrics": "accepted_frames; report acceptance coverage alongside scores",
                "all_frames_policy": "forced predictions including rejected zero-pose inputs; diagnostic only",
            }
    result["common_two_detected"] = {
        "image_ids": [r["image"] for r in reference_records if r["image"] in common_ids],
        "annotated": _subset(reference_records, baseline_predictions, common_ids, classifier.class_names),
        "detectors": {name: _subset(records, predicted_by_name[name], common_ids,
                                    classifier.class_names) for name in detector_names},
    }
    return result


def evaluate_external(args):
    """Evaluate checkpoints sequentially and write both per-checkpoint and aggregate JSON."""
    from src.Train.run_metadata import provenance, write_json
    import torch

    data = ExternalData(args.annotations, args.images_dir, args.metadata)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    names = (("yolo", "vitpose") if args.pose_source == "all" else
             (() if args.pose_source == "annotations" else (args.pose_source,)))
    result = {"schema_version": 1, "evaluation_kind": "external", "dataset": data.describe(),
              "requested_pose_source": args.pose_source, "runs": []}
    for index, checkpoint in enumerate(args.checkpoint, 1):
        classifier = ExternalClassifier(checkpoint, device=args.device)
        try:
            report = {"schema_version": 1, "evaluation_kind": "external", "dataset": data.describe(),
                      "checkpoint": classifier.describe(), "provenance": provenance(torch, classifier.device)}
            if classifier.model_name in POSE_MODEL_NAMES:
                report["pose_comparison"] = compare_pose_sources(
                    classifier, data.records, data.image_path, detector_names=names,
                    cache_dir=args.cache_dir, detection_device=args.detector_device or args.device,
                    batch_size=args.batch_size, progress=args.progress, reference_kind=data.reference_kind)
                if args.pose_source in ("yolo", "vitpose"):
                    source_result = report["pose_comparison"]["detectors"][args.pose_source]["all_frames"]
                    report["metrics_source"] = args.pose_source + "_all_frames"
                else:
                    source_result = report["pose_comparison"]["baseline"]
                    report["metrics_source"] = "supplied_reference_two_present"
                report.update(source_result)
            else:
                predictions = classifier.predict(data.records, data.image_path,
                                                 batch_size=args.batch_size, progress=args.progress)
                report.update(_result(data.records, predictions, classifier.class_names))
                report["metrics_source"] = "RGB_images_all_frames"
                report["population"] = {"input": len(data.records), "eligible": len(data.records),
                                        "pose_filter": "not_applicable_to_image_models"}
            target = output.with_name(f"{output.stem}-{index:02d}-{classifier.model_name}-"
                                      f"{classifier.checkpoint_sha256[:8]}.json")
            write_json(target, report)
            result["runs"].append({**report, "report_file": str(target)})
            metrics = report["metrics"]
            summary = (f"samples={metrics['samples']} accuracy={metrics['accuracy']:.4f} "
                       f"macro_f1={metrics['macro_f1']:.4f}" if metrics else "no eligible reference samples")
            print(f"{classifier.model_name} [{report['metrics_source']}]: {summary}; saved {target}")
            for name, detected in report.get("pose_comparison", {}).get("detectors", {}).items():
                scored = detected["all_frames"]["metrics"]
                if scored:
                    print(f"  {name}: samples={scored['samples']} accuracy={scored['accuracy']:.4f} "
                          f"macro_f1={scored['macro_f1']:.4f}; "
                          f"observed poses 0/1/2={detected['coverage']['0']}/"
                          f"{detected['coverage']['1']}/{detected['coverage']['2']}")
        finally:
            classifier.close()
    write_json(output, result)
    print(f"External evaluation saved: {output}")
    return result
