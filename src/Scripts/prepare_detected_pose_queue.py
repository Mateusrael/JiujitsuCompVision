"""Verify completed detector caches and prepare paired pose-classifier runs."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.Dataset.annotations import both_poses_present, load_annotations, sha256_file
from src.Eval.pose_detection import detector_config, validate_detection
from src.Eval.pose_cache_filter import validate_filtered_row, validate_postprocessing
from src.Loader.splits import filter_pose_records, read_manifest, split_records
from src.Train.queue_plan import prepare_plan
from src.Train.run_metadata import write_json


PROJECT = Path(__file__).resolve().parents[2]
DETECTORS = ("yolo", "vitpose")
DEFAULT_SEEDS = (46,)
TARGET_PROPORTIONS = {"train": .8, "val": .1, "test": .1}
RECIPES = (
    ("pose", {"model": "pose", "epochs": 300, "weight_decay": 1e-4, "dropout": .3}),
    ("wide", {"model": "pose-wide", "epochs": 150, "weight_decay": 1e-4, "dropout": .3}),
    ("attn32", {"model": "pose-attention", "epochs": 100, "weight_decay": .05,
                "attention_dim": 32, "attention_mlp_dim": 128}),
    ("attn64", {"model": "pose-attention", "epochs": 100, "weight_decay": .05,
                "attention_dim": 64, "attention_mlp_dim": 256}),
)


def verify_cache(directory, name, records, annotation_sha256):
    """Check final metadata and every serialized row without loading a detector."""
    path = Path(directory) / f"{name}.jsonl"
    metadata_path = path.with_suffix(".meta.json")
    if not path.is_file() or not metadata_path.is_file():
        raise ValueError(f"{name}: final cache and metadata are required; finish extraction first")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("complete") is not True:
        raise ValueError(f"{name}: cache is unfinished; finish extraction first")
    configuration = metadata.get("configuration")
    if configuration != detector_config(name):
        raise ValueError(f"{name}: detector configuration does not match the original extractor")
    filter_profile = validate_postprocessing(name, metadata)
    expected = {"annotation_sha256": annotation_sha256,
                "population": "all_original_annotations", "images": len(records)}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"{name}: final cache metadata {key} does not match the dataset/extractor")
    wanted = {record["image"] for record in records}
    seen = set()
    digest = hashlib.sha256()
    coverage = {"0": 0, "1": 0, "2": 0}
    frame_reasons, pose_reasons = Counter(), Counter()
    accepted = 0
    input_poses = 0
    discarded_joints = 0
    with path.open("rb") as handle:
        for line in handle:
            if not line.endswith(b"\n"):
                raise ValueError(f"{name}: incomplete final cache row; finish extraction first")
            digest.update(line)
            row = json.loads(line)
            if filter_profile:
                validate_filtered_row(row, name, profile=filter_profile)
            else:
                validate_detection(row)
                if any(key in row for key in ("quality_filter", "original_poses", "candidates", "selection")):
                    raise ValueError(f"{name}: raw cache contains unexpected selection/filter metadata")
            image = row["image"]
            if image not in wanted or image in seen:
                raise ValueError(f"{name}: unexpected or duplicate cache image {image}")
            seen.add(image)
            count = sum(any(joint[2] > 0 for joint in pose) for pose in row["poses"])
            coverage[str(count)] += 1
            if filter_profile:
                audit = row["quality_filter"]
                accepted += int(audit["accepted"])
                input_poses += audit["input_pose_count"]
                if not audit["accepted"]:
                    frame_reasons[audit["reason"]] += 1
                for pose in audit["pose_quality"]:
                    pose_reasons.update(pose["reasons"])
                    discarded_joints += len(pose["discarded_joint_indices"])
    if seen != wanted:
        raise ValueError(f"{name}: incomplete full-dataset cache ({len(seen)}/{len(wanted)} images)")
    if digest.hexdigest() != metadata.get("cache_sha256"):
        raise ValueError(f"{name}: final cache SHA-256 does not match completed metadata")
    if coverage != metadata.get("coverage"):
        raise ValueError(f"{name}: final cache coverage does not match completed metadata")
    audit = {"profile": filter_profile or "raw"}
    if filter_profile:
        audit.update(accepted_pairs=accepted, rejected_frames=len(seen) - accepted,
                     input_poses=input_poses, discarded_joints=discarded_joints,
                     frame_rejection_reasons=dict(sorted(frame_reasons.items())),
                     pose_rejection_reasons=dict(sorted(pose_reasons.items())))
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(),
            "configuration": metadata["configuration"], "images": len(seen),
            "postprocessing": metadata.get("postprocessing"),
            "detected_people": coverage, "quality_filter": audit}


def _class_counts(records, manifest):
    counts = Counter(manifest["label_map"][row["position"]] for row in records)
    return {label: counts[label] for label in manifest["class_names"]}


def _cohort(records, manifest):
    ids = sorted(row["image"] for row in records)
    return {"samples": len(ids), "class_counts": _class_counts(records, manifest),
            "image_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}


def _partition_counts(original, eligible, manifest, effective_manifest):
    baseline = filter_pose_records(original, manifest=manifest)
    baseline_ids = {row["image"] for row in baseline}
    eligible_ids = {row["image"] for row in eligible}
    added_ids = eligible_ids - baseline_ids
    added = [row for row in original if row["image"] in added_ids]
    removed = [row for row in baseline if row["image"] not in eligible_ids]
    recovered_missing = [row for row in added if not both_poses_present(row)]
    recovered_unpaired = [row for row in added if both_poses_present(row)
                          and manifest.get("pose_exclusion_reasons", {}).get(row["image"])
                          == "no_paired_training_view"]
    classified = {row["image"] for row in recovered_missing + recovered_unpaired}
    return {"original": _cohort(original, manifest),
            "annotated_pose_baseline": _cohort(baseline, manifest),
            "eligible": _cohort(eligible, manifest),
            "retained_from_annotated_baseline": len(baseline_ids & eligible_ids),
            "removed_from_annotated_baseline": len(removed),
            "removed_from_annotated_baseline_by_reason": dict(Counter(
                effective_manifest["pose_exclusion_reasons"].get(row["image"], "unspecified")
                for row in removed)),
            "added_to_annotated_baseline": _cohort(added, manifest),
            "recovered_missing_supplied_poses": _cohort(recovered_missing, manifest),
            "recovered_unpaired_supplied_poses": _cohort(recovered_unpaired, manifest),
            "other_additions": _cohort([row for row in added if row["image"] not in classified], manifest)}


def _split_balance(partitions, class_names):
    """Describe effective split shares; quality flags never reallocate records."""
    targets = TARGET_PROPORTIONS
    def distribution(counts):
        total = sum(counts.values())
        proportions = {part: counts[part] / total if total else None for part in targets}
        return {"counts": counts, "total": total, "proportions": proportions,
                "deviation_percentage_points": {
                    part: 100 * (value - targets[part]) if value is not None else None
                    for part, value in proportions.items()}}
    overall = distribution({part: partitions[part]["eligible"]["samples"] for part in targets})
    by_class = {label: distribution({part: partitions[part]["eligible"]["class_counts"][label]
                                    for part in targets}) for label in class_names}
    flags = []
    for part, deviation in overall["deviation_percentage_points"].items():
        if deviation is not None and abs(deviation) > 5 + 1e-9:
            flags.append({"code": "overall_split_deviation", "partition": part,
                          "deviation_percentage_points": deviation,
                          "message": f"{part} share differs from the 80/10/10 target by {deviation:+.2f} pp (>5 pp)."})
    for label, values in by_class.items():
        absent = [part for part, count in values["counts"].items() if not count]
        if absent:
            flags.append({"code": "absent_class", "class": label, "partitions": absent,
                          "message": f"{label} has no eligible samples in {', '.join(absent)}."})
        for part, deviation in values["deviation_percentage_points"].items():
            if deviation is not None and abs(deviation) > 10 + 1e-9:
                flags.append({"code": "class_split_deviation", "class": label, "partition": part,
                              "deviation_percentage_points": deviation,
                              "message": f"{label}/{part} share differs from target by {deviation:+.2f} pp (>10 pp)."})
    return {"target_proportions": dict(targets), "overall": overall, "by_class": by_class,
            "quality_flags": flags}


def _portable_path(path, project_root):
    path = Path(path).resolve()
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def prepare_queue(cache_dir, annotations, split, *, project_root=PROJECT, seeds=DEFAULT_SEEDS):
    """Return the recipe and audited cohorts; no training or file writes occur."""
    from src.Loader.pose_source import resolve_pose_data

    if (not isinstance(seeds, (list, tuple)) or not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be a nonempty sequence of unique nonnegative integers")
    project_root = Path(project_root).resolve()
    annotation_sha = sha256_file(annotations)
    split_sha = sha256_file(split)
    records = load_annotations(annotations)
    manifest = read_manifest(split, annotations, records)
    report = {"schema_version": 3, "annotation_sha256": annotation_sha,
              "split_sha256": split_sha, "split_method": manifest["split_method"],
              "checkpoint_selection": "pooled_validation_macro_f1",
              "pose_detections_scope": "all", "seeds": list(seeds),
              "detectors": {}, "common_cohort": {}}
    caches = {name: verify_cache(cache_dir, name, records, annotation_sha) for name in DETECTORS}
    profiles = {cache["quality_filter"]["profile"] for cache in caches.values()}
    if len(profiles) != 1:
        raise ValueError("Detector filtering profiles differ; use both raw caches or the same filtered profile")
    selected_ids = {}
    for name, cache in caches.items():
        config = {"model": "pose", "pose_detections": cache["path"],
                  "pose_detections_sha256": cache["sha256"],
                  "pose_detections_scope": "all", "pose_keep_annotated": False}
        inputs, effective_manifest = resolve_pose_data(records, manifest, config)
        detected_two_ids = {row["image"] for row in inputs if both_poses_present(row)}
        recovered = [row for row in records if row["image"] in detected_two_ids
                     and not both_poses_present(row)]
        full_coverage = {
            "supplied_two_athletes": _cohort([row for row in records if both_poses_present(row)], manifest),
            "detected_two_athletes": _cohort([row for row in records if row["image"] in detected_two_ids], manifest),
            "recovered_missing_supplied_poses": _cohort(recovered, manifest),
            "recovered_by_assignment": {
                part: _cohort([row for row in recovered if manifest["assignments"][row["image"]] == part], manifest)
                for part in ("train", "val", "test", "excluded")}}
        partitions = {}
        selected_ids[name] = {}
        for partition in ("train", "val", "test"):
            original = split_records(records, manifest, partition)
            eligible = filter_pose_records(
                split_records(inputs, effective_manifest, partition), manifest=effective_manifest)
            if not eligible:
                raise ValueError(f"{name}: {partition} is empty after detected-pose filtering")
            selected_ids[name][partition] = {row["image"] for row in eligible}
            counts = _partition_counts(original, eligible, manifest, effective_manifest)
            if partition != "train" and "evaluation_types" in manifest:
                groups = {}
                for kind in ("held_out_view", "unseen_moment"):
                    subset = split_records(eligible, effective_manifest, partition, kind)
                    if not subset:
                        raise ValueError(f"{name}: {partition}/{kind} is empty after detected-pose filtering")
                    groups[kind] = _cohort(subset, manifest)
                counts["by_evaluation_type"] = groups
            partitions[partition] = counts
        report["detectors"][name] = {**cache, "full_dataset_coverage": full_coverage,
                                     "partitions": partitions,
                                     "split_balance": _split_balance(partitions, manifest["class_names"])}
    for partition in ("train", "val", "test"):
        common = selected_ids["yolo"][partition] & selected_ids["vitpose"][partition]
        rows = [row for row in records if row["image"] in common]
        report["common_cohort"][partition] = {
            **_cohort(rows, manifest),
            "detector_cohorts_match": selected_ids["yolo"][partition] == selected_ids["vitpose"][partition]}
    if sha256_file(annotations) != annotation_sha or sha256_file(split) != split_sha:
        raise ValueError("Annotations or split changed during preflight; rerun preparation")
    defaults = {"annotations": _portable_path(annotations, project_root),
                "split": _portable_path(split, project_root), "compile": True, "workers": 2,
                "batch_size": 128, "lr": 3e-4, "swap_athletes": True,
                "horizontal_flip_prob": .5, "pose_detections_scope": "all",
                "pose_keep_annotated": False, "pose_joint_noise": 0.0,
                "pose_joint_dropout": 0.0, "pose_athlete_dropout": 0.0}
    jobs = []
    for seed in seeds:
        for recipe, settings in RECIPES:
            for name in DETECTORS:
                train = {**settings, "seed": seed,
                         "pose_detections": _portable_path(caches[name]["path"], project_root)}
                if settings["model"] == "pose-attention":
                    train.update(attention_heads=4, attention_layers=2, attention_dropout=0.0,
                                 attention_mlp_dropout=.4, attention_pooling="mean")
                jobs.append({"id": f"{'yolo26' if name == 'yolo' else name}-{recipe}-s{seed}",
                             "min_free_mib": 4096, "train": train})
    return {"version": 1, "max_parallel": 6, "poll_seconds": 10,
            "max_gpu_utilization": 100, "defaults": defaults, "jobs": jobs}, report


def print_report(report):
    for name, detector in report["detectors"].items():
        print(f"{name}: {detector['images']} verified cache rows; "
              f"observed people 0/1/2 = {detector['detected_people']}")
        quality = detector["quality_filter"]
        print(f"  Cache filtering: {quality['profile']}")
        if quality["profile"] != "raw":
            print(f"  Cached-pose filter: {quality['accepted_pairs']} accepted pairs / "
                  f"{quality['rejected_frames']} rejected frames from "
                  f"{quality['input_poses']} existing poses; "
                  f"{quality['discarded_joints']} unreliable/missing joint slots")
            print("    frame rejection reasons: " + ", ".join(
                f"{reason}={count}" for reason, count in quality["frame_rejection_reasons"].items()))
        full = detector["full_dataset_coverage"]
        print(f"  Full dataset: {full['detected_two_athletes']['samples']} detected pairs / "
              f"{full['supplied_two_athletes']['samples']} supplied pairs; "
              f"{full['recovered_missing_supplied_poses']['samples']} recovered missing supplied pairs "
              "before split/witness exclusions")
        for partition, counts in detector["partitions"].items():
            print(f"  {partition}: {counts['eligible']['samples']} eligible / "
                  f"{counts['annotated_pose_baseline']['samples']} annotated-pose baseline / "
                  f"{counts['original']['samples']} original")
            print("    classes: " + ", ".join(f"{key}={value}" for key, value in
                                             counts["eligible"]["class_counts"].items()))
            print(f"    baseline changes: +{counts['added_to_annotated_baseline']['samples']} / "
                  f"-{counts['removed_from_annotated_baseline']}; recovered missing supplied "
                  f"{counts['recovered_missing_supplied_poses']['samples']}, "
                  f"recovered old unpaired {counts['recovered_unpaired_supplied_poses']['samples']}")
            for kind, group in counts.get("by_evaluation_type", {}).items():
                print(f"    {kind}: {group['samples']}")
        balance = detector["split_balance"]
        print("  Effective split proportions (target train/val/test = 80/10/10):")
        for label, values in [("overall", balance["overall"]), *balance["by_class"].items()]:
            print(f"    {label}: " + "; ".join(
                f"{part} {share:.2%} ({values['deviation_percentage_points'][part]:+.2f} pp)"
                if share is not None else f"{part} undefined (no samples)"
                for part, share in values["proportions"].items()))
        for flag in balance["quality_flags"]:
            print(f"  Quality flag: {flag['message']}")
    for partition, cohort in report["common_cohort"].items():
        print(f"Common {partition}: {cohort['samples']} images; "
              f"identical detector cohorts: {cohort['detector_cohorts_match']}")
    print("Coverage is not a test score. Detector-specific cohorts may differ; "
          "compare common-image evaluation separately.")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path,
                        default=PROJECT / "Data/pose_detections/full_dataset_filtered_v2")
    parser.add_argument("--annotations", type=Path, default=PROJECT / "Data/annotations.json")
    parser.add_argument("--split", type=Path, default=PROJECT / "Data/splits/multiview_sections_shifted.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "launchers/detected-poses.local.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                        help="Seeds to run; default: 46 (eight jobs). Use 46 47 48 for repeats")
    parser.add_argument("--dry-run", action="store_true", help="Run all preflight checks without saving files")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        queue, report = prepare_queue(args.cache_dir, args.annotations, args.split,
                                      project_root=PROJECT, seeds=args.seeds)
        # Validate exact argv using the existing queue parser; this never starts a process.
        with tempfile.TemporaryDirectory(prefix="detected-poses-preflight-") as temporary:
            candidate = Path(temporary) / "queue.json"
            write_json(candidate, queue)
            resolved = prepare_plan(candidate, PROJECT, "detected-poses-preflight", Path(temporary) / "runs")
        print_report(report)
        print(f"Validated {len(resolved['jobs'])} fresh jobs; up to {resolved['max_parallel']} concurrent.")
        if args.dry_run:
            print("Dry run complete; no output files or training processes created.")
            return 0
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        report_path = output.with_name(output.stem.removesuffix(".local") + ".preflight.local.json")
        write_json(report_path, report)
        write_json(output, queue)
        command = f"bash launchers/QUEUE.sh start {shlex.quote(_portable_path(output, PROJECT))}"
        print(f"Saved queue: {output}\nSaved preflight: {report_path}")
        print(f"Review commands: {command} --dry-run")
        print(f"Start explicitly when ready: {command}")
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Detected-pose queue error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
