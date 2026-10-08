"""Export all run metrics and configurations in one ZIP, using only Python's standard library."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[2]
RESUME_NAME = re.compile(r"config_resume_epoch(\d+)_(.+)\.json\Z")
SETTINGS = ("model", "lr", "weight_decay", "dropout", "attention_dropout",
            "attention_mlp_dropout", "horizontal_flip_prob", "batch_size", "seed",
            "fine_tune", "compile", "annotation_sha256", "split_sha256", "pose_filter",
            "checkpoint_selection")
EVALUATION_TYPES = ("held_out_view", "unseen_moment")
GROUP_SCORES = ("accuracy", "loss", "macro_f1", "samples")
GROUP_SUMMARY_FIELDS = tuple(f"{stage}_val_{kind}_{score}"
                            for stage in ("best", "last")
                            for kind in EVALUATION_TYPES for score in GROUP_SCORES)
SUMMARY_FIELDS = ("run", *SETTINGS, "target_epochs", "epochs_logged", "last_epoch",
                  "best_epoch", "best_val_macro_f1", "best_val_accuracy", "best_val_loss",
                  "last_train_accuracy", "last_train_loss", "last_val_accuracy",
                  "last_val_loss", "last_val_macro_f1", *GROUP_SUMMARY_FIELDS, "warnings")


def reject_nonfinite(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def config_order(payload, match):
    """Resume events are chronological, even if an older checkpoint was restored."""
    provenance = payload.get("provenance")
    timestamp = provenance.get("timestamp_utc") if isinstance(provenance, dict) else None
    try:
        event_time = datetime.fromisoformat(timestamp)
        event_time = (event_time.replace(tzinfo=timezone.utc) if event_time.tzinfo is None
                      else event_time.astimezone(timezone.utc))
    except (ValueError, TypeError):
        try:
            event_time = datetime.strptime(match[2], "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            event_time = datetime.min.replace(tzinfo=timezone.utc)
    return event_time, int(match[1]) if match else -1, match[2] if match else ""


def snapshot(path):
    """Read at most the file size at open time, even when training appends to it."""
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        content = handle.read(before.st_size)
        after = os.fstat(handle.fileno())
    changed = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
    return content, changed


def metric_rows(content, warnings):
    rows = []
    lines = content.splitlines(keepends=True)
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        # The trainer terminates each completed record with a newline. Retain
        # unfinished bytes in the archive, but exclude them from the summary.
        if index == len(lines) and not line.endswith(b"\n"):
            warnings.append(f"Unterminated metrics line {index}; excluded from summary, preserved in ZIP")
            continue
        try:
            row = json.loads(line, parse_constant=reject_nonfinite)
            epoch = row["epoch"]
            if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
                raise ValueError("invalid epoch")
            for group, keys in (("train", ("accuracy", "loss")),
                                ("val", ("accuracy", "loss", "macro_f1"))):
                for key in keys:
                    value = row[group][key]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(f"invalid {group}.{key}")
            grouped = row["val"].get("by_evaluation_type")
            if grouped is not None:
                if not isinstance(grouped, dict) or set(grouped) != set(EVALUATION_TYPES):
                    raise ValueError("invalid val.by_evaluation_type groups")
                for kind in EVALUATION_TYPES:
                    for key in GROUP_SCORES:
                        value = grouped[kind][key]
                        if (isinstance(value, bool) or not isinstance(value, (int, float))
                                or not math.isfinite(value)):
                            raise ValueError(f"invalid val.by_evaluation_type.{kind}.{key}")
                    if not isinstance(grouped[kind]["samples"], int) or grouped[kind]["samples"] < 1:
                        raise ValueError(f"invalid val.by_evaluation_type.{kind}.samples")
                if sum(grouped[kind]["samples"] for kind in EVALUATION_TYPES) != row["val"]["samples"]:
                    raise ValueError("evaluation type sample counts do not sum to validation samples")
        except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
            warnings.append(f"Invalid metrics line {index}; excluded from summary ({exc})")
            continue
        rows.append(row)
    if [row["epoch"] for row in rows] != list(range(1, len(rows) + 1)):
        warnings.append("Metrics epochs are not consecutive from 1; inspect raw log before comparing")
    return rows


def summarize(name, files, warnings):
    configs = []
    for path, content in files.items():
        match = RESUME_NAME.fullmatch(path)
        if path != "config_start.json" and not match:
            continue
        try:
            payload = json.loads(content, parse_constant=reject_nonfinite)
            config = payload["config"]
            if not isinstance(config, dict):
                raise ValueError("config must be an object")
            configs.append((config_order(payload, match), config))
        except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
            warnings.append(f"Invalid {path}; configuration omitted from summary ({exc})")
    config = max(configs, key=lambda item: item[0])[1] if configs else {}
    if "config_start.json" not in files:
        warnings.append("Missing config_start.json")
    metrics = files.get("diagnostics/metrics.jsonl")
    if metrics is None:
        warnings.append("Missing diagnostics/metrics.jsonl; no epochs summarized")
    rows = metric_rows(metrics or b"", warnings)
    summary = {"run": name, **{key: config.get(key) for key in SETTINGS},
               "target_epochs": config.get("epochs"), "epochs_logged": len(rows)}
    if rows:
        best = max(rows, key=lambda row: row["val"]["macro_f1"])
        last = rows[-1]
        summary.update(last_epoch=last["epoch"], best_epoch=best["epoch"],
                       best_val_macro_f1=best["val"]["macro_f1"],
                       best_val_accuracy=best["val"]["accuracy"], best_val_loss=best["val"]["loss"])
        for group, keys in (("train", ("accuracy", "loss")),
                            ("val", ("accuracy", "loss", "macro_f1"))):
            summary.update({f"last_{group}_{key}": last[group][key] for key in keys})
        for stage, row in (("best", best), ("last", last)):
            for kind, scores in row["val"].get("by_evaluation_type", {}).items():
                summary.update({f"{stage}_val_{kind}_{key}": scores[key] for key in GROUP_SCORES})
    summary["warnings"] = " | ".join(warnings)
    return summary


def run_files(run):
    """Allowlist result files; never walk checkpoints, datasets or local settings."""
    yield run / "config_start.json"
    for path in sorted(run.glob("config_resume_epoch*.json")):
        if RESUME_NAME.fullmatch(path.name):
            yield path
    yield run / "diagnostics" / "metrics.jsonl"
    yield run / "diagnostics" / "test_metrics.json"


def build_export(trainings_dir, output):
    trainings_dir = Path(trainings_dir).expanduser().resolve()
    output = Path(output).expanduser().absolute()
    if not trainings_dir.is_dir():
        raise FileNotFoundError(f"Training directory does not exist: {trainings_dir}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Export already exists; choose another output: {output}")
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "notes": ["Read-only, per-file snapshot; not an atomic snapshot of all training processes.",
                  "Epochs logged do not establish that a process has finished or a checkpoint was saved.",
                  "Missing configuration fields remain blank; no hyperparameters are guessed.",
                  "Best summaries select validation macro F1; existing test metrics are copied without evaluation.",
                  "Raw file bytes are preserved, including any unfinished trailing metrics line."],
        "runs": [], "files": {}, "warnings": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=output.parent, suffix=".zip.tmp")
    os.close(descriptor)
    summaries = []
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for run in sorted(trainings_dir.iterdir()):
                if run.is_symlink():
                    manifest["warnings"].append(f"Skipped symlink: {run.name}")
                    continue
                if not run.is_dir() or run.name.startswith("."):
                    continue
                files, warnings = {}, []
                for path in run_files(run):
                    relative = path.relative_to(run).as_posix()
                    if path.is_symlink() or path.parent.is_symlink():
                        warnings.append(f"Skipped symlink: {relative}")
                        continue
                    if not path.exists():
                        continue
                    if not path.is_file():
                        warnings.append(f"Not a regular file: {relative}")
                        continue
                    try:
                        content, changed = snapshot(path)
                    except OSError as exc:
                        warnings.append(f"Could not read {relative}: {exc}")
                        continue
                    if changed:
                        warnings.append(f"File changed while reading: {relative}; export again for a newer snapshot")
                    files[relative] = content
                    member = f"runs/{run.name}/{relative}"
                    archive.writestr(member, content)
                    manifest["files"][member] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                summary = summarize(run.name, files, warnings)
                summaries.append(summary)
                manifest["runs"].append({"name": run.name, "warnings": warnings,
                                         "summary": {key: value for key, value in summary.items() if key != "warnings"}})
            if not manifest["files"]:
                raise ValueError("No run configuration or metrics files found; no export created")
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(summaries)
            archive.writestr("summary.csv", stream.getvalue())
            archive.writestr("manifest.json", json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        # Publish the complete archive without replacing any existing file.
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainings-dir", type=Path, default=ROOT / "trainings")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "diagnostics" / ("metrics-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".zip"))
    args = parser.parse_args(argv)
    try:
        manifest = build_export(args.trainings_dir, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Export failed: {exc}\n")
    count = len(manifest["files"])
    print(f"Exported {len(manifest['runs'])} run directories / {count} files: {args.output.absolute()}")
    for warning in manifest["warnings"]:
        print(f"WARNING: {warning}")
    for run in manifest["runs"]:
        for warning in run["warnings"]:
            print(f"WARNING [{run['name']}]: {warning}")


if __name__ == "__main__":
    main()
