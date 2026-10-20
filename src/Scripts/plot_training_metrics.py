"""Reproduce validation plots from an exported metrics ZIP or a trainings directory.

No training dependencies are imported. Matplotlib is loaded only when rendering;
``--data-only`` needs only Python's standard library.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import zipfile

from src.Scripts.export_metrics import config_order, metric_rows, reject_nonfinite, snapshot


MODELS = ("image", "image-dinov3", "pose", "pose-wide", "pose-attention")
MODEL_LABELS = {"image": "ResNet-18", "image-dinov3": "DINOv3 ViT-B/16",
                "pose": "MLP", "pose-wide": "MLP large", "pose-attention": "Pose attention"}
POSE_MODELS = ("pose", "pose-wide", "pose-attention")
GROUPS = ("held_out_view", "unseen_moment")
GROUP_LABELS = {"held_out_view": "Held-out view", "unseen_moment": "Unseen moment"}
RECALL_COLUMNS = ("overall", *GROUPS)
RECALL_LABELS = {"overall": "Overall", **GROUP_LABELS}
RECALL_PALETTE = ((0.0, "#FFFFD9"), (0.25, "#C7E9B4"), (0.5, "#41B6C4"),
                  (0.75, "#225EA8"), (1.0, "#081D58"))
RESUME = re.compile(r"config_resume_epoch(\d+)_(.+)\.json\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
MAX_MEMBER_BYTES = 256 * 1024 * 1024


def read_json(content):
    return json.loads(content, parse_constant=reject_nonfinite)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MetricsSource:
    """Read allowlisted files without extracting a ZIP or following symlinks."""

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.archive = None
        self.manifest = None
        self.read_hashes = {}
        if self.path.is_dir():
            self.runs = sorted(p.name for p in self.path.iterdir()
                               if p.is_dir() and not p.is_symlink() and not p.name.startswith("."))
            self.info = {"kind": "trainings_directory", "name": self.path.name,
                         "manifest_verified": False,
                         "snapshot_note": "Per-file snapshots, not an atomic snapshot of live training."}
        else:
            self.archive = zipfile.ZipFile(self.path)
            names = self.archive.namelist()
            if len(names) != len(set(names)):
                self.close()
                raise ValueError("ZIP contains duplicate member names")
            for name in names:
                relative = PurePosixPath(name)
                if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                        or ":" in name or "\x00" in name):
                    self.close()
                    raise ValueError(f"Unsafe ZIP member name: {name!r}")
            if "manifest.json" in names:
                self.manifest = read_json(self._zip_read("manifest.json"))
                if not isinstance(self.manifest.get("files"), dict):
                    raise ValueError("Export manifest must contain a files object")
            self.runs = sorted({PurePosixPath(name).parts[1] for name in names
                                if name.startswith("runs/") and len(PurePosixPath(name).parts) >= 3})
            self.info = {"kind": "metrics_zip", "name": self.path.name,
                         "sha256": sha256_file(self.path),
                         "export_created_utc": (self.manifest or {}).get("created_utc"),
                         "manifest_verified": self.manifest is not None,
                         "verification_scope": "Every configuration and metrics file read, not checkpoints."}

    def close(self):
        if self.archive is not None:
            self.archive.close()

    def _zip_read(self, member):
        info = self.archive.getinfo(member)
        if info.file_size > MAX_MEMBER_BYTES:
            raise ValueError(f"File exceeds {MAX_MEMBER_BYTES} bytes: {member}")
        return self.archive.read(member)

    def files(self, run):
        if self.archive is not None:
            prefix = f"runs/{run}/"
            return [name[len(prefix):] for name in self.archive.namelist()
                    if name.startswith(prefix)]
        directory = self.path / run
        return [p.name for p in directory.glob("config*.json") if not p.is_symlink()] + ["diagnostics/metrics.jsonl"]

    def read(self, run, relative):
        member = f"runs/{run}/{relative}"
        if self.archive is not None:
            try:
                content = self._zip_read(member)
            except KeyError:
                return None
            if self.manifest is not None:
                expected = self.manifest["files"].get(member)
                if expected is None:
                    raise ValueError(f"Missing manifest entry: {member}")
                if expected.get("bytes") != len(content) or expected.get("sha256") != hashlib.sha256(content).hexdigest():
                    raise ValueError(f"Manifest hash/size mismatch: {member}")
        else:
            path = self.path / run / relative
            if any(p.is_symlink() for p in (path, path.parent, self.path / run)):
                raise ValueError(f"Symlink is not a supported metrics input: {path}")
            if not path.is_file():
                return None
            if path.stat().st_size > MAX_MEMBER_BYTES:
                raise ValueError(f"File exceeds {MAX_MEMBER_BYTES} bytes: {path}")
            content, _changed = snapshot(path)
        self.read_hashes[member] = hashlib.sha256(content).hexdigest()
        return content


@dataclass
class Run:
    name: str
    config_events: list
    rows: list
    warnings: list

    @property
    def config(self):
        return self.config_events[-1]["config"]

    @property
    def best(self):
        # max retains the first row on ties, matching the trainer's strict > update.
        return max(self.rows, key=lambda row: row["val"]["macro_f1"])

    def config_at(self, epoch):
        events = [event for event in self.config_events if event["after_epoch"] < epoch]
        return events[-1]["config"]


def read_run(source, name):
    content = source.read(name, "config_start.json")
    if content is None:
        return None, "Missing config_start.json"
    start = read_json(content)
    if not isinstance(start.get("config"), dict):
        raise ValueError(f"{name}: config_start.json has no config object")
    events = [{"file": "config_start.json", "after_epoch": 0, **start}]
    resumes = []
    for relative in source.files(name):
        match = RESUME.fullmatch(relative)
        if match:
            payload = read_json(source.read(name, relative))
            if not isinstance(payload.get("config"), dict):
                raise ValueError(f"{name}: {relative} has no config object")
            resumes.append((config_order(payload, match),
                            {"file": relative, "after_epoch": int(match[1]), **payload}))
    events.extend(event for _, event in sorted(resumes, key=lambda item: item[0]))
    for key in ("model", "split_sha256", "annotation_sha256", "class_names", "pose_filter"):
        if any(event["config"].get(key) != events[0]["config"].get(key) for event in events):
            raise ValueError(f"{name}: resumed configuration changed {key}; cannot combine epochs")
    raw = source.read(name, "diagnostics/metrics.jsonl")
    if raw is None:
        return None, "Missing diagnostics/metrics.jsonl"
    warnings = []
    rows = metric_rows(raw, warnings)
    epochs = [row["epoch"] for row in rows]
    if len(epochs) != len(set(epochs)):
        raise ValueError(f"{name}: duplicated epochs; refusing ambiguous resume history")
    if epochs != sorted(epochs):
        raise ValueError(f"{name}: epochs out of order; refusing ambiguous resume history")
    # An incomplete final line is safe to omit; malformed completed rows are not.
    if any(warning.startswith("Invalid metrics line") for warning in warnings):
        raise ValueError(f"{name}: " + "; ".join(warnings))
    if not rows:
        return None, "No complete epoch records"
    return Run(name, events, rows, warnings), None


def _valid_hash(value, label):
    if not isinstance(value, str) or not HASH.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def select_runs(runs, models=MODELS, split_sha256=None, annotation_sha256=None, selection=None):
    """Select within exactly one split and annotation version, never using test data."""
    selection = selection or {}
    if not isinstance(selection, dict) or any(not isinstance(v, str) for v in selection.values()):
        raise ValueError("Selection models must map model names to run-name strings")
    unknown = set(selection) - set(models)
    if unknown:
        raise ValueError(f"Selection contains unrequested models: {sorted(unknown)}")
    names = {run.name: run for run in runs}
    for model, name in selection.items():
        if name not in names:
            raise ValueError(f"Selected run is unavailable or has no complete metrics: {name}")
        if names[name].config.get("model") != model:
            raise ValueError(f"Selected run {name} is not model {model}")
    candidates = [run for run in runs if run.config.get("model") in models
                  and (run.config.get("model") not in selection
                       or run.name == selection[run.config["model"]])]
    if split_sha256 is None:
        splits = {run.config.get("split_sha256") for run in candidates}
        if len(splits) != 1:
            raise ValueError("Multiple or no split versions available; pass --split-sha256 or selection metadata")
        split_sha256 = next(iter(splits))
    _valid_hash(split_sha256, "split_sha256")
    candidates = [run for run in candidates if run.config.get("split_sha256") == split_sha256]
    if not candidates:
        raise ValueError("No eligible run for the requested split and selection")
    if annotation_sha256 is None:
        annotations = {run.config.get("annotation_sha256") for run in candidates}
        if len(annotations) != 1:
            raise ValueError("Multiple or no annotation versions in split; pass --annotation-sha256")
        annotation_sha256 = next(iter(annotations))
    _valid_hash(annotation_sha256, "annotation_sha256")
    candidates = [run for run in candidates if run.config.get("annotation_sha256") == annotation_sha256]
    selected = []
    for model in models:
        choices = sorted((run for run in candidates if run.config["model"] == model), key=lambda run: run.name)
        if not choices:
            raise ValueError(f"No eligible run for {model} at the requested split and annotation hashes")
        selected.append(max(choices, key=lambda run: run.best["val"]["macro_f1"]))
    return selected, split_sha256, annotation_sha256


def class_metrics(scores, names, context):
    """Align by class name, never by the incidental order of per_class rows."""
    classes = scores.get("per_class")
    if not isinstance(names, list) or len(names) != len(set(names)) or not names:
        raise ValueError(f"{context}: missing or duplicate class names")
    if not isinstance(classes, list):
        raise ValueError(f"{context}: no per-class validation metrics")
    mapping = {row["class_name"]: row for row in classes}
    if len(mapping) != len(classes) or set(mapping) != set(names):
        raise ValueError(f"{context}: per-class names do not match configured class names")
    if scores.get("class_names") is not None and scores["class_names"] != names:
        raise ValueError(f"{context}: score class order disagrees with configured class order")
    result = []
    for name in names:
        row = mapping[name]
        for key in ("precision", "recall", "f1"):
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{context}: invalid {name}.{key}")
        if isinstance(row["support"], bool) or not isinstance(row["support"], int) or row["support"] < 0:
            raise ValueError(f"{context}: invalid {name}.support")
        result.append({key: row[key] for key in ("class_name", "precision", "recall", "f1", "support")})
    if sum(row["support"] for row in result) != scores["samples"]:
        raise ValueError(f"{context}: supports do not sum to validation samples")
    if not math.isclose(sum(row["f1"] for row in result) / len(result), scores["macro_f1"], abs_tol=1e-8):
        raise ValueError(f"{context}: pooled macro F1 differs from mean class F1")
    return result


def epoch_data(rows):
    result = []
    for row in rows:
        groups = row["val"].get("by_evaluation_type", {})
        result.append({"epoch": row["epoch"], "train_loss": row["train"]["loss"],
                       "val_loss": row["val"]["loss"], "train_accuracy": row["train"]["accuracy"],
                       "val_accuracy": row["val"]["accuracy"], "val_macro_f1": row["val"]["macro_f1"],
                       **{f"{group}_macro_f1": groups.get(group, {}).get("macro_f1") for group in GROUPS}})
    return result


def curve_y_axis(values, *, upper=None):
    """Round padded data bounds outward; never hide a finite plotted value."""
    values = [float(value) for value in values
              if value is not None and math.isfinite(value)]
    if any(value < 0 or (upper is not None and value > upper) for value in values):
        raise ValueError("Curve values fall outside their nonnegative metric domain")
    if not values:
        return {"min": 0.0, "max": upper or 1.0, "major_unit": (upper or 1.0) / 5}
    low, high = min(values), max(values)
    padding = (high - low) * .04 if high > low else max(abs(low) * .05, .01)
    padded_low = max(0.0, low - padding)
    padded_high = high + padding if upper is None else min(upper, high + padding)
    target = (padded_high - padded_low) / 5
    scale = 10 ** math.floor(math.log10(target))
    choices = [factor * scale for factor in (1, 2, 2.5, 5, 10)]
    step = min(choices, key=lambda value: abs(math.log(value / target)))
    minimum = max(0.0, math.floor(padded_low / step) * step)
    maximum = math.ceil(padded_high / step) * step
    if upper is not None:
        maximum = min(upper, maximum)
    # Keep enough precision for tiny spans; fixed significant digits can collapse
    # two nearby bounds to the same value.
    decimals = max(0, -math.floor(math.log10(step)) + 2)
    return {"min": min(round(minimum, decimals), low),
            "max": max(round(maximum, decimals), high),
            "major_unit": float(f"{step:.12g}")}


def curve_y_axes(rows):
    """One shared Y range for all series in each chart, independent per model."""
    groups = {"loss": ("train_loss", "val_loss"),
              "metrics": ("train_accuracy", "val_accuracy", "val_macro_f1"),
              "evaluation_metrics": ("val_macro_f1", *(f"{group}_macro_f1" for group in GROUPS))}
    return {name: curve_y_axis((row.get(key) for row in rows for key in keys),
                              upper=None if name == "loss" else 1.0)
            for name, keys in groups.items()}


def pose_curve_windows(rows, initial_epochs=20):
    """Keep the initial logged records separate without dropping or renumbering epochs."""
    if isinstance(initial_epochs, bool) or not isinstance(initial_epochs, int) or initial_epochs < 0:
        raise ValueError("pose_initial_epochs must be a nonnegative integer")
    if not initial_epochs or not rows:
        return []
    result = []
    for key, values in (("initial", rows[:initial_epochs]), ("later", rows[initial_epochs:])):
        if not values:
            continue
        axes = curve_y_axes(values)
        for bounds in axes.values():
            while (bounds["max"] - bounds["min"]) / bounds["major_unit"] > 5.0000001:
                bounds["major_unit"] *= 2
        result.append({"key": key, "first_epoch": values[0]["epoch"],
                       "last_epoch": values[-1]["epoch"], "epochs": values,
                       "curve_y_axes": axes})
    return result


def recall_colors(value):
    """Use the same editable-cell colors in JSON, native tables, PNG, and SVG."""
    if value is None:
        return {"fill": "#E8EDF0", "text": "#374151"}
    for (low, left), (high, right) in zip(RECALL_PALETTE, RECALL_PALETTE[1:]):
        if value <= high:
            ratio = (value - low) / (high - low)
            channels = [round(int(left[i:i + 2], 16) * (1 - ratio)
                              + int(right[i:i + 2], 16) * ratio) for i in (1, 3, 5)]
            fill = "#" + "".join(f"{channel:02X}" for channel in channels)
            luminance = sum(channel * weight for channel, weight in zip(channels, (0.299, 0.587, 0.114)))
            return {"fill": fill, "text": "#FFFFFF" if luminance < 145 else "#172033"}
    raise ValueError("Recall color requires a value from zero to one")


def class_recall(scores, names, context):
    """Derive exact correct/support counts, checking logged recall against them."""
    rows = class_metrics(scores, names, context)
    matrix = scores.get("confusion_matrix")
    if matrix is not None:
        if (not isinstance(matrix, list) or len(matrix) != len(names)
                or any(not isinstance(row, list) or len(row) != len(names) for row in matrix)):
            raise ValueError(f"{context}: confusion matrix shape does not match class names")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for row in matrix for value in row):
            raise ValueError(f"{context}: confusion matrix counts must be nonnegative integers")
        axes = scores.get("confusion_matrix_axes")
        if axes is not None and axes != {"rows": "true", "columns": "predicted"}:
            raise ValueError(f"{context}: unsupported confusion matrix axes")
    result = []
    for index, row in enumerate(rows):
        support = row["support"]
        correct = matrix[index][index] if matrix is not None else round(row["recall"] * support)
        if matrix is not None and sum(matrix[index]) != support:
            raise ValueError(f"{context}: confusion matrix row support differs for {row['class_name']}")
        expected = correct / support if support else 0.0
        if not math.isclose(expected, row["recall"], rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f"{context}: recall disagrees with correct/support for {row['class_name']}")
        recall = expected if support else None
        result.append({"class_name": row["class_name"], "correct": correct, "support": support,
                       "recall": recall, "available": True,
                       "count_source": "confusion_matrix" if matrix is not None else "recall_times_support",
                       "label": f"{correct}/{support}\n{100 * recall:.1f}%" if support else "N/A\n0 examples",
                       **recall_colors(recall)})
    return result


def class_recall_blocks(scores, names, context):
    columns = {"overall": scores, **scores.get("by_evaluation_type", {})}
    values = {key: class_recall(columns[key], names, f"{context}/{key}")
              for key in RECALL_COLUMNS if key in columns}
    missing = {"correct": None, "support": None, "recall": None, "available": False,
               "count_source": None, "label": "N/A", **recall_colors(None)}
    return {"columns": list(RECALL_COLUMNS), "column_labels": [RECALL_LABELS[key] for key in RECALL_COLUMNS],
            "metric": "class_recall", "units": "fractions",
            "rows": [{"class_name": name,
                      "cells": [{key: value for key, value in values[column][index].items()
                                 if key != "class_name"} if column in values else dict(missing)
                                for column in RECALL_COLUMNS]}
                     for index, name in enumerate(names)]}


def model_data(run, all_runs, *, pose_initial_epochs=20):
    best = run.best
    config = run.config_at(best["epoch"])
    names = config.get("class_names") or best["val"].get("class_names")
    per_class = class_metrics(best["val"], names, run.name)
    grouped = {group: class_metrics(scores, names, f"{run.name}/{group}")
               for group, scores in best["val"].get("by_evaluation_type", {}).items()}
    initialized = (run.config_events[0].get("initialized_from")
                   or run.config_events[0]["config"].get("initialized_from"))
    source_training = None
    if isinstance(initialized, dict):
        # Export the parent's trajectory separately, never silently splice stages.
        checkpoint_parts = PurePosixPath(str(initialized.get("checkpoint", "")).replace("\\", "/")).parts
        parent_name = checkpoint_parts[-3] if len(checkpoint_parts) >= 3 else None
        parent = next((item for item in all_runs if item.name == parent_name), None)
        source_epoch = initialized.get("source_epoch")
        if (parent and isinstance(source_epoch, int)
                and parent.config.get("split_sha256") == config["split_sha256"]
                and parent.config.get("annotation_sha256") == config["annotation_sha256"]
                and parent.config.get("model") == config["model"]
                and any(row["epoch"] == source_epoch for row in parent.rows)):
            source_training = {"run_name": parent.name, "through_epoch": source_epoch,
                               "config": parent.config_at(source_epoch),
                               "epochs": epoch_data([row for row in parent.rows if row["epoch"] <= source_epoch]),
                               "note": "Separate initialization trajectory through source checkpoint epoch. "
                                       "Link comes from logged checkpoint provenance; checkpoint bytes are not in the metrics export."}
            source_training["curve_y_axes"] = curve_y_axes(source_training["epochs"])
            source_training["curve_windows"] = (
                pose_curve_windows(source_training["epochs"], pose_initial_epochs)
                if config["model"] in POSE_MODELS else [])
    epochs = epoch_data(run.rows)
    return {"model": config["model"], "display_name": MODEL_LABELS.get(config["model"], config["model"]),
            "run_name": run.name, "config": config, "latest_config": run.config,
            "selected_epoch": best["epoch"], "best": best["val"], "epochs": epochs,
            "curve_y_axes": curve_y_axes(epochs),
            "curve_windows": (pose_curve_windows(epochs, pose_initial_epochs)
                              if config["model"] in POSE_MODELS else []),
            "per_class": per_class, "per_class_by_evaluation_type": grouped,
            "class_recall_blocks": class_recall_blocks(best["val"], names, run.name),
            "config_events": run.config_events,
            "checkpoint_provenance": {"initialized_from": initialized,
                                      "curve_scope": "current stage only" if initialized else "run history, including logged resumes",
                                      "checkpoint_bytes_verified": False},
            "initialization_source_training": source_training,
            "warnings": run.warnings, "files": {}}


def build_chart_data(source_path, *, models=MODELS, split_sha256=None, annotation_sha256=None,
                     selection=None, pose_initial_epochs=20):
    # Validate even an image-only request, so a typo does not silently pass.
    pose_curve_windows([], pose_initial_epochs)
    source = MetricsSource(source_path)
    runs, skipped = [], []
    try:
        for name in source.runs:
            run, reason = read_run(source, name)
            if run is not None:
                runs.append(run)
            else:
                skipped.append({"run_name": name, "reason": reason})
        selected, split_sha256, annotation_sha256 = select_runs(
            runs, models, split_sha256, annotation_sha256, selection)
        return {"schema_version": 1, "metric_units": "fractions", "loss_units": "cross_entropy",
                "checkpoint_selection": "Maximum pooled validation macro_f1; earliest epoch on a tie.",
                "run_selection": "Explicit run names where supplied, otherwise highest selected validation macro F1 within each model.",
                "split_sha256": split_sha256, "annotation_sha256": annotation_sha256,
                "figure_styles": {"class_recall": {"palette": list(RECALL_PALETTE),
                                  "minimum": 0, "maximum": 1, "missing": recall_colors(None),
                                  "cell_text": "Correct / support, then class recall percentage; N/A for zero support."}},
                "source": {**source.info, "read_file_sha256": source.read_hashes},
                "notes": ["Test metrics are never read or used for selection.",
                          "Selected checkpoints are validation-tuned results, not an unbiased final-test estimate.",
                          "Pooled macro F1 is not the average of held-out-view and unseen-moment macro F1.",
                          "Image and pose runs can have different eligible samples; compare their populations before interpreting rankings.",
                          "An epoch log does not prove process completion or checkpoint persistence."],
                "skipped_runs": skipped,
                "models": [model_data(run, runs, pose_initial_epochs=pose_initial_epochs)
                           for run in selected]}
    finally:
        source.close()


def render_figures(data, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
        from matplotlib.ticker import MaxNLocator, MultipleLocator, PercentFormatter
    except ImportError as exc:
        raise RuntimeError("Install analysis dependencies with: python -m pip install -r install/requirements-analysis.txt") from exc
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "svg.fonttype": "none"})
    colors = {"train": "#557084", "val": "#007F86", "f1": "#CE7825",
              "held_out_view": "#5865B2", "unseen_moment": "#A7557E"}

    def save(fig, model, name):
        destination = output / model["model"]
        destination.mkdir(parents=True, exist_ok=True)
        files = {}
        for extension in ("png", "svg"):
            path = destination / f"{name}.{extension}"
            fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight",
                        metadata={"Date": None} if extension == "svg" else {})
            files[extension] = path.relative_to(output).as_posix()
        plt.close(fig)
        model["files"][name] = files

    def curve_base(model, ylabel):
        fig, ax = plt.subplots(figsize=(8.8, 4.5), layout="constrained")
        ax.set_title(model["display_name"] + (" · current stage" if model["checkpoint_provenance"]["initialized_from"] else ""), loc="left", pad=12)
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E2E7EA", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.axvline(model["selected_epoch"], color="#353F49", linewidth=1.1, linestyle=":",
                   label=f"Selected checkpoint (epoch {model['selected_epoch']})")
        return fig, ax

    def class_plot(model, grouped=False, group=None):
        rows = model["per_class"] if group is None else model["per_class_by_evaluation_type"][group]
        fig, ax = plt.subplots(figsize=(8.8, 6.0), layout="constrained")
        names = [row["class_name"].replace("_", " ") for row in rows]
        positions = list(range(len(rows)))
        if grouped:
            series = [("Pooled validation", model["per_class"], colors["val"])]
            series.extend((GROUP_LABELS[kind], model["per_class_by_evaluation_type"][kind], colors[kind])
                          for kind in GROUPS if kind in model["per_class_by_evaluation_type"])
        else:
            series = [("Precision", rows, colors["train"]), ("Recall", rows, colors["f1"]), ("F1", rows, colors["val"])]
        width = 0.75 / len(series)
        for index, (label, values, color) in enumerate(series):
            metric = "f1" if grouped else label.lower()
            ys = [position + (index - (len(series) - 1) / 2) * width for position in positions]
            ax.barh(ys, [row[metric] for row in values], height=width * 0.9, label=label, color=color)
        ax.set_yticks(positions, names)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.02)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="x", color="#E2E7EA", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_xlabel("F1" if grouped else "Score")
        scope = "Validation" if group is None else GROUP_LABELS[group]
        ax.set_title(f"{model['display_name']} · {scope} · epoch {model['selected_epoch']}", loc="left", pad=12)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(series), frameon=False, fontsize=9)
        return fig

    def recall_plot(models):
        names = [row["class_name"] for row in models[0]["class_recall_blocks"]["rows"]]
        if any([row["class_name"] for row in model["class_recall_blocks"]["rows"]] != names
               for model in models[1:]):
            raise ValueError("Cannot combine recall blocks with different class names or class order")
        fig, axes = plt.subplots(1, len(models), figsize=(2.0 + 4.7 * len(models), 7.2),
                                 layout="constrained", squeeze=False)
        for index, (ax, model) in enumerate(zip(axes[0], models)):
            block = model["class_recall_blocks"]
            ax.imshow([[to_rgb(cell["fill"]) for cell in row["cells"]] for row in block["rows"]],
                      aspect="auto")
            ax.set_title(f"{model['display_name']}\nEpoch {model['selected_epoch']}", pad=12, fontsize=13)
            ax.set_xticks(range(3), ["Overall", "Held-out\nview", "Unseen\nmoment"], fontsize=10)
            ax.set_yticks(range(len(names)), [name.replace("_", " ") for name in names] if index == 0
                          else [""] * len(names))
            ax.tick_params(axis="both", length=0, pad=8)
            ax.set_xticks([value - .5 for value in range(4)], minor=True)
            ax.set_yticks([value - .5 for value in range(len(names) + 1)], minor=True)
            ax.grid(which="minor", color="white", linewidth=1.3)
            ax.tick_params(which="minor", length=0)
            for row_index, row in enumerate(block["rows"]):
                for column, cell in enumerate(row["cells"]):
                    ax.text(column, row_index, cell["label"], ha="center", va="center",
                            fontsize=9.5, color=cell["text"])
            for spine in ax.spines.values():
                spine.set_visible(False)
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1),
                                cmap=LinearSegmentedColormap.from_list("class_recall", RECALL_PALETTE)),
                                ax=axes[0].tolist(), orientation="horizontal", fraction=.04, pad=.05,
                                ticks=[0, .2, .4, .6, .8, 1], format=PercentFormatter(1), shrink=.6)
        colorbar.set_label("Correct predictions per class (recall). Each cell: correct / support and percentage.")
        colorbar.outline.set_visible(False)
        fig.suptitle("Correct predictions by class\nValidation checkpoint selected by overall macro F1",
                     fontsize=15, weight="bold")
        return fig

    def curves(model, prefix=""):
        rows = model["epochs"]
        xs = [row["epoch"] for row in rows]
        def apply_y_axis(ax, name):
            bounds = model["curve_y_axes"][name]
            ax.set_ylim(bounds["min"], bounds["max"])
            ax.yaxis.set_major_locator(MultipleLocator(bounds["major_unit"]))

        fig, ax = curve_base(model, "Cross-entropy loss")
        for key, label, color in (("train_loss", "Train", colors["train"]), ("val_loss", "Validation", colors["val"])):
            ax.plot(xs, [row[key] for row in rows], label=label, color=color, linewidth=1.8)
        apply_y_axis(ax, "loss")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=9)
        save(fig, model, prefix + "loss")
        fig, ax = curve_base(model, "Score")
        for key, label, color, style in (
                ("train_accuracy", "Train accuracy", colors["train"], "-"),
                ("val_accuracy", "Validation accuracy", colors["val"], "-"),
                ("val_macro_f1", "Validation macro F1", colors["f1"], "-")):
            if any(row[key] is not None for row in rows):
                ax.plot(xs, [float("nan") if row[key] is None else row[key] for row in rows],
                        label=label, color=color, linestyle=style, linewidth=1.5)
        apply_y_axis(ax, "metrics")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=9)
        save(fig, model, prefix + "metrics")
        if any(row[f"{group}_macro_f1"] is not None for row in rows for group in GROUPS):
            fig, ax = curve_base(model, "Macro F1")
            for key, label, color in (("val_macro_f1", "Pooled validation", colors["val"]),
                                     *((f"{group}_macro_f1", GROUP_LABELS[group], colors[group]) for group in GROUPS)):
                ax.plot(xs, [float("nan") if row[key] is None else row[key] for row in rows],
                        label=label, color=color, linewidth=1.5)
            apply_y_axis(ax, "evaluation_metrics")
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=9)
            save(fig, model, prefix + "evaluation_metrics")

    def window_curves(model, prefix=""):
        windows = model.get("curve_windows", [])
        if not windows:
            return
        series = {
            "loss": ("Cross-entropy loss",
                     (("train_loss", "Train", colors["train"]),
                      ("val_loss", "Validation", colors["val"]))),
            "metrics": ("Score",
                        (("train_accuracy", "Train accuracy", colors["train"]),
                         ("val_accuracy", "Validation accuracy", colors["val"]),
                         ("val_macro_f1", "Validation macro F1", colors["f1"]))),
            "evaluation_metrics": ("Macro F1",
                                   (("val_macro_f1", "Pooled validation", colors["val"]),
                                    *((f"{group}_macro_f1", GROUP_LABELS[group], colors[group])
                                      for group in GROUPS)))}
        for name, (ylabel, chart_series) in series.items():
            if name == "evaluation_metrics" and not any(
                    row[f"{group}_macro_f1"] is not None
                    for window in windows for row in window["epochs"] for group in GROUPS):
                continue
            fig, axes = plt.subplots(1, len(windows), figsize=(7.2 * len(windows), 4.8),
                                     layout="constrained", squeeze=False)
            stage = " · current stage" if model["checkpoint_provenance"]["initialized_from"] else ""
            fig.suptitle(f"{model['display_name']}{stage}\n"
                         "Separate epoch windows · independent Y scales", fontsize=14, weight="bold")
            for ax, window in zip(axes[0], windows):
                rows = window["epochs"]
                xs = [row["epoch"] for row in rows]
                first, last = window["first_epoch"], window["last_epoch"]
                label = "Initial epochs" if window["key"] == "initial" else "Later epochs"
                ax.set_title(f"{label}: {first}–{last}", loc="left", pad=12, fontsize=12)
                ax.set_xlabel("Epoch")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_ylabel(ylabel)
                ax.grid(axis="y", color="#E2E7EA", linewidth=0.7)
                ax.set_axisbelow(True)
                for key, label, color in chart_series:
                    if any(row[key] is not None for row in rows):
                        ax.plot(xs, [float("nan") if row[key] is None else row[key] for row in rows],
                                label=label, color=color, linewidth=1.5,
                                marker="o" if len(rows) == 1 else None)
                selected = model["selected_epoch"]
                if first <= selected <= last:
                    ax.axvline(selected, color="#353F49", linewidth=1.1, linestyle=":",
                               label=f"Selected checkpoint (epoch {selected})")
                # A checkpoint outside this window must not extend its X range.
                margin = max((last - first) * .02, .5)
                ax.set_xlim(first - margin, last + margin)
                bounds = window["curve_y_axes"][name]
                ax.set_ylim(bounds["min"], bounds["max"])
                ax.yaxis.set_major_locator(MultipleLocator(bounds["major_unit"]))
                if name != "loss":
                    ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16),
                          ncol=2, frameon=False, fontsize=9)
            save(fig, model, prefix + name + "_windows")

    for model in data["models"]:
        curves(model)
        window_curves(model)
        initialization = model.get("initialization_source_training")
        if initialization:
            source_model = {**model, "epochs": initialization["epochs"],
                            "curve_y_axes": initialization["curve_y_axes"],
                            "curve_windows": initialization["curve_windows"],
                            "selected_epoch": initialization["through_epoch"],
                            "display_name": model["display_name"] + " · initialization training",
                            "checkpoint_provenance": {"initialized_from": None}}
            curves(source_model, prefix="initialization_")
            window_curves(source_model, prefix="initialization_")
        save(class_plot(model), model, "per_class")
        save(class_plot(model, grouped=True), model, "class_f1")
        save(recall_plot([model]), model, "class_recall")
        for group in GROUPS:
            if group in model["per_class_by_evaluation_type"]:
                save(class_plot(model, group=group), model, f"per_class_{group}")
    for family, model_names in (("image", {"image", "image-dinov3"}),
                                ("pose", {"pose", "pose-wide", "pose-attention"})):
        models = [model for model in data["models"] if model["model"] in model_names]
        if models:
            # An empty model directory saves these family figures at the output root.
            destination = {"model": "", "files": {}}
            save(recall_plot(models), destination, f"class_recall_{family}")
            data.setdefault("files", {}).update(destination["files"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Metrics ZIP or local trainings directory")
    parser.add_argument("--output", type=Path, required=True, help="Destination for PNG, SVG, and chart_data.json")
    parser.add_argument("--split-sha256", help="Required when the input contains multiple split versions")
    parser.add_argument("--annotation-sha256", help="Required when the selected split has multiple annotation versions")
    parser.add_argument("--selection", type=Path, help="JSON model-to-run mapping, optionally with split/annotation hashes")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--pose-initial-epochs", type=int, default=20,
                        help="Separate the first N logged pose epochs in additional figures; 0 disables them")
    parser.add_argument("--data-only", action="store_true", help="Export chart JSON without importing matplotlib")
    args = parser.parse_args(argv)
    try:
        selection = read_json(args.selection.read_bytes()) if args.selection else {}
        if not isinstance(selection, dict):
            raise ValueError("Selection JSON must be an object")
        for key in ("split_sha256", "annotation_sha256"):
            cli_value = getattr(args, key)
            file_value = selection.get(key)
            if cli_value and file_value and cli_value != file_value:
                raise ValueError(f"Conflicting {key} in CLI and selection file")
            setattr(args, key, cli_value or file_value)
        mapping = selection.get("models", selection)
        data = build_chart_data(args.input, models=tuple(dict.fromkeys(args.models)),
                                split_sha256=args.split_sha256, annotation_sha256=args.annotation_sha256,
                                selection=mapping, pose_initial_epochs=args.pose_initial_epochs)
        args.output.mkdir(parents=True, exist_ok=True)
        if not args.data_only:
            render_figures(data, args.output)
        (args.output / "chart_data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        choice = {"split_sha256": data["split_sha256"], "annotation_sha256": data["annotation_sha256"],
                  "models": {model["model"]: model["run_name"] for model in data["models"]}}
        (args.output / "selection.json").write_text(json.dumps(choice, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Plot export failed: {exc}\n")
    for model in data["models"]:
        print(f"{model['display_name']}: {model['run_name']} | epoch {model['selected_epoch']} | "
              f"validation macro F1 {model['best']['macro_f1']:.4f}")
        for warning in model["warnings"]:
            print(f"WARNING [{model['run_name']}]: {warning}")
    print(f"Saved chart data and figures to {args.output.resolve()}")


if __name__ == "__main__":
    main()
