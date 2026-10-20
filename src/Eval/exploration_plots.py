"""Read-only, validation-only plots for notebooks, without training dependencies.

The metrics reader and range/class checks are shared with the presentation
plotter. Functions return Matplotlib figures; they never save files, call
``show()``, change the plotting backend, or read checkpoints or test results.
"""

import math
from pathlib import Path

from src.Scripts.export_metrics import config_order
from src.Scripts.plot_training_metrics import (
    GROUP_LABELS, MODEL_LABELS, POSE_MODELS, RESUME, MetricsSource, Run,
    class_recall, curve_y_axis, epoch_data, pose_curve_windows, read_json, read_run,
)


METRIC_LABELS = {
    "train_loss": "Train loss", "val_loss": "Validation loss",
    "train_accuracy": "Train accuracy", "val_accuracy": "Validation accuracy",
    "val_macro_f1": "Validation macro F1",
    "held_out_view_macro_f1": "Held-out view macro F1",
    "unseen_moment_macro_f1": "Unseen moment macro F1",
}
DEFAULT_METRICS = ("train_accuracy", "val_accuracy", "val_macro_f1")
COHORT_LABELS = {"overall": "Overall validation", **GROUP_LABELS}
COLORS = {"train_loss": "#557084", "val_loss": "#007F86",
          "train_accuracy": "#557084", "val_accuracy": "#007F86",
          "val_macro_f1": "#CE7825", "held_out_view_macro_f1": "#5865B2",
          "unseen_moment_macro_f1": "#A7557E"}


def _source(path):
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Metrics source does not exist: {path}")
    return MetricsSource(path)


def _unlogged_config(source, name):
    """Recover metadata for runs with no completed epoch, including resumed ones."""
    configs = []
    for relative in source.files(name):
        match = RESUME.fullmatch(relative)
        if relative != "config_start.json" and not match:
            continue
        content = source.read(name, relative)
        if content is None:
            continue
        payload = read_json(content)
        if not isinstance(payload.get("config"), dict):
            raise ValueError(f"{relative} has no config object")
        configs.append((config_order(payload, match), payload["config"]))
    return max(configs, key=lambda value: value[0])[1] if configs else {}


def catalog_runs(source, *, model=None, split_sha256=None, annotation_sha256=None):
    """List runs, including empty/live/interrupted logs, grouped by data version.

    Within each (split, annotation, model) group, the best validation macro F1
    comes first. Status describes the log only: ``target_epochs_logged`` does
    not prove the process exited or that a checkpoint was written. Re-run this
    function to refresh a live directory. ZIP exports are immutable snapshots.
    """
    reader = _source(source)
    result = []
    try:
        for name in reader.runs:
            config, error, warnings, run = {}, None, [], None
            try:
                run, reason = read_run(reader, name)
                config = run.config if run else _unlogged_config(reader, name)
                if reason:
                    warnings.append(reason)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                error = str(exc)
                try:
                    config = _unlogged_config(reader, name)
                except (ValueError, TypeError, KeyError, OSError):
                    pass
            if ((model is not None and config.get("model") != model)
                    or (split_sha256 is not None and config.get("split_sha256") != split_sha256)
                    or (annotation_sha256 is not None
                        and config.get("annotation_sha256") != annotation_sha256)):
                continue
            best, last = (run.best, run.rows[-1]) if run else (None, None)
            target = config.get("epochs")
            reached_target = (last is not None and isinstance(target, int)
                              and not isinstance(target, bool) and last["epoch"] >= target)
            status = ("unreadable" if error else "target_epochs_logged" if reached_target
                      else "partial_log" if run else "no_epochs")
            result.append({
                "run_name": name, "model": config.get("model"),
                "split_sha256": config.get("split_sha256"),
                "annotation_sha256": config.get("annotation_sha256"),
                "image_training": config.get("image_training"),
                "epochs": len(run.rows) if run else 0, "target_epochs": target,
                "last_epoch": last["epoch"] if last else None,
                "best_epoch": best["epoch"] if best else None,
                "best_macro_f1": best["val"]["macro_f1"] if best else None,
                "best_val_accuracy": best["val"]["accuracy"] if best else None,
                "best_val_loss": best["val"]["loss"] if best else None,
                "last_train_accuracy": last["train"]["accuracy"] if last else None,
                "last_train_loss": last["train"]["loss"] if last else None,
                "last_val_accuracy": last["val"]["accuracy"] if last else None,
                "last_val_loss": last["val"]["loss"] if last else None,
                "status": status, "error": error,
                "warnings": (run.warnings if run else []) + warnings,
            })
    finally:
        reader.close()
    return sorted(result, key=lambda row: (
        row["split_sha256"] or "", row["annotation_sha256"] or "", row["model"] or "",
        -(row["best_macro_f1"] if row["best_macro_f1"] is not None else -math.inf),
        row["run_name"]))


def load_run(source, run_name) -> Run:
    """Load an explicitly named run; an empty run has no graph to display."""
    reader = _source(source)
    try:
        if run_name not in reader.runs:
            raise ValueError(f"Unknown run: {run_name!r}")
        run, reason = read_run(reader, run_name)
        if run is None:
            raise ValueError(f"{run_name}: {reason}")
        return run
    finally:
        reader.close()


def _plotting():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, MultipleLocator, PercentFormatter
    except ImportError as exc:
        raise RuntimeError(
            "Install plotting dependencies with: "
            "python -m pip install -r install/requirements-analysis.txt"
        ) from exc
    return plt, MaxNLocator, MultipleLocator, PercentFormatter


def _name(run):
    return MODEL_LABELS.get(run.config.get("model"), run.config.get("model", "Model"))


def _axis(ax, rows, keys, *, upper):
    _plt, MaxNLocator, MultipleLocator, PercentFormatter = _plotting()
    bounds = curve_y_axis((row.get(key) for row in rows for key in keys), upper=upper)
    ax.set_ylim(bounds["min"], bounds["max"])
    ax.yaxis.set_major_locator(MultipleLocator(bounds["major_unit"]))
    if upper == 1:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Epoch")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E2E7EA", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)


def _metrics(keys, *, allow_loss=False):
    if isinstance(keys, str):
        keys = (keys,)
    keys = tuple(keys)
    available = tuple(METRIC_LABELS) if allow_loss else tuple(
        key for key in METRIC_LABELS if not key.endswith("_loss"))
    if not keys or len(set(keys)) != len(keys) or any(key not in available for key in keys):
        raise ValueError(f"Choose distinct metrics from: {', '.join(available)}")
    return keys


def plot_training_curves(run, *, pose_initial_epochs=20, metrics=DEFAULT_METRICS):
    """Plot loss and selected scores, with optional separate pose epoch windows.

    ``pose_initial_epochs=0`` shows the full history in a single window.
    Otherwise the first N logged pose epochs and the rest get independent Y
    axes, preserving all recorded epoch numbers. Image runs keep one window.
    """
    metrics = _metrics(metrics)
    rows = epoch_data(run.rows)
    # Validate the option for image models too.
    windows = pose_curve_windows(rows, pose_initial_epochs)
    if run.config.get("model") not in POSE_MODELS or not windows:
        windows = [{"epochs": rows, "first_epoch": rows[0]["epoch"],
                    "last_epoch": rows[-1]["epoch"]}]
    if not any(row.get(key) is not None for row in rows for key in metrics):
        raise ValueError("None of the selected metrics is recorded in this run")
    plt, *_ = _plotting()
    fig, axes = plt.subplots(2, len(windows), squeeze=False, layout="constrained",
                             figsize=(8 if len(windows) == 1 else 14, 8),
                             gridspec_kw={"width_ratios": [1, 2] if len(windows) == 2 else [1]})
    suffix = " · separate epoch windows, independent Y scales" if len(windows) > 1 else ""
    fig.suptitle(_name(run) + suffix, fontsize=14, weight="bold")
    for column, window in enumerate(windows):
        samples = window["epochs"]
        xs = [row["epoch"] for row in samples]
        for line, keys in enumerate((("train_loss", "val_loss"), metrics)):
            ax = axes[line, column]
            for key in keys:
                if any(row.get(key) is not None for row in samples):
                    ax.plot(xs, [float("nan") if row.get(key) is None else row[key] for row in samples],
                            color=COLORS[key], label=METRIC_LABELS[key],
                            linewidth=1.6, marker="o" if len(samples) == 1 else None)
            first, last = window["first_epoch"], window["last_epoch"]
            selected = run.best["epoch"]
            if first <= selected <= last:
                ax.axvline(selected, color="#353F49", linewidth=1, linestyle=":",
                           label=f"Best validation F1: epoch {selected}")
            margin = max((last - first) * .02, .5)
            ax.set_xlim(first - margin, last + margin)
            _axis(ax, samples, keys, upper=None if line == 0 else 1)
            ax.set_ylabel("Cross-entropy loss" if line == 0 else "Score")
            ax.set_title(f"Epochs {first}–{last}", loc="left", fontsize=11)
            ax.legend(fontsize=8, ncol=2, loc="best")
    return fig


def plot_run_comparison(runs, *, metric="val_macro_f1", epoch_range=None):
    """Compare explicitly chosen runs on the same data version.

    A matching split does not imply identical image/pose sample eligibility.
    The caller can inspect each run's ``config['pose_filter']`` before comparing.
    An optional inclusive epoch range changes the visible X range explicitly.
    """
    metric = _metrics((metric,), allow_loss=True)[0]
    runs = list(runs)
    if not runs:
        raise ValueError("Select at least one run")
    for key in ("split_sha256", "annotation_sha256"):
        versions = {run.config.get(key) for run in runs}
        if len(versions) != 1 or None in versions:
            raise ValueError(f"Run comparison requires the same known {key}")
    if epoch_range is not None:
        if (len(epoch_range) != 2 or any(isinstance(value, bool) or not isinstance(value, int)
                                      for value in epoch_range)
                or epoch_range[0] < 1 or epoch_range[0] > epoch_range[1]):
            raise ValueError("epoch_range must be an inclusive (first, last) pair of positive integers")
    plt, *_ = _plotting()
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    all_rows = []
    for index, run in enumerate(runs, 1):
        rows = [row for row in epoch_data(run.rows)
                if epoch_range is None or epoch_range[0] <= row["epoch"] <= epoch_range[1]]
        if not any(row.get(metric) is not None for row in rows):
            plt.close(fig)
            raise ValueError(f"No {metric} values in the requested window for {run.name}")
        all_rows.extend(rows)
        # The numbered legend stays compact; full identities are returned below.
        ax.plot([row["epoch"] for row in rows],
                [float("nan") if row.get(metric) is None else row[metric] for row in rows],
                label=f"{index}. {_name(run)}", linewidth=1.6,
                marker="o" if len(rows) == 1 else None)
    _axis(ax, all_rows, (metric,), upper=None if metric.endswith("_loss") else 1)
    if epoch_range is not None:
        first, last = epoch_range
        ax.set_xlim(first - .5, last + .5)
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(METRIC_LABELS[metric], loc="left")
    ax.legend(frameon=False, fontsize=9)
    # A notebook can access the exact mapping without oversized chart text.
    fig.run_names = tuple(run.name for run in runs)
    return fig


def _scores(run, epoch, cohort):
    if epoch == "best":
        row = run.best
    elif epoch in ("last", "latest"):
        row = run.rows[-1]
    elif isinstance(epoch, int) and not isinstance(epoch, bool):
        row = next((row for row in run.rows if row["epoch"] == epoch), None)
        if row is None:
            raise ValueError(f"Epoch {epoch} is not recorded in {run.name}")
    else:
        raise ValueError("epoch must be 'best', 'last'/'latest', or a recorded epoch number")
    if cohort not in COHORT_LABELS:
        raise ValueError(f"Unknown validation cohort: {cohort!r}")
    scores = row["val"] if cohort == "overall" else row["val"].get("by_evaluation_type", {}).get(cohort)
    if scores is None:
        raise ValueError(f"{cohort} metrics are not recorded at epoch {row['epoch']}")
    names = run.config_at(row["epoch"]).get("class_names") or row["val"].get("class_names")
    return row["epoch"], scores, names


def plot_class_recall(run, *, epoch="best",
                      cohorts=("overall", "held_out_view", "unseen_moment")):
    """Return class recall heatmaps with correct/support counts for one epoch.

    Missing cohorts remain visible as N/A; zero-support classes are also N/A.
    Explicitly requesting only an unavailable cohort raises a useful error.
    """
    if isinstance(cohorts, str):
        cohorts = (cohorts,)
    cohorts = tuple(cohorts)
    if (not cohorts or len(set(cohorts)) != len(cohorts)
            or any(cohort not in COHORT_LABELS for cohort in cohorts)):
        raise ValueError(f"Choose distinct validation cohorts from: {', '.join(COHORT_LABELS)}")
    number, overall, names = _scores(run, epoch, "overall")
    columns = []
    for cohort in cohorts:
        scores = overall if cohort == "overall" else overall.get("by_evaluation_type", {}).get(cohort)
        columns.append(class_recall(scores, names, f"{run.name}/{cohort}/epoch{number}")
                       if scores is not None else None)
    if not any(column is not None for column in columns):
        raise ValueError("None of the requested cohorts is recorded at this epoch")
    plt, *_ = _plotting()
    from matplotlib.colors import to_rgb
    fig, ax = plt.subplots(figsize=(3 + 2 * len(cohorts), max(4, .55 * len(names))),
                            layout="constrained")
    fills = [[to_rgb(column[index]["fill"]) if column else to_rgb("#E8EDF0")
              for column in columns] for index in range(len(names))]
    ax.imshow(fills, aspect="auto")
    for column_index, column in enumerate(columns):
        for row_index in range(len(names)):
            cell = column[row_index] if column else {"label": "N/A", "text": "#374151"}
            ax.text(column_index, row_index, cell["label"], ha="center", va="center",
                    color=cell["text"], fontsize=10)
    ax.set_xticks(range(len(cohorts)), [COHORT_LABELS[cohort] for cohort in cohorts])
    ax.set_yticks(range(len(names)), [name.replace("_", " ") for name in names])
    ax.tick_params(axis="both", length=0)
    ax.set_title(f"{_name(run)} · validation epoch {number}\nCorrect / total and recall",
                 loc="left", pad=12)
    return fig


def plot_confusion_matrix(run, *, epoch="best", cohort="overall", normalize=True):
    """Plot true rows vs predicted columns, optionally normalized by true class.

    Normalized cells display recall fraction plus the raw count. Rows with no
    examples are marked N/A instead of implying perfect or zero performance.
    """
    number, scores, names = _scores(run, epoch, cohort)
    counts = scores.get("confusion_matrix")
    if counts is None:
        raise ValueError("No confusion matrix was logged for the requested epoch/cohort")
    checked = class_recall(scores, names, f"{run.name}/{cohort}/epoch{number}")
    values = [[count / checked[row]["support"] if checked[row]["support"] else float("nan")
               for count in line] for row, line in enumerate(counts)] if normalize else counts
    plt, *_ = _plotting()
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    maximum = 1 if normalize else max((max(row) for row in counts), default=1)
    for row, line in enumerate(counts):
        for column, count in enumerate(line):
            value = values[row][column]
            text = (f"{value:.0%}\n({count})" if checked[row]["support"] else "N/A") if normalize else str(count)
            ax.text(column, row, text, ha="center", va="center", fontsize=8,
                    color="white" if math.isfinite(value) and value > maximum * .55 else "#172033")
    labels = [name.replace("_", " ") for name in names]
    ax.set_xticks(range(len(names)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{_name(run)} · {COHORT_LABELS[cohort]} · epoch {number}", loc="left", pad=12)
    fig.colorbar(image, ax=ax, fraction=.04, pad=.02,
                 label="Fraction of the true class" if normalize else "Samples")
    return fig
