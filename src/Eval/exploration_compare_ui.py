"""Explicit run selection and comparisons for the model explorer notebook."""

from pathlib import Path

from src.Eval.exploration_ui import (
    _new_output, _notebook_dependencies, _set_output_text, _show_figure,
)
from src.Modules.registry import MODEL_NAMES, POSE_MODEL_NAMES


def _run_label(row):
    return f"{row['model']} | F1 {row['best_macro_f1']:.3f} | {row['run_name']}"


class RunSelection:
    """Collect runs with the metrics browser's model/run dropdown pattern."""

    def __init__(self, trainings_dir):
        from src.Eval.exploration_plots import catalog_runs
        widgets, _, _ = _notebook_dependencies()
        self.trainings_dir = Path(trainings_dir)
        self._catalog_runs = catalog_runs
        self.catalog = []
        self._run_names = []
        self.model = widgets.Dropdown(
            options=[("All models", None), *[(name, name) for name in MODEL_NAMES]],
            description="Model:")
        self.search = widgets.Text(description="Search:", placeholder="Filter run names")
        self.run_name = widgets.Dropdown(options=[], description="Run:",
                                         layout=widgets.Layout(width="95%"),
                                         style={"description_width": "80px"})
        self.refresh_button = widgets.Button(description="Refresh runs", icon="refresh")
        self.add_button = widgets.Button(description="Add run", icon="plus")
        self.selected = widgets.Select(options=[], rows=4, description="Selected:",
                                        layout=widgets.Layout(width="95%"),
                                        style={"description_width": "80px"})
        self.remove_button = widgets.Button(description="Remove run", icon="minus")
        self.clear_button = widgets.Button(description="Clear selection", icon="trash")
        self.status = _new_output(widgets)
        self.widget = widgets.VBox([
            widgets.HBox([self.model, self.search, self.refresh_button]), self.run_name,
            self.add_button, self.selected,
            widgets.HBox([self.remove_button, self.clear_button]), self.status,
        ], layout=widgets.Layout(width="100%", height="auto", overflow="visible"))
        self.model.observe(lambda _: self._set_options(), names="value")
        self.search.observe(lambda _: self._set_options(), names="value")
        self.refresh_button.on_click(lambda _: self.refresh())
        self.add_button.on_click(lambda _: self.add())
        self.remove_button.on_click(lambda _: self.remove())
        self.clear_button.on_click(lambda _: self.clear())
        self.refresh()

    @property
    def run_names(self):
        """All added runs, in comparison order; highlight is only for removal."""
        return tuple(self._run_names)

    def _set_options(self):
        previous = self.run_name.value
        query = self.search.value.strip().casefold()
        rows = [row for row in self.catalog
                if self.model.value in (None, row.get("model"))
                and query in row["run_name"].casefold()]
        options = [(_run_label(row), row["run_name"]) for row in rows]
        self.run_name.options = options
        if previous in [name for _, name in options]:
            self.run_name.value = previous
        self.add_button.disabled = not options

    def _set_selected(self):
        previous = self.selected.value
        rows = {row["run_name"]: row for row in self.catalog}
        self.selected.options = [(_run_label(rows[name]), name) for name in self._run_names]
        if previous in self._run_names:
            self.selected.value = previous
        elif self._run_names:
            self.selected.value = self._run_names[-1]
        self.remove_button.disabled = not self._run_names
        self.clear_button.disabled = not self._run_names

    def refresh(self):
        try:
            rows = self._catalog_runs(self.trainings_dir)
            self.catalog = [row for row in rows if row.get("best_macro_f1") is not None]
            available = {row["run_name"] for row in self.catalog}
            removed = [name for name in self._run_names if name not in available]
            self._run_names = [name for name in self._run_names if name in available]
            self._set_options()
            self._set_selected()
            message = f"{len(self.catalog)} runs with metrics. Add each run you want to compare."
            if removed:
                message += "\nRemoved unavailable runs: " + ", ".join(removed)
            _set_output_text(self.status, message)
        except (OSError, ValueError, KeyError) as error:
            _set_output_text(self.status, f"{type(error).__name__}: {error}")

    def add(self):
        name = self.run_name.value
        if name is not None and name not in self._run_names:
            self._run_names.append(name)
            self._set_selected()

    def remove(self):
        if self.selected.value in self._run_names:
            self._run_names.remove(self.selected.value)
            self._set_selected()

    def clear(self):
        self._run_names.clear()
        self._set_selected()


class RunComparisonBrowser:
    """Compare saved curves after explicitly adding runs and clicking Compare."""

    def __init__(self, trainings_dir):
        from src.Eval.exploration_plots import METRIC_LABELS
        widgets, _, _ = _notebook_dependencies()
        self.selector = RunSelection(trainings_dir)
        self.runs = []
        self.metric = widgets.Dropdown(
            options=[(label, key) for key, label in METRIC_LABELS.items()],
            value="val_macro_f1", description="Metric:",
            layout=widgets.Layout(width="360px"))
        self.compare_button = widgets.Button(description="Compare runs", icon="line-chart",
                                              button_style="primary")
        self.output = _new_output(widgets)
        self.widget = widgets.VBox([self.selector.widget,
                                    widgets.HBox([self.metric, self.compare_button]), self.output],
                                   layout=widgets.Layout(width="100%", height="auto", overflow="visible"))
        self.compare_button.on_click(lambda _: self.compare())

    def compare(self):
        from src.Eval.exploration_plots import load_run, plot_run_comparison
        self.runs = []
        self.compare_button.disabled = True
        _set_output_text(self.output, "Loading selected metrics...")
        try:
            if not self.selector.run_names:
                raise ValueError("Add at least one run to the selection first")
            runs = [load_run(self.selector.trainings_dir, name)
                    for name in self.selector.run_names]
            figure = plot_run_comparison(runs, metric=self.metric.value)
            self.runs = runs
            mapping = "\n".join(f"{index}: {run.name}" for index, run in enumerate(runs, 1))
            _set_output_text(self.output, "Matching annotation and split hashes. Image and pose "
                             "models can still use different eligible examples.\n" + mapping)
            _show_figure(self.output, figure)
        except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
            _set_output_text(self.output, f"{type(error).__name__}: {error}")
        finally:
            self.compare_button.disabled = False


class PredictionComparisonBrowser:
    """Compare checkpoints on identical sampled frames, one loaded model at a time."""

    def __init__(self, data, trainings_dir, *, predictor=None, device="cpu", seed=42):
        widgets, _, _ = _notebook_dependencies()
        self.data, self.predictor = data, predictor
        self.selector = RunSelection(trainings_dir)
        self.records, self.predictions = [], {}
        self.checkpoint = widgets.Dropdown(options=["best", "last"], description="Checkpoint:")
        self.device = widgets.Text(value=device, description="Device:")
        self.split = widgets.Dropdown(options=["val", "train", "test"], description="Split:")
        self.cohort = widgets.Dropdown(options=[("Both", None), ("held_out_view", "held_out_view"),
                                               ("unseen_moment", "unseen_moment")],
                                       description="Eval type:")
        self.count = widgets.BoundedIntText(value=3, min=1, max=12, description="Show:")
        self.seed = widgets.IntText(value=seed, description="Seed:")
        self.overlay = widgets.Checkbox(value=True, description="Pose overlay")
        self.compare_button = widgets.Button(description="Compare predictions", icon="images",
                                              button_style="primary")
        self.output = _new_output(widgets)
        self.widget = widgets.VBox([
            self.selector.widget, widgets.HBox([self.checkpoint, self.device]),
            widgets.HBox([self.split, self.cohort]),
            widgets.HBox([self.count, self.seed, self.overlay]), self.compare_button, self.output,
        ], layout=widgets.Layout(width="100%", height="auto", overflow="visible"))
        self.split.observe(lambda _: self._update_cohort(), names="value")
        self.compare_button.on_click(lambda _: self.compare())

    def set_data(self, data):
        """Replace the verified dataset snapshot and discard prior comparisons."""
        self.data = data
        self.records, self.predictions = [], {}
        self.output.outputs = ()

    def _update_cohort(self):
        self.cohort.disabled = self.split.value == "train"
        if self.cohort.disabled:
            self.cohort.value = None

    def compare(self):
        from src.Eval.exploration import ModelExplorer
        from src.Eval.exploration_ui import plot_predictions
        self.records, self.predictions = [], {}
        self.compare_button.disabled = True
        _set_output_text(self.output, "Preparing shared samples...")
        try:
            names = self.selector.run_names
            if not names:
                raise ValueError("Add at least one run to the selection first")
            catalog = {row["run_name"]: row for row in self.selector.catalog}
            selected = [catalog[name] for name in names]
            for row in selected:
                if row.get("model") not in MODEL_NAMES:
                    raise ValueError(f"Unsupported model for {row['run_name']}")
                for key in ("annotation_sha256", "split_sha256"):
                    if row.get(key) != getattr(self.data, key):
                        raise ValueError(f"{row['run_name']} does not match the loaded data's {key}")
            population = "pose" if any(row["model"] in POSE_MODEL_NAMES for row in selected) else "image"
            self.records = self.data.sample(population, split=self.split.value,
                                             evaluation_type=self.cohort.value,
                                             count=self.count.value, seed=self.seed.value)
            if self.predictor is not None:
                self.predictor.close()
            ids = ", ".join(row["image"] for row in self.records)
            _set_output_text(self.output, f"Shared {self.split.value} frames: {ids}\n"
                             f"Eligibility: {population}. This sample is for inspection, not a model ranking.")
            for name in names:
                self.output.append_stdout(f"\nLoading {name}...\n")
                session = ModelExplorer.from_run(self.selector.trainings_dir / name, self.data,
                                                 checkpoint=self.checkpoint.value,
                                                 device=self.device.value.strip())
                try:
                    predictions = session.predict(self.records, top_k=3, batch_size=16)
                    epoch = session.checkpoint_epoch
                finally:
                    session.close()
                self.predictions[name] = predictions
                self.output.append_stdout(f"{name} | checkpoint epoch {epoch}\n")
                # Separate figures keep each sample readable in long widget outputs.
                for record, prediction in zip(self.records, predictions):
                    figure = plot_predictions(self.data, [record], [prediction],
                                               overlay=self.overlay.value)
                    _show_figure(self.output, figure)
        except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
            self.output.append_stderr(f"\n{type(error).__name__}: {error}\n")
        finally:
            self.compare_button.disabled = False
