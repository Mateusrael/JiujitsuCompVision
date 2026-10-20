"""Optional Jupyter controls over saved metrics and checkpoint inference."""

import json
from pathlib import Path

from src.Modules.registry import MODEL_NAMES, POSE_MODEL_NAMES


def _notebook_dependencies():
    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display
    except ImportError as error:
        raise ImportError("Install install/requirements-notebook.txt in this kernel's environment") from error
    return widgets, clear_output, display


class RunBrowser:
    """Browse saved logs; refreshing never loads a model or starts training."""

    def __init__(self, trainings_dir):
        from src.Eval.exploration_plots import catalog_runs
        widgets, self._clear, self._display = _notebook_dependencies()
        self.trainings_dir = Path(trainings_dir)
        self.catalog = []
        self.run = None
        self.model = widgets.Dropdown(options=[("All models", None), *[(m, m) for m in MODEL_NAMES]],
                                      description="Model:")
        self.run_name = widgets.Dropdown(options=[], description="Run:",
                                         layout=widgets.Layout(width="95%"),
                                         style={"description_width": "80px"})
        self.view = widgets.Dropdown(options=["Training curves", "Class recall", "Confusion matrix",
                                              "Configuration"], description="View:")
        self.cohort = widgets.Dropdown(options=["overall", "held_out_view", "unseen_moment"],
                                       description="Cohort:")
        self.score_view = widgets.Dropdown(options=["Accuracy + F1", "Evaluation groups"],
                                           description="Curve scores:")
        self.epoch = widgets.Text(value="best", description="Epoch:", placeholder="best, last, or number")
        self.initial_epochs = widgets.BoundedIntText(value=20, min=0, max=10000,
                                                     description="Pose initial:")
        self.refresh_button = widgets.Button(description="Refresh runs", icon="refresh")
        self.plot_button = widgets.Button(description="Show", icon="line-chart", button_style="primary")
        self.output = widgets.Output()
        self.widget = widgets.VBox([
            widgets.HBox([self.model, self.refresh_button]),
            self.run_name,
            widgets.HBox([self.view, self.cohort, self.epoch]),
            widgets.HBox([self.initial_epochs, self.score_view, self.plot_button]), self.output,
        ])
        self.refresh_button.on_click(lambda _: self.refresh())
        self.model.observe(lambda change: self._set_options(), names="value")
        self.plot_button.on_click(lambda _: self.show())
        self._catalog_runs = catalog_runs
        self.refresh()

    def _set_options(self):
        previous = self.run_name.value
        rows = [r for r in self.catalog if self.model.value in (None, r.get("model"))]
        rows = [r for r in rows if r.get("best_macro_f1") is not None]
        options = [(f"{r['model']} | F1 {r['best_macro_f1']:.3f} | {r['run_name']}", r["run_name"])
                   for r in rows]
        self.run_name.options = options
        if previous in [value for _, value in options]:
            self.run_name.value = previous
        elif options:
            self.run_name.value = options[0][1]

    def refresh(self):
        with self.output:
            self._clear(wait=True)
            try:
                self.catalog = self._catalog_runs(self.trainings_dir)
                self._set_options()
                usable = sum(r.get("best_macro_f1") is not None for r in self.catalog)
                print(f"{usable} runs with metrics; {len(self.catalog) - usable} empty or unreadable.")
                print("Scores use each run's own split. Compare only matching split/annotation hashes.")
                for row in self.catalog:
                    if row.get("best_macro_f1") is None:
                        print(f"  {row['run_name']}: {row.get('error') or row.get('status')}")
            except (OSError, ValueError) as error:
                self.catalog = []
                self._set_options()
                print(f"{type(error).__name__}: {error}")

    def show(self):
        import matplotlib.pyplot as plt
        from src.Eval.exploration_plots import (
            load_run, plot_class_recall, plot_confusion_matrix, plot_training_curves)
        with self.output:
            self._clear(wait=True)
            try:
                if not self.run_name.value:
                    raise ValueError("No run selected. Refresh after training has logged an epoch.")
                self.run = load_run(self.trainings_dir, self.run_name.value)
                epoch = self.epoch.value.strip()
                epoch = int(epoch) if epoch.isdigit() else epoch
                if self.view.value == "Configuration":
                    selected = (self.run.best["epoch"] if epoch == "best" else
                                self.run.rows[-1]["epoch"] if epoch == "last" else epoch)
                    if not isinstance(selected, int) or selected not in [r["epoch"] for r in self.run.rows]:
                        raise ValueError("Choose best, last, or a logged epoch number")
                    print(json.dumps(self.run.config_at(selected), indent=2))
                    return
                if self.view.value == "Training curves":
                    metrics = (("train_accuracy", "val_accuracy", "val_macro_f1")
                               if self.score_view.value == "Accuracy + F1" else
                               ("val_macro_f1", "held_out_view_macro_f1", "unseen_moment_macro_f1"))
                    fig = plot_training_curves(self.run, pose_initial_epochs=self.initial_epochs.value,
                                               metrics=metrics)
                elif self.view.value == "Class recall":
                    fig = plot_class_recall(self.run, epoch=epoch)
                else:
                    fig = plot_confusion_matrix(self.run, epoch=epoch, cohort=self.cohort.value)
                self._display(fig)
                plt.close(fig)
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                print(f"{type(error).__name__}: {error}")


def plot_predictions(data, records, predictions, *, overlay=True, min_confidence=0.0):
    """Return original images/pose overlays beside the top predicted classes."""
    import matplotlib.pyplot as plt
    if not records or len(records) != len(predictions):
        raise ValueError("Provide one prediction for each nonempty record list")
    fig, axes = plt.subplots(len(records), 2, figsize=(13, 3.8 * len(records)),
                             gridspec_kw={"width_ratios": [1.6, 1]}, squeeze=False,
                             layout="constrained")
    for (image_ax, score_ax), record, result in zip(axes, records, predictions):
        if record["image"] != result["image"]:
            raise ValueError("Record and prediction image IDs differ")
        try:
            image_ax.imshow(data.open_image(record))
            if overlay:
                data.draw_pose(image_ax, record, min_confidence=min_confidence)
        except FileNotFoundError as error:
            image_ax.text(.5, .5, str(error), ha="center", va="center",
                          wrap=True, transform=image_ax.transAxes)
        image_ax.set_axis_off()
        image_ax.set_title(f"{record['image']} | source {result['source']} | frame {result['frame']}\n"
                           f"Label: {result['actual']} | Prediction: {result['predicted']}",
                           fontsize=11, color="#166534" if result["correct"] else "#B91C1C")
        top = result["top_k"]
        score_ax.barh([entry["class_name"] for entry in top],
                      [entry["probability"] for entry in top], color="#147D78")
        score_ax.invert_yaxis()
        score_ax.set_xlim(0, 1)
        score_ax.set_xlabel("Softmax score")
        cohort = result.get("evaluation_type")
        score_ax.set_title(f"{result['split']} / {cohort}" if cohort else result["split"], fontsize=11)
        for index, entry in enumerate(top):
            score_ax.text(min(entry["probability"] + .015, .91), index,
                          f"{entry['probability']:.1%}", va="center", fontsize=10)
    return fig


def plot_model_input(session, record):
    """Inspect the exact deterministic tensor passed to the loaded model."""
    import matplotlib.pyplot as plt
    from src.Train.engine import make_dataset
    tensor, _ = make_dataset(session.model_name, [record], session.config, training=False)[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].imshow(session.data.open_image(record))
    session.data.draw_pose(axes[0], record)
    axes[0].set_title("Original frame and supplied poses")
    axes[0].set_axis_off()
    if session.model_name in POSE_MODEL_NAMES:
        joints = tensor.reshape(2, 17, 3).numpy()
        for athlete, color in enumerate(("#147D78", "#C36C2C")):
            observed = joints[athlete, :, 2] > 0
            axes[1].scatter(joints[athlete, observed, 0], joints[athlete, observed, 1],
                            color=color, label=f"Athlete {athlete + 1}")
            for index in range(17):
                if observed[index]:
                    axes[1].annotate(str(index), joints[athlete, index, :2], fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_aspect("equal")
        axes[1].legend()
        axes[1].set_title("102 features: 2 × 17 × (x, y, confidence)")
        axes[1].set_xlabel("Pair-normalized x")
        axes[1].set_ylabel("Pair-normalized y")
    else:
        import torch
        pixels = tensor * torch.tensor([.229, .224, .225])[:, None, None]
        pixels += torch.tensor([.485, .456, .406])[:, None, None]
        axes[1].imshow(pixels.permute(1, 2, 0).clamp(0, 1).numpy())
        axes[1].set_axis_off()
        axes[1].set_title("224 × 224 model input (normalization reversed for display)")
    return fig, tensor


class PredictionBrowser:
    """Load one checkpoint at a time and inspect filtered, bounded sample batches."""

    def __init__(self, data, run_browser, *, device="cpu"):
        widgets, self._clear, self._display = _notebook_dependencies()
        self.data, self.run_browser = data, run_browser
        self.session = None
        self.loaded_selection = None
        self.records, self.predictions = [], []
        self.checkpoint = widgets.Dropdown(options=["best", "last"], description="Checkpoint:")
        self.device = widgets.Text(value=device, description="Device:", placeholder="cpu or cuda:0")
        self.load_button = widgets.Button(description="Load selected run", icon="download")
        self.unload_button = widgets.Button(description="Unload model", icon="eject")
        self.split = widgets.Dropdown(options=["val", "train", "test"], description="Split:")
        self.class_name = widgets.Dropdown(options=[("All classes", None)], description="Class:")
        self.source = widgets.Dropdown(options=[("All sources", None)], description="Source:")
        self.cohort = widgets.Dropdown(options=[("Both", None), ("held_out_view", "held_out_view"),
                                               ("unseen_moment", "unseen_moment")],
                                       description="Eval type:")
        self.frame_min = widgets.IntText(value=0, description="Frame min:")
        self.frame_max = widgets.IntText(value=-1, description="Max (-1=all):")
        self.count = widgets.BoundedIntText(value=3, min=1, max=12, description="Show:")
        self.pool_size = widgets.BoundedIntText(value=64, min=1, max=1024, description="Search pool:")
        self.result_filter = widgets.Dropdown(options=["Any", "Incorrect", "Correct", "Lowest confidence"],
                                              description="Results:")
        self.fixed_seed = widgets.Checkbox(value=False, description="Repeatable sample")
        self.seed = widgets.IntText(value=42, description="Seed:")
        self.overlay = widgets.Checkbox(value=True, description="Pose overlay")
        self.confidence = widgets.FloatSlider(value=0.0, min=0, max=1, step=.05,
                                               description="Overlay min:")
        self.sample_button = widgets.Button(description="New random samples", button_style="primary",
                                             icon="random")
        self.image_id = widgets.Text(description="Image ID:", placeholder="Optional exact image ID")
        self.exact_button = widgets.Button(description="Predict this image")
        self.sample_id = widgets.Dropdown(options=[], description="Inspect:")
        self.input_button = widgets.Button(description="Show model input")
        self.status, self.output, self.input_output = widgets.Output(), widgets.Output(), widgets.Output()
        self.widget = widgets.VBox([
            widgets.HTML("<b>Uses the run selected in the metrics browser.</b>"),
            widgets.HBox([self.checkpoint, self.device, self.load_button, self.unload_button]),
            self.status,
            widgets.HBox([self.split, self.class_name, self.source]),
            widgets.HBox([self.cohort, self.frame_min, self.frame_max]),
            widgets.HBox([self.count, self.result_filter, self.pool_size]),
            widgets.HBox([self.fixed_seed, self.seed, self.overlay]),
            widgets.HBox([self.confidence, self.sample_button]),
            widgets.HBox([self.image_id, self.exact_button]),
            self.output,
            widgets.HBox([self.sample_id, self.input_button]), self.input_output,
        ])
        self.load_button.on_click(lambda _: self.load())
        self.unload_button.on_click(lambda _: self.close())
        self.sample_button.on_click(lambda _: self.sample())
        self.exact_button.on_click(lambda _: self.sample(exact=True))
        self.input_button.on_click(lambda _: self.show_input())
        self.split.observe(lambda change: self._update_cohort(), names="value")
        self.set_data(data)

    def set_data(self, data):
        """Replace the verified data snapshot and release a now-unmatched model."""
        self.close()
        self.data = data
        self.class_name.options = [("All classes", None), *[(n, n) for n in data.class_names]]
        self.source.options = [("All sources", None), *[(n, n) for n in data.sources]]
        self.class_name.value = None
        self.source.value = None

    def _update_cohort(self):
        self.cohort.disabled = self.split.value == "train"
        if self.cohort.disabled:
            self.cohort.value = None

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None
        self.loaded_selection = None
        self.records, self.predictions = [], []
        self.sample_id.options = []
        for output in (self.status, self.output, self.input_output):
            with output:
                self._clear(wait=False)

    def load(self):
        from src.Eval.exploration import ModelExplorer
        self.close()
        with self.status:
            try:
                name = self.run_browser.run_name.value
                if not name:
                    raise ValueError("Select a run in the metrics browser first")
                self.session = ModelExplorer.from_run(
                    self.run_browser.trainings_dir / name, self.data,
                    checkpoint=self.checkpoint.value, device=self.device.value.strip())
                self.loaded_selection = (name, self.checkpoint.value, self.device.value.strip())
                total = sum(p.numel() for p in self.session.model.parameters())
                print(f"Loaded {name}\n{self.session.model_name}, checkpoint epoch "
                      f"{self.session.checkpoint_epoch}, {total:,} parameters, {self.device.value}.")
                print("Eval mode; augmentation disabled. Softmax scores are not calibrated confidence.")
            except (OSError, ValueError, RuntimeError, ImportError) as error:
                print(f"{type(error).__name__}: {error}")

    def sample(self, *, exact=False):
        import matplotlib.pyplot as plt
        with self.output:
            self._clear(wait=True)
            try:
                if self.session is None:
                    raise ValueError("Load a checkpoint first")
                selected = (self.run_browser.run_name.value, self.checkpoint.value,
                            self.device.value.strip())
                if selected != self.loaded_selection:
                    raise ValueError("Run, checkpoint, or device selection changed. Click Load selected run.")
                frame_max = self.frame_max.value if self.frame_max.value >= 0 else None
                filters = dict(split=self.split.value, class_name=self.class_name.value,
                               source=self.source.value, evaluation_type=self.cohort.value,
                               frame_range=(self.frame_min.value, frame_max))
                if exact:
                    candidates = self.data.select(self.session.model_name, **filters)
                    records = [r for r in candidates if r["image"] == self.image_id.value.strip()]
                    if not records:
                        raise ValueError("That image is absent or outside the current eligible filters")
                else:
                    count = self.count.value if self.result_filter.value == "Any" else max(
                        self.pool_size.value, self.count.value)
                    records = self.data.sample(self.session.model_name, count=count,
                                               seed=self.seed.value if self.fixed_seed.value else None,
                                               **filters)
                predictions = self.session.predict(records, top_k=3, batch_size=16)
                pairs = list(zip(records, predictions))
                tested = len(pairs)
                if not exact:
                    if self.result_filter.value in ("Incorrect", "Correct"):
                        correct = self.result_filter.value == "Correct"
                        pairs = [(r, p) for r, p in pairs if p["correct"] == correct]
                    elif self.result_filter.value == "Lowest confidence":
                        pairs.sort(key=lambda item: max(item[1]["probabilities"]))
                    pairs = pairs[:self.count.value]
                print(f"{self.session.model_name}, checkpoint epoch {self.session.checkpoint_epoch}.")
                print(f"Evaluated {tested} sampled images; showing {len(pairs)}. "
                      "This sample is not a dataset-wide accuracy estimate.")
                self.records = [r for r, _ in pairs]
                self.predictions = [p for _, p in pairs]
                self.sample_id.options = [r["image"] for r in self.records]
                if self.records:
                    self.sample_id.value = self.records[0]["image"]
                if not pairs:
                    print("No matching results in this random pool. Try another batch or a larger pool.")
                    return
                fig = plot_predictions(self.data, self.records, self.predictions,
                                       overlay=self.overlay.value, min_confidence=self.confidence.value)
                self._display(fig)
                plt.close(fig)
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                print(f"{type(error).__name__}: {error}")

    def show_input(self):
        import matplotlib.pyplot as plt
        with self.input_output:
            self._clear(wait=True)
            try:
                if self.session is None or not self.sample_id.value:
                    raise ValueError("Predict samples first")
                record = next(r for r in self.records if r["image"] == self.sample_id.value)
                fig, self.last_input_tensor = plot_model_input(self.session, record)
                self._display(fig)
                plt.close(fig)
                print(f"Input tensor: {tuple(self.last_input_tensor.shape)}, {self.last_input_tensor.dtype}")
                if self.session.model_name in POSE_MODEL_NAMES:
                    print("The exact tensor is available as predictor.last_input_tensor.")
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                print(f"{type(error).__name__}: {error}")
