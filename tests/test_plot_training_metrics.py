"""Tests for validation-only selection, provenance, and class alignment."""

import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile

from src.Scripts import export_metrics, plot_training_metrics as plots
from tests.test_export_metrics import config_bytes, metric, metrics_bytes


class PlotTrainingMetricsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.trainings = self.root / "trainings"
        self.trainings.mkdir()

    def write(self, name, relative, content):
        path = self.trainings / name / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def run_fixture(self, name, scores, **settings):
        self.write(name, "config_start.json", config_bytes(**settings))
        self.write(name, "diagnostics/metrics.jsonl",
                   metrics_bytes(*(metric(epoch, score) for epoch, score in enumerate(scores, 1))))

    def data(self, source=None, **options):
        return plots.build_chart_data(source or self.trainings, models=("pose",), **options)

    def test_selects_correct_split_and_earliest_best_epoch_without_reading_test(self):
        self.run_fixture("old-high-score", [0.99], split_sha256="c" * 64)
        self.run_fixture("selected", [0.6, 0.8, 0.8])
        self.write("selected", "diagnostics/test_metrics.json", b"INVALID JSON MUST NEVER BE READ")
        data = self.data(split_sha256="b" * 64)
        self.assertEqual(data["models"][0]["run_name"], "selected")
        self.assertEqual(data["models"][0]["selected_epoch"], 2)
        self.assertFalse(any("test_metrics" in name for name in data["source"]["read_file_sha256"]))
        with self.assertRaisesRegex(ValueError, "Multiple or no split"):
            self.data()

    def test_multiple_annotation_versions_need_explicit_hash(self):
        self.run_fixture("first", [0.7])
        self.run_fixture("second", [0.9], annotation_sha256="d" * 64)
        with self.assertRaisesRegex(ValueError, "annotation versions"):
            self.data(split_sha256="b" * 64)
        data = self.data(split_sha256="b" * 64, annotation_sha256="a" * 64)
        self.assertEqual(data["models"][0]["run_name"], "first")

    def test_explicit_selection_never_crosses_requested_split(self):
        self.run_fixture("first", [0.7])
        self.run_fixture("second", [0.9], split_sha256="c" * 64)
        with self.assertRaisesRegex(ValueError, "No eligible run"):
            self.data(split_sha256="b" * 64, selection={"pose": "second"})
        data = self.data(selection={"pose": "first"})
        self.assertEqual(data["models"][0]["run_name"], "first")

    def test_duplicate_epochs_and_reversed_epochs_are_rejected(self):
        self.run_fixture("ambiguous", [0.7])
        self.write("ambiguous", "diagnostics/metrics.jsonl", metrics_bytes(metric(1), metric(1)))
        with self.assertRaisesRegex(ValueError, "duplicated epochs"):
            self.data()
        self.write("ambiguous", "diagnostics/metrics.jsonl", metrics_bytes(metric(2), metric(1)))
        with self.assertRaisesRegex(ValueError, "out of order"):
            self.data()

    def test_resume_preserves_unique_epochs_and_reports_checkpoint_config(self):
        self.run_fixture("resumed", [0.9, 0.7])
        self.write("resumed", "config_resume_epoch1_20261011T120000000000Z.json",
                   config_bytes(lr=0.0001, epochs=2, timestamp="2026-10-11T12:00:00+00:00"))
        data = self.data()
        model = data["models"][0]
        self.assertEqual(model["selected_epoch"], 1)
        self.assertEqual(model["config"]["lr"], 0.001)
        self.assertEqual(model["latest_config"]["lr"], 0.0001)
        self.assertEqual(len(model["config_events"]), 2)
        self.assertEqual([row["epoch"] for row in model["epochs"]], [1, 2])

    def test_resume_changed_split_is_rejected(self):
        self.run_fixture("resumed", [0.9, 0.7])
        self.write("resumed", "config_resume_epoch1_20261011T120000000000Z.json",
                   config_bytes(split_sha256="c" * 64))
        with self.assertRaisesRegex(ValueError, "changed split_sha256"):
            self.data()

    def test_per_class_is_aligned_by_name_and_duplicate_names_fail(self):
        self.run_fixture("first", [0.7])
        row = metric(1)
        names = row["val"]["class_names"]
        row["val"]["per_class"].reverse()
        self.write("first", "diagnostics/metrics.jsonl", metrics_bytes(row))
        data = self.data()
        self.assertEqual([value["class_name"] for value in data["models"][0]["per_class"]], names)
        row["val"]["per_class"][0] = row["val"]["per_class"][1]
        self.write("first", "diagnostics/metrics.jsonl", metrics_bytes(row))
        with self.assertRaisesRegex(ValueError, "per-class names"):
            self.data()

    def test_macro_f1_is_not_replaced_by_group_average(self):
        self.run_fixture("first", [0.7])
        row = metric(1)
        row["val"]["macro_f1"] = 0.95
        self.write("first", "diagnostics/metrics.jsonl", metrics_bytes(row))
        with self.assertRaisesRegex(ValueError, "mean class F1"):
            self.data()

    def test_recall_blocks_use_matrix_counts_and_align_class_names(self):
        self.run_fixture("first", [0.7])
        row = metric(1)
        row["val"]["per_class"].reverse()
        held = metric(1)["val"]
        held["confusion_matrix"][0] = [0] * 10
        held["per_class"][0].update(support=0, recall=0, f1=0)
        held["samples"] = 90
        held["macro_f1"] = 0.63
        unseen = metric(1)["val"]
        for index in range(1, 10):
            unseen["confusion_matrix"][index] = [0] * 10
            unseen["per_class"][index].update(support=0, recall=0, f1=0)
        unseen["samples"] = 10
        unseen["macro_f1"] = 0.07
        row["val"]["by_evaluation_type"] = {"held_out_view": held, "unseen_moment": unseen}
        self.write("first", "diagnostics/metrics.jsonl", metrics_bytes(row))
        block = self.data()["models"][0]["class_recall_blocks"]
        self.assertEqual(block["columns"], ["overall", "held_out_view", "unseen_moment"])
        self.assertEqual([item["class_name"] for item in block["rows"]], row["val"]["class_names"])
        overall, zero_support, _unseen = block["rows"][0]["cells"]
        self.assertEqual((overall["correct"], overall["support"], overall["recall"]), (7, 10, 0.7))
        self.assertEqual(overall["count_source"], "confusion_matrix")
        self.assertEqual(overall["label"], "7/10\n70.0%")
        self.assertEqual((zero_support["correct"], zero_support["support"], zero_support["recall"]), (0, 0, None))
        self.assertTrue(zero_support["available"])
        self.assertEqual(zero_support["label"], "N/A\n0 examples")
        legacy = metric(1)["val"]
        unavailable = plots.class_recall_blocks(legacy, legacy["class_names"], "fixture")["rows"][0]["cells"][2]
        self.assertFalse(unavailable["available"])
        self.assertIsNone(unavailable["support"])
        json.dumps(block, allow_nan=False)

    def test_recall_and_support_must_match_confusion_matrix(self):
        names = metric(1)["val"]["class_names"]
        scores = metric(1)["val"]
        scores["per_class"][0]["recall"] = 0.6
        with self.assertRaisesRegex(ValueError, "recall disagrees"):
            plots.class_recall(scores, names, "fixture")
        scores = metric(1)["val"]
        scores["confusion_matrix"][0][1] += 1
        with self.assertRaisesRegex(ValueError, "row support differs"):
            plots.class_recall(scores, names, "fixture")
        scores = metric(1)["val"]
        scores["confusion_matrix_axes"] = {"rows": "predicted", "columns": "true"}
        with self.assertRaisesRegex(ValueError, "unsupported confusion matrix axes"):
            plots.class_recall(scores, names, "fixture")

    def test_recall_without_matrix_requires_exact_integer_count(self):
        scores = metric(1)["val"]
        del scores["confusion_matrix"]
        counts = plots.class_recall(scores, scores["class_names"], "fixture")
        self.assertEqual(counts[0]["correct"], 7)
        self.assertEqual(counts[0]["count_source"], "recall_times_support")
        scores["per_class"][0]["recall"] = 0.75
        with self.assertRaisesRegex(ValueError, "recall disagrees"):
            plots.class_recall(scores, scores["class_names"], "fixture")

    def test_zip_manifest_hashes_are_checked(self):
        self.run_fixture("first", [0.7])
        output = self.root / "metrics.zip"
        export_metrics.build_export(self.trainings, output)
        data = self.data(output)
        self.assertTrue(data["source"]["manifest_verified"])
        altered = self.root / "tampered.zip"
        with zipfile.ZipFile(output) as source, zipfile.ZipFile(altered, "w") as target:
            for name in source.namelist():
                content = source.read(name)
                if name.endswith("metrics.jsonl"):
                    content = metrics_bytes(metric(1, 0.9))
                target.writestr(name, content)
        with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
            self.data(altered)

    def test_unsafe_zip_names_are_rejected_without_extracting(self):
        output = self.root / "unsafe.zip"
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("runs/../../escape", b"not extracted")
        with self.assertRaisesRegex(ValueError, "Unsafe ZIP"):
            self.data(output)
        self.assertFalse((self.root / "escape").exists())

    def test_initialization_trajectory_is_separate_and_stops_at_source_epoch(self):
        self.run_fixture("parent", [0.7, 0.8, 0.6])
        self.run_fixture("child", [0.9])
        payload = json.loads(config_bytes())
        payload["initialized_from"] = {
            "checkpoint": "/experiment/trainings/parent/checkpoints/best.pt", "source_epoch": 2,
            "checkpoint_sha256": "c" * 64}
        self.write("child", "config_start.json", json.dumps(payload).encode())
        model = self.data()["models"][0]
        self.assertEqual(model["run_name"], "child")
        self.assertEqual(len(model["epochs"]), 1)
        parent = model["initialization_source_training"]
        self.assertEqual(parent["run_name"], "parent")
        self.assertEqual([row["epoch"] for row in parent["epochs"]], [1, 2])
        self.assertEqual(model["checkpoint_provenance"]["curve_scope"], "current stage only")
        self.assertEqual(parent["curve_y_axes"], plots.curve_y_axes(parent["epochs"]))
        self.assertNotEqual(parent["curve_y_axes"]["metrics"], model["curve_y_axes"]["metrics"])

    def test_curve_bounds_keep_extrema_of_every_series_and_epoch(self):
        rows = [{"epoch": 1, "train_loss": 1.327, "val_loss": .81,
                 "train_accuracy": .55, "val_accuracy": .74, "val_macro_f1": .66,
                 "held_out_view_macro_f1": .43, "unseen_moment_macro_f1": .87},
                {"epoch": 18, "train_loss": .418, "val_loss": .442,
                 "train_accuracy": .84, "val_accuracy": .839, "val_macro_f1": .809,
                 "held_out_view_macro_f1": None, "unseen_moment_macro_f1": .91}]
        axes = plots.curve_y_axes(rows)
        for name, keys in (("loss", ("train_loss", "val_loss")),
                           ("metrics", ("train_accuracy", "val_accuracy", "val_macro_f1")),
                           ("evaluation_metrics", ("val_macro_f1", "held_out_view_macro_f1",
                                                   "unseen_moment_macro_f1"))):
            with self.subTest(chart=name):
                bounds = axes[name]
                for row in rows:
                    for key in keys:
                        if row[key] is not None:
                            self.assertLessEqual(bounds["min"], row[key])
                            self.assertGreaterEqual(bounds["max"], row[key])
                self.assertGreater(bounds["major_unit"], 0)
                self.assertGreater(bounds["min"], 0)
                if name != "loss":
                    self.assertLessEqual(bounds["max"], 1)
        self.assertLess(axes["evaluation_metrics"]["min"], axes["metrics"]["min"])

    def test_curve_bounds_handle_constant_empty_and_domain_edges(self):
        for upper in (None, 1.0):
            for values in ([0.0], [1.0, 1.0], [.8123], [.8123456789012, .81234567890121],
                           [], [None, float("nan")]):
                with self.subTest(upper=upper, values=values):
                    bounds = plots.curve_y_axis(values, upper=upper)
                    self.assertGreaterEqual(bounds["min"], 0)
                    self.assertGreater(bounds["max"], bounds["min"])
                    self.assertGreater(bounds["major_unit"], 0)
                    if upper is not None:
                        self.assertLessEqual(bounds["max"], upper)
                    for value in values:
                        if value is not None and value == value:
                            self.assertLessEqual(bounds["min"], value)
                            self.assertGreaterEqual(bounds["max"], value)
                    json.dumps(bounds, allow_nan=False)
        with self.assertRaisesRegex(ValueError, "outside"):
            plots.curve_y_axis([-.01])
        with self.assertRaisesRegex(ValueError, "outside"):
            plots.curve_y_axis([1.01], upper=1)

    def test_chart_data_exports_ranges_without_dropping_epochs_or_missing_groups(self):
        self.run_fixture("first", [.7, .8, .75])
        model = self.data()["models"][0]
        self.assertEqual([row["epoch"] for row in model["epochs"]], [1, 2, 3])
        self.assertEqual(model["curve_y_axes"], plots.curve_y_axes(model["epochs"]))
        self.assertEqual(set(model["curve_y_axes"]), {"loss", "metrics", "evaluation_metrics"})
        for row in model["epochs"]:
            row["held_out_view_macro_f1"] = None
            row["unseen_moment_macro_f1"] = None
        bounds = plots.curve_y_axes(model["epochs"])["evaluation_metrics"]
        self.assertLessEqual(bounds["min"], .7)
        self.assertGreaterEqual(bounds["max"], .8)

    def test_pose_windows_partition_every_logged_epoch_and_keep_true_numbers(self):
        self.run_fixture("first", [.7] * 25)
        # Gaps in a log must not become fabricated or renumbered epochs.
        rows = [metric(2 * epoch, .7) for epoch in range(1, 26)]
        for index, row in enumerate(rows):
            row["train"]["loss"] = 1.8 if index < 20 else .2
            row["val"]["loss"] = 1.5 if index < 20 else .3
            row["train"]["accuracy"] = .4 if index < 20 else .95
        self.write("first", "diagnostics/metrics.jsonl", metrics_bytes(*rows))
        model = self.data()["models"][0]
        windows = model["curve_windows"]
        self.assertEqual([window["key"] for window in windows], ["initial", "later"])
        self.assertEqual([(window["first_epoch"], window["last_epoch"]) for window in windows],
                         [(2, 40), (42, 50)])
        self.assertEqual([row for window in windows for row in window["epochs"]], model["epochs"])
        self.assertLess(windows[1]["curve_y_axes"]["loss"]["max"],
                        windows[0]["curve_y_axes"]["loss"]["min"])
        for window in windows:
            for name, keys in (("loss", ("train_loss", "val_loss")),
                               ("metrics", ("train_accuracy", "val_accuracy", "val_macro_f1")),
                               ("evaluation_metrics", ("val_macro_f1", "held_out_view_macro_f1",
                                                       "unseen_moment_macro_f1"))):
                bounds = window["curve_y_axes"][name]
                for row in window["epochs"]:
                    for key in keys:
                        if row[key] is not None:
                            self.assertLessEqual(bounds["min"], row[key])
                            self.assertGreaterEqual(bounds["max"], row[key])
                self.assertLessEqual((bounds["max"] - bounds["min"]) / bounds["major_unit"],
                                     5.0000001)

    def test_pose_windows_short_disabled_and_invalid_requests(self):
        self.run_fixture("first", [.7, .8, .75])
        short = self.data()["models"][0]
        self.assertEqual(len(short["curve_windows"]), 1)
        self.assertEqual(short["curve_windows"][0]["epochs"], short["epochs"])
        disabled = self.data(pose_initial_epochs=0)["models"][0]
        self.assertEqual(disabled["curve_windows"], [])
        self.assertEqual(short["epochs"], disabled["epochs"])
        self.assertEqual(short["curve_y_axes"], disabled["curve_y_axes"])
        self.assertEqual(len(self.data(pose_initial_epochs=1)["models"][0]["curve_windows"]), 2)
        for value in (-1, True, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonnegative integer"):
                self.data(pose_initial_epochs=value)

    def test_image_curves_are_unchanged_by_pose_window_setting(self):
        self.run_fixture("image", [.7, .8, .75], model="image")
        default = plots.build_chart_data(self.trainings, models=("image",))["models"][0]
        disabled = plots.build_chart_data(self.trainings, models=("image",),
                                          pose_initial_epochs=0)["models"][0]
        self.assertEqual(default["curve_windows"], [])
        self.assertEqual(default, disabled)

    def test_initialization_windows_use_own_epochs_and_bounds(self):
        self.run_fixture("parent", [.7] * 25)
        self.run_fixture("child", [.9])
        payload = json.loads(config_bytes())
        payload["initialized_from"] = {
            "checkpoint": "/experiment/trainings/parent/checkpoints/best.pt", "source_epoch": 24}
        self.write("child", "config_start.json", json.dumps(payload).encode())
        model = self.data()["models"][0]
        parent = model["initialization_source_training"]
        self.assertEqual([(window["first_epoch"], window["last_epoch"])
                          for window in parent["curve_windows"]], [(1, 20), (21, 24)])
        self.assertEqual([row for window in parent["curve_windows"] for row in window["epochs"]],
                         parent["epochs"])
        self.assertEqual(len(model["curve_windows"]), 1)
        self.assertNotEqual(parent["curve_windows"][0]["curve_y_axes"]["metrics"],
                            model["curve_windows"][0]["curve_y_axes"]["metrics"])

    def test_incomplete_final_row_is_ignored_and_reported(self):
        self.run_fixture("first", [0.7])
        self.write("first", "diagnostics/metrics.jsonl", metrics_bytes(metric(1)) + b'{"epoch": 2')
        model = self.data()["models"][0]
        self.assertEqual(len(model["epochs"]), 1)
        self.assertIn("Unterminated metrics line", model["warnings"][0])

    def test_presentation_epoch_ticks_are_numeric_with_round_interior_values(self):
        self.assertEqual(plots.presentation_epoch_axis(1, 20),
                         ((0, 20), [0, 5, 10, 15, 20]))
        self.assertEqual(plots.presentation_epoch_axis(21, 300),
                         ((21, 300), [21, 50, 100, 150, 200, 250, 300]))
        self.assertEqual(plots.presentation_epoch_axis(1, 30),
                         ((1, 30), [1, 5, 10, 15, 20, 25, 30]))
        self.assertEqual(plots.presentation_epoch_axis(42, 42), ((41.5, 42.5), [42]))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Optional matplotlib dependency")
    def test_presentation_figures_keep_logged_epoch_coordinates_and_fixed_canvas(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.run_fixture("pose", [.7] * 25)
        self.run_fixture("dino", [.8] * 30, model="image-dinov3")
        data = plots.build_chart_data(self.trainings, models=("pose", "image-dinov3"))
        pose, dino = data["models"]
        # Deliberately introduce a missing epoch: a category axis would compress
        # this gap or invent a value, while a numeric plot must preserve it.
        pose["curve_windows"][1]["epochs"][1]["epoch"] = 22.5
        for model, expected in ((pose, [list(range(1, 21)), [21, 22.5, 23, 24, 25]]),
                                (dino, [list(range(1, 31))])):
            fig = plots.presentation_curve_figure(model)
            try:
                self.assertEqual(len(fig.axes), len(expected) * 2)
                for index, ax in enumerate(fig.axes):
                    for line in ax.lines:
                        self.assertEqual(list(line.get_xdata()), expected[index // 2])
                    self.assertEqual(ax.get_xlim()[1], expected[index // 2][-1])
                    self.assertTrue(all(tick % 1 == 0 for tick in ax.get_xticks()))
            finally:
                plt.close(fig)
        plots.render_presentation_curves(data, self.root / "figures")
        for model in data["models"]:
            path = self.root / "figures" / model["files"]["presentation_curves"]["png"]
            self.assertEqual(struct.unpack(">II", path.read_bytes()[16:24]), (2304, 900))
        for family in ("pose", "image"):
            path = self.root / "figures" / data["files"][f"presentation_recall_{family}"]["png"]
            self.assertEqual(struct.unpack(">II", path.read_bytes()[16:24]), (2304, 1000))


if __name__ == "__main__":
    unittest.main()
