"""Notebook metrics reading is robust to unfinished jobs and explicit selection."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

from src.Eval import exploration_plots as plots
from src.Scripts import export_metrics
from tests.test_export_metrics import config_bytes, metric, metrics_bytes


class ExplorationFixture(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.trainings = self.root / "trainings"
        self.trainings.mkdir()

    def write(self, run_name, relative, content):
        path = self.trainings / run_name / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def run_fixture(self, name, scores, **config):
        self.write(name, "config_start.json", config_bytes(**config))
        if scores is not None:
            self.write(name, "diagnostics/metrics.jsonl",
                       metrics_bytes(*(metric(index, score)
                                       for index, score in enumerate(scores, 1))))


class ExplorationMetricsTests(ExplorationFixture):
    def test_catalog_retains_empty_partial_and_target_logs(self):
        self.run_fixture("empty", None)
        self.run_fixture("partial", [.6, .8])
        self.run_fixture("target", [.9, .8], epochs=2)
        self.write("partial", "diagnostics/metrics.jsonl",
                   metrics_bytes(metric(1, .6), metric(2, .8)) + b'{"epoch": 3')
        self.write("target", "diagnostics/test_metrics.json", b"NOT VALID JSON")
        catalog = plots.catalog_runs(self.trainings)
        self.assertEqual([row["run_name"] for row in catalog], ["target", "partial", "empty"])
        self.assertEqual([row["status"] for row in catalog],
                         ["target_epochs_logged", "partial_log", "no_epochs"])
        self.assertEqual((catalog[0]["best_epoch"], catalog[0]["last_epoch"]), (1, 2))
        self.assertEqual((catalog[1]["epochs"], catalog[1]["best_macro_f1"]), (2, .8))
        self.assertIn("Unterminated", catalog[1]["warnings"][0])
        self.assertIsNone(catalog[2]["best_macro_f1"])
        with self.assertRaisesRegex(ValueError, "Missing diagnostics"):
            plots.load_run(self.trainings, "empty")

    def test_invalid_run_does_not_hide_valid_siblings(self):
        self.run_fixture("ambiguous", [.7, .8])
        self.write("ambiguous", "diagnostics/metrics.jsonl",
                   metrics_bytes(metric(1), metric(1)))
        self.run_fixture("good", [.8])
        rows = {row["run_name"]: row for row in plots.catalog_runs(self.trainings)}
        self.assertEqual(rows["ambiguous"]["status"], "unreadable")
        self.assertIn("duplicated epochs", rows["ambiguous"]["error"])
        self.assertEqual(rows["good"]["best_macro_f1"], .8)
        with self.assertRaisesRegex(ValueError, "duplicated epochs"):
            plots.load_run(self.trainings, "ambiguous")

    def test_catalog_filters_versions_models_and_recovers_unlogged_resume_config(self):
        self.run_fixture("pose-old", [.9], split_sha256="c" * 64)
        self.run_fixture("image", [.8], model="image", image_training="input-head")
        self.run_fixture("resume", None)
        self.write("resume", "config_resume_epoch2_20261011T120000Z.json",
                   config_bytes(epochs=100, timestamp="2026-10-11T12:00:00+00:00"))
        rows = plots.catalog_runs(self.trainings, model="pose", split_sha256="b" * 64)
        self.assertEqual([row["run_name"] for row in rows], ["resume"])
        self.assertEqual(rows[0]["target_epochs"], 100)
        self.assertEqual(plots.catalog_runs(self.trainings, annotation_sha256="f" * 64), [])

    def test_explicit_selection_cannot_escape_source_and_zip_works(self):
        self.run_fixture("wanted", [.8, .8])
        self.run_fixture("higher", [.9])
        run = plots.load_run(self.trainings, "wanted")
        self.assertEqual((run.name, run.best["epoch"]), ("wanted", 1))
        with self.assertRaisesRegex(ValueError, "Unknown run"):
            plots.load_run(self.trainings, "../outside")
        archive = self.root / "metrics.zip"
        export_metrics.build_export(self.trainings, archive)
        self.assertEqual(plots.load_run(archive, "wanted").rows, run.rows)
        self.assertEqual([row["run_name"] for row in plots.catalog_runs(archive)],
                         ["higher", "wanted"])


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Optional Matplotlib is not installed")
class ExplorationFiguresTests(ExplorationFixture):
    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_pose_windows_cover_every_epoch_and_keep_independent_axes(self):
        self.run_fixture("pose", [.6] * 3 + [.9] * 3)
        run = plots.load_run(self.trainings, "pose")
        fig = plots.plot_training_curves(run, pose_initial_epochs=3)
        self.assertEqual(len(fig.axes), 4)
        first, last = fig.axes[:2]
        self.assertEqual(list(first.lines[0].get_xdata()), [1, 2, 3])
        self.assertEqual(list(last.lines[0].get_xdata()), [4, 5, 6])
        self.assertNotEqual(fig.axes[2].get_ylim(), fig.axes[3].get_ylim())
        self.assertLess(first.get_xlim()[1], 4)
        self.assertGreater(last.get_xlim()[0], 3)
        self.assertEqual(len(plots.plot_training_curves(run, pose_initial_epochs=0).axes), 2)
        with self.assertRaisesRegex(ValueError, "None of the selected"):
            plots.plot_training_curves(run, metrics=("held_out_view_macro_f1",))

    def test_image_history_unsplit_and_comparison_refuses_different_versions(self):
        self.run_fixture("image", [.6, .7], model="image")
        self.run_fixture("image-other", [.8, .9], model="image", split_sha256="c" * 64)
        first = plots.load_run(self.trainings, "image")
        second = plots.load_run(self.trainings, "image-other")
        self.assertEqual(len(plots.plot_training_curves(first, pose_initial_epochs=1).axes), 2)
        with self.assertRaisesRegex(ValueError, "same known split_sha256"):
            plots.plot_run_comparison([first, second])
        figure = plots.plot_run_comparison([first], epoch_range=(2, 2))
        self.assertEqual(list(figure.axes[0].lines[0].get_xdata()), [2])
        self.assertEqual(figure.run_names, ("image",))

    def test_confusion_epoch_selection_counts_and_missing_recall_cohorts(self):
        self.run_fixture("pose", [.9, .6])
        run = plots.load_run(self.trainings, "pose")
        best = plots.plot_confusion_matrix(run)
        latest = plots.plot_confusion_matrix(run, epoch="latest", normalize=False)
        last_alias = plots.plot_confusion_matrix(run, epoch="last", normalize=False)
        self.assertEqual(best.axes[0].texts[0].get_text(), "90%\n(9)")
        self.assertEqual(latest.axes[0].texts[0].get_text(), "6")
        self.assertEqual(last_alias.axes[0].texts[0].get_text(), "6")
        self.assertAlmostEqual(float(best.axes[0].images[0].get_array()[0][0]), .9)
        recall = plots.plot_class_recall(run, epoch=2)
        self.assertEqual(recall.axes[0].texts[0].get_text(), "6/10\n60.0%")
        self.assertIn("N/A", [text.get_text() for text in recall.axes[0].texts])
        with self.assertRaisesRegex(ValueError, "not recorded"):
            plots.plot_confusion_matrix(run, cohort="unseen_moment")
        with self.assertRaisesRegex(ValueError, "None of the requested"):
            plots.plot_class_recall(run, cohorts=("unseen_moment",))
        with self.assertRaisesRegex(ValueError, "not recorded"):
            plots.plot_class_recall(run, epoch=999)


if __name__ == "__main__":
    unittest.main()
