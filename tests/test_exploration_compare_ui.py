"""Notebook comparisons preserve selection, sample identity and model cleanup."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tests.test_exploration_plots import ExplorationFixture


ENABLED = os.environ.get("JIUJITSU_RUN_NOTEBOOK_TESTS") == "1"


def output_text(output):
    return "".join(item.get("text", "") for item in output.outputs)


def catalog_row(name, model="pose", score=.8, **hashes):
    return {"run_name": name, "model": model, "best_macro_f1": score,
            "annotation_sha256": "a" * 64, "split_sha256": "b" * 64, **hashes}


def add_run(selector, name):
    selector.model.value = None
    selector.run_name.value = name
    selector.add_button.click()


@unittest.skipUnless(ENABLED, "Set JIUJITSU_RUN_NOTEBOOK_TESTS=1 with notebook dependencies")
class RunSelectionTests(unittest.TestCase):
    def test_model_filters_add_remove_and_refresh_keep_the_comparison_selection(self):
        from src.Eval.exploration_compare_ui import RunSelection
        rows = [catalog_row("pose-a"), catalog_row("pose-b", score=.9),
                catalog_row("image-a", model="image"), catalog_row("empty", score=None)]
        with patch("src.Eval.exploration_plots.catalog_runs", return_value=rows) as catalog:
            selector = RunSelection("trainings")
            add_run(selector, "pose-b")
            add_run(selector, "pose-b")
            selector.model.value = "image"
            selector.add_button.click()
            self.assertEqual(selector.run_names, ("pose-b", "image-a"))
            self.assertEqual([value for _, value in selector.run_name.options], ["image-a"])
            self.assertIn("F1 0.800", selector.run_name.options[0][0])
            selector.refresh_button.click()
            self.assertEqual(selector.run_names, ("pose-b", "image-a"))
            self.assertEqual(selector.run_name.value, "image-a")
            selector.search.value = "POSE"
            self.assertEqual(selector.run_names, ("pose-b", "image-a"))
            self.assertEqual(selector.run_name.options, ())
            selector.search.value = ""
            selector.selected.value = "pose-b"
            selector.remove_button.click()
            self.assertEqual(selector.run_names, ("image-a",))
            catalog.return_value = [catalog_row("pose-a")]
            selector.refresh()
            self.assertEqual(selector.run_names, ())
            self.assertIn("Removed unavailable runs: image-a", output_text(selector.status))
            self.assertTrue(selector.add_button.disabled)
            selector.model.value = "pose"
            selector.add_button.click()
            selector.clear_button.click()
            self.assertEqual(selector.run_names, ())


@unittest.skipUnless(ENABLED, "Set JIUJITSU_RUN_NOTEBOOK_TESTS=1 with notebook dependencies")
class RunComparisonTests(ExplorationFixture):
    def setUp(self):
        super().setUp()
        import matplotlib
        matplotlib.use("Agg")

    def test_selected_curves_publish_a_png_and_reject_different_splits(self):
        from src.Eval.exploration_compare_ui import RunComparisonBrowser
        self.run_fixture("a", [.6, .8])
        self.run_fixture("b", [.7, .9])
        self.run_fixture("other-split", [.8], split_sha256="c" * 64)
        browser = RunComparisonBrowser(self.trainings)
        browser.compare_button.click()
        self.assertIn("Add at least one run", output_text(browser.output))
        add_run(browser.selector, "a")
        add_run(browser.selector, "b")
        self.assertEqual(browser.runs, [])
        browser.metric.value = "val_loss"
        browser.compare_button.click()
        self.assertEqual([run.name for run in browser.runs], ["a", "b"])
        self.assertTrue(any("image/png" in item.get("data", {}) for item in browser.output.outputs))
        self.assertIn("1: a\n2: b", output_text(browser.output))
        add_run(browser.selector, "other-split")
        browser.compare_button.click()
        self.assertIn("same known split_sha256", output_text(browser.output))
        self.assertEqual(browser.runs, [])
        self.assertFalse(any("image/png" in item.get("data", {}) for item in browser.output.outputs))
        self.assertFalse(browser.compare_button.disabled)


@unittest.skipUnless(ENABLED, "Set JIUJITSU_RUN_NOTEBOOK_TESTS=1 with notebook dependencies")
class PredictionComparisonTests(unittest.TestCase):
    def setUp(self):
        import matplotlib
        matplotlib.use("Agg")
        from tests.test_exploration import fixture
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.data = fixture(self.root, images=True)
        self.rows = [catalog_row("pose-a", annotation_sha256=self.data.annotation_sha256,
                                 split_sha256=self.data.split_sha256),
                     catalog_row("image-a", model="image",
                                 annotation_sha256=self.data.annotation_sha256,
                                 split_sha256=self.data.split_sha256)]
        self.events = []
        self.predictor = Mock()
        self.predictor.close.side_effect = lambda: self.events.append("unload-existing")
        selection_patch = patch("src.Eval.exploration.ModelExplorer.select_from_run",
                                side_effect=self.fake_selection)
        self.selection = selection_patch.start()
        self.addCleanup(selection_patch.stop)

    def fake_selection(self, run_path, data, *, checkpoint, **filters):
        model = "pose" if run_path.name.startswith("pose") else "image"
        return data.select(model, **filters)

    def browser(self):
        from src.Eval.exploration_compare_ui import PredictionComparisonBrowser
        with patch("src.Eval.exploration_plots.catalog_runs", return_value=self.rows):
            browser = PredictionComparisonBrowser(self.data, self.root / "trainings",
                                                   predictor=self.predictor, seed=19)
        browser.count.value = 2
        for row in self.rows:
            add_run(browser.selector, row["run_name"])
        return browser

    def fake_session(self, run_path, data, **kwargs):
        name = run_path.name
        self.events.append(("load", name, kwargs))
        session = Mock(checkpoint_epoch=7)
        session.pose_source = "annotations"
        session.draw_pose.side_effect = data.draw_pose

        def predict(records, **_):
            self.events.append(("predict", name, id(records), tuple(row["image"] for row in records)))
            return [{**data.describe(row), "predicted": data.describe(row)["actual"],
                     "correct": True, "probabilities": [.8, .1, .1],
                     "top_k": [{"class_name": data.describe(row)["actual"], "probability": .8}]}
                    for row in records]

        session.predict.side_effect = predict
        session.close.side_effect = lambda: self.events.append(("close", name))
        return session

    def test_same_eligible_frames_sequential_loading_repeatable_seed_and_explicit_button(self):
        browser = self.browser()
        with patch("src.Eval.exploration.ModelExplorer.from_run", side_effect=self.fake_session) as load:
            self.assertFalse(load.called)
            browser.compare_button.click()
        self.assertEqual(list(browser.predictions), ["pose-a", "image-a"])
        self.assertEqual(len(browser.records), 2)
        self.assertTrue(all(self.data.describe(row)["both_poses_present"] for row in browser.records))
        predict_events = [event for event in self.events if isinstance(event, tuple) and event[0] == "predict"]
        self.assertEqual(predict_events[0][2:], predict_events[1][2:])
        self.assertEqual(self.events[0], "unload-existing")
        self.assertEqual([event[:2] for event in self.events[1:]],
                         [("load", "pose-a"), ("predict", "pose-a"), ("close", "pose-a"),
                          ("load", "image-a"), ("predict", "image-a"), ("close", "image-a")])
        self.assertEqual(sum("image/png" in item.get("data", {}) for item in browser.output.outputs), 4)
        first_ids = [row["image"] for row in browser.records]
        with patch("src.Eval.exploration.ModelExplorer.from_run", side_effect=self.fake_session):
            browser.compare()
        self.assertEqual(first_ids, [row["image"] for row in browser.records])
        browser.set_data(self.data)
        self.assertEqual((browser.records, browser.predictions, browser.output.outputs), ([], {}, ()))

    def test_failed_inference_closes_model_and_does_not_load_the_next_one(self):
        browser = self.browser()
        failed = Mock()
        failed.predict.side_effect = RuntimeError("synthetic inference failure")
        with patch("src.Eval.exploration.ModelExplorer.from_run", return_value=failed) as load:
            browser.compare()
        failed.close.assert_called_once()
        self.assertEqual(load.call_count, 1)
        self.assertIn("synthetic inference failure", output_text(browser.output))
        self.assertEqual(browser.predictions, {})
        self.assertFalse(browser.compare_button.disabled)

    def test_data_mismatch_rejected_before_sampling_or_loading(self):
        browser = self.browser()
        browser.selector.catalog[0] = {**browser.selector.catalog[0], "split_sha256": "c" * 64}
        with patch("src.Eval.exploration.ModelExplorer.from_run") as load, \
                patch.object(self.data, "sample", wraps=self.data.sample) as sample:
            browser.compare()
        self.assertFalse(sample.called)
        self.assertFalse(load.called)
        self.predictor.close.assert_not_called()
        self.assertIn("does not match the loaded data's split_sha256", output_text(browser.output))

    def test_image_only_comparison_keeps_image_eligibility_and_training_has_no_cohort(self):
        browser = self.browser()
        browser.selector.selected.value = "pose-a"
        browser.selector.remove()
        browser.cohort.value = "held_out_view"
        browser.split.value = "train"
        self.assertIsNone(browser.cohort.value)
        self.assertTrue(browser.cohort.disabled)
        with patch.object(self.data, "select", wraps=self.data.select) as select, \
                patch("src.Eval.exploration.ModelExplorer.from_run", side_effect=self.fake_session):
            browser.compare()
        self.assertEqual(select.call_args.args, ("image",))
        self.assertEqual(select.call_args.kwargs["split"], "train")
        self.assertIsNone(select.call_args.kwargs["evaluation_type"])

    def test_checkpoint_populations_are_intersected_before_sampling(self):
        browser = self.browser()
        browser.count.value = 3
        eligible = {"pose-a": {"0100000", "0100001", "0100004", "0100005"},
                    "image-a": {"0100000", "0100004", "0100005"}}
        self.selection.side_effect = lambda run_path, data, **_: [
            row for row in data.records if row["image"] in eligible[run_path.name]]
        with patch("src.Eval.exploration.ModelExplorer.from_run", side_effect=self.fake_session):
            browser.compare()
        self.assertEqual({row["image"] for row in browser.records},
                         {"0100000", "0100004", "0100005"})
        self.assertEqual(set(browser.predictions), {"pose-a", "image-a"})
        eligible["image-a"] = {"0100002"}
        with patch("src.Eval.exploration.ModelExplorer.from_run") as load:
            browser.compare()
        load.assert_not_called()
        self.assertIn("No common eligible", output_text(browser.output))

    def test_annotation_and_detected_models_use_their_own_eligibility(self):
        self.rows.append(catalog_row("pose-annotations",
                                     annotation_sha256=self.data.annotation_sha256,
                                     split_sha256=self.data.split_sha256))
        browser = self.browser()
        browser.count.value = 3
        eligible = {"pose-a": {"0100000", "0100004", "0100005"},
                    "image-a": {"0100000", "0100004", "0100005"},
                    "pose-annotations": {"0100004", "0100005"}}
        self.selection.side_effect = lambda run_path, data, **_: [
            row for row in data.records if row["image"] in eligible[run_path.name]]
        with patch("src.Eval.exploration.ModelExplorer.from_run", side_effect=self.fake_session):
            browser.compare()
        self.assertEqual({row["image"] for row in browser.records}, {"0100004", "0100005"})
        self.assertEqual(set(browser.predictions), set(eligible))


if __name__ == "__main__":
    unittest.main()
