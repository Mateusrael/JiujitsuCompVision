"""Opt-in notebook controls exercised against small local checkpoint fixtures."""

import contextlib
import base64
import io
import json
import os
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("JIUJITSU_RUN_NOTEBOOK_TESTS") == "1",
                     "Set JIUJITSU_RUN_NOTEBOOK_TESTS=1 with optional notebook dependencies")
class ExplorationUITests(unittest.TestCase):
    def setUp(self):
        import torch
        import matplotlib
        matplotlib.use("Agg")
        from tests.test_exploration import fixture, ExplorationInferenceTests
        from tests.test_export_metrics import metrics_bytes, metric
        torch.set_num_threads(1)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.data = fixture(self.root, images=True)
        self.trainings = self.root / "trainings"
        self.checkpoint, _ = ExplorationInferenceTests.checkpoint(self.trainings, self.data)
        run = self.checkpoint.parent.parent
        config = torch.load(self.checkpoint, weights_only=False)["config"]
        (run / "config_start.json").write_text(json.dumps({"config": config}), encoding="utf-8")
        (run / "diagnostics").mkdir()
        (run / "diagnostics/metrics.jsonl").write_bytes(metrics_bytes(metric(1), metric(12, .8)))

    def browsers(self):
        from src.Eval.exploration_ui import RunBrowser, PredictionBrowser
        with contextlib.redirect_stdout(io.StringIO()):
            browser = RunBrowser(self.trainings)
        browser._display = lambda _: None
        predictor = PredictionBrowser(self.data, browser)
        predictor._display = lambda _: None
        self.addCleanup(predictor.close)
        return browser, predictor

    def test_filters_sampling_input_preview_and_stale_checkpoint_selection(self):
        browser, predictor = self.browsers()
        self.assertIsNone(predictor.cohort.value)
        with contextlib.redirect_stdout(io.StringIO()):
            predictor.load()
        self.assertIsNotNone(predictor.session)
        predictor.count.value = 2
        predictor.fixed_seed.value = True
        with contextlib.redirect_stdout(io.StringIO()):
            predictor.sample()
        self.assertEqual(len(predictor.records), 2)
        self.assertTrue(all(p["split"] == "val" for p in predictor.predictions))
        ids = [r["image"] for r in predictor.records]
        with contextlib.redirect_stdout(io.StringIO()):
            predictor.sample()
            predictor.show_input()
        self.assertEqual(ids, [r["image"] for r in predictor.records])
        self.assertEqual(tuple(predictor.last_input_tensor.shape), (102,))
        from src.Eval.exploration_ui import plot_predictions
        figure = plot_predictions(self.data, predictor.records, predictor.predictions)
        self.assertEqual(figure.axes[1].get_title(), "val")
        import matplotlib.pyplot as plt
        plt.close(figure)
        predictor.cohort.value = "unseen_moment"
        predictor.split.value = "train"
        self.assertIsNone(predictor.cohort.value)
        self.assertTrue(predictor.cohort.disabled)
        predictor.checkpoint.value = "last"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            predictor.sample()
        self.assertIn("selection changed", predictor.output.outputs[0]["text"])
        self.assertEqual(predictor.records, [])
        self.assertEqual(predictor.sample_id.options, ())
        self.assertEqual(predictor.input_output.outputs, ())
        predictor.set_data(self.data)
        self.assertIsNone(predictor.session)
        self.assertIsNone(predictor.loaded_selection)
        self.assertIsNone(predictor.class_name.value)
        self.assertIsNone(predictor.source.value)

    def test_run_graphs_and_bounded_correctness_search(self):
        browser, predictor = self.browsers()
        for view in ("Training curves", "Class recall", "Confusion matrix", "Configuration"):
            browser.view.value = view
            browser.epoch.value = "last"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                browser.show()
            self.assertNotIn("Error:", output.getvalue())
            self.assertIsNotNone(browser.run)
        with contextlib.redirect_stdout(io.StringIO()):
            predictor.load()
        predictor.result_filter.value = "Incorrect"
        predictor.pool_size.value = 8
        predictor.count.value = 2
        with contextlib.redirect_stdout(io.StringIO()):
            predictor.sample()
        self.assertLessEqual(len(predictor.predictions), 2)
        self.assertTrue(all(not row["correct"] for row in predictor.predictions))
        predictor.result_filter.value = "Any"
        predictor.image_id.value = "0100003"
        with contextlib.redirect_stdout(io.StringIO()):
            predictor.sample(exact=True)
        self.assertEqual([r["image"] for r in predictor.records], ["0100003"])

    def test_prediction_callbacks_publish_complete_pngs_and_replace_old_results(self):
        from PIL import Image, ImageStat
        import matplotlib.pyplot as plt
        _, predictor = self.browsers()
        predictor.load_button.click()
        # Exercise the actual callbacks without an IPython display-capture context.
        predictor.count.value = 3
        predictor.fixed_seed.value = True
        for exact in (False, True, False):
            if exact:
                predictor.image_id.value = "0100003"
                predictor.exact_button.click()
            else:
                predictor.sample_button.click()
            images = [o["data"]["image/png"] for o in predictor.output.outputs
                      if o["output_type"] == "display_data"]
            self.assertEqual(len(images), 1)
            with Image.open(io.BytesIO(base64.b64decode(images[0]))) as image:
                expected = 1 if exact else 3
                self.assertEqual(len(predictor.records), expected)
                self.assertGreater(image.width, 1000)
                self.assertGreater(image.height, expected * 300)
                # Each sample row contains visible content, including the bottom row.
                for row in range(expected):
                    band = image.crop((0, row * image.height // expected,
                                       image.width, (row + 1) * image.height // expected))
                    self.assertGreater(max(ImageStat.Stat(band.convert("RGB")).stddev), 20)
            self.assertEqual(plt.get_fignums(), [])
        predictor.input_button.click()
        self.assertTrue(any(o["output_type"] == "display_data" for o in predictor.input_output.outputs))
        from unittest.mock import patch
        with patch("src.Eval.exploration_ui.plot_model_input", side_effect=FileNotFoundError("missing frame")):
            predictor.input_button.click()
        self.assertIsNone(predictor.last_input_tensor)
        self.assertIn("missing frame", predictor.input_output.outputs[0]["text"])
        predictor.image_id.value = "not-an-image"
        predictor.exact_button.click()
        self.assertIn("absent", predictor.output.outputs[0]["text"])
        self.assertEqual(predictor.records, [])
        self.assertEqual(predictor.input_output.outputs, ())

    def test_png_renderer_closes_figure_even_when_encoding_fails(self):
        import matplotlib.pyplot as plt
        import ipywidgets as widgets
        from unittest.mock import patch
        from src.Eval.exploration_ui import _show_figure
        figure = plt.figure()
        with patch.object(figure, "savefig", side_effect=OSError("render failed")):
            with self.assertRaisesRegex(OSError, "render failed"):
                _show_figure(widgets.Output(), figure)
        self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
