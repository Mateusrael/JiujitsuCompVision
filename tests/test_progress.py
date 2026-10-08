"""Progress reporting must leave optimization, scores, and RNG state unchanged."""

import copy
import os
import unittest
from unittest.mock import patch


class RecordingProgress:
    """Record the information shown to users without writing terminal output."""

    instances = []

    def __init__(self, *args, **kwargs):
        self.options = kwargs
        self.updates = 0
        self.postfixes = []
        self.closed = False
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def update(self, count=1):
        self.updates += count

    def set_postfix(self, ordered_dict=None, **kwargs):
        kwargs.pop("refresh", None)
        self.postfixes.append(dict(ordered_dict or {}, **kwargs))

    def close(self):
        self.closed = True


@unittest.skipUnless(os.environ.get("JIUJITSU_RUN_TORCH_TESTS") == "1",
                     "Set JIUJITSU_RUN_TORCH_TESTS=1 for CPU progress checks")
class ProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def setUp(self):
        self.torch.manual_seed(17)
        RecordingProgress.instances.clear()

    def fixtures(self):
        torch = self.torch
        from torch.utils.data import TensorDataset
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(),
                                    torch.nn.Dropout(0.3), torch.nn.Linear(8, 3))
        dataset = TensorDataset(torch.randn(7, 4), torch.tensor([0, 1, 2, 0, 2, 1, 0]))
        return model, dataset

    def loader(self, dataset):
        torch = self.torch
        return torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True,
                                          generator=torch.Generator().manual_seed(91))

    def test_training_progress_preserves_metrics_weights_and_rng(self):
        from src.Train.engine import train_epoch
        torch = self.torch
        original, dataset = self.fixtures()
        initial_rng = torch.get_rng_state()
        results, states, random_states = [], [], []
        for visible in (False, True):
            model = copy.deepcopy(original)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            torch.set_rng_state(initial_rng)
            with patch("src.Train.engine.tqdm", RecordingProgress):
                results.append(train_epoch(model, self.loader(dataset), optimizer,
                    torch.device("cpu"), progress=visible, description="Training fixture"))
            states.append(model.state_dict())
            random_states.append(torch.get_rng_state())
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1]["samples"], 7)
        for key in states[0]:
            torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
        self.assertTrue(torch.equal(*random_states))
        hidden, shown = RecordingProgress.instances
        self.assertTrue(hidden.options["disable"])
        self.assertFalse(shown.options["disable"])
        self.assertEqual(shown.options["desc"], "Training fixture")
        self.assertEqual(shown.options["total"], 3)
        self.assertEqual(shown.updates, 3)
        self.assertEqual(len(shown.postfixes), 3)
        for key in ("loss", "accuracy"):
            self.assertAlmostEqual(float(shown.postfixes[-1][key]), results[1][key], delta=0.000051)
        self.assertTrue(hidden.closed and shown.closed)

    def test_validation_and_test_progress_preserve_scores_and_report_every_batch(self):
        from src.Train.engine import score_model
        torch = self.torch
        model, dataset = self.fixtures()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        initial_rng = torch.get_rng_state()
        results, random_states = [], []
        for visible, description in ((False, "Validation fixture"), (True, "Test fixture")):
            torch.set_rng_state(initial_rng)
            with patch("src.Train.engine.tqdm", RecordingProgress):
                results.append(score_model(model, self.loader(dataset), torch.device("cpu"),
                    ["a", "b", "c"], progress=visible, description=description))
            random_states.append(torch.get_rng_state())
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1]["samples"], 7)
        self.assertTrue(torch.equal(*random_states))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        shown = RecordingProgress.instances[-1]
        self.assertEqual(shown.options["desc"], "Test fixture")
        self.assertEqual(shown.updates, 3)
        self.assertEqual(len(shown.postfixes), 3)
        for key in ("loss", "accuracy"):
            self.assertAlmostEqual(float(shown.postfixes[-1][key]), results[1][key], delta=0.000051)
        self.assertTrue(shown.closed)

    def test_progress_closes_if_model_execution_fails(self):
        from src.Train.engine import score_model
        torch = self.torch
        model, dataset = self.fixtures()
        with patch.object(model, "forward", side_effect=RuntimeError("fixture failure")), \
                patch("src.Train.engine.tqdm", RecordingProgress):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                score_model(model, self.loader(dataset), torch.device("cpu"), ["a", "b", "c"])
        self.assertTrue(RecordingProgress.instances[-1].closed)


if __name__ == "__main__":
    unittest.main()
