"""Apple Silicon device selection; the hardware checks run only where MPS exists."""

import os
import unittest
from unittest.mock import patch

RUN_TORCH = os.environ.get("JIUJITSU_RUN_TORCH_TESTS") == "1"


@unittest.skipUnless(RUN_TORCH, "Set JIUJITSU_RUN_TORCH_TESTS=1 for MPS checks")
class MpsTests(unittest.TestCase):
    def test_unavailable_mps_fails_clearly(self):
        import torch
        from src.Train.engine import resolve_device
        with patch.object(torch.backends.mps, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "MPS was requested"):
                resolve_device("mps")

    def test_checkpoints_without_mps_state_still_restore(self):
        import torch
        from src.Train.checkpoints import random_state, restore_random_state
        generator = torch.Generator()
        state = random_state({"train": generator})
        state.pop("mps")
        restore_random_state(state, {"train": generator})

    def test_mps_random_state_round_trip(self):
        import torch
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable")
        from src.Train.checkpoints import random_state, restore_random_state
        state = random_state({})
        first = torch.rand(4, device="mps").cpu()
        restore_random_state(state, {})
        self.assertTrue(torch.equal(first, torch.rand(4, device="mps").cpu()))


if __name__ == "__main__":
    unittest.main()
