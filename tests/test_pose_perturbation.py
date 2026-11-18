"""Training-only pose degradation that imitates detector errors."""

import argparse
import os
import unittest

from src.Train.config import resolve_settings

RUN_TORCH = os.environ.get("JIUJITSU_RUN_TORCH_TESTS") == "1"


def arguments(model="pose-wide", **values):
    return argparse.Namespace(model=model, **values)


class ConfigurationTests(unittest.TestCase):
    def test_defaults_are_disabled(self):
        settings = resolve_settings(arguments())
        for name in ("pose_joint_noise", "pose_joint_dropout", "pose_athlete_dropout"):
            self.assertEqual(settings[name], 0.0)

    def test_invalid_values_and_image_models_are_rejected(self):
        for values in ({"pose_joint_noise": -0.1}, {"pose_joint_dropout": 1.0},
                       {"pose_athlete_dropout": 1.5}):
            with self.assertRaises(ValueError):
                resolve_settings(arguments(**values))
        with self.assertRaisesRegex(ValueError, "only to pose models"):
            resolve_settings(arguments(model="image", pose_joint_noise=0.05))

    def test_resume_cannot_change_and_old_runs_default_to_zero(self):
        saved = resolve_settings(arguments(pose_joint_noise=0.05))
        with self.assertRaisesRegex(ValueError, "Cannot change"):
            resolve_settings(arguments(pose_joint_noise=0.1), saved)
        old = {key: value for key, value in saved.items() if not key.startswith("pose_")}
        self.assertEqual(resolve_settings(arguments(), old)["pose_joint_noise"], 0.0)


@unittest.skipUnless(RUN_TORCH, "Set JIUJITSU_RUN_TORCH_TESTS=1 for perturbation checks")
class PerturbationTests(unittest.TestCase):
    def setUp(self):
        import torch
        from src.Loader.pose import pose_features
        torch.manual_seed(0)
        self.record = {"image": "0100000", "frame": 0, "position": "mount1",
                       "pose1": [[10 + j * 3, 20 + j * 2, 0.6] for j in range(17)],
                       "pose2": [[90 + j * 2, 60 - j, 0.9] for j in range(17)]}
        self.features = torch.tensor(pose_features(self.record["pose1"], self.record["pose2"]),
                                     dtype=torch.float32)

    def test_input_is_preserved_and_confidence_is_unchanged(self):
        from src.Loader.datasets import perturb_pose
        original = self.features.clone()
        result = perturb_pose(self.features, joint_noise=0.05)
        self.assertTrue(original.equal(self.features))
        self.assertTrue(result.reshape(34, 3)[:, 2].equal(original.reshape(34, 3)[:, 2]))
        self.assertFalse(result.equal(original))

    def test_removed_athlete_matches_normalizing_the_remaining_pose(self):
        import torch
        from src.Loader.datasets import perturb_pose
        from src.Loader.pose import pose_features
        result = perturb_pose(self.features, athlete_dropout=0.999999)
        remaining = (pose_features(self.record["pose1"], None) if result[53] == 0
                     else pose_features(None, self.record["pose2"]))
        self.assertTrue(torch.allclose(result, torch.tensor(remaining, dtype=torch.float32), atol=1e-5))

    def test_either_athlete_can_be_removed(self):
        from src.Loader.datasets import perturb_pose
        removed = set()
        for _ in range(50):
            result = perturb_pose(self.features, athlete_dropout=0.999999).reshape(34, 3)
            removed.add("first" if result[0, 2] == 0 else "second")
        self.assertEqual(removed, {"first", "second"})

    def test_removing_every_joint_keeps_the_frame_observed(self):
        from src.Loader.datasets import perturb_pose
        result = perturb_pose(self.features, joint_dropout=0.999999)
        self.assertGreater(int((result.reshape(34, 3)[:, 2] > 0).sum()), 0)

    def test_evaluation_never_perturbs(self):
        import torch
        from src.Dataset.annotations import CLASS_NAMES_10, LABEL_MAP_10
        from src.Train.engine import make_dataset
        config = {"class_names": CLASS_NAMES_10, "label_map": LABEL_MAP_10, "swap_athletes": False,
                  "pose_joint_noise": 0.5, "pose_joint_dropout": 0.5, "pose_athlete_dropout": 0.5}
        evaluation = make_dataset("pose-wide", [self.record], config)
        self.assertTrue(evaluation[0][0].equal(self.features))
        training = make_dataset("pose-wide", [self.record], config, training=True)
        self.assertFalse(training[0][0].equal(self.features))


if __name__ == "__main__":
    unittest.main()
