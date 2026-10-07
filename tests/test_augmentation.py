"""Horizontal pose/image mirroring preserves labels and evaluation inputs."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.Dataset.annotations import CLASS_NAMES_10, LABEL_MAP_10
from src.Loader.pose import horizontal_flip_pose, pose_features


COCO_MIRROR_ORDER = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)


def example_record():
    return {"image": "0100000", "frame": 0,
            "position": next(raw for raw, mapped in LABEL_MAP_10.items() if mapped == "mount"),
            "pose1": [[10 + joint * 3, 20 + joint * 2, 0.6] for joint in range(17)],
            "pose2": [[90 + joint * 2, 60 - joint, 0.9] for joint in range(17)]}


class PoseMirroringTests(unittest.TestCase):
    def test_mirror_exchanges_left_right_joints_inside_each_athlete(self):
        joints = [[index + 1.0, index + 100.0, (index + 1) / 40] for index in range(34)]
        features = [value for joint in joints for value in joint]
        original = list(features)
        mirrored = horizontal_flip_pose(features)
        for athlete in range(2):
            for output_joint, source_joint in enumerate(COCO_MIRROR_ORDER):
                source = joints[athlete * 17 + source_joint]
                start = (athlete * 17 + output_joint) * 3
                self.assertEqual(mirrored[start:start + 3], [-source[0], source[1], source[2]])
        self.assertEqual(features, original)
        self.assertEqual(horizontal_flip_pose(mirrored), features)
        self.assertEqual(horizontal_flip_pose(features[51:] + features[:51]),
                         mirrored[51:] + mirrored[:51])

    def test_normalized_mirror_matches_mirroring_raw_pixel_coordinates(self):
        record = example_record()
        # Missing joints must remain missing, even though reflecting their raw
        # coordinates would produce nonzero values.
        record["pose1"][5] = [0, 0, 0]
        original = pose_features(record["pose1"], record["pose2"])
        raw_mirrors = []
        for pose in (record["pose1"], record["pose2"]):
            raw_mirrors.append([[639 - pose[index][0], pose[index][1], pose[index][2]]
                                for index in COCO_MIRROR_ORDER])
        expected = pose_features(*raw_mirrors)
        mirrored = horizontal_flip_pose(original)
        for actual, target in zip(mirrored, expected):
            self.assertAlmostEqual(actual, target, places=12)
        self.assertEqual(mirrored[6 * 3:6 * 3 + 3], [0.0, 0.0, 0.0])

    def test_missing_athletes_and_all_missing_frames_remain_zero(self):
        record = example_record()
        features = horizontal_flip_pose(pose_features(record["pose1"], None))
        self.assertEqual(features[51:], [0.0] * 51)
        self.assertEqual(horizontal_flip_pose(pose_features(None, None)), [0.0] * 102)


@unittest.skipUnless(os.environ.get("JIUJITSU_RUN_TORCH_TESTS") == "1",
                     "Set JIUJITSU_RUN_TORCH_TESTS=1 for CPU augmentation checks")
class DatasetMirroringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def test_pose_probability_endpoints_preserve_labels_and_cached_features(self):
        from src.Loader.datasets import PoseDataset
        torch = self.torch
        record = example_record()
        for probability in (0.0, 1.0):
            with self.subTest(probability=probability):
                dataset = PoseDataset([record], CLASS_NAMES_10, LABEL_MAP_10,
                                      horizontal_flip_prob=probability)
                cached = dataset.features.clone()
                expected = cached[0] if probability == 0 else torch.tensor(
                    horizontal_flip_pose(cached[0].tolist()), dtype=torch.float32)
                for _ in range(3):
                    features, label = dataset[0]
                    torch.testing.assert_close(features, expected, rtol=0, atol=0)
                    self.assertEqual(label.item(), CLASS_NAMES_10.index("mount"))
                torch.testing.assert_close(dataset.features, cached, rtol=0, atol=0)
        disabled = PoseDataset([record], CLASS_NAMES_10, LABEL_MAP_10)
        state = torch.get_rng_state()
        disabled[0]
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_pose_swap_and_mirror_use_independent_random_draws(self):
        from src.Loader.datasets import PoseDataset
        torch = self.torch
        dataset = PoseDataset([example_record()], CLASS_NAMES_10, LABEL_MAP_10,
                              swap_athletes=True, horizontal_flip_prob=0.5)
        cached = dataset.features[0].clone()
        mirror = torch.tensor(horizontal_flip_pose(cached.tolist()))
        only_swap = torch.cat((cached[51:], cached[:51]))
        outcomes = []
        for draws in ((0.1, 0.9), (0.9, 0.1)):
            with patch("src.Loader.datasets.torch.rand",
                       side_effect=[torch.tensor(value) for value in draws]) as random_draw:
                features, _ = dataset[0]
            self.assertEqual(random_draw.call_count, 2)
            outcomes.append(features)
        self.assertTrue(any(torch.equal(outcome, mirror) for outcome in outcomes))
        self.assertTrue(any(torch.equal(outcome, only_swap) for outcome in outcomes))

    def test_training_only_mirroring_for_all_pose_models(self):
        from src.Train.engine import make_dataset
        torch = self.torch
        records = [example_record()]
        config = {"class_names": CLASS_NAMES_10, "label_map": LABEL_MAP_10,
                  "swap_athletes": False, "horizontal_flip_prob": 1.0}
        original = torch.tensor(pose_features(records[0]["pose1"], records[0]["pose2"]))
        mirrored = torch.tensor(horizontal_flip_pose(original.tolist()))
        for name in ("pose", "pose-wide", "pose-attention"):
            with self.subTest(model=name):
                training = make_dataset(name, records, config, training=True)
                evaluation = make_dataset(name, records, config, training=False)
                torch.testing.assert_close(training[0][0], mirrored, rtol=0, atol=0)
                torch.testing.assert_close(evaluation[0][0], original, rtol=0, atol=0)

    def test_image_mirror_precedes_letterbox_without_vertical_flip_or_eval_changes(self):
        from PIL import Image, ImageOps
        from src.Loader.datasets import ImageDataset, letterbox
        from src.Train.engine import make_dataset
        torch = self.torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # A portrait image produces asymmetric horizontal padding at 224px.
            image = Image.new("RGB", (2, 3))
            image.putdata([(255, 0, 0), (0, 255, 0), (0, 0, 255),
                           (255, 255, 0), (255, 0, 255), (0, 255, 255)])
            path = root / "0100000.png"
            image.save(path)
            before = path.read_bytes()
            record = example_record()
            plain = ImageDataset([record], CLASS_NAMES_10, LABEL_MAP_10, root,
                                 horizontal_flip_prob=0.0)
            config = {"class_names": CLASS_NAMES_10, "label_map": LABEL_MAP_10,
                      "images_dir": str(root), "horizontal_flip_prob": 1.0}
            training = make_dataset("image", [record], config, training=True)
            evaluation = make_dataset("image", [record], config, training=False)
            expected = plain.normalize(plain.to_tensor(letterbox(ImageOps.mirror(image))).float() / 255)
            vertically_flipped = plain.normalize(
                plain.to_tensor(letterbox(ImageOps.flip(image))).float() / 255)
            actual, label = training[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertFalse(torch.equal(actual, vertically_flipped))
            self.assertFalse(torch.equal(actual, plain[0][0].flip(-1)))
            torch.testing.assert_close(evaluation[0][0], plain[0][0], rtol=0, atol=0)
            self.assertEqual(label, CLASS_NAMES_10.index("mount"))
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
