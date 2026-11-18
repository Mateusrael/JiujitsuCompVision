"""Training pose models on poses from a detector's cache."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from src.Loader.pose import with_detected_poses
from src.Train.config import resolve_settings

POSE_A = [[1.0, 2.0, 0.9]] * 17
POSE_B = [[3.0, 4.0, 0.8]] * 17
ANNOTATED = [[5.0, 6.0, 1.0]] * 17


def record(image):
    return {"image": image, "position": "mount", "pose1": ANNOTATED, "pose2": ANNOTATED}


class DetectedPoseTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "cache.jsonl"
        rows = [{"image": "two", "poses": [POSE_A, POSE_B]}, {"image": "one", "poses": [POSE_A]},
                {"image": "none", "poses": []}, {"image": "other", "poses": [POSE_B]}]
        self.path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_replaces_poses_and_drops_frames_without_people(self):
        original = [record("two"), record("one"), record("none")]
        result = with_detected_poses(original, self.path)
        self.assertEqual([row["image"] for row in result], ["two", "one"])
        self.assertEqual((result[0]["pose1"], result[0]["pose2"]), (POSE_A, POSE_B))
        self.assertEqual((result[1]["pose1"], result[1]["pose2"]), (POSE_A, None))
        self.assertEqual(original[0]["pose1"], ANNOTATED)

    def test_keep_annotated_adds_detected_copies(self):
        result = with_detected_poses([record("two"), record("none")], self.path, keep_annotated=True)
        self.assertEqual([row["image"] for row in result], ["two", "none", "two"])
        self.assertEqual((result[0]["pose1"], result[2]["pose1"]), (ANNOTATED, POSE_A))

    def test_uncached_frame_is_an_error(self):
        with self.assertRaises(ValueError):
            with_detected_poses([record("absent")], self.path)


class ConfigurationTests(unittest.TestCase):
    def test_settings(self):
        pose = lambda **values: argparse.Namespace(model="pose-wide", **values)
        self.assertIsNone(resolve_settings(pose())["pose_detections"])
        self.assertEqual(resolve_settings(pose(pose_detections="c.jsonl"))["pose_detections"], "c.jsonl")
        with self.assertRaises(ValueError):
            resolve_settings(pose(pose_keep_annotated=True))
        with self.assertRaises(ValueError):
            resolve_settings(argparse.Namespace(model="image", pose_detections="c.jsonl"))


if __name__ == "__main__":
    unittest.main()
