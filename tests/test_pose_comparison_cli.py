"""Original-split comparison CLI with a real pose checkpoint and stub detections."""

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.Dataset.annotations import CLASS_NAMES_10, LABEL_MAP_10, sha256_file


@unittest.skipUnless(os.environ.get("JIUJITSU_RUN_TORCH_TESTS") == "1", "opt-in CPU inference")
class PoseComparisonIntegrationTests(unittest.TestCase):
    def test_real_checkpoint_and_shared_comparator_write_traceable_report(self):
        import torch
        from experiments.compare_pose_detectors import main
        from src.Modules.models import build_model
        from src.Train.config import resolve_settings
        from src.Train.engine import configured_architecture
        from tests.test_training import synthetic_files

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations, split, images, _ = synthetic_files(root, images=True)
            config = resolve_settings(argparse.Namespace(model="pose"))
            config.update(model="pose", annotations=str(annotations), split=str(split),
                          images_dir=str(images), annotation_sha256=sha256_file(annotations),
                          split_sha256=sha256_file(split), class_names=CLASS_NAMES_10,
                          label_map=LABEL_MAP_10, pose_filter="both_present")
            config["model_config"] = configured_architecture("pose", 10, config)
            model = build_model(config["model_config"], pretrained=False)
            checkpoint = root / "best.pt"
            torch.save({"schema_version": 1, "config": config, "model_state": model.state_dict(),
                        "optimizer_state": {}, "epoch": 7, "best_macro_f1": 0,
                        "random_state": {}}, checkpoint)

            def detect(name, records, image_path, cache_dir, device, progress=True):
                return {r["image"]: {"image": r["image"], "poses": [r["pose1"], r["pose2"]],
                                      "seconds": 0.01, "image_sha256": sha256_file(image_path(r))}
                        for r in records}

            with (patch("src.Eval.pose_detection.detect_records", side_effect=detect),
                  patch("src.Train.engine.evaluation_types_for_records",
                        return_value=["held_out_view", "unseen_moment"] * 2)):
                main(["--checkpoint", str(checkpoint), "--device", "cpu", "--no-progress",
                      "--limit", "4", "--output-dir", str(root / "reports"),
                      "--cache-dir", str(root / "cache")])
            reports = list((root / "reports").glob("*.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            self.assertEqual(report["checkpoint_sha256"], sha256_file(checkpoint))
            self.assertEqual(report["split_sha256"], sha256_file(split))
            self.assertEqual(report["epoch"], 7)
            result = report["comparison"]
            self.assertEqual(result["baseline"]["metrics"]["samples"], 4)
            self.assertEqual(len(result["common_two_detected"]["image_ids"]), 4)
            for detector in result["detectors"].values():
                self.assertEqual(detector["all_frames"], result["baseline"])
                self.assertEqual(detector["localization"]["mean_normalized_joint_error"], 0)
            for kind in ("held_out_view", "unseen_moment"):
                metrics = result["by_evaluation_type"][kind]
                self.assertEqual(metrics["annotated"]["samples"], 2)
                for detector_metrics in metrics["detectors"].values():
                    self.assertEqual(detector_metrics, metrics["annotated"])


if __name__ == "__main__":
    unittest.main()
