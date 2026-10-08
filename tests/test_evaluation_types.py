"""One inference pass must account for both evaluation populations correctly."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("JIUJITSU_RUN_TORCH_TESTS") == "1",
                     "Set JIUJITSU_RUN_TORCH_TESTS=1 for CPU grouped evaluation checks")
class EvaluationTypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def fixtures(self):
        torch = self.torch

        class CountedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, features):
                self.calls += 1
                return features

        logits = torch.tensor([[4., 0.], [0., 4.], [3., 0.], [0., 2.], [0., 1.]])
        targets = torch.tensor([0, 0, 1, 1, 1])
        dataset = torch.utils.data.TensorDataset(logits, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
        types = ["held_out_view", "unseen_moment", "held_out_view",
                 "unseen_moment", "unseen_moment"]
        return CountedModel(), loader, types

    def test_group_scores_account_for_pooled_samples_without_extra_inference(self):
        from src.Train.engine import score_model
        model, loader, types = self.fixtures()
        scores = score_model(model, loader, "cpu", ["a", "b"], progress=False,
                             evaluation_types=types)
        self.assertEqual(model.calls, len(loader))
        groups = scores["by_evaluation_type"]
        self.assertEqual(groups["held_out_view"]["confusion_matrix"], [[1, 0], [1, 0]])
        self.assertEqual(groups["unseen_moment"]["confusion_matrix"], [[0, 1], [0, 2]])
        self.assertEqual(scores["samples"], sum(group["samples"] for group in groups.values()))
        self.assertAlmostEqual(scores["loss"], sum(group["loss"] * group["samples"]
                                                  for group in groups.values()) / 5, places=6)
        for actual in range(2):
            for predicted in range(2):
                self.assertEqual(scores["confusion_matrix"][actual][predicted],
                                 sum(group["confusion_matrix"][actual][predicted]
                                     for group in groups.values()))
        # Old manifests receive the same pooled scores, without added populations.
        plain = score_model(model, loader, "cpu", ["a", "b"], progress=False)
        self.assertNotIn("by_evaluation_type", plain)
        self.assertAlmostEqual(plain.pop("loss"), scores["loss"], places=6)
        self.assertEqual(plain, {key: value for key, value in scores.items()
                                 if key not in ("loss", "by_evaluation_type")})

    def test_incomplete_types_unknown_types_or_unordered_samples_fail_before_inference(self):
        from src.Train.engine import score_model
        model, loader, types = self.fixtures()
        for invalid in (types[:-1], types + ["unseen_moment"], types[:-1] + ["unknown"]):
            with self.subTest(types=invalid), self.assertRaises(ValueError):
                score_model(model, loader, "cpu", ["a", "b"], progress=False,
                            evaluation_types=invalid)
        for options in ({"shuffle": True}, {"drop_last": True}):
            invalid_loader = self.torch.utils.data.DataLoader(loader.dataset, batch_size=2, **options)
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "sequential loader"):
                score_model(model, invalid_loader, "cpu", ["a", "b"], progress=False,
                            evaluation_types=types)
        self.assertEqual(model.calls, 0)

    def test_missing_image_type_fails_and_filtering_preserves_remaining_type_order(self):
        from src.Train.engine import evaluation_types_for_records, select_model_records
        observed = [[0, 0, 0.5]] * 17
        records = [{"image": "a", "pose1": observed, "pose2": observed},
                   {"image": "b", "pose1": observed, "pose2": None},
                   {"image": "c", "pose1": observed, "pose2": observed}]
        manifest = {"assignments": {row["image"]: "val" for row in records},
                    "evaluation_types": {"a": "unseen_moment", "b": "held_out_view",
                                         "c": "held_out_view"}}
        eligible, counts = select_model_records("pose", records, manifest, "val")
        self.assertEqual([row["image"] for row in eligible], ["a", "c"])
        self.assertEqual(evaluation_types_for_records(eligible, manifest),
                         ["unseen_moment", "held_out_view"])
        self.assertEqual(counts["eligible"], 2)
        self.assertEqual(counts["by_evaluation_type"]["held_out_view"],
                         {"original": 2, "eligible": 1, "removed": 1})
        image_records, image_counts = select_model_records("image", records, manifest, "val")
        self.assertEqual(image_records, records)
        self.assertEqual(image_counts["removed"], 0)
        del manifest["evaluation_types"]["c"]
        with self.assertRaisesRegex(ValueError, "Missing or unsupported"):
            evaluation_types_for_records(eligible, manifest)
        self.assertIsNone(evaluation_types_for_records(records, {"assignments": {}}))

    def test_filtered_training_and_test_log_the_same_population_policy(self):
        from src.Dataset.annotations import sha256_file
        from src.Dataset.splits import build_manifest
        from src.Eval.evaluate import evaluate
        from src.Scripts.evaluate import build_parser as evaluation_parser
        from src.Scripts.train import build_parser
        from src.Train.checkpoints import load_checkpoint
        from src.Train.engine import train, validate_checkpoint_data
        from tests.test_training import synthetic_files

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations, split, _, records = synthetic_files(root)
            manifest = json.loads(split.read_text())
            for prefix in ("01", "02", "03"):
                extra = dict(next(row for row in records if row["image"].startswith(prefix)))
                extra.update(image=prefix + "00010", frame=10, pose2=None)
                records.append(extra)
            annotations.write_text(json.dumps(records))
            manifest = build_manifest(records, manifest["recording_groups"], sha256_file(annotations))
            manifest["evaluation_types"] = {
                row["image"]: ("held_out_view" if row["frame"] % 2 else "unseen_moment")
                for row in records if manifest["assignments"][row["image"]] in ("val", "test")}
            split.write_text(json.dumps(manifest))
            run = root / "run"
            # Manifest construction/verification is covered by split tests; this
            # fixture isolates how train and evaluate consume verified metadata.
            with patch("src.Train.engine.read_manifest", return_value=manifest):
                train(build_parser().parse_args([
                    "--model", "pose", "--annotations", str(annotations), "--split", str(split),
                    "--run-dir", str(run), "--epochs", "1", "--device", "cpu",
                    "--workers", "0", "--batch-size", "4", "--no-progress"]))
            best = run / "checkpoints" / "best.pt"
            checkpoint = load_checkpoint(best)
            config = checkpoint["config"]
            self.assertEqual(config["pose_filter"], "both_present")
            self.assertEqual(config["checkpoint_selection"], "pooled_validation_macro_f1")
            for part in ("train", "val", "test"):
                self.assertEqual(config["sample_counts"][part]["original"], 11)
                self.assertEqual(config["sample_counts"][part]["eligible"], 10)
                self.assertEqual(config["sample_counts"][part]["removed"], 1)
            row = json.loads((run / "diagnostics" / "metrics.jsonl").read_text())
            self.assertEqual(row["train"]["samples"], 10)
            self.assertEqual(row["val"]["samples"], 10)
            self.assertEqual(set(row["val"]["by_evaluation_type"]), {"held_out_view", "unseen_moment"})
            with patch("src.Eval.evaluate.read_manifest", return_value=manifest):
                result = evaluate(evaluation_parser().parse_args([
                    "--checkpoint", str(best), "--device", "cpu", "--no-progress"]))
            self.assertEqual(result["pose_filter"], "both_present")
            self.assertEqual(result["sample_counts"], config["sample_counts"]["test"])
            self.assertEqual(result["metrics"]["samples"], 10)
            self.assertEqual(set(result["metrics"]["by_evaluation_type"]),
                             {"held_out_view", "unseen_moment"})
            del config["pose_filter"]
            with self.assertRaisesRegex(ValueError, "fresh training"):
                validate_checkpoint_data(config, "pose", annotations, split, manifest)

    def test_complete_pose_without_training_view_is_excluded_by_pose_assignments(self):
        from src.Train.engine import evaluation_types_for_records, select_model_records
        observed = [[1, 2, 0.8]] * 17
        records = [{"image": name, "pose1": observed, "pose2": observed}
                   for name in ("paired", "unpaired", "unseen")]
        manifest = {
            "assignments": {record["image"]: "val" for record in records},
            "pose_assignments": {"paired": "val", "unpaired": "excluded", "unseen": "val"},
            "pose_exclusion_reasons": {"unpaired": "no_paired_training_view"},
            "evaluation_types": {"paired": "held_out_view", "unpaired": "held_out_view",
                                 "unseen": "unseen_moment"}}
        selected, counts = select_model_records("pose-attention", records, manifest, "val")
        self.assertEqual([record["image"] for record in selected], ["paired", "unseen"])
        self.assertEqual(evaluation_types_for_records(selected, manifest),
                         ["held_out_view", "unseen_moment"])
        self.assertEqual(counts["by_evaluation_type"]["held_out_view"],
                         {"original": 2, "eligible": 1, "removed": 1})
        self.assertEqual(counts["removed_by_reason"], {"no_paired_training_view": 1})
        image_selected, counts = select_model_records("image", records, manifest, "val")
        self.assertEqual(image_selected, records)
        self.assertEqual(counts["removed"], 0)

    def test_generated_multiview_manifest_trains_evaluates_and_exports_both_populations(self):
        from src.Dataset.annotations import sha256_file
        from src.Dataset.multiview import build_multiview_manifest
        from src.Eval.evaluate import evaluate
        from src.Scripts.evaluate import build_parser as evaluation_parser
        from src.Scripts.export_metrics import build_export
        from src.Scripts.train import build_parser
        from src.Train.checkpoints import load_checkpoint
        from src.Train.engine import train
        from tests.test_multiview import fixture

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, plan = fixture()
            annotations = root / "annotations.json"
            annotations.write_text(json.dumps(records), encoding="utf-8")
            manifest = build_multiview_manifest(records, sha256_file(annotations), section_plan=plan)
            self.assertEqual(manifest["schema_version"], 3)
            split = root / "multiview.json"
            split.write_text(json.dumps(manifest), encoding="utf-8")
            run = root / "trainings" / "pose"
            # Use the actual manifest verifier, model, optimizer and evaluator.
            # The shared fixture includes missing poses and unpaired views, so
            # agreement here also checks model-specific population integration.
            train(build_parser().parse_args([
                "--model", "pose", "--annotations", str(annotations), "--split", str(split),
                "--run-dir", str(run), "--epochs", "2", "--device", "cpu",
                "--workers", "0", "--batch-size", "256", "--no-progress"]))
            rows = [json.loads(line) for line in
                    (run / "diagnostics" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["epoch"] for row in rows], [1, 2])
            best_row = max(rows, key=lambda row: row["val"]["macro_f1"])
            best_path = run / "checkpoints" / "best.pt"
            checkpoint = load_checkpoint(best_path)
            self.assertEqual(checkpoint["epoch"], best_row["epoch"])
            self.assertEqual(checkpoint["best_macro_f1"], best_row["val"]["macro_f1"])
            self.assertEqual(checkpoint["config"]["split_method"], "multiview-sections")
            self.assertEqual(checkpoint["config"]["checkpoint_selection"],
                             "pooled_validation_macro_f1")
            for partition in ("train", "val", "test"):
                counts = checkpoint["config"]["sample_counts"][partition]
                self.assertEqual(counts["original"], manifest["counts"][partition])
                self.assertEqual(counts["eligible"], manifest["pose_counts"][partition])
                self.assertEqual(counts["removed"], counts["original"] - counts["eligible"])
                self.assertEqual(sum(counts["removed_by_reason"].values()), counts["removed"])
            for row in rows:
                self.assertEqual(row["train"]["samples"], manifest["pose_counts"]["train"])
                self.assertEqual(row["val"]["samples"], manifest["pose_counts"]["val"])
                for kind, metrics in row["val"]["by_evaluation_type"].items():
                    self.assertEqual(metrics["samples"], manifest["pose_evaluation_counts"]["val"][kind])
                    self.assertTrue(all(value["support"] > 0 for value in metrics["per_class"]))
            result = evaluate(evaluation_parser().parse_args([
                "--checkpoint", str(best_path), "--device", "cpu", "--no-progress"]))
            self.assertEqual(result["checkpoint_epoch"], best_row["epoch"])
            self.assertEqual(result["sample_counts"], checkpoint["config"]["sample_counts"]["test"])
            self.assertEqual(result["metrics"]["samples"], manifest["pose_counts"]["test"])
            for kind, metrics in result["metrics"]["by_evaluation_type"].items():
                self.assertEqual(metrics["samples"], manifest["pose_evaluation_counts"]["test"][kind])
            exported = build_export(root / "trainings", root / "results.zip")
            summary = exported["runs"][0]["summary"]
            self.assertEqual(summary["best_epoch"], best_row["epoch"])
            self.assertEqual(summary["best_val_macro_f1"], best_row["val"]["macro_f1"])
            self.assertEqual(exported["runs"][0]["warnings"], [])
            self.assertIn("runs/pose/diagnostics/test_metrics.json", exported["files"])
            for kind in ("held_out_view", "unseen_moment"):
                self.assertEqual(summary[f"best_val_{kind}_macro_f1"],
                                 best_row["val"]["by_evaluation_type"][kind]["macro_f1"])
                self.assertEqual(summary[f"last_val_{kind}_samples"],
                                 manifest["pose_evaluation_counts"]["val"][kind])


if __name__ == "__main__":
    unittest.main()
