"""Small standard-library fixtures for safe, useful training-metrics exports."""

import csv
import hashlib
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from src.Dataset.annotations import CLASS_NAMES_10
from src.Scripts import export_metrics


def config_bytes(model="pose", epochs=30, *, timestamp="2026-10-10T12:00:00+00:00", **settings):
    payload = {"config": {"model": model, "epochs": epochs, "lr": 0.001,
                          "annotation_sha256": "a" * 64, "split_sha256": "b" * 64,
                          **settings},
               "provenance": {"timestamp_utc": timestamp}, "resumed_from": None}
    return (json.dumps(payload, indent=2) + "\n").encode()


def metric(epoch, score=0.7, loss=1.0):
    correct = round(score * 10)
    matrix = [[correct if row == column else (10 - correct if column == (row + 1) % 10 else 0)
               for column in range(10)] for row in range(10)]
    return {"epoch": epoch, "train": {"accuracy": 0.9, "loss": 0.2, "samples": 100},
            "val": {"accuracy": score, "macro_f1": score, "loss": loss, "samples": 100,
                    "class_names": CLASS_NAMES_10,
                    "per_class": [{"class_name": name, "precision": score, "recall": score,
                                   "f1": score, "support": 10} for name in CLASS_NAMES_10],
                    "confusion_matrix": matrix,
                    "confusion_matrix_axes": {"rows": "true", "columns": "predicted"}},
            "seconds": 1.5, "best_macro_f1": score}


def metrics_bytes(*rows):
    return b"".join(json.dumps(row).encode() + b"\r\n" for row in rows)


class ExportMetricsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.trainings = self.root / "trainings"
        self.trainings.mkdir()
        self.output = self.root / "exports" / "metrics.zip"

    def write(self, relative, content):
        path = self.trainings / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def summary(self, manifest, name):
        return next(run["summary"] for run in manifest["runs"] if run["name"] == name)

    def test_multiple_runs_preserve_raw_files_hashes_and_only_allowlisted_results(self):
        first = "20261010-pose-wide"
        second = "20261010-pose-attention"
        expected = {
            f"{first}/config_start.json": config_bytes("pose-wide", weight_decay=0.001),
            f"{first}/diagnostics/metrics.jsonl": metrics_bytes(
                metric(1, 0.6, 1.2), metric(2, 0.8, 0.9), metric(3, 0.8, 1.1)),
            f"{second}/config_start.json": config_bytes("pose-attention", epochs=100,
                attention_dropout=0.1, attention_mlp_dropout=0.15),
            f"{second}/diagnostics/metrics.jsonl": metrics_bytes(metric(1)),
            f"{second}/diagnostics/test_metrics.json": json.dumps({
                "split": "test", "checkpoint_epoch": 1, "checkpoint": "/remote/checkpoints/best.pt",
                "model": "pose-attention", "metrics": metric(1)["val"]}).encode(),
        }
        for path, content in expected.items():
            self.write(path, content)
        for path in (f"{first}/checkpoints/best.pt", f"{first}/.env",
                     f"{first}/config_start.json.tmp", f"{first}/local_config.sh",
                     f"{first}/diagnostics/environment.json", f"{first}/diagnostics/custom_test.json",
                     f"{first}/images/frame.png", ".hidden-run/config_start.json"):
            self.write(path, b"excluded fixture")
        manifest = export_metrics.build_export(self.trainings, self.output)
        self.assertEqual({run["name"] for run in manifest["runs"]}, {first, second})
        with zipfile.ZipFile(self.output) as archive:
            self.assertEqual(set(archive.namelist()),
                             {f"runs/{path}" for path in expected} | {"manifest.json", "summary.csv"})
            self.assertEqual(json.loads(archive.read("manifest.json")), manifest)
            for relative, content in expected.items():
                member = f"runs/{relative}"
                self.assertEqual(archive.read(member), content)
                self.assertEqual(manifest["files"][member],
                                 {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
                self.assertEqual((self.trainings / relative).read_bytes(), content)
            summaries = {row["run"]: row for row in csv.DictReader(
                io.StringIO(archive.read("summary.csv").decode()))}
        summary = self.summary(manifest, first)
        self.assertEqual(summary["best_epoch"], 2)  # Equal F1 keeps the earlier best checkpoint.
        self.assertEqual(summary["best_val_loss"], 0.9)
        self.assertEqual(summary["last_epoch"], 3)
        self.assertEqual(summary["epochs_logged"], 3)
        self.assertEqual(summary["target_epochs"], 30)
        self.assertEqual(summaries[first]["weight_decay"], "0.001")
        self.assertEqual(summaries[second]["weight_decay"], "")
        self.assertIsNone(self.summary(manifest, second)["weight_decay"])

    def test_latest_resume_timestamp_wins_even_if_resumed_epoch_is_lower(self):
        run = "resumed"
        self.write(f"{run}/config_start.json", config_bytes(epochs=30))
        self.write(f"{run}/config_resume_epoch100_20261011T120000000000Z.json",
                   config_bytes(epochs=150, timestamp="2026-10-11T12:00:00+00:00"))
        # A later restoration can legitimately start from an earlier checkpoint.
        self.write(f"{run}/config_resume_epoch9_20261012T120000000000Z.json",
                   config_bytes(epochs=60, timestamp="2026-10-12T12:00:00+00:00"))
        self.write(f"{run}/diagnostics/metrics.jsonl", metrics_bytes(metric(1)))
        manifest = export_metrics.build_export(self.trainings, self.output)
        self.assertEqual(self.summary(manifest, run)["target_epochs"], 60)
        self.assertEqual(len(manifest["files"]), 4)

    def test_resume_config_order_falls_back_to_filename_time_then_numeric_epoch(self):
        match = export_metrics.RESUME_NAME.fullmatch
        older = match("config_resume_epoch100_20261011T120000000000Z.json")
        newer = match("config_resume_epoch9_20261012T120000000000Z.json")
        self.assertGreater(export_metrics.config_order({"provenance": None}, newer),
                           export_metrics.config_order({}, older))
        ninth = match("config_resume_epoch9_unknown.json")
        tenth = match("config_resume_epoch10_unknown.json")
        self.assertGreater(export_metrics.config_order({"provenance": []}, tenth),
                           export_metrics.config_order({}, ninth))
        export_metrics.config_order({}, None)  # A start configuration may lack provenance.

    def test_partial_malformed_and_nonfinite_lines_are_preserved_but_not_summarized(self):
        invalid = metric(3)
        invalid["val"]["loss"] = float("nan")
        content = (metrics_bytes(metric(1)) + b"not valid JSON\n"
                   + metrics_bytes(invalid, metric(4, 0.8)) + b'{"epoch": 5, "train":')
        path = self.write("running/diagnostics/metrics.jsonl", content)
        manifest = export_metrics.build_export(self.trainings, self.output)
        run = manifest["runs"][0]
        self.assertEqual(run["summary"]["epochs_logged"], 2)
        self.assertEqual(run["summary"]["last_epoch"], 4)
        self.assertEqual(run["summary"]["best_epoch"], 4)
        self.assertIsNone(run["summary"]["model"])
        warnings = "\n".join(run["warnings"])
        self.assertIn("Missing config_start.json", warnings)
        self.assertIn("Invalid metrics line 2", warnings)
        self.assertIn("Nonfinite JSON number", warnings)
        self.assertIn("Unterminated metrics line 5", warnings)
        self.assertIn("not consecutive", warnings)
        with zipfile.ZipFile(self.output) as archive:
            self.assertEqual(archive.read("runs/running/diagnostics/metrics.jsonl"), content)
        self.assertEqual(path.read_bytes(), content)

    def test_unterminated_valid_json_is_treated_as_an_unfinished_log_record(self):
        warnings = []
        content = metrics_bytes(metric(1)) + json.dumps(metric(2, 0.8)).encode()
        rows = export_metrics.metric_rows(content, warnings)
        self.assertEqual([row["epoch"] for row in rows], [1])
        self.assertIn("Unterminated metrics line 2", warnings[0])

    def test_nonfinite_config_is_copied_with_warning_without_poisoning_other_runs(self):
        bad = config_bytes(weight_decay=float("inf"))
        self.write("bad-config/config_start.json", bad)
        self.write("bad-config/diagnostics/metrics.jsonl", metrics_bytes(metric(1)))
        self.write("waiting/config_start.json", config_bytes("image", epochs=50))
        manifest = export_metrics.build_export(self.trainings, self.output)
        bad_run = next(run for run in manifest["runs"] if run["name"] == "bad-config")
        self.assertIsNone(bad_run["summary"]["model"])
        self.assertTrue(any("Invalid config_start.json" in warning for warning in bad_run["warnings"]))
        waiting = next(run for run in manifest["runs"] if run["name"] == "waiting")
        self.assertEqual(waiting["summary"]["epochs_logged"], 0)
        self.assertEqual(waiting["summary"]["target_epochs"], 50)
        self.assertTrue(any("Missing diagnostics/metrics.jsonl" in warning for warning in waiting["warnings"]))
        with zipfile.ZipFile(self.output) as archive:
            self.assertEqual(archive.read("runs/bad-config/config_start.json"), bad)

    def test_symlink_files_runs_and_diagnostics_directories_are_skipped(self):
        self.write("safe/config_start.json", config_bytes())
        self.write("safe/diagnostics/metrics.jsonl", metrics_bytes(metric(1)))
        linked_file = self.write("file-link/config_start.json", b"external config bytes")
        self.write("file-link/diagnostics/metrics.jsonl", metrics_bytes(metric(1)))
        self.write("dir-link/config_start.json", config_bytes())
        linked_directory = self.write("dir-link/diagnostics/metrics.jsonl", b"external log bytes").parent
        linked_run = self.write("run-link/config_start.json", b"external run bytes").parent
        blocked = {path.resolve() for path in (linked_file, linked_directory, linked_run)}
        original = Path.is_symlink

        # Simulate filesystem link markers so this regression test does not
        # require Windows' symlink-creation privilege.
        def is_link(path):
            return path.resolve() in blocked or original(path)

        with patch.object(Path, "is_symlink", is_link):
            manifest = export_metrics.build_export(self.trainings, self.output)
        members = set(manifest["files"])
        self.assertNotIn("runs/file-link/config_start.json", members)
        self.assertNotIn("runs/dir-link/diagnostics/metrics.jsonl", members)
        self.assertFalse(any(member.startswith("runs/run-link/") for member in members))
        self.assertIn("runs/safe/diagnostics/metrics.jsonl", members)
        self.assertTrue(any("run-link" in warning for warning in manifest["warnings"]))

    def test_snapshot_caps_bytes_at_open_size_and_reports_changes(self):
        path = self.write("fixture/diagnostics/metrics.jsonl", b"old\nnew\n")
        self.assertEqual(export_metrics.snapshot(path), (b"old\nnew\n", False))
        before = SimpleNamespace(st_size=4, st_mtime_ns=1)
        after = SimpleNamespace(st_size=8, st_mtime_ns=2)
        with patch.object(export_metrics.os, "fstat", side_effect=[before, after]):
            self.assertEqual(export_metrics.snapshot(path), (b"old\n", True))

    def test_changed_or_unreadable_files_warn_without_discarding_readable_metrics(self):
        self.write("running/config_start.json", config_bytes())
        raw = metrics_bytes(metric(1))
        self.write("running/diagnostics/metrics.jsonl", raw)
        original = export_metrics.snapshot

        def read_file(path):
            if path.name == "config_start.json":
                raise PermissionError("fixture read failure")
            content, _ = original(path)
            return content, True

        with patch.object(export_metrics, "snapshot", side_effect=read_file):
            manifest = export_metrics.build_export(self.trainings, self.output)
        warnings = "\n".join(manifest["runs"][0]["warnings"])
        self.assertIn("Could not read config_start.json", warnings)
        self.assertIn("File changed while reading: diagnostics/metrics.jsonl", warnings)
        self.assertEqual(self.summary(manifest, "running")["epochs_logged"], 1)
        with zipfile.ZipFile(self.output) as archive:
            self.assertEqual(archive.read("runs/running/diagnostics/metrics.jsonl"), raw)

    def test_missing_or_empty_training_directories_do_not_create_an_export(self):
        with self.assertRaises(FileNotFoundError):
            export_metrics.build_export(self.root / "missing", self.output)
        with self.assertRaisesRegex(ValueError, "No run configuration or metrics"):
            export_metrics.build_export(self.trainings, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob("*.zip.tmp")), [])

    def test_existing_destination_is_preserved_without_reading_run_files(self):
        self.output.parent.mkdir()
        self.output.write_bytes(b"existing export")
        with patch.object(export_metrics, "snapshot", side_effect=AssertionError("must not read")):
            with self.assertRaises(FileExistsError):
                export_metrics.build_export(self.trainings, self.output)
        self.assertEqual(self.output.read_bytes(), b"existing export")

    def test_publish_failure_cleans_temporary_zip_and_never_overwrites_a_racing_export(self):
        self.write("pose/config_start.json", config_bytes())

        def another_export_wins(temporary, output):
            Path(output).write_bytes(b"other export")
            raise FileExistsError("fixture publication race")

        with patch.object(export_metrics.os, "link", side_effect=another_export_wins):
            with self.assertRaises(FileExistsError):
                export_metrics.build_export(self.trainings, self.output)
        self.assertEqual(self.output.read_bytes(), b"other export")
        self.assertEqual(list(self.output.parent.glob("*.zip.tmp")), [])

    def test_zip_write_failure_removes_temporary_output(self):
        source = self.write("pose/config_start.json", config_bytes())
        before = source.read_bytes()
        with patch.object(zipfile.ZipFile, "writestr", side_effect=OSError("fixture ZIP failure")):
            with self.assertRaisesRegex(OSError, "fixture ZIP failure"):
                export_metrics.build_export(self.trainings, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob("*.zip.tmp")), [])
        self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
