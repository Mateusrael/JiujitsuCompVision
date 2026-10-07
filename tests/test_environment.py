"""Probe failure-path tests; requires no PyTorch, GPU, internet, or installation."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "check_environment", Path(__file__).resolve().parents[1] / "install" / "check_environment.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class EnvironmentTests(unittest.TestCase):
    def test_missing_command_and_timeout_are_reported(self):
        for failure in (FileNotFoundError("missing"), subprocess.TimeoutExpired(["probe"], 1)):
            with self.subTest(failure=type(failure).__name__), patch.object(probe.subprocess, "run", side_effect=failure):
                self.assertFalse(probe.run_command(["probe"], 1)["ok"])

    def test_import_crash_is_contained(self):
        failed = {"ok": False, "returncode": -11, "stderr": "torch import failed"}
        with patch.object(probe, "run_command", return_value=failed):
            self.assertEqual(probe.probe_torch(1), failed)

    def test_import_failure_and_extra_stdout_are_parsed(self):
        detail = {"ok": False, "imports": {"torch": {"ok": False, "error": "No module named torch"}}}
        result = {"ok": True, "stdout": "import message\nVISAOCOMP_JSON=" + json.dumps(detail), "stderr": ""}
        with patch.object(probe, "run_command", return_value=result):
            self.assertEqual(probe.probe_torch(1), detail)

    def test_child_reports_broken_imports_without_torch_installed(self):
        with tempfile.TemporaryDirectory() as folder:
            for module in ("torch", "torchvision"):
                (Path(folder) / (module + ".py")).write_text("raise RuntimeError('simulated broken import')", encoding="utf-8")
            code = "import sys; sys.path.insert(0, sys.argv[2]);\n" + probe.TORCH_CODE
            result = probe.run_command([sys.executable, "-B", "-c", code, "", folder], 10)
            self.assertTrue(result["ok"], result)
            report = json.loads(result["stdout"].split("VISAOCOMP_JSON=")[1])
            self.assertFalse(report["ok"])
            self.assertFalse(report["cuda"]["available"])
            for module in ("torch", "torchvision"):
                self.assertIn("simulated broken import", report["imports"][module]["error"])

    def test_nonexistent_disk_path_uses_existing_ancestor(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "not-created" / "Data"
            result = probe.disk_report(target)
            self.assertFalse(result["exists"])
            self.assertEqual(Path(result["measured_path"]), Path(folder).resolve())
            self.assertGreaterEqual(result["free"], 0)
            self.assertIn("quota", result["note"])
            self.assertFalse(target.exists())

    def test_require_cuda_fails_and_saves_report(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "reports" / "environment.json"
            with patch.object(probe, "run_command", return_value={"ok": False, "error": "missing"}), \
                    patch.object(probe, "probe_torch", return_value={"ok": True, "cuda": {"available": False}}), \
                    patch.object(probe, "network_report") as network, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(probe.main(["--require-cuda", "--output", str(output)]), 1)
                network.assert_not_called()
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["ok"])
            self.assertIn("CUDA", report["errors"][0])

    def test_missing_torch_is_descriptive_unless_required(self):
        with patch.object(probe, "run_command", return_value={"ok": False}), \
                patch.object(probe, "probe_torch", return_value={"ok": False, "error": "missing torch"}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(probe.main([]), 0)
            self.assertEqual(probe.main(["--require-torch"]), 1)
            self.assertEqual(probe.main(["--smoke-test", "cpu"]), 1)

    def test_failed_atomic_replace_preserves_previous_report(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "environment.json"
            output.write_text("previous", encoding="utf-8")
            with patch.object(probe.os, "replace", side_effect=OSError("denied")):
                with self.assertRaises(OSError):
                    probe.write_report(output, "replacement")
            self.assertEqual(output.read_text(encoding="utf-8"), "previous")
            self.assertEqual(list(Path(folder).iterdir()), [output])


if __name__ == "__main__":
    unittest.main()
