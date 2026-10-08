"""Whole-model compilation stays optional and preserves portable checkpoints."""

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, sentinel

from src.Modules.execution import compile_model, validate_compile_settings
from src.Scripts.evaluate import build_parser as evaluation_parser
from src.Scripts.train import build_parser as training_parser
from src.Train.config import resolve_settings


class ExecutionConfigurationTests(unittest.TestCase):
    def test_compile_defaults_off_and_progress_defaults_on(self):
        settings = resolve_settings(training_parser().parse_args(["--model", "pose"]))
        self.assertFalse(settings["compile"])
        self.assertEqual(settings["compile_mode"], "default")
        self.assertTrue(settings["progress"])
        evaluation = evaluation_parser().parse_args(["--checkpoint", "best.pt"])
        self.assertFalse(evaluation.compile)
        self.assertEqual(evaluation.compile_mode, "default")
        self.assertTrue(evaluation.progress)
        self.assertIs(compile_model(sentinel.model), sentinel.model)

    def test_execution_settings_can_change_when_resuming(self):
        parser = training_parser()
        previous = resolve_settings(parser.parse_args([
            "--model", "pose", "--compile", "--compile-mode", "reduce-overhead", "--no-progress"]))
        resumed = resolve_settings(parser.parse_args([
            "--model", "pose", "--no-compile", "--progress"]), previous)
        self.assertFalse(resumed["compile"])
        self.assertTrue(resumed["progress"])
        enabled = resolve_settings(parser.parse_args([
            "--model", "pose", "--compile", "--compile-mode", "max-autotune"]), resumed)
        self.assertTrue(enabled["compile"])
        self.assertEqual(enabled["compile_mode"], "max-autotune")

    def test_invalid_execution_options_fail_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, "requires --compile"):
            resolve_settings(training_parser().parse_args([
                "--model", "pose", "--compile-mode", "reduce-overhead"]))
        for enabled, mode in ((1, "default"), (False, "unsupported"), (True, None)):
            with self.subTest(enabled=enabled, mode=mode):
                with self.assertRaises(ValueError):
                    validate_compile_settings(enabled, mode)


@unittest.skipUnless(os.environ.get("JIUJITSU_RUN_TORCH_TESTS") == "1",
                     "Set JIUJITSU_RUN_TORCH_TESTS=1 for CPU execution checks")
class TorchExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def test_compile_receives_the_whole_model_and_is_never_called_when_disabled(self):
        model = self.torch.nn.Sequential(self.torch.nn.Linear(3, 4), self.torch.nn.ReLU(),
                                         self.torch.nn.Linear(4, 2))
        with patch("torch.compile", return_value=sentinel.compiled_model) as compiler:
            self.assertIs(compile_model(model), model)
            compiler.assert_not_called()
            result = compile_model(model, enabled=True, mode="reduce-overhead")
            self.assertIs(result, sentinel.compiled_model)
            compiler.assert_called_once_with(model, mode="reduce-overhead")

    def test_missing_or_failed_compiler_is_reported_without_silent_fallback(self):
        model = self.torch.nn.Linear(2, 2)
        with patch("torch.compile", None):
            with self.assertRaisesRegex(RuntimeError, "does not provide torch.compile"):
                compile_model(model, enabled=True)
        with patch("torch.compile", side_effect=RuntimeError("fixture compiler failure")):
            with self.assertRaisesRegex(RuntimeError, "could not initialize") as caught:
                compile_model(model, enabled=True)
        self.assertIn("fixture compiler failure", str(caught.exception.__cause__))

    def test_real_torch_compile_eager_backend_preserves_outputs_and_gradients(self):
        from src.Modules.models import architecture_config, build_model
        torch = self.torch
        original_compile = torch.compile

        def cpu_graph_capture(model, **options):
            # Exercise real Dynamo graph capture on CPU without requiring the
            # platform-specific Inductor compiler used for DGX acceleration.
            options.pop("mode", None)
            return original_compile(model, backend="eager", **options)

        for name in ("pose", "pose-wide", "pose-attention", "image"):
            with self.subTest(model=name):
                torch.manual_seed(27)
                config = architecture_config(name, attention_dim=32, attention_heads=2,
                                              attention_layers=1, attention_mlp_dim=64)
                eager = build_model(config).train()
                raw = copy.deepcopy(eager)
                with patch("torch.compile", side_effect=cpu_graph_capture) as compiler:
                    compiled = compile_model(raw, enabled=True)
                compiler.assert_called_once_with(raw, mode="default")
                if name == "image":
                    features = torch.randn(2, 3, 32, 32)
                else:
                    joints = torch.randn(2, 34, 3)
                    joints[..., 2] = 0.8
                    features = joints.reshape(2, 102)
                targets = torch.tensor([0, 3])
                eager_logits = eager(features)
                compiled_logits = compiled(features)
                torch.testing.assert_close(compiled_logits, eager_logits, atol=1e-6, rtol=1e-5)
                torch.nn.functional.cross_entropy(eager_logits, targets).backward()
                torch.nn.functional.cross_entropy(compiled_logits, targets).backward()
                for (key, parameter), (other_key, other) in zip(
                        eager.named_parameters(), raw.named_parameters()):
                    self.assertEqual(key, other_key)
                    if parameter.requires_grad:
                        self.assertIsNotNone(other.grad)
                        torch.testing.assert_close(other.grad, parameter.grad, atol=1e-6, rtol=1e-5)

    def test_compiled_training_saves_raw_weights_and_can_resume_or_evaluate_eagerly(self):
        from src.Eval.evaluate import evaluate
        from src.Modules.models import build_model
        from src.Train.checkpoints import load_checkpoint
        from src.Train.engine import train
        from tests.test_training import synthetic_files
        torch = self.torch
        wrappers = []

        class CompiledWrapper(torch.nn.Module):
            def __init__(self, raw):
                super().__init__()
                self._orig_mod = raw
                self.forward_calls = 0

            def forward(self, *args, **kwargs):
                self.forward_calls += 1
                return self._orig_mod(*args, **kwargs)

        def fake_compile(model, **options):
            wrapper = CompiledWrapper(model)
            wrappers.append(wrapper)
            return wrapper

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations, split, _, _ = synthetic_files(root)
            run = root / "compiled"
            options = ["--model", "pose", "--annotations", str(annotations),
                "--split", str(split), "--device", "cpu", "--workers", "0",
                "--batch-size", "10", "--epochs", "1", "--run-dir", str(run),
                "--compile", "--compile-mode", "reduce-overhead", "--no-progress"]
            with patch("torch.compile", side_effect=fake_compile) as compiler:
                train(training_parser().parse_args(options))
            self.assertEqual(compiler.call_count, 1)
            self.assertGreaterEqual(wrappers[0].forward_calls, 2)
            self.assertTrue(all(key.startswith("_orig_mod.") for key in wrappers[0].state_dict()))
            last = run / "checkpoints" / "last.pt"
            first = load_checkpoint(last)
            self.assertTrue(first["config"]["compile"])
            self.assertEqual(first["config"]["compile_mode"], "reduce-overhead")
            plain = build_model(first["config"]["model_config"])
            self.assertEqual(set(first["model_state"]), set(plain.state_dict()))
            plain.load_state_dict(first["model_state"], strict=True)
            with patch("torch.compile", side_effect=AssertionError("Evaluation should default to eager")):
                eager_result = evaluate(evaluation_parser().parse_args([
                    "--checkpoint", str(last), "--device", "cpu", "--no-progress"]))
            self.assertFalse(eager_result["execution"]["compile"])
            with patch("torch.compile", side_effect=fake_compile) as compiler:
                compiled_result = evaluate(evaluation_parser().parse_args([
                    "--checkpoint", str(last), "--device", "cpu", "--no-progress", "--compile"]))
            self.assertEqual(compiler.call_count, 1)
            self.assertEqual(compiled_result["metrics"], eager_result["metrics"])
            self.assertTrue(compiled_result["execution"]["compile"])
            with patch("torch.compile", side_effect=AssertionError("Resume should be eager")):
                train(training_parser().parse_args([
                    "--resume", str(last), "--epochs", "2", "--no-compile", "--no-progress"]))
            second = load_checkpoint(last)
            self.assertEqual(second["epoch"], 2)
            self.assertFalse(second["config"]["compile"])
            self.assertEqual(set(second["model_state"]), set(plain.state_dict()))


if __name__ == "__main__":
    unittest.main()
