"""GPU scheduling uses observed UUID visibility and conservative telemetry."""

import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from src.Train.queue_gpu import GPU, choose_gpu, discover_allowed_gpus, query_gpus


UUIDS = [f"GPU-00000000-0000-0000-0000-{index:012x}" for index in range(6)]
ROOT = Path("project")


def output(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def probe(values):
    return output("dependency notice\nJIUJITSU_VISIBLE_GPU_UUIDS=" + json.dumps(values) + "\n")


class GPUDiscoveryTests(unittest.TestCase):
    def test_probe_uuid_mapping_is_authority_for_reordered_mask(self):
        with patch("src.Train.queue_gpu.subprocess.run", return_value=probe(
                [UUIDS[5][4:].upper(), UUIDS[1]])) as run:
            allowed = discover_allowed_gpus("venv-python", ROOT)
        self.assertEqual(allowed, {UUIDS[5], UUIDS[1]})
        args, kwargs = run.call_args
        self.assertEqual(args[0][:2], ["venv-python", "-c"])
        self.assertEqual(kwargs["cwd"], ROOT)
        self.assertNotIn("env", kwargs)
        self.assertIn("get_device_properties", args[0][2])
        self.assertNotIn("torch.empty", args[0][2])

    def test_void_container_variable_does_not_override_visible_torch_devices(self):
        with patch.dict("os.environ", {"NVIDIA_VISIBLE_DEVICES": "void", "CUDA_VISIBLE_DEVICES": "5,1"}):
            with patch("src.Train.queue_gpu.subprocess.run", return_value=probe(UUIDS)):
                self.assertEqual(discover_allowed_gpus("python", ROOT), set(UUIDS))

    def test_zero_visible_devices_returns_empty_set(self):
        with patch("src.Train.queue_gpu.subprocess.run", return_value=probe([])):
            self.assertEqual(discover_allowed_gpus("python", ROOT), set())

    def test_bad_or_ambiguous_probe_response_is_error(self):
        responses = [output(""), output("JIUJITSU_VISIBLE_GPU_UUIDS=not-json"),
                     probe({"gpu": UUIDS[0]}), probe([None]), probe([""]),
                     probe(["MIG-" + UUIDS[0][4:]]), probe([UUIDS[0], UUIDS[0]]),
                     output(probe([]).stdout + probe([]).stdout)]
        for response in responses:
            with self.subTest(response=response.stdout):
                with patch("src.Train.queue_gpu.subprocess.run", return_value=response):
                    with self.assertRaises(RuntimeError):
                        discover_allowed_gpus("python", ROOT)

    def test_probe_failure_preserves_useful_error(self):
        with patch("src.Train.queue_gpu.subprocess.run", return_value=output(stderr="missing torch", returncode=1)):
            with self.assertRaisesRegex(RuntimeError, "missing torch"):
                discover_allowed_gpus("python", ROOT)


class GPUTelemetryTests(unittest.TestCase):
    def test_parse_csv_and_skip_prohibited_or_unavailable_gpus(self):
        rows = (f'0, {UUIDS[0]}, "NVIDIA H100, example", 81559, 12000, 20, Default\n'
                f'1, {UUIDS[1]}, NVIDIA H100, 81559, 40000, 0, Prohibited\n'
                f'2, {UUIDS[2]}, NVIDIA H100, 81559, N/A, 0, Default\n'
                f'3, {UUIDS[3]}, NVIDIA H100, 81559, 30000, [Not Supported], Default\n'
                f'4, {UUIDS[4]}, NVIDIA H100, 81559, 20000, 0, Exclusive_Process\n')
        with patch("src.Train.queue_gpu.subprocess.run", return_value=output(rows)) as run:
            actual = query_gpus()
        self.assertEqual(actual, [GPU(0, UUIDS[0], "NVIDIA H100, example", 81559, 12000, 20),
                                  GPU(4, UUIDS[4], "NVIDIA H100", 81559, 20000, 0)])
        self.assertEqual(run.call_args.args[0][0], "nvidia-smi")
        self.assertIn("--format=csv,noheader,nounits", run.call_args.args[0])
        self.assertGreater(run.call_args.kwargs["timeout"], 0)

    def test_empty_query_has_no_gpus(self):
        with patch("src.Train.queue_gpu.subprocess.run", return_value=output("\n")):
            self.assertEqual(query_gpus(), [])

    def test_reject_malformed_and_out_of_range_telemetry(self):
        template = ["0", UUIDS[0], "NVIDIA H100", "81559", "20000", "0", "Default"]
        examples = ["wrong fields", '0, "unfinished', ", ".join(template) + "\n" + ", ".join(template)]
        for column, invalid in [(0, "-1"), (1, "bad-uuid"), (2, ""), (3, "0"),
                                (4, "90000"), (4, "20 GiB"), (4, "1.5"),
                                (5, "101"), (5, "-1"), (6, "unknown")]:
            fields = list(template)
            fields[column] = invalid
            examples.append(", ".join(fields))
        for text in examples:
            with self.subTest(text=text):
                with patch("src.Train.queue_gpu.subprocess.run", return_value=output(text)):
                    with self.assertRaises(RuntimeError):
                        query_gpus()

    def test_command_absence_timeout_and_failure_are_errors(self):
        for failure in (FileNotFoundError("no nvidia-smi"), subprocess.TimeoutExpired("nvidia-smi", 30)):
            with self.subTest(failure=failure):
                with patch("src.Train.queue_gpu.subprocess.run", side_effect=failure):
                    with self.assertRaisesRegex(RuntimeError, "GPU discovery command failed"):
                        query_gpus()
        with patch("src.Train.queue_gpu.subprocess.run", return_value=output(stderr="driver unavailable", returncode=9)):
            with self.assertRaisesRegex(RuntimeError, "driver unavailable"):
                query_gpus()


class GPUSelectionTests(unittest.TestCase):
    def test_masks_reservations_thresholds_and_deterministic_ranking(self):
        gpus = [GPU(0, UUIDS[0], "H100", 80000, 70000, 0),
                GPU(1, UUIDS[1], "H100", 80000, 60000, 0),
                GPU(2, UUIDS[2], "H100", 80000, 50000, 10),
                GPU(3, UUIDS[3], "H100", 80000, 30000, 20),
                GPU(4, UUIDS[4], "H100", 80000, 40000, 20),
                GPU(5, UUIDS[5], "H100", 80000, 40000, 20)]
        allowed = set(UUIDS[1:])
        busy = {UUIDS[1], UUIDS[2]}
        self.assertEqual(choose_gpu(list(reversed(gpus)), allowed, busy, 40000), gpus[4])
        self.assertEqual(choose_gpu(gpus, allowed, {UUIDS[1]}, 40000), gpus[2])
        self.assertIsNone(choose_gpu(gpus, allowed, busy, 40001))
        self.assertIsNone(choose_gpu(gpus, allowed, busy, 40000, 19))
        self.assertEqual(choose_gpu(gpus, allowed, busy, 40000, 20), gpus[4])

    def test_no_visible_or_eligible_device_never_falls_back(self):
        gpu = GPU(0, UUIDS[0], "H100", 80000, 1, 100)
        self.assertIsNone(choose_gpu([gpu], set(), set(), 0))
        self.assertIsNone(choose_gpu([gpu], {gpu.uuid}, {gpu.uuid}, 0))
        self.assertIsNone(choose_gpu([], {gpu.uuid}, set(), 0))
        self.assertEqual(choose_gpu([gpu], {gpu.uuid}, set(), 0), gpu)

    def test_invalid_thresholds_are_rejected(self):
        for minimum in (-1, 1.5, True):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                choose_gpu([], set(), set(), minimum)
        for utilization in (-1, 101, 1.5, True):
            with self.subTest(utilization=utilization), self.assertRaises(ValueError):
                choose_gpu([], set(), set(), 0, utilization)


if __name__ == "__main__":
    unittest.main()
