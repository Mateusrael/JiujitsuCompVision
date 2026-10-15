"""Training output must stay isolated, readable, and bounded on cancellation."""

import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from src.Train.queue_process import TrainingProcess, follow_log


class QueueProcessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.log = self.root / "run.log"

    def launch(self, code, **kwargs):
        child_env = dict(os.environ, QUEUE_TEST_VALUE="inherited")
        child = TrainingProcess([sys.executable, "-u", "-c", code], self.root,
                                child_env, self.log, **kwargs)
        self.addCleanup(child.terminate, grace_seconds=0.1)
        return child

    def test_captures_both_streams_terminal_bytes_environment_and_exit_status(self):
        child = self.launch(
            "import os, sys; "
            "print(os.environ['QUEUE_TEST_VALUE']); "
            "os.write(1, b'\\x1b[32mstep 1\\rstep 2\\x1b[0m'); "
            "os.write(2, b'error stream\\n'); sys.exit(7)")
        self.assertGreater(child.pid, 0)
        self.assertEqual(child.wait(timeout=10), 7)
        self.assertEqual(child.poll(), 7)
        payload = self.log.read_bytes()
        self.assertIn(b"inherited", payload)
        self.assertIn(b"\x1b[32mstep 1\rstep 2\x1b[0m", payload)
        self.assertIn(b"error stream", payload)
        self.assertFalse(child._reader.is_alive())
        self.assertIsNone(child._output_fd)
        self.assertTrue(child._log.closed)

    def test_wait_timeout_does_not_kill_training_and_termination_reaps_it(self):
        child = self.launch("import time; print('started', flush=True); time.sleep(30)")
        with self.assertRaises(subprocess.TimeoutExpired):
            child.wait(timeout=0.05)
        self.assertIsNone(child.poll())
        result = child.terminate(grace_seconds=0.1)
        self.assertNotEqual(result, 0)
        self.assertEqual(child.wait(timeout=1), result)
        self.assertFalse(child._reader.is_alive())

    def test_start_failure_closes_log(self):
        opened = []
        actual_open = Path.open

        def record_open(path, *args, **kwargs):
            stream = actual_open(path, *args, **kwargs)
            opened.append(stream)
            return stream

        with patch.object(Path, "open", record_open), \
                patch("src.Train.queue_process.subprocess.Popen",
                      side_effect=FileNotFoundError("missing executable")):
            with self.assertRaisesRegex(FileNotFoundError, "missing executable"):
                TrainingProcess(["missing"], self.root, dict(os.environ), self.log)
        self.assertEqual(len(opened), 1)
        self.assertTrue(opened[0].closed)

    def test_reader_start_failure_reaps_launched_child(self):
        started = []
        actual_popen = subprocess.Popen

        def record_process(*args, **kwargs):
            child = actual_popen(*args, **kwargs)
            started.append(child)
            return child

        with patch("src.Train.queue_process.subprocess.Popen", record_process), \
                patch("src.Train.queue_process.threading.Thread.start",
                      side_effect=RuntimeError("cannot start reader")):
            with self.assertRaisesRegex(RuntimeError, "cannot start reader"):
                TrainingProcess(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    self.root, dict(os.environ), self.log)
        self.assertEqual(len(started), 1)
        self.assertIsNotNone(started[0].poll())
        if started[0].stdout is not None:
            self.assertTrue(started[0].stdout.closed)

    def test_repeated_cleanup_never_signals_retired_process_group(self):
        child = self.launch("print('finished')")
        self.assertEqual(child.wait(timeout=10), 0)
        with patch.object(child, "_signal_group") as signal_group, \
                patch.object(child, "_kill_group") as kill_group:
            self.assertEqual(child.terminate(), 0)
            self.assertEqual(child.poll(), 0)
            self.assertEqual(child.wait(), 0)
        signal_group.assert_not_called()
        kill_group.assert_not_called()

    def test_log_failure_is_reported_and_stops_child(self):
        actual_open = Path.open

        class BrokenLog:
            def __init__(self, stream):
                self.stream = stream

            @property
            def closed(self):
                return self.stream.closed

            def write(self, data):
                raise OSError("disk is full")

            def close(self):
                self.stream.close()

        with patch.object(Path, "open", lambda path, *args, **kwargs:
                          BrokenLog(actual_open(path, *args, **kwargs))):
            child = TrainingProcess(
                [sys.executable, "-u", "-c",
                 "import time; print('first output', flush=True); time.sleep(30)"],
                self.root, dict(os.environ), self.log)
        try:
            with self.assertRaisesRegex(RuntimeError, "disk is full"):
                child.wait(timeout=10)
            self.assertIsNotNone(child._process.poll())
            self.assertFalse(child._reader.is_alive())
            self.assertTrue(child._log.closed)
        finally:
            if child._process.poll() is None:
                child._process.kill()
                child._process.wait()

    @unittest.skipUnless(os.name == "posix", "PTY and process groups require POSIX")
    def test_linux_child_has_terminal_dimensions_and_own_process_group(self):
        child = self.launch(
            "import os, sys; "
            "print('tty', sys.stdin.isatty(), sys.stdout.isatty(), sys.stderr.isatty()); "
            "print('size', *os.get_terminal_size(1)); "
            "print('group', os.getpid() == os.getpgrp())",
            columns=132, rows=45)
        self.assertEqual(child.wait(timeout=10), 0)
        payload = self.log.read_text()
        self.assertIn("tty False True True", payload)
        self.assertIn("size 132 45", payload)
        self.assertIn("group True", payload)

    @unittest.skipUnless(os.name == "posix", "PTY descendants require POSIX")
    def test_descendant_holding_output_cannot_hang_wait(self):
        child = self.launch(
            "import subprocess, sys; "
            "worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
            "print(worker.pid, flush=True)")
        before = time.monotonic()
        self.assertEqual(child.wait(timeout=10), 0)
        self.assertLess(time.monotonic() - before, 4)
        self.assertFalse(child._reader.is_alive())
        self.assertIsNone(child._output_fd)
        # Cleanup targets the owned group, including the surviving worker.
        child.terminate(grace_seconds=0.1)

    @unittest.skipUnless(os.name == "posix", "POSIX signal escalation")
    def test_termination_escalates_when_child_ignores_sigterm(self):
        child = self.launch(
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready', flush=True); time.sleep(30)")
        deadline = time.monotonic() + 10
        while b"ready" not in self.log.read_bytes():
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
        self.assertEqual(child.terminate(grace_seconds=0.05), -9)
        self.assertFalse(child._reader.is_alive())


class FollowLogTests(unittest.TestCase):
    def test_replays_tail_and_follows_new_bytes_without_losing_terminal_controls(self):
        source = io.BytesIO(b"old output\n\x1b[32mfirst\r")
        output = io.BytesIO()
        checks = 0

        def finished():
            nonlocal checks
            checks += 1
            if checks == 1:
                position = source.tell()
                source.seek(0, os.SEEK_END)
                source.write(b"second\x1b[0m\n")
                source.seek(position)
                return False
            return True

        with patch.object(Path, "open", return_value=source), \
                patch("src.Train.queue_process.sys.stdout") as stdout, \
                patch("src.Train.queue_process.time.sleep"):
            stdout.buffer = output
            follow_log(Path("run.log"), finished, tail_bytes=len(b"\x1b[32mfirst\r"))
        self.assertEqual(output.getvalue(), b"\x1b[32mfirst\rsecond\x1b[0m\n")
        self.assertTrue(source.closed)

    def test_ctrl_c_detaches_and_closes_reader_without_signalling_a_process(self):
        source = io.BytesIO(b"output")
        with patch.object(Path, "open", return_value=source), \
                patch("src.Train.queue_process.sys.stdout") as stdout, \
                patch("src.Train.queue_process.time.sleep", side_effect=KeyboardInterrupt), \
                patch("src.Train.queue_process.os.kill") as kill:
            stdout.buffer = io.BytesIO()
            follow_log(Path("run.log"), lambda: False)
        self.assertTrue(source.closed)
        kill.assert_not_called()

    def test_queued_log_can_appear_later_and_missing_finished_log_returns(self):
        source = io.BytesIO(b"eventual output")
        with patch.object(Path, "open", side_effect=[FileNotFoundError(), source]), \
                patch("src.Train.queue_process.sys.stdout") as stdout, \
                patch("src.Train.queue_process.time.sleep"):
            stdout.buffer = io.BytesIO()
            follow_log(Path("run.log"), lambda: source.tell() > 0)
            self.assertEqual(stdout.buffer.getvalue(), b"eventual output")
        with patch.object(Path, "open", side_effect=FileNotFoundError()), \
                patch("src.Train.queue_process.sys.stdout") as stdout:
            stdout.buffer = io.BytesIO()
            follow_log(Path("run.log"), lambda: True)
            self.assertEqual(stdout.buffer.getvalue(), b"")

    def test_invalid_follow_options_are_rejected(self):
        for options in ({"tail_bytes": -1}, {"interval": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                follow_log(Path("run.log"), lambda: True, **options)


if __name__ == "__main__":
    unittest.main()
