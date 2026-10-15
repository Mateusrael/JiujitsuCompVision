"""Capture one training process without mixing its terminal output with others.

On Linux a pseudo-terminal keeps tqdm enabled. Each log contains terminal bytes,
including carriage returns and ANSI sequences, so it can be followed separately.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Callable


class TrainingProcess:
    """Own a child process group and continuously copy its output to a log.

    ``poll`` and ``wait`` also report log-capture failures. ``wait`` drains output
    for at most one second after the child exits, even if a descendant inherited
    the output descriptor. Windows uses a pipe for local verification; the Linux
    queue uses a PTY with a fixed width (120 columns by default).
    """

    _DRAIN_SECONDS = 1.0

    def __init__(self, command: list[str], cwd: Path, env: dict[str, str],
                 log_path: Path, columns: int = 120, rows: int = 40):
        if not command:
            raise ValueError("Training command cannot be empty")
        if not (0 < columns <= 65535 and 0 < rows <= 65535):
            raise ValueError("Terminal dimensions must be between 1 and 65535")
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop_reader = threading.Event()
        self._reader_error: BaseException | None = None
        self._reader: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._group_finished = False
        self._output_fd: int | None = None
        self._output_pipe = None
        self._log = self.log_path.open("ab", buffering=0)
        slave_fd = None
        try:
            if os.name == "posix":
                import fcntl
                import pty
                import struct
                import termios

                self._output_fd, slave_fd = pty.openpty()
                fcntl.ioctl(slave_fd, termios.TIOCSWINSZ,
                            struct.pack("HHHH", rows, columns, 0, 0))
                os.set_blocking(self._output_fd, False)
                self._process = subprocess.Popen(
                    command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                    stdout=slave_fd, stderr=slave_fd, start_new_session=True,
                    close_fds=True)
            else:
                self._process = subprocess.Popen(
                    command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    bufsize=0, close_fds=True)
                self._output_pipe = self._process.stdout
                self._output_fd = self._output_pipe.fileno()
            self._reader = threading.Thread(
                target=self._capture, name=f"training-log-{self.pid}", daemon=True)
            self._reader.start()
        except BaseException:
            if self._process is not None:
                self._signal_group(signal.SIGTERM)
                try:
                    self._process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self._kill_group()
                    self._process.wait()
            self._close_output()
            self._log.close()
            raise
        finally:
            if slave_fd is not None:
                os.close(slave_fd)

    @property
    def pid(self) -> int:
        return self._process.pid

    def _read_chunk(self) -> bytes | None:
        """Return bytes, None when waiting, or empty bytes at EOF."""
        if os.name == "posix":
            import select

            if not select.select([self._output_fd], [], [], 0.1)[0]:
                return None
            try:
                return os.read(self._output_fd, 65536)
            except BlockingIOError:
                return None
            except OSError as error:
                # Linux PTYs report EIO once the final slave descriptor closes.
                if error.errno == errno.EIO:
                    return b""
                raise

        # Pipes cannot be selected on Windows. Peek avoids an uninterruptible
        # reader when a descendant retains the pipe after its parent exits.
        import ctypes
        from ctypes import wintypes
        import msvcrt

        available = wintypes.DWORD()
        peek = ctypes.WinDLL("kernel32", use_last_error=True).PeekNamedPipe
        peek.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                         ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD),
                         ctypes.c_void_p)
        peek.restype = wintypes.BOOL
        if not peek(msvcrt.get_osfhandle(self._output_fd), None, 0, None,
                    ctypes.byref(available), None):
            error = ctypes.get_last_error()
            if error in (109, 232):  # Broken pipe / no data after closing.
                return b""
            raise ctypes.WinError(error)
        if available.value:
            return os.read(self._output_fd, min(available.value, 65536))
        self._stop_reader.wait(0.05)
        return None

    def _capture(self) -> None:
        exited_at = None
        try:
            while not self._stop_reader.is_set():
                if self._process.poll() is not None:
                    if exited_at is None:
                        exited_at = time.monotonic()
                    elif time.monotonic() - exited_at >= self._DRAIN_SECONDS:
                        break
                chunk = self._read_chunk()
                if chunk == b"":
                    break
                if chunk:
                    remaining = memoryview(chunk)
                    while remaining:
                        written = self._log.write(remaining)
                        if not written:
                            raise OSError("Training log write made no progress")
                        remaining = remaining[written:]
        except BaseException as error:
            self._reader_error = error
        finally:
            try:
                self._close_output()
            except BaseException as error:
                self._reader_error = self._reader_error or error
            try:
                self._log.close()
            except BaseException as error:
                self._reader_error = self._reader_error or error

    def _close_output(self) -> None:
        if self._output_pipe is not None:
            self._output_pipe.close()
            self._output_pipe = None
            self._output_fd = None
        elif self._output_fd is not None:
            os.close(self._output_fd)
            self._output_fd = None

    def _signal_group(self, sig: int) -> None:
        try:
            if os.name == "posix":
                os.killpg(self.pid, sig)
            elif self._process.poll() is None:
                self._process.terminate()
        except ProcessLookupError:
            pass

    def _kill_group(self) -> None:
        if os.name == "posix":
            self._signal_group(signal.SIGKILL)
        elif self._process.poll() is None:
            self._process.kill()

    def _finish_reader(self) -> None:
        self._reader.join(timeout=self._DRAIN_SECONDS + 0.3)
        if self._reader.is_alive():
            self._stop_reader.set()
            self._reader.join(timeout=0.5)
        if self._reader.is_alive():
            raise RuntimeError(f"Training log reader did not stop: {self.log_path}")
        if not self._group_finished and self._process.poll() is not None:
            # A failed training may leave worker processes alive. Once its log
            # is drained, retire that group exactly once, before reusing this
            # object could ever target a recycled process identifier.
            self._kill_group()
            self._group_finished = True

    def _raise_capture_error(self) -> None:
        if self._reader_error is not None:
            error = self._reader_error
            # A full disk must not leave a live training consuming a GPU after
            # the manager has marked the run failed.
            self._stop_owned_process(grace_seconds=1)
            self._finish_reader()
            raise RuntimeError(f"Cannot capture training log {self.log_path}: {error}") from error

    def poll(self) -> int | None:
        self._raise_capture_error()
        result = self._process.poll()
        if result is not None:
            self._finish_reader()
            self._raise_capture_error()
        return result

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            self._raise_capture_error()
            result = self._process.poll()
            if result is not None:
                self._finish_reader()
                self._raise_capture_error()
                return result
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise subprocess.TimeoutExpired(self._process.args, timeout)
            try:
                self._process.wait(timeout=min(0.1, remaining) if remaining is not None else 0.1)
            except subprocess.TimeoutExpired:
                pass

    def _stop_owned_process(self, grace_seconds: float) -> int:
        if grace_seconds < 0:
            raise ValueError("Termination grace must be nonnegative")
        if self._group_finished:
            return self._process.returncode
        self._signal_group(signal.SIGTERM)
        try:
            self._process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            self._kill_group()
            self._process.wait()
        # The parent may exit before its DataLoader workers. A final signal to
        # its own group prevents those descendants from surviving cancellation.
        self._kill_group()
        self._group_finished = True
        return self._process.returncode

    def terminate(self, grace_seconds: float = 10) -> int:
        """Stop this training and its process group, drain the log, and reap it."""
        result = self._stop_owned_process(grace_seconds)
        self._finish_reader()
        self._raise_capture_error()
        return result


def follow_log(path: Path, finished: Callable[[], bool], tail_bytes: int = 65536,
               interval: float = 0.2) -> None:
    """Replay recent terminal output and follow updates; Ctrl+C only detaches.

    A missing file is expected while the selected training is still queued.
    The caller supplies a completion predicate for the selected job, not the
    entire queue. Use a terminal at least as wide as the captured training PTY.
    """
    if tail_bytes < 0 or interval <= 0:
        raise ValueError("tail_bytes must be nonnegative and interval positive")
    output = sys.stdout.buffer
    source = None
    try:
        while True:
            if source is None:
                try:
                    source = Path(path).open("rb")
                except FileNotFoundError:
                    if finished():
                        return
                    time.sleep(interval)
                    continue
                size = source.seek(0, os.SEEK_END)
                source.seek(max(0, size - tail_bytes))
            chunk = source.read(65536)
            if chunk:
                output.write(chunk)
                output.flush()
                continue
            if finished():
                # Completion may have happened between the previous read and
                # status check. Read once more after the writer has finished.
                chunk = source.read()
                if chunk:
                    output.write(chunk)
                    output.flush()
                return
            time.sleep(interval)
    except KeyboardInterrupt:
        return
    finally:
        if source is not None:
            source.close()
