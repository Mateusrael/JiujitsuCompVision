"""Allow independent queues while keeping each training output single-writer.

Claims cover pending and running jobs, but never reserve GPUs. A legacy manager
may continue holding the old project-wide lock: its output paths are respected
without preventing a new queue from training into different directories.
"""

from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path


class QueueAlreadyRunning(RuntimeError):
    """A second supervisor must not overwrite the first supervisor's state."""


@contextmanager
def _exclusive(path, message, error_type=RuntimeError):
    import fcntl

    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise error_type(message) from error
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _record(handle):
    handle.seek(0)
    contents = handle.read().strip()
    if not contents:
        return None
    try:
        record = json.loads(contents)
        directory = record["directory"]
        if not isinstance(directory, str) or not Path(directory).is_absolute():
            raise ValueError("directory must be an absolute path")
    except (ValueError, TypeError, KeyError) as error:
        raise RuntimeError(f"Invalid queue ownership record: {handle.name}") from error
    return Path(directory).resolve()


def _state(directory):
    path = directory / "state.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or not isinstance(state.get("jobs"), list):
            raise ValueError("state must contain a jobs list")
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot inspect previous queue state: {path}") from error
    return state


def _check_owner(owner, requested, current_directory, *, lock_held=False):
    # Import lazily: queue_manager uses queue_claims to enter its run loop.
    from src.Train.queue_manager import FINAL, manager_alive, process_token

    state = _state(owner)
    active = lock_held or manager_alive(state)
    if active and owner == current_directory:
        raise QueueAlreadyRunning(f"This queue already has an active manager: {owner}")
    jobs = state["jobs"]
    if active and not jobs:
        # The detached launcher initially writes an empty starting state. Its
        # plan already identifies all reserved output paths during that window.
        try:
            jobs = json.loads((owner / "plan.json").read_text(encoding="utf-8"))["jobs"]
            if not isinstance(jobs, list):
                raise ValueError("plan must contain a jobs list")
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise RuntimeError(f"Cannot inspect active queue plan: {owner / 'plan.json'}") from error
    for job in jobs:
        if not isinstance(job, dict) or not isinstance(job.get("run_dir"), str):
            raise RuntimeError(f"Queue contains a job without an output directory: {owner}")
        run_dir = Path(job["run_dir"]).resolve()
        if run_dir not in requested:
            continue
        pid, token = job.get("pid"), job.get("process_token")
        if pid and token and process_token(pid) == token:
            raise RuntimeError(f"Training output is still owned by live job {job.get('id')} "
                               f"(PID {pid}) in {owner}: {run_dir}")
        if active and job.get("status", "pending") not in FINAL:
            raise RuntimeError(f"Training output is already scheduled by {owner}: {run_dir}")


def _check_legacy(path, requested, current_directory):
    import fcntl

    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError:
        return
    with handle:
        acquired = False
        try:
            # Shared probing works with a read-only descriptor. The old
            # supervisor holds an exclusive lock for its entire lifetime.
            try:
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                pass
            owner = _record(handle)
            if owner is None:
                if not acquired:
                    raise RuntimeError("Legacy queue is starting without an ownership record; retry shortly")
                return
            _check_owner(owner, requested, current_directory, lock_held=not acquired)
        finally:
            if acquired:
                fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def queue_claims(directory, plan):
    """Claim this supervisor and all of its output paths until it exits.

    Existing claim records are inspected before replacement so a surviving
    training child remains protected even after its manager has crashed.
    Separate output paths may run concurrently, including on the same GPU.
    """
    directory = Path(directory).resolve()
    diagnostics = Path(plan["project_root"]).resolve() / "diagnostics"
    claims = diagnostics / ".training-run-locks"
    requested = {Path(job["run_dir"]).resolve() for job in plan["jobs"]}
    if len(requested) != len(plan["jobs"]):
        raise ValueError("Two queue jobs would write the same training output")
    with ExitStack() as stack:
        stack.enter_context(_exclusive(
            directory / ".manager.lock", f"This queue already has an active manager: {directory}",
            QueueAlreadyRunning))
        _check_legacy(diagnostics / ".training-queue.lock", requested, directory)
        claims.mkdir(parents=True, exist_ok=True)
        handles = []
        for run_dir in sorted(requested):
            digest = hashlib.sha256(str(run_dir).encode("utf-8")).hexdigest()
            handle = stack.enter_context(_exclusive(
                claims / f"{digest}.lock",
                f"Training output is already claimed by another queue: {run_dir}"))
            owner = _record(handle)
            if owner is not None:
                _check_owner(owner, {run_dir}, directory)
            handles.append(handle)
        # Keep every previous ownership record intact until every claim and
        # orphan check succeeds. Failed starts release all acquired handles.
        for handle in handles:
            handle.seek(0)
            handle.truncate()
            json.dump({"directory": str(directory)}, handle)
            handle.flush()
        yield
