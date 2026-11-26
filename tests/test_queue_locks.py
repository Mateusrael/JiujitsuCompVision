"""Independent queue ownership, legacy coexistence, and orphan writers."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.Train.queue_locks import QueueAlreadyRunning, queue_claims


class FakeFcntl:
    """Model descriptor-owned advisory locks for the Windows test environment."""
    LOCK_SH, LOCK_EX, LOCK_NB, LOCK_UN = 1, 2, 4, 8

    def __init__(self):
        self.owners = {}

    def flock(self, handle, operation):
        fd = handle.fileno()
        stat = os.fstat(fd)
        key = (stat.st_dev, stat.st_ino)
        owners = self.owners.setdefault(key, {})
        if operation == self.LOCK_UN:
            owners.pop(fd, None)
            return
        mode = operation & (self.LOCK_SH | self.LOCK_EX)
        if any(other != fd and (mode == self.LOCK_EX or held == self.LOCK_EX)
               for other, held in owners.items()):
            raise BlockingIOError("already locked")
        owners[fd] = mode


class QueueClaimTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fake = FakeFcntl()
        patcher = patch.dict(sys.modules, {"fcntl": self.fake})
        patcher.start()
        self.addCleanup(patcher.stop)

    def queue(self, name, outputs):
        directory = self.root / "diagnostics" / "queues" / name
        directory.mkdir(parents=True)
        plan = {"name": name, "project_root": str(self.root), "jobs": [
            {"id": f"job{i}", "run_dir": str(self.root / "trainings" / output)}
            for i, output in enumerate(outputs)]}
        (directory / "plan.json").write_text(json.dumps(plan))
        (directory / "state.json").write_text(json.dumps({
            "status": "starting", "jobs": [{**job, "status": "pending"} for job in plan["jobs"]]}))
        return directory, plan

    def update(self, directory, **updates):
        path = directory / "state.json"
        state = json.loads(path.read_text())
        state.update(updates)
        path.write_text(json.dumps(state))

    def legacy(self, owner):
        path = self.root / "diagnostics" / ".training-queue.lock"
        path.write_text(json.dumps({"directory": str(owner)}))
        return path

    def test_distinct_queues_claim_outputs_concurrently_without_gpu_restrictions(self):
        first, first_plan = self.queue("first", ["a"])
        second, second_plan = self.queue("second", ["b"])
        with queue_claims(first, first_plan), queue_claims(second, second_plan):
            claims = list((self.root / "diagnostics" / ".training-run-locks").glob("*.lock"))
            self.assertEqual(len(claims), 2)

    def test_duplicate_manager_cannot_claim_or_change_original_state(self):
        directory, plan = self.queue("first", ["a"])
        before = (directory / "state.json").read_bytes()
        with queue_claims(directory, plan):
            with self.assertRaises(QueueAlreadyRunning):
                with queue_claims(directory, plan):
                    self.fail("Duplicate manager entered")
        self.assertEqual(before, (directory / "state.json").read_bytes())

    def test_pending_output_is_reserved_across_queues(self):
        first, first_plan = self.queue("first", ["shared"])
        second, second_plan = self.queue("second", ["shared"])
        with queue_claims(first, first_plan):
            with self.assertRaisesRegex(RuntimeError, "already claimed"):
                with queue_claims(second, second_plan):
                    self.fail("Output had two owners")

    def test_failed_partial_acquisition_releases_claims_and_preserves_prior_records(self):
        previous, previous_plan = self.queue("previous", ["a"])
        with queue_claims(previous, previous_plan):
            pass
        claim = next((self.root / "diagnostics" / ".training-run-locks").glob("*.lock"))
        before = claim.read_bytes()
        busy, busy_plan = self.queue("busy", ["z"])
        blocked, blocked_plan = self.queue("blocked", ["a", "z"])
        subsequent, subsequent_plan = self.queue("subsequent", ["a"])
        with queue_claims(busy, busy_plan):
            with self.assertRaisesRegex(RuntimeError, "already claimed"):
                with queue_claims(blocked, blocked_plan):
                    self.fail("Conflicting output was acquired")
            self.assertEqual(before, claim.read_bytes())
            with queue_claims(subsequent, subsequent_plan):
                pass

    def test_live_orphan_blocks_same_output_but_reused_pid_does_not(self):
        previous, previous_plan = self.queue("previous", ["shared"])
        with queue_claims(previous, previous_plan):
            pass
        self.update(previous, status="failed", jobs=[{
            **previous_plan["jobs"][0], "status": "running", "pid": 123, "process_token": "boot:old"}])
        next_dir, next_plan = self.queue("next", ["shared"])
        with patch("src.Train.queue_manager.process_token", return_value="boot:old"):
            with self.assertRaisesRegex(RuntimeError, "still owned by live job"):
                with queue_claims(next_dir, next_plan):
                    self.fail("Orphan output was acquired")
        with patch("src.Train.queue_manager.process_token", return_value="boot:new"):
            with queue_claims(next_dir, next_plan):
                pass

    def test_missing_previous_state_fails_closed(self):
        previous, previous_plan = self.queue("previous", ["shared"])
        with queue_claims(previous, previous_plan):
            pass
        (previous / "state.json").unlink()
        next_dir, next_plan = self.queue("next", ["shared"])
        with self.assertRaisesRegex(RuntimeError, "Cannot inspect previous queue state"):
            with queue_claims(next_dir, next_plan):
                self.fail("Unknown output owner was discarded")

    def test_legacy_lock_allows_unrelated_queue_but_preserves_its_pending_outputs(self):
        legacy, legacy_plan = self.queue("legacy", ["image-run"])
        other, other_plan = self.queue("other", ["pose-run"])
        conflict, conflict_plan = self.queue("conflict", ["image-run"])
        with self.legacy(legacy).open("r") as handle:
            self.fake.flock(handle, self.fake.LOCK_EX | self.fake.LOCK_NB)
            try:
                with queue_claims(other, other_plan):
                    pass
                with self.assertRaisesRegex(RuntimeError, "already scheduled"):
                    with queue_claims(conflict, conflict_plan):
                        self.fail("Legacy pending output was reused")
            finally:
                self.fake.flock(handle, self.fake.LOCK_UN)

    def test_legacy_same_directory_raises_special_exception_without_state_change(self):
        legacy, plan = self.queue("legacy", ["image-run"])
        before = (legacy / "state.json").read_bytes()
        with self.legacy(legacy).open("r") as handle:
            self.fake.flock(handle, self.fake.LOCK_EX | self.fake.LOCK_NB)
            try:
                with self.assertRaises(QueueAlreadyRunning):
                    with queue_claims(legacy, plan):
                        self.fail("Legacy manager was duplicated")
            finally:
                self.fake.flock(handle, self.fake.LOCK_UN)
        self.assertEqual(before, (legacy / "state.json").read_bytes())

    def test_legacy_starting_state_uses_plan_to_reserve_outputs(self):
        legacy, _ = self.queue("legacy", ["shared"])
        self.update(legacy, jobs=[], manager_pid=321, manager_token="boot:manager")
        self.legacy(legacy)
        next_dir, next_plan = self.queue("next", ["shared"])
        with patch("src.Train.queue_manager.process_token", return_value="boot:manager"):
            with self.assertRaisesRegex(RuntimeError, "already scheduled"):
                with queue_claims(next_dir, next_plan):
                    self.fail("Starting queue's output was reused")

    def test_legacy_orphan_is_protected_even_without_live_manager_or_held_lock(self):
        legacy, plan = self.queue("legacy", ["shared"])
        self.update(legacy, status="failed", jobs=[{
            **plan["jobs"][0], "pid": 123, "process_token": "boot:child", "status": "running"}])
        self.legacy(legacy)
        next_dir, next_plan = self.queue("next", ["shared"])
        with patch("src.Train.queue_manager.process_token", return_value="boot:child"):
            with self.assertRaisesRegex(RuntimeError, "still owned by live job"):
                with queue_claims(next_dir, next_plan):
                    self.fail("Orphan legacy output was reused")


@unittest.skipUnless(sys.platform == "linux", "Real flock requires Linux")
class RealQueueClaimTests(unittest.TestCase):
    def test_real_file_locks_allow_independent_queues_and_block_duplicate_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plans = []
            for name, output in (("first", "shared"), ("second", "other"), ("third", "shared")):
                directory = root / name
                directory.mkdir()
                plan = {"project_root": str(root), "jobs": [{"id": "job", "run_dir": str(root / output)}]}
                (directory / "state.json").write_text(json.dumps({"jobs": plan["jobs"]}))
                plans.append((directory, plan))
            with queue_claims(*plans[0]), queue_claims(*plans[1]):
                with self.assertRaisesRegex(RuntimeError, "already claimed"):
                    with queue_claims(*plans[2]):
                        self.fail("Real flock permitted duplicate output")


if __name__ == "__main__":
    unittest.main()
