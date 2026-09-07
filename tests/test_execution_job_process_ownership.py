"""Phase J process-ownership verification tests
(research/execution_job/process_ownership.py). No real process touched."""

from __future__ import annotations

import pytest

from research.execution_job.process_ownership import (
    LiveProcessSnapshot,
    ProcessIdentity,
    ProcessOwnershipError,
    verify_process_identity,
)

RECORDED = ProcessIdentity(
    pid=4242,
    launch_timestamp="2026-09-06T00:00:00+00:00",
    command_fingerprint="fake-runner::JOB-0001",
    job_id="JOB-0001",
    cwd="research/execution_jobs",
    executable="fake-runner",
)


class TestVerifyProcessIdentity:
    def test_exact_match_verifies(self):
        live = LiveProcessSnapshot(
            pid=4242,
            start_timestamp="2026-09-06T00:00:00+00:00",
            command_fingerprint="fake-runner::JOB-0001",
            cwd="research/execution_jobs",
        )
        assert verify_process_identity(RECORDED, live) is True

    def test_process_not_found_returns_false(self):
        live = LiveProcessSnapshot(pid=4242, start_timestamp=None, command_fingerprint=None, cwd=None, exists=False)
        assert verify_process_identity(RECORDED, None) is False
        assert verify_process_identity(RECORDED, live) is False

    def test_pid_reused_by_different_process_returns_false(self):
        """Same PID, but the live process's own start time / command don't
        match what was recorded -- classic PID-reuse scenario. Must NOT be
        treated as a match."""
        live = LiveProcessSnapshot(
            pid=4242,
            start_timestamp="2026-09-07T00:00:00+00:00",  # different process, reused PID
            command_fingerprint="some-unrelated-process",
            cwd="/some/other/cwd",
        )
        assert verify_process_identity(RECORDED, live) is False

    def test_different_pid_returns_false(self):
        live = LiveProcessSnapshot(
            pid=9999,
            start_timestamp="2026-09-06T00:00:00+00:00",
            command_fingerprint="fake-runner::JOB-0001",
            cwd="research/execution_jobs",
        )
        assert verify_process_identity(RECORDED, live) is False

    def test_ambiguous_missing_fields_raises_not_guesses(self):
        """A process exists at that PID but its identifying metadata could
        not be read -- must BLOCK (raise), never silently treat as
        matching or not matching."""
        live = LiveProcessSnapshot(pid=4242, start_timestamp=None, command_fingerprint=None, cwd=None, exists=True)
        with pytest.raises(ProcessOwnershipError):
            verify_process_identity(RECORDED, live)

    def test_cwd_mismatch_returns_false(self):
        live = LiveProcessSnapshot(
            pid=4242,
            start_timestamp="2026-09-06T00:00:00+00:00",
            command_fingerprint="fake-runner::JOB-0001",
            cwd="/completely/different/dir",
        )
        assert verify_process_identity(RECORDED, live) is False
