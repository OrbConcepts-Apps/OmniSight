"""Phase J JobManager end-to-end tests (research/execution_job/manager.py).
Every runner here is FakeRunner -- no real process, no real GPU, no real
training. Covers the Phase J authorization section 25 test matrix."""

from __future__ import annotations

import pytest

from research import git_isolation, operational_state
from research.execution_budget import ExecutionBudgetConfig, ExecutionBudgetError, ResourceEstimate
from research.execution_job import state as s
from research.execution_job.checkpoint import CheckpointRef
from research.execution_job.manager import (
    JobManager,
    JobManagerError,
    LaunchPreconditions,
    requires_human_intervention,
)
from research.execution_job.process_ownership import LiveProcessSnapshot, ProcessOwnershipError
from research.execution_job.runner import FakeRunner, RunnerStatus
from research.execution_job.checkpoint import IncompatibleCheckpointError


@pytest.fixture(autouse=True)
def _isolated_jobs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "JOBS_DIR", tmp_path / "execution_jobs")


@pytest.fixture
def ledger_path(tmp_path):
    return tmp_path / "ledger.jsonl"


@pytest.fixture
def authorized_budget():
    return ExecutionBudgetConfig(gpu_execution_authorized=True, max_concurrent_training_jobs=5)


NO_GIT_CHECK = LaunchPreconditions(require_clean_git_tree=False)


def _manager(ledger_path, runner=None):
    return JobManager(runner or FakeRunner(), ledger_path=ledger_path)


class TestLaunchDeniedGates:
    def test_denied_without_gpu_authorization(self, ledger_path):
        mgr = _manager(ledger_path)
        with pytest.raises(ExecutionBudgetError):
            mgr.launch(
                experiment_id="EXP-9003", spec={}, budget_config=ExecutionBudgetConfig(),
                preconditions=NO_GIT_CHECK,
            )

    def test_denied_without_execution_budget_config(self, ledger_path):
        mgr = _manager(ledger_path)
        with pytest.raises(ExecutionBudgetError):
            mgr.launch(
                experiment_id="EXP-9003", spec={}, budget_config=None,
                preconditions=NO_GIT_CHECK,
            )

    def test_denied_while_paused(self, ledger_path, authorized_budget):
        operational_state.pause("test pause")
        mgr = _manager(ledger_path)
        with pytest.raises(operational_state.OperationalPausedError):
            mgr.launch(
                experiment_id="EXP-9003", spec={}, budget_config=authorized_budget,
                preconditions=NO_GIT_CHECK,
            )

    def test_denied_while_stopped(self, ledger_path, authorized_budget):
        operational_state.stop("test stop")
        mgr = _manager(ledger_path)
        with pytest.raises(operational_state.OperationalStoppedError):
            mgr.launch(
                experiment_id="EXP-9003", spec={}, budget_config=authorized_budget,
                preconditions=NO_GIT_CHECK,
            )

    def test_denied_with_dirty_tree(self, ledger_path, authorized_budget, monkeypatch):
        def _raise(*args, **kwargs):
            raise git_isolation.GitIsolationError("simulated dirty tree")

        monkeypatch.setattr(git_isolation, "require_clean_tree", _raise)
        mgr = _manager(ledger_path)
        with pytest.raises(git_isolation.GitIsolationError):
            mgr.launch(
                experiment_id="EXP-9003", spec={}, budget_config=authorized_budget,
                preconditions=LaunchPreconditions(require_clean_git_tree=True),
            )

    def test_denied_with_unfrozen_spec(self, ledger_path, authorized_budget):
        def _unfrozen():
            raise RuntimeError("spec is not frozen/hash-verified")

        mgr = _manager(ledger_path)
        with pytest.raises(RuntimeError, match="not frozen"):
            mgr.launch(
                experiment_id="EXP-9003", spec={}, budget_config=authorized_budget,
                preconditions=LaunchPreconditions(require_clean_git_tree=False, frozen_spec_check=_unfrozen),
            )

    def test_denied_with_retry_above_default_zero_cap(self, ledger_path, authorized_budget):
        mgr = _manager(ledger_path)
        with pytest.raises(ExecutionBudgetError):
            mgr.launch(
                experiment_id="EXP-9003", spec={}, budget_config=authorized_budget,
                preconditions=NO_GIT_CHECK, retry_count=1,
            )

    def test_concurrent_job_limit_blocks_next_launch(self, ledger_path):
        config = ExecutionBudgetConfig(gpu_execution_authorized=True, max_concurrent_training_jobs=1)
        mgr = _manager(ledger_path)
        with pytest.raises(ExecutionBudgetError):
            mgr.launch(
                experiment_id="EXP-9003", spec={}, budget_config=config,
                preconditions=NO_GIT_CHECK, current_running_training_jobs=1,
            )


class TestLaunchSuccess:
    def test_successful_launch_reaches_running(self, ledger_path, authorized_budget):
        mgr = _manager(ledger_path)
        record = mgr.launch(
            experiment_id="EXP-9003", spec={"foo": "bar"}, budget_config=authorized_budget,
            preconditions=NO_GIT_CHECK,
        )
        assert record.state == s.RUNNING
        assert record.pid is not None
        reloaded = s.load_job(record.job_id)
        assert reloaded.state == s.RUNNING

    def test_launch_writes_started_ledger_entry(self, ledger_path, authorized_budget):
        from research.execution_job import accounting

        mgr = _manager(ledger_path)
        record = mgr.launch(
            experiment_id="EXP-9003", spec={}, budget_config=authorized_budget,
            preconditions=NO_GIT_CHECK,
        )
        entries = accounting.read_ledger(ledger_path)
        assert any(e["job_id"] == record.job_id and e["event"] == "STARTED" for e in entries)

    def test_prepare_failure_marks_job_failed_not_silently_dropped(self, ledger_path, authorized_budget):
        runner = FakeRunner()

        def _boom(job_id, spec):
            raise RuntimeError("prepare exploded")

        runner.prepare = _boom
        mgr = _manager(ledger_path, runner=runner)
        with pytest.raises(JobManagerError):
            mgr.launch(
                experiment_id="EXP-9003", spec={}, budget_config=authorized_budget,
                preconditions=NO_GIT_CHECK,
            )
        jobs = s.list_all_jobs()
        assert len(jobs) == 1
        assert jobs[0].state == s.FAILED

    def test_launch_failure_marks_job_failed(self, ledger_path, authorized_budget):
        runner = FakeRunner()
        mgr = _manager(ledger_path, runner=runner)
        runner.launch = lambda job_id: (_ for _ in ()).throw(RuntimeError("launch failed"))
        with pytest.raises(JobManagerError):
            mgr.launch(
                experiment_id="EXP-9003", spec={}, budget_config=authorized_budget,
                preconditions=NO_GIT_CHECK,
            )
        jobs = s.list_all_jobs()
        assert jobs[0].state == s.FAILED


class TestWallClockEnforcement:
    def _running_job(self, ledger_path, authorized_budget, runner):
        mgr = _manager(ledger_path, runner=runner)
        record = mgr.launch(
            experiment_id="EXP-9003", spec={}, budget_config=authorized_budget,
            preconditions=NO_GIT_CHECK,
        )
        return mgr, record

    def _owned_snapshot(self, record):
        return LiveProcessSnapshot(
            pid=record.pid,
            start_timestamp=record.launched_at,
            command_fingerprint=record.command_fingerprint,
            cwd=record.cwd,
        )

    def test_within_budget_no_action(self, ledger_path, authorized_budget):
        runner = FakeRunner()
        mgr, record = self._running_job(ledger_path, authorized_budget, runner)
        result = mgr.enforce_wall_clock(
            record, elapsed_sec=10, max_wall_clock_sec=3600,
            live_snapshot=self._owned_snapshot(record), grace_expired=False,
        )
        assert result.state == s.RUNNING
        assert not runner._stopped
        assert not runner._force_terminated

    def test_no_limit_configured_never_times_out(self, ledger_path, authorized_budget):
        runner = FakeRunner()
        mgr, record = self._running_job(ledger_path, authorized_budget, runner)
        result = mgr.enforce_wall_clock(
            record, elapsed_sec=999999, max_wall_clock_sec=None,
            live_snapshot=self._owned_snapshot(record), grace_expired=True,
        )
        assert result.state == s.RUNNING

    def test_timeout_graceful_stop_requested_first(self, ledger_path, authorized_budget):
        runner = FakeRunner()
        mgr, record = self._running_job(ledger_path, authorized_budget, runner)
        result = mgr.enforce_wall_clock(
            record, elapsed_sec=7200, max_wall_clock_sec=3600,
            live_snapshot=self._owned_snapshot(record), grace_expired=False,
        )
        assert record.job_id in runner._stopped
        assert result.state == s.RUNNING  # still waiting out the grace period

    def test_timeout_escalates_to_force_terminate_if_still_running(self, ledger_path, authorized_budget):
        runner = FakeRunner()
        mgr, record = self._running_job(ledger_path, authorized_budget, runner)
        runner.script_poll_sequence(record.job_id, [RunnerStatus(is_running=True)])
        result = mgr.enforce_wall_clock(
            record, elapsed_sec=7200, max_wall_clock_sec=3600,
            live_snapshot=self._owned_snapshot(record), grace_expired=True,
        )
        assert record.job_id in runner._force_terminated
        assert result.state == s.TIMED_OUT

    def test_timeout_no_force_terminate_if_runner_stopped_in_time(self, ledger_path, authorized_budget):
        runner = FakeRunner()
        mgr, record = self._running_job(ledger_path, authorized_budget, runner)
        runner.script_poll_sequence(record.job_id, [RunnerStatus(is_running=False, is_complete=True)])
        result = mgr.enforce_wall_clock(
            record, elapsed_sec=7200, max_wall_clock_sec=3600,
            live_snapshot=self._owned_snapshot(record), grace_expired=True,
        )
        assert record.job_id not in runner._force_terminated
        assert result.state == s.TIMED_OUT

    def test_unowned_process_never_touched_but_still_timed_out(self, ledger_path, authorized_budget):
        """Live snapshot doesn't exist at all -- nothing to terminate, but
        the job is still correctly marked TIMED_OUT."""
        runner = FakeRunner()
        mgr, record = self._running_job(ledger_path, authorized_budget, runner)
        result = mgr.enforce_wall_clock(
            record, elapsed_sec=7200, max_wall_clock_sec=3600,
            live_snapshot=None, grace_expired=True,
        )
        assert not runner._stopped
        assert not runner._force_terminated
        assert result.state == s.TIMED_OUT

    def test_pid_reused_by_different_process_not_terminated(self, ledger_path, authorized_budget):
        """A live process exists at the recorded PID, but its identity
        doesn't match (PID reuse) -- must not be killed."""
        runner = FakeRunner()
        mgr, record = self._running_job(ledger_path, authorized_budget, runner)
        mismatched = LiveProcessSnapshot(
            pid=record.pid, start_timestamp="1999-01-01T00:00:00+00:00",
            command_fingerprint="totally-unrelated-process", cwd="/somewhere/else",
        )
        result = mgr.enforce_wall_clock(
            record, elapsed_sec=7200, max_wall_clock_sec=3600,
            live_snapshot=mismatched, grace_expired=True,
        )
        assert not runner._force_terminated
        assert result.state == s.TIMED_OUT

    def test_ambiguous_ownership_raises_not_guesses(self, ledger_path, authorized_budget):
        runner = FakeRunner()
        mgr, record = self._running_job(ledger_path, authorized_budget, runner)
        ambiguous = LiveProcessSnapshot(
            pid=record.pid, start_timestamp=None, command_fingerprint=None, cwd=None, exists=True,
        )
        with pytest.raises(ProcessOwnershipError):
            mgr.enforce_wall_clock(
                record, elapsed_sec=7200, max_wall_clock_sec=3600,
                live_snapshot=ambiguous, grace_expired=True,
            )


class TestPauseResume:
    def test_pause_then_confirm(self, ledger_path, authorized_budget):
        runner = FakeRunner()
        mgr = _manager(ledger_path, runner=runner)
        record = mgr.launch(
            experiment_id="EXP-9003", spec={}, budget_config=authorized_budget,
            preconditions=NO_GIT_CHECK,
        )
        record = mgr.request_pause(record)
        assert record.state == s.PAUSING
        assert record.job_id in runner._stopped
        ckpt = CheckpointRef(
            path="x", step=1, seed=42, config_hash="c", dataset_manifest_hash="d",
            code_commit_hash="e", created_at="2026-01-01T00:00:00+00:00", experiment_id="EXP-9003",
        )
        runner.set_checkpoint(record.job_id, ckpt.to_dict())
        record = mgr.confirm_pause(record)
        assert record.state == s.PAUSED
        assert record.checkpoint_ref == ckpt.to_dict()

    def test_resume_with_incompatible_checkpoint_raises(self, ledger_path, authorized_budget):
        runner = FakeRunner()
        mgr = _manager(ledger_path, runner=runner)
        record = mgr.launch(
            experiment_id="EXP-9003", spec={}, budget_config=authorized_budget,
            preconditions=NO_GIT_CHECK,
        )
        record = mgr.request_pause(record)
        ckpt = CheckpointRef(
            path="x", step=1, seed=42, config_hash="c", dataset_manifest_hash="d",
            code_commit_hash="e", created_at="2026-01-01T00:00:00+00:00", experiment_id="EXP-9003",
        )
        runner.set_checkpoint(record.job_id, ckpt.to_dict())
        record = mgr.confirm_pause(record)
        with pytest.raises(IncompatibleCheckpointError):
            mgr.resume(
                record, expected_config_hash="DIFFERENT", expected_dataset_manifest_hash="d",
                expected_code_commit_hash="e", expected_seed=42,
            )
        assert s.load_job(record.job_id).state == s.PAUSED

    def test_resume_with_compatible_checkpoint_succeeds(self, ledger_path, authorized_budget):
        runner = FakeRunner()
        mgr = _manager(ledger_path, runner=runner)
        record = mgr.launch(
            experiment_id="EXP-9003", spec={}, budget_config=authorized_budget,
            preconditions=NO_GIT_CHECK,
        )
        record = mgr.request_pause(record)
        ckpt = CheckpointRef(
            path="x", step=1, seed=42, config_hash="c", dataset_manifest_hash="d",
            code_commit_hash="e", created_at="2026-01-01T00:00:00+00:00", experiment_id="EXP-9003",
        )
        runner.set_checkpoint(record.job_id, ckpt.to_dict())
        record = mgr.confirm_pause(record)
        record = mgr.resume(
            record, expected_config_hash="c", expected_dataset_manifest_hash="d",
            expected_code_commit_hash="e", expected_seed=42,
        )
        assert record.state == s.RUNNING
        assert runner._resumed[record.job_id] == ckpt.to_dict()


class TestCancel:
    def test_cancel_owned_process_stops_then_cancels(self, ledger_path, authorized_budget):
        runner = FakeRunner()
        mgr = _manager(ledger_path, runner=runner)
        record = mgr.launch(
            experiment_id="EXP-9003", spec={}, budget_config=authorized_budget,
            preconditions=NO_GIT_CHECK,
        )
        owned = LiveProcessSnapshot(
            pid=record.pid, start_timestamp=record.launched_at,
            command_fingerprint=record.command_fingerprint, cwd=record.cwd,
        )
        runner.script_poll_sequence(record.job_id, [RunnerStatus(is_running=True)])
        record = mgr.request_cancel(record, reason="operator stop", live_snapshot=owned, grace_expired=True)
        assert record.state == s.CANCELLED
        assert record.job_id in runner._stopped
        assert record.job_id in runner._force_terminated

    def test_cancel_never_touches_unowned_process(self, ledger_path, authorized_budget):
        runner = FakeRunner()
        mgr = _manager(ledger_path, runner=runner)
        record = mgr.launch(
            experiment_id="EXP-9003", spec={}, budget_config=authorized_budget,
            preconditions=NO_GIT_CHECK,
        )
        record = mgr.request_cancel(record, reason="operator stop", live_snapshot=None, grace_expired=True)
        assert record.state == s.CANCELLED
        assert not runner._stopped
        assert not runner._force_terminated


class TestFinalize:
    def test_finalize_completed(self, ledger_path, authorized_budget):
        runner = FakeRunner()
        mgr = _manager(ledger_path, runner=runner)
        record = mgr.launch(
            experiment_id="EXP-9003", spec={}, budget_config=authorized_budget,
            preconditions=NO_GIT_CHECK,
        )
        record = mgr.finalize(record, outcome=s.COMPLETED, elapsed_sec=100)
        assert record.state == s.COMPLETED
        with pytest.raises(s.JobStateError):
            mgr.finalize(record, outcome=s.FAILED, elapsed_sec=1)  # terminal, immutable

    def test_finalize_failed_preserved(self, ledger_path, authorized_budget):
        runner = FakeRunner()
        mgr = _manager(ledger_path, runner=runner)
        record = mgr.launch(
            experiment_id="EXP-9003", spec={}, budget_config=authorized_budget,
            preconditions=NO_GIT_CHECK,
        )
        record = mgr.finalize(record, outcome=s.FAILED, elapsed_sec=50, reason="runner reported failure")
        assert record.state == s.FAILED
        assert record.stopped_reason == "runner reported failure"
        assert s.load_job(record.job_id).state == s.FAILED

    def test_finalize_rejects_non_terminal_outcome(self, ledger_path, authorized_budget):
        runner = FakeRunner()
        mgr = _manager(ledger_path, runner=runner)
        record = mgr.launch(
            experiment_id="EXP-9003", spec={}, budget_config=authorized_budget,
            preconditions=NO_GIT_CHECK,
        )
        with pytest.raises(JobManagerError):
            mgr.finalize(record, outcome=s.RUNNING, elapsed_sec=1)


class TestCrashRecovery:
    def test_no_duplicate_launch_on_recovery_scan(self, ledger_path, authorized_budget):
        runner = FakeRunner()
        mgr = _manager(ledger_path, runner=runner)
        record = mgr.launch(
            experiment_id="EXP-9003", spec={}, budget_config=authorized_budget,
            preconditions=NO_GIT_CHECK,
        )
        launches_before = len(runner._launched)
        classifications = mgr.crash_recovery_scan({record.job_id: True})
        assert classifications[record.job_id] == s.RecoveryClassification.STILL_RUNNING_OWNED
        assert len(runner._launched) == launches_before  # never re-launched

    def test_ambiguous_job_requires_human_intervention(self, ledger_path, authorized_budget):
        runner = FakeRunner()
        mgr = _manager(ledger_path, runner=runner)
        record = mgr.launch(
            experiment_id="EXP-9003", spec={}, budget_config=authorized_budget,
            preconditions=NO_GIT_CHECK,
        )
        classifications = mgr.crash_recovery_scan({})
        assert classifications[record.job_id] == s.RecoveryClassification.AMBIGUOUS
        assert record.job_id in requires_human_intervention(classifications)

    def test_terminal_job_not_flagged_for_intervention(self, ledger_path, authorized_budget):
        runner = FakeRunner()
        mgr = _manager(ledger_path, runner=runner)
        record = mgr.launch(
            experiment_id="EXP-9003", spec={}, budget_config=authorized_budget,
            preconditions=NO_GIT_CHECK,
        )
        mgr.finalize(record, outcome=s.COMPLETED, elapsed_sec=10)
        classifications = mgr.crash_recovery_scan({})
        assert record.job_id not in requires_human_intervention(classifications)
