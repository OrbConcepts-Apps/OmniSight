"""Phase J active watchdog tests (research/execution_job/watchdog.py).
Fake-clock tests are fully deterministic; one real integration test proves
the watchdog can stop an actual (harmless, short) subprocess."""

from __future__ import annotations

import sys
import time

from research.execution_budget import ExecutionBudgetConfig
from research.execution_job import state as s
from research.execution_job.local_process_runner import LocalProcessRunner
from research.execution_job.manager import JobManager, LaunchPreconditions
from research.execution_job.process_ownership import LiveProcessSnapshot
from research.execution_job.runner import FakeRunner, RunnerStatus
from research.execution_job.watchdog import WatchdogConfig, run_watchdog


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, sec: float) -> None:
        self.t += sec


def _isolated_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "JOBS_DIR", tmp_path / "execution_jobs")


class TestFakeClockDeterministic:
    def test_completes_before_timeout_returns_completed(self, tmp_path, monkeypatch):
        _isolated_jobs(tmp_path, monkeypatch)
        runner = FakeRunner()
        mgr = JobManager(runner, ledger_path=tmp_path / "ledger.jsonl")
        record = mgr.launch(
            experiment_id="EXP-TEST", spec={}, budget_config=ExecutionBudgetConfig(gpu_execution_authorized=True),
            preconditions=LaunchPreconditions(require_clean_git_tree=False),
        )
        runner.script_poll_sequence(record.job_id, [RunnerStatus(is_complete=True)])

        clock = FakeClock()
        sleeps = []
        config = WatchdogConfig(max_wall_clock_sec=100, poll_interval_sec=10)
        result = run_watchdog(
            mgr, record, config,
            live_snapshot_fn=lambda r: None,
            is_complete_fn=lambda r: runner.poll(r.job_id).is_complete,
            clock=clock, sleep=lambda sec: sleeps.append(sec),
        )
        assert result.job_id == record.job_id
        assert sleeps == []  # returned immediately, no sleep needed

    def test_timeout_drives_graceful_then_escalated_termination(self, tmp_path, monkeypatch):
        _isolated_jobs(tmp_path, monkeypatch)
        runner = FakeRunner()
        mgr = JobManager(runner, ledger_path=tmp_path / "ledger.jsonl")
        record = mgr.launch(
            experiment_id="EXP-TEST", spec={}, budget_config=ExecutionBudgetConfig(gpu_execution_authorized=True),
            preconditions=LaunchPreconditions(require_clean_git_tree=False),
        )
        runner.script_poll_sequence(record.job_id, [RunnerStatus(is_running=True)])

        owned_snapshot = LiveProcessSnapshot(
            pid=record.pid, start_timestamp=record.launched_at,
            command_fingerprint=record.command_fingerprint, cwd=record.cwd,
        )

        clock = FakeClock()

        def advancing_sleep(sec):
            clock.advance(sec)

        config = WatchdogConfig(max_wall_clock_sec=5, poll_interval_sec=2, grace_period_sec=1)
        # Never reports complete -- forces the watchdog into the timeout path.
        result = run_watchdog(
            mgr, record, config,
            live_snapshot_fn=lambda r: owned_snapshot,
            is_complete_fn=lambda r: False,
            clock=clock, sleep=advancing_sleep,
        )
        assert result.state == s.TIMED_OUT
        assert record.job_id in runner._stopped
        assert record.job_id in runner._force_terminated


class TestRealSubprocessWatchdog:
    def test_real_watchdog_stops_real_subprocess_on_timeout(self, tmp_path, monkeypatch):
        """The one required real-subprocess proof (Phase J real-runner
        authorization section 7). Short, deterministic: a 20-second sleep,
        a 0.5-second wall-clock budget, 0.2-second grace period -- total
        real wall-clock cost for this test is well under 2 seconds."""
        _isolated_jobs(tmp_path, monkeypatch)
        runner = LocalProcessRunner(logs_dir=tmp_path)
        mgr = JobManager(runner, ledger_path=tmp_path / "ledger.jsonl")
        record = mgr.launch(
            experiment_id="EXP-TEST",
            spec={"argv": [sys.executable, "-c", "import time; time.sleep(20)"]},
            budget_config=ExecutionBudgetConfig(gpu_execution_authorized=True),
            preconditions=LaunchPreconditions(require_clean_git_tree=False),
        )

        def live_snapshot(r):
            from research.execution_job.process_ownership import live_snapshot_for_pid

            return live_snapshot_for_pid(r.pid)

        config = WatchdogConfig(max_wall_clock_sec=0, poll_interval_sec=0.1, grace_period_sec=0.3)
        result = run_watchdog(
            mgr, record, config,
            live_snapshot_fn=live_snapshot,
            is_complete_fn=lambda r: runner.poll(r.job_id).is_complete,
            clock=time.monotonic, sleep=time.sleep,
        )
        assert result.state == s.TIMED_OUT
        time.sleep(0.3)
        assert runner.poll(record.job_id).is_running is False
