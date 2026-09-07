"""Phase J real-integration tests: JobManager + LocalProcessRunner +
GpuSlotManager together, against real (harmless, short) subprocesses. No
GPU/CUDA is actually used -- "GPU-declared" here means the job's own
gpu_required flag and the concurrency-slot bookkeeping around it, exactly
as a future real training job would declare, without any CUDA workload."""

from __future__ import annotations

import sys
import time

import pytest

from research.execution_budget import ExecutionBudgetConfig, ExecutionBudgetError
from research.execution_job import accounting, state as s
from research.execution_job.gpu_slots import GpuSlotError, GpuSlotManager
from research.execution_job.local_process_runner import LocalProcessRunner
from research.execution_job.manager import JobManager, LaunchPreconditions
from research.execution_job.process_ownership import LiveProcessSnapshot, live_snapshot_for_pid


def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "JOBS_DIR", tmp_path / "execution_jobs")


NO_GIT = LaunchPreconditions(require_clean_git_tree=False)


class TestCpuVsGpuAuthorization:
    def test_cpu_only_job_does_not_require_gpu_authorization(self, tmp_path, monkeypatch):
        _isolated(tmp_path, monkeypatch)
        runner = LocalProcessRunner(logs_dir=tmp_path)
        mgr = JobManager(runner, ledger_path=tmp_path / "ledger.jsonl")
        record = mgr.launch(
            experiment_id="EXP-TEST", spec={"argv": [sys.executable, "-c", "print(1)"]},
            budget_config=ExecutionBudgetConfig(),  # gpu_execution_authorized=False (default)
            preconditions=NO_GIT, gpu_required=False,
        )
        assert record.state == s.RUNNING
        mgr.finalize(record, outcome=s.COMPLETED, elapsed_sec=0.1)

    def test_gpu_declared_job_requires_gpu_authorization(self, tmp_path, monkeypatch):
        _isolated(tmp_path, monkeypatch)
        runner = LocalProcessRunner(logs_dir=tmp_path)
        mgr = JobManager(runner, ledger_path=tmp_path / "ledger.jsonl")
        with pytest.raises(ExecutionBudgetError):
            mgr.launch(
                experiment_id="EXP-TEST", spec={"argv": [sys.executable, "-c", "print(1)"]},
                budget_config=ExecutionBudgetConfig(),  # not authorized
                preconditions=NO_GIT, gpu_required=True,
            )


class TestGpuConcurrency:
    def test_first_gpu_job_obtains_slot_second_rejected(self, tmp_path, monkeypatch):
        _isolated(tmp_path, monkeypatch)
        slots = GpuSlotManager(path=tmp_path / "slots.json", max_concurrent_gpu_jobs=1)
        runner = LocalProcessRunner(logs_dir=tmp_path)
        mgr = JobManager(runner, ledger_path=tmp_path / "ledger.jsonl", gpu_slots=slots)
        budget = ExecutionBudgetConfig(gpu_execution_authorized=True, max_concurrent_training_jobs=5)

        record1 = mgr.launch(
            experiment_id="EXP-A", spec={"argv": [sys.executable, "-c", "import time; time.sleep(2)"]},
            budget_config=budget, preconditions=NO_GIT, gpu_required=True,
        )
        assert slots.is_held(record1.job_id)

        with pytest.raises(Exception):  # GpuSlotError wrapped as JobManagerError
            mgr.launch(
                experiment_id="EXP-B", spec={"argv": [sys.executable, "-c", "print(1)"]},
                budget_config=budget, preconditions=NO_GIT, gpu_required=True,
            )

        mgr.finalize(record1, outcome=s.COMPLETED, elapsed_sec=0.1)
        assert not slots.is_held(record1.job_id)

        # Slot now free -- a second GPU job can proceed.
        record2 = mgr.launch(
            experiment_id="EXP-C", spec={"argv": [sys.executable, "-c", "print(1)"]},
            budget_config=budget, preconditions=NO_GIT, gpu_required=True,
        )
        assert slots.is_held(record2.job_id)
        mgr.finalize(record2, outcome=s.COMPLETED, elapsed_sec=0.1)

    def test_slot_released_on_abnormal_exit_via_cancel(self, tmp_path, monkeypatch):
        _isolated(tmp_path, monkeypatch)
        slots = GpuSlotManager(path=tmp_path / "slots.json", max_concurrent_gpu_jobs=1)
        runner = LocalProcessRunner(logs_dir=tmp_path)
        mgr = JobManager(runner, ledger_path=tmp_path / "ledger.jsonl", gpu_slots=slots)
        budget = ExecutionBudgetConfig(gpu_execution_authorized=True)
        record = mgr.launch(
            experiment_id="EXP-A", spec={"argv": [sys.executable, "-c", "import time; time.sleep(20)"]},
            budget_config=budget, preconditions=NO_GIT, gpu_required=True,
        )
        assert slots.is_held(record.job_id)
        owned = LiveProcessSnapshot(
            pid=record.pid, start_timestamp=record.launched_at,
            command_fingerprint=record.command_fingerprint, cwd=record.cwd,
        )
        mgr.request_cancel(record, reason="test cancel", live_snapshot=owned, grace_expired=True)
        assert not slots.is_held(record.job_id)

    def test_slot_released_on_prepare_failure(self, tmp_path, monkeypatch):
        """A job that never reaches RUNNING (fails at prepare()) must not
        leak its GPU slot -- release happens on the FAILED-transition path
        inside launch() itself, not only via finalize()."""
        _isolated(tmp_path, monkeypatch)
        slots = GpuSlotManager(path=tmp_path / "slots.json", max_concurrent_gpu_jobs=1)
        runner = LocalProcessRunner(logs_dir=tmp_path)
        mgr = JobManager(runner, ledger_path=tmp_path / "ledger.jsonl", gpu_slots=slots)
        budget = ExecutionBudgetConfig(gpu_execution_authorized=True)
        with pytest.raises(Exception):
            mgr.launch(
                experiment_id="EXP-A", spec={"argv": "not-a-list"},  # invalid -> prepare() raises
                budget_config=budget, preconditions=NO_GIT, gpu_required=True,
            )
        jobs = s.list_all_jobs()
        assert jobs[0].state == s.FAILED
        assert not slots.is_held(jobs[0].job_id)


class TestCancellationAndRestartRecovery:
    def test_cancellation_path_real_subprocess(self, tmp_path, monkeypatch):
        _isolated(tmp_path, monkeypatch)
        runner = LocalProcessRunner(logs_dir=tmp_path)
        mgr = JobManager(runner, ledger_path=tmp_path / "ledger.jsonl")
        record = mgr.launch(
            experiment_id="EXP-TEST", spec={"argv": [sys.executable, "-c", "import time; time.sleep(20)"]},
            budget_config=ExecutionBudgetConfig(gpu_execution_authorized=True), preconditions=NO_GIT,
        )
        owned = LiveProcessSnapshot(
            pid=record.pid, start_timestamp=record.launched_at,
            command_fingerprint=record.command_fingerprint, cwd=record.cwd,
        )
        record = mgr.request_cancel(record, reason="operator stop", live_snapshot=owned, grace_expired=True)
        assert record.state == s.CANCELLED
        time.sleep(0.2)
        assert runner.poll(record.job_id).is_running is False

    def test_restart_reattach_no_duplicate_launch_via_manager(self, tmp_path, monkeypatch):
        """Full manager-level version of section 12's A-G recovery flow."""
        _isolated(tmp_path, monkeypatch)
        original_runner = LocalProcessRunner(logs_dir=tmp_path)
        mgr = JobManager(original_runner, ledger_path=tmp_path / "ledger.jsonl")
        record = mgr.launch(
            experiment_id="EXP-TEST", spec={"argv": [sys.executable, "-c", "import time; time.sleep(3)"]},
            budget_config=ExecutionBudgetConfig(gpu_execution_authorized=True), preconditions=NO_GIT,
        )

        # Simulate OmniLab restart: fresh manager/runner instances, only
        # persisted state.json survives.
        fresh_runner = LocalProcessRunner(logs_dir=tmp_path)
        fresh_mgr = JobManager(fresh_runner, ledger_path=tmp_path / "ledger.jsonl")
        reloaded = s.load_job(record.job_id)

        fresh_runner.reattach(
            reloaded.job_id, pid=reloaded.pid,
            argv=[sys.executable, "-c", "import time; time.sleep(3)"],
            cwd=reloaded.cwd, creation_time=reloaded.launched_at,
        )
        live = live_snapshot_for_pid(reloaded.pid)
        owned = live.exists  # real ownership check via the real OS process

        classifications = fresh_mgr.crash_recovery_scan({reloaded.job_id: owned})
        assert classifications[reloaded.job_id] == s.RecoveryClassification.STILL_RUNNING_OWNED

        launches_before = len(fresh_runner._prepared)
        fresh_runner.poll(reloaded.job_id)  # reattach-based poll, no launch
        assert len(fresh_runner._prepared) == launches_before

        original_runner.force_terminate(record.job_id)  # cleanup


class TestResourceLedgerPersisted:
    def test_ledger_entries_persisted_for_real_job(self, tmp_path, monkeypatch):
        _isolated(tmp_path, monkeypatch)
        ledger_path = tmp_path / "ledger.jsonl"
        runner = LocalProcessRunner(logs_dir=tmp_path)
        mgr = JobManager(runner, ledger_path=ledger_path)
        record = mgr.launch(
            experiment_id="EXP-TEST", spec={"argv": [sys.executable, "-c", "print(1)"]},
            budget_config=ExecutionBudgetConfig(gpu_execution_authorized=True), preconditions=NO_GIT,
        )
        for _ in range(50):
            if not runner.poll(record.job_id).is_running:
                break
            time.sleep(0.05)
        mgr.finalize(record, outcome=s.COMPLETED, elapsed_sec=0.05)

        entries = accounting.read_ledger(ledger_path)
        events = {e["event"] for e in entries if e["job_id"] == record.job_id}
        assert events == {"STARTED", "FINISHED"}
        finished = [e for e in entries if e["job_id"] == record.job_id and e["event"] == "FINISHED"][0]
        assert finished["state"] == s.COMPLETED
