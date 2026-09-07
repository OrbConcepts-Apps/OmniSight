"""Phase J GPU concurrency slot tests (research/execution_job/gpu_slots.py).
No real GPU/CUDA involved -- pure bookkeeping."""

from __future__ import annotations

import pytest

from research.execution_job.gpu_slots import GpuSlotError, GpuSlotManager


class TestAcquireRelease:
    def test_first_job_obtains_slot(self, tmp_path):
        mgr = GpuSlotManager(path=tmp_path / "slots.json", max_concurrent_gpu_jobs=1)
        mgr.acquire("JOB-1")
        assert mgr.is_held("JOB-1")

    def test_second_gpu_job_rejected_while_first_active(self, tmp_path):
        mgr = GpuSlotManager(path=tmp_path / "slots.json", max_concurrent_gpu_jobs=1)
        mgr.acquire("JOB-1")
        with pytest.raises(GpuSlotError):
            mgr.acquire("JOB-2")

    def test_slot_released_after_completion(self, tmp_path):
        mgr = GpuSlotManager(path=tmp_path / "slots.json", max_concurrent_gpu_jobs=1)
        mgr.acquire("JOB-1")
        mgr.release("JOB-1")
        mgr.acquire("JOB-2")  # must not raise -- slot freed
        assert mgr.is_held("JOB-2")

    def test_release_of_unheld_job_is_a_safe_noop(self, tmp_path):
        mgr = GpuSlotManager(path=tmp_path / "slots.json", max_concurrent_gpu_jobs=1)
        mgr.release("JOB-NEVER-HELD")  # must not raise

    def test_acquire_idempotent_for_same_job(self, tmp_path):
        mgr = GpuSlotManager(path=tmp_path / "slots.json", max_concurrent_gpu_jobs=1)
        mgr.acquire("JOB-1")
        mgr.acquire("JOB-1")  # must not raise or double-count
        assert mgr.held_jobs() == ["JOB-1"]

    def test_higher_concurrency_limit_allows_multiple(self, tmp_path):
        mgr = GpuSlotManager(path=tmp_path / "slots.json", max_concurrent_gpu_jobs=2)
        mgr.acquire("JOB-1")
        mgr.acquire("JOB-2")  # must not raise
        with pytest.raises(GpuSlotError):
            mgr.acquire("JOB-3")


class TestPersistenceAcrossInstances:
    def test_state_persists_across_new_manager_instances(self, tmp_path):
        """Simulates a real restart: a fresh GpuSlotManager instance reads
        the same on-disk reservation."""
        path = tmp_path / "slots.json"
        GpuSlotManager(path=path, max_concurrent_gpu_jobs=1).acquire("JOB-1")
        fresh = GpuSlotManager(path=path, max_concurrent_gpu_jobs=1)
        assert fresh.is_held("JOB-1")
        with pytest.raises(GpuSlotError):
            fresh.acquire("JOB-2")


class TestStaleReservationRecovery:
    def test_reconcile_releases_stale_reservation(self, tmp_path):
        """JOB-1 held a slot before a crash; crash-recovery confirms it is
        NOT still running -- reconcile() must release the stale slot so a
        future job isn't starved forever."""
        path = tmp_path / "slots.json"
        mgr = GpuSlotManager(path=path, max_concurrent_gpu_jobs=1)
        mgr.acquire("JOB-1")
        released = mgr.reconcile(verified_still_running_job_ids=set())
        assert released == ["JOB-1"]
        assert not mgr.is_held("JOB-1")
        mgr.acquire("JOB-2")  # slot now available

    def test_reconcile_preserves_genuinely_running_job(self, tmp_path):
        path = tmp_path / "slots.json"
        mgr = GpuSlotManager(path=path, max_concurrent_gpu_jobs=1)
        mgr.acquire("JOB-1")
        released = mgr.reconcile(verified_still_running_job_ids={"JOB-1"})
        assert released == []
        assert mgr.is_held("JOB-1")


class TestCorruptSlotFileFailsClosed:
    def test_corrupt_file_refuses_new_acquisition(self, tmp_path):
        path = tmp_path / "slots.json"
        path.write_text("{not valid json", encoding="utf-8")
        mgr = GpuSlotManager(path=path, max_concurrent_gpu_jobs=1)
        with pytest.raises(GpuSlotError):
            mgr.acquire("JOB-1")
