"""Phase J job state machine tests (research/execution_job/state.py).
No real process/GPU/training anywhere in this file."""

from __future__ import annotations

import pytest

from research.execution_job import state as s


@pytest.fixture(autouse=True)
def _isolated_jobs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "JOBS_DIR", tmp_path / "execution_jobs")


class TestJobIdAllocation:
    def test_first_id_is_job_0001(self):
        record = s.create_job(experiment_id="EXP-9001")
        assert record.job_id == "JOB-0001"

    def test_ids_increment(self):
        s.create_job()
        s.create_job()
        r3 = s.create_job()
        assert r3.job_id == "JOB-0003"


class TestTransitions:
    def test_full_happy_path(self):
        r = s.create_job()
        r = s.transition(r, s.VALIDATED)
        r = s.transition(r, s.AUTHORIZED)
        r = s.transition(r, s.PREPARING)
        r = s.transition(r, s.RUNNING)
        r = s.transition(r, s.COMPLETED)
        assert r.state == s.COMPLETED
        assert len(r.history) == 5

    def test_illegal_transition_rejected(self):
        r = s.create_job()
        with pytest.raises(s.JobStateError):
            s.transition(r, s.RUNNING)  # cannot skip straight from CREATED

    def test_terminal_state_immutable(self):
        r = s.create_job()
        r = s.transition(r, s.VALIDATED)
        r = s.transition(r, s.AUTHORIZED)
        r = s.transition(r, s.PREPARING)
        r = s.transition(r, s.RUNNING)
        r = s.transition(r, s.CANCELLED, reason="test")
        with pytest.raises(s.JobStateError):
            s.transition(r, s.RUNNING)
        with pytest.raises(s.JobStateError):
            s.transition(r, s.COMPLETED)

    def test_unknown_state_rejected(self):
        r = s.create_job()
        with pytest.raises(s.JobStateError):
            s.transition(r, "NOT_A_REAL_STATE")

    def test_pause_resume_cycle(self):
        r = s.create_job()
        r = s.transition(r, s.VALIDATED)
        r = s.transition(r, s.AUTHORIZED)
        r = s.transition(r, s.PREPARING)
        r = s.transition(r, s.RUNNING)
        r = s.transition(r, s.PAUSING)
        r = s.transition(r, s.PAUSED)
        r = s.transition(r, s.RESUMING)
        r = s.transition(r, s.RUNNING)
        assert r.state == s.RUNNING

    def test_reason_persisted_for_terminal_stop_states(self):
        r = s.create_job()
        r = s.transition(r, s.VALIDATED, reason="")
        r = s.transition(r, s.FAILED, reason="budget exhausted")
        assert r.stopped_reason == "budget exhausted"


class TestPersistenceAndReload:
    def test_reload_matches_saved_state(self):
        r = s.create_job(experiment_id="EXP-9002", seed=42)
        r = s.transition(r, s.VALIDATED)
        reloaded = s.load_job(r.job_id)
        assert reloaded.state == s.VALIDATED
        assert reloaded.seed == 42
        assert reloaded.experiment_id == "EXP-9002"

    def test_load_missing_job_raises(self):
        with pytest.raises(s.JobStateError):
            s.load_job("JOB-9999")

    def test_corrupt_state_file_raises(self, tmp_path):
        job_dir = s.JOBS_DIR / "JOB-0001"
        job_dir.mkdir(parents=True)
        (job_dir / "state.json").write_text("{not valid json", encoding="utf-8")
        with pytest.raises(s.JobStateError):
            s.load_job("JOB-0001")

    def test_list_all_jobs_skips_corrupt_entries(self):
        s.create_job()
        corrupt_dir = s.JOBS_DIR / "JOB-9999"
        corrupt_dir.mkdir(parents=True)
        (corrupt_dir / "state.json").write_text("{not valid", encoding="utf-8")
        jobs = s.list_all_jobs()
        assert len(jobs) == 1


class TestCrashRecoveryClassification:
    def _running_job(self):
        r = s.create_job()
        r = s.transition(r, s.VALIDATED)
        r = s.transition(r, s.AUTHORIZED)
        r = s.transition(r, s.PREPARING)
        return s.transition(r, s.RUNNING)

    def test_running_and_verified_owned(self):
        r = self._running_job()
        cls = s.classify_for_recovery(r, process_is_verifiably_owned=True)
        assert cls == s.RecoveryClassification.STILL_RUNNING_OWNED

    def test_running_but_not_owned_without_checkpoint_is_non_resumable(self):
        r = self._running_job()
        cls = s.classify_for_recovery(r, process_is_verifiably_owned=False)
        assert cls == s.RecoveryClassification.NON_RESUMABLE

    def test_running_but_not_owned_with_checkpoint_is_resumable(self):
        r = self._running_job()
        r.checkpoint_ref = {"path": "x"}
        cls = s.classify_for_recovery(r, process_is_verifiably_owned=False)
        assert cls == s.RecoveryClassification.RESUMABLE

    def test_ambiguous_ownership_is_ambiguous(self):
        r = self._running_job()
        cls = s.classify_for_recovery(r, process_is_verifiably_owned=None)
        assert cls == s.RecoveryClassification.AMBIGUOUS

    def test_terminal_job_is_non_resumable(self):
        r = self._running_job()
        r = s.transition(r, s.COMPLETED)
        cls = s.classify_for_recovery(r, process_is_verifiably_owned=None)
        assert cls == s.RecoveryClassification.NON_RESUMABLE

    def test_never_launched_job_is_non_resumable(self):
        r = s.create_job()
        cls = s.classify_for_recovery(r, process_is_verifiably_owned=None)
        assert cls == s.RecoveryClassification.NON_RESUMABLE
