"""Phase J resource-accounting ledger tests
(research/execution_job/accounting.py). No real GPU telemetry involved --
elapsed_sec values here are synthetic test fixtures."""

from __future__ import annotations

from research.execution_job import accounting as acc


class TestAppendOnlyLedger:
    def test_record_and_read_roundtrip(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        acc.record_event(job_id="JOB-0001", experiment_id="EXP-9001", event="STARTED", ledger_path=path)
        acc.record_event(
            job_id="JOB-0001", experiment_id="EXP-9001", event="FINISHED",
            elapsed_sec=120.5, termination_reason="completed", state="COMPLETED", ledger_path=path,
        )
        entries = acc.read_ledger(path)
        assert len(entries) == 2
        assert entries[0]["event"] == "STARTED"
        assert entries[1]["elapsed_sec"] == 120.5

    def test_never_measured_gpu_active_sec_stays_none(self, tmp_path):
        """Per Phase J authorization section 10: never fabricate precision
        telemetry cannot support."""
        path = tmp_path / "ledger.jsonl"
        entry = acc.record_event(job_id="JOB-0001", experiment_id="EXP-9001", event="FINISHED", elapsed_sec=10, ledger_path=path)
        assert entry.gpu_active_sec is None
        assert acc.read_ledger(path)[0]["gpu_active_sec"] is None

    def test_appending_never_truncates_prior_entries(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        for i in range(5):
            acc.record_event(job_id=f"JOB-{i:04d}", experiment_id="EXP-9001", event="STARTED", ledger_path=path)
        assert len(acc.read_ledger(path)) == 5

    def test_corrupt_line_skipped_not_fatal(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        acc.record_event(job_id="JOB-0001", experiment_id="EXP-9001", event="STARTED", ledger_path=path)
        with path.open("a", encoding="utf-8") as f:
            f.write("{not valid json\n")
        acc.record_event(job_id="JOB-0002", experiment_id="EXP-9001", event="STARTED", ledger_path=path)
        assert len(acc.read_ledger(path)) == 2

    def test_missing_ledger_file_returns_empty(self, tmp_path):
        assert acc.read_ledger(tmp_path / "does_not_exist.jsonl") == []


class TestCumulativeWallClock:
    def test_sums_finished_entries_only(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        acc.record_event(job_id="JOB-1", experiment_id="EXP-9001", event="STARTED", ledger_path=path)
        acc.record_event(job_id="JOB-1", experiment_id="EXP-9001", event="FINISHED", elapsed_sec=100, ledger_path=path)
        acc.record_event(job_id="JOB-2", experiment_id="EXP-9001", event="FINISHED", elapsed_sec=200, ledger_path=path)
        assert acc.cumulative_wall_clock_sec(experiment_id="EXP-9001", ledger_path=path) == 300

    def test_filters_by_experiment_id(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        acc.record_event(job_id="JOB-1", experiment_id="EXP-A", event="FINISHED", elapsed_sec=100, ledger_path=path)
        acc.record_event(job_id="JOB-2", experiment_id="EXP-B", event="FINISHED", elapsed_sec=999, ledger_path=path)
        assert acc.cumulative_wall_clock_sec(experiment_id="EXP-A", ledger_path=path) == 100

    def test_filters_by_date(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        acc.record_event(job_id="JOB-1", experiment_id="EXP-A", event="FINISHED", elapsed_sec=50, ledger_path=path)
        today = acc.read_ledger(path)[0]["at"][:10]
        assert acc.cumulative_wall_clock_sec(on_date=today, ledger_path=path) == 50
        assert acc.cumulative_wall_clock_sec(on_date="1999-01-01", ledger_path=path) == 0
