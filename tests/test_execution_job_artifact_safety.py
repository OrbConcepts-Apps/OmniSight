"""Phase J output-directory collision tests
(research/execution_job/artifact_safety.py). No real experiment output."""

from __future__ import annotations

import pytest

from research.execution_job.artifact_safety import (
    OutputDirCollisionError,
    reserve_job_output_dir,
)


class TestReserveJobOutputDir:
    def test_first_reservation_creates_and_marks(self, tmp_path):
        out_dir = tmp_path / "JOB-0001"
        out = reserve_job_output_dir("JOB-0001", out_dir)
        assert out.exists()
        assert (out / ".omnilab_job_owner").exists()

    def test_same_job_id_reservation_is_idempotent(self, tmp_path):
        out_dir = tmp_path / "JOB-0001"
        reserve_job_output_dir("JOB-0001", out_dir)
        out2 = reserve_job_output_dir("JOB-0001", out_dir)  # must not raise
        assert out2.exists()

    def test_different_job_id_same_path_collision_blocked(self, tmp_path):
        shared_dir = tmp_path / "shared_output"
        reserve_job_output_dir("JOB-0001", shared_dir)
        with pytest.raises(OutputDirCollisionError):
            reserve_job_output_dir("JOB-0002", shared_dir)

    def test_unrecognized_existing_dir_blocked(self, tmp_path):
        """A directory that already exists but was never stamped by
        OmniLab (e.g. left over from something unrelated) must never be
        silently adopted as a job's output directory."""
        out_dir = tmp_path / "JOB-0001"
        out_dir.mkdir()
        with pytest.raises(OutputDirCollisionError):
            reserve_job_output_dir("JOB-0001", out_dir)
