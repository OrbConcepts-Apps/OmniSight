"""Phase J checkpoint/resume compatibility tests
(research/execution_job/checkpoint.py). No real checkpoint file involved."""

from __future__ import annotations

import pytest

from research.execution_job.checkpoint import (
    CheckpointRef,
    IncompatibleCheckpointError,
    check_resume_compatibility,
)

VALID = CheckpointRef(
    path="research/execution_jobs/JOB-0001/ckpt.pt",
    step=10,
    seed=42,
    config_hash="cfg-abc",
    dataset_manifest_hash="ds-abc",
    code_commit_hash="commit-abc",
    created_at="2026-09-06T00:00:00+00:00",
    experiment_id="EXP-9003",
)


def _check(**overrides):
    kwargs = dict(
        checkpoint=VALID,
        expected_config_hash="cfg-abc",
        expected_dataset_manifest_hash="ds-abc",
        expected_code_commit_hash="commit-abc",
        expected_experiment_id="EXP-9003",
        expected_seed=42,
    )
    kwargs.update(overrides)
    check_resume_compatibility(**kwargs)


class TestCompatibleResume:
    def test_exact_match_does_not_raise(self):
        _check()  # must not raise


class TestIncompatibleResume:
    def test_missing_checkpoint_rejected(self):
        with pytest.raises(IncompatibleCheckpointError):
            _check(checkpoint=None)

    def test_config_hash_mismatch_rejected(self):
        with pytest.raises(IncompatibleCheckpointError) as exc:
            _check(expected_config_hash="cfg-different")
        assert "config_hash" in str(exc.value)

    def test_dataset_manifest_hash_mismatch_rejected(self):
        with pytest.raises(IncompatibleCheckpointError) as exc:
            _check(expected_dataset_manifest_hash="ds-different")
        assert "dataset_manifest_hash" in str(exc.value)

    def test_code_commit_hash_mismatch_rejected(self):
        with pytest.raises(IncompatibleCheckpointError) as exc:
            _check(expected_code_commit_hash="commit-different")
        assert "code_commit_hash" in str(exc.value)

    def test_experiment_id_mismatch_rejected(self):
        with pytest.raises(IncompatibleCheckpointError) as exc:
            _check(expected_experiment_id="EXP-0005")
        assert "experiment_id" in str(exc.value)

    def test_seed_mismatch_rejected(self):
        with pytest.raises(IncompatibleCheckpointError) as exc:
            _check(expected_seed=43)
        assert "seed" in str(exc.value)

    def test_all_mismatches_reported_together(self):
        with pytest.raises(IncompatibleCheckpointError) as exc:
            _check(
                expected_config_hash="x",
                expected_dataset_manifest_hash="y",
                expected_seed=999,
            )
        assert len(exc.value.reasons) == 3
