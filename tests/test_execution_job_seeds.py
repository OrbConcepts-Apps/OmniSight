"""Phase J multi-seed cherry-pick-proofing tests
(research/execution_job/seeds.py). No real training/seed execution."""

from __future__ import annotations

import pytest

from research.execution_job.seeds import (
    CherryPickError,
    SeedPlan,
    SeedRunResult,
    aggregate_seed_results,
)


class TestSeedPlan:
    def test_duplicate_seeds_rejected(self):
        with pytest.raises(ValueError):
            SeedPlan(experiment_id="EXP-9003", seeds=(42, 42, 43))

    def test_empty_seed_list_rejected(self):
        with pytest.raises(ValueError):
            SeedPlan(experiment_id="EXP-9003", seeds=())

    def test_valid_plan_constructs(self):
        plan = SeedPlan(experiment_id="EXP-9003", seeds=(42, 43, 44))
        assert plan.seeds == (42, 43, 44)


class TestAggregateSeedResults:
    def test_all_completed_preserved(self):
        plan = SeedPlan(experiment_id="EXP-9003", seeds=(42, 43, 44))
        results = [
            SeedRunResult(seed=42, job_id="JOB-1", outcome="COMPLETED", metrics={"recall": 0.25}),
            SeedRunResult(seed=43, job_id="JOB-2", outcome="COMPLETED", metrics={"recall": 0.24}),
            SeedRunResult(seed=44, job_id="JOB-3", outcome="COMPLETED", metrics={"recall": 0.26}),
        ]
        report = aggregate_seed_results(plan, results)
        assert len(report.completed) == 3
        assert report.all_accounted_for

    def test_failed_seed_preserved_not_dropped(self):
        plan = SeedPlan(experiment_id="EXP-9003", seeds=(42, 43, 44))
        results = [
            SeedRunResult(seed=42, job_id="JOB-1", outcome="COMPLETED", metrics={"recall": 0.25}),
            SeedRunResult(seed=43, job_id="JOB-2", outcome="FAILED"),
            SeedRunResult(seed=44, job_id="JOB-3", outcome="COMPLETED", metrics={"recall": 0.26}),
        ]
        report = aggregate_seed_results(plan, results)
        assert len(report.completed) == 2
        assert len(report.failed) == 1
        assert report.failed[0].seed == 43
        assert report.all_accounted_for

    def test_missing_seed_result_raises_cherry_pick_error(self):
        """Only 2 of 3 pre-registered seeds have results -- must refuse to
        aggregate a favorable-subset report."""
        plan = SeedPlan(experiment_id="EXP-9003", seeds=(42, 43, 44))
        results = [
            SeedRunResult(seed=42, job_id="JOB-1", outcome="COMPLETED", metrics={"recall": 0.30}),
            SeedRunResult(seed=43, job_id="JOB-2", outcome="COMPLETED", metrics={"recall": 0.31}),
        ]
        with pytest.raises(CherryPickError):
            aggregate_seed_results(plan, results)

    def test_unplanned_seed_rejected(self):
        plan = SeedPlan(experiment_id="EXP-9003", seeds=(42, 43))
        results = [
            SeedRunResult(seed=42, job_id="JOB-1", outcome="COMPLETED"),
            SeedRunResult(seed=43, job_id="JOB-2", outcome="COMPLETED"),
            SeedRunResult(seed=99, job_id="JOB-3", outcome="COMPLETED"),
        ]
        with pytest.raises(CherryPickError):
            aggregate_seed_results(plan, results)

    def test_duplicate_result_for_same_seed_rejected(self):
        plan = SeedPlan(experiment_id="EXP-9003", seeds=(42, 43))
        results = [
            SeedRunResult(seed=42, job_id="JOB-1", outcome="COMPLETED"),
            SeedRunResult(seed=42, job_id="JOB-1-retry", outcome="COMPLETED"),
            SeedRunResult(seed=43, job_id="JOB-2", outcome="COMPLETED"),
        ]
        with pytest.raises(CherryPickError):
            aggregate_seed_results(plan, results)

    def test_pending_seed_accounted_for(self):
        plan = SeedPlan(experiment_id="EXP-9003", seeds=(42, 43))
        results = [
            SeedRunResult(seed=42, job_id="JOB-1", outcome="COMPLETED"),
            SeedRunResult(seed=43, job_id="JOB-2", outcome="PENDING"),
        ]
        report = aggregate_seed_results(plan, results)
        assert len(report.pending) == 1
        assert report.all_accounted_for
