"""Phase J -- multi-seed bookkeeping, cherry-pick-proof by construction.

This module does NOT compute any scientific verdict (that stays the
reviewer's/evaluation-policy's job, per research/experiment_validator.py
and research/evaluation_policy.py). It guarantees that whatever downstream
aggregation a future experiment performs, it is handed every seed's result
-- attempted, completed, or failed -- never a pre-filtered favorable subset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SeedPlan:
    """An immutable, ordered seed list pre-registered before any job in the
    plan is launched. Per Phase J authorization section 12, this must be
    fixed up front -- never extended after seeing early results."""

    experiment_id: str
    seeds: tuple

    def __post_init__(self):
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"SeedPlan for {self.experiment_id!r}: duplicate seeds in {self.seeds!r}")
        if len(self.seeds) == 0:
            raise ValueError(f"SeedPlan for {self.experiment_id!r}: seed list must be non-empty")


@dataclass(frozen=True)
class SeedRunResult:
    seed: int
    job_id: str
    outcome: str  # "COMPLETED" | "FAILED" | "TIMED_OUT" | "CANCELLED" | "PENDING"
    metrics: Optional[dict] = None


class CherryPickError(RuntimeError):
    """Raised when an aggregation input set does not account for every seed
    in the plan -- the one failure mode this module exists to prevent."""


@dataclass(frozen=True)
class SeedAggregateReport:
    experiment_id: str
    total_seeds: int
    completed: tuple
    failed: tuple
    pending: tuple

    @property
    def all_accounted_for(self) -> bool:
        return len(self.completed) + len(self.failed) + len(self.pending) == self.total_seeds


def aggregate_seed_results(plan: SeedPlan, results: list) -> SeedAggregateReport:
    """Builds the full-disclosure report over `results` against `plan`.
    Raises CherryPickError if `results` contains a result for a seed not in
    the plan, more than one result for the same seed, or is missing a
    result for a seed that is neither reported as PENDING nor present at
    all -- every seed in the plan must appear exactly once."""
    seen: dict = {}
    for r in results:
        if r.seed not in plan.seeds:
            raise CherryPickError(
                f"{plan.experiment_id}: result for seed {r.seed} is not in the pre-registered "
                f"seed plan {plan.seeds!r} -- refusing to aggregate an unplanned seed."
            )
        if r.seed in seen:
            raise CherryPickError(
                f"{plan.experiment_id}: duplicate result for seed {r.seed} -- exactly one "
                "result per seed is required."
            )
        seen[r.seed] = r

    missing = [s for s in plan.seeds if s not in seen]
    if missing:
        raise CherryPickError(
            f"{plan.experiment_id}: missing result(s) for seed(s) {missing} -- every "
            "pre-registered seed must be accounted for (as COMPLETED, FAILED, or PENDING) "
            "before aggregation; a report built from only the seeds that happened to finish "
            "is exactly the cherry-picking failure mode this check exists to prevent."
        )

    completed = tuple(r for r in results if r.outcome == "COMPLETED")
    failed = tuple(r for r in results if r.outcome in ("FAILED", "TIMED_OUT", "CANCELLED"))
    pending = tuple(r for r in results if r.outcome == "PENDING")

    return SeedAggregateReport(
        experiment_id=plan.experiment_id,
        total_seeds=len(plan.seeds),
        completed=completed,
        failed=failed,
        pending=pending,
    )
