"""Deterministic GPU/runtime execution-budget framework (Phase-I-readiness
HIGH finding #4).

Phase H/this hardening pass does NOT implement autonomous training or
benchmarking -- that remains explicit human-gated territory (Phase F's
`new_training_approved` flag, still hard-coded False on every LLM-authored
proposal). This module is the AUTHORIZATION/BUDGET BOUNDARY future execution
code (a later phase's real training/benchmark runner) MUST cross before
consuming GPU/training resources -- built now, deliberately with nothing
wired to launch or kill a real GPU job, so that boundary exists in code
before any execution path can reach past it.

Fail-closed by construction: `require_execution_budget()` raises unless it
is given an explicit `ExecutionBudgetConfig` with `gpu_execution_authorized
=True` and the specific operation's resource estimate fits within every
configured limit. Missing configuration is refused, never defaulted to
"unlimited" or "assume it's fine".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ExecutionBudgetConfig:
    """Configurable limits for GPU/training-resource-consuming operations.
    Every field must be explicitly set by a human/operator -- there is no
    default instance implying "unlimited"; see require_execution_budget()'s
    fail-closed behavior when config is None."""

    # Human authorization gate -- distinct from Phase F's per-experiment
    # `new_training_approved` flag (that approves ONE specific experiment;
    # this authorizes GPU execution AT ALL for the current session/run).
    gpu_execution_authorized: bool = False

    max_wall_clock_sec_per_experiment: Optional[int] = None
    max_cumulative_runtime_sec_per_cycle: Optional[int] = None
    max_concurrent_training_jobs: Optional[int] = None

    # Phase J additions -- a generic long-running-job execution boundary
    # needs a per-JOB ceiling distinct from the pre-existing per-EXPERIMENT
    # one (an experiment may launch several jobs, e.g. one per seed), a
    # per-day GPU-runtime ceiling distinct from the pre-existing per-cycle
    # one, an explicit retry cap (default 0 -- see require_execution_budget's
    # retry_count check), and optional disk/RAM floors. All remain Optional
    # with no "unlimited" default; None simply means "this particular limit
    # is not being enforced", never inferred as permission.
    max_wall_clock_sec_per_job: Optional[int] = None
    max_cumulative_gpu_runtime_sec_per_day: Optional[int] = None
    max_retry_count: int = 0
    min_free_disk_gb: Optional[float] = None
    min_free_ram_gb: Optional[float] = None


@dataclass(frozen=True)
class ResourceEstimate:
    """A single operation's proposal-stage resource estimate -- e.g. what a
    training_data proposal's `compute_resource_estimate` field (Phase H
    schema-mapping fix) declared. Never a measured/observed value; this is
    what's being CHECKED against budget, not what actually happened."""

    estimated_wall_clock_sec: Optional[int] = None
    estimated_gpu_hours: Optional[float] = None


class ExecutionBudgetError(RuntimeError):
    """Raised when an execution-budget-consuming operation is refused --
    either because no budget configuration was supplied at all (fail
    closed), GPU execution was not explicitly authorized, no concurrent-job
    slot is available, or the operation's own resource estimate exceeds a
    configured limit."""


def require_execution_budget(
    operation: str,
    config: Optional[ExecutionBudgetConfig],
    *,
    estimate: Optional[ResourceEstimate] = None,
    current_cumulative_runtime_sec: int = 0,
    current_running_training_jobs: int = 0,
    current_cumulative_gpu_runtime_sec_today: int = 0,
    retry_count: int = 0,
    current_free_disk_gb: Optional[float] = None,
    current_free_ram_gb: Optional[float] = None,
    gpu_required: bool = True,
) -> None:
    """Fail-closed gate a future GPU/training-consuming operation MUST call
    before starting. Raises ExecutionBudgetError for any of:
      - config is None (no budget configuration exists for this operation
        at all -- refused, never treated as "unlimited")
      - config.gpu_execution_authorized is not True
      - current_running_training_jobs >= config.max_concurrent_training_jobs
        (when that limit is configured)
      - estimate.estimated_wall_clock_sec exceeds
        config.max_wall_clock_sec_per_experiment (when both are set)
      - current_cumulative_runtime_sec + the estimate's wall-clock exceeds
        config.max_cumulative_runtime_sec_per_cycle (when both are set)
    Never launches, monitors, or kills any real process -- this is purely
    the authorization/budget check a caller consults before doing so."""
    if config is None:
        raise ExecutionBudgetError(
            f"refusing {operation!r}: no ExecutionBudgetConfig supplied. GPU/training-"
            "resource-consuming operations require an explicit, human-configured budget "
            "-- there is no 'unlimited' default."
        )
    if gpu_required and not config.gpu_execution_authorized:
        raise ExecutionBudgetError(
            f"refusing {operation!r}: gpu_execution_authorized=False. GPU execution "
            "requires explicit human authorization, distinct from any per-experiment "
            "new_training_approved flag."
        )
    if config.max_concurrent_training_jobs is not None:
        if current_running_training_jobs >= config.max_concurrent_training_jobs:
            raise ExecutionBudgetError(
                f"refusing {operation!r}: {current_running_training_jobs} training job(s) "
                f"already running, at or above the configured limit of "
                f"{config.max_concurrent_training_jobs}."
            )
    if estimate is not None:
        if (
            estimate.estimated_wall_clock_sec is not None
            and config.max_wall_clock_sec_per_experiment is not None
            and estimate.estimated_wall_clock_sec > config.max_wall_clock_sec_per_experiment
        ):
            raise ExecutionBudgetError(
                f"refusing {operation!r}: estimated wall-clock "
                f"{estimate.estimated_wall_clock_sec}s exceeds the configured per-experiment "
                f"limit of {config.max_wall_clock_sec_per_experiment}s."
            )
        if (
            estimate.estimated_wall_clock_sec is not None
            and config.max_cumulative_runtime_sec_per_cycle is not None
            and (current_cumulative_runtime_sec + estimate.estimated_wall_clock_sec)
            > config.max_cumulative_runtime_sec_per_cycle
        ):
            raise ExecutionBudgetError(
                f"refusing {operation!r}: cumulative runtime "
                f"{current_cumulative_runtime_sec}s + estimated {estimate.estimated_wall_clock_sec}s "
                f"would exceed the configured per-cycle limit of "
                f"{config.max_cumulative_runtime_sec_per_cycle}s."
            )
        if (
            estimate.estimated_wall_clock_sec is not None
            and config.max_wall_clock_sec_per_job is not None
            and estimate.estimated_wall_clock_sec > config.max_wall_clock_sec_per_job
        ):
            raise ExecutionBudgetError(
                f"refusing {operation!r}: estimated wall-clock "
                f"{estimate.estimated_wall_clock_sec}s exceeds the configured per-job "
                f"limit of {config.max_wall_clock_sec_per_job}s."
            )
    if config.max_cumulative_gpu_runtime_sec_per_day is not None:
        estimated_sec = estimate.estimated_wall_clock_sec if estimate is not None else 0
        if (current_cumulative_gpu_runtime_sec_today + (estimated_sec or 0)) > config.max_cumulative_gpu_runtime_sec_per_day:
            raise ExecutionBudgetError(
                f"refusing {operation!r}: today's GPU runtime "
                f"{current_cumulative_gpu_runtime_sec_today}s + estimated {estimated_sec or 0}s "
                f"would exceed the configured per-day limit of "
                f"{config.max_cumulative_gpu_runtime_sec_per_day}s."
            )
    if retry_count > config.max_retry_count:
        raise ExecutionBudgetError(
            f"refusing {operation!r}: retry_count={retry_count} exceeds the configured "
            f"max_retry_count={config.max_retry_count} (default 0 -- automatic "
            "retry-from-scratch requires explicit future authorization, see Phase J "
            "authorization section 13)."
        )
    if config.min_free_disk_gb is not None:
        if current_free_disk_gb is None:
            raise ExecutionBudgetError(
                f"refusing {operation!r}: min_free_disk_gb={config.min_free_disk_gb} is "
                "configured but current_free_disk_gb was not supplied -- fail closed rather "
                "than assume disk space is fine."
            )
        if current_free_disk_gb < config.min_free_disk_gb:
            raise ExecutionBudgetError(
                f"refusing {operation!r}: {current_free_disk_gb}GB free disk is below the "
                f"configured floor of {config.min_free_disk_gb}GB."
            )
    if config.min_free_ram_gb is not None:
        if current_free_ram_gb is None:
            raise ExecutionBudgetError(
                f"refusing {operation!r}: min_free_ram_gb={config.min_free_ram_gb} is "
                "configured but current_free_ram_gb was not supplied -- fail closed rather "
                "than assume RAM is fine."
            )
        if current_free_ram_gb < config.min_free_ram_gb:
            raise ExecutionBudgetError(
                f"refusing {operation!r}: {current_free_ram_gb}GB free RAM is below the "
                f"configured floor of {config.min_free_ram_gb}GB."
            )
