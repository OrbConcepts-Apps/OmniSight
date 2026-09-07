"""Phase J -- GPU concurrency slot enforcement.

A single-GPU research machine has exactly one real GPU to share across
jobs; this module enforces `max_concurrent_gpu_jobs` (default 1) as a
persisted, restart-safe reservation set. Persistence matters specifically
for "stale reservation recovered safely after restart" (Phase J real-runner
authorization section 16): if OmniLab crashes while a GPU job is running,
the slot must not silently vanish (which would let a second GPU job start
concurrently) or stay permanently stuck (which would starve every future
GPU job forever) -- `reconcile()` resolves this against a caller-supplied
set of jobs verified STILL_RUNNING_OWNED after a crash-recovery scan.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from research.config import REPO_ROOT

SLOTS_PATH = REPO_ROOT / "research" / "execution_jobs" / "gpu_slots.json"

DEFAULT_MAX_CONCURRENT_GPU_JOBS = 1


class GpuSlotError(RuntimeError):
    """Raised when a GPU slot cannot be acquired -- fail closed, never
    silently oversubscribe the single real GPU."""


@dataclass
class GpuSlotState:
    max_concurrent_gpu_jobs: int = DEFAULT_MAX_CONCURRENT_GPU_JOBS
    held_by: list = field(default_factory=list)  # list[str] job_ids

    def to_dict(self) -> dict:
        return {"max_concurrent_gpu_jobs": self.max_concurrent_gpu_jobs, "held_by": list(self.held_by)}

    @classmethod
    def from_dict(cls, data: dict) -> "GpuSlotState":
        return cls(
            max_concurrent_gpu_jobs=data.get("max_concurrent_gpu_jobs", DEFAULT_MAX_CONCURRENT_GPU_JOBS),
            held_by=list(data.get("held_by", [])),
        )


class GpuSlotManager:
    def __init__(self, path: Path = SLOTS_PATH, max_concurrent_gpu_jobs: int = DEFAULT_MAX_CONCURRENT_GPU_JOBS):
        self.path = path
        self._default_max = max_concurrent_gpu_jobs

    def _load(self) -> GpuSlotState:
        if not self.path.exists():
            return GpuSlotState(max_concurrent_gpu_jobs=self._default_max)
        try:
            return GpuSlotState.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            # Corrupt slot file -- fail closed by treating it as fully held
            # (max_concurrent_gpu_jobs=0 effectively refuses new
            # acquisitions) rather than silently resetting to "empty",
            # which could let an oversubscription slip through unnoticed.
            return GpuSlotState(max_concurrent_gpu_jobs=0, held_by=["<corrupt-slot-file>"])

    def _save(self, state: GpuSlotState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def acquire(self, job_id: str) -> None:
        """Raises GpuSlotError if no slot is available. Idempotent if
        `job_id` already holds a slot (does not double-count)."""
        state = self._load()
        if job_id in state.held_by:
            return
        if len(state.held_by) >= state.max_concurrent_gpu_jobs:
            raise GpuSlotError(
                f"refusing GPU slot for {job_id!r}: {len(state.held_by)} of "
                f"{state.max_concurrent_gpu_jobs} concurrent GPU job slot(s) already held "
                f"({state.held_by!r})."
            )
        state.held_by.append(job_id)
        self._save(state)

    def release(self, job_id: str) -> None:
        """Always safe to call, including for a job that never held a slot
        (e.g. a launch that failed before acquire()) -- releasing on every
        terminal path (Phase J real-runner authorization section 15) must
        never itself raise."""
        state = self._load()
        if job_id in state.held_by:
            state.held_by.remove(job_id)
            self._save(state)

    def is_held(self, job_id: str) -> bool:
        return job_id in self._load().held_by

    def held_jobs(self) -> list:
        return list(self._load().held_by)

    def reconcile(self, verified_still_running_job_ids: set) -> list:
        """Restart-safety: drop any held slot whose job_id is NOT in
        `verified_still_running_job_ids` (the caller's crash-recovery scan
        result -- see JobManager.crash_recovery_scan). A slot held by a job
        that is no longer verifiably running is a stale reservation and is
        released; a slot held by a job that IS still running is correctly
        preserved (not double-released). Returns the list of job_ids whose
        stale reservations were released."""
        state = self._load()
        stale = [j for j in state.held_by if j not in verified_still_running_job_ids]
        if stale:
            state.held_by = [j for j in state.held_by if j in verified_still_running_job_ids]
            self._save(state)
        return stale
