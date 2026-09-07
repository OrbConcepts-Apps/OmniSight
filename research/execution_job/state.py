"""Phase J -- generic long-running job state machine.

Same discipline as research/phase_i/candidate_state.py: one JSON state file
per job (research/execution_jobs/<JOB-ID>/state.json), one explicit
transition function, fail-closed on anything ambiguous, terminal states
immutable. This is deliberately generic (not tied to any one experiment
family) so a future training runner, a future benchmark runner, or any
other long-running OmniLab-owned process can reuse it.

States
------
CREATED    -- job id allocated, nothing else has happened.
VALIDATED  -- preconditions (spec freeze, integrity, dataset prerequisites)
              checked and passed; not yet authorized to consume resources.
AUTHORIZED -- execution-budget check passed (require_execution_budget did
              not raise); resources reserved, GPU work not yet started.
PREPARING  -- runner.prepare() in progress (write configs, stage inputs).
RUNNING    -- runner.launch() succeeded; process/job is live.
PAUSING    -- graceful-pause requested, waiting for runner to confirm.
PAUSED     -- runner confirmed a resumable paused state.
RESUMING   -- resume requested, waiting for runner to confirm restart.
COMPLETED  -- terminal: runner reported successful completion.
FAILED     -- terminal: runner reported failure, or crashed/errored.
CANCELLED  -- terminal: stopped by human/operational-gate action, not a
              failure of the job itself.
TIMED_OUT  -- terminal: wall-clock budget exceeded, job force-terminated.

COMPLETED/FAILED/CANCELLED/TIMED_OUT are the only terminals -- once reached,
no further transition is ever allowed. PAUSED is deliberately NOT terminal
(mirrors Phase I's BLOCKED): a paused job is expected to eventually resume
or be explicitly cancelled, never silently discarded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from research.config import REPO_ROOT

JOBS_DIR = REPO_ROOT / "research" / "execution_jobs"

CREATED = "CREATED"
VALIDATED = "VALIDATED"
AUTHORIZED = "AUTHORIZED"
PREPARING = "PREPARING"
RUNNING = "RUNNING"
PAUSING = "PAUSING"
PAUSED = "PAUSED"
RESUMING = "RESUMING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
CANCELLED = "CANCELLED"
TIMED_OUT = "TIMED_OUT"

ALL_STATES = frozenset({
    CREATED, VALIDATED, AUTHORIZED, PREPARING, RUNNING, PAUSING, PAUSED,
    RESUMING, COMPLETED, FAILED, CANCELLED, TIMED_OUT,
})

TERMINAL_STATES = frozenset({COMPLETED, FAILED, CANCELLED, TIMED_OUT})

# Explicit transition table -- anything not listed here is refused. Every
# non-terminal state can reach FAILED/CANCELLED (a crash or an operator stop
# can happen from almost anywhere) in addition to its "normal" next state(s).
ALLOWED_TRANSITIONS: dict[str, frozenset] = {
    CREATED: frozenset({VALIDATED, FAILED, CANCELLED}),
    VALIDATED: frozenset({AUTHORIZED, FAILED, CANCELLED}),
    AUTHORIZED: frozenset({PREPARING, FAILED, CANCELLED}),
    PREPARING: frozenset({RUNNING, FAILED, CANCELLED}),
    RUNNING: frozenset({PAUSING, COMPLETED, FAILED, CANCELLED, TIMED_OUT}),
    PAUSING: frozenset({PAUSED, FAILED, CANCELLED, TIMED_OUT}),
    PAUSED: frozenset({RESUMING, CANCELLED, FAILED}),
    RESUMING: frozenset({RUNNING, FAILED, CANCELLED}),
    COMPLETED: frozenset(),
    FAILED: frozenset(),
    CANCELLED: frozenset(),
    TIMED_OUT: frozenset(),
}


class JobStateError(RuntimeError):
    """Raised for an illegal transition, or when persisted state is
    inconsistent with what's on disk -- fail closed, never guess."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRecord:
    job_id: str
    experiment_id: str = ""
    state: str = CREATED
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)
    seed: Optional[int] = None
    gpu_required: bool = False
    pid: Optional[int] = None
    launched_at: Optional[str] = None
    command_fingerprint: str = ""
    cwd: str = ""
    executable: str = ""
    checkpoint_ref: Optional[dict] = None
    stopped_reason: str = ""
    history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "experiment_id": self.experiment_id,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "seed": self.seed,
            "gpu_required": self.gpu_required,
            "pid": self.pid,
            "launched_at": self.launched_at,
            "command_fingerprint": self.command_fingerprint,
            "cwd": self.cwd,
            "executable": self.executable,
            "checkpoint_ref": self.checkpoint_ref,
            "stopped_reason": self.stopped_reason,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "JobRecord":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _state_path(job_id: str) -> Path:
    return _job_dir(job_id) / "state.json"


def next_job_id() -> str:
    """Allocate the next JOB-NNNN id. Distinct namespace from EXP-XXXX,
    DRYRUN-NNNN, and CANDIDATE-NNNN -- a job is an execution-layer concept,
    never an experiment/candidate identity by itself."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(p.name for p in JOBS_DIR.iterdir() if p.is_dir() and p.name.startswith("JOB-"))
    n = len(existing) + 1
    return f"JOB-{n:04d}"


def create_job(experiment_id: str = "", seed: Optional[int] = None, gpu_required: bool = False) -> JobRecord:
    job_id = next_job_id()
    record = JobRecord(job_id=job_id, experiment_id=experiment_id, seed=seed, gpu_required=gpu_required)
    _save(record)
    return record


def load_job(job_id: str) -> JobRecord:
    path = _state_path(job_id)
    if not path.exists():
        raise JobStateError(f"no persisted state for {job_id!r} at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise JobStateError(f"{job_id}: state.json is corrupt/unreadable: {e}") from e
    return JobRecord.from_dict(data)


def list_all_jobs() -> list:
    """All persisted jobs, oldest first. A corrupt individual job's
    state.json is skipped (fail-closed for THAT job only)."""
    if not JOBS_DIR.exists():
        return []
    records = []
    for p in sorted(JOBS_DIR.iterdir()):
        if not (p.is_dir() and p.name.startswith("JOB-")):
            continue
        try:
            records.append(load_job(p.name))
        except JobStateError:
            continue
    return records


def _save(record: JobRecord) -> None:
    d = _job_dir(record.job_id)
    d.mkdir(parents=True, exist_ok=True)
    _state_path(record.job_id).write_text(
        json.dumps(record.to_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8"
    )


def transition(record: JobRecord, new_state: str, *, reason: str = "") -> JobRecord:
    """The only sanctioned way to change a job's state. Raises JobStateError
    for any transition not explicitly listed in ALLOWED_TRANSITIONS,
    including any transition attempted FROM a terminal state. Persists
    immediately (crash-safe)."""
    if new_state not in ALL_STATES:
        raise JobStateError(f"unknown state {new_state!r}")
    allowed = ALLOWED_TRANSITIONS.get(record.state, frozenset())
    if new_state not in allowed:
        raise JobStateError(
            f"{record.job_id}: illegal transition {record.state!r} -> {new_state!r} "
            f"(allowed from {record.state!r}: {sorted(allowed) or 'none (terminal)'})"
        )
    record.history.append({"from": record.state, "to": new_state, "at": _utcnow(), "reason": reason})
    record.state = new_state
    record.updated_at = _utcnow()
    if reason and new_state in (FAILED, CANCELLED, TIMED_OUT):
        record.stopped_reason = reason
    _save(record)
    return record


def save(record: JobRecord) -> None:
    """Persist the record without a state transition -- used for non-state
    field updates (pid, checkpoint_ref, ...) mid-stage."""
    _save(record)


class RecoveryClassification:
    STILL_RUNNING_OWNED = "STILL_RUNNING_OWNED"
    RESUMABLE = "RESUMABLE"
    NON_RESUMABLE = "NON_RESUMABLE"
    AMBIGUOUS = "AMBIGUOUS"


def classify_for_recovery(record: JobRecord, *, process_is_verifiably_owned: Optional[bool]) -> str:
    """Classify a persisted, non-terminal job at OmniLab restart time, per
    Phase J authorization section 18. `process_is_verifiably_owned` must be
    supplied by the caller (via process_ownership.verify_process_identity)
    for RUNNING/PAUSING/RESUMING jobs -- None means "could not be
    determined", which this function treats as AMBIGUOUS (fail closed,
    never guessed)."""
    if record.state in TERMINAL_STATES:
        return RecoveryClassification.NON_RESUMABLE  # already resolved, nothing to recover
    if record.state == PAUSED:
        return RecoveryClassification.RESUMABLE if record.checkpoint_ref else RecoveryClassification.NON_RESUMABLE
    if record.state in (RUNNING, PAUSING, RESUMING):
        if process_is_verifiably_owned is True:
            return RecoveryClassification.STILL_RUNNING_OWNED
        if process_is_verifiably_owned is False:
            return RecoveryClassification.RESUMABLE if record.checkpoint_ref else RecoveryClassification.NON_RESUMABLE
        return RecoveryClassification.AMBIGUOUS
    if record.state in (CREATED, VALIDATED, AUTHORIZED, PREPARING):
        # Never actually launched (or died before launch was confirmed) --
        # safe to treat as non-resumable; a fresh job should be created
        # rather than resuming something that never ran.
        return RecoveryClassification.NON_RESUMABLE
    return RecoveryClassification.AMBIGUOUS
