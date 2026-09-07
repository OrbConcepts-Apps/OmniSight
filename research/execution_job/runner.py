"""Phase J -- generic long-running-job runner interface.

`Runner` is the abstraction any future execution engine (a real training
loop, a real long benchmark run) must implement to be driven by
`JobManager`. This module deliberately implements NO real training/GPU
invocation -- `FakeRunner` below is the only concrete implementation that
exists today, used exclusively by tests and by any future dry-run of the
manager's own control flow. A future phase that adds a real training runner
subclasses `Runner`; `JobManager` never needs to change to support it.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RunnerStatus:
    """A single poll's answer. `is_running`/`is_complete`/`is_failed` are
    mutually exclusive; a runner in a transient state (e.g. still starting
    up) reports is_running=False, is_complete=False, is_failed=False and
    `detail` explaining why -- never guessed as one of the three."""

    is_running: bool = False
    is_complete: bool = False
    is_failed: bool = False
    detail: str = ""
    checkpoint_ref: Optional[dict] = None


class RunnerError(RuntimeError):
    """Raised by a Runner method when it cannot honor the request (e.g.
    finalize() called before the job reached a terminal poll state)."""


class Runner(abc.ABC):
    """Interface every long-running-job backend must implement. No method
    here performs resource-BUDGET checks -- that is JobManager's job, via
    research.execution_budget, before prepare()/launch() are ever called."""

    @abc.abstractmethod
    def prepare(self, job_id: str, spec: dict) -> None:
        """Stage configuration/inputs for the job. Must not start any
        long-running work yet."""

    @abc.abstractmethod
    def launch(self, job_id: str) -> dict:
        """Start the job. Returns process-identity fields (pid,
        launch_timestamp, command_fingerprint, cwd, executable) the caller
        persists via research.execution_job.process_ownership."""

    @abc.abstractmethod
    def poll(self, job_id: str) -> RunnerStatus:
        """Non-blocking status check."""

    @abc.abstractmethod
    def graceful_stop(self, job_id: str) -> None:
        """Request the job pause/stop cleanly (e.g. save a checkpoint and
        exit). Must not block indefinitely -- JobManager owns the escalation
        timeout to force_terminate()."""

    @abc.abstractmethod
    def force_terminate(self, job_id: str) -> None:
        """Immediately terminate the job's process/process-tree. Called
        only after graceful_stop() has been given its grace period and the
        job is still confirmed to be OmniLab-owned (see
        process_ownership.verify_process_identity) -- never called on an
        unverified process."""

    @abc.abstractmethod
    def checkpoint_metadata(self, job_id: str) -> Optional[dict]:
        """Return the job's current checkpoint reference (see
        research.execution_job.checkpoint.CheckpointRef.to_dict()), or None
        if no checkpoint exists yet."""

    @abc.abstractmethod
    def resume(self, job_id: str, checkpoint_ref: dict) -> None:
        """Resume from a previously-validated-compatible checkpoint. Caller
        (JobManager) must have already called
        checkpoint.check_resume_compatibility() before invoking this --
        Runner.resume() itself does not re-derive compatibility."""

    @abc.abstractmethod
    def finalize(self, job_id: str) -> dict:
        """Called once poll() reports is_complete or is_failed. Returns a
        result payload (metrics, log references, failure reason) to be
        persisted by the caller. Must be idempotent -- calling it twice for
        the same terminal job must not duplicate side effects."""


class FakeRunner(Runner):
    """Deterministic, in-memory Runner for tests and for exercising
    JobManager's control flow without touching any real process or GPU.
    Behavior is driven entirely by a script the test supplies -- no real
    subprocess, no real training. See Phase J authorization section 21:
    'All process/GPU/training behavior in tests must use mocked process
    handles, fake runners, fake clocks, deterministic fixtures.'"""

    def __init__(self) -> None:
        self._prepared: set = set()
        self._launched: dict = {}
        self._poll_scripts: dict = {}
        self._stopped: set = set()
        self._force_terminated: set = set()
        self._checkpoints: dict = {}
        self._finalized: set = set()
        self._resumed: dict = {}
        self._fail_on_launch: set = set()

    def script_poll_sequence(self, job_id: str, sequence: list) -> None:
        """Test hook: queue a sequence of RunnerStatus objects poll() will
        return, one per call, holding the last one once exhausted."""
        self._poll_scripts[job_id] = list(sequence)

    def fail_on_launch(self, job_id: str) -> None:
        self._fail_on_launch.add(job_id)

    def prepare(self, job_id: str, spec: dict) -> None:
        self._prepared.add(job_id)

    def launch(self, job_id: str) -> dict:
        if job_id in self._fail_on_launch:
            raise RunnerError(f"{job_id}: simulated launch failure")
        identity = {
            "pid": 100000 + len(self._launched),
            "launch_timestamp": "2026-09-06T00:00:00+00:00",
            "command_fingerprint": f"fake-runner::{job_id}",
            "cwd": "research/execution_jobs",
            "executable": "fake-runner",
        }
        self._launched[job_id] = identity
        return identity

    def poll(self, job_id: str) -> RunnerStatus:
        script = self._poll_scripts.get(job_id)
        if not script:
            return RunnerStatus(is_running=True, detail="fake runner: default running")
        if len(script) > 1:
            return script.pop(0)
        return script[0]

    def graceful_stop(self, job_id: str) -> None:
        self._stopped.add(job_id)

    def force_terminate(self, job_id: str) -> None:
        self._force_terminated.add(job_id)

    def checkpoint_metadata(self, job_id: str) -> Optional[dict]:
        return self._checkpoints.get(job_id)

    def set_checkpoint(self, job_id: str, checkpoint_ref: dict) -> None:
        self._checkpoints[job_id] = checkpoint_ref

    def resume(self, job_id: str, checkpoint_ref: dict) -> None:
        self._resumed[job_id] = checkpoint_ref

    def finalize(self, job_id: str) -> dict:
        self._finalized.add(job_id)
        return {"job_id": job_id, "finalized": True}
