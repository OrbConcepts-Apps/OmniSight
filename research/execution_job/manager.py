"""Phase J -- JobManager: the generic long-running-job execution boundary.

Wires together operational_state (pause/stop kill switch), execution_budget
(fail-closed GPU/runtime authorization), git_isolation (clean-tree check),
process_ownership (identity-verified termination), the job state machine,
the checkpoint/resume contract, and the resource-accounting ledger. This is
the ONE place a future real training runner is meant to be driven through
-- it never launches, polls, or kills anything itself; all of that is
delegated to the injected Runner, which in every test here is FakeRunner.

No autonomous experiment selection, no EXP-0006, no real training. See
research/execution_job/__init__.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from research import git_isolation, operational_state
from research.execution_budget import (
    ExecutionBudgetConfig,
    ExecutionBudgetError,
    ResourceEstimate,
    require_execution_budget,
)
from research.execution_job import accounting, state
from research.execution_job.checkpoint import (
    CheckpointRef,
    IncompatibleCheckpointError,
    check_resume_compatibility,
)
from research.execution_job.process_ownership import (
    LiveProcessSnapshot,
    ProcessIdentity,
    ProcessOwnershipError,
    verify_process_identity,
)
from research.execution_job.runner import Runner, RunnerError


class JobManagerError(RuntimeError):
    """Raised for any precondition/authorization refusal at the JobManager
    boundary -- fail closed, per Phase J authorization section 19."""


@dataclass
class LaunchPreconditions:
    """Explicit knobs for what to check before a job may launch. Every
    field defaults to the SAFEST (most-checked) behavior; a caller must
    deliberately opt out (e.g. in a test) rather than a missing kwarg
    silently skipping a check."""

    require_clean_git_tree: bool = True
    frozen_spec_check: Optional[Callable[[], None]] = None  # raises if unfrozen/tampered


class JobManager:
    def __init__(
        self,
        runner: Runner,
        *,
        ledger_path=accounting.LEDGER_PATH,
        gpu_slots=None,
    ) -> None:
        self._runner = runner
        self._ledger_path = ledger_path
        self._gpu_slots = gpu_slots

    def _release_gpu_slot(self, record: state.JobRecord) -> None:
        """Safe to call unconditionally (including for a non-GPU job, or a
        job that never reached slot acquisition) -- release() itself is a
        no-op if the job never held a slot. Called on EVERY terminal path
        (Phase J real-runner authorization section 15)."""
        if self._gpu_slots is not None and record.gpu_required:
            self._gpu_slots.release(record.job_id)

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------
    def launch(
        self,
        *,
        experiment_id: str,
        spec: dict,
        budget_config: Optional[ExecutionBudgetConfig],
        estimate: Optional[ResourceEstimate] = None,
        seed: Optional[int] = None,
        preconditions: Optional[LaunchPreconditions] = None,
        current_running_training_jobs: int = 0,
        current_cumulative_runtime_sec: int = 0,
        current_cumulative_gpu_runtime_sec_today: int = 0,
        retry_count: int = 0,
        current_free_disk_gb: Optional[float] = None,
        current_free_ram_gb: Optional[float] = None,
        gpu_required: bool = True,
    ) -> state.JobRecord:
        """Full precondition chain, per Phase J authorization sections 3-4,
        15-16, 19: operational state RUNNING -> clean git tree (unless
        disabled) -> frozen-spec check (if supplied) -> execution-budget
        authorization -> only THEN prepare()/launch() the runner. Any
        refusal raises before a job record is even created -- nothing is
        persisted for a launch attempt that never got past its gates."""
        preconditions = preconditions or LaunchPreconditions()

        # 1. Operational gate -- fails closed on PAUSED/STOPPED.
        operational_state.check_gate("phase_j_job_launch")

        # 2. Git isolation -- a training job must not start against a dirty
        #    tree (Phase J authorization section 15).
        if preconditions.require_clean_git_tree:
            git_isolation.require_clean_tree("phase_j_job_launch")

        # 3. Frozen-spec / integrity check, if the caller supplied one
        #    (Phase J authorization section 16 -- must not bypass Phase F
        #    freeze semantics; JobManager does not know how to verify a
        #    proposal hash itself, so it delegates to the caller's check).
        if preconditions.frozen_spec_check is not None:
            preconditions.frozen_spec_check()

        # 4. Execution-budget authorization -- fail closed by construction.
        #    gpu_required=False (a CPU-only harmless fixture) exempts only
        #    the GPU-authorization check itself; every other dimension
        #    (concurrency/wall-clock/retry/disk/ram) still applies exactly
        #    as before.
        require_execution_budget(
            f"launch:{experiment_id}",
            budget_config,
            estimate=estimate,
            current_cumulative_runtime_sec=current_cumulative_runtime_sec,
            current_running_training_jobs=current_running_training_jobs,
            current_cumulative_gpu_runtime_sec_today=current_cumulative_gpu_runtime_sec_today,
            retry_count=retry_count,
            current_free_disk_gb=current_free_disk_gb,
            current_free_ram_gb=current_free_ram_gb,
            gpu_required=gpu_required,
        )

        record = state.create_job(experiment_id=experiment_id, seed=seed, gpu_required=gpu_required)

        # 5. GPU concurrency slot -- only for jobs that declare they need
        #    one (Phase J real-runner authorization section 15/16).
        if gpu_required and self._gpu_slots is not None:
            try:
                self._gpu_slots.acquire(record.job_id)
            except Exception as e:  # noqa: BLE001 -- GpuSlotError or similar
                state.transition(record, state.FAILED, reason=f"GPU slot unavailable: {e}")
                raise JobManagerError(f"{record.job_id}: GPU slot unavailable: {e}") from e

        record = state.transition(record, state.VALIDATED, reason="preconditions passed")
        record = state.transition(record, state.AUTHORIZED, reason="execution budget authorized")
        record = state.transition(record, state.PREPARING, reason="runner.prepare() starting")

        try:
            self._runner.prepare(record.job_id, spec)
        except Exception as e:  # noqa: BLE001 -- any prepare failure is a job failure
            state.transition(record, state.FAILED, reason=f"prepare() failed: {e}")
            self._release_gpu_slot(record)
            raise JobManagerError(f"{record.job_id}: prepare() failed: {e}") from e

        try:
            identity = self._runner.launch(record.job_id)
        except Exception as e:  # noqa: BLE001
            state.transition(record, state.FAILED, reason=f"launch() failed: {e}")
            self._release_gpu_slot(record)
            raise JobManagerError(f"{record.job_id}: launch() failed: {e}") from e

        record.pid = identity.get("pid")
        record.launched_at = identity.get("launch_timestamp")
        record.command_fingerprint = identity.get("command_fingerprint", "")
        record.cwd = identity.get("cwd", "")
        record.executable = identity.get("executable", "")
        state.save(record)
        record = state.transition(record, state.RUNNING, reason="runner.launch() succeeded")

        accounting.record_event(
            job_id=record.job_id,
            experiment_id=experiment_id,
            event="STARTED",
            state=record.state,
            ledger_path=self._ledger_path,
        )
        return record

    def _identity_of(self, record: state.JobRecord) -> ProcessIdentity:
        return ProcessIdentity(
            pid=record.pid or -1,
            launch_timestamp=record.launched_at or "",
            command_fingerprint=record.command_fingerprint,
            job_id=record.job_id,
            cwd=record.cwd,
            executable=record.executable,
        )

    # ------------------------------------------------------------------
    # Wall-clock enforcement
    # ------------------------------------------------------------------
    def enforce_wall_clock(
        self,
        record: state.JobRecord,
        *,
        elapsed_sec: float,
        max_wall_clock_sec: Optional[int],
        live_snapshot: Optional[LiveProcessSnapshot],
        grace_expired: bool,
    ) -> state.JobRecord:
        """Caller supplies `elapsed_sec` (from its own clock/fake clock --
        JobManager never sleeps or measures time itself, per Phase J
        authorization section 9: no real multi-hour sleep in this code
        path). If the job is within budget, returns the record unchanged.
        If it has exceeded `max_wall_clock_sec`, requests a graceful stop;
        if `grace_expired` is True (caller decides how long to wait,
        entirely outside this method), escalates to force_terminate and
        marks the job TIMED_OUT. Only ever kills a process whose identity
        verifies against what was recorded at launch -- an ambiguous or
        mismatched identity blocks the kill and raises."""
        if max_wall_clock_sec is None or elapsed_sec <= max_wall_clock_sec:
            return record

        identity = self._identity_of(record)
        owned = verify_process_identity(identity, live_snapshot)
        if not owned:
            # Nothing (verified) to stop -- record the timeout without
            # attempting to touch an unowned/nonexistent process.
            record = state.transition(
                record, state.TIMED_OUT,
                reason=f"wall-clock exceeded ({elapsed_sec}s > {max_wall_clock_sec}s); "
                "no verified live process to terminate",
            )
            self._record_finish(record, elapsed_sec, "wall_clock_timeout")
            return record

        self._runner.graceful_stop(record.job_id)
        if not grace_expired:
            return record  # still waiting out the grace period; caller polls again

        status = self._runner.poll(record.job_id)
        if status.is_running:
            self._runner.force_terminate(record.job_id)
        record = state.transition(
            record, state.TIMED_OUT,
            reason=f"wall-clock exceeded ({elapsed_sec}s > {max_wall_clock_sec}s)",
        )
        self._record_finish(record, elapsed_sec, "wall_clock_timeout")
        return record

    def _record_finish(self, record: state.JobRecord, elapsed_sec: float, reason: str) -> None:
        # Called from every terminal path (timeout x2, cancel, finalize) --
        # the single place a GPU slot is guaranteed to be released, per
        # Phase J real-runner authorization section 15: "release slot on
        # every terminal path, including exception/timeout/cancel."
        self._release_gpu_slot(record)
        accounting.record_event(
            job_id=record.job_id,
            experiment_id=record.experiment_id,
            event="FINISHED",
            elapsed_sec=elapsed_sec,
            termination_reason=reason,
            state=record.state,
            ledger_path=self._ledger_path,
        )

    # ------------------------------------------------------------------
    # Pause / resume (operational PAUSED semantics, section 7)
    # ------------------------------------------------------------------
    def request_pause(self, record: state.JobRecord) -> state.JobRecord:
        record = state.transition(record, state.PAUSING, reason="operational pause requested")
        self._runner.graceful_stop(record.job_id)
        return record

    def confirm_pause(self, record: state.JobRecord) -> state.JobRecord:
        checkpoint_ref = self._runner.checkpoint_metadata(record.job_id)
        record.checkpoint_ref = checkpoint_ref
        state.save(record)
        return state.transition(record, state.PAUSED, reason="runner confirmed paused state")

    def resume(
        self,
        record: state.JobRecord,
        *,
        expected_config_hash: str,
        expected_dataset_manifest_hash: str,
        expected_code_commit_hash: str,
        expected_seed: int,
    ) -> state.JobRecord:
        """Raises IncompatibleCheckpointError (never silently starts over)
        if the persisted checkpoint doesn't match every identity field."""
        checkpoint = (
            CheckpointRef.from_dict(record.checkpoint_ref) if record.checkpoint_ref else None
        )
        check_resume_compatibility(
            checkpoint=checkpoint,
            expected_config_hash=expected_config_hash,
            expected_dataset_manifest_hash=expected_dataset_manifest_hash,
            expected_code_commit_hash=expected_code_commit_hash,
            expected_experiment_id=record.experiment_id,
            expected_seed=expected_seed,
        )
        record = state.transition(record, state.RESUMING, reason="checkpoint verified compatible")
        self._runner.resume(record.job_id, record.checkpoint_ref)
        identity = self._runner.launch(record.job_id)
        record.pid = identity.get("pid")
        record.launched_at = identity.get("launch_timestamp")
        record.command_fingerprint = identity.get("command_fingerprint", "")
        record.cwd = identity.get("cwd", "")
        record.executable = identity.get("executable", "")
        state.save(record)
        return state.transition(record, state.RUNNING, reason="resumed")

    # ------------------------------------------------------------------
    # Cancel (operational STOPPED semantics, section 7)
    # ------------------------------------------------------------------
    def request_cancel(
        self,
        record: state.JobRecord,
        *,
        reason: str,
        live_snapshot: Optional[LiveProcessSnapshot],
        grace_expired: bool,
    ) -> state.JobRecord:
        identity = self._identity_of(record)
        owned = verify_process_identity(identity, live_snapshot)
        if owned:
            self._runner.graceful_stop(record.job_id)
            if grace_expired:
                status = self._runner.poll(record.job_id)
                if status.is_running:
                    self._runner.force_terminate(record.job_id)
            else:
                return record  # still in the grace window; caller polls again
        record = state.transition(record, state.CANCELLED, reason=reason)
        self._record_finish(record, 0.0, reason)
        return record

    # ------------------------------------------------------------------
    # Finalize (normal completion/failure, no timeout/cancel involved)
    # ------------------------------------------------------------------
    def finalize(
        self, record: state.JobRecord, *, outcome: str, elapsed_sec: float, reason: str = ""
    ) -> state.JobRecord:
        if outcome not in (state.COMPLETED, state.FAILED):
            raise JobManagerError(f"finalize() outcome must be COMPLETED or FAILED, got {outcome!r}")
        result = self._runner.finalize(record.job_id)
        record = state.transition(record, outcome, reason=reason or str(result))
        self._record_finish(record, elapsed_sec, reason or outcome)
        return record

    # ------------------------------------------------------------------
    # Crash recovery (section 18) -- classification only, never re-launches.
    # ------------------------------------------------------------------
    def crash_recovery_scan(
        self, ownership_checks: dict
    ) -> dict:
        """`ownership_checks` maps job_id -> Optional[bool] (True/False/None
        per process_ownership.verify_process_identity's contract; a caller
        that hasn't checked a given non-running job may simply omit it, or
        pass None). Returns {job_id: classification}. Never calls
        self._runner.launch()/resume() -- recovery only classifies; a
        human/a separate explicit call decides whether to actually resume
        a RESUMABLE job, per Phase J authorization section 18 ('mark
        resumable, do not automatically resume unless policy/authorization
        permits')."""
        results = {}
        for record in state.list_all_jobs():
            owned = ownership_checks.get(record.job_id)
            results[record.job_id] = state.classify_for_recovery(
                record, process_is_verifiably_owned=owned
            )
        return results


def requires_human_intervention(classifications: dict) -> list:
    """Filters a crash_recovery_scan() result down to the job ids that need
    a human decision before anything else touches them."""
    return [
        job_id
        for job_id, cls in classifications.items()
        if cls == state.RecoveryClassification.AMBIGUOUS
    ]
