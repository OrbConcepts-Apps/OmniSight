# Phase J — Resource Management and Safe Long-Run Execution Infrastructure

Infrastructure only. No EXP-0006, no training, no benchmark execution, no
device work, no live LLM calls anywhere in this phase. Every process/GPU
behavior exercised by this phase's own tests uses `FakeRunner` — no real
subprocess, no real GPU.

## Architecture

```
research/execution_job/
  __init__.py
  state.py             -- job state machine (JobRecord, transitions, crash-recovery classification)
  process_ownership.py -- ProcessIdentity / LiveProcessSnapshot / verify_process_identity
  runner.py             -- Runner ABC + FakeRunner (test/dry-run only)
  checkpoint.py         -- CheckpointRef + check_resume_compatibility
  accounting.py         -- append-only resource ledger (research/execution_ledger.jsonl)
  seeds.py               -- SeedPlan / aggregate_seed_results (cherry-pick-proof)
  artifact_safety.py    -- reserve_job_output_dir (collision/ownership guard)
  manager.py             -- JobManager: the one execution boundary everything above is wired through

research/execution_budget.py  -- extended (additive) with Phase J budget fields
```

`JobManager` is the only class a future real training runner is meant to be
driven through. It never launches, polls, or kills anything itself — every
such action is delegated to an injected `Runner` implementation. Today the
only `Runner` implementation is `FakeRunner`; a future phase adding a real
training runner subclasses `Runner` and `JobManager` does not need to
change.

## Job state machine

States: `CREATED -> VALIDATED -> AUTHORIZED -> PREPARING -> RUNNING ->
{PAUSING -> PAUSED -> RESUMING -> RUNNING} -> {COMPLETED | FAILED |
CANCELLED | TIMED_OUT}`.

`COMPLETED`/`FAILED`/`CANCELLED`/`TIMED_OUT` are the only terminals — once
reached, `transition()` refuses any further change (`JobStateError`).
`PAUSED` is deliberately non-terminal, mirroring Phase I's `BLOCKED`: a
paused job is expected to resume or be explicitly cancelled, never silently
discarded. Every transition is persisted to
`research/execution_jobs/<JOB-ID>/state.json` immediately (crash-safe:
process death right after a transition still leaves the new state on disk).

Valid/invalid transitions are an explicit table (`ALLOWED_TRANSITIONS`) —
anything not listed is refused with the full allowed-set named in the
error. Almost every non-terminal state can additionally reach
`FAILED`/`CANCELLED` (a crash or an operator stop can happen from nearly
anywhere).

## Authority boundary

`JobManager.launch()` runs this exact precondition chain, in order, before
a job record is even created:

1. `operational_state.check_gate()` — refuses if PAUSED/STOPPED.
2. `git_isolation.require_clean_tree()` — refuses on a dirty working tree
   (skippable via `LaunchPreconditions.require_clean_git_tree=False`, used
   only by this phase's own tests to avoid touching the real repo tree).
3. An optional caller-supplied `frozen_spec_check()` — lets a caller plug
   in Phase F's proposal-hash/approval verification without `JobManager`
   needing to know anything about `ExperimentProposal` itself.
4. `require_execution_budget()` (see below) — fail-closed GPU/runtime
   authorization.

Any refusal at any of these four steps raises before `state.create_job()`
is ever called — a refused launch attempt leaves no job record behind.

## Execution-budget semantics

`research/execution_budget.py`'s `ExecutionBudgetConfig` gained (additive,
non-breaking) fields this phase:

- `max_wall_clock_sec_per_job` — distinct from the pre-existing
  per-EXPERIMENT ceiling (an experiment may launch several jobs, e.g. one
  per seed).
- `max_cumulative_gpu_runtime_sec_per_day` — distinct from the pre-existing
  per-CYCLE ceiling.
- `max_retry_count` (default **0**) — automatic retry-from-scratch requires
  explicit future authorization; resume-from-checkpoint is a distinct code
  path (`JobManager.resume()`), never counted as a "retry".
- `min_free_disk_gb` / `min_free_ram_gb` — optional floors; when configured,
  the caller MUST supply the current reading or the check fails closed
  (never assumes "probably fine").

`require_execution_budget()` remains fail-closed by construction: no
config at all, `gpu_execution_authorized=False`, any configured limit
exceeded, or a configured floor with no reading supplied — all raise
`ExecutionBudgetError`. A resource *estimate* (e.g. a proposal's
`compute_resource_estimate`) is never treated as authorization by itself;
distinguishing estimate / authorization / reservation / consumption /
accounting was an explicit Phase J requirement (section 3) — the estimate
is only ever an input CHECKED against budget, never itself a grant.

## Resource accounting

`research/execution_ledger.jsonl` — append-only, one JSON object per line,
written via `accounting.record_event()`. Never rewritten or truncated by
this code; `read_ledger()` skips (never raises on) a corrupt individual
line. `gpu_active_sec` is always `None` — true GPU-active-cycle telemetry
is not implemented, and per Phase J authorization section 10 this
limitation is documented rather than papered over with fabricated
precision. The enforceable budget metric is wall-clock job occupancy
(`elapsed_sec`), which `cumulative_wall_clock_sec()` sums, optionally
filtered by UTC calendar date and/or experiment id.

## Process ownership

Every launched job's identity (`pid`, `launch_timestamp`,
`command_fingerprint`, `cwd`, `executable`) is persisted at launch time in
its `JobRecord`. Before any pause/cancel/timeout action touches a process,
`process_ownership.verify_process_identity()` compares a live snapshot
against every recorded field — a PID match alone is never sufficient (PID
reuse is real). Three outcomes:

- **Match** → the action proceeds.
- **No match** (wrong pid, wrong start time, wrong command, wrong cwd, or
  the process doesn't exist) → the action is skipped; the job is still
  driven to its terminal state (e.g. `TIMED_OUT`), but nothing is killed.
- **Ambiguous** (process exists but its identifying metadata could not be
  read) → raises `ProcessOwnershipError`. This propagates out of
  `JobManager` uncaught — the caller must treat it as "block and require
  human intervention," per Phase J authorization section 8. It is never
  swallowed or treated as a pass/fail default.

## Stop/kill (pause/cancel/timeout) semantics

- **Wall-clock timeout** (`JobManager.enforce_wall_clock`): caller supplies
  `elapsed_sec` from its own clock (real or fake — `JobManager` never
  sleeps or measures time itself, so no test needs a real multi-hour
  wait). Over budget → `runner.graceful_stop()` requested first; only once
  the caller reports `grace_expired=True` does `JobManager` poll and, if
  still running, escalate to `runner.force_terminate()`. Final state is
  always `TIMED_OUT`.
- **Operational PAUSED** (`request_pause`/`confirm_pause`): drives
  `RUNNING -> PAUSING` (graceful stop requested) `-> PAUSED` (once the
  runner confirms and reports a checkpoint reference).
- **Operational STOPPED** (`request_cancel`): drives toward `CANCELLED`
  with the same graceful-stop-then-escalate pattern, but only if the
  process is verified-owned; an unowned/nonexistent process is never
  touched, the job is still marked `CANCELLED`.

No kill path in this code ever targets an unrelated process — every
termination call is preceded by an identity check.

## Checkpoint/resume contract

`checkpoint.CheckpointRef` carries `path, step, seed, config_hash,
dataset_manifest_hash, code_commit_hash, created_at, experiment_id`.
`check_resume_compatibility()` raises `IncompatibleCheckpointError`
(listing every mismatch found, not just the first) unless every one of
config hash, dataset-manifest hash, code-commit hash, experiment id, and
seed matches exactly — or the checkpoint is missing. `JobManager.resume()`
calls this BEFORE transitioning the job out of `PAUSED`, so a refused
resume leaves the job's state untouched (still `PAUSED`) — it never
silently starts over and calls that a resume.

## Crash recovery

`JobManager.crash_recovery_scan()` classifies every persisted, non-terminal
job via `state.classify_for_recovery()`, using an externally-supplied
`{job_id: True/False/None}` ownership map (a real caller would build this
from a live process scan):

- `STILL_RUNNING_OWNED` — reattach monitoring; no relaunch.
- `RESUMABLE` — has a checkpoint reference; marked, NOT auto-resumed.
- `NON_RESUMABLE` — no checkpoint, or never actually launched; preserved as
  evidence, not auto-retried.
- `AMBIGUOUS` — ownership could not be determined; surfaced via
  `requires_human_intervention()`, never silently resolved either way.

`crash_recovery_scan()` never calls `runner.launch()`/`runner.resume()`
itself — recovery is classification only, confirmed by
`test_no_duplicate_launch_on_recovery_scan`.

## Multi-seed handling

`seeds.SeedPlan` is an immutable, ordered, deduplicated seed tuple, fixed
before any job in the plan launches. `aggregate_seed_results()` raises
`CherryPickError` if any pre-registered seed is missing from the result
set, if an unplanned seed appears, or if a seed has more than one result —
every seed must be accounted for as `COMPLETED`, one of
`FAILED`/`TIMED_OUT`/`CANCELLED`, or `PENDING` before an aggregate report
can be built at all. This module computes no scientific verdict itself —
that stays `research/experiment_validator.py`/`research/evaluation_policy.py`'s
job — it only guarantees the input to that judgment can never be a
favorable-subset selection.

## Residual limitations (honest, not fabricated)

- **No real training runner exists.** `FakeRunner` is the only
  implementation; a real one (subprocess/CUDA-aware) is future work, not
  built in this phase (explicitly out of scope per Phase J authorization
  section 5/21).
- **No true GPU-utilization telemetry.** The ledger's `gpu_active_sec` is
  always `None`; wall-clock job occupancy is the enforced metric instead.
- **`git_isolation.require_clean_tree()` is a real git subprocess call** —
  `JobManager`'s own tests disable it (`require_clean_git_tree=False`) to
  avoid depending on the ambient repo state; a real caller must leave it
  enabled.
- **No queue-integration code was built** (Phase J authorization section
  20 explicitly excludes this) — a future queue must independently
  recheck approvals/budget/operational-state/integrity immediately before
  calling `JobManager.launch()`; a queued entry never implies permission
  on its own.
- **Disk/RAM readings are caller-supplied**, not measured by this package
  (`research/resources.py` already has real psutil-based readers; wiring
  them into a real launch call is future integration work, not part of
  this infrastructure-only phase).
