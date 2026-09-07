# Phase J — Safety Audit

Self-audit against the Phase J authorization's completion criteria
(section 27) and the human-authority/no-real-execution constraints running
through the whole authorization. Zero live LLM calls; zero real
process/GPU/training work anywhere in this phase.

## Completion criteria (section 27) — checked

- [x] Long-running job abstraction exists — `research/execution_job/manager.py::JobManager`
  + `runner.py::Runner`.
- [x] Budgets are actively enforceable — `require_execution_budget()` extended and wired
  into `JobManager.launch()`; `TestLaunchDeniedGates` proves every dimension refuses.
- [x] Stop/timeout behavior is code-enforced — `JobManager.enforce_wall_clock()` /
  `request_cancel()`, both with graceful-stop-then-escalate and identity-gated termination.
- [x] Crash recovery/idempotency exists — `state.classify_for_recovery()` +
  `JobManager.crash_recovery_scan()`, proven to never re-launch
  (`test_no_duplicate_launch_on_recovery_scan`).
- [x] Checkpoint contract exists — `checkpoint.py`, `check_resume_compatibility()`
  rejects on any of 5 identity mismatches, never silently restarts.
- [x] Multi-seed accounting exists — `seeds.py`, `CherryPickError` blocks any incomplete
  or unplanned aggregation input.
- [x] Failed runs preserved — `JobManager.finalize(outcome=FAILED)` leaves the job record
  and ledger entry on disk, immutable once terminal (`JobStateError` on re-transition).
- [x] Human execution gates remain fail-closed — `new_training_approved`,
  `mac_iphone_deployment_approved`, and every other Phase F approval flag are untouched by
  this phase; `JobManager` has no code path that grants any of them.
- [x] Tests pass — 818/818 (720 pre-existing + 98 new), 0 failures.
- [x] No real experiment was executed — `research/db.py` still lists exactly
  EXP-0001–EXP-0005; no EXP-0006; `git diff --stat -- ios/ benchmark/config.py` empty;
  `research/candidates/CANDIDATE-0003/` byte-unchanged.

## Human-authority boundary re-verified

Grepped `research/execution_job/` and `research/execution_budget.py` for any reference to
`new_training_approved`, `mac_iphone_deployment_approved`, `queue_experiment_from_spec`,
`OmniLabDB`, `ExperimentProposal`, `ExperimentSpec` — **`research/execution_job/` has zero
matches for any of these**; `research/execution_budget.py` mentions `new_training_approved`
only in comments/docstrings explaining how it's a DIFFERENT, HUMAN-only gate from this
package's own GPU-authorization flag, never writing to it. This package cannot grant an
approval, register an experiment, or queue one — it has no import of, or reference to,
`ExperimentProposal`/`ExperimentSpec`/`OmniLabDB` at all. `JobManager.launch()`'s
`frozen_spec_check` parameter is an injection point for a FUTURE caller to plug Phase F
verification into — `JobManager` itself does not know how to verify a proposal hash and
cannot bypass Phase F freeze semantics because it never touches them.

## Kill-path safety re-verified

Every one of `enforce_wall_clock()`, `request_cancel()` calls
`process_ownership.verify_process_identity()` before `runner.graceful_stop()`/
`force_terminate()`. Three tests specifically target the failure modes named in the
authorization: `test_pid_reused_by_different_process_not_terminated` (PID reuse),
`test_unowned_process_never_touched_but_still_timed_out` (nonexistent process), and
`test_ambiguous_ownership_raises_not_guesses` (missing identity metadata correctly raises
rather than defaulting either way).

## Residual risk / what Phase J does NOT close

- **No real Runner implementation.** `FakeRunner` proves the control flow; it proves
  nothing about how a real subprocess/CUDA training job will actually behave (signal
  handling on graceful stop, checkpoint I/O latency, GPU memory reclaim timing on
  force-terminate). A real Runner subclass needs its own dedicated tests before EXP-0006
  (or any training-heavy experiment) executes against it.
- **`git_isolation.require_clean_tree()` and `resources.py`'s disk/RAM readers are real,
  unmocked in production use** — this phase's tests bypass/inject them for isolation, which
  is correct for unit tests but means the FULL integration path (real git tree + real
  psutil readings + `JobManager`) has not itself been exercised end-to-end in this phase.
  A pre-EXP-0006 integration test wiring all of it together (still with `FakeRunner`, real
  everything else) would close this gap cheaply.
- **GPU-active-cycle accounting remains unmeasured** — wall-clock occupancy is the
  enforced proxy; a genuinely idle-but-still-"running" job would still consume its
  wall-clock budget without any actual GPU work happening. Acceptable for now (documented,
  not silently assumed away) but worth flagging for a future, more precise accounting pass.
- **Queue-vs-execution semantics are audited, not implemented** — no code in this repo yet
  re-checks approvals/budget/state/integrity at a queue-consumption boundary, because no
  autonomous queue-consumption code exists (deliberately out of scope, section 20). Building
  one is future work that must call `JobManager.launch()`'s full precondition chain, never
  bypass it.

## Recommendation

Phase J's own completion criteria are met for what it was scoped to build (infrastructure,
zero real execution). It does **not** by itself make EXP-0006 ready to execute — a real
Runner implementation and the CANDIDATE-0003 spec corrections
(`reports/phase_i/CANDIDATE_0003_EXP0006_ELIGIBILITY.md`) remain separate, still-open
prerequisites.
