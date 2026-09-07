# Phase J — Real Local Process Runner Implementation

Infrastructure only. EXP-0006 was NOT executed, NOT trained, remains `BLOCKED`. Every
subprocess used by this phase's own tests is a tiny, harmless `sys.executable -c "..."`
fixture — no YOLO/Ultralytics, no GPU/CUDA, no network, no personal data.

## New modules

```
research/execution_job/
  local_process_runner.py  -- LocalProcessRunner(Runner): real argv subprocess control
  gpu_slots.py               -- GpuSlotManager: persisted GPU concurrency reservation
  watchdog.py                -- run_watchdog(): active monotonic-clock polling loop
```
Plus targeted additions to already-existing Phase J modules: `process_ownership.py`
(`live_snapshot_for_pid()`, `command_fingerprint_of()` — real psutil-backed identity),
`manager.py` (GPU-slot acquire/release wired into `launch()`/every terminal path,
`gpu_required` param, `require_execution_budget(..., gpu_required=...)`),
`execution_budget.py` (`gpu_required: bool = True` param on `require_execution_budget()` —
default preserves every prior test's behavior exactly), `state.py`
(`JobRecord.gpu_required` field, additive, non-hashed).

## Launch mechanism

`subprocess.Popen(argv, cwd=cwd, stdout=f_out, stderr=f_err, env=_safe_env(...),
creationflags=CREATE_NEW_PROCESS_GROUP, close_fds=True, shell=False)`. Argument-vector
only — `shell=True` never appears anywhere in this codebase (verified by grep; the only
match is this sentence's own docstring mention). A string containing shell metacharacters
(e.g. `$(whoami)`) is passed through and printed literally by a test fixture, proving no
shell interpretation occurs.

## Process identity

Recorded at launch: `pid`, `launch_timestamp` (process creation time, `psutil`), `cwd`,
`executable` (argv[0]), `command_fingerprint` (`" ".join(argv)`, identical formatting
used at both launch time and live-check time so a genuinely-owned process always
matches exactly). Persisted into the same `JobRecord`/`state.json` shape Phase J already
uses — no new schema.

## Windows process-tree strategy

Termination uses `psutil.Process(pid).children(recursive=True)` to enumerate the job's
own descendant tree, then `terminate()` on every member (parent + children), a bounded
`psutil.wait_procs(timeout=2)`, and `kill()` on any survivor. Never targets a process by
executable name — only processes discovered via the verified job's own process-tree
relationship. A dedicated integration test (`test_child_process_tree_terminated`)
launches a parent that spawns one child sleeper and confirms `force_terminate()` kills
both.

## Ownership verification (real)

`process_ownership.live_snapshot_for_pid(pid)` builds a real `LiveProcessSnapshot` from
`psutil` (`pid`, `create_time()`, `cmdline()`, `exe()`). `force_terminate()` re-verifies
identity **immediately before acting** — closing the race between an earlier check and
the kill itself — and distinguishes two outcomes: the process already exited (benign,
silent no-op) vs. a **different, live** process now exists at that PID (refused,
`ProcessOwnershipError`, never killed). PID-reuse is explicitly tested
(`test_pid_mismatch_blocks_kill`): a job's recorded identity is pointed at this test
process's own real, live PID, and `force_terminate()` correctly refuses to touch it.

## Active wall-clock watchdog

`watchdog.run_watchdog()` polls on a bounded interval using an injectable monotonic
clock (real `time.monotonic`/`time.sleep` in production; a `FakeClock` in unit tests).
On timeout it drives `JobManager.enforce_wall_clock()`'s existing graceful-stop →
grace-period → forced-termination escalation automatically, never sleeping for the
full budget in one blocking call. One real-subprocess integration test
(`test_real_watchdog_stops_real_subprocess_on_timeout`) proves the whole chain against
an actual 20-second-sleep process with a 0-second budget and a 0.3-second grace period
— total real wall-clock cost under 2 seconds.

## Graceful stop — real, verified Windows behavior

Windows has no POSIX `SIGTERM`. `graceful_stop()` sends `CTRL_BREAK_EVENT` to the job's
own process group (`CREATE_NEW_PROCESS_GROUP` at launch). **Empirically verified in this
session** (not assumed):
- A **cooperative** child that installs a `signal.SIGBREAK` handler AND polls/sleeps in
  short intervals (not one long blocking `time.sleep(N)`) shuts down cleanly, exit code
  0, in well under half a second.
- A **non-cooperative** child with no handler is also terminated by the CTRL_BREAK
  event's own default OS action — reasonably promptly in practice.
- A child that explicitly does `signal.signal(signal.SIGBREAK, signal.SIG_IGN)` is
  correctly **not** affected by `graceful_stop()` and requires the `force_terminate()`
  escalation, which is proven to work (`test_non_cooperative_child_escalates_to_forced_termination`).
- **Discovered limitation, documented honestly**: a child blocked in ONE long
  `time.sleep(N)` C-level call does not notice a delivered signal until that call
  returns — Python only checks for pending signals between bytecode instructions. A
  cooperative shutdown handler is only actually effective if the child's own loop
  yields control periodically (a short-sleep loop, not one long sleep). This is real,
  reproducible Windows/CPython behavior, not a defect in this implementation — a future
  real training script must poll/checkpoint periodically to be gracefully stoppable at
  all, and this is now documented rather than silently assumed to "just work".

## Exit-code / outcome mapping

`exit_code == 0` → `RunnerStatus.is_complete`; nonzero → `is_failed` (detail names the
exact code). A **reattached** job (no live `Popen` handle — post-restart) that is found
no longer running has its exit code **honestly reported as unknown** — `is_failed` with
an explicit "ended while unmonitored" detail, never fabricated as `COMPLETED`.
`JobManager` retains sole responsibility for the final persisted job state
(`TIMED_OUT`/`CANCELLED`/`COMPLETED`/`FAILED`), unchanged from the existing architecture.

## Checkpoint / resume — honest non-capability

`checkpoint_metadata()` always returns `None`; `resume()` always raises `RunnerError`.
A generic subprocess has no checkpoint concept — nothing is fabricated. A process that
dies unexpectedly is `NON_RESUMABLE`, consistent with `state.classify_for_recovery()`'s
existing logic.

## Execution-budget / GPU integration

`JobManager.launch(..., gpu_required: bool)`. When `True` (default, preserving every
prior test unchanged): `require_execution_budget(..., gpu_required=True)` demands
`gpu_execution_authorized=True`, AND a `GpuSlotManager` slot is acquired immediately
after the job record is created. When `False` (a CPU-only harmless fixture):
`gpu_execution_authorized` is not required at all, and no GPU slot is touched. The
slot is released on **every** terminal path — both `launch()` failure branches
(`prepare()`/`launch()` raising), and via `_record_finish()`, which every one of
`enforce_wall_clock`'s two `TIMED_OUT` branches, `request_cancel`'s `CANCELLED` branch,
and `finalize()`'s `COMPLETED`/`FAILED` branches already call — one single release
point covers every exit.

## GPU concurrency

`GpuSlotManager` (default `max_concurrent_gpu_jobs=1`, matching the single-GPU research
machine) persists its reservation set to `research/execution_jobs/gpu_slots.json`.
`reconcile(verified_still_running_job_ids)` drops a stale reservation left behind by a
job no longer verifiably running after a crash-recovery scan — restart-safe without
either leaking the slot forever or silently dropping a genuinely-still-running job's
reservation.

## Resource accounting

Unchanged ledger shape (`research/execution_ledger.jsonl`) — `STARTED`/`FINISHED`
events, `elapsed_sec`, `termination_reason`, `state`. `gpu_active_sec` remains always
`None` — still no true GPU-utilization telemetry, still honestly documented as such,
never fabricated.

## Real integration tests run (all against actual local subprocesses)

37 new tests, 0 failures, covering exactly the required real-integration list: successful
launch/completion, nonzero exit → FAILED, watchdog timeout → TIMED_OUT, stdout/stderr
captured, cancellation, forced termination after graceful stop fails/ignored, real
identity verification, PID-mismatch blocks kill, child-process-tree termination,
restart/reattach without duplicate launch, CPU-only job skips GPU authorization,
GPU-declared job requires it, concurrent GPU slot enforcement + rejection, slot cleanup
on prepare-failure and on cancellation, resource ledger entries persisted for a real job.
