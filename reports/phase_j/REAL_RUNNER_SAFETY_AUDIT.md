# Phase J — Real Runner Security/Safety Audit

Self-audit against the exact checklist in this task's authorization section 21, run
against `research/execution_job/local_process_runner.py`, `gpu_slots.py`, `watchdog.py`,
and the `manager.py`/`process_ownership.py`/`execution_budget.py`/`state.py` changes.

## Checklist

- **`shell=True`**: not present anywhere. Grepped the whole `research/execution_job/`
  package; the only string match is this sentence's own docstring mention in
  `local_process_runner.py` explaining it is NEVER used. `Popen(..., shell=False)` is
  passed explicitly (redundant with the default, kept explicit as a visible guarantee).
- **Unsafe command concatenation**: none — `argv` is always a `list[str]` passed
  directly to `Popen`; no string formatting/concatenation ever builds a command line.
  A test (`test_no_shell_true_argument_vector_only`) proves a shell-metacharacter string
  (`$(whoami)`) is passed through literally, never interpreted.
- **Broad process kills**: none. `force_terminate()` only ever acts on
  `psutil.Process(pid).children(recursive=True) + [proc]` — the verified job's own
  process tree. No code path enumerates or kills by executable name, and none matches
  "all python.exe" or similar.
- **PID-only termination**: `force_terminate()` re-verifies full identity
  (pid + creation_time + command_fingerprint) via `process_ownership.verify_process_identity()`
  immediately before acting — a PID match alone is never sufficient. Proven by
  `test_pid_mismatch_blocks_kill`, which points a job's recorded identity at a
  different, genuinely-live process and confirms the kill is refused.
- **Inherited-secret serialization**: `_safe_env()` passes ONLY a fixed 7-key whitelist
  (`SYSTEMROOT, PATH, PATHEXT, TEMP, TMP, COMSPEC, NUMBER_OF_PROCESSORS`) — the full
  `os.environ` (which would include `OPENROUTER_API_KEY` if set) is never passed to
  `Popen`. Verified by reading `_safe_env()`'s implementation directly (single
  dict-comprehension over the whitelist, no fallback to the full environment).
- **Command-line secret leakage**: `argv` is persisted verbatim into `JobRecord`
  (`command_fingerprint`) and the resource ledger. **Residual risk, disclosed, not
  code-fixed**: if a future caller passed a secret as a literal CLI argument, it would
  be persisted in cleartext in `state.json`/the ledger. This is a caller-contract
  requirement, not something this module can enforce for arbitrary future callers —
  documented here so a future real training-job wrapper knows to pass secrets (if any
  are ever needed) via the whitelisted env mechanism or a file reference, never as a
  bare argv element.
- **Path traversal**: `job_id` (used to build `self._logs_dir / job_id / "process"`) is
  always system-generated (`state.next_job_id()`, format `JOB-NNNN`) — never derived
  from external/untrusted input in any code path that exists today. `cwd`/`argv` ARE
  caller-supplied and not path-validated; this is an accepted trust boundary (only
  OmniLab's own code calls `JobManager.launch()` — no external/LLM-authored text
  reaches `argv`/`cwd` directly today), explicitly flagged for re-review before any
  future caller derives `argv`/`cwd` from less-trusted input (e.g. LLM-authored
  proposal text) — that caller must sanitize/validate before construction, not this
  layer.
- **Arbitrary output-directory overwrite**: `LocalProcessRunner.prepare()` calls
  `artifact_safety.reserve_job_output_dir()` (existing Phase J module) before writing
  any log file — a directory already owned by a different job_id, or an unmarked
  pre-existing directory, is refused, never silently written into.
- **Race between ownership validation and termination**: `force_terminate()`'s
  verification and the actual `terminate()`/`kill()` calls are not a single atomic OS
  operation — a sub-millisecond window exists between `live_snapshot_for_pid()` and the
  `psutil.Process(pid)` calls that follow. **Accepted, documented residual risk**:
  Windows exposes no simple "kill iff creation-time still matches" primitive; closing
  this window fully would require a kernel-level compare-and-kill this project has no
  access to. The window is sub-millisecond in practice, immeasurably smaller than the
  time between a real PID's reuse events, and is disclosed here rather than silently
  assumed away.

## Windows-specific limitations (all empirically verified in this session, not assumed)

1. **`psutil.Process.cwd()` is unreliable on this Windows/psutil combination** — it was
   observed to report the OS temp directory regardless of the actual directory passed
   to `CreateProcess`. Fix applied: `live_snapshot_for_pid()` never populates `cwd` on
   `win32` (left `None`, which `verify_process_identity()`'s existing contract already
   treats as "skip this comparison"). Identity now rests on `pid + creation_time +
   command_fingerprint`, all three independently confirmed reliable.
2. **`CTRL_BREAK_EVENT` delivery requires the child to poll/sleep in short intervals**
   to notice it promptly — a child blocked in one long `time.sleep(N)` will not run its
   signal handler until that call returns (Python only checks for pending signals
   between bytecode instructions; Windows sleep is not itself interrupted by the
   event). Documented as a requirement for any future cooperative training script.
3. **No POSIX `SIGTERM` equivalent** — `CTRL_BREAK_EVENT` is the closest available
   primitive, and its "graceful" character depends entirely on whether the child
   installed a handler; a non-cooperative child is simply terminated (verified:
   the OS's own default action for the event ends a non-handling process; the ONE case
   requiring the `force_terminate()` escalation is a child that explicitly ignores the
   signal via `SIG_IGN`, which was tested directly).
4. **No exit-code recovery after an unmonitored process ends** — `psutil` alone cannot
   retrieve the exit code of a process this instance did not `Popen()` itself; a
   reattached job discovered no-longer-running is reported `is_failed` with an explicit
   "unknown outcome" detail, never a fabricated `COMPLETED`.

## Findings requiring a fix within this task's scope

None discovered rose to CRITICAL/HIGH after the two fixes already applied during
development and testing (both already committed in the code, not merely noted):
- `live_snapshot_for_pid()`'s `cwd` fix (item 1 above) — without it, ownership
  verification would falsely reject every genuinely-owned Windows process.
- `force_terminate()`'s "already exited" vs. "different live process" distinction —
  without it, a benign race (the process died from `graceful_stop()` moments before
  `force_terminate()` ran) would incorrectly raise `ProcessOwnershipError` instead of
  silently no-op'ing, which would have broken the cancellation path for every real job
  that responds promptly to a graceful stop.

## Recommendation

No CRITICAL/HIGH security issues remain open. The residual items above (secret-in-argv
caller contract, argv/cwd trust boundary, sub-millisecond TOCTOU window) are MEDIUM at
most, cannot be fully eliminated by this layer alone, and are disclosed rather than
silently accepted.
