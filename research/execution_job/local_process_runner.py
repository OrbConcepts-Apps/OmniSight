"""Phase J -- LocalProcessRunner: a real, generic local subprocess Runner.

Concrete implementation of research.execution_job.runner.Runner for a
Windows research machine. Generic (no YOLO/Ultralytics knowledge
whatsoever) -- any future training job is just "a program with an argv".

Security posture (see reports/phase_j/REAL_RUNNER_SAFETY_AUDIT.md for the
full audit):
  - argument-vector `subprocess.Popen(argv, ...)` ONLY -- never `shell=True`,
    never a concatenated command string.
  - a minimal environment whitelist is passed, never the full inherited
    `os.environ` (secrets like OPENROUTER_API_KEY are never propagated).
  - termination always re-verifies process identity via
    research.execution_job.process_ownership immediately before acting,
    and only ever targets the job's own process tree (via psutil parent/
    child relationships) -- never a bare "kill by PID" or "kill by
    executable name".

Windows graceful-stop honesty (Phase J real-runner authorization section 8):
Windows has no POSIX SIGTERM. `graceful_stop()` sends CTRL_BREAK_EVENT to
the job's own process group (created via CREATE_NEW_PROCESS_GROUP at
launch). A COOPERATIVE child that installs a SIGBREAK handler can shut
down cleanly on this signal (the test fixture in this test suite does
exactly that). A non-cooperative child is simply terminated by the
default OS handler for that event -- this is the best "graceful" signal
Windows offers without an application-level protocol, and is documented
here as exactly that, never oversold as POSIX SIGTERM-equivalent.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from research.config import REPO_ROOT
from research.execution_job.artifact_safety import reserve_job_output_dir
from research.execution_job.process_ownership import (
    ProcessIdentity,
    ProcessOwnershipError,
    command_fingerprint_of,
    live_snapshot_for_pid,
    verify_process_identity,
)
from research.execution_job.runner import Runner, RunnerError, RunnerStatus

LOGS_DIR = REPO_ROOT / "research" / "execution_jobs"

# Never inherit the full environment (Phase J real-runner authorization
# section 4/10: no secrets, no complete inherited environment). This is
# the minimum Windows needs to reliably launch/locate an executable.
_ENV_WHITELIST = ("SYSTEMROOT", "PATH", "PATHEXT", "TEMP", "TMP", "COMSPEC", "NUMBER_OF_PROCESSORS")


def _safe_env(extra: Optional[dict] = None) -> dict:
    env = {k: os.environ[k] for k in _ENV_WHITELIST if k in os.environ}
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


@dataclass
class _ProcessHandle:
    pid: int
    argv: list
    cwd: str
    creation_time: str  # canonical string form, see process_ownership.live_snapshot_for_pid
    stdout_path: Path
    stderr_path: Path
    popen: Optional[subprocess.Popen] = None  # None after a reattach() (fresh-process simulation)
    finalized: bool = False


class LocalProcessRunner(Runner):
    """Generic argv-based subprocess Runner. `spec` passed to prepare()
    must contain:
        argv: list[str]        -- required, the exact command vector
        cwd: str                -- optional, defaults to REPO_ROOT
        env: dict[str, str]      -- optional, additive to the safe whitelist
        gpu_required: bool        -- informational only; JobManager (not
                                      this class) is responsible for GPU
                                      slot acquisition/release around launch
    """

    def __init__(self, logs_dir: Path = LOGS_DIR):
        self._logs_dir = logs_dir
        self._prepared: dict = {}
        self._handles: dict = {}

    # -- prepare / launch --------------------------------------------------

    def prepare(self, job_id: str, spec: dict) -> None:
        argv = spec.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
            raise RunnerError(f"{job_id}: spec['argv'] must be a non-empty list[str], got {argv!r}")
        cwd = spec.get("cwd", str(REPO_ROOT))
        out_dir = reserve_job_output_dir(job_id, self._logs_dir / job_id / "process")
        self._prepared[job_id] = {
            "argv": list(argv),
            "cwd": cwd,
            "env": dict(spec.get("env") or {}),
            "out_dir": out_dir,
        }

    def launch(self, job_id: str) -> dict:
        prep = self._prepared.get(job_id)
        if prep is None:
            raise RunnerError(f"{job_id}: launch() called before prepare()")

        argv = prep["argv"]
        cwd = prep["cwd"]
        out_dir = prep["out_dir"]
        stdout_path = out_dir / "stdout.log"
        stderr_path = out_dir / "stderr.log"

        creationflags = 0
        if sys.platform == "win32":
            # New process GROUP so graceful_stop() can target CTRL_BREAK_EVENT
            # at exactly this job's tree, never at OmniLab's own console
            # group (which a bare CTRL_BREAK would otherwise also signal).
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        with open(stdout_path, "wb") as f_out, open(stderr_path, "wb") as f_err:
            popen = subprocess.Popen(
                argv,
                cwd=cwd,
                stdout=f_out,
                stderr=f_err,
                env=_safe_env(prep["env"]),
                creationflags=creationflags,
                close_fds=True,
                shell=False,  # NEVER True -- see module docstring / security audit
            )
        # File handles above are closed by the `with` block on the PARENT
        # side once Popen has duplicated them for the child -- this is the
        # standard, safe pattern (the child keeps its own OS-level handle).

        snapshot = live_snapshot_for_pid(popen.pid)
        creation_time = snapshot.start_timestamp or ""

        handle = _ProcessHandle(
            pid=popen.pid, argv=argv, cwd=cwd, creation_time=creation_time,
            stdout_path=stdout_path, stderr_path=stderr_path, popen=popen,
        )
        self._handles[job_id] = handle

        return {
            "pid": popen.pid,
            "launch_timestamp": creation_time,
            "command_fingerprint": command_fingerprint_of(argv),
            "cwd": cwd,
            "executable": argv[0],
        }

    # -- reattach (restart/recovery only; not part of the Runner ABC) -----

    def reattach(self, job_id: str, *, pid: int, argv: list, cwd: str, creation_time: str) -> None:
        """Rebuild internal tracking for an already-launched job after a
        simulated/real OmniLab restart, WITHOUT launching anything. The
        caller (crash-recovery flow) is responsible for having already
        confirmed via process_ownership.verify_process_identity that this
        PID is genuinely the job's own process before calling this."""
        out_dir = self._logs_dir / job_id / "process"
        self._handles[job_id] = _ProcessHandle(
            pid=pid, argv=argv, cwd=cwd, creation_time=creation_time,
            stdout_path=out_dir / "stdout.log", stderr_path=out_dir / "stderr.log", popen=None,
        )

    def recorded_identity(self, job_id: str) -> ProcessIdentity:
        h = self._handles[job_id]
        return ProcessIdentity(
            pid=h.pid, launch_timestamp=h.creation_time,
            command_fingerprint=command_fingerprint_of(h.argv), job_id=job_id, cwd=h.cwd,
            executable=h.argv[0] if h.argv else "",
        )

    # -- poll ---------------------------------------------------------------

    def poll(self, job_id: str) -> RunnerStatus:
        h = self._handles.get(job_id)
        if h is None:
            raise RunnerError(f"{job_id}: poll() called before launch()/reattach()")

        if h.popen is not None:
            rc = h.popen.poll()
            if rc is None:
                return RunnerStatus(is_running=True, detail="subprocess running")
            if rc == 0:
                return RunnerStatus(is_complete=True, detail="exit_code=0")
            return RunnerStatus(is_failed=True, detail=f"exit_code={rc}")

        # Reattached (no Popen handle) -- exit code is only obtainable via
        # Popen.wait(); psutil alone cannot retrieve the exit code of a
        # process this instance did not create. If still alive, that's
        # fine (still-running is fully observable). If it already ended
        # while unmonitored, its outcome is HONESTLY unknown -- never
        # fabricated as COMPLETED or a specific exit code.
        snapshot = live_snapshot_for_pid(h.pid)
        if snapshot.exists:
            return RunnerStatus(is_running=True, detail="reattached, subprocess still running")
        return RunnerStatus(
            is_failed=True,
            detail="reattached process is no longer running and its exit code could not be "
            "recovered (it ended while unmonitored) -- reported as failed rather than "
            "fabricating a COMPLETED outcome with no evidence.",
        )

    # -- stop / terminate ----------------------------------------------------

    def graceful_stop(self, job_id: str) -> None:
        h = self._handles.get(job_id)
        if h is None:
            return
        if sys.platform != "win32":
            raise RunnerError("LocalProcessRunner's graceful_stop() is implemented for win32 only")
        import signal

        try:
            os.kill(h.pid, signal.CTRL_BREAK_EVENT)
        except (OSError, ProcessLookupError):
            pass  # already gone -- nothing to signal

    def force_terminate(self, job_id: str) -> None:
        h = self._handles.get(job_id)
        if h is None:
            return

        # Re-verify identity IMMEDIATELY before acting (Phase J real-runner
        # authorization section 9) -- closes the race between an earlier
        # ownership check and the actual kill.
        recorded = self.recorded_identity(job_id)
        live = live_snapshot_for_pid(h.pid)
        if not live.exists:
            # Already gone (e.g. a preceding graceful_stop already ended
            # it) -- benign, nothing to terminate, not a safety concern.
            # Distinct from the case below (a DIFFERENT live process now
            # occupies this PID), which IS refused.
            return
        if not verify_process_identity(recorded, live):
            raise ProcessOwnershipError(
                f"{job_id}: refusing force_terminate -- a live process exists at pid {h.pid} "
                "but does not verify as this job's own process (PID reused by an unrelated "
                "process). No kill attempted."
            )

        import psutil

        try:
            proc = psutil.Process(h.pid)
        except psutil.NoSuchProcess:
            return  # already gone

        # Only this job's own process tree -- never anything matched by
        # executable name alone.
        try:
            children = proc.children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        targets = children + [proc]

        for p in targets:
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(targets, timeout=2)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass

    # -- checkpoint / resume (generic subprocess: none, never fabricated) --

    def checkpoint_metadata(self, job_id: str) -> Optional[dict]:
        return None

    def resume(self, job_id: str, checkpoint_ref: dict) -> None:
        raise RunnerError(
            f"{job_id}: LocalProcessRunner has no checkpoint/resume capability for a generic "
            "subprocess -- a process that died unexpectedly is NON_RESUMABLE, never fabricated "
            "as resumable."
        )

    # -- finalize -------------------------------------------------------------

    def finalize(self, job_id: str) -> dict:
        h = self._handles.get(job_id)
        if h is None:
            raise RunnerError(f"{job_id}: finalize() called before launch()/reattach()")
        exit_code = h.popen.poll() if h.popen is not None else None
        h.finalized = True
        return {
            "job_id": job_id,
            "exit_code": exit_code,
            "stdout_path": str(h.stdout_path),
            "stderr_path": str(h.stderr_path),
        }
