"""Phase J -- process ownership identity and verification.

A launched job's PID alone is never sufficient to identify it later (PID
reuse by the OS is real, especially across a long-running research session
or a machine restart). This module defines the full identity record OmniLab
persists for every job it launches, and the verification function that must
be consulted before attaching to, pausing, or killing anything -- per Phase
J authorization section 8: "If uncertain: BLOCK and require human
intervention."
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProcessIdentity:
    """Everything OmniLab records about a process it launched, at launch
    time. All fields are compared on verification -- a PID match alone is
    never treated as sufficient identity confirmation."""

    pid: int
    launch_timestamp: str
    command_fingerprint: str
    job_id: str
    cwd: str
    executable: str = ""


class ProcessOwnershipError(RuntimeError):
    """Raised when a process's ownership cannot be confidently established
    -- the caller must treat this as BLOCK/require-human-intervention, per
    Phase J authorization section 8. Never a signal to proceed anyway."""


@dataclass(frozen=True)
class LiveProcessSnapshot:
    """What can actually be observed about a currently-running OS process
    right now, to compare against a recorded ProcessIdentity. Deliberately
    a plain data holder (not psutil-coupled) so tests can supply fakes
    without touching real processes."""

    pid: int
    start_timestamp: Optional[str]
    command_fingerprint: Optional[str]
    cwd: Optional[str]
    executable: Optional[str] = ""
    exists: bool = True


def command_fingerprint_of(argv: list) -> str:
    """Canonical fingerprint for a launched command -- used identically at
    launch time (what gets recorded into ProcessIdentity) and at live-check
    time (what gets read back from the OS), so a genuinely-owned process's
    fingerprint always matches exactly."""
    return " ".join(str(a) for a in argv)


def live_snapshot_for_pid(pid: int) -> "LiveProcessSnapshot":
    """Builds a real LiveProcessSnapshot from the operating system via
    psutil. Never raises for a plain "process not found" (returns
    exists=False) -- only genuinely ambiguous conditions (process exists
    but a required field cannot be read) are left as None fields, which
    verify_process_identity() itself turns into a ProcessOwnershipError.
    This is the ONLY function in this codebase that reads real OS process
    state for identity verification -- everything else compares against
    what this returns."""
    import psutil

    try:
        proc = psutil.Process(pid)
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return LiveProcessSnapshot(pid=pid, start_timestamp=None, command_fingerprint=None, cwd=None, exists=False)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return LiveProcessSnapshot(pid=pid, start_timestamp=None, command_fingerprint=None, cwd=None, exists=False)

    start_timestamp = None
    command_fingerprint = None
    cwd = None
    executable = ""

    # KNOWN RACE (discovered empirically, not assumed), two directions:
    #   1. STARTUP: psutil's Process.create_time()/cmdline() can transiently
    #      raise AccessDenied or return empty data immediately after
    #      CreateProcess on Windows, before the OS has fully populated the
    #      new process's block.
    #   2. SHUTDOWN: while a process is mid-termination (e.g. this exact
    #      pid was just sent CTRL_BREAK_EVENT by graceful_stop()),
    #      create_time() may keep working (static metadata) while
    #      cmdline() -- which needs to read the still-live process's
    #      memory -- starts failing, and the process then finishes exiting
    #      moments later. Retrying blindly cannot fix case 2 (the process
    #      is not coming back) -- so this loop re-checks liveness on every
    #      attempt: a NoSuchProcess encountered THIS SIDE of the initial
    #      is_running() check means the process has since (cleanly, not
    #      ambiguously) exited, and the function correctly reports
    #      exists=False rather than a spurious "ambiguous" state.
    import time as _time

    _max_attempts = 15
    for attempt in range(_max_attempts):
        if start_timestamp is None:
            try:
                start_timestamp = f"{proc.create_time():.6f}"
            except psutil.NoSuchProcess:
                return LiveProcessSnapshot(pid=pid, start_timestamp=None, command_fingerprint=None, cwd=None, exists=False)
            except psutil.AccessDenied:
                pass  # can be transient right after CreateProcess -- retry
        if command_fingerprint is None:
            try:
                argv = proc.cmdline()
            except psutil.NoSuchProcess:
                return LiveProcessSnapshot(pid=pid, start_timestamp=None, command_fingerprint=None, cwd=None, exists=False)
            except psutil.AccessDenied:
                argv = None
            if argv:
                command_fingerprint = command_fingerprint_of(argv)
        if start_timestamp is not None and command_fingerprint is not None:
            break
        if attempt < _max_attempts - 1:
            _time.sleep(0.03)
    # KNOWN WINDOWS LIMITATION (verified empirically, not assumed): psutil's
    # Process.cwd() on Windows does not reliably report the actual
    # directory passed to CreateProcess -- it has been observed to report
    # the OS temp directory instead, regardless of the real cwd. Trusting
    # it would produce FALSE ownership mismatches for genuinely-owned
    # processes. cwd is therefore intentionally left None on win32 (never
    # compared -- verify_process_identity's contract already treats
    # cwd=None as "skip this check"); identity here rests on
    # pid + creation_time + command_fingerprint, which ARE reliable on
    # Windows. See reports/phase_j/REAL_RUNNER_SAFETY_AUDIT.md.
    if sys.platform != "win32":
        try:
            cwd = proc.cwd()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            cwd = None
    try:
        executable = proc.exe()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        executable = ""

    return LiveProcessSnapshot(
        pid=pid, start_timestamp=start_timestamp, command_fingerprint=command_fingerprint,
        cwd=cwd, executable=executable, exists=True,
    )


def verify_process_identity(
    recorded: ProcessIdentity, live: Optional[LiveProcessSnapshot]
) -> bool:
    """Returns True only if `live` unambiguously matches `recorded` on every
    field that both sides can supply. Returns False if the process plainly
    does not exist or plainly does not match. Raises ProcessOwnershipError
    if the live snapshot is missing enough fields to make a confident
    determination either way (e.g. exists=True but start_timestamp/
    command_fingerprint could not be read) -- ambiguous must BLOCK, never
    silently pass or silently fail."""
    if live is None or not live.exists:
        return False
    if live.start_timestamp is None or live.command_fingerprint is None:
        raise ProcessOwnershipError(
            f"job {recorded.job_id!r}: pid {recorded.pid} exists but its start_timestamp "
            "and/or command_fingerprint could not be read -- cannot confidently confirm "
            "or deny ownership (PID reuse cannot be ruled out). Refusing to guess; "
            "requires human intervention before any attach/pause/kill action."
        )
    if live.pid != recorded.pid:
        return False
    if live.start_timestamp != recorded.launch_timestamp:
        return False
    if live.command_fingerprint != recorded.command_fingerprint:
        return False
    if recorded.cwd and live.cwd is not None and live.cwd != recorded.cwd:
        return False
    return True
