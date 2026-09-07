"""Phase J -- active monotonic-clock wall-clock watchdog.

Closes the restriction noted in reports/phase_j/PHASE_J_SAFETY_AUDIT.md:
`JobManager.enforce_wall_clock()` previously only reacted to a caller-
supplied elapsed-time value once, on demand. This module actively polls a
running job on a bounded interval, using a real (or injectable, for tests)
monotonic clock, and drives it through `JobManager.enforce_wall_clock()`'s
existing graceful-stop-then-escalate logic automatically.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from research.execution_job import state
from research.execution_job.manager import JobManager
from research.execution_job.process_ownership import LiveProcessSnapshot


@dataclass(frozen=True)
class WatchdogConfig:
    max_wall_clock_sec: int
    poll_interval_sec: float = 1.0
    grace_period_sec: float = 5.0


def run_watchdog(
    manager: JobManager,
    record: state.JobRecord,
    config: WatchdogConfig,
    *,
    live_snapshot_fn: Callable[[state.JobRecord], Optional[LiveProcessSnapshot]],
    is_complete_fn: Callable[[state.JobRecord], bool],
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> state.JobRecord:
    """Actively polls until the job completes/fails on its own, or the
    wall-clock budget is exceeded, in which case it drives the job through
    graceful-stop -> (grace period) -> forced-termination -> TIMED_OUT via
    `manager.enforce_wall_clock()`.

    `clock`/`sleep` are injectable so tests never need a real wait longer
    than a fraction of a second; `live_snapshot_fn`/`is_complete_fn` let
    the caller supply how to check the real (or fake) process without this
    module needing to know about any specific Runner implementation.

    Never sleeps for the full `max_wall_clock_sec` in one call -- it wakes
    every `poll_interval_sec` so a job that finishes early returns
    immediately, and so the grace-period escalation itself proceeds in
    bounded, observable steps rather than one long blocking sleep."""
    start = clock()
    while True:
        if is_complete_fn(record):
            return record

        elapsed = clock() - start
        record = manager.enforce_wall_clock(
            record,
            elapsed_sec=elapsed,
            max_wall_clock_sec=config.max_wall_clock_sec,
            live_snapshot=live_snapshot_fn(record),
            grace_expired=False,
        )
        if record.state in state.TERMINAL_STATES:
            return record

        if elapsed > config.max_wall_clock_sec:
            # Already over budget and enforce_wall_clock's first call above
            # only requested a graceful stop -- wait out the grace period,
            # then escalate for real.
            sleep(config.grace_period_sec)
            record = manager.enforce_wall_clock(
                record,
                elapsed_sec=clock() - start,
                max_wall_clock_sec=config.max_wall_clock_sec,
                live_snapshot=live_snapshot_fn(record),
                grace_expired=True,
            )
            return record

        sleep(min(config.poll_interval_sec, max(config.max_wall_clock_sec - elapsed, 0.01)))
