"""Phase J -- append-only resource-consumption ledger.

One JSON object per line, appended, never rewritten or deleted -- the
ledger itself is the audit trail. Per Phase J authorization section 10: "If
exact GPU utilization cannot be reliably measured, document the limitation
and use conservative wall-clock GPU-job occupancy as the enforceable budget
metric. Never fabricate precision that telemetry cannot support." This
module reports wall-clock occupancy only; it does not claim to measure true
GPU-active cycles (see PHASE_J_IMPLEMENTATION.md's residual-limitations
section).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from research.config import REPO_ROOT

LEDGER_PATH = REPO_ROOT / "research" / "execution_ledger.jsonl"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class LedgerEntry:
    job_id: str
    experiment_id: str
    event: str  # e.g. "STARTED" | "FINISHED"
    at: str
    elapsed_sec: Optional[float] = None
    gpu_active_sec: Optional[float] = None  # None -- unmeasurable, never fabricated
    termination_reason: str = ""
    state: str = ""

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "experiment_id": self.experiment_id,
            "event": self.event,
            "at": self.at,
            "elapsed_sec": self.elapsed_sec,
            "gpu_active_sec": self.gpu_active_sec,
            "termination_reason": self.termination_reason,
            "state": self.state,
        }


def record_event(
    *,
    job_id: str,
    experiment_id: str,
    event: str,
    elapsed_sec: Optional[float] = None,
    gpu_active_sec: Optional[float] = None,
    termination_reason: str = "",
    state: str = "",
    ledger_path: Path = LEDGER_PATH,
) -> LedgerEntry:
    entry = LedgerEntry(
        job_id=job_id,
        experiment_id=experiment_id,
        event=event,
        at=_utcnow(),
        elapsed_sec=elapsed_sec,
        gpu_active_sec=gpu_active_sec,
        termination_reason=termination_reason,
        state=state,
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
    return entry


def read_ledger(ledger_path: Path = LEDGER_PATH) -> list:
    """Returns every entry, in append order. Skips (never raises on) any
    corrupt individual line -- a single bad line must not make the whole
    ledger unreadable."""
    if not ledger_path.exists():
        return []
    entries = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def cumulative_wall_clock_sec(
    *,
    on_date: Optional[str] = None,
    experiment_id: Optional[str] = None,
    ledger_path: Path = LEDGER_PATH,
) -> float:
    """Sums `elapsed_sec` across FINISHED entries, optionally filtered to a
    given UTC calendar date (YYYY-MM-DD, matched against the entry's `at`
    prefix) and/or a given experiment_id. This is the wall-clock-occupancy
    figure Phase J authorization section 10 designates as the enforceable
    budget metric in place of unmeasurable true GPU-active time."""
    total = 0.0
    for entry in read_ledger(ledger_path):
        if entry.get("event") != "FINISHED":
            continue
        if on_date is not None and not str(entry.get("at", "")).startswith(on_date):
            continue
        if experiment_id is not None and entry.get("experiment_id") != experiment_id:
            continue
        elapsed = entry.get("elapsed_sec")
        if elapsed is not None:
            total += float(elapsed)
    return total
