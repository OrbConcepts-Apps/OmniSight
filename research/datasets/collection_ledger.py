"""OMNISIGHT-PILOT-001 cap-enforcement ledger + sequence admission +
bystander stop rule + pilot-data default status. Metadata-only -- no real
media is created or referenced anywhere in this module; every test
exercising it uses synthetic ids.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from research.config import REPO_ROOT
from research.datasets.pilot_plan import (
    OMNISIGHT_PILOT_001_MAX_SEQUENCES,
    OMNISIGHT_PILOT_001_PLAN,
    OMNISIGHT_PILOT_001_PLAN_HASH,
    PILOT_ID,
)

LEDGER_PATH = REPO_ROOT / "research" / "datasets" / "pilot_manifests" / "OMNISIGHT-PILOT-001-ledger.json"

# A future collected record's status until an explicit, later, separate
# human decision promotes it -- never automatic (Phase authorization
# section 12).
PILOT_PLANNING_ONLY = "PILOT_PLANNING_ONLY"

# Field rule (Phase authorization section 11): no automatic face-redaction
# workaround is engineered here -- the rule is procedural, applied by the
# person collecting, not a code path that "fixes" a bad sequence.
BYSTANDER_STOP_RULE = (
    "If an unplanned non-consenting person becomes materially identifiable in a staged "
    "sequence, the collector stops or invalidates/excludes that ENTIRE sequence under the "
    "protocol -- never treats the bystander as training data, and never applies automatic "
    "face redaction as a substitute for exclusion."
)


class PilotCapExceededError(RuntimeError):
    pass


@dataclass
class LedgerState:
    participant_ids: list = field(default_factory=list)
    session_ids: list = field(default_factory=list)
    sequence_ids: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "participant_ids": list(self.participant_ids),
            "session_ids": list(self.session_ids),
            "sequence_ids": list(self.sequence_ids),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LedgerState":
        return cls(
            participant_ids=list(data.get("participant_ids", [])),
            session_ids=list(data.get("session_ids", [])),
            sequence_ids=list(data.get("sequence_ids", [])),
        )


class PilotCollectionLedger:
    """Deterministic counters -- never relies on documentation alone
    (Phase authorization section 9). Every register_* call is idempotent
    for an already-registered id (never double-counts), and raises
    PilotCapExceededError, never silently truncates, once a cap would be
    exceeded by a genuinely NEW id."""

    def __init__(self, path: Path = LEDGER_PATH):
        self.path = path

    def _load(self) -> LedgerState:
        if not self.path.exists():
            return LedgerState()
        try:
            return LedgerState.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            # Corrupt ledger -- fail closed by reporting caps as already
            # exhausted, never silently reset to empty (which could let a
            # cap be quietly exceeded across a corruption event).
            return LedgerState(
                participant_ids=["<corrupt-ledger>"] * OMNISIGHT_PILOT_001_PLAN.num_participants,
                session_ids=["<corrupt-ledger>"] * OMNISIGHT_PILOT_001_PLAN.num_sessions,
                sequence_ids=["<corrupt-ledger>"] * OMNISIGHT_PILOT_001_MAX_SEQUENCES,
            )

    def _save(self, state: LedgerState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def counts(self) -> dict:
        state = self._load()
        return {
            "participants": len(state.participant_ids),
            "sessions": len(state.session_ids),
            "sequences": len(state.sequence_ids),
        }

    def register_participant(self, participant_id: str) -> None:
        state = self._load()
        if participant_id in state.participant_ids:
            return
        if len(state.participant_ids) >= OMNISIGHT_PILOT_001_PLAN.num_participants:
            raise PilotCapExceededError(
                f"refusing participant {participant_id!r}: "
                f"{len(state.participant_ids)} of {OMNISIGHT_PILOT_001_PLAN.num_participants} "
                "approved participant slots already used"
            )
        state.participant_ids.append(participant_id)
        self._save(state)

    def register_session(self, session_id: str, *, participant_id: str) -> None:
        state = self._load()
        if participant_id not in state.participant_ids:
            raise PilotCapExceededError(
                f"refusing session {session_id!r}: participant {participant_id!r} is not registered"
            )
        if session_id in state.session_ids:
            return
        if len(state.session_ids) >= OMNISIGHT_PILOT_001_PLAN.num_sessions:
            raise PilotCapExceededError(
                f"refusing session {session_id!r}: "
                f"{len(state.session_ids)} of {OMNISIGHT_PILOT_001_PLAN.num_sessions} "
                "approved session slots already used"
            )
        state.session_ids.append(session_id)
        self._save(state)

    def register_sequence(self, sequence_id: str, *, session_id: str) -> None:
        state = self._load()
        if session_id not in state.session_ids:
            raise PilotCapExceededError(
                f"refusing sequence {sequence_id!r}: session {session_id!r} is not registered"
            )
        if sequence_id in state.sequence_ids:
            return
        if len(state.sequence_ids) >= OMNISIGHT_PILOT_001_MAX_SEQUENCES:
            raise PilotCapExceededError(
                f"refusing sequence {sequence_id!r}: "
                f"{len(state.sequence_ids)} of {OMNISIGHT_PILOT_001_MAX_SEQUENCES} "
                "approved sequence slots already used"
            )
        state.sequence_ids.append(sequence_id)
        self._save(state)


# ---------------------------------------------------------------------------
# Sequence admission (Phase authorization section 10) -- metadata-only.
# ---------------------------------------------------------------------------

from research.datasets.collection_authorization import CollectionAdmissionResult  # noqa: E402


@dataclass(frozen=True)
class SequenceAdmissionRequest:
    pilot_id: str
    pilot_plan_hash: str
    session_id: str
    scenario_id: str
    privacy_class: str
    sensitive_context: bool = False
    unconsented_bystander_present: bool = False
    storage_path_outside_tracked_git_raw: bool = True


def check_sequence_admission(
    request: SequenceAdmissionRequest,
    *,
    session_is_registered: bool,
    cap_available: bool,
) -> CollectionAdmissionResult:
    blockers = []
    if request.pilot_id != PILOT_ID:
        blockers.append(f"pilot_id {request.pilot_id!r} is not {PILOT_ID!r}")
    if request.pilot_plan_hash != OMNISIGHT_PILOT_001_PLAN_HASH:
        blockers.append("pilot_plan_hash does not match the current frozen plan")
    if not session_is_registered:
        blockers.append(f"session {request.session_id!r} is not a registered session")
    if not cap_available:
        blockers.append("sequence cap already exhausted")
    if request.privacy_class != "STAGED_CONSENTED":
        blockers.append(f"privacy_class {request.privacy_class!r} is not STAGED_CONSENTED")
    if request.sensitive_context:
        blockers.append("sensitive_context=True -- never authorized")
    if request.unconsented_bystander_present:
        blockers.append(f"unconsented bystander present -- {BYSTANDER_STOP_RULE}")
    if not request.storage_path_outside_tracked_git_raw:
        blockers.append("storage path not confirmed outside the tracked git raw-data area")
    return CollectionAdmissionResult(admitted=len(blockers) == 0, blockers=tuple(blockers))
