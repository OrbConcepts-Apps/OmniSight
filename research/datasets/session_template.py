"""OMNISIGHT-PILOT-001 empty session-record template. Logical/pseudonymous
metadata only -- no names, no emails, no addresses, no real timestamps
pretending collection happened, no media paths pretending files exist.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from research.datasets.ethics_review import NOT_ASSESSED
from research.datasets.pilot_plan import OMNISIGHT_PILOT_001_PLAN_HASH, PILOT_ID


@dataclass(frozen=True)
class PilotSessionRecord:
    pilot_id: str = PILOT_ID
    pilot_plan_hash: str = OMNISIGHT_PILOT_001_PLAN_HASH
    participant_pseudonymous_id: str = ""
    session_id: str = ""
    sequence_ids: tuple = field(default_factory=tuple)
    planned_scenario_ids: tuple = field(default_factory=tuple)
    capture_device_logical_id: str = ""
    consent_status_reference: str = ""  # points at a consent record kept elsewhere, never the record itself
    ethics_review_status_reference: str = NOT_ASSESSED
    start_end_metadata_policy: str = "DATE_ONLY"  # matches MediaUnitRecord.capture_timestamp_policy's vocabulary
    exclusions: tuple = field(default_factory=tuple)
    collection_counters: dict = field(default_factory=lambda: {"sequences_recorded": 0, "frames_sampled": 0})

    def to_dict(self) -> dict:
        return asdict(self)


def build_empty_session_template() -> dict:
    """A single EMPTY session record -- no session_id, no participant id,
    zero counters, no populated sequence/scenario ids. Never a completed
    record."""
    return PilotSessionRecord().to_dict()
