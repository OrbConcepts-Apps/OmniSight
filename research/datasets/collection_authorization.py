"""OMNISIGHT-PILOT-001 collection admission -- deterministic, separate from
experiment registration, queue admission, and training execution admission
(Phase authorization: "Implement deterministic collection admission
separate from: experiment registration; queue admission; training
execution admission.").

No real consent-management system is built here -- `is_pilot_media_collectible()`
is a minimal boundary function proving a media record cannot be treated as
collectible unless its provenance satisfies the staged-consented policy. No
names, emails, or signatures are represented anywhere in this module --
only pseudonymous ids and policy enums (see
research.datasets.manifest_schema for the shared vocabulary).
"""

from __future__ import annotations

from dataclasses import dataclass

from research.datasets.manifest_schema import CONSENT_STATUSES, PRIVACY_CLASSES
from research.datasets.pilot_plan import (
    OMNISIGHT_PILOT_001_MAX_SEQUENCES,
    OMNISIGHT_PILOT_001_PLAN,
    OMNISIGHT_PILOT_001_PLAN_HASH,
    PILOT_ID,
)

# Only staged-consented media may be collected under this pilot's scope
# (Phase authorization section 6) -- narrower than the general dataset
# schema's PRIVACY_CLASSES, which also allows INCIDENTAL_CONSENTED for the
# eventual full dataset.
PILOT_ALLOWED_PRIVACY_CLASS = "STAGED_CONSENTED"
PILOT_ALLOWED_CONSENT_STATUS = "STAGED_CONSENTED"


@dataclass(frozen=True)
class CollectionRequest:
    """What a future collection-admission caller declares before any
    capture happens -- no real media referenced here."""

    pilot_id: str
    pilot_plan_hash: str
    num_participants: int
    num_sessions: int
    num_sequences: int
    privacy_class: str
    sensitive_context: bool = False
    storage_path_outside_tracked_git_raw: bool = True


@dataclass(frozen=True)
class CollectionAdmissionResult:
    admitted: bool
    blockers: tuple  # tuple[str, ...] -- empty iff admitted


def check_collection_admission(
    request: CollectionRequest,
    *,
    operational_state_running: bool,
    staged_pilot_collection_approved: bool,
) -> CollectionAdmissionResult:
    """Pure, deterministic. Never admits unless EVERY check passes --
    fail-closed, same discipline as research.execution_budget.
    require_execution_budget(). Does not require new_training_approved or
    mac_iphone_deployment_approved for ordinary staged-consented media
    capture (Phase authorization section 7)."""
    blockers = []

    if not operational_state_running:
        blockers.append("operational state is not RUNNING")

    if request.pilot_id != PILOT_ID:
        blockers.append(f"pilot_id {request.pilot_id!r} is not the approved pilot {PILOT_ID!r}")

    if request.pilot_plan_hash != OMNISIGHT_PILOT_001_PLAN_HASH:
        blockers.append(
            f"pilot_plan_hash {request.pilot_plan_hash!r} does not match the current frozen "
            f"plan hash {OMNISIGHT_PILOT_001_PLAN_HASH!r} -- the plan has changed since this "
            "request was scoped, or the request is stale/mistargeted"
        )

    if not staged_pilot_collection_approved:
        blockers.append("staged_pilot_collection_approved is False")

    if request.num_participants > OMNISIGHT_PILOT_001_PLAN.num_participants:
        blockers.append(
            f"num_participants={request.num_participants} exceeds the approved cap of "
            f"{OMNISIGHT_PILOT_001_PLAN.num_participants}"
        )
    if request.num_sessions > OMNISIGHT_PILOT_001_PLAN.num_sessions:
        blockers.append(
            f"num_sessions={request.num_sessions} exceeds the approved cap of "
            f"{OMNISIGHT_PILOT_001_PLAN.num_sessions}"
        )
    if request.num_sequences > OMNISIGHT_PILOT_001_MAX_SEQUENCES:
        blockers.append(
            f"num_sequences={request.num_sequences} exceeds the approved cap of "
            f"{OMNISIGHT_PILOT_001_MAX_SEQUENCES}"
        )

    if request.privacy_class != PILOT_ALLOWED_PRIVACY_CLASS:
        blockers.append(
            f"privacy_class {request.privacy_class!r} is not {PILOT_ALLOWED_PRIVACY_CLASS!r} "
            "-- this pilot's scope permits staged/consented collection only"
        )

    if request.sensitive_context:
        blockers.append(
            "sensitive_context=True -- minors/schools/medical/bathrooms-changing-areas/"
            "private-home-interiors/other sensitive contexts are never authorized"
        )

    if not request.storage_path_outside_tracked_git_raw:
        blockers.append(
            "storage path is not confirmed outside the tracked git raw-data area -- refusing "
            "to admit a collection request with no confirmed safe storage target"
        )

    return CollectionAdmissionResult(admitted=len(blockers) == 0, blockers=tuple(blockers))


def check_collection_admission_live(
    request: CollectionRequest, *, staged_pilot_collection_approved: bool,
) -> CollectionAdmissionResult:
    """Convenience wrapper reading the REAL operational state (RUNNING iff
    not paused and not stopped) -- the pure function above stays fully
    unit-testable without touching research.operational_state's real,
    shared state file."""
    from research.operational_state import current_state

    state = current_state()
    running = not state.paused and not state.stopped
    return check_collection_admission(
        request, operational_state_running=running,
        staged_pilot_collection_approved=staged_pilot_collection_approved,
    )


def is_pilot_media_collectible(consent_status: str, privacy_class: str) -> bool:
    """Minimal consent/provenance boundary (Phase authorization section
    11): a media record can never be treated as collectible under this
    pilot's scope unless BOTH its consent_status and privacy_class are
    exactly the pilot's allowed staged-consented values -- no real
    consent-management system, no name/email/signature represented
    anywhere, pseudonymous policy enums only."""
    if consent_status not in CONSENT_STATUSES or privacy_class not in PRIVACY_CLASSES:
        return False
    return consent_status == PILOT_ALLOWED_CONSENT_STATUS and privacy_class == PILOT_ALLOWED_PRIVACY_CLASS
