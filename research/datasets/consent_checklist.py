"""OMNISIGHT-PILOT-001 consent readiness checklist -- a TEMPLATE proving
what must be true before an individual session, never a completed consent
record. No signatures, no names, no fake consent evidence -- pseudonymous
references only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConsentReadinessChecklist:
    """Every field defaults False -- a checklist is "ready" only once a
    human has actually confirmed each item for a real, specific session.
    This dataclass never holds a name/email/signature -- only booleans and
    a pseudonymous reference id."""

    participant_pseudonymous_id: str = ""
    participant_eligible_under_protocol: bool = False
    consent_requirement_satisfied_under_institutional_policy: bool = False
    participant_understands_capture: bool = False
    participant_understands_annotation: bool = False
    participant_understands_research_use: bool = False
    publication_choice_recorded: bool = False
    withdrawal_retention_terms_communicated: bool = False
    consent_evidence_stored_separately_from_manifest: bool = False
    no_pii_copied_into_tracked_artifacts: bool = False


def is_session_consent_ready(checklist: ConsentReadinessChecklist) -> bool:
    """True only if every readiness item is confirmed AND a pseudonymous
    id has actually been assigned (an empty id is never ready, even if
    every boolean happens to be True)."""
    if not checklist.participant_pseudonymous_id:
        return False
    return all(
        (
            checklist.participant_eligible_under_protocol,
            checklist.consent_requirement_satisfied_under_institutional_policy,
            checklist.participant_understands_capture,
            checklist.participant_understands_annotation,
            checklist.participant_understands_research_use,
            checklist.publication_choice_recorded,
            checklist.withdrawal_retention_terms_communicated,
            checklist.consent_evidence_stored_separately_from_manifest,
            checklist.no_pii_copied_into_tracked_artifacts,
        )
    )
