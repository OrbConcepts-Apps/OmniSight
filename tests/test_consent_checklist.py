"""OMNISIGHT-PILOT-001 consent readiness checklist tests
(research/datasets/consent_checklist.py). No real consent records, no PII."""

from __future__ import annotations

import dataclasses

from research.datasets.consent_checklist import ConsentReadinessChecklist, is_session_consent_ready


class TestChecklistDefaults:
    def test_all_fields_default_false_or_empty(self):
        checklist = ConsentReadinessChecklist()
        assert checklist.participant_pseudonymous_id == ""
        for f in dataclasses.fields(checklist):
            if f.name != "participant_pseudonymous_id":
                assert getattr(checklist, f.name) is False

    def test_default_checklist_is_not_ready(self):
        assert is_session_consent_ready(ConsentReadinessChecklist()) is False


class TestReadinessLogic:
    def _fully_ready(self):
        return ConsentReadinessChecklist(
            participant_pseudonymous_id="participant-001",
            participant_eligible_under_protocol=True,
            consent_requirement_satisfied_under_institutional_policy=True,
            participant_understands_capture=True,
            participant_understands_annotation=True,
            participant_understands_research_use=True,
            publication_choice_recorded=True,
            withdrawal_retention_terms_communicated=True,
            consent_evidence_stored_separately_from_manifest=True,
            no_pii_copied_into_tracked_artifacts=True,
        )

    def test_fully_confirmed_checklist_is_ready(self):
        assert is_session_consent_ready(self._fully_ready()) is True

    def test_missing_pseudonymous_id_never_ready(self):
        checklist = dataclasses.replace(self._fully_ready(), participant_pseudonymous_id="")
        assert is_session_consent_ready(checklist) is False

    def test_single_missing_item_blocks_readiness(self):
        checklist = dataclasses.replace(self._fully_ready(), no_pii_copied_into_tracked_artifacts=False)
        assert is_session_consent_ready(checklist) is False


class TestNoPiiFields:
    def test_no_name_email_address_signature_fields(self):
        field_names = {f.name for f in dataclasses.fields(ConsentReadinessChecklist)}
        for forbidden in ("name", "email", "address", "signature", "phone"):
            assert not any(forbidden in fn.lower() for fn in field_names)
