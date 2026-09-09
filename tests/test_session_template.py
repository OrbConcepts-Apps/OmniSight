"""OMNISIGHT-PILOT-001 empty session-record template tests
(research/datasets/session_template.py). No real session data."""

from __future__ import annotations

import dataclasses

from research.datasets.ethics_review import NOT_ASSESSED
from research.datasets.pilot_plan import OMNISIGHT_PILOT_001_PLAN_HASH, PILOT_ID
from research.datasets.session_template import PilotSessionRecord, build_empty_session_template


class TestEmptyTemplate:
    def test_template_is_empty(self):
        template = build_empty_session_template()
        assert template["session_id"] == ""
        assert template["participant_pseudonymous_id"] == ""
        assert template["sequence_ids"] == ()
        assert template["collection_counters"] == {"sequences_recorded": 0, "frames_sampled": 0}

    def test_template_carries_pilot_identity_and_defaults(self):
        template = build_empty_session_template()
        assert template["pilot_id"] == PILOT_ID
        assert template["pilot_plan_hash"] == OMNISIGHT_PILOT_001_PLAN_HASH
        assert template["ethics_review_status_reference"] == NOT_ASSESSED


class TestNoPiiOrFabricatedData:
    def test_no_pii_fields(self):
        field_names = {f.name for f in dataclasses.fields(PilotSessionRecord)}
        for forbidden in ("name", "email", "address", "phone", "signature"):
            assert not any(forbidden in fn.lower() for fn in field_names)

    def test_no_media_path_field(self):
        field_names = {f.name for f in dataclasses.fields(PilotSessionRecord)}
        assert not any("path" in fn.lower() and "storage" not in fn.lower() for fn in field_names)

    def test_start_end_metadata_is_a_policy_not_a_real_timestamp(self):
        record = PilotSessionRecord()
        assert record.start_end_metadata_policy in ("DATE_ONLY", "NONE", "FULL")
