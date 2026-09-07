"""EXP-0006 Amendment 001 tests (research/amend_exp_0006_001.py). Reads the
REAL, already-amended registered spec (the amendment was applied once, for
real, as part of this authorized task) plus the guard behavior on a
would-be second application. No live LLM calls, no training, no data
collection."""

from __future__ import annotations

import pytest

from research.amend_exp_0006_001 import NEW_QA_SENTENCE, OLD_QA_SENTENCE, apply_amendment
from research.backfill_experiment_specs import load_spec
from research.db import OmniLabDB
from research.preregistration import load_preregistration


class TestAmendmentApplied:
    def test_old_ambiguous_sentence_no_longer_present(self):
        spec = load_spec("EXP-0006")
        assert OLD_QA_SENTENCE not in spec.proposal.isolation_requirements

    def test_new_wording_present_and_marks_thresholds_provisional(self):
        spec = load_spec("EXP-0006")
        text = spec.proposal.isolation_requirements
        assert "PREREGISTERED PILOT QA THRESHOLD" in text
        assert "not yet evidence-backed" in text
        assert "PROVISIONAL" in text

    def test_wording_specifies_all_required_elements(self):
        """Section 2's minimum content list."""
        spec = load_spec("EXP-0006")
        text = spec.proposal.isolation_requirements
        for required_phrase in (
            "greedy IoU assignment",           # box matching
            "ignore/ambiguous regions",         # ignore-region treatment
            "intersection-over-union",           # IoU metric definition
            "per-image",                          # per-image vs aggregate
            "unmatched-box rate",                  # unmatched-box treatment
            "double-annotated at 100%",             # minimum sampled quantity
            "held at annotation_status=FIRST_PASS",  # pass/fail behavior
            "re-annotated or adjudicated",            # adjudication behavior
        ):
            assert required_phrase in text, f"missing required amendment element: {required_phrase!r}"

    def test_amendment_history_integrity(self):
        spec = load_spec("EXP-0006")
        assert len(spec.amendments) == 1
        a = spec.amendments[0]
        assert a.field_name == "isolation_requirements"
        assert a.old_value == OLD_QA_SENTENCE or OLD_QA_SENTENCE in str(a.old_value)
        assert a.reason
        assert a.approved_by

    def test_frozen_hash_changed_and_verifies(self):
        spec = load_spec("EXP-0006")
        spec.verify_integrity()  # must not raise
        assert spec.frozen_hash is not None

    def test_original_frozen_hash_recoverable_from_preregistration(self):
        """The ORIGINAL (pre-amendment) hash is preserved -- not in the
        amended twin's frozen_hash (which is now the post-amendment hash
        by design), but in the untouched original preregistration."""
        preregistration = load_preregistration()
        preregistration.verify_integrity()
        assert preregistration.frozen_hash == "e0b4a954e6a6ffb1c3bd067d8a10363dafe236a3c386d51949300a79a806289a"
        # And the amended twin's hash is DIFFERENT from the original.
        amended = load_spec("EXP-0006")
        assert amended.frozen_hash != preregistration.frozen_hash

    def test_preregistration_artifact_untouched(self):
        """research/preregistrations/EXP-0006-PROPOSED.json is NEVER
        amended -- it still contains the original ambiguous wording,
        proving Amendment 001 only touched the registered twin."""
        preregistration = load_preregistration()
        assert OLD_QA_SENTENCE in preregistration.proposal.isolation_requirements

    def test_db_amendment_event_visible(self):
        with OmniLabDB() as db:
            events = db.get_events("EXP-0006")
        assert any("amend:isolation_requirements" in (e.from_status or "") for e in events)

    def test_exp_0006_still_blocked_verdict_pending(self):
        with OmniLabDB() as db:
            exp = db.get_experiment("EXP-0006")
        assert exp.execution_status == "BLOCKED"
        assert exp.research_verdict == "PENDING"

    def test_approvals_still_all_false(self):
        spec = load_spec("EXP-0006")
        p = spec.proposal
        assert p.new_training_approved is False
        assert p.private_user_data_use_approved is False
        assert p.mac_iphone_deployment_approved is False


class TestGuardAgainstBlindReapplication:
    def test_reapplying_after_already_amended_refuses(self):
        """apply_amendment() checks the OLD sentence is present verbatim
        before touching anything -- since it's already gone (amended for
        real), calling it again must refuse rather than silently no-op or
        double-amend."""
        with pytest.raises(RuntimeError):
            apply_amendment()

    def test_new_sentence_is_a_real_constant_not_fabricated_at_call_time(self):
        assert "PREREGISTERED PILOT QA THRESHOLD" in NEW_QA_SENTENCE
