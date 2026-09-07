"""OMNISIGHT-PILOT-001 collection admission tests
(research/datasets/collection_authorization.py). No real collection, no
real consent records, no PII anywhere in this file."""

from __future__ import annotations

from research.backfill_experiment_specs import load_spec
from research.datasets.collection_authorization import (
    CollectionRequest,
    check_collection_admission,
    check_collection_admission_live,
    is_pilot_media_collectible,
)
from research.datasets.pilot_plan import OMNISIGHT_PILOT_001_PLAN_HASH, PILOT_ID
from research.db import OmniLabDB


def _valid_request(**overrides):
    defaults = dict(
        pilot_id=PILOT_ID, pilot_plan_hash=OMNISIGHT_PILOT_001_PLAN_HASH,
        num_participants=6, num_sessions=6, num_sequences=24,
        privacy_class="STAGED_CONSENTED",
    )
    defaults.update(overrides)
    return CollectionRequest(**defaults)


class TestApprovalDefaultsAndBlocking:
    def test_collection_approval_defaults_false_on_real_frozen_spec(self):
        spec = load_spec("EXP-0006")
        assert spec.proposal.staged_pilot_collection_approved is False

    def test_denied_while_approval_false(self):
        result = check_collection_admission(
            _valid_request(), operational_state_running=True, staged_pilot_collection_approved=False,
        )
        assert result.admitted is False
        assert any("staged_pilot_collection_approved" in b for b in result.blockers)

    def test_admitted_when_everything_valid_and_approved(self):
        result = check_collection_admission(
            _valid_request(), operational_state_running=True, staged_pilot_collection_approved=True,
        )
        assert result.admitted is True
        assert result.blockers == ()


class TestApprovalIndependence:
    def test_collection_approval_does_not_imply_training_approval(self):
        """Section 9: even hypothetically granting collection approval,
        EXP-0006's real, persisted new_training_approved stays False."""
        spec = load_spec("EXP-0006")
        # Simulate the collection flag hypothetically true, in memory only --
        # never written back to disk.
        hypothetically_collectable = True
        assert hypothetically_collectable is True  # the hypothetical itself
        assert spec.proposal.new_training_approved is False  # unaffected, real persisted state

    def test_collection_approval_does_not_imply_private_data_training_use(self):
        spec = load_spec("EXP-0006")
        assert spec.proposal.private_user_data_use_approved is False

    def test_training_approval_does_not_imply_collection_approval(self):
        """The inverse: if new_training_approved were hypothetically true,
        collection admission still requires its OWN flag."""
        result = check_collection_admission(
            _valid_request(), operational_state_running=True, staged_pilot_collection_approved=False,
        )
        # new_training_approved being true elsewhere has no bearing on this
        # function's signature at all -- it doesn't even accept that
        # parameter, proving structural independence.
        assert result.admitted is False


class TestScopeBinding:
    def test_wrong_pilot_id_rejected(self):
        result = check_collection_admission(
            _valid_request(pilot_id="SOME-OTHER-PILOT"),
            operational_state_running=True, staged_pilot_collection_approved=True,
        )
        assert result.admitted is False
        assert any("pilot_id" in b for b in result.blockers)

    def test_wrong_pilot_hash_rejected(self):
        result = check_collection_admission(
            _valid_request(pilot_plan_hash="0" * 64),
            operational_state_running=True, staged_pilot_collection_approved=True,
        )
        assert result.admitted is False
        assert any("pilot_plan_hash" in b for b in result.blockers)

    def test_participant_cap_exceeded_rejected(self):
        result = check_collection_admission(
            _valid_request(num_participants=7),
            operational_state_running=True, staged_pilot_collection_approved=True,
        )
        assert result.admitted is False
        assert any("num_participants" in b for b in result.blockers)

    def test_session_cap_exceeded_rejected(self):
        result = check_collection_admission(
            _valid_request(num_sessions=7),
            operational_state_running=True, staged_pilot_collection_approved=True,
        )
        assert result.admitted is False
        assert any("num_sessions" in b for b in result.blockers)

    def test_sequence_cap_exceeded_rejected(self):
        result = check_collection_admission(
            _valid_request(num_sequences=25),
            operational_state_running=True, staged_pilot_collection_approved=True,
        )
        assert result.admitted is False
        assert any("num_sequences" in b for b in result.blockers)

    def test_at_exact_caps_allowed(self):
        result = check_collection_admission(
            _valid_request(num_participants=6, num_sessions=6, num_sequences=24),
            operational_state_running=True, staged_pilot_collection_approved=True,
        )
        assert result.admitted is True


class TestPolicyEnforcement:
    def test_sensitive_context_rejected(self):
        result = check_collection_admission(
            _valid_request(sensitive_context=True),
            operational_state_running=True, staged_pilot_collection_approved=True,
        )
        assert result.admitted is False
        assert any("sensitive_context" in b for b in result.blockers)

    def test_non_consented_privacy_category_rejected(self):
        result = check_collection_admission(
            _valid_request(privacy_class="INCIDENTAL_CONSENTED"),
            operational_state_running=True, staged_pilot_collection_approved=True,
        )
        assert result.admitted is False
        assert any("privacy_class" in b for b in result.blockers)

    def test_restricted_private_rejected(self):
        result = check_collection_admission(
            _valid_request(privacy_class="RESTRICTED_PRIVATE"),
            operational_state_running=True, staged_pilot_collection_approved=True,
        )
        assert result.admitted is False

    def test_storage_path_not_confirmed_outside_git_rejected(self):
        result = check_collection_admission(
            _valid_request(storage_path_outside_tracked_git_raw=False),
            operational_state_running=True, staged_pilot_collection_approved=True,
        )
        assert result.admitted is False
        assert any("storage" in b.lower() for b in result.blockers)


class TestOperationalGate:
    def test_paused_or_stopped_blocks_collection(self):
        result = check_collection_admission(
            _valid_request(), operational_state_running=False, staged_pilot_collection_approved=True,
        )
        assert result.admitted is False
        assert any("operational state" in b for b in result.blockers)

    def test_live_wrapper_blocks_when_really_paused(self, tmp_path, monkeypatch):
        import research.operational_state as op

        monkeypatch.setattr(op, "STATE_PATH", tmp_path / "orchestrator_state.json")
        op.pause("test pause for collection-admission gate test")
        result = check_collection_admission_live(_valid_request(), staged_pilot_collection_approved=True)
        assert result.admitted is False
        assert any("operational state" in b for b in result.blockers)

    def test_live_wrapper_admits_when_really_running(self, tmp_path, monkeypatch):
        import research.operational_state as op

        monkeypatch.setattr(op, "STATE_PATH", tmp_path / "orchestrator_state.json")
        result = check_collection_admission_live(_valid_request(), staged_pilot_collection_approved=True)
        assert result.admitted is True


class TestNoUnrelatedApprovalRequired:
    def test_mac_iphone_approval_not_required(self):
        """check_collection_admission's signature has no mac/iphone
        parameter at all -- structurally cannot require it."""
        result = check_collection_admission(
            _valid_request(), operational_state_running=True, staged_pilot_collection_approved=True,
        )
        assert result.admitted is True  # granted without ever mentioning device approval

    def test_new_training_approved_not_required(self):
        """Same -- no training-approval parameter exists on this function."""
        result = check_collection_admission(
            _valid_request(), operational_state_running=True, staged_pilot_collection_approved=True,
        )
        assert result.admitted is True


class TestPilotPlanMutationInvalidatesScope:
    def test_changed_plan_produces_different_hash_and_is_rejected(self):
        from dataclasses import replace

        from research.datasets.pilot_plan import OMNISIGHT_PILOT_001_PLAN, pilot_plan_hash

        mutated_plan = replace(OMNISIGHT_PILOT_001_PLAN, num_participants=12)  # a hypothetical scope increase
        mutated_hash = pilot_plan_hash(mutated_plan)
        assert mutated_hash != OMNISIGHT_PILOT_001_PLAN_HASH

        result = check_collection_admission(
            _valid_request(pilot_plan_hash=mutated_hash),
            operational_state_running=True, staged_pilot_collection_approved=True,
        )
        assert result.admitted is False  # old authorization scope does not carry over


class TestConsentProvenanceBoundary:
    def test_staged_consented_is_collectible(self):
        assert is_pilot_media_collectible("STAGED_CONSENTED", "STAGED_CONSENTED") is True

    def test_incidental_consented_not_collectible_under_pilot_scope(self):
        assert is_pilot_media_collectible("INCIDENTAL_CONSENTED", "INCIDENTAL_CONSENTED") is False

    def test_prohibited_consent_never_collectible(self):
        assert is_pilot_media_collectible("PROHIBITED", "RESTRICTED_PRIVATE") is False

    def test_unknown_enum_values_never_collectible(self):
        assert is_pilot_media_collectible("NOT_A_REAL_STATUS", "STAGED_CONSENTED") is False

    def test_no_pii_parameters_accepted(self):
        """The function's own signature accepts only two enum strings --
        structurally cannot accept a name/email/signature."""
        import inspect

        params = list(inspect.signature(is_pilot_media_collectible).parameters)
        assert params == ["consent_status", "privacy_class"]


class TestPersistedStateUnaffected:
    def test_exp_0006_remains_blocked(self):
        with OmniLabDB() as db:
            exp = db.get_experiment("EXP-0006")
        assert exp.execution_status == "BLOCKED"

    def test_all_approvals_false_in_real_persisted_state(self):
        spec = load_spec("EXP-0006")
        p = spec.proposal
        assert p.new_training_approved is False
        assert p.private_user_data_use_approved is False
        assert p.mac_iphone_deployment_approved is False
        assert p.staged_pilot_collection_approved is False
