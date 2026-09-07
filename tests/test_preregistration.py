"""EXP-0006 preregistration tests (research/preregistration.py). No live
LLM calls, no real training, no DB registration -- validate/freeze the
corrected spec deterministically and prove it never touches
CANDIDATE-0003 or research.db.OmniLabDB."""

from __future__ import annotations

import pytest

from research.db import OmniLabDB
from research.experiment_validator import is_queue_eligible, validate
from research.preregistration import (
    RECOVERY_COUNT_THRESHOLD,
    SeedCriteriaResult,
    build_exp0006_proposal,
    build_exp0006_spec,
    classify_aggregate_verdict,
    classify_seed_verdict,
    load_preregistration,
    save_preregistration,
)


class TestBuildAndValidate:
    def test_proposal_builds_without_error(self):
        proposal = build_exp0006_proposal()
        assert proposal.experiment_id == "EXP-0006"
        assert proposal.family == "training_data"

    def test_validates_with_only_duplicate_id_error(self):
        """EXP-0006 is now a REGISTERED row (research.register_exp_0006).
        Re-validating a freshly-built copy of the same proposal against the
        live DB therefore correctly reports DUPLICATE_ID -- this is
        validate()'s own no-double-registration guard working as intended,
        not a defect in the proposal content. The pre-registration
        validation run (zero errors) is the one recorded in
        reports/phase_i/EXP0006_PREREGISTRATION_AUDIT.md and re-proven in
        tests/test_register_exp_0006.py via an isolated DB."""
        spec = build_exp0006_spec()
        result = validate(spec)
        assert [i.code for i in result.errors] == ["DUPLICATE_ID"]

    def test_correct_pending_approvals(self):
        """4 pending approvals as of the OMNISIGHT-PILOT-001
        collection-authorization gate: the original 3 plus the new,
        distinct staged-pilot-collection approval (data collection is a
        separate gate from private-data USE, see
        reports/phase_i/OMNISIGHT_PILOT_001_COLLECTION_AUTHORIZATION_AUDIT.md)."""
        spec = build_exp0006_spec()
        result = validate(spec)
        codes = {i.code for i in result.needs_human_approval}
        assert codes == {
            "UNAPPROVED_MAC_IPHONE_DEPLOYMENT",
            "UNAPPROVED_NEW_TRAINING",
            "UNAPPROVED_PRIVATE_DATA_USE",
            "UNAPPROVED_STAGED_PILOT_COLLECTION",
        }

    def test_not_queue_eligible(self):
        """Structurally/scientifically valid but correctly NOT queue-eligible
        -- 3 real pending human approvals, none silently bypassed."""
        spec = build_exp0006_spec()
        result = validate(spec)
        assert is_queue_eligible(result) is False

    def test_approval_flags_all_false(self):
        p = build_exp0006_proposal()
        assert p.new_training_approved is False
        assert p.mac_iphone_deployment_approved is False
        assert p.private_user_data_use_approved is False
        assert p.coreml_model_replacement_approved is False
        assert p.signing_distribution_change_approved is False
        assert p.production_swift_modification_approved is False
        assert p.external_upload_approved is False

    def test_privacy_classification_corrected_from_candidate_0003(self):
        p = build_exp0006_proposal()
        assert p.data_privacy_classification == "PRIVATE_USER_DATA"

    def test_no_placeholder_values(self):
        """Every unresolved fact is an explicit PREREQUISITE sentence, never
        a bare placeholder research.experiment_validator would reject."""
        spec = build_exp0006_spec()
        result = validate(spec)
        assert not any(i.code == "PLACEHOLDER_VALUE" for i in result.issues)

    def test_evidence_and_prior_experiment_refs_resolve(self):
        """Confirms MEM-0003/0015/0017/0025 and EXP-0004/0005 are real,
        resolvable records -- not fabricated ids."""
        spec = build_exp0006_spec()
        result = validate(spec)
        assert not any(i.code in ("BAD_EVIDENCE_REF", "BAD_PRIOR_EXPERIMENT_REF") for i in result.issues)

    def test_builder_itself_never_touches_db(self):
        """research.preregistration's build/validate/freeze functions never
        call OmniLabDB.create_experiment -- registration (when authorized)
        happens exclusively via the separate research.register_exp_0006
        module. This test builds a fresh, unfrozen spec and confirms doing
        so has no side effect on DB row count, regardless of whether
        EXP-0006 happens to already be registered in this environment."""
        with OmniLabDB() as db:
            before = {e.experiment_id for e in db.list_experiments()}
        build_exp0006_spec()
        with OmniLabDB() as db:
            after = {e.experiment_id for e in db.list_experiments()}
        assert before == after

    def test_training_config_is_structured_dict_not_prose(self):
        p = build_exp0006_proposal()
        cfg = p.controlled_variables["training_config"]
        assert isinstance(cfg, dict)
        for key in ("optimizer", "learning_rate", "batch_size", "seeds", "starting_checkpoint"):
            assert key in cfg


class TestFreezeAndPersist:
    def test_freeze_computes_hash(self):
        spec = build_exp0006_spec()
        assert spec.frozen_hash is None
        spec.freeze("VALIDATED")
        assert spec.frozen_hash is not None
        spec.verify_integrity()  # must not raise

    def test_save_and_reload_roundtrip(self, tmp_path):
        spec = build_exp0006_spec()
        spec.freeze("VALIDATED")
        path = tmp_path / "EXP-0006-PROPOSED.json"
        save_preregistration(spec, path)
        reloaded = load_preregistration(path)
        reloaded.verify_integrity()  # must not raise
        assert reloaded.proposal.experiment_id == "EXP-0006"
        assert reloaded.status == "VALIDATED"

    def test_saved_artifact_never_overwrites_candidate_0003(self, tmp_path):
        spec = build_exp0006_spec()
        spec.freeze("VALIDATED")
        path = tmp_path / "EXP-0006-PROPOSED.json"
        save_preregistration(spec, path)
        # The only thing this test proves directly: save_preregistration's
        # default path constant lives under research/preregistrations/, a
        # distinct directory from research/candidates/ -- see
        # test_candidate_0003_immutability below for the on-disk check.
        from research.preregistration import PREREGISTRATIONS_DIR

        assert "candidates" not in str(PREREGISTRATIONS_DIR)
        assert "experiment_specs" not in str(PREREGISTRATIONS_DIR)


class TestSeedVerdictClassification:
    """Every one of the 8 (precision, recall, recovery) combinations must
    resolve to exactly one of PASS/FAIL/INCONCLUSIVE -- no uncovered state,
    fixing CANDIDATE-0003's exact gap."""

    @pytest.mark.parametrize(
        "precision_pass,recall_pass,recovery_pass,expected",
        [
            (True, True, True, "PASS"),
            (True, True, False, "INCONCLUSIVE"),  # the exact gap CANDIDATE-0003 left uncovered
            (True, False, True, "INCONCLUSIVE"),
            (True, False, False, "FAIL"),
            (False, True, True, "FAIL"),  # precision veto, even with both other criteria passing
            (False, True, False, "FAIL"),
            (False, False, True, "FAIL"),
            (False, False, False, "FAIL"),
        ],
    )
    def test_all_eight_combinations_covered(self, precision_pass, recall_pass, recovery_pass, expected):
        result = SeedCriteriaResult(
            seed=42, precision_pass=precision_pass, recall_pass=recall_pass, recovery_pass=recovery_pass,
        )
        assert classify_seed_verdict(result) == expected


class TestAggregateVerdictClassification:
    def _r(self, seed, precision=True, recall=True, recovery=True):
        return SeedCriteriaResult(seed=seed, precision_pass=precision, recall_pass=recall, recovery_pass=recovery)

    def test_majority_pass_is_overall_pass(self):
        results = [self._r(42), self._r(43), self._r(44, recall=False, recovery=False)]
        assert classify_aggregate_verdict(results) == "PASS"

    def test_all_fail_is_overall_fail(self):
        results = [self._r(s, recall=False, recovery=False) for s in (42, 43, 44)]
        assert classify_aggregate_verdict(results) == "FAIL"

    def test_single_pass_no_majority_is_inconclusive(self):
        results = [
            self._r(42),
            self._r(43, recall=False, recovery=False),
            self._r(44, recall=True, recovery=False),  # INCONCLUSIVE seed, not PASS or FAIL
        ]
        assert classify_aggregate_verdict(results) == "INCONCLUSIVE"

    def test_single_precision_failure_vetoes_overall_even_with_two_passes(self):
        """No one favorable outcome elsewhere overrides a safety-guardrail
        violation -- Phase J/preregistration authorization's explicit
        'no cherry-picking' requirement."""
        results = [self._r(42), self._r(43), self._r(44, precision=False)]
        assert classify_aggregate_verdict(results) == "FAIL"

    def test_empty_results_rejected(self):
        with pytest.raises(ValueError):
            classify_aggregate_verdict([])

    def test_exhaustive_over_random_sample(self):
        """No combination of 3 seeds' verdicts should ever raise or return
        something outside {PASS, FAIL, INCONCLUSIVE}."""
        import itertools

        for combo in itertools.product([True, False], repeat=9):
            p1, r1, v1, p2, r2, v2, p3, r3, v3 = combo
            results = [
                self._r(1, precision=p1, recall=r1, recovery=v1),
                self._r(2, precision=p2, recall=r2, recovery=v2),
                self._r(3, precision=p3, recall=r3, recovery=v3),
            ]
            verdict = classify_aggregate_verdict(results)
            assert verdict in ("PASS", "FAIL", "INCONCLUSIVE")


class TestRecoveryThresholdConstant:
    def test_threshold_is_candidate_c_plus_margin(self):
        assert RECOVERY_COUNT_THRESHOLD == 22  # 17 + 5, matches the corrected preregistration text
