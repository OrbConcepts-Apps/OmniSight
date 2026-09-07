"""EXP-0006 pilot planning tests (research/datasets/pilot_plan.py). No real
media, no real collection -- schema/template and decision-logic only."""

from __future__ import annotations

import itertools

from research.datasets.pilot_plan import (
    PILOT_ID,
    VERDICTS,
    BaselineEvalConfig,
    PilotCompletenessReport,
    PilotSizePlan,
    build_empty_pilot_manifest,
    classify_pilot_readiness,
)

_PLAN = PilotSizePlan(
    num_participants=6, num_sessions=6, num_sequences_per_session=4,
    approx_sequence_duration_sec=20, frame_sampling_rule="1fps baseline",
)


class TestPilotIdentity:
    def test_pilot_id_distinct_from_final_dataset_versioning(self):
        assert PILOT_ID == "OMNISIGHT-PILOT-001"
        assert not PILOT_ID.lower().startswith("omnisight_v")  # final dataset naming convention is different


class TestEmptyPilotManifest:
    def test_manifest_has_zero_records(self):
        manifest = build_empty_pilot_manifest(_PLAN)
        assert manifest["records"] == []

    def test_manifest_not_flagged_as_training_data(self):
        manifest = build_empty_pilot_manifest(_PLAN)
        assert manifest["is_training_data"] is False

    def test_manifest_carries_pilot_id_and_schema_version(self):
        manifest = build_empty_pilot_manifest(_PLAN)
        assert manifest["pilot_id"] == PILOT_ID
        assert manifest["schema_version"]

    def test_manifest_scenario_taxonomy_covers_authorized_factors(self):
        manifest = build_empty_pilot_manifest(_PLAN)
        taxonomy = set(manifest["planned_scenario_taxonomy"])
        for required in ("low_illumination", "motion_blur", "partial_occlusion", "no_person_negative"):
            assert required in taxonomy

    def test_save_and_reload(self, tmp_path):
        import json

        from research.datasets.pilot_plan import save_empty_pilot_manifest

        path = tmp_path / "pilot.json"
        save_empty_pilot_manifest(_PLAN, path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["records"] == []
        assert data["pilot_id"] == PILOT_ID


class TestBaselineEvalConfig:
    def test_config_holds_frozen_procedure_fields_only(self):
        config = BaselineEvalConfig(
            baseline_model_sha256="21ffa3718c577ac23e708e4c0544c49a20682efa03914d5f816166b54e8fd3fe",
            confidence_threshold=0.4, iou_threshold=0.7, imgsz=640,
            evaluation_code_version="benchmark.evaluate@unspecified",
        )
        assert config.confidence_threshold == 0.4
        assert len(config.baseline_model_sha256) == 64


class TestPilotReadinessVerdictExhaustiveness:
    def test_all_true_is_ready(self):
        report = PilotCompletenessReport(
            dataset_size_estimable=True, annotation_cost_estimable=True,
            usable_yield_estimable=True, failure_prevalence_estimable=True,
            qa_feasibility_validated=True, storage_privacy_workflow_validated=True,
            leakage_controls_validated=True,
        )
        assert classify_pilot_readiness(report) == "READY"

    def test_all_false_is_extend_pilot(self):
        assert classify_pilot_readiness(PilotCompletenessReport()) == "EXTEND_PILOT"

    def test_protocol_defect_always_wins(self):
        report = PilotCompletenessReport(
            dataset_size_estimable=True, annotation_cost_estimable=True,
            usable_yield_estimable=True, failure_prevalence_estimable=True,
            qa_feasibility_validated=True, storage_privacy_workflow_validated=True,
            leakage_controls_validated=True, protocol_defect_found=True,
        )
        assert classify_pilot_readiness(report) == "REDESIGN_PROTOCOL"

    def test_exhaustive_over_all_256_combinations(self):
        """Every combination of the 8 booleans must resolve to exactly one
        of the 3 defined verdicts -- no uncovered state."""
        for combo in itertools.product([True, False], repeat=8):
            report = PilotCompletenessReport(*combo)
            verdict = classify_pilot_readiness(report)
            assert verdict in VERDICTS

    def test_partial_completeness_without_defect_is_extend_pilot(self):
        report = PilotCompletenessReport(dataset_size_estimable=True)
        assert classify_pilot_readiness(report) == "EXTEND_PILOT"


class TestPilotScopePolicy:
    """Section 6's staged-consented-only / sensitive-context exclusion
    policy, checked against the pilot's own planned taxonomy and the
    shared privacy vocabulary it must use."""

    def test_scenario_taxonomy_contains_no_sensitive_context(self):
        manifest = build_empty_pilot_manifest(_PLAN)
        taxonomy_text = " ".join(manifest["planned_scenario_taxonomy"]).lower()
        for prohibited in ("school", "medical", "bathroom", "changing", "minor", "home_interior"):
            assert prohibited not in taxonomy_text

    def test_pilot_reuses_staged_consented_privacy_vocabulary(self):
        from research.datasets.manifest_schema import PRIVACY_CLASSES

        assert "STAGED_CONSENTED" in PRIVACY_CLASSES

    def test_pilot_manifest_has_no_pii_shaped_fields(self):
        manifest = build_empty_pilot_manifest(_PLAN)
        flat_keys = set(manifest.keys()) | set(manifest.get("planned_size", {}).keys())
        for pii_hint in ("name", "email", "phone", "address", "ssn"):
            assert not any(pii_hint in k.lower() for k in flat_keys)
