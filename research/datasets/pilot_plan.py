"""EXP-0006 staged/consented pilot -- planning utilities. No real
collection, annotation, or evaluation happens here. See
reports/phase_i/EXP0006_PILOT_PLAN.md for the full protocol this module
supports.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

PILOT_ID = "OMNISIGHT-PILOT-001"
SCHEMA_VERSION = "1.0"

# The pilot is a DISTINCT dataset identity from any eventual final training
# dataset -- promotion of any pilot capture into a training split requires
# a separate, explicit, later versioned decision (never automatic).
PILOT_IS_TRAINING_DATA = False


@dataclass(frozen=True)
class PilotSizePlan:
    """A PROPOSED bounded pilot size -- planning-only, never a claim of
    statistical power for a final research conclusion."""

    num_participants: int
    num_sessions: int
    num_sequences_per_session: int
    approx_sequence_duration_sec: int
    frame_sampling_rule: str


def pilot_plan_hash(plan: PilotSizePlan) -> str:
    """Canonical SHA-256 of a pilot size plan's fields. Any change to
    caps/sampling-rule/duration produces a different hash -- a collection
    authorization scoped to one hash is invalidated the instant the plan
    changes (Phase authorization section 10)."""
    payload = json.dumps(asdict(plan), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# The canonical, frozen OMNISIGHT-PILOT-001 plan -- these are the ONLY
# caps a collection authorization may be scoped against today. Increasing
# any of them requires a NEW plan (and therefore a new hash), which in
# turn requires a fresh, separate human authorization -- never a silent
# scope increase under the same approval.
OMNISIGHT_PILOT_001_PLAN = PilotSizePlan(
    num_participants=6,
    num_sessions=6,
    num_sequences_per_session=4,
    approx_sequence_duration_sec=20,
    frame_sampling_rule=(
        "1 fps baseline sampling, plus dense 5fps sampling within any pre-identified "
        "hard-event window (motion-blur onset, occlusion transition, distance-band "
        "change), capped at 40 sampled frames per sequence"
    ),
)
OMNISIGHT_PILOT_001_PLAN_HASH = pilot_plan_hash(OMNISIGHT_PILOT_001_PLAN)
OMNISIGHT_PILOT_001_MAX_SEQUENCES = (
    OMNISIGHT_PILOT_001_PLAN.num_sessions * OMNISIGHT_PILOT_001_PLAN.num_sequences_per_session
)  # 6 * 4 = 24


@dataclass(frozen=True)
class BaselineEvalConfig:
    """The FROZEN baseline evaluation procedure a future pilot would run
    once data exists -- never executed by this module."""

    baseline_model_sha256: str
    confidence_threshold: float
    iou_threshold: float
    imgsz: int
    evaluation_code_version: str


def build_empty_pilot_manifest(size_plan: PilotSizePlan) -> dict:
    """Returns a manifest TEMPLATE -- schema/version metadata and the
    planned scenario taxonomy, with ZERO records. Never populated with
    fabricated captures by this function."""
    return {
        "pilot_id": PILOT_ID,
        "pilot_plan_hash": pilot_plan_hash(size_plan),
        "schema_version": SCHEMA_VERSION,
        "is_training_data": PILOT_IS_TRAINING_DATA,
        "planned_size": asdict(size_plan),
        "planned_scenario_taxonomy": [
            "normal_illumination", "low_illumination", "motion_blur",
            "partial_occlusion", "small_distant_person", "clutter",
            "unusual_assistive_viewpoint", "indoor", "outdoor_controlled",
            "no_person_negative",
        ],
        "expected_logical_identifiers": [
            "session_id", "sequence_id", "frame_id", "environment_domain", "device",
        ],
        "records": [],  # zero real captures -- see reports/phase_i/EXP0006_PILOT_PLAN.md
    }


def save_empty_pilot_manifest(size_plan: PilotSizePlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_empty_pilot_manifest(size_plan), indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Pilot readiness verdict -- exhaustive, deterministic
# ---------------------------------------------------------------------------

READY = "READY"
EXTEND_PILOT = "EXTEND_PILOT"
REDESIGN_PROTOCOL = "REDESIGN_PROTOCOL"
VERDICTS = (READY, EXTEND_PILOT, REDESIGN_PROTOCOL)


@dataclass(frozen=True)
class PilotCompletenessReport:
    """Booleans a real pilot run would eventually report -- all False
    until real data exists. This dataclass never holds fabricated True
    values; every field here is meant to be set only from real pilot
    output."""

    dataset_size_estimable: bool = False
    annotation_cost_estimable: bool = False
    usable_yield_estimable: bool = False
    failure_prevalence_estimable: bool = False
    qa_feasibility_validated: bool = False
    storage_privacy_workflow_validated: bool = False
    leakage_controls_validated: bool = False
    protocol_defect_found: bool = False  # e.g. capture equipment/consent-flow failure


def classify_pilot_readiness(report: PilotCompletenessReport) -> str:
    """Exhaustive over every combination of the 8 boolean fields --
    REDESIGN_PROTOCOL always wins (a protocol defect must be fixed before
    anything else matters); READY requires every other estimate to be in
    hand; otherwise EXTEND_PILOT (some real signal, not yet complete)."""
    if report.protocol_defect_found:
        return REDESIGN_PROTOCOL
    all_estimable = (
        report.dataset_size_estimable
        and report.annotation_cost_estimable
        and report.usable_yield_estimable
        and report.failure_prevalence_estimable
        and report.qa_feasibility_validated
        and report.storage_privacy_workflow_validated
        and report.leakage_controls_validated
    )
    return READY if all_estimable else EXTEND_PILOT


# Predefined descriptive metrics a real pilot run would eventually report
# (Phase authorization section 19). No values populated -- names only.
PILOT_OUTPUT_METRIC_NAMES = (
    "total_sessions", "total_independent_sequences", "total_sampled_frames",
    "person_positive_frames", "no_person_frames", "excluded_frames",
    "excluded_sequences", "exclusion_reasons", "privacy_bystander_exclusion_rate",
    "bounding_box_count", "inter_annotator_agreement", "adjudication_rate",
    "exact_duplicate_rate", "near_duplicate_candidate_rate", "baseline_person_recall",
    "baseline_person_precision", "derived_true_detector_miss_count",
    "derived_true_detector_miss_rate", "hard_condition_prevalence",
    "annotation_minutes_per_sampled_frame", "storage_per_minute_or_session",
    "effective_sample_size_estimate", "sequence_correlation_estimate",
)
