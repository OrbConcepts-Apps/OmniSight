"""EXP-0006 preregistration -- the corrected specification for the
experiment that MAY become EXP-0006, built per the Phase I admission-
boundary and eligibility audits (reports/phase_i/CANDIDATE_0003_EXP0006_ELIGIBILITY.md).

Deliberately NOT `research/_exp0005_preregister.py`'s pattern -- that
module writes directly into `OmniLabDB` because EXP-0005 was already a real
DB row at the QUEUED stage. This module does the opposite on purpose: it
builds a fully-formed `ExperimentSpec`, runs it through the same
deterministic `research.experiment_validator.validate()` every real
proposal goes through, and -- if validation allows it -- freezes its
content hash, but NEVER calls anything in `research.db`/`OmniLabDB`. The
resulting artifact is saved under `research/preregistrations/`, a
namespace distinct from both `research/experiment_specs/` (DB-registered,
backfilled specs) and `research/candidates/` (Phase I's autonomous-loop
artifacts) -- so its mere existence on disk can never be mistaken for
either a registered experiment or a Phase I candidate outcome.

CANDIDATE-0003 (research/candidates/CANDIDATE-0003/) is read-only input
here -- this module never writes to it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from research.config import REPO_ROOT
from research.experiment_spec import ExperimentProposal, ExperimentSpec

PREREGISTRATIONS_DIR = REPO_ROOT / "research" / "preregistrations"

# Pre-registered seed plan -- fixed here, before any training happens.
EXP_0006_SEEDS = (42, 43, 44)

# The fixed population size established by EXP-0005: 92 TRUE_DETECTOR_MISS
# Person cases in the frozen baseline evaluation. Candidate C (YOLO11m/COCO,
# MEM-0015) recovered 17 of them. This module treats both numbers as fixed
# historical facts, never re-derives or re-measures them.
TRUE_DETECTOR_MISS_POPULATION = 92
CANDIDATE_C_RECOVERY_COUNT = 17
RECOVERY_MARGIN = 5  # pre-registered absolute increment required beyond candidate C
RECOVERY_COUNT_THRESHOLD = CANDIDATE_C_RECOVERY_COUNT + RECOVERY_MARGIN  # 22


# ---------------------------------------------------------------------------
# Exhaustive, deterministic per-seed and aggregate verdict classification
# (Section 9's outcome-table requirement). No uncovered combination.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedCriteriaResult:
    """One seed's three PRE-REGISTERED pass/fail criteria -- computed
    elsewhere (a future real evaluation), never fabricated here."""

    seed: int
    precision_pass: bool  # hazard.precision >= 0.757
    recall_pass: bool  # person.recall delta >= 0.03
    recovery_pass: bool  # recovery_count >= RECOVERY_COUNT_THRESHOLD


def classify_seed_verdict(result: SeedCriteriaResult) -> str:
    """Exactly one of PASS/FAIL/INCONCLUSIVE for every one of the 8 possible
    (precision_pass, recall_pass, recovery_pass) combinations -- fixes the
    exact gap CANDIDATE-0003's success criteria left uncovered (recall pass
    + precision pass + recovery fail had no assigned verdict there).

    The hazard-precision guardrail is an absolute safety veto, consistent
    with every prior EXP in this lab: a precision failure is FAIL
    regardless of recall/recovery. Otherwise: both recall and recovery
    passing is PASS; both failing is FAIL; exactly one passing (precision
    intact) is INCONCLUSIVE -- partial signal, safety intact, insufficient
    to fully support the hypothesis."""
    if not result.precision_pass:
        return "FAIL"
    if result.recall_pass and result.recovery_pass:
        return "PASS"
    if not result.recall_pass and not result.recovery_pass:
        return "FAIL"
    return "INCONCLUSIVE"


def classify_aggregate_verdict(results: list) -> str:
    """Aggregate verdict across the full pre-registered seed plan. Requires
    EVERY seed in the plan to be present (cherry-pick-proofing is
    research.execution_job.seeds' job, upstream of this call) -- this
    function only classifies, it does not check completeness itself.

    Rule (Section 2's 'no one favorable seed determines PASS' requirement):
      - any seed FAILED specifically on the precision guardrail -> overall FAIL
        (a single safety-guardrail violation vetoes the whole aggregate,
        never overridden by a majority of favorable seeds elsewhere)
      - >= majority (more than half) of seeds verdict == PASS -> overall PASS
      - zero seeds PASS -> overall FAIL
      - otherwise (some but not a majority PASS, no safety veto) -> INCONCLUSIVE
    Exhaustive over any non-empty seed-result list of any size (not just 3)."""
    if not results:
        raise ValueError("classify_aggregate_verdict: results must be non-empty")

    verdicts = [classify_seed_verdict(r) for r in results]
    safety_veto = any(not r.precision_pass for r in results)
    if safety_veto:
        return "FAIL"

    n = len(verdicts)
    n_pass = sum(1 for v in verdicts if v == "PASS")
    if n_pass > n / 2:
        return "PASS"
    if n_pass == 0:
        return "FAIL"
    return "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# The proposal itself
# ---------------------------------------------------------------------------


def _training_config() -> dict:
    """Structured, hashable training configuration -- deliberately embedded
    as a nested dict inside `controlled_variables` (an existing
    ExperimentProposal field) rather than a new dataclass field, per the
    Phase-I schema-evolution/hash-tolerance lesson: adding a new field to a
    frozen-hashed dataclass is a real, previously-encountered hazard, and
    is unnecessary here since `controlled_variables` already accepts an
    arbitrary dict and participates in the whole-proposal content hash."""
    return {
        "starting_checkpoint": "benchmark/models/yolov8m-oiv7.pt",
        "epochs": 50,
        "epochs_status": "ESTIMATE_UNCALIBRATED -- see compute_resource_estimate",
        "stopping_rule": (
            "fixed-epoch (no early stopping) -- PREREQUISITE: replace with an explicit "
            "early-stopping-on-held-out-validation-split rule once a calibration run exists; "
            "a fixed-epoch budget without a stopping rule risks over/under-fitting relative "
            "to the (currently unknown) actual convergence behavior of this dataset size."
        ),
        "optimizer": "SGD (ultralytics YOLOv8 default)",
        "learning_rate": 0.01,
        "lr_scheduler": "PREREQUISITE -- exact schedule (ultralytics default assumed, unverified) to be confirmed from a calibration run",
        "batch_size": 16,
        "imgsz": 640,
        "augmentation_policy": "PREREQUISITE -- ultralytics YOLOv8 default augmentation assumed; must be explicitly frozen (not merely assumed) before any real training run",
        "seeds": list(EXP_0006_SEEDS),
        "data_manifest_hash": "PREREQUISITE -- dataset does not yet exist, no hash can be computed",
        "framework": "ultralytics",
        "framework_version": "PREREQUISITE -- exact pinned version not yet recorded",
        "device": "CUDA (RTX 3070 Ti, Windows) -- offline screening only, never iPhone/ANE",
        "checkpoint_selection_rule": (
            "best held-out-validation person.recall among epochs where hazard.precision >= 0.757; "
            "held-out validation split is disjoint from BOTH the frozen 380-image eval manifest AND "
            "the training data (see isolation_requirements)"
        ),
        "evaluation_thresholds": {"conf": 0.4, "iou": 0.7},
    }


def build_exp0006_proposal() -> ExperimentProposal:
    """Builds the corrected EXP-0006 preregistration proposal. Every
    unresolved fact is stated as an explicit PREREQUISITE sentence -- never
    fabricated, never a bare placeholder (which
    research.experiment_validator._is_placeholder would correctly reject)."""

    checkpoint_provenance = (
        "Starting checkpoint: benchmark/models/yolov8m-oiv7.pt. KNOWN: architecture=YOLOv8m "
        "(confirmed via ultralytics model introspection, same convention as EXP-0005); "
        "class_set=Open Images V7, 601 classes (confirmed via live model.names, same as production). "
        "UNKNOWN/PREREQUISITE (blocking, must be verified before any real training run -- never "
        "inferred from filename): exact upstream source (believed to be an Ultralytics official "
        "OIV7 release asset per BENCHMARK_PLAN.md's description, unverified against the actual "
        "release manifest); SHA-256 of the checkpoint file; the checkpoint's own weight-file "
        "license (distinct from the image/annotation licenses documented in docs/DATASETS.md, "
        "which cover the EVALUATION data, not this model's training data); the checkpoint's own "
        "original training dataset and hyperparameters; whether any fine-tuning beyond stock OIV7 "
        "pretraining has already been silently applied to this specific .pt file; exact "
        "ultralytics/framework version used to produce it."
    )

    dataset_manifest_contract = (
        "DATASET DOES NOT YET EXIST. This section is a MANIFEST CONTRACT (schema + policy), not a "
        "dataset -- no image is collected, staged, or referenced as if it existed. "
        "Capture domain: chest-height camera angle, indoor and outdoor pedestrian-accessibility "
        "scenarios, explicitly including low-light and motion-blur conditions. Inclusion "
        "criterion: a frame is included only if it contains at least one person with an "
        "annotatable bounding box; frames with zero persons are excluded (this is a Person-"
        "detection-focused dataset, not a general re-training set). Class definition: reuses the "
        "existing Open Images V7 'Person' class definition unchanged (no new subtypes/age/pose "
        "categories) for direct comparability with the baseline and EXP-0001-0005. Annotation "
        "format: matches data/manifests/eval_manifest.jsonl's existing schema exactly "
        "(sample_id, source, filename, split, labels[{class_name, bbox [x,y,w,h] normalized, "
        "is_occluded, is_truncated, is_group_of}], scene_category, lighting_category, difficulty, "
        "license{...}) plus two NEW required fields per record: consent_status "
        "(one of STAGED_CONSENTED | INCIDENTAL_CONSENTED | PROHIBITED -- INCIDENTAL_NONCONSENTED "
        "is never a valid value, see privacy section) and capture_session_id (used for the "
        "session-disjoint train/val split below). Annotation QA: dual independent annotation with "
        "adjudication on any disagreement; inter-annotator bounding-box IoU agreement target "
        ">= 0.7 on >= 95% of boxes before a batch is accepted; annotation-guideline version string "
        "recorded per batch. Image/video source provenance: staged, consented capture sessions "
        "only (see privacy section) -- no scraped or incidentally-sourced imagery. Train/validation "
        "split: split by capture_session_id, never by individual frame -- no two frames from the "
        "same capture session may appear on both sides of the split, which also serves as the "
        "primary near-duplicate-leakage control (consecutive video frames from one session are "
        "near-duplicates by construction). Frozen baseline-evaluation isolation: every candidate "
        "image's SHA-256 is checked against data/manifests/eval_manifest.jsonl's existing 380 "
        "images for exact-duplicate exclusion (existing convention); ADDITIONALLY, a perceptual "
        "hash (e.g. pHash) comparison against the same 380 images is required to catch near-"
        "duplicate leakage that a SHA-256-only check would miss (a near-identical re-capture of "
        "the same eval scene, not merely the identical file). Versioning: dataset_version follows "
        "'omnisight_v{capture_end_date}_annotated', tied 1:1 to a recorded annotation-guideline "
        "version. Manifest hash: SHA-256 of the finished manifest JSONL, to be computed and frozen "
        "once the dataset exists -- PREREQUISITE, not yet computable. Dataset SIZE: not fabricated "
        "here; a planning estimate only (see compute_resource_estimate), not a committed target, "
        "since no power analysis for this specific effect size has been run."
    )

    privacy_note = (
        "CORRECTED from CANDIDATE-0003 (which used data_privacy_classification=NONE): the "
        "described capture domain (real people, in real indoor/outdoor accessibility scenarios, "
        "at chest-height camera angle) is PII-bearing by construction -- this is set to "
        "PRIVATE_USER_DATA, not NONE. Explicit distinction of source categories for this dataset: "
        "public/non-sensitive source (NOT used -- no scraped/public imagery); staged/synthetic "
        "source (not used for this design, though a future revision could substitute this to "
        "avoid the private-data approval entirely); consented collection (the REQUIRED capture "
        "method for every included frame -- consent_status=STAGED_CONSENTED); incidental "
        "bystanders (permitted ONLY as consent_status=INCIDENTAL_CONSENTED, meaning a bystander "
        "who appears incidentally but has separately given consent -- e.g. a public demo/pilot "
        "with posted signage and an opt-out process); private imagery of a non-consenting subject "
        "(PROHIBITED -- any frame containing a non-consenting identifiable bystander must be "
        "excluded or the person masked/blurred before annotation, never included as-is); "
        "prohibited/unsupported collection (any covert, non-consented, or ToS-violating capture "
        "method -- explicitly out of scope, never authorized by this proposal). "
        "private_user_data_use_approved is NOT granted by this proposal and remains False -- this "
        "is flagged as a REQUIRED human approval, not silently assumed."
    )

    causal_control = (
        "CONTROL arm: the SAME starting checkpoint (benchmark/models/yolov8m-oiv7.pt, byte-"
        "identical) fine-tuned for the IDENTICAL epoch budget, optimizer, learning-rate schedule, "
        "batch size, image size, and augmentation policy as the INTERVENTION arm (see "
        "controlled_variables.training_config), using an equal-sized RESAMPLED SUBSET of the "
        "EXISTING Open Images V7 training distribution -- i.e. additional training happens in "
        "both arms, the ONLY difference is whether that additional training data is OmniSight-"
        "domain or resampled-OIV7. This isolates 'additional training of this duration occurred "
        "at all' from 'training specifically on OmniSight-domain data helped'. INTERVENTION arm: "
        "identical procedure, but the additional training data is the OmniSight-specific dataset "
        "(see dataset manifest contract) instead of resampled OIV7. REFERENCE (kept for continuity "
        "with EXP-0001-0005's baseline_metrics convention, explicitly NOT treated as the causal "
        "control for this comparison): the shipped, never-further-trained yolov8m-oiv7.pt "
        "(baseline_run_id=RUN-20260904-002) -- its own pretraining history/provenance is unverified "
        "(see checkpoint provenance), so a difference between it and the intervention arm would "
        "conflate 'more training happened' with 'OmniSight data specifically helped'; only the "
        "CONTROL-vs-INTERVENTION comparison supports a causal claim about training-data "
        "composition specifically."
    )

    return ExperimentProposal(
        schema_version="1.0",
        experiment_id="EXP-0006",
        title="OmniSight-Domain Fine-Tuning vs. Matched Resampled-OIV7 Control for Person TRUE_DETECTOR_MISS Recovery",
        family="training_data",
        hypothesis=(
            "Fine-tuning YOLOv8m on an OmniSight-domain dataset (chest-height, indoor/outdoor, "
            "low-light, motion-blurred accessibility-scenario frames) recovers more of the 92 "
            "fixed baseline TRUE_DETECTOR_MISS Person cases than a MATCHED control arm trained "
            "identically except for using resampled existing Open Images V7 data instead -- "
            "specifically, at least 22 of the 92 cases (5 more than candidate C's 17, EXP-0005/"
            "MEM-0015) -- while keeping hazard.precision >= 0.757, across a majority of 3 "
            "pre-registered training seeds."
        ),
        motivation=(
            "Corrects CANDIDATE-0003's hypothesis/metric wording mismatch identified in "
            "reports/phase_i/CANDIDATE_0003_EXP0006_ELIGIBILITY.md section 1: the prior wording "
            "('25% more than baseline') was vacuous since the baseline recovers 0/92 by "
            "definition (TRUE_DETECTOR_MISS is DEFINED as zero baseline detections); the actual "
            "binding comparison was always against candidate C's 17/92, restated here explicitly "
            "as an absolute-count threshold (interpretation C from that audit's three offered "
            "models), never as a percentage-of-a-percentage."
        ),
        research_question=(
            "Does OmniSight-domain training data, specifically (isolated from the mere fact of "
            "additional training), recover more TRUE_DETECTOR_MISS Person cases than a matched "
            "control receiving equivalent additional training on existing Open Images V7 data, "
            "without violating the hazard-precision guardrail, and is this effect consistent "
            "across multiple training seeds?"
        ),
        evidence_references=("MEM-0003", "MEM-0015", "MEM-0017", "MEM-0025"),
        prior_experiment_ids=("EXP-0004", "EXP-0005"),
        baseline_run_id="RUN-20260904-002",
        baseline_metrics={
            "hazard_precision": 0.8070175438596491,
            "hazard_recall": 0.4804177545691906,
            "hazard_f1": 0.602291325695581,
            "overall_recall": 0.2467453213995118,
            "overall_precision": 0.7342615012106537,
            "num_images_evaluated": 380,
            "run_id": "RUN-20260904-002",
        },
        independent_variables=("training_dataset_composition",),
        dependent_variables=("person.recall", "hazard.precision", "true_detector_miss_recovery_count"),
        controlled_variables={
            "confidence_threshold": 0.4,
            "iou_threshold": 0.7,
            "imgsz": 640,
            "model_architecture": "YOLOv8m",
            "evaluation_dataset": "data/manifests/eval_manifest.jsonl (frozen, 380 images)",
            "training_config": _training_config(),
        },
        procedure=(
            "1. [PREREQUISITE, not done here] Collect+annotate the OmniSight-domain dataset per "
            "the manifest contract below. 2. [PREREQUISITE] Verify yolov8m-oiv7.pt provenance per "
            "the checkpoint-provenance section. 3. Build the CONTROL arm's resampled-OIV7 training "
            "set (equal size to the OmniSight-domain set, drawn from existing OIV7 training data, "
            "excluding any image in the frozen 380-image eval manifest). 4. For each of the 3 "
            "pre-registered seeds (42, 43, 44): train CONTROL and INTERVENTION arms independently, "
            "identical procedure/hyperparameters (see training_config), differing only in training "
            "data. 5. Evaluate every trained checkpoint against the frozen eval manifest at "
            "conf=0.4/iou=0.7, computing person.recall, hazard.precision, and "
            "true_detector_miss_recovery_count (per-seed criteria). 6. Classify each seed via "
            "classify_seed_verdict() and the aggregate via classify_aggregate_verdict() "
            "(research/preregistration.py) -- both deterministic, pre-registered, no post-hoc "
            "discretion. 7. Report every seed's result (COMPLETED or FAILED), never a "
            "favorable-subset selection."
        ),
        dataset_version=(
            "PREREQUISITE -- dataset does not yet exist; will be assigned "
            "'omnisight_v{capture_end_date}_annotated' once collection+annotation completes per "
            "the manifest contract in isolation_requirements/model_config_ref context below."
        ),
        model_config_ref=(
            "Starting checkpoint benchmark/models/yolov8m-oiv7.pt (SHA-256/exact source/license "
            "PREREQUISITE, see checkpoint-provenance notes in isolation_requirements). Final "
            "trained config refs: 'omnisight_control_v1_seed{42,43,44}' and "
            "'omnisight_intervention_v1_seed{42,43,44}' (6 total trained checkpoints)."
        ),
        implementation_scope=(
            "Experiment conducted entirely within research/ and a new experiments/ output "
            "directory; no change to ios/, benchmark/config.py, or the frozen baseline. Requires "
            "a real Phase J Runner implementation to execute (does not exist yet, per "
            "reports/phase_j/PHASE_J_SAFETY_AUDIT.md's residual-risk section) -- this proposal is "
            "a specification only, not an executable plan against current infrastructure."
        ),
        expected_artifacts=(
            "research/experiments/exp_0006/control_seed42/weights/best.pt",
            "research/experiments/exp_0006/control_seed43/weights/best.pt",
            "research/experiments/exp_0006/control_seed44/weights/best.pt",
            "research/experiments/exp_0006/intervention_seed42/weights/best.pt",
            "research/experiments/exp_0006/intervention_seed43/weights/best.pt",
            "research/experiments/exp_0006/intervention_seed44/weights/best.pt",
            "research/experiments/exp_0006/seed_results.json",
            "research/experiments/exp_0006/aggregate_verdict.json",
        ),
        reproducibility_requirements=(
            "Every training_config field (see controlled_variables.training_config) is a "
            "structured, hashable dict participating in this proposal's own content hash -- not "
            "prose-only. All 3 seeds' full results (not a favorable subset) must be reported per "
            "research.execution_job.seeds' cherry-pick-proofing contract. Fixed random seed per "
            "run; CONTROL and INTERVENTION arms for the same seed use the identical seed value "
            "(seed varies the training stochasticity, not the arm identity)."
        ),
        control_condition=causal_control,
        baseline_comparison=(
            "PRIMARY causal comparison: INTERVENTION arm vs. matched CONTROL arm (see "
            "control_condition). SECONDARY, non-causal reference: shipped baseline RUN-20260904-002 "
            "(person.recall=0.2467, hazard.precision=0.8070), retained for continuity with "
            "EXP-0001-0005's reporting convention only."
        ),
        isolation_requirements=(
            "Train/eval isolation: SHA-256 exact-duplicate exclusion (existing convention) PLUS "
            "perceptual-hash near-duplicate exclusion against the frozen 380-image eval manifest; "
            "session-disjoint train/validation split (see dataset manifest contract) as the "
            "primary near-duplicate-leakage control within the training data itself. Checkpoint "
            "provenance (see checkpoint_provenance below) is a BLOCKING PREREQUISITE, verified "
            "before any real training run, never inferred from filename. "
            + checkpoint_provenance
            + " DATASET MANIFEST CONTRACT (no data collected here): "
            + dataset_manifest_contract
            + " PRIVACY CLASSIFICATION: "
            + privacy_note
        ),
        success_criteria={
            "primary_metric": "person.recall",
            "min_meaningful_delta": 0.03,
            "guardrails": ["hazard.precision >= 0.757"],
            "recovery_metric_definition": {
                "name": "true_detector_miss_recovery_count",
                "numerator": (
                    "count of the 92 fixed baseline TRUE_DETECTOR_MISS Person cases (established "
                    "by EXP-0005) detected by the INTERVENTION model at conf=0.4"
                ),
                "denominator": TRUE_DETECTOR_MISS_POPULATION,
                "reference_condition": (
                    f"candidate C (YOLO11m/COCO, EXP-0005/MEM-0015), which recovered "
                    f"{CANDIDATE_C_RECOVERY_COUNT} of the {TRUE_DETECTOR_MISS_POPULATION} cases"
                ),
                "direction": "must exceed the reference by a pre-registered absolute margin, never a relative-percentage-of-a-percentage",
                "threshold": f"recovery_count >= {RECOVERY_COUNT_THRESHOLD} (reference {CANDIDATE_C_RECOVERY_COUNT} + margin {RECOVERY_MARGIN})",
                "interpretation_chosen": (
                    "C -- incremental absolute count beyond the prior best candidate's recovery "
                    "(reports/phase_i/CANDIDATE_0003_EXP0006_ELIGIBILITY.md section 1's three "
                    "offered models); NOT interpretation A (bare fraction of 92) alone and NOT "
                    "interpretation B (relative-improvement percentage), because the baseline's "
                    "own recovery rate is tautologically 0/92 and a 'percent more than baseline' "
                    "framing is therefore vacuous."
                ),
            },
            "seed_plan": {
                "seeds": list(EXP_0006_SEEDS),
                "per_seed_verdict_function": "research.preregistration.classify_seed_verdict",
                "aggregate_verdict_function": "research.preregistration.classify_aggregate_verdict",
                "no_cherry_picking": "all 3 seeds' results reported regardless of outcome; aggregation requires a MAJORITY (>=2/3), never a single favorable seed",
            },
        },
        supports_hypothesis_if=(
            "classify_aggregate_verdict() == 'PASS': >=2 of 3 seeds independently satisfy ALL "
            "THREE per-seed criteria (hazard.precision>=0.757 AND person.recall delta>=0.03 AND "
            f"recovery_count>={RECOVERY_COUNT_THRESHOLD}), with zero seeds vetoed on the precision "
            "guardrail."
        ),
        rejects_hypothesis_if=(
            "classify_aggregate_verdict() == 'FAIL': either zero of 3 seeds achieve a per-seed "
            "PASS, or ANY seed fails specifically on the hazard-precision guardrail (a single "
            "safety-guardrail violation vetoes the whole aggregate, regardless of other seeds)."
        ),
        inconclusive_if=(
            "classify_aggregate_verdict() == 'INCONCLUSIVE': exactly 1 of 3 seeds achieves a "
            "per-seed PASS with no precision-guardrail veto -- mixed, non-majority signal."
        ),
        production_impact=False,
        production_impact_description=(
            "Experimental evaluation only; any production model swap requires a separate, later, "
            "explicit human decision and is never implied by this proposal or by registration "
            "itself."
        ),
        data_privacy_classification="PRIVATE_USER_DATA",
        external_api_required=False,
        mac_iphone_required=True,
        coreml_replacement_required=False,
        signing_distribution_change_required=False,
        compute_resource_estimate={
            "status": "ESTIMATE_UNCALIBRATED -- no calibration run has been performed (explicitly out of scope for this preregistration task)",
            "estimated_gpu_hours_per_seed_per_arm": 24,
            "estimated_gpu_hours_total": 24 * 2 * len(EXP_0006_SEEDS),  # 2 arms x 3 seeds
            "planning_method": (
                "naive multiplication of CANDIDATE-0003's single unvalidated 24-GPU-hour estimate "
                "by (2 arms x 3 seeds) -- a PLANNING CEILING, not a validated figure; the actual "
                "resource authorization will be set only after a real Phase J Runner and a real "
                "calibration run exist (no calibration run is performed in this task)"
            ),
            "gpu": "RTX 3070 Ti",
            "disk_requirement_status": "PREREQUISITE -- 6 checkpoints x unknown per-checkpoint size (depends on final architecture export); not estimated here to avoid fabricating a number with no basis",
            "checkpoint_storage_status": "PREREQUISITE -- storage/versioning location for 6 intermediate checkpoints not yet decided",
        },
        allowed_path_scope=("benchmark/", "data/", "research/", "experiments/"),
        acknowledges_rejected_hypothesis_ids=(),
        materially_new_rationale=(
            "Materially distinct from every prior EXP-0001-0005 (threshold/resolution/class-remap/"
            "preprocessing/checkpoint-size, none of which changed training data) and from "
            "CANDIDATE-0001 (temporal_pipeline, a post-hoc confirmation filter -- structurally "
            "cannot add a detection to a zero-detection frame, per "
            "reports/phase_i/CANDIDATE_0001_POSTMORTEM.md) and CANDIDATE-0002 (model_variant + "
            "gamma preprocessing, rejected for missing rejected-hypothesis acknowledgment and a "
            "mac_iphone_required schema gap, and independently assessed PLAUSIBLE_BUT_UNDERJUSTIFIED "
            "for lacking a mechanistic link between gamma correction and YOLO11m's own precision "
            "collapse). This design (a) is the first to isolate training-DATA composition via a "
            "matched control arm rather than comparing against the shipped baseline directly, "
            "closing the causal-confound gap CANDIDATE-0003 left open, and (b) is the first to use "
            "a multi-seed, cherry-pick-proof aggregate verdict rather than a single training run. "
            "NOTE (validator limitation, disclosed honestly): research.experiment_validator's "
            "rejected-hypothesis check requires an EXACT family match against existing EXP rows; "
            "since no EXP-0001-0005 row has family='training_data', that mechanical check cannot "
            "and will not fire for this proposal regardless of content -- this acknowledgment "
            "section is therefore the ONLY place any conceptual-overlap accounting exists, and is "
            "provided voluntarily, not because the deterministic gate required it."
        ),
    )


def build_exp0006_spec() -> ExperimentSpec:
    return ExperimentSpec(proposal=build_exp0006_proposal())


def save_preregistration(spec: ExperimentSpec, path: Optional[Path] = None) -> Path:
    path = path or (PREREGISTRATIONS_DIR / "EXP-0006-PROPOSED.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.to_json(), encoding="utf-8")
    return path


def load_preregistration(path: Optional[Path] = None) -> ExperimentSpec:
    path = path or (PREREGISTRATIONS_DIR / "EXP-0006-PROPOSED.json")
    return ExperimentSpec.from_json(path.read_text(encoding="utf-8"))
