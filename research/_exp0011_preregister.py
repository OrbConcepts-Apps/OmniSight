"""One-shot registration of EXP-0011: inference/NMS IoU sensitivity sweep.

Run via:

    uv run python -m research._exp0011_preregister

Idempotent: no-ops if EXP-0011 already exists.

Preregistered BEFORE any grid-point result was computed or inspected --
this script is committed and the DB row created before
benchmark/diagnostics/nms_iou_sweep.py is ever executed.

VERIFIED (not assumed): benchmark/config.py's own docstring and
benchmark/model.py::predict_at() confirm `iou=0.7` is the model's internal
NMS (duplicate-box suppression) threshold passed to ultralytics'
model.predict(), NOT the evaluation-matching IoU (benchmark/config.py's
separate MAP50_IOU=0.5, used by benchmark/metrics.py's greedy_match()).
These two IoUs are kept programmatically and conceptually separate
throughout this experiment.

Distinct from, and does not reopen, the now-closed EXP-0007-0010 Person
confidence-threshold branch (frozen conclusion: no tested confidence
threshold provides both the lab's meaningful recall gain and robust
hazard-precision protection). Confidence threshold is FIXED at the
production value (0.4) throughout this experiment -- NMS IoU is the only
variable, deliberately not jointly optimized with Person threshold.

No training, no private data, no participant data, no device deployment,
no production changes, no new human approvals, no live LLM calls. Runs new
(non-training) inference over the existing frozen public 380-image eval
set -- explicitly authorized for this experiment, unlike EXP-0007-0010
which only re-filtered an existing capture.
"""

from __future__ import annotations

from research.config import EXPERIMENTS_DIR
from research.db import Experiment, OmniLabDB
from research.experiment_lifecycle import move_to_status
from research.experiment_schema import write_queued_artifacts

EXPERIMENT_ID = "EXP-0011"


def _experiment() -> Experiment:
    return Experiment(
        experiment_id=EXPERIMENT_ID,
        hypothesis=(
            "Changing the model's internal NMS IoU threshold (production=0.7) -- confidence "
            "threshold held fixed at the production value (0.4) -- can recover meaningful "
            "Person recall (>= +0.03 vs the iou=0.7 control) while keeping hazard-aggregate "
            "precision at or above the standard guardrail (>= 0.757). Mechanism: NMS chooses "
            "among ALREADY-PROPOSED overlapping candidate boxes of the same class; it cannot "
            "invent a detection where the raw detector proposed nothing. It can only plausibly "
            "help by letting a currently-suppressed second detection near an ADJACENT person "
            "survive (raising IoU, less aggressive suppression) -- it has no plausible channel "
            "to recover a genuine 'detector saw nothing' TRUE_DETECTOR_MISS case, and lowering "
            "IoU (more aggressive suppression) can only remove surviving boxes, never add one."
        ),
        motivation=(
            "The Person confidence-threshold line of inquiry (EXP-0007-0010) is closed: no "
            "tested threshold is both robust and meaningful. NMS IoU is a genuinely distinct, "
            "previously untested independent variable (verified in code, not assumed, to be "
            "the model's own duplicate-suppression threshold, separate from the evaluation-"
            "matching IoU). This experiment tests it in isolation, with an explicit, falsifiable "
            "mechanism analysis stated before any result is computed -- including an honest "
            "statement of what this mechanism CANNOT plausibly fix (TRUE_DETECTOR_MISS misses "
            "with no raw candidate box at all)."
        ),
        rationale=(
            "Uses only the existing frozen 380-image public eval set and the existing shipped "
            "checkpoint. Runs new, non-training inference (explicitly authorized for this "
            "experiment) -- unlike EXP-0007-0010, NMS IoU cannot be swept by re-filtering an "
            "existing capture, since NMS happens inside the model call before any prediction is "
            "returned. No private data, no participant data, no device deployment, no "
            "production change, no new approval. Orthogonal to and unaffected by the blocked "
            "EXP-0006 pilot."
        ),
        independent_variable=(
            "nms_inference_iou (benchmark/model.py::predict_at()'s `iou` parameter, passed to "
            "ultralytics model.predict()) -- grid [0.9,0.8,0.7,0.6,0.5], 0.7=production control, "
            "fixed BEFORE any grid-point result was computed. Confidence threshold fixed at "
            "0.4 (production) throughout -- NOT jointly optimized with the now-closed "
            "EXP-0007-0010 Person-threshold branch."
        ),
        controls={
            "model": "yolov8m-oiv7.pt (same weights as the canonical baseline)",
            "manifest": "data/manifests/eval_manifest.jsonl (unchanged)",
            "imgsz": 640,
            "confidence_threshold": 0.4,
            "evaluation_matching_iou": 0.5,
        },
        evaluation_method=(
            "benchmark/diagnostics/nms_iou_sweep.py runs real (non-training) inference over all "
            "380 eval images at each grid IoU value (conf fixed at 0.4), computing hazard/Person "
            "metrics via the same benchmark.metrics matching code as every other experiment in "
            "this lab. The iou=0.7 control point must exactly reproduce the official baseline "
            "(identity/correctness check). TRUE_DETECTOR_MISS recovery is checked directly "
            "against the 92 known baseline cases (person_confusion_analysis.json) -- did any "
            "grid point's own Person predictions newly match (IoU>=0.5) one of those specific "
            "92 GT boxes. A duplicate-Person-detection proxy (mutual IoU>=0.5 among a single "
            "image's surviving Person boxes) is also reported."
        ),
        success_criteria={
            "primary_metric": "person.recall",
            "min_meaningful_delta": 0.03,
            "precision_floor": 0.757,
            "guardrail_metrics": ["hazard.precision"],
            "sample_size_requirements": {"person": 100},
            "note": (
                "Both thresholds (0.757, 0.03) are UNCHANGED from EXP-0001 onward -- no new "
                "significance criterion is introduced for this experiment. If any grid point "
                "passes on the point estimate, the SAME image-level bootstrap robustness check "
                "that exposed EXP-0008's fragility (EXP-0009's methodology) is required before "
                "treating it as a credible positive finding -- a point-estimate PASS alone is "
                "explicitly NOT sufficient here, matching the precedent this lab just set."
            ),
        },
        risks=(
            "Runs real (non-training) GPU inference over the existing eval set -- no training, "
            "no private data, no production/config changes (uses predict_at(), never predict(); "
            "benchmark/config.py's real IOU_THRESHOLD=0.7 is never modified). The only risk is "
            "scientific: an honest negative result must be reported as such, not reframed."
        ),
        expected_outcome=(
            "Either: (a) no grid point clears both the guardrail and the minimum meaningful "
            "delta -- consistent with the mechanism analysis's prediction that NMS IoU has no "
            "plausible channel to fix the dominant TRUE_DETECTOR_MISS failure mode, closing this "
            "branch too as a genuine negative result; or (b) a grid point passes the point "
            "estimate -- in which case its robustness must be checked (per EXP-0009's precedent) "
            "before being treated as credible, and the TRUE_DETECTOR_MISS recovery count "
            "specifically should be inspected to see whether the adjacent-person-suppression "
            "mechanism is actually responsible, or something else is."
        ),
        parent_experiment_id=None,
        experiment_family="threshold_postprocessing",
        model_version="yolov8m-oiv7.pt",
        dataset_version="data/manifests/eval_manifest.jsonl",
        baseline_run_id="RUN-20260904-002",
        validation_requirement="OFFLINE_SIMULATABLE",
        estimated_cost=0.0,
    )


def main() -> None:
    with OmniLabDB() as db:
        existing_ids = {e.experiment_id for e in db.list_experiments()}
        if EXPERIMENT_ID in existing_ids:
            print(f"{EXPERIMENT_ID} already registered -- no-op.")
            return
        exp = _experiment()
        db.create_experiment(exp)
        write_queued_artifacts(exp, EXPERIMENTS_DIR / "queued" / exp.experiment_id)
        move_to_status(exp.experiment_id, "QUEUED")
        print(f"{EXPERIMENT_ID} registered, status=QUEUED.")


if __name__ == "__main__":
    main()
