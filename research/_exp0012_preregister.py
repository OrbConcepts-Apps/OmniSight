"""One-shot registration of EXP-0012: test-time augmentation (TTA)
sensitivity test.

Run via:

    uv run python -m research._exp0012_preregister

Idempotent: no-ops if EXP-0012 already exists.

Preregistered BEFORE any result was computed or inspected.

Distinct mechanism from both now-closed lines of inquiry (EXP-0007-0010
Person confidence threshold, EXP-0011 NMS IoU): TTA (ultralytics'
`augment=True`) runs several transformed views of the same image through
the model and merges candidates across views -- unlike re-thresholding or
re-selecting among a SINGLE pass's candidates, a genuinely different view
CAN produce a detection the original pass did not, giving TTA a real
channel to potentially recover a TRUE_DETECTOR_MISS case.

Single ON/OFF test (augment=False control vs augment=True candidate), not
a hyperparameter grid. Confidence (0.4) and NMS IoU (0.7) fixed at
production values throughout. No training, no private data, no participant
data, no device deployment, no production changes, no new human approvals,
no live LLM calls. Runs new (non-training) inference over the existing
frozen public 380-image eval set.
"""

from __future__ import annotations

from research.config import EXPERIMENTS_DIR
from research.db import Experiment, OmniLabDB
from research.experiment_lifecycle import move_to_status
from research.experiment_schema import write_queued_artifacts

EXPERIMENT_ID = "EXP-0012"


def _experiment() -> Experiment:
    return Experiment(
        experiment_id=EXPERIMENT_ID,
        hypothesis=(
            "Test-time augmentation (ultralytics augment=True: multi-view flip/scale inference "
            "merged via NMS), confidence and NMS IoU fixed at production values, recovers "
            "meaningful Person recall (>= +0.03 vs the augment=False control) while keeping "
            "hazard-aggregate precision at or above the standard guardrail (>= 0.757) AND "
            "keeping p95 latency within the lab's existing +50% regression guardrail. Mechanism: "
            "unlike confidence-threshold or NMS-IoU changes (which can only re-select among a "
            "single pass's already-proposed candidates), TTA runs multiple transformed views "
            "through the model and can produce a genuinely NEW candidate detection in a view "
            "where the original pass proposed nothing -- the one channel tested so far with a "
            "real mechanism to potentially recover a TRUE_DETECTOR_MISS case."
        ),
        motivation=(
            "Both prior post-hoc, single-pass levers on the shipped checkpoint (EXP-0007-0010 "
            "Person confidence threshold, EXP-0011 NMS IoU) are closed as negative results, and "
            "both were mechanistically incapable of ever recovering a TRUE_DETECTOR_MISS case "
            "(92/239 baseline Person false negatives, the dominant failure category) -- neither "
            "can add a detection where the single forward pass proposed none. TTA is "
            "mechanistically different: it is real inference (still no training, no new data), "
            "but genuinely multi-pass, giving it an actual channel this lab has not yet tested."
        ),
        rationale=(
            "Uses only the existing frozen 380-image public eval set and the existing shipped "
            "checkpoint. Not a hyperparameter sweep -- a single ON/OFF test, so there is no grid "
            "point to cherry-pick from. Real cost is measured directly and checked against the "
            "lab's EXISTING latency guardrail (not a new criterion) -- a recall improvement "
            "bought with unacceptable real-time latency cost is a genuine, reportable constraint "
            "for an assistive-vision system, not something to omit. Orthogonal to and unaffected "
            "by the blocked EXP-0006 pilot."
        ),
        independent_variable=(
            "tta_augment (benchmark/model.py::predict_at()'s new `augment` parameter, passed to "
            "ultralytics model.predict()) -- binary on/off, confidence (0.4) and NMS IoU (0.7) "
            "fixed at production values throughout, not jointly optimized with either closed "
            "branch."
        ),
        controls={
            "model": "yolov8m-oiv7.pt (same weights as the canonical baseline)",
            "manifest": "data/manifests/eval_manifest.jsonl (unchanged)",
            "imgsz": 640,
            "confidence_threshold": 0.4,
            "nms_iou": 0.7,
        },
        evaluation_method=(
            "benchmark/diagnostics/tta_sweep.py runs real (non-training) inference over all 380 "
            "eval images under augment=False (control, must exactly reproduce the official "
            "baseline) and augment=True (candidate), computing hazard/Person metrics via the "
            "same benchmark.metrics matching code as every other experiment in this lab, plus "
            "p50/p95 latency and the same TRUE_DETECTOR_MISS recovery check as EXP-0011 against "
            "the 92 known baseline cases."
        ),
        success_criteria={
            "primary_metric": "person.recall",
            "min_meaningful_delta": 0.03,
            "precision_floor": 0.757,
            "guardrail_metrics": ["hazard.precision", "latency.p95_ms"],
            "max_latency_regression_pct": 50.0,
            "sample_size_requirements": {"person": 100},
            "note": (
                "All three thresholds (0.757, 0.03, 50%) are UNCHANGED from this lab's existing "
                "policy (research/evaluation_policy.py::default_hazard_policy) -- no new "
                "significance criterion is introduced. If the point estimate passes, the same "
                "image-level bootstrap robustness check that exposed EXP-0008's fragility "
                "(EXP-0009's methodology) is required before treating it as credible."
            ),
        },
        risks=(
            "Runs real (non-training) GPU inference, 2-3x the normal per-image cost expected -- "
            "no training, no private data, no production/config changes (uses predict_at(), "
            "never predict(); benchmark/config.py is never modified). The only risk is "
            "scientific: an honest negative or latency-constrained result must be reported as "
            "such, not reframed as a win."
        ),
        expected_outcome=(
            "Either: (a) no meaningful recall improvement -- closes this line of inquiry as a "
            "third consecutive negative result on the shipped checkpoint's inference-time levers; "
            "(b) a meaningful recall improvement but at an unacceptable latency cost -- a real, "
            "reportable constraint, not a usable production candidate; or (c) a genuine, "
            "guardrail-clearing, latency-acceptable improvement -- the strongest candidate this "
            "lab has found on the shipped checkpoint, still requiring the same robustness check "
            "before being treated as credible."
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
