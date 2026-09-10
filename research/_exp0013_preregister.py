"""One-shot registration of EXP-0013: image tiling for small/distant Person
recovery.

Run via:

    uv run python -m research._exp0013_preregister

Idempotent: no-ops if EXP-0013 already exists.

Preregistered BEFORE any tiling-specific result was computed or inspected.

## 1. Mechanism validation (done BEFORE writing benchmark/diagnostics/tile_sweep.py)

Verified against the actual pipeline, not assumed:
- ultralytics' model.predict(imgsz=640, ...) applies a LetterBox transform
  (confirmed by reading ultralytics/data/augment.py::LetterBox directly),
  resizing whatever source it is given -- full frame or a crop -- to fit
  within 640x640, preserving aspect ratio.
- All 380 eval images are larger than 640x640 on at least one axis
  (verified via PIL: min 311x446, max 1024x1024, median 1024x768, zero
  images with both dimensions <=640) -- so the full-image pass already
  downscales every object, including small/distant Person instances,
  before the network ever sees it.
- A CROP containing that same instance, letterboxed to the SAME fixed
  640x640 input, presents it at a LARGER effective scale (less information
  lost to downscaling), because tiling keeps the network's own input
  resolution FIXED at 640 -- only what portion of the original image
  occupies that fixed frame changes. This is the mechanistic difference
  from EXP-0002 (which changed imgsz itself to 960/1280 and made Person
  recall WORSE, 0.211->0.165, a likely out-of-distribution scale mismatch
  for a model calibrated at 640). The distinction survives verification --
  this branch proceeds.

## 2-3. Implementation + correctness tests (done BEFORE this preregistration)

benchmark/diagnostics/tiling.py (deterministic tile geometry, coordinate
remapping, cross-tile NMS merge) + tests/test_tiling.py (19 synthetic
geometry fixtures with hand-computed expected values) -- ALL PASS, per this
task's explicit requirement that the scientific benchmark not run until
these pass. Committed in a prior commit, before this preregistration.

No training, no private data, no participant data, no device deployment,
no production changes, no new human approvals, no live LLM calls. Runs new
(non-training) inference over the existing frozen public 380-image eval
set for the 4 crop tiles per image; the full-image pass is REUSED directly
from benchmark/results/baseline/predictions.jsonl (bit-identical to the
official baseline by construction, not re-inferred) rather than re-run.
"""

from __future__ import annotations

from research.config import EXPERIMENTS_DIR
from research.db import Experiment, OmniLabDB
from research.experiment_lifecycle import move_to_status
from research.experiment_schema import write_queued_artifacts

EXPERIMENT_ID = "EXP-0013"

# ## 5. Bounded, preregistered primary configuration (chosen BEFORE any
# tiling-specific result was computed -- NOT a hyperparameter search):
PRIMARY_CONFIG = {
    "tile_layout": "2x2 grid + full image (5 logical passes/image; full-image pass reused from the official baseline capture, not re-inferred)",
    "n_cols": 2,
    "n_rows": 2,
    "overlap_fraction": 0.20,
    "per_tile_inference_imgsz": 640,  # == production IMGSZ, unchanged
    "confidence_threshold": 0.4,      # == production CONF_THRESHOLD, unchanged
    "per_tile_nms_iou": 0.7,          # == production IOU_THRESHOLD, unchanged
    "cross_tile_merge_algorithm": "greedy, confidence-descending, same-class-only NMS (benchmark/diagnostics/tiling.py::merge_detections_nms)",
    "cross_tile_merge_iou": 0.7,      # matches production NMS IoU, for consistency
    "model_sha256": "21ffa3718c577ac23e708e4c0544c49a20682efa03914d5f816166b54e8fd3fe",
    "dataset_manifest_sha256": "617190fd433c26c1bb916e6844c705c64edf73defe49bac1d72eba5623915327",
    "code_commit_at_preregistration": "34563c8c465568000b083b56fa3847ad23e0e112",
}


def _experiment() -> Experiment:
    return Experiment(
        experiment_id=EXPERIMENT_ID,
        hypothesis=(
            "Tiling (cropping each image into 4 overlapping 2x2-grid regions plus the full frame, "
            "running inference on each at the fixed production imgsz=640/conf=0.4/iou=0.7, then "
            "merging via cross-tile NMS) recovers meaningful Person recall (>= +0.03 vs the frozen "
            "single-pass baseline) while keeping hazard-aggregate precision at or above the "
            "standard guardrail (>= 0.757), because it increases the EFFECTIVE detector-input "
            "scale of small/distant Person instances without changing the network's own input "
            "resolution (unlike EXP-0002's failed global-resize approach)."
        ),
        motivation=(
            "68 of the 92 baseline TRUE_DETECTOR_MISS cases are 'small' (<2% image area, "
            "person_confusion_analysis.json). Three prior inference-time levers (EXP-0007-0010 "
            "Person confidence threshold, EXP-0011 NMS IoU, EXP-0012 TTA) are all closed as "
            "negative and none had a plausible mechanism to change effective object scale. "
            "Tiling is mechanistically distinct and directly targets this specific subset -- but "
            "is NOT assumed to prove the remaining problem is a training-data limitation; it is "
            "an unresolved alternative mechanism, tested here before drawing that conclusion."
        ),
        rationale=(
            "Uses only the existing frozen 380-image public eval set and the existing shipped "
            "checkpoint. Not a hyperparameter search -- ONE preregistered primary configuration "
            "(PRIMARY_CONFIG above), chosen before any tiling-specific result was computed. No "
            "training, no private data, no device deployment, no production change, no new "
            "approval. Orthogonal to and unaffected by the blocked EXP-0006 pilot."
        ),
        independent_variable=(
            "inference_tiling (on/off: single full-frame pass vs the 5-pass tiled+merged "
            "pipeline described in PRIMARY_CONFIG). Confidence (0.4) and per-tile/merge NMS IoU "
            "(0.7) fixed at production values throughout -- not jointly optimized with any "
            "closed branch."
        ),
        controls=dict(PRIMARY_CONFIG),
        evaluation_method=(
            "benchmark/diagnostics/tile_sweep.py: for each of the 380 eval images, reuses the "
            "official baseline's full-image predictions (benchmark/results/baseline/"
            "predictions.jsonl, bit-identical, not re-inferred) plus NEW inference "
            "(predict_array_at, non-training) on the 4 crop tiles generated by "
            "benchmark/diagnostics/tiling.py::generate_tiles(). All 5 passes' hazard-class "
            "detections are remapped to full-image coordinates and merged via "
            "merge_detections_nms(). Evaluated against ground truth via the same "
            "benchmark.metrics matching code as every other experiment in this lab. "
            "TRUE_DETECTOR_MISS recovery is audited via the ACTUAL greedy-match exclusion logic "
            "(matches_at_fixed_iou['Person'].matched_gt_ids, mapped back to per-sample "
            "gt_index via a reconstructed global-to-per-sample index), not a naive per-case IoU "
            "check -- explicitly avoiding the EXP-0011 neighboring-GT artifact. Every claimed "
            "recovery is reported with its (sample_id, gt_index) for independent audit."
        ),
        success_criteria={
            "primary_metric": "person.recall",
            "min_meaningful_delta": 0.03,
            "precision_floor": 0.757,
            "guardrail_metrics": ["hazard.precision"],
            "sample_size_requirements": {"person": 100},
            "true_detector_miss_recovery": (
                "Reported DESCRIPTIVELY (count, fraction, per-case audit trail, small-vs-non-small "
                "breakdown) -- NOT used as a separate pass/fail gate. EXP-0006's own preregistered "
                "recovery threshold (>=22, calibrated against a specific training-based candidate's "
                "17/92) does not transfer to this inference-time-only context without a fabricated "
                "justification, so no numeric recovery threshold is invented here. The primary "
                "person.recall delta captures whether recovery, if any, is large enough to matter."
            ),
            "note": "All thresholds (0.757, 0.03) are UNCHANGED from EXP-0001 onward. If the point estimate passes, the same image-level bootstrap robustness check as EXP-0009 is required before treating it as credible.",
        },
        risks=(
            "Runs real (non-training) GPU inference (4 new tile passes per image, full-image pass "
            "reused) -- no training, no private data, no production/config changes (uses "
            "predict_array_at(), never predict(); benchmark/config.py never modified). New, "
            "correctness-sensitive spatial logic (coordinate remapping, cross-tile merge) is "
            "covered by 19 passing synthetic-geometry tests before this benchmark runs."
        ),
        expected_outcome=(
            "Possible outcomes, per this task's own framing: (a) positive -- meaningful recovery "
            "with robust precision; (b) detection-positive but impractical -- recovery at "
            "unacceptable Windows-runtime cost (evidence that effective scale matters even if not "
            "deployable); (c) negative -- no meaningful recovery, materially strengthening the "
            "representation/training-data hypothesis; (d) precision failure -- recall gain at an "
            "unacceptable false-positive cost. All four are informative and none is assumed here."
        ),
        parent_experiment_id=None,
        experiment_family="small_object",
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
