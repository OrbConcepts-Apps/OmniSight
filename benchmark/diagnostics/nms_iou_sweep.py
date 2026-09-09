"""EXP-0011: inference/NMS IoU sensitivity sweep.

VERIFIED (not assumed) meaning of the production `iou=0.7` parameter, before
writing this script: benchmark/config.py's own module docstring and
benchmark/model.py::predict_at() confirm `IOU_THRESHOLD` is passed straight
into ultralytics' `model.predict(..., iou=iou, ...)` -- the model's own
internal NMS (duplicate-box suppression) threshold, applied BEFORE any
prediction is ever returned. This is a COMPLETELY SEPARATE parameter from
`MAP50_IOU=0.5` (benchmark/config.py), which is the IoU used by
benchmark/metrics.py's greedy_match() to decide whether a *returned*
prediction counts as matching a *ground-truth* box for scoring -- that
number is never touched here. Unlike the Person confidence-threshold work
(EXP-0007-0010), NMS IoU CANNOT be swept by re-filtering an existing capture
-- it changes what the model itself returns, so this script runs NEW (but
non-training) inference at each grid point.

PRE-REGISTERED GRID (fixed before any grid-point result was computed;
symmetric and bounded around the production value, not an open-ended
search): NMS_IOU_GRID = [0.9, 0.8, 0.7, 0.6, 0.5]. 0.7 is the identity/
control point (must exactly reproduce the official baseline -- a
correctness check). Confidence threshold is FIXED at the production value
(0.4, benchmark/config.py::CONF_THRESHOLD) throughout -- this experiment
isolates NMS IoU as the only variable, deliberately NOT jointly optimizing
it with the Person-specific confidence threshold from EXP-0007-0010 (that
branch is closed; see research/memory/current_state.md).

MECHANISM ANALYSIS (written before any grid-point result was computed):
NMS IoU governs which of several MUTUALLY OVERLAPPING candidate boxes of
the SAME class survive as separate detections -- it can only choose among
boxes the detector ALREADY PROPOSED internally; it can never invent a
detection where the raw detector proposed nothing near a location. This
means:
  - It CANNOT recover a TRUE_DETECTOR_MISS case in the sense of "the model
    saw nothing there at all" -- there is no raw candidate box for NMS to
    keep in that scenario, at any IoU setting.
  - It COULD plausibly reveal a MISCLASSIFIED "TRUE_DETECTOR_MISS": if two
    real, adjacent people are both detected internally by the model but
    their boxes overlap by >= the current 0.7 threshold, NMS suppresses the
    lower-confidence one as if it were a duplicate of the SAME person --
    RAISING the IoU threshold (0.8, 0.9, less aggressive suppression) could
    let that second, genuinely-different-person detection survive, which
    would show up as a NEWLY MATCHED prediction near one of the 92 known
    baseline TRUE_DETECTOR_MISS GT boxes. This is the one concrete,
    falsifiable channel by which this experiment could affect
    TRUE_DETECTOR_MISS, and it is checked directly below (not inferred).
  - LOWERING the IoU threshold (0.6, 0.5, more aggressive suppression) can
    only ever REMOVE surviving boxes relative to production, never add one
    -- its only plausible benefit is fewer duplicate-detection false
    positives over a single person (a precision effect), at the risk of
    additionally suppressing genuine adjacent-person detections (a recall
    cost). It has no plausible channel to recover any TRUE_DETECTOR_MISS
    case.
  - Neither direction has any plausible mechanism for LOCALIZATION_FAILURE
    (box exists but IoU with GT is too low) or SEMANTIC_CLASS_CONFUSION
    (wrong class label) -- NMS operates within the SAME predicted class and
    does not alter box coordinates or class labels.

TRUE_DETECTOR_MISS recovery check (derivable consistently, computed
directly from each grid point's own new inference, no extra low-confidence
capture needed): benchmark/results/diagnostics/person_confusion_analysis.json
already lists the exact 92 baseline (iou=0.7) TRUE_DETECTOR_MISS cases as
(sample_id, gt_index, gt_bbox) records. For each grid point, this script
checks whether that grid point's own Person-class predictions (at
production conf=0.4) now match (IoU>=0.5, the same MATCH_IOU used
throughout this lab) any of those 92 specific GT boxes -- a true recovered
detection, not merely a nearby candidate -- and separately reports a looser
IoU>=0.3 "spatially associated" count for context (mirroring
person_confusion_analysis.py's own SPATIAL_ASSOC_IOU floor).

Duplicate-detection proxy: for each image's surviving Person-class boxes
(post-NMS, as returned by the model), count pairs with mutual IoU>=0.5 --
this is explicitly a proxy for "boxes plausibly covering the same physical
person" (could occasionally include two genuinely close-together people),
not a ground-truth duplicate label; its DIRECTION of change across the grid
is the interpretable signal, not its absolute value.

Run with: uv run python -m benchmark.diagnostics.nms_iou_sweep
Writes: benchmark/results/diagnostics/nms_iou_sweep.json
(runs real, non-training inference -- takes a few minutes on GPU)
"""

from __future__ import annotations

import json
import time

from benchmark.config import (
    CONF_THRESHOLD,
    EVAL_MANIFEST_PATH,
    HAZARD_CLASS_MAP,
    IOU_THRESHOLD,
    RAW_IMAGE_DIR,
    REPO_ROOT,
)
from benchmark.dataset import assert_eval_only, load_manifest
from benchmark.diagnostics.person_confusion_analysis import OUT_PATH as CONFUSION_ANALYSIS_PATH
from benchmark.metrics import Detection, GroundTruth, evaluate_detections, iou_xywh
from benchmark.model import BaselineModel

HAZARD_CLASSES = tuple(HAZARD_CLASS_MAP.values())

DIAG_DIR = REPO_ROOT / "benchmark" / "results" / "diagnostics"
OUT_PATH = DIAG_DIR / "nms_iou_sweep.json"

NMS_IOU_GRID = [0.9, 0.8, 0.7, 0.6, 0.5]
CONTROL_IOU = IOU_THRESHOLD  # 0.7, production
MATCH_IOU = 0.5
SPATIAL_ASSOC_IOU = 0.3
GUARDRAIL_FLOOR = 0.757  # unchanged from EXP-0001/0007/0008/0009/0010
MIN_MEANINGFUL_DELTA = 0.03  # unchanged, established since EXP-0001


def _load_true_detector_miss_cases() -> list:
    d = json.loads(CONFUSION_ANALYSIS_PATH.read_text(encoding="utf-8"))
    return [r for r in d["records"] if r["primary_category"] == "TRUE_DETECTOR_MISS"]


def run_one_iou(model: BaselineModel, samples: list, iou: float) -> dict:
    """Runs REAL inference over all 380 eval images at (conf=CONF_THRESHOLD,
    iou=iou), never touching benchmark/config.py or benchmark/results/baseline/
    (uses predict_at(), never predict())."""
    dets: list = []
    dets_by_sample: dict = {}
    latencies: list = []

    for sample in samples:
        image_path = RAW_IMAGE_DIR / sample.filename
        t0 = time.perf_counter()
        raw_preds = model.predict_at(image_path, conf=CONF_THRESHOLD, iou=iou)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        per_sample: list = []
        for p in raw_preds:
            det = Detection(sample_id=sample.sample_id, class_name=p.class_name, bbox=p.bbox, confidence=p.confidence)
            per_sample.append(det)
            if p.class_name in HAZARD_CLASSES:
                dets.append(det)
        dets_by_sample[sample.sample_id] = per_sample

    all_gts = [
        GroundTruth(sample_id=s.sample_id, class_name=lbl.class_name, bbox=lbl.bbox)
        for s in samples for lbl in s.labels
    ]

    overall, per_class, _ = evaluate_detections(dets, all_gts, list(HAZARD_CLASSES), map_ious=(0.5,))
    person = per_class["Person"]

    # Duplicate-Person-pair proxy (post-NMS surviving boxes, per image).
    duplicate_pairs = 0
    for sample_id, preds in dets_by_sample.items():
        person_boxes = [p for p in preds if p.class_name == "Person"]
        for i in range(len(person_boxes)):
            for j in range(i + 1, len(person_boxes)):
                if iou_xywh(person_boxes[i].bbox, person_boxes[j].bbox) >= MATCH_IOU:
                    duplicate_pairs += 1

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    return {
        "hazard": {"precision": overall.precision, "recall": overall.recall, "tp": overall.tp, "fp": overall.fp, "fn": overall.fn},
        "person": {
            "recall": person.recall, "precision": person.precision,
            "tp": person.tp, "fp": person.fp, "fn": person.fn, "num_gt": person.num_gt,
        },
        "duplicate_person_pairs": duplicate_pairs,
        "latency_ms": {"p50": p50, "p95": p95},
        "_dets_by_sample": dets_by_sample,  # internal, stripped before writing
    }


def sweep() -> dict:
    samples = load_manifest(EVAL_MANIFEST_PATH)
    assert_eval_only(samples)
    model = BaselineModel()
    tdm_cases = _load_true_detector_miss_cases()

    results: dict = {}
    for iou in NMS_IOU_GRID:
        r = run_one_iou(model, samples, iou)
        dets_by_sample = r.pop("_dets_by_sample")

        matched_strict = 0
        matched_loose = 0
        for case in tdm_cases:
            sid, gt_bbox = case["sample_id"], tuple(case["gt_bbox"])
            person_preds = [p for p in dets_by_sample.get(sid, []) if p.class_name == "Person"]
            best_iou = max((iou_xywh(p.bbox, gt_bbox) for p in person_preds), default=0.0)
            if best_iou >= MATCH_IOU:
                matched_strict += 1
            if best_iou >= SPATIAL_ASSOC_IOU:
                matched_loose += 1

        r["true_detector_miss_recovery"] = {
            "denominator": len(tdm_cases),
            "recovered_strict_iou_ge_0.5": matched_strict,
            "spatially_associated_iou_ge_0.3": matched_loose,
        }
        results[str(iou)] = r
        print(f"  iou={iou}: hazard.P={r['hazard']['precision']:.4f} person.R={r['person']['recall']:.4f} "
              f"person.P={r['person']['precision']:.4f} dup_pairs={r['duplicate_person_pairs']} "
              f"TDM_recovered={matched_strict}/{len(tdm_cases)}")

    control = results[str(CONTROL_IOU)]
    baseline_hazard_precision = control["hazard"]["precision"]
    baseline_person_recall = control["person"]["recall"]

    grid_summary: dict = {}
    for iou in NMS_IOU_GRID:
        r = results[str(iou)]
        delta_recall = r["person"]["recall"] - baseline_person_recall
        guardrail_satisfied = r["hazard"]["precision"] >= GUARDRAIL_FLOOR
        meaningful = delta_recall >= MIN_MEANINGFUL_DELTA
        grid_summary[str(iou)] = {
            "person_recall_delta": delta_recall,
            "guardrail_satisfied": guardrail_satisfied,
            "meaningful_delta": meaningful,
            "point_estimate_pass": guardrail_satisfied and meaningful and iou != CONTROL_IOU,
        }

    return {
        "procedure": {
            "nms_iou_grid": NMS_IOU_GRID,
            "control_iou": CONTROL_IOU,
            "confidence_threshold_fixed_at": CONF_THRESHOLD,
            "guardrail_floor": GUARDRAIL_FLOOR,
            "min_meaningful_delta": MIN_MEANINGFUL_DELTA,
            "match_iou": MATCH_IOU,
            "spatial_assoc_iou": SPATIAL_ASSOC_IOU,
        },
        "grid": results,
        "summary": grid_summary,
        "any_point_estimate_pass": any(g["point_estimate_pass"] for g in grid_summary.values()),
    }


def main() -> None:
    results = sweep()
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print(f"any_point_estimate_pass={results['any_point_estimate_pass']}")


if __name__ == "__main__":
    main()
