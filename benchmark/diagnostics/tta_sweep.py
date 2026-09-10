"""EXP-0012: test-time augmentation (TTA) sensitivity test.

Distinct mechanism from BOTH now-closed lines of inquiry (EXP-0007-0010
Person confidence threshold, EXP-0011 NMS IoU): those can only choose among
or re-threshold candidate boxes the detector's single forward pass already
proposed -- neither can ever produce a detection where the original pass
proposed nothing. TTA (ultralytics' built-in `augment=True`) runs SEVERAL
transformed views of the same image (typically horizontal flip plus one or
more scales) through the SAME model and merges all views' candidate boxes
via NMS -- a genuinely different view CAN produce a candidate detection the
original, single, un-augmented pass did not, giving TTA a real (if
unproven) channel to recover a TRUE_DETECTOR_MISS case, not just
re-arrange existing candidates.

This is a single ON/OFF test (augment=False control vs augment=True
candidate), NOT a hyperparameter grid -- ultralytics' TTA implementation is
not exposed as a tunable parameter this script sweeps; there is nothing to
"cherry-pick" a value from. Confidence (0.4) and NMS IoU (0.7) are FIXED at
the production values throughout -- TTA on/off is the only variable,
deliberately not jointly optimized with either closed branch.

Real (non-training) inference, reusing benchmark/model.py::BaselineModel.
predict_at()'s new `augment` parameter (default False, byte-identical to
every existing caller's prior behavior). Never touches
benchmark/results/baseline/ or benchmark/config.py.

COST TRADEOFF (stated before any result is computed): TTA typically runs
the model 2-3x per image (each transformed view is a separate forward
pass), so a real latency increase is expected and is measured directly here
against this lab's existing latency guardrail (max +50% p95 vs baseline,
research/evaluation_policy.py::default_hazard_policy) -- a recall
improvement that comes with an unacceptable latency cost is a real,
reportable constraint for a real-time assistive-vision system, not
something to omit.

TRUE_DETECTOR_MISS recovery check: identical method to EXP-0011 (checks
each of the 92 known baseline TRUE_DETECTOR_MISS cases,
person_confusion_analysis.json, for a newly-matching Person prediction at
IoU>=0.5 / spatially-associated at IoU>=0.3).

Run with: uv run python -m benchmark.diagnostics.tta_sweep
Writes: benchmark/results/diagnostics/tta_sweep.json
(runs real, non-training inference -- takes a few minutes on GPU)
"""

from __future__ import annotations

import json
import time

from benchmark.config import CONF_THRESHOLD, EVAL_MANIFEST_PATH, HAZARD_CLASS_MAP, IOU_THRESHOLD, RAW_IMAGE_DIR, REPO_ROOT
from benchmark.dataset import assert_eval_only, load_manifest
from benchmark.diagnostics.person_confusion_analysis import OUT_PATH as CONFUSION_ANALYSIS_PATH
from benchmark.metrics import Detection, GroundTruth, evaluate_detections, iou_xywh
from benchmark.model import BaselineModel

HAZARD_CLASSES = tuple(HAZARD_CLASS_MAP.values())

DIAG_DIR = REPO_ROOT / "benchmark" / "results" / "diagnostics"
OUT_PATH = DIAG_DIR / "tta_sweep.json"

MATCH_IOU = 0.5
SPATIAL_ASSOC_IOU = 0.3
GUARDRAIL_FLOOR = 0.757
MIN_MEANINGFUL_DELTA = 0.03
MAX_LATENCY_REGRESSION_PCT = 50.0  # matches default_hazard_policy's existing convention


def _load_true_detector_miss_cases() -> list:
    d = json.loads(CONFUSION_ANALYSIS_PATH.read_text(encoding="utf-8"))
    return [r for r in d["records"] if r["primary_category"] == "TRUE_DETECTOR_MISS"]


def run_one_condition(model: BaselineModel, samples: list, augment: bool) -> dict:
    dets: list = []
    dets_by_sample: dict = {}
    latencies: list = []

    for sample in samples:
        image_path = RAW_IMAGE_DIR / sample.filename
        t0 = time.perf_counter()
        raw_preds = model.predict_at(image_path, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, augment=augment)
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

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    return {
        "hazard": {"precision": overall.precision, "recall": overall.recall, "tp": overall.tp, "fp": overall.fp, "fn": overall.fn},
        "person": {
            "recall": person.recall, "precision": person.precision,
            "tp": person.tp, "fp": person.fp, "fn": person.fn, "num_gt": person.num_gt,
        },
        "latency_ms": {"p50": p50, "p95": p95},
        "_dets_by_sample": dets_by_sample,
    }


def sweep() -> dict:
    samples = load_manifest(EVAL_MANIFEST_PATH)
    assert_eval_only(samples)
    model = BaselineModel()
    tdm_cases = _load_true_detector_miss_cases()

    results: dict = {}
    for augment in (False, True):
        r = run_one_condition(model, samples, augment)
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
        results["augment_" + str(augment)] = r
        print(f"  augment={augment}: hazard.P={r['hazard']['precision']:.4f} person.R={r['person']['recall']:.4f} "
              f"person.P={r['person']['precision']:.4f} p95_ms={r['latency_ms']['p95']:.1f} "
              f"TDM_recovered={matched_strict}/{len(tdm_cases)}")

    control = results["augment_False"]
    candidate = results["augment_True"]
    baseline_hazard_precision = control["hazard"]["precision"]
    baseline_person_recall = control["person"]["recall"]
    baseline_p95 = control["latency_ms"]["p95"]

    delta_recall = candidate["person"]["recall"] - baseline_person_recall
    guardrail_satisfied = candidate["hazard"]["precision"] >= GUARDRAIL_FLOOR
    meaningful = delta_recall >= MIN_MEANINGFUL_DELTA
    latency_regression_pct = (candidate["latency_ms"]["p95"] - baseline_p95) / baseline_p95 * 100.0
    latency_ok = latency_regression_pct <= MAX_LATENCY_REGRESSION_PCT

    return {
        "procedure": {
            "confidence_threshold_fixed_at": CONF_THRESHOLD,
            "nms_iou_fixed_at": IOU_THRESHOLD,
            "guardrail_floor": GUARDRAIL_FLOOR,
            "min_meaningful_delta": MIN_MEANINGFUL_DELTA,
            "max_latency_regression_pct": MAX_LATENCY_REGRESSION_PCT,
        },
        "conditions": results,
        "summary": {
            "person_recall_delta": delta_recall,
            "guardrail_satisfied": guardrail_satisfied,
            "meaningful_delta": meaningful,
            "latency_regression_pct": latency_regression_pct,
            "latency_guardrail_satisfied": latency_ok,
            "point_estimate_pass": guardrail_satisfied and meaningful and latency_ok,
        },
    }


def main() -> None:
    results = sweep()
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
