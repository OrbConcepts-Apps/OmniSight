"""EXP-0007 diagnostic: PER-CLASS confidence threshold policy.

Distinct independent variable from EXP-0001's GLOBAL threshold sweep
(benchmark/diagnostics/threshold_sweep.py, which lowers ALL hazard classes'
confidence cutoff uniformly). Here, only the Person class's confidence
threshold is varied; every OTHER hazard class stays fixed at the production
cutoff (0.4). Mechanism under test: EXP-0001 showed a GLOBAL threshold drop
recovers Person recall but collapses hazard precision -- but per
person_confusion_analysis.md's counterfactual work and the raw per-class
sweep, classes like Car have far worse low-confidence precision than Person
does (Car precision=0.295 at conf=0.05 vs the aggregate hazard figure of
0.381), meaning a global drop's precision damage is disproportionately driven
by NON-Person classes. Isolating the threshold change to Person alone tests
whether that collateral damage -- not the Person/recall tradeoff itself -- was
the real cause of EXP-0001's guardrail violation.

Reuses the SAME single low-confidence capture used by threshold_sweep.py
(benchmark/results/diagnostics/low_conf_predictions.jsonl, conf=0.01,
produced by capture_low_conf.py) and the SAME matching code
(benchmark/metrics.py) -- no new model inference, no training, no private
data, no device deployment.

PRE-REGISTERED GRID (fixed before any per-class-isolated result was computed
or inspected -- see research/_exp0007_preregister.py):
  PERSON_THRESHOLDS = [0.40, 0.30, 0.20, 0.10]
  0.40 is the identity/control point (must exactly reproduce the official
  baseline's Person and hazard-aggregate metrics -- a correctness check).
  OTHER_HAZARD_CLASS_THRESHOLD = 0.40 (fixed, matches production for every
  hazard class except Person).

PRE-REGISTERED SELECTION RULE (fixed before inspecting results): among grid
points whose hazard-aggregate precision satisfies the guardrail
(>= baseline_hazard_precision - 0.05), select the one with the highest
person.recall. If none satisfy the guardrail, the representative is 0.40
(no viable candidate).

Run with: uv run python -m benchmark.diagnostics.per_class_threshold_sweep
Writes: benchmark/results/diagnostics/per_class_threshold_sweep.json
"""

from __future__ import annotations

import json

from benchmark.config import EVAL_MANIFEST_PATH, HAZARD_CLASS_MAP, REPO_ROOT
from benchmark.dataset import load_manifest
from benchmark.metrics import Detection, GroundTruth, evaluate_detections

HAZARD_CLASSES = tuple(HAZARD_CLASS_MAP.values())

DIAG_DIR = REPO_ROOT / "benchmark" / "results" / "diagnostics"
LOW_CONF_PATH = DIAG_DIR / "low_conf_predictions.jsonl"
OUT_PATH = DIAG_DIR / "per_class_threshold_sweep.json"

PERSON_THRESHOLDS = [0.40, 0.30, 0.20, 0.10]
OTHER_HAZARD_CLASS_THRESHOLD = 0.40
PRECISION_GUARDRAIL_MARGIN = 0.05


def load_low_conf_detections() -> list:
    dets = []
    with open(LOW_CONF_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for p in rec["predictions"]:
                dets.append(
                    Detection(
                        sample_id=rec["sample_id"],
                        class_name=p["class_name"],
                        bbox=tuple(p["bbox"]),
                        confidence=p["confidence"],
                    )
                )
    return dets


def load_ground_truths() -> list:
    gts = []
    for sample in load_manifest(EVAL_MANIFEST_PATH):
        for lbl in sample.labels:
            gts.append(GroundTruth(sample_id=sample.sample_id, class_name=lbl.class_name, bbox=lbl.bbox))
    return gts


def sweep() -> dict:
    all_dets = load_low_conf_detections()
    all_gts = load_ground_truths()

    results = {"person_thresholds": {}}

    for pt in PERSON_THRESHOLDS:
        filtered = [
            d for d in all_dets
            if d.class_name in HAZARD_CLASSES and (
                (d.class_name == "Person" and d.confidence >= pt)
                or (d.class_name != "Person" and d.confidence >= OTHER_HAZARD_CLASS_THRESHOLD)
            )
        ]
        overall, per_class, _ = evaluate_detections(filtered, all_gts, list(HAZARD_CLASSES), map_ious=(0.5,))
        results["person_thresholds"][str(pt)] = {
            "hazard_overall": {
                "precision": overall.precision, "recall": overall.recall, "f1": overall.f1,
                "tp": overall.tp, "fp": overall.fp, "fn": overall.fn,
                "num_gt": overall.num_gt, "num_predictions": overall.num_predictions,
            },
            "person": {
                "precision": per_class["Person"].precision, "recall": per_class["Person"].recall,
                "tp": per_class["Person"].tp, "fp": per_class["Person"].fp, "fn": per_class["Person"].fn,
                "num_gt": per_class["Person"].num_gt,
            },
        }

    baseline_hazard_precision = results["person_thresholds"]["0.4"]["hazard_overall"]["precision"]
    guardrail_floor = baseline_hazard_precision - PRECISION_GUARDRAIL_MARGIN

    eligible = [
        pt for pt in PERSON_THRESHOLDS
        if results["person_thresholds"][str(pt)]["hazard_overall"]["precision"] >= guardrail_floor
    ]
    representative = max(eligible, key=lambda pt: results["person_thresholds"][str(pt)]["person"]["recall"]) if eligible else 0.40

    results["pre_registered_selection_rule"] = (
        "Among PERSON_THRESHOLDS whose hazard.precision >= baseline_hazard_precision - 0.05, "
        "pick the one with highest person.recall; if none qualify, representative=0.40 (no viable candidate)."
    )
    results["baseline_hazard_precision"] = baseline_hazard_precision
    results["guardrail_floor"] = guardrail_floor
    results["representative_person_threshold"] = representative
    results["other_hazard_class_threshold"] = OTHER_HAZARD_CLASS_THRESHOLD
    return results


def main() -> None:
    if not LOW_CONF_PATH.exists():
        raise FileNotFoundError(
            f"{LOW_CONF_PATH} not found. Run `uv run python -m benchmark.diagnostics.capture_low_conf` first."
        )
    results = sweep()
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    for pt in PERSON_THRESHOLDS:
        b = results["person_thresholds"][str(pt)]
        print(f"  person_conf>={pt:.2f}: hazard.P={b['hazard_overall']['precision']:.3f} "
              f"hazard.R={b['hazard_overall']['recall']:.3f} person.R={b['person']['recall']:.3f} "
              f"person.P={b['person']['precision']:.3f}")
    print(f"Representative (pre-registered rule): person_threshold={results['representative_person_threshold']}")


if __name__ == "__main__":
    main()
