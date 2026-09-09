"""EXP-0010: threshold-sensitivity bootstrap around EXP-0008/0009's
person_threshold=0.30 finding.

EXP-0009 showed person_threshold=0.30's hazard-precision guardrail margin
(+0.0098 point estimate) is FRAGILE under image-level bootstrap resampling
(36.9% of replicates violate the 0.757 floor). This raises the direct next
question the coarse original grid ([0.40,0.30,0.20,0.10],
per_class_threshold_sweep.py) cannot answer: is there a MORE conservative
threshold, between 0.30 and 0.40, that keeps a meaningful recall
improvement while holding a robust (not razor-thin) guardrail margin -- or
was 0.30 simply the most favorable-looking point on a coarse grid, with
nothing in between actually safer?

Reuses the SAME conf=0.01 capture and image-level bootstrap methodology as
EXP-0009 (person_threshold_bootstrap.py) -- fixed seed, image-level
resampling with synthetic per-occurrence sample_ids for duplicates, metrics
recomputed from scratch per replicate -- but draws each bootstrap replicate
ONCE and evaluates the ENTIRE finer grid on that same resampled image set
(more efficient than re-resampling per threshold, and statistically cleaner:
comparisons across thresholds within one replicate share the same resample).

PRE-REGISTERED GRID (fixed before any replicate was computed):
  PERSON_THRESHOLDS = [0.40, 0.38, 0.36, 0.34, 0.32, 0.30]
  (0.40 = production control; 0.30 = EXP-0008/0009's candidate, included for
  direct continuity/comparison; the four intermediate points are the new
  sensitivity probes).

PRE-REGISTERED PER-THRESHOLD CLASSIFICATION (identical rule to EXP-0009,
applied independently to each grid point): ROBUST if
guardrail_violation_rate<=0.05 AND 2.5th-percentile(delta person.recall)>0;
FRAGILE if guardrail_violation_rate>0.05; else INCONCLUSIVE.

Run with: uv run python -m benchmark.diagnostics.person_threshold_sensitivity_bootstrap
Writes: benchmark/results/diagnostics/person_threshold_sensitivity_bootstrap.json
"""

from __future__ import annotations

import json

import numpy as np

from benchmark.config import EVAL_MANIFEST_PATH, HAZARD_CLASS_MAP, REPO_ROOT
from benchmark.dataset import load_manifest
from benchmark.metrics import Detection, GroundTruth, evaluate_detections

HAZARD_CLASSES = tuple(HAZARD_CLASS_MAP.values())

DIAG_DIR = REPO_ROOT / "benchmark" / "results" / "diagnostics"
LOW_CONF_PATH = DIAG_DIR / "low_conf_predictions.jsonl"
OUT_PATH = DIAG_DIR / "person_threshold_sensitivity_bootstrap.json"

PERSON_THRESHOLDS = [0.40, 0.38, 0.36, 0.34, 0.32, 0.30]
BASELINE_THRESHOLD = 0.40
OTHER_HAZARD_CLASS_THRESHOLD = 0.40
GUARDRAIL_FLOOR = 0.757
N_REPLICATES = 2000
SEED = 20260909  # same seed as EXP-0009 -- identical resamples, extended grid

GUARDRAIL_VIOLATION_RATE_MAX = 0.05
CI_LOW_PCT, CI_HIGH_PCT = 2.5, 97.5


def load_by_image() -> tuple[dict, dict, list]:
    dets_by_image: dict = {}
    with open(LOW_CONF_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            sid = rec["sample_id"]
            dets_by_image.setdefault(sid, [])
            for p in rec["predictions"]:
                if p["class_name"] in HAZARD_CLASSES:
                    dets_by_image[sid].append(
                        Detection(sample_id=sid, class_name=p["class_name"], bbox=tuple(p["bbox"]), confidence=p["confidence"])
                    )

    gts_by_image: dict = {}
    image_ids: list = []
    for sample in load_manifest(EVAL_MANIFEST_PATH):
        sid = sample.sample_id
        image_ids.append(sid)
        gts_by_image[sid] = [
            GroundTruth(sample_id=sid, class_name=lbl.class_name, bbox=lbl.bbox) for lbl in sample.labels
        ]
        dets_by_image.setdefault(sid, [])

    return dets_by_image, gts_by_image, image_ids


def _build_virtual_pool(sampled_ids: list, dets_by_image: dict, gts_by_image: dict) -> tuple[list, list]:
    """One resample's full detection/GT pool at the lowest threshold in the
    grid (0.30) -- every higher threshold is a strict subset filter of this
    same pool, computed per-threshold below without re-resampling."""
    dets: list = []
    gts: list = []
    for occurrence_index, sid in enumerate(sampled_ids):
        virtual_sid = f"{sid}__occ{occurrence_index}"
        for d in dets_by_image[sid]:
            dets.append(Detection(sample_id=virtual_sid, class_name=d.class_name, bbox=d.bbox, confidence=d.confidence))
        for g in gts_by_image[sid]:
            gts.append(GroundTruth(sample_id=virtual_sid, class_name=g.class_name, bbox=g.bbox))
    return dets, gts


def _evaluate_at_threshold(all_dets: list, all_gts: list, person_threshold: float) -> dict:
    filtered = [
        d for d in all_dets
        if (d.class_name == "Person" and d.confidence >= person_threshold)
        or (d.class_name != "Person" and d.confidence >= OTHER_HAZARD_CLASS_THRESHOLD)
    ]
    overall, per_class, _ = evaluate_detections(filtered, all_gts, list(HAZARD_CLASSES), map_ious=(0.5,))
    person = per_class["Person"]
    return {"hazard_precision": overall.precision, "person_recall": person.recall, "person_precision": person.precision}


def sensitivity_bootstrap() -> dict:
    dets_by_image, gts_by_image, image_ids = load_by_image()
    n_images = len(image_ids)
    rng = np.random.default_rng(SEED)

    per_threshold_results: dict = {str(pt): [] for pt in PERSON_THRESHOLDS}

    for _ in range(N_REPLICATES):
        sampled = list(rng.choice(image_ids, size=n_images, replace=True))
        all_dets, all_gts = _build_virtual_pool(sampled, dets_by_image, gts_by_image)
        for pt in PERSON_THRESHOLDS:
            per_threshold_results[str(pt)].append(_evaluate_at_threshold(all_dets, all_gts, pt))

    baseline_recall = np.array([r["person_recall"] for r in per_threshold_results[str(BASELINE_THRESHOLD)]])

    def _pct(arr):
        return {"mean": float(np.mean(arr)), "ci_2.5": float(np.percentile(arr, CI_LOW_PCT)), "ci_97.5": float(np.percentile(arr, CI_HIGH_PCT))}

    grid_results: dict = {}
    for pt in PERSON_THRESHOLDS:
        recs = per_threshold_results[str(pt)]
        hazard_precision = np.array([r["hazard_precision"] for r in recs])
        person_recall = np.array([r["person_recall"] for r in recs])
        person_precision = np.array([r["person_precision"] for r in recs])
        delta_recall = person_recall - baseline_recall

        violation_rate = float(np.mean(hazard_precision < GUARDRAIL_FLOOR))
        ci_low = float(np.percentile(delta_recall, CI_LOW_PCT))
        robust = violation_rate <= GUARDRAIL_VIOLATION_RATE_MAX and ci_low > 0.0
        fragile = violation_rate > GUARDRAIL_VIOLATION_RATE_MAX
        classification = "ROBUST" if robust else ("FRAGILE" if fragile else "INCONCLUSIVE")

        grid_results[str(pt)] = {
            "hazard_precision": _pct(hazard_precision),
            "person_recall": _pct(person_recall),
            "person_precision": _pct(person_precision),
            "recall_delta_vs_0.40": _pct(delta_recall),
            "guardrail_violation_rate": violation_rate,
            "classification": classification,
        }

    robust_thresholds = [pt for pt in PERSON_THRESHOLDS if grid_results[str(pt)]["classification"] == "ROBUST" and pt != BASELINE_THRESHOLD]
    # Most conservative (highest, i.e. smallest recall gain but safest precision) robust candidate, if any.
    safest_robust = max(robust_thresholds) if robust_thresholds else None

    return {
        "procedure": {
            "resampling_unit": "image (all detections+GT for a sampled image travel together)",
            "n_replicates": N_REPLICATES,
            "n_images": n_images,
            "seed": SEED,
            "person_thresholds_tested": PERSON_THRESHOLDS,
            "guardrail_floor": GUARDRAIL_FLOOR,
        },
        "pre_registered_criterion": (
            "Per grid point: ROBUST if guardrail_violation_rate<=0.05 AND "
            "2.5th-percentile(delta person.recall vs 0.40)>0; FRAGILE if "
            "guardrail_violation_rate>0.05; else INCONCLUSIVE. Fixed before any replicate computed."
        ),
        "grid": grid_results,
        "any_robust_threshold_found": len(robust_thresholds) > 0,
        "safest_robust_threshold": safest_robust,
    }


def main() -> None:
    if not LOW_CONF_PATH.exists():
        raise FileNotFoundError(f"{LOW_CONF_PATH} not found.")
    results = sensitivity_bootstrap()
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    for pt in PERSON_THRESHOLDS:
        g = results["grid"][str(pt)]
        print(f"  person_conf>={pt:.2f}: hazard.P_mean={g['hazard_precision']['mean']:.3f} "
              f"violation_rate={g['guardrail_violation_rate']:.3f} classification={g['classification']}")
    print(f"any_robust_threshold_found={results['any_robust_threshold_found']} safest={results['safest_robust_threshold']}")


if __name__ == "__main__":
    main()
