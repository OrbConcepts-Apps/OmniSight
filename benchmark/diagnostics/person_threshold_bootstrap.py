"""EXP-0009: image-level bootstrap uncertainty analysis of EXP-0008's
person_threshold=0.30 result.

POST-HOC ROBUSTNESS ANALYSIS, not a retroactive change to EXP-0008. EXP-0008
preregistered no uncertainty criterion (a single deterministic point-estimate
evaluation, like every other experiment in this lab to date) -- this script
answers a different, explicitly-labeled question: does EXP-0008's PASS
survive image-level resampling of the SAME frozen 380-image eval set, or is
the guardrail margin (+0.0098) an artifact of this one particular sample.

No new inference, no training, no private data. Reuses the SAME conf=0.01
capture (low_conf_predictions.jsonl) and GT manifest (eval_manifest.jsonl)
EXP-0001/0007/0008 already use.

METHOD (image-level, not box-level, resampling):
  - The resampling unit is the WHOLE IMAGE: every detection and every
    ground-truth box belonging to a sampled image travel together. This
    respects the real dependence structure (boxes within one image are not
    independent draws) -- box-level resampling would be invalid here.
  - Each bootstrap replicate draws 380 image ids WITH REPLACEMENT from the
    380 real images. When an image is drawn more than once in one replicate,
    each occurrence is treated as an independent "virtual" image (its
    detections/GTs relabeled with a synthetic per-occurrence sample_id) so
    the greedy IoU matcher (benchmark/metrics.py, sample_id-keyed) does not
    let duplicate occurrences of the same image compete for the same GT
    boxes -- each occurrence gets its own full, unclaimed GT set, which is
    the statistically correct way to bootstrap a matching-based metric.
  - Metrics are RECOMPUTED FROM SCRATCH on each resampled image set (full
    re-run of evaluate_detections), never bootstrapped from already-
    aggregated point estimates.
  - RNG: numpy.random.default_rng(SEED) with a fixed, recorded seed --
    fully reproducible.

PRE-REGISTERED ROBUSTNESS CRITERION (fixed here, before any bootstrap
replicate is computed or inspected -- see research/_exp0009_preregister.py
for the formal DB record):
  ROBUST (maps to research_verdict=PASS) if BOTH:
    (a) guardrail_violation_rate <= 0.05 -- hazard.precision@person=0.30
        stays >= 0.757 in at least 95% of bootstrap replicates, AND
    (b) the 2.5th percentile of the bootstrap distribution of
        (person.recall@0.30 - person.recall@0.40) is > 0 -- the recall
        improvement's direction is robust at the 95% (two-sided) level,
        not just its point estimate.
  FRAGILE (maps to research_verdict=FAIL) if guardrail_violation_rate > 0.05
    (the guardrail is unreliable under resampling -- a safety-relevant
    finding, reported honestly even though EXP-0008 itself PASSED).
  Otherwise INCONCLUSIVE (e.g. guardrail mostly holds but the recall-
    improvement direction is not robust, or vice versa).
This criterion was chosen for standard reasons (a 95%-level percentile
interval on each of the two things EXP-0008's PASS actually depends on) and
was NOT tuned after inspecting any bootstrap result.

Run with: uv run python -m benchmark.diagnostics.person_threshold_bootstrap
Writes: benchmark/results/diagnostics/person_threshold_bootstrap.json
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
OUT_PATH = DIAG_DIR / "person_threshold_bootstrap.json"

PERSON_THRESHOLD = 0.30
OTHER_HAZARD_CLASS_THRESHOLD = 0.40
GUARDRAIL_FLOOR = 0.757  # baseline_hazard_precision(0.8070...) - 0.05, same value EXP-0001/0007/0008 use
N_REPLICATES = 2000
SEED = 20260909  # fixed, recorded -- chosen once, before any replicate was computed

GUARDRAIL_VIOLATION_RATE_MAX = 0.05
CI_LOW_PCT, CI_HIGH_PCT = 2.5, 97.5


def load_by_image() -> tuple[dict, dict, list]:
    """Returns (dets_by_image, gts_by_image, image_ids) -- every detection/GT
    grouped by its real sample_id, plus the ordered list of the 380 real
    image ids to resample from."""
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
            GroundTruth(sample_id=sid, class_name=lbl.class_name, bbox=lbl.bbox)
            for lbl in sample.labels
        ]
        dets_by_image.setdefault(sid, [])

    return dets_by_image, gts_by_image, image_ids


def _evaluate_replicate(sampled_ids: list, dets_by_image: dict, gts_by_image: dict, person_threshold: float) -> dict:
    """Builds one virtual image set from sampled_ids (relabeling duplicate
    occurrences with a synthetic per-occurrence sample_id so they don't
    compete for the same GT boxes), filters at person_threshold (Person) /
    OTHER_HAZARD_CLASS_THRESHOLD (everything else), and recomputes hazard +
    Person metrics from scratch via evaluate_detections."""
    dets: list = []
    gts: list = []
    for occurrence_index, sid in enumerate(sampled_ids):
        virtual_sid = f"{sid}__occ{occurrence_index}"
        for d in dets_by_image[sid]:
            if (d.class_name == "Person" and d.confidence >= person_threshold) or (
                d.class_name != "Person" and d.confidence >= OTHER_HAZARD_CLASS_THRESHOLD
            ):
                dets.append(Detection(sample_id=virtual_sid, class_name=d.class_name, bbox=d.bbox, confidence=d.confidence))
        for g in gts_by_image[sid]:
            gts.append(GroundTruth(sample_id=virtual_sid, class_name=g.class_name, bbox=g.bbox))

    overall, per_class, _ = evaluate_detections(dets, gts, list(HAZARD_CLASSES), map_ious=(0.5,))
    person = per_class["Person"]
    return {"hazard_precision": overall.precision, "person_recall": person.recall, "person_precision": person.precision}


def bootstrap() -> dict:
    dets_by_image, gts_by_image, image_ids = load_by_image()
    n_images = len(image_ids)
    rng = np.random.default_rng(SEED)

    at_030: list = []
    at_040: list = []

    for _ in range(N_REPLICATES):
        sampled = list(rng.choice(image_ids, size=n_images, replace=True))
        at_030.append(_evaluate_replicate(sampled, dets_by_image, gts_by_image, PERSON_THRESHOLD))
        at_040.append(_evaluate_replicate(sampled, dets_by_image, gts_by_image, OTHER_HAZARD_CLASS_THRESHOLD))

    hazard_precision_030 = np.array([r["hazard_precision"] for r in at_030])
    person_recall_030 = np.array([r["person_recall"] for r in at_030])
    person_precision_030 = np.array([r["person_precision"] for r in at_030])
    person_recall_040 = np.array([r["person_recall"] for r in at_040])
    hazard_precision_040 = np.array([r["hazard_precision"] for r in at_040])

    delta_recall = person_recall_030 - person_recall_040
    delta_hazard_precision = hazard_precision_030 - hazard_precision_040

    guardrail_violation_rate = float(np.mean(hazard_precision_030 < GUARDRAIL_FLOOR))
    recall_delta_ci_low = float(np.percentile(delta_recall, CI_LOW_PCT))
    recall_delta_ci_high = float(np.percentile(delta_recall, CI_HIGH_PCT))
    delta_ge_0p03_rate = float(np.mean(delta_recall >= 0.03))

    robust = guardrail_violation_rate <= GUARDRAIL_VIOLATION_RATE_MAX and recall_delta_ci_low > 0.0
    fragile = guardrail_violation_rate > GUARDRAIL_VIOLATION_RATE_MAX
    classification = "ROBUST" if robust else ("FRAGILE" if fragile else "INCONCLUSIVE")

    def _pct(arr):
        return {
            "mean": float(np.mean(arr)),
            "ci_2.5": float(np.percentile(arr, CI_LOW_PCT)),
            "ci_97.5": float(np.percentile(arr, CI_HIGH_PCT)),
        }

    return {
        "procedure": {
            "resampling_unit": "image (all detections+GT for a sampled image travel together)",
            "n_replicates": N_REPLICATES,
            "n_images": n_images,
            "seed": SEED,
            "rng": "numpy.random.default_rng",
            "person_threshold_tested": PERSON_THRESHOLD,
            "baseline_threshold": OTHER_HAZARD_CLASS_THRESHOLD,
            "guardrail_floor": GUARDRAIL_FLOOR,
            "recomputed_from_scratch_per_replicate": True,
        },
        "pre_registered_robustness_criterion": (
            "ROBUST if guardrail_violation_rate <= 0.05 AND 2.5th-percentile(delta person.recall) > 0; "
            "FRAGILE if guardrail_violation_rate > 0.05; else INCONCLUSIVE. Fixed before any replicate was computed."
        ),
        "metrics_at_person_threshold_0.30": {
            "hazard_precision": _pct(hazard_precision_030),
            "person_recall": _pct(person_recall_030),
            "person_precision": _pct(person_precision_030),
        },
        "delta_vs_baseline_0.40": {
            "person_recall_delta": _pct(delta_recall),
            "hazard_precision_delta": _pct(delta_hazard_precision),
        },
        "guardrail_violation_rate": guardrail_violation_rate,
        "person_recall_delta_ge_0.03_rate": delta_ge_0p03_rate,
        "classification": classification,
    }


def main() -> None:
    if not LOW_CONF_PATH.exists():
        raise FileNotFoundError(f"{LOW_CONF_PATH} not found.")
    results = bootstrap()
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
