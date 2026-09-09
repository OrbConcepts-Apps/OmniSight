"""One-shot registration of EXP-0007 (per-class confidence threshold policy),
mirroring research/seed_experiments.py's EXP-0001 pattern exactly: a
confirmatory/diagnostic-only experiment, no new model inference, no training,
no private data, no device deployment, no new human approval required.

Run via:

    uv run python -m research._exp0007_preregister

Idempotent: no-ops if EXP-0007 already exists.

Orthogonal to the blocked EXP-0006 pilot: this experiment answers a
different, currently-authorized question -- whether ISOLATING a lower
confidence threshold to the Person class alone (holding every other hazard
class fixed at the production 0.4 cutoff) recovers Person recall without
violating the hazard-precision guardrail, unlike EXP-0001's GLOBAL threshold
sweep (which lowered every hazard class uniformly and collapsed precision).
Evidence source: benchmark/results/diagnostics/per_class_threshold_sweep.json
(benchmark/diagnostics/per_class_threshold_sweep.py), built from the SAME
already-captured low_conf_predictions.jsonl (conf=0.01) used by EXP-0001 --
no new inference.
"""

from __future__ import annotations

from research.config import EXPERIMENTS_DIR
from research.db import Experiment, OmniLabDB
from research.experiment_lifecycle import move_to_status
from research.experiment_schema import write_queued_artifacts

EXPERIMENT_ID = "EXP-0007"


def _experiment() -> Experiment:
    return Experiment(
        experiment_id=EXPERIMENT_ID,
        hypothesis=(
            "Isolating the confidence-threshold reduction to the Person class ONLY (holding "
            "every other hazard class fixed at the production conf=0.4 cutoff) recovers "
            "meaningful Person recall (>= +0.03 over baseline) while keeping hazard-aggregate "
            "precision within the standard guardrail (>= baseline - 0.05), unlike EXP-0001's "
            "GLOBAL uniform threshold drop, which collapsed hazard precision well past that "
            "guardrail. Mechanism: non-Person hazard classes (esp. Car, precision=0.295 at "
            "conf=0.05 per threshold_sweep.json) contribute disproportionate low-confidence "
            "false positives to the AGGREGATE hazard-precision guardrail under a global drop; "
            "isolating the threshold change to Person alone should avoid that collateral damage."
        ),
        motivation=(
            "EXP-0001 confirmed threshold reduction cannot fix Person recall GLOBALLY without "
            "unacceptable precision loss. That result conflates two effects: (a) Person's own "
            "confidence/precision tradeoff, and (b) collateral false-positive cost from OTHER "
            "hazard classes' low-confidence predictions, which the aggregate hazard-precision "
            "guardrail also penalizes. This experiment isolates (a) from (b) -- a genuinely "
            "different independent variable, not a re-run of EXP-0001, and answerable entirely "
            "from data already captured (no new inference, no training, no private data)."
        ),
        rationale=(
            "Orthogonal to the currently-blocked EXP-0006 pilot (ethics status NOT_ASSESSED) -- "
            "requires zero participant data, zero new training, zero device deployment, zero new "
            "human approval. Deterministic, reuses existing benchmark.metrics matching code and "
            "the existing low_conf_predictions.jsonl (conf=0.01) capture used by EXP-0001. A "
            "negative result (guardrail still violated even isolated to Person) would be a "
            "stronger, more specific finding than EXP-0001's global-only test, closing off this "
            "entire class of pure-thresholding approaches definitively."
        ),
        independent_variable=(
            "person_confidence_threshold (evaluated post-hoc from the existing conf=0.01 "
            "capture; all other hazard classes and benchmark/config.py's real conf=0.4 are "
            "unchanged; PERSON_THRESHOLDS=[0.40,0.30,0.20,0.10] pre-registered in "
            "benchmark/diagnostics/per_class_threshold_sweep.py before any per-class-isolated "
            "result was computed)"
        ),
        controls={
            "model": "yolov8m-oiv7.pt (same weights as the canonical baseline)",
            "manifest": "data/manifests/eval_manifest.jsonl (unchanged)",
            "iou_threshold": 0.7,
            "imgsz": 640,
            "other_hazard_class_threshold": 0.40,
        },
        evaluation_method=(
            "benchmark/diagnostics/per_class_threshold_sweep.py filters the existing "
            "low_conf_predictions.jsonl capture at a Person-only threshold while holding every "
            "other hazard class at conf=0.4, using the same greedy IoU>=0.5 matching as the "
            "official baseline (benchmark/metrics.py). The 0.40 grid point is an identity/"
            "control check (must exactly reproduce the official baseline). The representative "
            "candidate is selected by a rule fixed BEFORE inspecting results: among grid points "
            "satisfying the hazard-precision guardrail, pick the one with highest person.recall; "
            "if none qualify, representative=0.40 (no viable candidate). "
            "research.evaluation_policy's default hazard policy then judges that representative "
            "against the baseline."
        ),
        success_criteria={
            "primary_metric": "person.recall",
            "min_meaningful_delta": 0.03,
            "precision_floor": 0.757,
            "guardrail_metrics": ["hazard.precision", "hazard.recall", "latency.p95_ms"],
            "max_latency_regression_pct": 50.0,
            "sample_size_requirements": {"person": 100},
            "required_tests_pass": True,
        },
        risks=(
            "Read-only re-analysis of already-captured, already-approved diagnostic data (same "
            "capture EXP-0001 used) plus one new, reviewable analysis script -- no code touches "
            "production, no new inference, no training. Guardrail margin at the pre-registered "
            "grid may be thin (small-sample noise risk, same caveat threshold_sweep.py's own "
            "docstring names for low-GT-count classes) -- reported honestly either way, not "
            "cherry-picked."
        ),
        expected_outcome=(
            "Either: (a) the representative candidate clears both the precision guardrail and "
            "the minimum meaningful recall delta -- a genuine, narrow positive finding about "
            "decision-threshold policy (not a production change by itself); or (b) even isolated "
            "to Person, the guardrail is still violated at any meaningfully-improved recall "
            "point -- a stronger negative result than EXP-0001's global-only test."
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
