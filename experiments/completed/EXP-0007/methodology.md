# EXP-0007 — Methodology

## Independent variable

person_confidence_threshold (evaluated post-hoc from the existing conf=0.01 capture; all other hazard classes and benchmark/config.py's real conf=0.4 are unchanged; PERSON_THRESHOLDS=[0.40,0.30,0.20,0.10] pre-registered in benchmark/diagnostics/per_class_threshold_sweep.py before any per-class-isolated result was computed)

## Controls (held constant)

- `model`: yolov8m-oiv7.pt (same weights as the canonical baseline)
- `manifest`: data/manifests/eval_manifest.jsonl (unchanged)
- `iou_threshold`: 0.7
- `imgsz`: 640
- `other_hazard_class_threshold`: 0.4

## Evaluation method

benchmark/diagnostics/per_class_threshold_sweep.py filters the existing low_conf_predictions.jsonl capture at a Person-only threshold while holding every other hazard class at conf=0.4, using the same greedy IoU>=0.5 matching as the official baseline (benchmark/metrics.py). The 0.40 grid point is an identity/control check (must exactly reproduce the official baseline). The representative candidate is selected by a rule fixed BEFORE inspecting results: among grid points satisfying the hazard-precision guardrail, pick the one with highest person.recall; if none qualify, representative=0.40 (no viable candidate). research.evaluation_policy's default hazard policy then judges that representative against the baseline.

## Success criteria (checked by research/evaluation_policy.py)

- `primary_metric`: person.recall
- `min_meaningful_delta`: 0.03
- `precision_floor`: 0.757
- `guardrail_metrics`: ['hazard.precision', 'hazard.recall', 'latency.p95_ms']
- `max_latency_regression_pct`: 50.0
- `sample_size_requirements`: {'person': 100}
- `required_tests_pass`: True

## Baseline compared against

`RUN-20260904-002` (see `benchmark/results/baseline/run_metadata.json`
if this is the canonical baseline, or `benchmark/results/diagnostics/` for a
diagnostic-derived baseline).
