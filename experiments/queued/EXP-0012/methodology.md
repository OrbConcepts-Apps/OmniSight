# EXP-0012 — Methodology

## Independent variable

tta_augment (benchmark/model.py::predict_at()'s new `augment` parameter, passed to ultralytics model.predict()) -- binary on/off, confidence (0.4) and NMS IoU (0.7) fixed at production values throughout, not jointly optimized with either closed branch.

## Controls (held constant)

- `model`: yolov8m-oiv7.pt (same weights as the canonical baseline)
- `manifest`: data/manifests/eval_manifest.jsonl (unchanged)
- `imgsz`: 640
- `confidence_threshold`: 0.4
- `nms_iou`: 0.7

## Evaluation method

benchmark/diagnostics/tta_sweep.py runs real (non-training) inference over all 380 eval images under augment=False (control, must exactly reproduce the official baseline) and augment=True (candidate), computing hazard/Person metrics via the same benchmark.metrics matching code as every other experiment in this lab, plus p50/p95 latency and the same TRUE_DETECTOR_MISS recovery check as EXP-0011 against the 92 known baseline cases.

## Success criteria (checked by research/evaluation_policy.py)

- `primary_metric`: person.recall
- `min_meaningful_delta`: 0.03
- `precision_floor`: 0.757
- `guardrail_metrics`: ['hazard.precision', 'latency.p95_ms']
- `max_latency_regression_pct`: 50.0
- `sample_size_requirements`: {'person': 100}
- `note`: All three thresholds (0.757, 0.03, 50%) are UNCHANGED from this lab's existing policy (research/evaluation_policy.py::default_hazard_policy) -- no new significance criterion is introduced. If the point estimate passes, the same image-level bootstrap robustness check that exposed EXP-0008's fragility (EXP-0009's methodology) is required before treating it as credible.

## Baseline compared against

`RUN-20260904-002` (see `benchmark/results/baseline/run_metadata.json`
if this is the canonical baseline, or `benchmark/results/diagnostics/` for a
diagnostic-derived baseline).
