# EXP-0011 — Methodology

## Independent variable

nms_inference_iou (benchmark/model.py::predict_at()'s `iou` parameter, passed to ultralytics model.predict()) -- grid [0.9,0.8,0.7,0.6,0.5], 0.7=production control, fixed BEFORE any grid-point result was computed. Confidence threshold fixed at 0.4 (production) throughout -- NOT jointly optimized with the now-closed EXP-0007-0010 Person-threshold branch.

## Controls (held constant)

- `model`: yolov8m-oiv7.pt (same weights as the canonical baseline)
- `manifest`: data/manifests/eval_manifest.jsonl (unchanged)
- `imgsz`: 640
- `confidence_threshold`: 0.4
- `evaluation_matching_iou`: 0.5

## Evaluation method

benchmark/diagnostics/nms_iou_sweep.py runs real (non-training) inference over all 380 eval images at each grid IoU value (conf fixed at 0.4), computing hazard/Person metrics via the same benchmark.metrics matching code as every other experiment in this lab. The iou=0.7 control point must exactly reproduce the official baseline (identity/correctness check). TRUE_DETECTOR_MISS recovery is checked directly against the 92 known baseline cases (person_confusion_analysis.json) -- did any grid point's own Person predictions newly match (IoU>=0.5) one of those specific 92 GT boxes. A duplicate-Person-detection proxy (mutual IoU>=0.5 among a single image's surviving Person boxes) is also reported.

## Success criteria (checked by research/evaluation_policy.py)

- `primary_metric`: person.recall
- `min_meaningful_delta`: 0.03
- `precision_floor`: 0.757
- `guardrail_metrics`: ['hazard.precision']
- `sample_size_requirements`: {'person': 100}
- `note`: Both thresholds (0.757, 0.03) are UNCHANGED from EXP-0001 onward -- no new significance criterion is introduced for this experiment. If any grid point passes on the point estimate, the SAME image-level bootstrap robustness check that exposed EXP-0008's fragility (EXP-0009's methodology) is required before treating it as a credible positive finding -- a point-estimate PASS alone is explicitly NOT sufficient here, matching the precedent this lab just set.

## Baseline compared against

`RUN-20260904-002` (see `benchmark/results/baseline/run_metadata.json`
if this is the canonical baseline, or `benchmark/results/diagnostics/` for a
diagnostic-derived baseline).
