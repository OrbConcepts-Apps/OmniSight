# EXP-0013 — Hypothesis

**Family**: small_object
**Validation requirement**: OFFLINE_SIMULATABLE
**Parent experiment**: (none)

## Hypothesis

Tiling (cropping each image into 4 overlapping 2x2-grid regions plus the full frame, running inference on each at the fixed production imgsz=640/conf=0.4/iou=0.7, then merging via cross-tile NMS) recovers meaningful Person recall (>= +0.03 vs the frozen single-pass baseline) while keeping hazard-aggregate precision at or above the standard guardrail (>= 0.757), because it increases the EFFECTIVE detector-input scale of small/distant Person instances without changing the network's own input resolution (unlike EXP-0002's failed global-resize approach).

## Motivation

68 of the 92 baseline TRUE_DETECTOR_MISS cases are 'small' (<2% image area, person_confusion_analysis.json). Three prior inference-time levers (EXP-0007-0010 Person confidence threshold, EXP-0011 NMS IoU, EXP-0012 TTA) are all closed as negative and none had a plausible mechanism to change effective object scale. Tiling is mechanistically distinct and directly targets this specific subset -- but is NOT assumed to prove the remaining problem is a training-data limitation; it is an unresolved alternative mechanism, tested here before drawing that conclusion.

## Rationale

Uses only the existing frozen 380-image public eval set and the existing shipped checkpoint. Not a hyperparameter search -- ONE preregistered primary configuration (PRIMARY_CONFIG above), chosen before any tiling-specific result was computed. No training, no private data, no device deployment, no production change, no new approval. Orthogonal to and unaffected by the blocked EXP-0006 pilot.

## Expected outcome

Possible outcomes, per this task's own framing: (a) positive -- meaningful recovery with robust precision; (b) detection-positive but impractical -- recovery at unacceptable Windows-runtime cost (evidence that effective scale matters even if not deployable); (c) negative -- no meaningful recovery, materially strengthening the representation/training-data hypothesis; (d) precision failure -- recall gain at an unacceptable false-positive cost. All four are informative and none is assumed here.

## Risks

Runs real (non-training) GPU inference (4 new tile passes per image, full-image pass reused) -- no training, no private data, no production/config changes (uses predict_array_at(), never predict(); benchmark/config.py never modified). New, correctness-sensitive spatial logic (coordinate remapping, cross-tile merge) is covered by 19 passing synthetic-geometry tests before this benchmark runs.
