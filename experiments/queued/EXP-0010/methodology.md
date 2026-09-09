# EXP-0010 — Methodology

## Independent variable

person_confidence_threshold, swept finely between 0.30 and 0.40 (benchmark/diagnostics/person_threshold_sensitivity_bootstrap.py, grid fixed before any replicate was computed)

## Controls (held constant)

- `model`: yolov8m-oiv7.pt (same weights as the canonical baseline)
- `manifest`: data/manifests/eval_manifest.jsonl (unchanged, resampled at image level)
- `n_replicates`: 2000
- `seed`: 20260909
- `other_hazard_class_threshold`: 0.4

## Evaluation method

benchmark/diagnostics/person_threshold_sensitivity_bootstrap.py: 2000 image-level bootstrap replicates (same seed as EXP-0009), evaluating the entire threshold grid on each resampled image set. Per-threshold classification uses EXP-0009's exact rule. The experiment-level verdict asks whether any ROBUST threshold ALSO clears the pre-existing 0.03 minimum-meaningful-delta bar.

## Success criteria (checked by research/evaluation_policy.py)

- `compound_criterion`: PASS if >=1 threshold is ROBUST (guardrail_violation_rate<=0.05) AND its mean recall delta vs 0.40 is >=0.03. FAIL otherwise (including the case where a threshold is robust but its recall gain is below 0.03 -- robustness and meaningfulness in tension is a real, reportable negative finding, not an excuse to lower the bar).
- `note`: Both component thresholds (0.05 violation tolerance, 0.03 minimum delta) were fixed before this grid was computed, in EXP-0009 and since EXP-0001 respectively.

## Baseline compared against

`RUN-20260904-002` (see `benchmark/results/baseline/run_metadata.json`
if this is the canonical baseline, or `benchmark/results/diagnostics/` for a
diagnostic-derived baseline).
