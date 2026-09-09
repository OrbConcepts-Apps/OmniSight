# EXP-0009 — Methodology

## Independent variable

none in the traditional sense -- this experiment resamples the EXISTING evidence (low_conf_predictions.jsonl, eval_manifest.jsonl) at the image level and recomputes metrics from scratch per replicate; the model/weights/threshold under test (person_threshold=0.30 vs baseline 0.40) is unchanged from EXP-0008.

## Controls (held constant)

- `model`: yolov8m-oiv7.pt (same weights as the canonical baseline)
- `manifest`: data/manifests/eval_manifest.jsonl (unchanged, resampled at image level only)
- `iou_threshold`: 0.7
- `imgsz`: 640
- `person_threshold_tested`: 0.3
- `n_replicates`: 2000
- `seed`: 20260909

## Evaluation method

benchmark/diagnostics/person_threshold_bootstrap.py draws 2000 image-level bootstrap replicates (fixed seed=20260909, numpy.random.default_rng) of the 380 real eval images, recomputing hazard/Person metrics from scratch on each resampled set (never bootstrapping already-aggregated point estimates); duplicate image occurrences within one replicate are relabeled with synthetic per-occurrence sample_ids so they do not compete for the same GT boxes. Reports percentile (2.5/97.5) CIs and the fraction of replicates violating the 0.757 hazard-precision guardrail.

## Success criteria (checked by research/evaluation_policy.py)

- `pre_registered_robustness_criterion`: ROBUST (PASS) if guardrail_violation_rate<=0.05 AND 2.5th-percentile(delta person.recall)>0; FRAGILE (FAIL) if guardrail_violation_rate>0.05; else INCONCLUSIVE. Fixed in benchmark/diagnostics/person_threshold_bootstrap.py before any bootstrap replicate was computed or inspected.
- `note`: Not the same shape as this lab's other experiments' guardrail-vs-baseline point-estimate criteria -- this is a distributional robustness check, and is explicitly reported as such, never as a production readiness claim.

## Baseline compared against

`RUN-20260904-002` (see `benchmark/results/baseline/run_metadata.json`
if this is the canonical baseline, or `benchmark/results/diagnostics/` for a
diagnostic-derived baseline).
