# EXP-0006 — Methodology

## Independent variable

training_dataset_composition

## Controls (held constant)

- `confidence_threshold`: 0.4
- `evaluation_dataset`: data/manifests/eval_manifest.jsonl (frozen, 380 images)
- `imgsz`: 640
- `iou_threshold`: 0.7
- `model_architecture`: YOLOv8m

## Evaluation method

1. [PREREQUISITE, not done here] Collect+annotate the OmniSight-domain dataset per the manifest contract below. 2. [PREREQUISITE] Verify yolov8m-oiv7.pt provenance per the checkpoint-provenance section. 3. Build the CONTROL arm's resampled-OIV7 training set (equal size to the OmniSight-domain set, drawn from existing OIV7 training data, excluding any image in the frozen 380-image eval manifest). 4. For each of the 3 pre-registered seeds (42, 43, 44): train CONTROL and INTERVENTION arms independently, identical procedure/hyperparameters (see training_config), differing only in training data. 5. Evaluate every trained checkpoint against the frozen eval manifest at conf=0.4/iou=0.7, computing person.recall, hazard.precision, and true_detector_miss_recovery_count (per-seed criteria). 6. Classify each seed via classify_seed_verdict() and the aggregate via classify_aggregate_verdict() (research/preregistration.py) -- both deterministic, pre-registered, no post-hoc discretion. 7. Report every seed's result (COMPLETED or FAILED), never a favorable-subset selection.

## Success criteria (checked by research/evaluation_policy.py)

- `guardrails`: ['hazard.precision >= 0.757']
- `min_meaningful_delta`: 0.03
- `primary_metric`: person.recall
- `recovery_metric_definition`: {'denominator': 92, 'direction': 'must exceed the reference by a pre-registered absolute margin, never a relative-percentage-of-a-percentage', 'interpretation_chosen': "C -- incremental absolute count beyond the prior best candidate's recovery (reports/phase_i/CANDIDATE_0003_EXP0006_ELIGIBILITY.md section 1's three offered models); NOT interpretation A (bare fraction of 92) alone and NOT interpretation B (relative-improvement percentage), because the baseline's own recovery rate is tautologically 0/92 and a 'percent more than baseline' framing is therefore vacuous.", 'name': 'true_detector_miss_recovery_count', 'numerator': 'count of the 92 fixed baseline TRUE_DETECTOR_MISS Person cases (established by EXP-0005) detected by the INTERVENTION model at conf=0.4', 'reference_condition': 'candidate C (YOLO11m/COCO, EXP-0005/MEM-0015), which recovered 17 of the 92 cases', 'threshold': 'recovery_count >= 22 (reference 17 + margin 5)'}
- `seed_plan`: {'aggregate_verdict_function': 'research.preregistration.classify_aggregate_verdict', 'no_cherry_picking': "all 3 seeds' results reported regardless of outcome; aggregation requires a MAJORITY (>=2/3), never a single favorable seed", 'per_seed_verdict_function': 'research.preregistration.classify_seed_verdict', 'seeds': [42, 43, 44]}

## Baseline compared against

`RUN-20260904-002` (see `benchmark/results/baseline/run_metadata.json`
if this is the canonical baseline, or `benchmark/results/diagnostics/` for a
diagnostic-derived baseline).
