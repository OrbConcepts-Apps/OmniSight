# EXP-0006 — Hypothesis

**Family**: training_data
**Validation requirement**: REQUIRES_IPHONE
**Parent experiment**: (none)

## Hypothesis

Fine-tuning YOLOv8m on an OmniSight-domain dataset (chest-height, indoor/outdoor, low-light, motion-blurred accessibility-scenario frames) recovers more of the 92 fixed baseline TRUE_DETECTOR_MISS Person cases than a MATCHED control arm trained identically except for using resampled existing Open Images V7 data instead -- specifically, at least 22 of the 92 cases (5 more than candidate C's 17, EXP-0005/MEM-0015) -- while keeping hazard.precision >= 0.757, across a majority of 3 pre-registered training seeds.

## Motivation

Corrects CANDIDATE-0003's hypothesis/metric wording mismatch identified in reports/phase_i/CANDIDATE_0003_EXP0006_ELIGIBILITY.md section 1: the prior wording ('25% more than baseline') was vacuous since the baseline recovers 0/92 by definition (TRUE_DETECTOR_MISS is DEFINED as zero baseline detections); the actual binding comparison was always against candidate C's 17/92, restated here explicitly as an absolute-count threshold (interpretation C from that audit's three offered models), never as a percentage-of-a-percentage.

## Rationale

Corrects CANDIDATE-0003's hypothesis/metric ambiguity and causal-control gap (reports/phase_i/CANDIDATE_0003_EXP0006_ELIGIBILITY.md); see materially_new_rationale in research/experiment_specs/EXP-0006.json for the full relationship to EXP-0001-0005 and CANDIDATE-0001/0002.

## Expected outcome

Does OmniSight-domain training data, specifically (isolated from the mere fact of additional training), recover more TRUE_DETECTOR_MISS Person cases than a matched control receiving equivalent additional training on existing Open Images V7 data, without violating the hazard-precision guardrail, and is this effect consistent across multiple training seeds?

## Risks

Checkpoint provenance unresolved; dataset does not yet exist; epoch/resource estimates uncalibrated; no real training Runner exists yet (reports/phase_j/PHASE_J_SAFETY_AUDIT.md). See execution_status=BLOCKED note for the complete, current blocker list.
