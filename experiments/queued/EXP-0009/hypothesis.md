# EXP-0009 — Hypothesis

**Family**: threshold_postprocessing
**Validation requirement**: OFFLINE_SIMULATABLE
**Parent experiment**: EXP-0008

## Hypothesis

EXP-0008's PASS (person_threshold=0.30 recovers person.recall +0.066 while hazard -aggregate precision clears the 0.757 guardrail by +0.0098) is ROBUST under image-level bootstrap resampling of the same frozen 380-image eval set: the guardrail is held in at least 95% of resamples, and the recall-improvement direction is robust at the 95% level (2.5th percentile of the resampled recall delta is > 0).

## Motivation

EXP-0008's PASS rested on a guardrail margin of only +0.0098 -- thin relative to sampling noise on a 380-image manifest. Before treating that PASS as evidence of anything beyond 'clears the guardrail on this one particular sample', its robustness must be checked. This is a post-hoc robustness analysis, explicitly labeled as such -- EXP-0008 itself preregistered no uncertainty criterion, and none is invented for it retroactively; this experiment's OWN criterion (below) was fixed before any bootstrap replicate was computed.

## Rationale

Uses only the existing frozen 380-image eval set (no private data, no new inference, no training, no device deployment, no new human approval). Image-level (not box-level) resampling respects the real dependence structure -- boxes within one image are not independent draws. Orthogonal to and unaffected by the blocked EXP-0006 pilot.

## Expected outcome

Either: (a) ROBUST -- EXP-0008's finding survives resampling, and the next cheapest controlled test (e.g. sensitivity around threshold=0.30, or an independent eval set) becomes the natural next step before any production consideration; or (b) FRAGILE/INCONCLUSIVE -- the guardrail margin is a sampling artifact, and EXP-0008's PASS must be reinterpreted as fragile, not a credible positive finding, without altering EXP-0008's own preserved deterministic record.

## Risks

Read-only, deterministic re-analysis of already-captured, already-approved diagnostic data -- no code touches production, no new inference, no training. The only 'risk' is scientific: this may show EXP-0008's PASS does not survive resampling, which must be reported honestly regardless of outcome (no new significance threshold invented after seeing the result).
