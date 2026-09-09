# EXP-0010 — Hypothesis

**Family**: threshold_postprocessing
**Validation requirement**: OFFLINE_SIMULATABLE
**Parent experiment**: EXP-0009

## Hypothesis

At least one person_threshold in the finer grid [0.40,0.38,0.36,0.34,0.32,0.30] is simultaneously (a) ROBUST under image-level bootstrap resampling -- guardrail_violation_rate<=0.05, per EXP-0009's criterion -- AND (b) achieves a mean recall delta vs the production baseline (0.40) of at least the lab's established minimum meaningful delta (0.03, used by every experiment since EXP-0001).

## Motivation

EXP-0009 showed person_threshold=0.30's guardrail margin is FRAGILE (36.9% bootstrap violation rate). The coarse original grid ([0.40,0.30,0.20,0.10]) cannot distinguish '0.30 was simply the best of a few bad options' from 'a nearby, more conservative threshold is actually safe.' This experiment answers that directly, as the cheapest available non-training controlled test, per this lab's own explicit next-step guidance after a fragile positive finding: check sensitivity around the chosen threshold before treating anything as a credible positive result.

## Rationale

Uses only the existing frozen 380-image eval set (no private data, no new inference, no training, no device deployment, no new human approval). Reuses EXP-0009's exact image-level bootstrap methodology and fixed seed, extended to a finer grid computed in one pass per replicate (statistically cleaner than re-resampling per threshold). Orthogonal to and unaffected by the blocked EXP-0006 pilot.

## Expected outcome

Either: (a) a genuinely robust AND meaningful threshold exists -- the closest thing to a credible positive finding this line of inquiry could produce, still requiring independent validation before any production consideration; or (b) no threshold clears both bars -- the per-class threshold-policy line of inquiry is closed as a viable production lever under current evidence, a genuine negative result.

## Risks

Read-only, deterministic re-analysis of already-captured data -- no code touches production, no new inference, no training.
