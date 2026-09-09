# Per-Class (Person-Only) Confidence Threshold Policy (EXP-0007/EXP-0008)

Real result, deterministic, zero new inference, zero training, zero private data. Orthogonal
to the blocked EXP-0006 pilot (`ethics_or_institutional_review_status=NOT_ASSESSED`) --
this line of work does not touch it and is not affected by it.

## Question

EXP-0001 showed a GLOBAL confidence-threshold drop (all 8 hazard classes lowered uniformly)
cannot recover Person recall without collapsing hazard-aggregate precision. Does ISOLATING
the threshold drop to Person alone (every other hazard class fixed at the production 0.4
cutoff) avoid that collapse?

## Method

`benchmark/diagnostics/per_class_threshold_sweep.py`, reusing the same conf=0.01 capture
(`benchmark/results/diagnostics/low_conf_predictions.jsonl`) and matching code
(`benchmark/metrics.py`) EXP-0001 already used -- no new model inference. Grid
`PERSON_THRESHOLDS=[0.40,0.30,0.20,0.10]` and the selection rule (highest person.recall
among grid points clearing the hazard-precision guardrail) were fixed in the script before
any per-class-isolated number was computed. EXP-0007 was registered and run first; its
branch run was REJECTED for a purely structural reason (pytest invariants left over from the
EXP-0006 era hardcoded "no experiment beyond EXP-0006 exists") -- not a scientific rejection.
Those invariants were corrected and the identical design was re-registered as EXP-0008,
`parent_experiment_id=EXP-0007`, which produced the real verdict below.

## Result

| person_threshold | hazard.precision | hazard.recall | person.recall | person.precision |
|---|---|---|---|---|
| 0.40 (control, = production) | 0.807 | 0.480 | 0.211 | 0.667 |
| **0.30 (representative)** | **0.767** | **0.507** | **0.277** | **0.575** |
| 0.20 | 0.717 | 0.533 | 0.343 | 0.498 |
| 0.10 | 0.641 | 0.564 | 0.422 | 0.408 |

Guardrail floor: `hazard.precision >= 0.757` (baseline 0.807 - 0.05). Only 0.40 (control)
and 0.30 clear it; 0.20 and 0.10 do not.

**EXP-0008 verdict: PASS.** At person_threshold=0.30: person.recall improves +0.0660 (above
the +0.03 minimum meaningful delta) and hazard.precision holds at 0.767, clearing the
guardrail -- but by a margin of only **+0.0098** (0.767 vs 0.757). This is thin, not a wide
margin like EXP-0002/0003's clear violations. person.precision itself drops materially
(0.667 -> 0.575) -- the real cost of the recall gain, paid entirely within the Person class
rather than smeared across the hazard aggregate the way EXP-0001's global drop was.

## What this is NOT

- **Not** a production change. `benchmark/config.py`'s real conf=0.4 is unchanged. Shipping
  this would require a separate, explicit human decision plus the usual production-change
  approvals (`production_swift_modification_approved` etc.) -- not evaluated or requested here.
- **Not** a single-run, unreplicated claim with no sample-size context: Person GT=303 (the
  largest, most trustworthy hazard class in this eval set), same 380-image manifest as every
  other experiment in this lab.
- **Not** independent of EXP-0001's finding -- it refines it (the global test's guardrail
  violation was driven disproportionately by non-Person classes' low-confidence false
  positives, as this result's precision-vs-recall shape at 0.30 vs 0.20/0.10 shows), rather
  than contradicting it.

## Caveats (stated plainly, not glossed over)

1. **Guardrail margin is thin (+0.0098).** A 380-image eval manifest carries real sampling
   noise; this margin would not survive a materially unlucky re-sample. Before treating this
   as a robust finding (let alone a shippable one), it should be checked against a bootstrap
   CI over images (see the reproducibility-gap note in `research/memory/open_questions.md`)
   or a held-out/second eval set.
2. **Single evaluation, no seed variation** -- this is a deterministic re-scoring of one fixed
   model's one fixed inference capture, not a multi-seed training comparison; "seed" isn't a
   meaningful axis here the way it is for EXP-0006.
3. **Person-precision cost is real.** Person precision drops 0.667 -> 0.575, i.e. the
   false-positive share of Person detections rises from about 1-in-3 (33.3%) at the
   production threshold to about 2-in-5 (42.5%) at threshold 0.30. Whether that tradeoff is
   acceptable for an assistive-vision hazard alert is a product/UX judgment this report does
   not make.
4. This finding says nothing about EXP-0006's domain-matched-training-data question -- it is
   a decision-threshold policy result, not a training-data or model-architecture result.

## Addendum: robustness and sensitivity analysis (EXP-0009, EXP-0010)

Caveat 1 above was checked. **The result is FRAGILE, not robust.**

**EXP-0009** (image-level bootstrap, 2000 replicates, fixed seed=20260909, resampling the
frozen 380-image manifest, recomputing metrics from scratch per replicate -- never
bootstrapping already-aggregated point estimates): at person_threshold=0.30, hazard-aggregate
precision falls below the 0.757 guardrail in **36.9% of resamples** (mean 0.766, 95% CI
[0.704, 0.823] -- the guardrail floor sits well inside this interval). The Person-recall
improvement itself IS robust (95% CI [0.0365, 0.1026], 99.4% of resamples clear +0.03) -- only
the precision side of EXP-0008's result is fragile.

**EXP-0010** (same bootstrap methodology, finer grid [0.40,0.38,0.36,0.34,0.32,0.30] in one
pass per replicate): only **person_threshold=0.38** is robust on the guardrail
(violation_rate=0.042), but its mean recall gain is only **+0.0199** -- below this lab's own
+0.03 minimum-meaningful-delta bar (used since EXP-0001). Every threshold with a
point-estimate recall gain >=0.03 (0.34, 0.32, 0.30) is fragile (violation rates 0.202, 0.267,
0.369 respectively, rising monotonically as the threshold drops). **No threshold in the tested
grid is both robust and meaningful.**

### Three things, kept explicitly distinct (per this analysis's own governing instructions)

1. **Original EXP-0008 deterministic PASS**: unchanged, still on record exactly as computed --
   person_threshold=0.30 clears the guardrail and the minimum-delta bar on the single frozen
   380-image point estimate. This record is not retroactively altered.
2. **Robustness of that PASS**: FRAGILE (EXP-0009), and the follow-up sensitivity sweep
   (EXP-0010) confirms this isn't a threshold-tuning fix -- the entire testable region between
   the fragile candidate and the production baseline fails to clear both bars at once.
3. **Evidence sufficient for a production recommendation**: NO. This line of inquiry
   (per-class Person confidence-threshold policy, EXP-0007/8/9/10) is closed under current
   evidence as a genuine, defensible negative result -- not because EXP-0008's arithmetic was
   wrong, but because its guardrail margin does not survive the sampling noise inherent in a
   380-image eval set. No production change is recommended or should be inferred from EXP-0008
   alone.

## Bottom line

EXP-0008 was a real, deterministic, preregistered, orthogonal PASS -- but a post-hoc
robustness check (explicitly labeled as such, since EXP-0008 preregistered no uncertainty
criterion) shows that PASS does not survive image-level bootstrap resampling, and a
sensitivity sweep confirms no nearby threshold does either. The honest final interpretation of
this whole line of inquiry is a **negative result**: no per-class confidence-threshold policy
in the tested range is both robust and meaningful given the current 380-image eval set. No
production, training, or approval action is taken or requested by any part of this analysis.
