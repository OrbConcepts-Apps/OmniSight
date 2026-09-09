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

## Bottom line

A real, deterministic, preregistered, orthogonal, positive finding -- distinct from and not
contradicted by EXP-0001-0005 -- but with a thin guardrail margin that should be treated as
"worth a closer look" rather than "ready to ship." No production, training, or approval
action is taken or requested by this analysis.
