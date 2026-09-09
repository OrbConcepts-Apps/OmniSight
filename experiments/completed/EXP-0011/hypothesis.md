# EXP-0011 — Hypothesis

**Family**: threshold_postprocessing
**Validation requirement**: OFFLINE_SIMULATABLE
**Parent experiment**: (none)

## Hypothesis

Changing the model's internal NMS IoU threshold (production=0.7) -- confidence threshold held fixed at the production value (0.4) -- can recover meaningful Person recall (>= +0.03 vs the iou=0.7 control) while keeping hazard-aggregate precision at or above the standard guardrail (>= 0.757). Mechanism: NMS chooses among ALREADY-PROPOSED overlapping candidate boxes of the same class; it cannot invent a detection where the raw detector proposed nothing. It can only plausibly help by letting a currently-suppressed second detection near an ADJACENT person survive (raising IoU, less aggressive suppression) -- it has no plausible channel to recover a genuine 'detector saw nothing' TRUE_DETECTOR_MISS case, and lowering IoU (more aggressive suppression) can only remove surviving boxes, never add one.

## Motivation

The Person confidence-threshold line of inquiry (EXP-0007-0010) is closed: no tested threshold is both robust and meaningful. NMS IoU is a genuinely distinct, previously untested independent variable (verified in code, not assumed, to be the model's own duplicate-suppression threshold, separate from the evaluation-matching IoU). This experiment tests it in isolation, with an explicit, falsifiable mechanism analysis stated before any result is computed -- including an honest statement of what this mechanism CANNOT plausibly fix (TRUE_DETECTOR_MISS misses with no raw candidate box at all).

## Rationale

Uses only the existing frozen 380-image public eval set and the existing shipped checkpoint. Runs new, non-training inference (explicitly authorized for this experiment) -- unlike EXP-0007-0010, NMS IoU cannot be swept by re-filtering an existing capture, since NMS happens inside the model call before any prediction is returned. No private data, no participant data, no device deployment, no production change, no new approval. Orthogonal to and unaffected by the blocked EXP-0006 pilot.

## Expected outcome

Either: (a) no grid point clears both the guardrail and the minimum meaningful delta -- consistent with the mechanism analysis's prediction that NMS IoU has no plausible channel to fix the dominant TRUE_DETECTOR_MISS failure mode, closing this branch too as a genuine negative result; or (b) a grid point passes the point estimate -- in which case its robustness must be checked (per EXP-0009's precedent) before being treated as credible, and the TRUE_DETECTOR_MISS recovery count specifically should be inspected to see whether the adjacent-person-suppression mechanism is actually responsible, or something else is.

## Risks

Runs real (non-training) GPU inference over the existing eval set -- no training, no private data, no production/config changes (uses predict_at(), never predict(); benchmark/config.py's real IOU_THRESHOLD=0.7 is never modified). The only risk is scientific: an honest negative result must be reported as such, not reframed.
