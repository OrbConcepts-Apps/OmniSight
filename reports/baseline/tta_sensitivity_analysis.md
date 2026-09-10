# Test-Time Augmentation (TTA) Sensitivity Analysis (EXP-0012)

Real result, real (non-training) inference, zero training, zero private data, zero device
deployment. Orthogonal to the blocked EXP-0006 pilot. Third and final test in the
inference-time-lever sequence (confidence threshold, NMS IoU, TTA).

## Mechanism (why this is different from EXP-0007-0011)

Confidence threshold and NMS IoU can only re-select or re-threshold among candidate boxes a
**single** forward pass already proposed. TTA (`ultralytics` `augment=True`) runs several
transformed views of the same image (flip/scale) through the **same** model and merges
candidates across views via NMS -- a genuinely different view can propose a detection the
original pass did not, giving TTA a real, previously-untested channel toward recovering a
TRUE_DETECTOR_MISS case. This was a single ON/OFF test (not a grid) -- there is no value to
cherry-pick.

## Method

`benchmark/diagnostics/tta_sweep.py`. `augment=False` (control) vs `augment=True` (candidate),
confidence (0.4) and NMS IoU (0.7) fixed at production values throughout. Same guardrails as
every experiment since EXP-0001 (hazard.precision >= 0.757, person.recall delta >= +0.03),
plus this lab's existing latency guardrail (p95 regression <= 50%) -- no new criteria invented.

## Result

| Condition | hazard.precision | person.recall | person.precision | p95 latency (ms) | TDM recovered |
|---|---|---|---|---|---|
| augment=False (control) | 0.8070 | 0.2112 | 0.6667 | 63.0 | 1/92 |
| augment=True (candidate) | 0.7438 | 0.2343 | 0.5868 | 83.1 | 1/92 |

Control exactly reproduces the official baseline -- correctness check passed.

- **person.recall delta = +0.0231** -- below the +0.03 minimum meaningful delta.
- **hazard.precision hard-violates the guardrail**: 0.7438 vs the 0.757 floor, margin -0.0132
  (outside the 0.01 noise-margin tolerance) -- this is a genuine, non-noisy failure, not a
  borderline case.
- **Latency regression = +32%** -- within the 50% guardrail (lower than the 2-3x initially
  estimated; this model's TTA implementation is apparently closer to 1.3x cost).
- **TRUE_DETECTOR_MISS recovery unchanged at 1/92** (the same known artifact case documented in
  `nms_iou_sensitivity_analysis.md`, not a genuine recovery) -- TTA's modest recall gain is
  **not** coming from the specific 92 cases this mechanism was hypothesized to help.

**Verdict: FAIL** (hard guardrail violation).

## Interpretation

TTA trades away *more* precision for *less* recall gain than the already-closed
Person-threshold=0.30 candidate (EXP-0008: +0.066 recall for -0.092 precision; TTA: +0.023
recall for -0.063 precision) -- a strictly worse tradeoff on this dataset, not a close call.
The recall gain that does exist appears to come from elsewhere in the failure taxonomy
(LOW_CONFIDENCE_PERSON or LOCALIZATION_FAILURE cases nudged over the line by a merged,
higher-confidence box), not from the dominant TRUE_DETECTOR_MISS category this mechanism was
specifically hoped to address.

## Synthesis: three inference-time levers, three negative results

Person confidence threshold (EXP-0007-0010, closed: fragile), NMS IoU (EXP-0011, closed: zero
effect), and TTA (EXP-0012, closed: worse tradeoff) have now all been tested with an explicit,
falsifiable mechanism analysis, on the same shipped checkpoint, using only the existing frozen
380-image eval set. None can fix TRUE_DETECTOR_MISS (92/239 baseline Person false negatives,
38.5%, the dominant failure mode) -- none of them can make the model recognize something it
does not represent well enough at any decision-time setting. This is a representational/
training-data limitation, not a decision-policy one, and it materially strengthens (rather than
merely coexists with) the case that EXP-0006's original hypothesis -- domain-matched training
data -- is the most promising remaining lever, still blocked on the same human-only approvals
(ethics status, then training/private-data approval), unaffected by any of this work.

No production, training, or approval action is taken or requested by this analysis.
