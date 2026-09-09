# NMS/Inference-IoU Sensitivity Analysis (EXP-0011)

Real result, real (non-training) inference, zero training, zero private data, zero device
deployment. Orthogonal to the blocked EXP-0006 pilot. Direct follow-up to the now-closed
EXP-0007-0010 Person confidence-threshold branch, testing a genuinely different, independently
verified inference-time variable.

## What `iou=0.7` actually is (verified, not assumed)

`benchmark/config.py`'s own docstring and `benchmark/model.py::predict_at()` confirm
`IOU_THRESHOLD=0.7` is passed straight into ultralytics' `model.predict(..., iou=iou, ...)` --
the model's own internal NMS (non-max suppression / duplicate-box suppression) threshold,
applied inside the model before any prediction is returned. This is a completely separate
number from `MAP50_IOU=0.5`, the IoU `benchmark/metrics.py`'s `greedy_match()` uses to decide
whether a *returned* prediction counts as matching a *ground-truth* box for scoring. The two
were kept programmatically and conceptually separate throughout this experiment.

Because NMS happens inside the model call, this variable -- unlike Person confidence
threshold -- cannot be swept by re-filtering an existing capture. This experiment runs real,
new (non-training) inference over the existing frozen 380-image eval set at each grid point.

## Method

`benchmark/diagnostics/nms_iou_sweep.py`. Pre-registered grid `[0.9, 0.8, 0.7, 0.6, 0.5]`
(0.7 = production control), confidence threshold fixed at the production value (0.4)
throughout -- NMS IoU is the only variable, deliberately not jointly optimized with the
now-closed Person-threshold branch. Same guardrail (hazard.precision >= 0.757) and minimum
meaningful delta (person.recall delta >= +0.03) as every experiment since EXP-0001 -- no new
criteria invented.

## Mechanism analysis (written before any result was computed)

NMS chooses among candidate boxes the detector's backbone *already proposed* for the same
class; it can never invent a detection where the raw detector proposed nothing near a
location. This means:

- It **cannot** recover a genuine TRUE_DETECTOR_MISS ("the model saw nothing there at all") --
  there is no raw candidate box for NMS to keep in that scenario, at any IoU setting.
- It **could plausibly** reveal a *misclassified* TRUE_DETECTOR_MISS: if two real, adjacent
  people are both detected internally but their boxes overlap by >= the current 0.7 threshold,
  NMS suppresses the lower-confidence one as if it were a duplicate of the same person --
  raising IoU (less aggressive suppression) could let that second detection survive.
- Lowering IoU (more aggressive suppression) can only ever *remove* surviving boxes relative
  to production, never add one -- its only plausible benefit is fewer duplicate-detection false
  positives (a precision effect), at the risk of suppressing genuine adjacent-person detections.
- Neither direction has any plausible mechanism for LOCALIZATION_FAILURE or
  SEMANTIC_CLASS_CONFUSION -- NMS does not alter box coordinates or class labels.

## Result

| NMS IoU | hazard.precision | person.recall | person.precision | duplicate_person_pairs | TDM strict match |
|---|---|---|---|---|---|
| 0.9 | 0.7747 | 0.21122 | 0.6598 | 1 | 1/92 |
| 0.8 | 0.7983 | 0.21122 | 0.6667 | 0 | 1/92 |
| **0.7 (control)** | **0.8070** | **0.21122** | **0.6667** | **0** | **1/92** |
| 0.6 | 0.8106 | 0.21122 | 0.6667 | 0 | 1/92 |
| 0.5 | 0.8124 | 0.21122 | 0.6667 | 0 | 1/92 |

Control exactly reproduces the official baseline bit-for-bit
(hazard.precision=0.8070175438596491, person.recall=0.21122112211221122, Person TP=64/FP=32/
FN=239) -- correctness check passed.

**Person recall is literally identical (0.21122112211221122) at every single grid point** --
`person.recall_delta = +0.0000` everywhere. This is the cleanest possible negative result: no
ambiguity, no bootstrap-robustness question even arises, since there is zero variation to be
fragile or robust about. Person precision is also essentially flat (only the most permissive
setting, 0.9, shows a small drop to 0.6598, matching its 1 extra duplicate-Person-pair).
Hazard-aggregate precision *does* move materially across the grid (0.775 at 0.9 to 0.812 at
0.5) -- but this is driven entirely by the other 7 hazard classes' duplicate-suppression
behavior, not Person.

**Verdict: INCONCLUSIVE** (`person.recall` delta below the +0.03 minimum meaningful delta --
there is no candidate to even evaluate as a PASS or FAIL, since none differs from control on
the metric this experiment was designed to move).

## A methodological correction, found and reported honestly

The TRUE_DETECTOR_MISS recovery check shows `1/92` recovered **at every grid point, including
the control**. Investigated directly (not accepted at face value): this is a real Person
prediction (confidence 0.48) that overlaps GT box index 0 (image `oiv7-591eff0709a14d1c`) by
IoU=0.65 -- but the official `person_confusion_analysis.py::classify_false_negatives()`
correctly assigned that same prediction to a *neighboring* GT box (index 1, two adjacent Person
instances sharing one raw detection), per its documented cross-GT candidate-exclusion logic,
which this sweep's simplified per-GT recovery check does not implement. Because the count is
identical at every grid point *including the control*, it does not distort the grid comparison
itself (still zero net change from NMS-IoU) -- but it is not a genuine recovery, and reporting
it as one without this investigation would have been a real (if minor) error. This is exactly
the kind of self-check this lab's process discipline requires before trusting a derived metric.

## Latency

p50/p95 stay in the 18-28ms / 22-62ms range across the whole grid, matching the known baseline
(p50~18-24ms) -- NMS post-processing cost is negligible relative to backbone inference, as
expected; NMS-IoU changes have no meaningful latency effect.

## Bottom line

A genuine, defensible negative result, exactly consistent with the mechanism analysis's own
prediction: NMS IoU has no plausible channel to fix TRUE_DETECTOR_MISS (92/239 baseline Person
false negatives, the dominant failure category), and empirically, this dataset's Person
instances are not densely-packed enough (0-1 duplicate-Person-pairs across all 380 images at
any tested setting) for NMS-IoU in this bounded range to matter for Person at all. **Both
post-hoc, inference-time-only levers on the shipped checkpoint -- confidence threshold
(EXP-0007-0010) and NMS IoU (EXP-0011) -- are now exhausted without finding a robust,
meaningful Person-recall improvement.** No production, training, or approval action is taken
or requested by this analysis.
