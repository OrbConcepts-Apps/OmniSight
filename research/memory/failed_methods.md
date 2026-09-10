# Failed Methods

> Read `research/memory/README.md` first.

## Already established (Phase B.5, treated as prior art here, not a new experiment)

- **Lowering the confidence threshold to fix Person recall.** conf=0.05
  roughly doubles Person recall (0.211->0.479) but collapses precision
  (0.667->0.312) — see `research/memory/known_failures.md`. This is not a
  fresh negative result from this lab's own orchestrator; it's Phase B.5
  diagnostic evidence being carried forward so nobody re-proposes "just lower
  the threshold" as if it were a novel idea. EXP-0001 formally confirms this
  via the orchestrator pipeline.

This file is otherwise empty at Phase C seed time — updated by
`research/orchestrator.py`'s `experiment()` command whenever a verdict is
FAILED or REJECTED, with enough detail (independent variable, what broke,
which guardrail) that the same mistake isn't repeated.

## EXP-0002 (2026-09-04T23:21:45.894618+00:00)

- Family: small_object
- Status: FAILED
- Hypothesis: Increased inference-time input resolution (640->960 or 640->1280) meaningfully improves Person recall, at some measurable latency cost.
- Reasons: guardrail 'hazard.recall' violated: 0.4491 does not satisfy gte 0.4604 (hazard recall must not drop more than 0.02 below baseline)

## EXP-0003 (2026-09-04T23:53:42.838654+00:00)

- Family: class_confusion
- Status: FAILED
- Hypothesis: A meaningful fraction of Person recall loss at the baseline conf=0.4 operating point is attributable to the detector correctly localizing a human-shaped region but labeling it with a semantically related non-Person class (Man/Woman/Boy/Girl/Human body/person-subparts), NOT to the detector failing to notice a person at all. Every Person ground-truth false negative is classified into exactly one of 5 mutually-exclusive primary categories (so percentages sum to 100%): (A) TRUE_DETECTOR_MISS -- no spatially relevant human-like prediction of any class exists near the GT box at any non-noise confidence; (B) LOW_CONFIDENCE_PERSON -- a Person prediction exists at sufficient IoU (>=0.5) but below the 0.4 production confidence threshold; (C) SEMANTIC_CLASS_CONFUSION -- a different human-related class (Man/Woman/Boy/Girl/Human body/a person subpart) is predicted at sufficient IoU (>=0.5) and at/above a diagnostic confidence floor (0.4, matching the production threshold); (D) LOCALIZATION_FAILURE -- a human-related prediction exists nearby but its IoU with the GT box is below the 0.5 match threshold (though still >=0.3, the 'spatially associated' floor); (E) DUPLICATE_MULTI_LABEL is reported as a secondary/overlapping flag (not a primary bucket) when >=2 distinct human-related classes are predicted over the same GT box.
- Reasons: guardrail 'hazard.precision' violated: 0.6785 does not satisfy gte 0.7570 (hazard precision must not drop more than 0.05 below baseline)

## EXP-0007 (2026-09-09T22:55:51.643446+00:00)

- Family: threshold_postprocessing
- Execution status: COMPLETED
- Research verdict: REJECTED
- Hypothesis: Isolating the confidence-threshold reduction to the Person class ONLY (holding every other hazard class fixed at the production conf=0.4 cutoff) recovers meaningful Person recall (>= +0.03 over baseline) while keeping hazard-aggregate precision within the standard guardrail (>= baseline - 0.05), unlike EXP-0001's GLOBAL uniform threshold drop, which collapsed hazard precision well past that guardrail. Mechanism: non-Person hazard classes (esp. Car, precision=0.295 at conf=0.05 per threshold_sweep.json) contribute disproportionate low-confidence false positives to the AGGREGATE hazard-precision guardrail under a global drop; isolating the threshold change to Person alone should avoid that collateral damage.
- Reasons: test_failure: pytest failed on the experiment branch

## EXP-0009 (2026-09-09T23:22:12.123786+00:00)

- Family: threshold_postprocessing
- Execution status: COMPLETED
- Research verdict: FAIL
- Hypothesis: EXP-0008's PASS (person_threshold=0.30 recovers person.recall +0.066 while hazard -aggregate precision clears the 0.757 guardrail by +0.0098) is ROBUST under image-level bootstrap resampling of the same frozen 380-image eval set: the guardrail is held in at least 95% of resamples, and the recall-improvement direction is robust at the 95% level (2.5th percentile of the resampled recall delta is > 0).
- Reasons: FRAGILE: bootstrap guardrail_violation_rate=0.369 exceeds the pre-registered tolerance (0.05) -- the hazard-precision guardrail is not reliably held under image-level resampling.

## EXP-0010 (2026-09-09T23:29:02.583776+00:00)

- Family: threshold_postprocessing
- Execution status: COMPLETED
- Research verdict: FAIL
- Hypothesis: At least one person_threshold in the finer grid [0.40,0.38,0.36,0.34,0.32,0.30] is simultaneously (a) ROBUST under image-level bootstrap resampling -- guardrail_violation_rate<=0.05, per EXP-0009's criterion -- AND (b) achieves a mean recall delta vs the production baseline (0.40) of at least the lab's established minimum meaningful delta (0.03, used by every experiment since EXP-0001).
- Reasons: threshold(s) ['0.38'] are ROBUST on the guardrail, but their mean recall delta falls short of the established minimum meaningful delta (0.03) -- robustness and meaningfulness are in tension across the tested grid; no threshold clears both.

## EXP-0012 (2026-09-10T00:02:49.857013+00:00)

- Family: threshold_postprocessing
- Execution status: COMPLETED
- Research verdict: FAIL
- Hypothesis: Test-time augmentation (ultralytics augment=True: multi-view flip/scale inference merged via NMS), confidence and NMS IoU fixed at production values, recovers meaningful Person recall (>= +0.03 vs the augment=False control) while keeping hazard-aggregate precision at or above the standard guardrail (>= 0.757) AND keeping p95 latency within the lab's existing +50% regression guardrail. Mechanism: unlike confidence-threshold or NMS-IoU changes (which can only re-select among a single pass's already-proposed candidates), TTA runs multiple transformed views through the model and can produce a genuinely NEW candidate detection in a view where the original pass proposed nothing -- the one channel tested so far with a real mechanism to potentially recover a TRUE_DETECTOR_MISS case.
- Reasons: guardrail 'hazard.precision' violated: 0.7438 does not satisfy gte 0.7570 (hazard precision must not drop more than 0.05 below baseline)

## EXP-0013 (2026-09-10T01:19:29.174903+00:00)

- Family: small_object
- Execution status: COMPLETED
- Research verdict: FAIL
- Hypothesis: Tiling (cropping each image into 4 overlapping 2x2-grid regions plus the full frame, running inference on each at the fixed production imgsz=640/conf=0.4/iou=0.7, then merging via cross-tile NMS) recovers meaningful Person recall (>= +0.03 vs the frozen single-pass baseline) while keeping hazard-aggregate precision at or above the standard guardrail (>= 0.757), because it increases the EFFECTIVE detector-input scale of small/distant Person instances without changing the network's own input resolution (unlike EXP-0002's failed global-resize approach).
- Reasons: guardrail 'hazard.precision' violated: 0.3139 does not satisfy gte 0.7570 (hazard precision must not drop more than 0.05 below baseline); guardrail 'latency.p95_ms' violated: 166.2649 does not satisfy lte 85.6887 (p95 latency must not exceed 1.5x baseline)
