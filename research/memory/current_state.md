# Current State

> Read `research/memory/README.md` first — this file and its four siblings
> are mandatory reading before proposing any new experiment.

Last updated: 2026-09-09 (post-Amendment-002, Phase I/J complete, EXP-0006
registered/blocked/amended, pilot collection software-authorized pending
ethics determination).

## Canonical baseline

- Run: `RUN-20260904-002` (`benchmark/results/baseline/`). Model
  `yolov8m-oiv7.pt`, imgsz=640, conf=0.4, iou=0.7 — this is the shipped
  production operating point and must never be changed by experiment code.
- Hazard classes (person, car, truck, bus, bicycle, motorcycle, stairs, dog):
  Precision=0.807, Recall=0.480, F1=0.602, mAP50=0.582.
- Person (GT=303, the largest and most trustworthy sample): Recall=0.211,
  Precision=0.667. Worst hazard-class recall in the dataset.
- Latency (Windows/CUDA proxy only): p50=18.0ms, p95=57.1ms, p99=65.8ms.
- Checkpoint provenance: `yolov8m-oiv7.pt` SHA-256
  `21ffa3718c577ac23e708e4c0544c49a20682efa03914d5f816166b54e8fd3fe`, audited
  in `reports/phase_i/EXP0006_CHECKPOINT_PROVENANCE_AUDIT.md` — pretraining
  history beyond the shipped weights is UNKNOWN (not fabricated as known).

## Completed experiments (see `known_failures.md`/`failed_methods.md`/
`successful_methods.md`/`open_questions.md` for full per-experiment detail)

- EXP-0001 (threshold_postprocessing) — COMPLETED / PASSED (confirmatory
  negative control: threshold reduction alone cannot fix Person recall
  without unacceptable precision collapse).
- EXP-0002 (resolution sweep, 640->960/1280) — COMPLETED / FAILED (hazard
  recall guardrail violated).
- EXP-0003 (semantic class-confusion recovery) — COMPLETED / FAILED (hazard
  precision guardrail violated).
- EXP-0004 (generic preprocessing) — COMPLETED / INCONCLUSIVE.
- EXP-0005 (model variant, e.g. YOLO11m) — COMPLETED / INCONCLUSIVE overall;
  YOLO11m recovered some TRUE_DETECTOR_MISS cases (17/92) but its advantage
  mostly disappeared at precision-matched thresholds. Simple model scaling
  did not solve the problem.
- EXP-0006 (domain-matched training data, matched CONTROL/INTERVENTION
  design, 3 seeds) — REGISTERED, execution_status=BLOCKED,
  research_verdict=PENDING. Preregistration frozen + Amendment 001 (fixed
  ambiguous IoU-QA rule) + Amendment 002 (staged_pilot_collection_approved
  granted, scoped exactly to OMNISIGHT-PILOT-001). No training has occurred.
  `new_training_approved=False` and `private_user_data_use_approved=False`
  remain the two flags that must separately be granted before any real
  training run — collection approval does not imply either.
- EXP-0007 (per-class Person confidence threshold policy) — COMPLETED /
  REJECTED for a purely structural reason (stale pytest invariants left
  over from the EXP-0006 era hardcoded "no experiment beyond EXP-0006
  exists"), not a scientific rejection. Preserved honestly, not deleted.
- EXP-0008 (identical design, parent_experiment_id=EXP-0007, run after the
  stale invariants were corrected) — COMPLETED / PASS (deterministic, single
  point estimate, unchanged/unretracted). Isolating the confidence threshold
  to Person only (0.30, other hazard classes fixed at 0.4) recovers
  person.recall 0.211->0.277 (+0.0660) while hazard-aggregate precision
  holds at 0.767, clearing the 0.757 guardrail by a THIN margin (+0.0098).
- EXP-0009 (post-hoc image-level bootstrap robustness check of EXP-0008,
  parent_experiment_id=EXP-0008, 2000 replicates, seed=20260909) —
  COMPLETED / FAIL (FRAGILE): hazard-aggregate precision at
  person_threshold=0.30 falls below the 0.757 guardrail in 36.9% of
  image-level bootstrap resamples. The recall improvement itself IS robust
  (95% CI [0.0365, 0.1026]) — only the precision-guardrail margin is fragile.
- EXP-0010 (threshold-sensitivity bootstrap, parent_experiment_id=EXP-0009,
  finer grid [0.40,0.38,0.36,0.34,0.32,0.30], same seed) — COMPLETED / FAIL:
  no threshold is BOTH robust on the guardrail AND clears the lab's
  established +0.03 minimum-meaningful-delta bar. person_threshold=0.38 is
  the only robust point (violation_rate=0.042) but its mean recall gain is
  only +0.0199. **This closes the per-class Person confidence-threshold line
  of inquiry (EXP-0007/8/9/10) as a genuine, defensible negative result** —
  see `reports/baseline/person_per_class_threshold_analysis.md` (full
  addendum) for the complete chain, explicitly distinguishing EXP-0008's
  unretracted deterministic PASS from its (fragile) robustness from
  (insufficient) evidence for any production recommendation. Zero new
  inference, zero training, zero private data throughout EXP-0007-0010 —
  all reuse EXP-0001's existing conf=0.01 capture. Orthogonal to and
  unaffected by the blocked EXP-0006 pilot.
- EXP-0011 (NMS/inference-IoU sensitivity, real non-training inference over
  the 380-image eval set, grid=[0.9,0.8,0.7,0.6,0.5], confidence fixed at
  production 0.4) — COMPLETED / INCONCLUSIVE. person.recall is literally
  identical (0.21122112211221122) at EVERY grid point tested — the cleanest
  possible negative result, no ambiguity. Verified (in code, not assumed)
  that iou=0.7 is the model's own internal NMS threshold, distinct from the
  eval-matching IoU (0.5). Mechanism analysis (written before the result):
  NMS can only choose among already-proposed candidate boxes, never invent
  one, so it has no plausible channel to fix TRUE_DETECTOR_MISS (the
  dominant Person failure mode). A methodological artifact was found and
  corrected: a constant "1/92 TRUE_DETECTOR_MISS recovered" at every grid
  point (including control) was investigated and shown to be a real
  detection legitimately claimed by a NEIGHBORING GT box under the official
  classifier's cross-GT exclusion logic, not a genuine recovery — see
  `reports/baseline/nms_iou_sensitivity_analysis.md` for the full writeup.
  **Both post-hoc inference-time levers on the shipped checkpoint —
  confidence threshold (EXP-0007-0010) and NMS IoU (EXP-0011) — are now
  exhausted without a robust, meaningful Person-recall improvement.**

## Pilot data collection (OMNISIGHT-PILOT-001)

- Software collection gate: ADMITTED (scope: <=6 participants, <=6 sessions,
  <=24 sequences, ~20s/sequence, STAGED_CONSENTED only; bound to
  `pilot_plan_hash=e0488d752b679be2e7d010be2f1ded8f75e3dc1f9d8bdfd85219905b1a257dc3`).
- Ethics/institutional-review status: `NOT_ASSESSED` (default, conservative,
  never self-set by any code path) — this is the current human-only blocker
  on real participant recording. See
  `reports/phase_i/OMNISIGHT_PILOT_001_ETHICS_REVIEW_REQUEST.md` for the
  reviewer-facing handoff packet, and
  `reports/phase_i/OMNISIGHT_PILOT_001_FIELD_CHECKLIST.md` for the internal
  engineering readiness state.
- Zero media collected, zero participants contacted, zero consent records
  exist anywhere in this repository.

## EXP-0012 and the inference-time-lever synthesis

- EXP-0012 (test-time augmentation, ultralytics `augment=True` vs
  `augment=False` control, real non-training inference, confidence/NMS-IoU
  fixed at production) — COMPLETED / FAIL. person.recall improved +0.0231
  (below the +0.03 bar) while hazard-aggregate precision hard-violated the
  guardrail (0.7438 vs 0.757, margin -0.0132). TRUE_DETECTOR_MISS recovery
  stayed 1/92 (same known artifact case) — TTA's modest gain does not come
  from the targeted TRUE_DETECTOR_MISS cases. Strictly worse cost/benefit
  tradeoff than the already-closed threshold=0.30 candidate.
- **Synthesis**: three mechanistically distinct, non-training,
  inference-time-only levers on the shipped checkpoint have now all been
  tested and closed as negative: Person confidence threshold
  (EXP-0007-0010), NMS IoU (EXP-0011), test-time augmentation (EXP-0012).
  None can fix the dominant Person failure mode (TRUE_DETECTOR_MISS, 92/239
  baseline FNs, 38.5%) because none can make the model recognize something
  it fundamentally does not represent well enough at ANY decision-time
  setting — a representational/training-data limitation, not a
  decision-policy one. This strengthens (does not merely coexist with) the
  case for EXP-0006's original hypothesis (domain-matched training data) as
  the most promising remaining lever — still blocked on the same two human
  approvals (ethics status, then training/private-data approval), unchanged
  by any of this inference-time work.
- EXP-0013 (image tiling: 2x2 crop grid + reused full-image pass, 20%
  overlap, cross-tile NMS merge, real non-training inference, confidence/
  NMS-IoU fixed at production) — COMPLETED / FAIL. Mechanism verified
  against the actual ultralytics LetterBox pipeline before running (not
  assumed): all 380 images exceed 640x640, so tiling genuinely increases
  effective object scale within the fixed 640 network input, distinct from
  EXP-0002's failed global-resize (which changed the network's own input
  resolution and made recall worse). **Zero TRUE_DETECTOR_MISS recovery
  (0/92, 0/68 small subset)**, including 0/76 of the cases that were
  geometrically fully contained within a single crop tile — the most
  favorable possible condition for the hypothesized mechanism. Hard
  hazard-precision guardrail violation (0.314 vs 0.757), root-caused (not
  hand-waved) to large objects (Bicycle/Car) spanning tiles producing
  low-mutual-IoU partial-view duplicates that same-class NMS can't merge.
  2.91x latency. Preceded by 19 hand-computed synthetic-geometry
  correctness tests (tests/test_tiling.py), all passing, before any
  benchmark ran. See `reports/baseline/tiling_analysis.md`.
- **SYNTHESIS**: all four mechanistically distinct, non-training,
  image-only levers on the shipped checkpoint are now tested and closed as
  negative — Person confidence threshold (EXP-0007-0010), NMS IoU
  (EXP-0011), test-time augmentation (EXP-0012), image tiling (EXP-0013).
  No further mechanistically distinct, non-training, image-only candidate
  on this single checkpoint has been identified. This materially
  strengthens — without by itself proving — the hypothesis that the
  dominant Person failure mode (TRUE_DETECTOR_MISS) is a representational/
  training-data limitation. EXP-0006 (domain-matched training data)
  remains the most promising identified remaining lever, still blocked on
  `ethics_or_institutional_review_status` (human-only, unchanged) and then
  separately on `new_training_approved`/`private_user_data_use_approved`.

## Phase I/J infrastructure status

- Phase I proposal-only autonomous loop: built and exercised (CANDIDATE-0001,
  0002 rejected with genuine reasons; CANDIDATE-0003 accepted, became
  EXP-0006's basis).
- Phase J execution infrastructure: JobManager, job state machine, real
  `LocalProcessRunner` (Windows-safe subprocess control, active watchdog,
  GPU concurrency slots, process-tree ownership verification) — built and
  tested against real (harmless) subprocesses. Never used for EXP-0006
  execution.

## What has deliberately NOT been done

- No real EXP-0006 training (blocked on `new_training_approved` and
  `private_user_data_use_approved`, both False).
- No real pilot participant recording (blocked on
  `ethics_or_institutional_review_status`).
- No live LLM completion call this session.
- No device-validated (Mac/iPhone/CoreML) execution of anything.
- No production Swift/CoreML modification.
