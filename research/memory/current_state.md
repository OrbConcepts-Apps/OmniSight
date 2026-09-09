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
  stale invariants were corrected) — COMPLETED / PASS. Isolating the
  confidence threshold to Person only (0.30, other hazard classes fixed at
  0.4) recovers person.recall 0.211->0.277 (+0.0660) while hazard-aggregate
  precision holds at 0.767, clearing the 0.757 guardrail by a THIN margin
  (+0.0098) — see `reports/baseline/person_per_class_threshold_analysis.md`
  for the full result table and caveats (person.precision cost, sampling-
  noise risk on the thin margin, not a production change by itself). Zero
  new inference, zero training — reuses EXP-0001's existing conf=0.01
  capture. Orthogonal to and unaffected by the blocked EXP-0006 pilot.

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
