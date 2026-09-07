# EXP-0006 Staged/Consented Pilot Plan

Protocol design only. **No participant recruited. No media captured. No annotation
performed. No baseline inference run.** This document does not authorize collection —
it prepares the exact bounded request a human could later approve.

## 5. Pilot purpose

Answers EXP-0006's **planning questions**, not its final hypothesis: Person prevalence,
hard-condition prevalence, sequence correlation, usable-frame yield, annotation burden,
annotation agreement, near-duplicate rate, privacy/exclusion rate, approximate baseline
TRUE_DETECTOR_MISS prevalence, and rough capture/runtime/storage burden. **Not** intended
to demonstrate model improvement — no training occurs on pilot data by default (§18).

## 6. Pilot participant/scenario policy

`STAGED_CONSENTED` only. No incidental non-consenting bystanders as intended subjects.
Hard exclusions, no exceptions: minors, schools, medical environments, bathrooms/
changing areas, private-home interiors, any other sensitive context. Controlled public
or private-neutral environments (e.g. an office space, a consented outdoor walking
route) where every intentional participant has given consent.

## 7. Proposed bounded pilot size

`research/datasets/pilot_plan.py::PilotSizePlan` (real proposed numbers, not fabricated
observations):

- **6 participants**
- **6 collection sessions** (one per participant)
- **4 independent sequences per session** (~24 sequences total)
- **~20 seconds per sequence**
- **Frame sampling**: 1 fps baseline, plus dense 5 fps sampling within any
  pre-identified hard-event window (motion-blur onset, occlusion transition,
  distance-band change), capped at 40 sampled frames per sequence

**Why this is enough for planning but not a final conclusion**: the independent unit is
the SEQUENCE (24 of them, from 6 participants), not the frame — a sequence sampled at
up to 40 frames yields at most ~960 total sampled frames, but these are heavily
correlated within each sequence (near-identical adjacent content), so the EFFECTIVE
independent sample size for any prevalence estimate is bounded by the sequence/
participant count (~24 and ~6 respectively), not the frame count. This is sufficient to
get a rough, order-of-magnitude planning estimate (e.g. "is Person prevalence closer to
30% or 70%?", "does annotation take 30 seconds or 5 minutes per frame?") but is far too
small and too correlated to support any final statistical claim about detection
improvement — exactly the distinction §7's instruction requires stating explicitly.

## 8. Capture matrix (modest, information-value-prioritized)

Not exhaustive combinatorial coverage — prioritized by prior-research-cited failure
value: normal illumination, low illumination, motion blur/camera motion, partial
occlusion, small/distant Person, clutter, unusual assistive-phone viewpoint (chest-
height/hand-held), indoor, outdoor-where-controlled. Each of the 24 sequences is
assigned to cover 1-2 of these factors deliberately (a per-sequence checklist), rather
than attempting all combinations across only 24 sequences.

## 9. Negative examples

Controlled no-Person / hard-negative sequences (person-shaped distractors: mannequins,
posters, statues) are included among the 24 sequences (a subset, not the majority).
Necessary for: **false-positive assessment** (the hazard-precision guardrail concerns
false alarms, not just misses); **annotation process validation** (confirms annotators
correctly produce zero Person boxes on a true negative, a basic QA sanity check);
**prevalence estimation** (a dataset with only positive examples cannot estimate real
Person prevalence at all). Pilot negative-sequence proportion is NOT tuned to match any
assumed final training prevalence — it exists to inform that later decision, not
pre-empt it.

## 10. Device / capture source

The pilot is for dataset planning, not deployment validation. Any consistent, consumer-
grade smartphone camera capture is acceptable for the pilot (device metadata recorded
per `MediaUnitRecord.device`/`camera_configuration`). **Recording video with an iPhone
is simple media capture — it is explicitly NOT authorization for CoreML/device
validation, Mac/iPhone model deployment, or any on-device inference testing.**
`mac_iphone_deployment_approved` is not granted by this document and is unaffected by
whatever device is used to record pilot video.

## 11. Consent protocol (data contract, no signatures collected now)

A future consent form must cover, at minimum: (a) research image/video capture; (b)
Person bounding-box annotation of the captured media; (c) use of the annotated data for
model training (a SEPARATE clause from mere capture — see §13's governance-gap
finding); (d) potential research-result publication (aggregate findings, not raw
imagery, unless separately and explicitly agreed); (e) explicit statement of whether
example imagery may ever be published, or must remain private indefinitely — defaulting
to **private, never published**, unless a participant separately opts in; (f)
withdrawal/retention policy (a participant may request removal; how removal is honored
once data has entered aggregate statistics is addressed in the final legal-reviewed
form, not decided here); (g) who has storage access and under what conditions. **This
document does not claim legal sufficiency** — the actual consent language requires
human/legal review before any real use.

## 12. Participant identifiers

Pseudonymous IDs only in the tracked research dataset — no names, no contact
information. Consent records (who actually consented, contact info, signed form) are
kept in a SEPARATE system from the pseudonymous dataset manifest; the dataset manifest
never contains the linkage key itself, only the pseudonymous id
(`MediaUnitRecord.annotator_ids`/session identifiers already enforce this structurally
— `validate_media_unit()` rejects any id that looks like a secret/contact-shaped
string, and the schema has no name/email/phone field at all).

## 13. Privacy review — required, not granted here

`private_user_data_use_approved` remains **False**, left untouched by this task.

**Governance gap identified, not resolved**: the current approval schema
(`ExperimentProposal.private_user_data_use_approved`, gated by
`data_privacy_classification=PRIVATE_USER_DATA`) has only ONE approval flag covering
"private data use" — it does not distinguish **collection authorization** (may this
project capture staged/consented private data at all) from **training-use
authorization** (may already-collected data be used to train a model). A human might
reasonably want to approve a small, bounded pilot COLLECTION without yet committing to
approving its later use in an actual training run, or vice versa. This is reported here
as a real gap in the existing approval schema, per this task's explicit instruction —
**not self-resolved, not self-granted**. If a human wishes to authorize the pilot next,
the narrowest correct action is a new, explicitly-scoped approval distinct from
`private_user_data_use_approved` and `new_training_approved` (see §47 below for the
exact proposed scope).

## 14. Empty pilot manifest

Created: `research/datasets/pilot_manifests/OMNISIGHT-PILOT-001.json` — schema/pilot
version metadata, the planned scenario taxonomy, expected logical identifiers, and
`"records": []`. **Zero real captures.** Generated via
`research.datasets.pilot_plan.build_empty_pilot_manifest()`/`save_empty_pilot_manifest()`.

## 15. Pilot version — distinct identity

`PILOT_ID = "OMNISIGHT-PILOT-001"` (`research/datasets/pilot_plan.py`) — deliberately a
different naming convention from the final dataset's `omnisight_v{date}_annotated`
format, so the two can never be confused. `is_training_data: false` is a field
IN the manifest itself (§18) — pilot data does not silently become the final training
dataset; promotion requires a separate, later, explicit versioned decision.

## 16. Frame sampling policy

1 fps baseline + dense 5 fps sampling only within pre-identified hard-event windows
(motion-blur onset, occlusion transition, distance-band change), capped at 40 sampled
frames per ~20-second sequence — NOT every adjacent frame. This bounds annotation
burden while still surfacing hard-condition transitions the dense sampling is meant to
catch, and keeps the correlated-frame count per sequence small and explicit rather than
implicit.

## 17. Baseline inference procedure (designed, NOT run)

`research.datasets.pilot_plan.BaselineEvalConfig` — records exactly what a future
evaluation run must fix before executing: `baseline_model_sha256` (the VERIFIED
`21ffa3718c577ac23e708e4c0544c49a20682efa03914d5f816166b54e8fd3fe`, per
`EXP0006_CHECKPOINT_PROVENANCE_AUDIT.md`), `confidence_threshold=0.4`,
`iou_threshold=0.7`, `imgsz=640`, `evaluation_code_version` (a placeholder field for the
actual eval-code commit/version once it runs). This is the exact procedure §17's
derivation rule (ground truth + frozen baseline inference → TRUE_DETECTOR_MISS) would
apply to real pilot data — designed here, not executed.

## 18. Pilot data does not enter training automatically

Default: pilot captures are **planning/evaluation evidence only**. No automatic
inclusion in EXP-0006's training set. `PILOT_IS_TRAINING_DATA = False` is a hardcoded
module constant, not a per-record toggle a caller could accidentally flip. If later
reuse is scientifically acceptable, it requires an explicit, separately-versioned
decision and split assignment before any training run — never inherited implicitly from
having been "already collected."

## 19. Pilot output metrics (predefined, no values yet)

`research.datasets.pilot_plan.PILOT_OUTPUT_METRIC_NAMES` — 23 named metrics: total
sessions/sequences/sampled frames, Person-positive/no-Person frame counts, exclusion
counts/reasons, privacy/bystander exclusion rate, bounding-box count, inter-annotator
agreement, adjudication rate, exact/near-duplicate rates, baseline Person recall/
precision, derived TRUE_DETECTOR_MISS count/rate, hard-condition prevalence, annotation
minutes per frame, storage per minute/session, effective-sample-size and sequence-
correlation estimates. No value populated for any of them.

## 20. Pilot readiness verdict — exhaustive, deterministic

`research.datasets.pilot_plan.classify_pilot_readiness()` takes a
`PilotCompletenessReport` (8 booleans: dataset-size/annotation-cost/usable-yield/
failure-prevalence estimability, QA-feasibility/storage-privacy-workflow/leakage-
controls validation, and a `protocol_defect_found` flag) and returns exactly one of
`READY` / `EXTEND_PILOT` / `REDESIGN_PROTOCOL` — proven exhaustive over all 256
combinations by test. `protocol_defect_found` always wins (a defect must be fixed
before anything else matters); `READY` requires every other estimate in hand;
otherwise `EXTEND_PILOT`.

## 21. No model training — reaffirmed

Even a fully successful pilot **does not** make EXP-0006 executable. The next
authorization, if granted, would permit only bounded pilot collection/annotation/
evaluation preparation — never `new_training_approved`, never a real training run.
