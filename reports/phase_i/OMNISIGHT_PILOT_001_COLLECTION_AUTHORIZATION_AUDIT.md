# OMNISIGHT-PILOT-001 Collection Authorization Audit

Governance/schema work only. Zero live LLM calls. Zero data collection.

## Test-count reporting check (this task's item 1)

The prior chat message stated "39 tests added" (breakdown: 12 + 15 = 27). Grepped both
persisted reports (`EXP0006_AMENDMENT_001_REPORT.md`, `EXP0006_PILOT_PLAN.md`) for "39"
— **zero matches**. As with the prior "75" miscount, the error existed only in
conversational chat text, never in a persisted file. No file correction was needed or
made.

## Approval semantics

Three human-authority flags now exist on `ExperimentProposal`, each independent, none
inferring another:

| Flag | Means | Does NOT mean |
|---|---|---|
| `staged_pilot_collection_approved` (**new**) | Permission to perform ONE specifically-scoped staged/consented collection, bound to a pilot id + frozen pilot-plan hash | Training use, private-data-training use, external upload, device validation, production modification, CoreML replacement, signing/distribution changes |
| `private_user_data_use_approved` (existing, unchanged semantics) | May the already-collected private data be USED for the research purpose EXP-0006 governs (including later training, only once `new_training_approved` is ALSO separately granted) | Permission to collect NEW private data in the first place |
| `new_training_approved` (existing, unchanged semantics) | May the actual new-model-training procedure run | Anything about collection or general data use |

All three default `False`, are never self-granted, and all three remain `False` in the
real persisted `EXP-0006` spec after this task.

## Schema/hash-tolerance bug found and fixed (real, not fabricated)

Adding `staged_pilot_collection_approved` to `ExperimentProposal` initially broke
`EXP-0006`'s (but not EXP-0001–0005's) `verify_integrity()` — a genuine latent bug in
the existing `_FIELDS_ADDED_AFTER_PHASE_F_FREEZE` tolerance mechanism, which excluded
**all** tolerated fields as a single all-or-nothing fallback. EXP-0006 was frozen/amended
*after* the CoreML/signing fields already existed but *before* this new field did — a
case the all-or-nothing fallback could not handle (excluding the CoreML/signing fields
too, which WERE genuinely part of its real frozen payload, broke the comparison).
**Fixed**: `ExperimentSpec.verify_integrity()` now tries every subset of the
currently-at-default tolerated fields, not just the full set — confirmed all of
EXP-0001–0006 and the untouched `EXP-0006-PROPOSED.json` preregistration still verify
correctly (`uv run pytest` — no regressions, 997/997 passing).

## Scope binding

`research/datasets/pilot_plan.py::OMNISIGHT_PILOT_001_PLAN` — the frozen, canonical
plan (6 participants, 6 sessions, 4 sequences/session = 24 max sequences, ~20s/sequence,
the exact caps from `EXP0006_PILOT_PLAN.md`, never increased here).
`OMNISIGHT_PILOT_001_PLAN_HASH = e0488d752b679be2e7d010be2f1ded8f75e3dc1f9d8bdfd85219905b1a257dc3`
— a SHA-256 of the plan's own fields. Any change to caps/sampling-rule/duration produces
a different hash automatically (tested: `TestPilotPlanMutationInvalidatesScope`).

## Hash binding — collection admission

`research/datasets/collection_authorization.py::check_collection_admission()` — a
**deterministic function, structurally separate from experiment registration, queue
admission (`research.experiment_validator.is_queue_eligible`), and training execution
admission (`research.execution_budget.require_execution_budget`)**. Requires ALL of:
operational state RUNNING; `pilot_id == "OMNISIGHT-PILOT-001"`; `pilot_plan_hash ==` the
current frozen hash; `staged_pilot_collection_approved == True`; participant/session/
sequence counts within the frozen caps; `privacy_class == "STAGED_CONSENTED"` (stricter
than the general dataset schema, which also allows `INCIDENTAL_CONSENTED`);
`sensitive_context == False`; storage path confirmed outside the tracked git raw-data
area. Fails closed on any single missing check — never partially admits.

## Collection-vs-training distinction — proven by test

`check_collection_admission()`'s signature has NO parameter for
`new_training_approved`, `private_user_data_use_approved`, or Mac/iPhone approval at
all — it is structurally incapable of requiring or being satisfied by them
(`TestNoUnrelatedApprovalRequired`, `TestApprovalIndependence`). Ordinary staged-
consented smartphone media capture requires neither device-validation approval nor
training approval — confirmed granted (`admitted=True`) using only the collection flag.

## Consent/provenance boundary

`is_pilot_media_collectible(consent_status, privacy_class)` — a minimal, 2-parameter,
enum-only function (confirmed by inspecting its own signature in a test) proving a
media record can never be treated as collectible under this pilot's scope unless BOTH
fields are exactly `STAGED_CONSENTED`. No real consent-management system built; no
name/email/signature field exists anywhere in this module or the underlying
`MediaUnitRecord` schema.

## Remaining human decision

`staged_pilot_collection_approved` remains `False`. Collection admission for
`OMNISIGHT-PILOT-001` returns `admitted=False`, blocker: `"staged_pilot_collection_approved is False"`
(the only blocker when every other field is valid — confirmed by test). No collection
is possible. The narrow authorization that would unblock it is stated in this task's
final report §40.
