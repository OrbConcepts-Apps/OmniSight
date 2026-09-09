# OMNISIGHT-PILOT-001 Collection Approval Report

Human-authorized approval applied. Zero collection performed. Zero live LLM calls.

## Approval semantics

`staged_pilot_collection_approved = True` on `EXP-0006` (Amendment 002). Means:
permission to perform ONE specifically-scoped staged/consented collection, bound to
`pilot_id=OMNISIGHT-PILOT-001` and `pilot_plan_hash=e0488d752b679be2e7d010be2f1ded8f75e3dc1f9d8bdfd85219905b1a257dc3`.
Implies nothing about training use, private-data-training use, external upload, device
validation, production modification, CoreML replacement, or signing/distribution.

## Scope binding (unchanged from prior audit)

- Pilot ID: `OMNISIGHT-PILOT-001`
- Pilot-plan hash: `e0488d752b679be2e7d010be2f1ded8f75e3dc1f9d8bdfd85219905b1a257dc3`
- Caps: ≤6 participants, ≤6 sessions, ≤24 sequences, ~20 seconds/sequence
- Privacy class: `STAGED_CONSENTED` only

## Amendment / audit mechanism

`research/amend_exp_0006_002.py::apply_amendment()` — `ExperimentSpec.amend()`, the
same mechanism used for Amendment 001. Sets ONLY `staged_pilot_collection_approved`.
Old hash `8a25cd096a921c209f4741676adabe02d68d2b644a3f9811d7f5868332ccfb3c` → new hash
`09c68c7e8c6772a8cde886255de87cc7838a374b49779608b5fd268bbae21ea5`. `spec.amendments`
now has 2 entries (Amendment 001, Amendment 002), both preserved, neither overwritten.
`research/db.py`'s `experiment_events` table carries a new
`amend:staged_pilot_collection_approved` event pair. `research/preregistrations/
EXP-0006-PROPOSED.json` (the original historical preregistration) remains completely
untouched — still shows `staged_pilot_collection_approved=False`, still verifies at its
original hash `e0b4a954e6a6ffb1c3bd067d8a10363dafe236a3c386d51949300a79a806289a`.

## Real bug found and fixed this task (production robustness, not just test flakiness)

While verifying the collection-vs-training separation with real subprocess tests, a
genuine race was discovered and fixed in `research/execution_job/process_ownership.py::
live_snapshot_for_pid()`: when a job is cancelled, `graceful_stop()`'s `CTRL_BREAK_EVENT`
can already be terminating the target process WHILE `force_terminate()`'s own identity
re-verification is reading that same process's `cmdline()` — a legitimate, in-flight
process teardown, not "ambiguous ownership." Previously this could be misreported as an
ambiguous identity (raising `ProcessOwnershipError` incorrectly). Fixed: a `NoSuchProcess`
encountered mid-check (after the function's initial liveness check already passed) is
now correctly treated as a clean "process has since exited" result (`exists=False`,
handled safely by `verify_process_identity`) rather than an ambiguous one. Verified
stable across repeated real-subprocess test runs (12/12, then full-suite 2x) after the
fix; the same race was intermittently failing `test_cancellation_path_real_subprocess`
before it.

## What remains false (unchanged, verified)

`private_user_data_use_approved`, `new_training_approved`,
`mac_iphone_deployment_approved`, `external_upload_approved`,
`production_swift_modification_approved`, `coreml_model_replacement_approved`,
`signing_distribution_change_approved` — all confirmed `False` on the real, amended spec.

## EXP-0006 status (unchanged)

`execution_status=BLOCKED`, `research_verdict=PENDING`. `validate()` on the real spec:
`errors=['DUPLICATE_ID']` (the expected registration-path guard, per the accepted
correction — **not** an execution blocker), `needs_human_approval=`
`['UNAPPROVED_MAC_IPHONE_DEPLOYMENT', 'UNAPPROVED_NEW_TRAINING', 'UNAPPROVED_PRIVATE_DATA_USE']`
(3 remaining, down from 4 — `UNAPPROVED_STAGED_PILOT_COLLECTION` no longer fires, since
it is now satisfied). `is_queue_eligible()=False`. `require_execution_budget(config=None)`
still refuses training execution admission before touching any Runner.

## Collection readiness vs. occurrence

`COLLECTION_AUTHORIZED = YES` (software gate). `COLLECTION_OCCURRED = NO` — zero media
collected, zero participants contacted, zero sessions conducted, zero sequences
recorded. No automated collector was launched.
