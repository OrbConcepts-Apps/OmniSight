# EXP-0006 Registration Report

Registration authorized this task. Execution remains NOT authorized. Zero live LLM
calls, zero real training/GPU/device work, zero data collection.

## Source of truth

`research/preregistrations/EXP-0006-PROPOSED.json` (frozen last task, accepted by
`reports/phase_i/EXP0006_PREREGISTRATION_AUDIT.md`). CANDIDATE-0003 was never touched.

## Registration path

`research/register_exp_0006.py::register_exp_0006()`. Follows the SAME canonical path
already established for EXP-0001–0005 (`research/seed_experiments.py` +
`_exp0004_preregister.py`/`_exp0005_preregister.py`'s idiom): a `research.db.Experiment`
row (execution/business-tracking layer) plus a twin
`research/experiment_specs/EXP-0006.json` (`ExperimentSpec`, the full Phase F scientific
layer). Immediately transitioned `QUEUED → BLOCKED` — the exact same idiom EXP-0005's
own registration used (`db.transition_status("EXP-0005", "BLOCKED", ...)`), reused
rather than inventing a new status value.

**Schema mismatch resolution (per this task's §1 instruction)**: `research.db.Experiment`
cannot represent approval flags, `data_privacy_classification`, `evidence_references`,
`acknowledges_rejected_hypothesis_ids`, the structured seed/recovery-metric definitions,
or `supports/rejects/inconclusive_if` — none of these fields exist on that dataclass.
Rather than drop them, the twin `research/experiment_specs/EXP-0006.json` carries the
COMPLETE, byte-identical `ExperimentSpec` (same object, same frozen hash) — nothing is
lost; this mirrors exactly how EXP-0001–0005 already work (their DB rows also lack
approval flags; the full content lives in their own `experiment_specs/EXP-000N.json`
twins, built by `research/backfill_experiment_specs.py`). No STOP was required because
this resolution already exists as established codebase convention, not a gap.

## Verified facts (all checked directly against the live registration, not assumed)

- `research/db.py`: `sorted(experiment_id for e in list_experiments())` =
  `['EXP-0001', 'EXP-0002', 'EXP-0003', 'EXP-0004', 'EXP-0005', 'EXP-0006']`.
- EXP-0006: `execution_status=BLOCKED`, `research_verdict=PENDING`,
  `experiment_family=training_data`, `validation_requirement=REQUIRES_IPHONE`
  (downstream signal, same convention as EXP-0005's own `REQUIRES_MAC` despite
  executing entirely on Windows/CUDA — see below).
- EXP-0001–0005 verdicts unchanged: PASS / FAIL / FAIL / INCONCLUSIVE / INCONCLUSIVE.
- `research/experiment_specs/EXP-0006.json`'s `frozen_hash` ==
  `research/preregistrations/EXP-0006-PROPOSED.json`'s `frozen_hash` (both
  `e0b4a954e6a6ffb1c3bd067d8a10363dafe236a3c386d51949300a79a806289a`), each independently
  round-trip-verified via `verify_integrity()`.
- All 7 approval flags on the twin spec: **False** — `new_training_approved`,
  `private_user_data_use_approved`, `mac_iphone_deployment_approved`,
  `coreml_model_replacement_approved`, `external_upload_approved`,
  `production_swift_modification_approved`, `signing_distribution_change_approved`.
  Registration performed zero writes to any of these — confirmed by reading the twin
  spec fresh from disk after registration.
- `research.experiment_validator.validate()` on the live twin spec: **1 ERROR
  (`DUPLICATE_ID`) + 3 NEEDS_HUMAN_APPROVAL** (`UNAPPROVED_NEW_TRAINING`,
  `UNAPPROVED_PRIVATE_DATA_USE`, `UNAPPROVED_MAC_IPHONE_DEPLOYMENT`). `DUPLICATE_ID` is
  **expected and correct** post-registration — it is `validate()`'s own no-
  double-registration guard confirming EXP-0006 is genuinely, uniquely registered, not
  a defect. (Re-run in isolation against a scratch DB where EXP-0006 does not yet exist:
  zero errors — see `tests/test_register_exp_0006.py::TestIsolatedProposalValidation`.)
- `is_queue_eligible(result)` = **False**.
- `research.execution_budget.require_execution_budget("...", config=None)` raises
  `ExecutionBudgetError` before any `Runner` method is called — verified directly:
  `FakeRunner.prepare()`/`launch()` were never invoked.
- `git status --short research/candidates/`: only the untracked `CANDIDATE-0003/`
  directory, no modification.
- `git diff --stat -- ios/ benchmark/config.py`: empty.
- `research/llm_usage.json`: unchanged (`{"2026-09-05": 11, "2026-09-06": 2}`) — zero
  live calls this task.

## Status semantics audit (§5)

`BLOCKED` (existing `EXECUTION_STATUSES` value, already used for exactly this purpose
by EXP-0005) correctly means "registered scientific experiment, not authorized/
executable yet" — no new status was invented. `research.experiment_validator.
is_queue_eligible()` is entirely independent of `db.py`'s `execution_status` string
(it never reads the DB row's `execution_status` field at all) — registration, queue
admission, and execution admission were already three separate mechanisms in this
codebase; this task did not need to fix an architecture conflation because none existed.

## Full test suite

864 passed (818 prior + 46 new registration/preregistration tests). One pre-existing
test (`test_experiment_immutability.py::test_no_exp_0006_exists`) asserted the OLD
invariant ("EXP-0006 must not exist") and was updated, per this task's own authorization,
to assert the NEW correct invariant (EXP-0006 exists, BLOCKED, no EXP-0007) — not
weakened, not deleted, renamed to `test_exp_0006_registered_blocked_no_exp_0007`.
