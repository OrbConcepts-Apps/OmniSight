# OMNISIGHT-PILOT-001 Field Readiness Checklist

Empty template/checklist document. No real participant, session, or media data exists
anywhere in this file or the artifacts it references.

## Two independent gates — both must clear before real participant recording

1. **Software collection-admission gate** (`research.datasets.collection_authorization
   .check_collection_admission`) — currently: **admitted=True** for a request matching
   the exact authorized pilot ID/hash/caps/privacy class.
2. **Ethics/institutional-review determination** (`research.datasets.ethics_review`) —
   currently: **`CURRENT_ETHICS_STATUS = NOT_ASSESSED`** — the conservative default,
   never self-set. `check_field_clearance()` combines both gates; with ethics status at
   `NOT_ASSESSED`, **field clearance is NOT granted**, regardless of the software gate.

**Neither gate alone is sufficient.** Real participant recording remains blocked until
BOTH clear.

## Ethics/institutional-review status — action required from a human

Current value: `NOT_ASSESSED`. Valid values: `NOT_ASSESSED` (default) /
`NOT_REQUIRED_BY_INSTITUTION` / `APPROVED_OR_EXEMPT` / `BLOCKED_PENDING_REVIEW`. This
codebase does NOT claim IRB (or equivalent) is legally required in every circumstance —
only a human operator can determine and record which of the four applies, by editing
`research/datasets/ethics_review.py::CURRENT_ETHICS_STATUS` (or a future equivalent
config the human controls). No automated code path in this repository sets it to a
cleared value.

## Consent readiness checklist (template — `research/datasets/consent_checklist.py`)

Per participant, before their session, a human must confirm ALL of:

- [ ] Participant is eligible under the protocol (staged, consented, no sensitive
      context, per `reports/phase_i/EXP0006_PILOT_PLAN.md` §6/§15)
- [ ] Adult/appropriate consent requirement satisfied under applicable institutional
      policy
- [ ] Participant understands video/image capture
- [ ] Participant understands Person bounding-box annotation
- [ ] Participant understands the proposed research/model-development use
- [ ] Publication/image-use choice recorded (default: private, never published, unless
      the participant separately opts in)
- [ ] Withdrawal/retention terms communicated
- [ ] Pseudonymous participant ID assigned
- [ ] Consent evidence stored SEPARATELY from the research manifest
- [ ] No PII copied into any tracked dataset artifact

`is_session_consent_ready()` returns `True` only when every item above is confirmed AND
a pseudonymous id has been assigned — no signatures, no names, no fake consent records
exist anywhere in this repository.

## Empty session-record template (`research/datasets/session_template.py`)

```json
{
  "pilot_id": "OMNISIGHT-PILOT-001",
  "pilot_plan_hash": "e0488d752b679be2e7d010be2f1ded8f75e3dc1f9d8bdfd85219905b1a257dc3",
  "participant_pseudonymous_id": "",
  "session_id": "",
  "sequence_ids": [],
  "planned_scenario_ids": [],
  "capture_device_logical_id": "",
  "consent_status_reference": "",
  "ethics_review_status_reference": "NOT_ASSESSED",
  "start_end_metadata_policy": "DATE_ONLY",
  "exclusions": [],
  "collection_counters": {"sequences_recorded": 0, "frames_sampled": 0}
}
```

No names, emails, addresses, real timestamps, or media paths anywhere in this template.

## Cap enforcement (`research.datasets.collection_ledger.PilotCollectionLedger`)

Deterministic counters, not documentation alone: ≤6 participants, ≤6 sessions, ≤24
sequences. A 7th participant/session or 25th sequence is rejected with
`PilotCapExceededError` — tested (`tests/test_collection_ledger.py`). Current real
ledger state: **empty** (0/6 participants, 0/6 sessions, 0/24 sequences) — no
registration has occurred.

## Sequence admission (`research.datasets.collection_ledger.check_sequence_admission`)

Before any future sequence is accepted into the pilot manifest, requires: authorized
pilot ID/hash, a registered session, `STAGED_CONSENTED` provenance, no sensitive
context, no known non-consenting-bystander issue, storage-target compliance, and
available cap headroom.

## Bystander stop rule

> If an unplanned non-consenting person becomes materially identifiable in a staged
> sequence, the collector stops or invalidates/excludes that ENTIRE sequence under the
> protocol — never treats the bystander as training data, and never applies automatic
> face redaction as a substitute for exclusion.

No automatic face-redaction workaround exists or is proposed as a substitute for this
rule.

## Pilot data status

Any future collected record defaults to `PILOT_PLANNING_ONLY`
(`research.datasets.collection_ledger.PILOT_PLANNING_ONLY`). Promotion into EXP-0006's
training dataset requires a separate, later, explicit human decision, private-data-use
approval, dataset freeze/versioning, and ultimately `new_training_approved` — never
automatic.

## No external sync

Raw pilot media must remain local, under `data/raw/` (confirmed gitignored,
`tests/test_dataset_storage_contract.py`) or an equivalent configured private path
outside the repository. No commit to git, no cloud upload, no LLM/API transmission, no
external sync of any kind is authorized by this document.

## Current overall status

- Software collection gate: **ADMITTED** (scoped to OMNISIGHT-PILOT-001)
- Ethics/institutional-review status: **NOT_ASSESSED** (blocking)
- Combined field clearance: **NOT GRANTED**
- Real participant recording: **NOT YET PERMITTED**

**Exact remaining human action before the first real participant can be recorded**: a
human operator must record the applicable ethics/institutional-review determination
(`NOT_REQUIRED_BY_INSTITUTION` or `APPROVED_OR_EXEMPT`) — this document, and the
software collection gate alone, are not sufficient.
