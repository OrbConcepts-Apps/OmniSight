# EXP-0006 Amendment 001 Report

Applied via the existing, already-tested `ExperimentSpec.amend()` mechanism
(`research/amend_exp_0006_001.py`). Zero live LLM calls. No scientific content changed
beyond the one identified ambiguity.

## Reporting correction (this task's item 1)

The prior session's final chat message stated "75 tests added" while its own itemized
breakdown summed to 40 (12 provenance + 25 manifest + 3 storage). **No persisted report
file contained the incorrect count** — grepped both `EXP0006_DATASET_PROTOCOL.md` and
`EXP0006_CHECKPOINT_PROVENANCE_AUDIT.md`; neither mentions a test-count figure at all.
The miscount existed only in that turn's conversational final-report text, which is not
a persisted artifact. No file correction was required or made.

## What changed

- **File amended**: `research/experiment_specs/EXP-0006.json` (the registered twin —
  the canonical record `ExperimentSpec.amend()` operates on).
- **File explicitly NOT touched**: `research/preregistrations/EXP-0006-PROPOSED.json`
  (the original, immutable historical preregistration) — confirmed still contains the
  original ambiguous sentence verbatim, hash unchanged.
- **Field amended**: `isolation_requirements` (one sentence replaced within a larger
  unchanged string).

## Before / after wording

**Before** (the exact frozen sentence):
> "inter-annotator bounding-box IoU agreement target >= 0.7 on >= 95% of boxes before a
> batch is accepted"

**After**:
> "annotator-pair bounding-box agreement rule (PREREGISTERED PILOT QA THRESHOLD, not yet
> evidence-backed — see EXP-0006 Amendment 001): match annotator A's and B's Person
> boxes via greedy IoU assignment (IoU = intersection-over-union of the two boxes'
> areas; ignore/ambiguous regions per the annotation ontology are excluded from this
> metric entirely, on both sides); per QA-sampled batch, three separate statistics
> apply: (a) >=95% of MATCHED pairs must achieve IoU>=0.7 (a PROVISIONAL threshold the
> pilot tests for achievability, not an already-observed performance fact); (b)
> unmatched-box rate (a box with no counterpart at IoU>=0.1, counted as a miss, never
> silently folded into (a)'s IoU statistic) must be <=5% of all boxes across both
> annotators; (c) no more than 5% of images in the batch may fall below an 80%
> per-image matched-pair agreement rate. The first full batch of any new session is
> double-annotated at 100% before any agreement rate is computed. If any of (a)/(b)/(c)
> is missed, the ENTIRE batch is held at annotation_status=FIRST_PASS, fully
> re-annotated or adjudicated, and the guideline is reviewed for ambiguity before any
> further batch — never silently accepted at a lower bar, and thresholds are never
> retuned post-hoc without a further explicit amendment"

## Minimum content checklist (this task's section 2) — all present in the new wording

- Box matching between independent annotators: greedy IoU assignment.
- Metric definition: IoU = intersection-over-union of two boxes' areas.
- Included/excluded from the metric: ignore/ambiguous regions excluded on both sides.
- Scope: matched-pair rate, unmatched-box rate, AND a per-image minimum — not aggregate
  alone.
- Unmatched-box treatment: counted separately as a miss, never folded into the IoU
  statistic.
- Ignore/ambiguous-region treatment: excluded from the metric entirely.
- Minimum sampled quantity: 100% double-annotation for a session's first batch.
- Pass/fail behavior: whole-batch hold at `FIRST_PASS`, re-annotation/adjudication,
  guideline review.
- Adjudication behavior: referenced explicitly as part of the failure path.

## Threshold evidence status (this task's section 3)

The 0.70/95% numbers were **kept, not changed** — this task's authorization explicitly
allows keeping them provisionally "if changing the thresholds themselves would require
evidence not yet available." No pilot has run; no empirical annotator-agreement data
exists. The amendment therefore **explicitly labels both numbers "PREREGISTERED PILOT
QA THRESHOLD, not yet evidence-backed"** and **"a PROVISIONAL threshold the pilot tests
for achievability"** — the numbers are unchanged, but their epistemic status is now
stated honestly rather than left to read as an established fact. No observed annotation
performance was invented to justify keeping or changing them.

## Hash / immutability

- **Original frozen hash** (unchanged, preserved in `research/preregistrations/
  EXP-0006-PROPOSED.json`, still verifiable): `e0b4a954e6a6ffb1c3bd067d8a10363dafe236a3c386d51949300a79a806289a`
- **Amended hash** (the registered twin, `research/experiment_specs/EXP-0006.json`,
  post-Amendment-001): `8a25cd096a921c209f4741676adabe02d68d2b644a3f9811d7f5868332ccfb3c`
- Both independently pass `ExperimentSpec.verify_integrity()`.
- `spec.amendments` on the twin now has exactly 1 entry: `field_name=
  "isolation_requirements"`, full old/new values, `reason`, `approved_by`, timestamp —
  the complete Phase F amendment record, never a silent overwrite.

## DB history visibility

`research/db.py`'s `experiment_events` table for `EXP-0006` now shows, in order:
`None -> QUEUED`, `QUEUED -> BLOCKED` (registration, prior task), and a new
`amend:isolation_requirements (...)='<old>' -> amend:isolation_requirements (...)='<pointer to this report>'`
event (this task) — confirmed by direct query. `execution_status` remains `BLOCKED`,
`research_verdict` remains `PENDING`, unchanged by the amendment. This reused
`OmniLabDB._log_amendment()` — the exact internal primitive `update_fields(
allow_amendment=True)` already uses for this purpose — directly, since EXP-0006's
`execution_status` is `BLOCKED`, not `COMPLETED`, so `update_fields()`'s stricter
already-finalized-record gate does not apply here; this amendment concerns a
still-open, not-yet-executable spec, which is exactly the class of record that gate was
never meant to lock down.

## Guard behavior confirmed

`research/amend_exp_0006_001.py::apply_amendment()` checks the OLD sentence is present
**verbatim** before touching anything. Running it again now (already amended) correctly
raises `RuntimeError` rather than double-amending or silently no-op'ing — tested
(`tests/test_amend_exp_0006_001.py::TestGuardAgainstBlindReapplication`).

## Approvals / execution status — unchanged

All 7 approval flags remain `False`. `execution_status=BLOCKED`,
`research_verdict=PENDING`. No training, no data collection, no execution occurred.
