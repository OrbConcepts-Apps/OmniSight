"""EXP-0006 Amendment 001 -- corrects the ambiguous IoU annotation-QA
sentence in the REGISTERED twin spec (research/experiment_specs/EXP-0006.json)
via the existing, already-tested ExperimentSpec.amend() mechanism.

Run via:

    uv run python -m research.amend_exp_0006_001

Does NOT touch research/preregistrations/EXP-0006-PROPOSED.json (the
original, immutable historical preregistration record) -- only the
registered twin, which is the canonical record amendments apply to (same
distinction established at registration time, see
research/register_exp_0006.py's module docstring). Does NOT change any
scientific content beyond the one sentence identified in
reports/phase_i/EXP0006_DATASET_PROTOCOL.md's Amendment Candidate 1.
"""

from __future__ import annotations

from research.backfill_experiment_specs import SPECS_DIR, load_spec

OLD_QA_SENTENCE = (
    "inter-annotator bounding-box IoU agreement target >= 0.7 on >= 95% of boxes "
    "before a batch is accepted"
)

NEW_QA_SENTENCE = (
    "annotator-pair bounding-box agreement rule (PREREGISTERED PILOT QA THRESHOLD, "
    "not yet evidence-backed -- see EXP-0006 Amendment 001): match annotator A's and "
    "B's Person boxes via greedy IoU assignment (IoU = intersection-over-union of the "
    "two boxes' areas; ignore/ambiguous regions per the annotation ontology are "
    "excluded from this metric entirely, on both sides); per QA-sampled batch, three "
    "separate statistics apply: (a) >=95% of MATCHED pairs must achieve IoU>=0.7 "
    "(a PROVISIONAL threshold the pilot tests for achievability, not an already-"
    "observed performance fact); (b) unmatched-box rate (a box with no counterpart "
    "at IoU>=0.1, counted as a miss, never silently folded into (a)'s IoU statistic) "
    "must be <=5% of all boxes across both annotators; (c) no more than 5% of images "
    "in the batch may fall below an 80% per-image matched-pair agreement rate. The "
    "first full batch of any new session is double-annotated at 100% before any "
    "agreement rate is computed. If any of (a)/(b)/(c) is missed, the ENTIRE batch "
    "is held at annotation_status=FIRST_PASS, fully re-annotated or adjudicated, and "
    "the guideline is reviewed for ambiguity before any further batch -- never "
    "silently accepted at a lower bar, and thresholds are never retuned post-hoc "
    "without a further explicit amendment"
)

AMENDMENT_REASON = (
    "EXP-0006 Amendment 001, per reports/phase_i/EXP0006_DATASET_PROTOCOL.md section "
    "19's audit (Amendment Candidate 1): the original wording ('IoU >= 0.7 on >= 95% "
    "of boxes') did not specify box correspondence between independent annotators, "
    "unmatched-box treatment, ignore-region handling, or whether the rule applies "
    "per-image or only in aggregate -- all four ambiguities are closed by this "
    "amendment, without changing the underlying 0.70/95% numbers themselves (marked "
    "provisional/pilot-tested, not evidence-backed, per the same audit's section 3 "
    "instruction not to preserve the numbers as if they were established facts)."
)

APPROVED_BY = "human-authorized preregistration amendment (EXP-0006 Amendment 001)"


def apply_amendment():
    spec = load_spec("EXP-0006")
    spec.verify_integrity()
    original_hash = spec.frozen_hash

    current_text = spec.proposal.isolation_requirements
    if OLD_QA_SENTENCE not in current_text:
        raise RuntimeError(
            "EXP-0006's isolation_requirements no longer contains the expected "
            "ambiguous QA sentence verbatim -- refusing to apply Amendment 001 "
            "blindly (the spec may have already been amended, or drifted "
            "unexpectedly). No change made."
        )
    new_text = current_text.replace(OLD_QA_SENTENCE, NEW_QA_SENTENCE)

    amendment = spec.amend(
        "isolation_requirements", new_text, reason=AMENDMENT_REASON, approved_by=APPROVED_BY,
    )
    spec.verify_integrity()

    out_path = SPECS_DIR / "EXP-0006.json"
    out_path.write_text(spec.to_json(), encoding="utf-8")

    # DB audit-trail visibility (research/db.py's experiment_events table).
    # EXP-0006's execution_status is BLOCKED, not COMPLETED, so
    # OmniLabDB.update_fields()'s stricter allow_amendment gate (which
    # exists to protect an already-FINALIZED scientific record) does not
    # apply here -- this amendment is to a still-open, not-yet-executable
    # spec. Reusing the same internal audit-log primitive
    # (OmniLabDB._log_amendment) that gate uses for exactly this purpose
    # (an 'amend:field=old' -> 'amend:field=new' event pair), so the DB's
    # own event history shows the amendment happened, without touching any
    # DB column (isolation_requirements text lives only in the spec twin,
    # never duplicated into the Experiment row).
    from research.db import OmniLabDB

    with OmniLabDB() as db:
        db.get_experiment("EXP-0006")  # confirms it still exists, fails loudly otherwise
        from datetime import datetime, timezone

        db._log_amendment(
            "EXP-0006", "isolation_requirements (spec twin, IoU-QA sentence)",
            OLD_QA_SENTENCE, "see reports/phase_i/EXP0006_AMENDMENT_001_REPORT.md",
            AMENDMENT_REASON, datetime.now(timezone.utc).isoformat(),
        )
        db._conn.commit()

    return original_hash, spec.frozen_hash, amendment


def main() -> None:
    old_hash, new_hash, amendment = apply_amendment()
    print(f"EXP-0006 Amendment 001 applied.\nold_hash={old_hash}\nnew_hash={new_hash}")


if __name__ == "__main__":
    main()
