"""EXP-0006 Amendment 002 -- grants staged_pilot_collection_approved=True,
scoped exclusively to OMNISIGHT-PILOT-001's frozen plan/caps, via the
existing ExperimentSpec.amend() mechanism.

Run via:

    uv run python -m research.amend_exp_0006_002

Sets ONLY staged_pilot_collection_approved. Does not touch any other
approval flag. Does not touch research/preregistrations/EXP-0006-PROPOSED.json
(the original, immutable historical preregistration).
"""

from __future__ import annotations

from research.backfill_experiment_specs import SPECS_DIR, load_spec
from research.datasets.pilot_plan import OMNISIGHT_PILOT_001_PLAN_HASH, PILOT_ID

AMENDMENT_REASON = (
    "EXP-0006 Amendment 002 -- human operator authorization: 'AUTHORIZE OMNISIGHT-PILOT-001 "
    "STAGED/CONSENTED COLLECTION ONLY'. Grants staged_pilot_collection_approved=True, scoped "
    f"exclusively to pilot_id={PILOT_ID!r}, pilot_plan_hash={OMNISIGHT_PILOT_001_PLAN_HASH!r}, "
    "max_participants=6, max_sessions=6, max_sequences=24, ~20 seconds/sequence, "
    "privacy_class=STAGED_CONSENTED only. This authorization permits ONLY the bounded pilot "
    "collection defined by the frozen pilot plan -- it does not authorize actual collection "
    "during the task that applied it, and does not touch any other human-authority flag "
    "(private_user_data_use_approved, new_training_approved, mac_iphone_deployment_approved, "
    "external_upload_approved, production_swift_modification_approved, "
    "coreml_model_replacement_approved, signing_distribution_change_approved all remain "
    "exactly as they were before this amendment)."
)

APPROVED_BY = "human operator (EXP-0006 Amendment 002)"


def apply_amendment():
    spec = load_spec("EXP-0006")
    spec.verify_integrity()
    original_hash = spec.frozen_hash

    if spec.proposal.staged_pilot_collection_approved:
        raise RuntimeError(
            "staged_pilot_collection_approved is already True -- refusing to apply "
            "Amendment 002 again (no-op guard, never a silent double-amend)."
        )

    amendment = spec.amend(
        "staged_pilot_collection_approved", True, reason=AMENDMENT_REASON, approved_by=APPROVED_BY,
    )
    spec.verify_integrity()

    out_path = SPECS_DIR / "EXP-0006.json"
    out_path.write_text(spec.to_json(), encoding="utf-8")

    from datetime import datetime, timezone

    from research.db import OmniLabDB

    with OmniLabDB() as db:
        db.get_experiment("EXP-0006")
        db._log_amendment(
            "EXP-0006", "staged_pilot_collection_approved",
            False, True, AMENDMENT_REASON, datetime.now(timezone.utc).isoformat(),
        )
        db._conn.commit()

    return original_hash, spec.frozen_hash, amendment


def main() -> None:
    old_hash, new_hash, amendment = apply_amendment()
    print(f"EXP-0006 Amendment 002 applied.\nold_hash={old_hash}\nnew_hash={new_hash}")


if __name__ == "__main__":
    main()
