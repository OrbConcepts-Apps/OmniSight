"""One-shot registration of EXP-0006 from its frozen preregistration.

Run via:

    uv run python -m research.register_exp_0006

Reads the ALREADY-FROZEN, ALREADY-AUDITED spec at
research/preregistrations/EXP-0006-PROPOSED.json (built by
research/preregistration.py, accepted by
reports/phase_i/EXP0006_PREREGISTRATION_AUDIT.md) and performs the
"normal canonical registration path" this codebase already established for
EXP-0001..0005 (research/seed_experiments.py + research/_exp0004_preregister.py
/ research/_exp0005_preregister.py's pattern): a `research.db.Experiment`
row (execution/business-tracking layer) plus a twin
`research/experiment_specs/EXP-0006.json` (`ExperimentSpec`, the full
Phase F scientific-content layer). This mirrors the established two-layer
split documented in `research/experiment_spec.py`'s module docstring:
`research.db.Experiment` was never meant to carry approval flags, evidence
references, structured multi-seed criteria, etc. -- that full-fidelity
content is what `ExperimentSpec`/`research/experiment_specs/EXP-XXXX.json`
is FOR, exactly as it already is for EXP-0001..0005 (see
research/backfill_experiment_specs.py). Registering EXP-0006 the SAME way
means no field is silently dropped: everything the DB `Experiment` row
cannot represent (data_privacy_classification, all 7 approval flags,
evidence_references, acknowledges_rejected_hypothesis_ids,
supports/rejects/inconclusive_if, the structured seed/recovery-metric
definitions, etc.) lives byte-identical, hash-verified, in the twin JSON
file this script ALSO writes -- it is never rebuilt from scratch, only
copied from the frozen preregistration, so the frozen_hash relationship is
preserved exactly.

Immediately after creation, this script transitions execution_status
QUEUED -> BLOCKED (same idiom as EXP-0005's own registration -- see
research/seed_experiments.py's `db.transition_status("EXP-0005", "BLOCKED",
...)` call) with a note naming every real, current blocker. BLOCKED is the
correct EXISTING state for "registered scientific experiment, not yet
authorized/executable" -- no new status value is introduced.

Never grants any approval flag. Never queues (BLOCKED, not QUEUED, is the
terminal state this script leaves the row in). Never touches
CANDIDATE-0003 or research/preregistrations/EXP-0006-PROPOSED.json (loaded
read-only)."""

from __future__ import annotations

from research.db import Experiment, ExperimentNotFoundError, OmniLabDB
from research.experiment_lifecycle import move_to_status
from research.experiment_schema import write_queued_artifacts
from research.preregistration import PREREGISTRATIONS_DIR, load_preregistration

SPECS_DIR_NAME = "experiment_specs"

BLOCK_NOTE = (
    "Registered from the corrected, frozen preregistration "
    "(research/preregistrations/EXP-0006-PROPOSED.json, "
    "reports/phase_i/EXP0006_PREREGISTRATION_AUDIT.md). Execution remains "
    "blocked on: new_training_approved=False; private_user_data_use_approved=False "
    "(data_privacy_classification=PRIVATE_USER_DATA); dataset does not yet exist "
    "(collection/annotation prerequisite unresolved); yolov8m-oiv7.pt checkpoint "
    "provenance unresolved (source/SHA-256/license/training-history prerequisite); "
    "no real Phase J training Runner exists yet; epoch/resource estimates are "
    "uncalibrated. mac_iphone_deployment_approved is also False but is NOT a "
    "reason this offline-stage experiment is blocked -- device validation is an "
    "explicit, separate, downstream gate (see materially_new_rationale / "
    "isolation_requirements), never a prerequisite for offline registration or "
    "offline scientific eligibility."
)


class SchemaMismatchError(RuntimeError):
    """Raised if a field on the frozen ExperimentProposal cannot be
    represented on the research.db.Experiment row WITHOUT being carried,
    unchanged, in the twin research/experiment_specs/EXP-0006.json this
    same script writes. This should never actually fire in normal
    operation -- it exists so a future schema change that silently drops
    a field is caught loudly rather than shipped quietly."""


def _build_experiment_row(proposal) -> Experiment:
    """Maps the frozen ExperimentProposal onto a research.db.Experiment
    row. Every field NOT representable here (approval flags, privacy
    classification, evidence_references, acknowledges_rejected_hypothesis_ids,
    supports/rejects/inconclusive_if, the structured seed/recovery-metric
    definitions, isolation_requirements' full prose, etc.) is preserved
    losslessly in the twin ExperimentSpec JSON this script also writes --
    see this module's docstring. `success_criteria` and `configuration`
    below intentionally carry the FULL nested dicts (not just the
    DB-representable subset), since both are already free-form JSON
    columns with no narrower schema of their own."""
    controlled = dict(proposal.controlled_variables)
    training_config = controlled.pop("training_config", {})
    controls = dict(controlled)  # top-level scalars only, for methodology.md readability

    return Experiment(
        experiment_id=proposal.experiment_id,
        hypothesis=proposal.hypothesis,
        motivation=proposal.motivation,
        rationale=(
            "Corrects CANDIDATE-0003's hypothesis/metric ambiguity and causal-control gap "
            "(reports/phase_i/CANDIDATE_0003_EXP0006_ELIGIBILITY.md); see "
            "materially_new_rationale in research/experiment_specs/EXP-0006.json for the "
            "full relationship to EXP-0001-0005 and CANDIDATE-0001/0002."
        ),
        independent_variable=", ".join(proposal.independent_variables),
        controls=controls,
        evaluation_method=proposal.procedure,
        success_criteria=dict(proposal.success_criteria),
        risks=(
            "Checkpoint provenance unresolved; dataset does not yet exist; "
            "epoch/resource estimates uncalibrated; no real training Runner exists yet "
            "(reports/phase_j/PHASE_J_SAFETY_AUDIT.md). See execution_status=BLOCKED note "
            "for the complete, current blocker list."
        ),
        expected_outcome=proposal.research_question,
        experiment_family=proposal.family,
        baseline_run_id=proposal.baseline_run_id,
        model_version=proposal.model_config_ref,
        dataset_version=proposal.dataset_version,
        configuration={**controlled, "training_config": training_config},
        # mac_iphone_required=True on the proposal is a downstream-only
        # signal (this offline experiment never touches a device) -- same
        # established convention as EXP-0005's own DB row, which recorded
        # REQUIRES_MAC even though EXP-0005 executed entirely on
        # Windows/CUDA (see research/backfill_experiment_specs.py's
        # module docstring and this project's own memory notes on
        # mac_iphone_required's real semantics).
        validation_requirement="REQUIRES_IPHONE" if proposal.mac_iphone_required else "OFFLINE_SIMULATABLE",
        estimated_cost=dict(proposal.compute_resource_estimate),
    )


def register_exp_0006() -> Experiment:
    spec = load_preregistration()
    spec.verify_integrity()  # fail closed if the frozen artifact was ever tampered with
    proposal = spec.proposal

    if proposal.experiment_id != "EXP-0006":
        raise SchemaMismatchError(
            f"refusing to register: preregistration's experiment_id is "
            f"{proposal.experiment_id!r}, not 'EXP-0006'."
        )

    with OmniLabDB() as db:
        try:
            db.get_experiment("EXP-0006")
            raise SchemaMismatchError("EXP-0006 already exists in the database -- refusing to re-register.")
        except ExperimentNotFoundError:
            pass

        exp = _build_experiment_row(proposal)
        db.create_experiment(exp)
        exp = db.transition_status("EXP-0006", "BLOCKED", note=BLOCK_NOTE)

    move_to_status("EXP-0006", "BLOCKED")
    from research.experiment_lifecycle import find_current_dir

    exp_dir = find_current_dir("EXP-0006")
    if exp_dir is not None:
        write_queued_artifacts(exp, exp_dir)

    # Twin, full-fidelity Phase F spec artifact -- byte-copied content from
    # the already-frozen preregistration (never rebuilt), so its
    # frozen_hash is identical to research/preregistrations/EXP-0006-PROPOSED.json's.
    from research.config import REPO_ROOT

    specs_dir = REPO_ROOT / "research" / SPECS_DIR_NAME
    specs_dir.mkdir(parents=True, exist_ok=True)
    (specs_dir / "EXP-0006.json").write_text(spec.to_json(), encoding="utf-8")

    return exp


def main() -> None:
    exp = register_exp_0006()
    print(f"EXP-0006 registered. execution_status={exp.execution_status}, research_verdict={exp.research_verdict}")


if __name__ == "__main__":
    main()
