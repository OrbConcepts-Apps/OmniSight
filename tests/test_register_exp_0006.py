"""EXP-0006 registration tests (research/register_exp_0006.py). No live LLM
calls, no real training, no data collection, no GPU/device work.

Two kinds of coverage here:
  - "live" tests read the ALREADY-registered production EXP-0006 row
    (research.register_exp_0006.register_exp_0006() was run once, for
    real, as part of this authorized task) and prove real invariants about
    it -- DB unchanged elsewhere, hash preserved, approvals false, etc.
  - "isolated" tests re-run validate() against a scratch OmniLabDB/MemoryDB
    seeded with just enough fixture data to re-prove the PROPOSAL content
    itself validates cleanly independent of whether EXP-0006 happens to
    already be registered in this environment (avoids the DUPLICATE_ID
    self-collision a live re-validation would otherwise show)."""

from __future__ import annotations

import pytest

from research.db import Experiment, OmniLabDB
from research.execution_budget import ExecutionBudgetError
from research.execution_job.manager import JobManager, LaunchPreconditions
from research.execution_job.runner import FakeRunner
from research.experiment_spec import FrozenProposalTamperedError
from research.experiment_validator import is_queue_eligible, validate
from research.memory_db import MemoryDB, MemoryRecord
from research.preregistration import build_exp0006_proposal, load_preregistration


def _seeded_isolated_dbs(tmp_path):
    """A scratch OmniLabDB (EXP-0004/0005 stubs only, no EXP-0006) and a
    scratch MemoryDB (MEM-0003/0015/0017/0025 stubs), so validate() can be
    re-run against the proposal's actual references without touching the
    real production databases."""
    db = OmniLabDB(db_path=tmp_path / "omnilab.db")
    for exp_id in ("EXP-0004", "EXP-0005"):
        db.create_experiment(
            Experiment(
                experiment_id=exp_id, hypothesis="stub", motivation="stub",
                rationale="stub", independent_variable="stub",
            )
        )
    memory_db = MemoryDB(db_path=tmp_path / "memory.db")
    for mem_id in ("MEM-0003", "MEM-0015", "MEM-0017", "MEM-0025"):
        memory_db.insert(MemoryRecord(record_id=mem_id, claim="stub", tag="VERIFIED", experiment_id="EXP-0004"))
    return db, memory_db


class TestIsolatedProposalValidation:
    """Re-proves the proposal content's own quality, independent of
    registration state."""

    def test_zero_errors_when_not_yet_registered(self, tmp_path):
        db, memory_db = _seeded_isolated_dbs(tmp_path)
        try:
            spec = build_exp0006_proposal()
            from research.experiment_spec import ExperimentSpec

            result = validate(ExperimentSpec(proposal=spec), db=db, memory_db=memory_db)
            assert result.errors == []
        finally:
            db.close()
            memory_db.close()

    def test_downstream_mac_iphone_approval_does_not_block_registration(self, tmp_path):
        """mac_iphone_required=True produces a WARNING + NEEDS_HUMAN_APPROVAL
        (device validation is downstream), never an ERROR -- confirmed
        against an isolated DB where EXP-0006 is not yet registered, so
        this is purely about the mac/iphone semantics, not a DUPLICATE_ID
        side effect."""
        db, memory_db = _seeded_isolated_dbs(tmp_path)
        try:
            spec = build_exp0006_proposal()
            from research.experiment_spec import ExperimentSpec

            result = validate(ExperimentSpec(proposal=spec), db=db, memory_db=memory_db)
            assert not any(
                i.code == "MAC_IPHONE_REQUIRED_MISMATCH" for i in result.errors
            )
            assert any(i.code == "UNAPPROVED_MAC_IPHONE_DEPLOYMENT" for i in result.needs_human_approval)
        finally:
            db.close()
            memory_db.close()


class TestLiveRegistration:
    """Reads the real, already-registered production EXP-0006 row."""

    def test_exp_0006_registered(self):
        with OmniLabDB() as db:
            exp = db.get_experiment("EXP-0006")
        assert exp.experiment_family == "training_data"

    def test_db_contains_at_least_the_first_six_ids(self):
        """Was `ids == {EXP-0001..EXP-0006}` (exact equality) when this
        invariant was written, before EXP-0007 (an orthogonal, non-training,
        non-private-data experiment; research/_exp0007_preregister.py) was
        legitimately registered. Relaxed to a subset check rather than
        hardcoding an ever-growing exact set, since further legitimately
        registered experiment ids are expected as the lab continues."""
        with OmniLabDB() as db:
            ids = {e.experiment_id for e in db.list_experiments()}
        assert {"EXP-0001", "EXP-0002", "EXP-0003", "EXP-0004", "EXP-0005", "EXP-0006"}.issubset(ids)

    def test_exp_0001_through_0005_unchanged(self):
        with OmniLabDB() as db:
            assert db.get_experiment("EXP-0001").research_verdict == "PASS"
            assert db.get_experiment("EXP-0002").research_verdict == "FAIL"
            assert db.get_experiment("EXP-0003").research_verdict == "FAIL"
            assert db.get_experiment("EXP-0004").research_verdict == "INCONCLUSIVE"
            assert db.get_experiment("EXP-0005").research_verdict == "INCONCLUSIVE"

    def test_execution_status_is_blocked_not_queued_or_completed(self):
        with OmniLabDB() as db:
            exp = db.get_experiment("EXP-0006")
        assert exp.execution_status == "BLOCKED"
        assert exp.research_verdict == "PENDING"

    def test_frozen_hash_traceable_to_preregistration(self):
        """As registered (before EXP-0006 Amendment 001), the twin's hash
        matched the original preregistration exactly. Amendment 001
        (see tests/test_amend_exp_0006_001.py) intentionally changed the
        twin's hash while leaving the preregistration untouched -- both
        specs must still independently verify, and the preregistration's
        hash remains the recorded ORIGINAL hash for history."""
        from research.backfill_experiment_specs import load_spec

        twin = load_spec("EXP-0006")
        preregistration = load_preregistration()
        twin.verify_integrity()
        preregistration.verify_integrity()
        assert preregistration.frozen_hash == "e0b4a954e6a6ffb1c3bd067d8a10363dafe236a3c386d51949300a79a806289a"
        assert len(twin.amendments) >= 1  # at least Amendment 001

    def test_all_approval_flags_false_on_twin_spec(self):
        from research.backfill_experiment_specs import load_spec

        p = load_spec("EXP-0006").proposal
        assert p.new_training_approved is False
        assert p.private_user_data_use_approved is False
        assert p.mac_iphone_deployment_approved is False
        assert p.coreml_model_replacement_approved is False
        assert p.external_upload_approved is False
        assert p.production_swift_modification_approved is False
        assert p.signing_distribution_change_approved is False

    def test_registration_does_not_imply_queue_eligibility(self):
        from research.backfill_experiment_specs import load_spec

        result = validate(load_spec("EXP-0006"))
        assert is_queue_eligible(result) is False

    def test_missing_training_approval_blocks(self):
        from research.backfill_experiment_specs import load_spec

        result = validate(load_spec("EXP-0006"))
        assert any(i.code == "UNAPPROVED_NEW_TRAINING" for i in result.needs_human_approval)

    def test_missing_private_data_approval_blocks(self):
        from research.backfill_experiment_specs import load_spec

        result = validate(load_spec("EXP-0006"))
        assert any(i.code == "UNAPPROVED_PRIVATE_DATA_USE" for i in result.needs_human_approval)

    def test_dataset_prerequisite_honestly_unresolved(self):
        from research.backfill_experiment_specs import load_spec

        p = load_spec("EXP-0006").proposal
        assert "PREREQUISITE" in p.dataset_version

    def test_checkpoint_provenance_honestly_unresolved(self):
        from research.backfill_experiment_specs import load_spec

        p = load_spec("EXP-0006").proposal
        assert "UNKNOWN/PREREQUISITE" in p.isolation_requirements

    def test_missing_real_runner_blocks_execution(self):
        """No Runner is invoked -- execution_budget refuses before
        FakeRunner.prepare()/launch() is ever called."""
        runner = FakeRunner()
        mgr = JobManager(runner)
        with pytest.raises(ExecutionBudgetError):
            mgr.launch(
                experiment_id="EXP-0006", spec={}, budget_config=None,
                preconditions=LaunchPreconditions(require_clean_git_tree=False),
            )
        assert not runner._prepared
        assert not runner._launched

    def test_immutable_registered_spec_detects_tampering(self):
        from research.backfill_experiment_specs import load_spec
        from dataclasses import replace

        spec = load_spec("EXP-0006")
        spec.proposal = replace(spec.proposal, new_training_approved=True)  # simulate tampering
        with pytest.raises(FrozenProposalTamperedError):
            spec.verify_integrity()

    def test_amendment_path_still_works(self):
        """The established amend() mechanism (not a silent edit) still
        functions on the registered spec -- proving registration didn't
        freeze the spec into something un-amendable, only un-silently-editable."""
        from research.backfill_experiment_specs import load_spec

        spec = load_spec("EXP-0006")
        original_hash = spec.frozen_hash
        amendment = spec.amend(
            "reproducibility_requirements",
            "TEST AMENDMENT -- not a real change, exercised by test_register_exp_0006.py only",
            reason="test: prove amendment path works", approved_by="test",
        )
        assert amendment.field_name == "reproducibility_requirements"
        assert spec.frozen_hash != original_hash
        spec.verify_integrity()  # re-frozen at the new hash, must not raise

    def test_exp_0007_is_orthogonal_not_training_data(self):
        """Was `test_no_exp_0007_exists` -- EXP-0007 was subsequently,
        legitimately registered (research/_exp0007_preregister.py), an
        orthogonal per-class-threshold diagnostic requiring no training, no
        private data, no device deployment, no new human approval. This
        test's real invariant is that EXP-0007 is NOT the training_data
        family EXP-0006 belongs to -- registering it never silently expanded
        EXP-0006's own scope or approvals."""
        with OmniLabDB() as db:
            exp = db.get_experiment("EXP-0007")
        assert exp.experiment_family != "training_data"

    def test_candidate_0003_unchanged(self):
        import subprocess

        from research.config import REPO_ROOT

        out = subprocess.run(
            ["git", "status", "--short", "research/candidates/"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout
        assert "M " not in out  # only untracked ("??"), never modified
