"""OMNISIGHT-PILOT-001 cap-enforcement ledger + sequence admission tests
(research/datasets/collection_ledger.py). No real media, all synthetic ids."""

from __future__ import annotations

import pytest

from research.datasets.collection_ledger import (
    BYSTANDER_STOP_RULE,
    PILOT_PLANNING_ONLY,
    PilotCapExceededError,
    PilotCollectionLedger,
    SequenceAdmissionRequest,
    check_sequence_admission,
)
from research.datasets.pilot_plan import OMNISIGHT_PILOT_001_PLAN_HASH, PILOT_ID


def _ledger(tmp_path):
    return PilotCollectionLedger(path=tmp_path / "ledger.json")


class TestParticipantCap:
    def test_six_participants_allowed(self, tmp_path):
        ledger = _ledger(tmp_path)
        for i in range(6):
            ledger.register_participant(f"p{i}")
        assert ledger.counts()["participants"] == 6

    def test_seventh_participant_rejected(self, tmp_path):
        ledger = _ledger(tmp_path)
        for i in range(6):
            ledger.register_participant(f"p{i}")
        with pytest.raises(PilotCapExceededError):
            ledger.register_participant("p6")

    def test_reregistering_same_participant_is_idempotent(self, tmp_path):
        ledger = _ledger(tmp_path)
        for i in range(6):
            ledger.register_participant(f"p{i}")
        ledger.register_participant("p0")  # already registered -- must not raise
        assert ledger.counts()["participants"] == 6


class TestSessionCap:
    def test_session_requires_registered_participant(self, tmp_path):
        ledger = _ledger(tmp_path)
        with pytest.raises(PilotCapExceededError):
            ledger.register_session("s0", participant_id="never-registered")

    def test_six_sessions_allowed(self, tmp_path):
        ledger = _ledger(tmp_path)
        ledger.register_participant("p0")
        for i in range(6):
            ledger.register_session(f"s{i}", participant_id="p0")
        assert ledger.counts()["sessions"] == 6

    def test_seventh_session_rejected(self, tmp_path):
        ledger = _ledger(tmp_path)
        ledger.register_participant("p0")
        for i in range(6):
            ledger.register_session(f"s{i}", participant_id="p0")
        with pytest.raises(PilotCapExceededError):
            ledger.register_session("s6", participant_id="p0")


class TestSequenceCap:
    def _seeded_ledger(self, tmp_path):
        ledger = _ledger(tmp_path)
        ledger.register_participant("p0")
        ledger.register_session("s0", participant_id="p0")
        return ledger

    def test_sequence_requires_registered_session(self, tmp_path):
        ledger = _ledger(tmp_path)
        with pytest.raises(PilotCapExceededError):
            ledger.register_sequence("seq0", session_id="never-registered")

    def test_24_sequences_allowed(self, tmp_path):
        ledger = self._seeded_ledger(tmp_path)
        for i in range(24):
            ledger.register_sequence(f"seq{i}", session_id="s0")
        assert ledger.counts()["sequences"] == 24

    def test_25th_sequence_rejected(self, tmp_path):
        ledger = self._seeded_ledger(tmp_path)
        for i in range(24):
            ledger.register_sequence(f"seq{i}", session_id="s0")
        with pytest.raises(PilotCapExceededError):
            ledger.register_sequence("seq24", session_id="s0")


class TestCorruptLedgerFailsClosed:
    def test_corrupt_file_reports_caps_exhausted(self, tmp_path):
        path = tmp_path / "ledger.json"
        path.write_text("{not valid json", encoding="utf-8")
        ledger = PilotCollectionLedger(path=path)
        with pytest.raises(PilotCapExceededError):
            ledger.register_participant("p0")


def _valid_sequence_request(**overrides):
    defaults = dict(
        pilot_id=PILOT_ID, pilot_plan_hash=OMNISIGHT_PILOT_001_PLAN_HASH,
        session_id="s0", scenario_id="normal_illumination", privacy_class="STAGED_CONSENTED",
    )
    defaults.update(overrides)
    return SequenceAdmissionRequest(**defaults)


class TestSequenceAdmission:
    def test_admitted_when_everything_valid(self):
        result = check_sequence_admission(
            _valid_sequence_request(), session_is_registered=True, cap_available=True,
        )
        assert result.admitted is True

    def test_unregistered_session_rejected(self):
        result = check_sequence_admission(
            _valid_sequence_request(), session_is_registered=False, cap_available=True,
        )
        assert result.admitted is False

    def test_cap_exhausted_rejected(self):
        result = check_sequence_admission(
            _valid_sequence_request(), session_is_registered=True, cap_available=False,
        )
        assert result.admitted is False

    def test_sensitive_context_rejected(self):
        result = check_sequence_admission(
            _valid_sequence_request(sensitive_context=True), session_is_registered=True, cap_available=True,
        )
        assert result.admitted is False

    def test_unconsented_bystander_rejected(self):
        result = check_sequence_admission(
            _valid_sequence_request(unconsented_bystander_present=True),
            session_is_registered=True, cap_available=True,
        )
        assert result.admitted is False
        assert any("bystander" in b for b in result.blockers)

    def test_wrong_pilot_id_rejected(self):
        result = check_sequence_admission(
            _valid_sequence_request(pilot_id="OTHER"), session_is_registered=True, cap_available=True,
        )
        assert result.admitted is False


class TestBystanderStopRule:
    def test_rule_mentions_exclusion_not_redaction_as_fix(self):
        assert "excludes" in BYSTANDER_STOP_RULE.lower() or "exclude" in BYSTANDER_STOP_RULE.lower()
        assert "never" in BYSTANDER_STOP_RULE.lower()


class TestPilotDataStatus:
    def test_default_status_is_planning_only(self):
        assert PILOT_PLANNING_ONLY == "PILOT_PLANNING_ONLY"
