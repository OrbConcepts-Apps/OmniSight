"""Future OmniSight-domain dataset manifest schema tests
(research/datasets/manifest_schema.py). No real collected data anywhere in
this file -- every record below is a synthetic fixture."""

from __future__ import annotations

import pytest

from research.datasets.manifest_schema import (
    MediaUnitRecord,
    check_exact_overlap_with_frozen_eval,
    contains_prohibited_path,
    dataset_version_hash,
    detect_session_split_leakage,
    flag_near_duplicates,
    hamming_distance_hex,
    validate_media_unit,
)

VALID_SHA = "a" * 64


def _record(**overrides):
    defaults = dict(
        dataset_version="omnisight_v20260101_annotated",
        session_id="session-001",
        sequence_id="seq-001",
        frame_id="frame-001",
        source_type="omnisight-staged-capture",
        sha256=VALID_SHA,
        consent_status="STAGED_CONSENTED",
        privacy_class="STAGED_CONSENTED",
        split="train",
    )
    defaults.update(overrides)
    return MediaUnitRecord(**defaults)


class TestMediaUnitValidation:
    def test_valid_record_has_no_issues(self):
        assert validate_media_unit(_record()) == []

    def test_invalid_privacy_class_rejected(self):
        issues = validate_media_unit(_record(privacy_class="NOT_A_REAL_CLASS"))
        assert any("privacy_class" in i for i in issues)

    def test_invalid_split_rejected(self):
        issues = validate_media_unit(_record(split="somewhere_else"))
        assert any("split" in i for i in issues)

    def test_prohibited_consent_status_must_be_excluded(self):
        issues = validate_media_unit(_record(consent_status="PROHIBITED", split="train"))
        assert any("PROHIBITED" in i for i in issues)
        assert validate_media_unit(_record(consent_status="PROHIBITED", split="excluded")) == []

    def test_overlap_flag_requires_exclusion(self):
        issues = validate_media_unit(_record(baseline_eval_overlap_status="EXACT_MATCH", split="train"))
        assert any("baseline_eval_overlap_status" in i for i in issues)
        assert validate_media_unit(_record(baseline_eval_overlap_status="EXACT_MATCH", split="excluded")) == []

    def test_short_sha256_rejected(self):
        issues = validate_media_unit(_record(sha256="deadbeef"))
        assert any("sha256" in i for i in issues)

    def test_prohibited_absolute_path_in_field_rejected(self):
        issues = validate_media_unit(_record(device=r"C:\Users\armaa\camera1"))
        assert any("prohibited absolute personal path" in i for i in issues)

    def test_secret_looking_annotator_id_rejected(self):
        issues = validate_media_unit(_record(annotator_ids=("api_key_12345",)))
        assert any("looks like a secret" in i for i in issues)

    def test_pseudonymous_annotator_id_accepted(self):
        assert validate_media_unit(_record(annotator_ids=("annotator_007",))) == []


class TestSessionSplitLeakage:
    def test_no_leakage_when_sessions_disjoint(self):
        records = [
            _record(session_id="s1", split="train"),
            _record(session_id="s2", split="val"),
        ]
        assert detect_session_split_leakage(records) == []

    def test_leakage_detected_when_session_spans_splits(self):
        records = [
            _record(session_id="s1", frame_id="f1", split="train"),
            _record(session_id="s1", frame_id="f2", split="val"),
        ]
        assert detect_session_split_leakage(records) == ["s1"]

    def test_excluded_frames_never_count_as_leakage(self):
        records = [
            _record(session_id="s1", frame_id="f1", split="train"),
            _record(session_id="s1", frame_id="f2", split="excluded"),
        ]
        assert detect_session_split_leakage(records) == []


class TestFrozenEvalIsolation:
    def test_exact_overlap_detected(self):
        frozen = {"hash-a", "hash-b"}
        candidates = {"new-1": "hash-a", "new-2": "hash-c"}
        overlap = check_exact_overlap_with_frozen_eval(candidates, frozen)
        assert overlap == {"new-1": "hash-a"}

    def test_no_overlap_when_disjoint(self):
        overlap = check_exact_overlap_with_frozen_eval({"new-1": "hash-x"}, {"hash-a"})
        assert overlap == {}


class TestNearDuplicateFlagging:
    def test_hamming_distance_identical_is_zero(self):
        assert hamming_distance_hex("ff00", "ff00") == 0

    def test_hamming_distance_counts_differing_bits(self):
        assert hamming_distance_hex("0000", "0001") == 1  # last hex digit differs by 1 bit
        assert hamming_distance_hex("0000", "000f") == 4

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            hamming_distance_hex("ff", "ffff")

    def test_flags_within_threshold(self):
        flags = flag_near_duplicates({"c1": "0000"}, {"r1": "0001"}, threshold=1)
        assert flags == [("c1", "r1", 1)]

    def test_no_flag_beyond_threshold(self):
        flags = flag_near_duplicates({"c1": "0000"}, {"r1": "000f"}, threshold=1)
        assert flags == []

    def test_never_auto_excludes_only_flags(self):
        """This module's contract: flagging is advisory, never a silent
        auto-decision -- confirmed by the function's return type (a list
        of candidates for review, not a filtered/mutated dataset)."""
        flags = flag_near_duplicates({"c1": "0000"}, {"r1": "0000"}, threshold=0)
        assert isinstance(flags, list)
        assert flags[0][:2] == ("c1", "r1")


class TestDatasetVersionHash:
    def test_stable_for_same_inputs(self):
        kwargs = dict(
            manifest_hash="m1", annotation_guideline_hash="g1",
            split_assignment_hash="s1", provenance_summary_hash="p1",
        )
        assert dataset_version_hash(**kwargs) == dataset_version_hash(**kwargs)

    def test_different_for_changed_input(self):
        base = dict(
            manifest_hash="m1", annotation_guideline_hash="g1",
            split_assignment_hash="s1", provenance_summary_hash="p1",
        )
        changed = dict(base, manifest_hash="m2")
        assert dataset_version_hash(**base) != dataset_version_hash(**changed)

    def test_changing_data_after_freeze_implies_new_version_never_same_hash(self):
        v1 = dataset_version_hash(manifest_hash="m1", annotation_guideline_hash="g1", split_assignment_hash="s1", provenance_summary_hash="p1")
        v2 = dataset_version_hash(manifest_hash="m1-corrected", annotation_guideline_hash="g1", split_assignment_hash="s1", provenance_summary_hash="p1")
        assert v1 != v2


class TestProhibitedPathDetection:
    def test_windows_user_path_detected(self):
        assert contains_prohibited_path(r"C:\Users\armaa\Desktop\photo.jpg") is True

    def test_relative_logical_path_allowed(self):
        assert contains_prohibited_path("session-001/frame-042.jpg") is False
