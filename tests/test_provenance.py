"""Provenance registry tests (research/provenance.py). No live web fetch,
no fabricated values -- deterministic hashing and metadata extraction only."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.config import REPO_ROOT
from research.provenance import (
    STATUS_VALUES,
    UNKNOWN,
    VERIFIED,
    ProvenanceRecord,
    ProvenanceRegistry,
    build_exp0006_checkpoint_registry,
    extract_mlmodel_strings,
    extract_pt_checkpoint_metadata,
    find_marker,
    hash_file,
)


class TestHashFile:
    def test_stable_sha256(self, tmp_path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"hello world")
        h1 = hash_file(p)
        h2 = hash_file(p)
        assert h1 == h2
        assert len(h1) == 64

    def test_different_content_different_hash(self, tmp_path):
        p1 = tmp_path / "a.bin"
        p2 = tmp_path / "b.bin"
        p1.write_bytes(b"aaaa")
        p2.write_bytes(b"bbbb")
        assert hash_file(p1) != hash_file(p2)

    def test_chunked_reading_matches_whole_file_hash(self, tmp_path):
        import hashlib

        p = tmp_path / "big.bin"
        p.write_bytes(b"x" * (5 * 1024 * 1024 + 37))  # not a clean multiple of chunk_size
        expected = hashlib.sha256(p.read_bytes()).hexdigest()
        assert hash_file(p, chunk_size=1024 * 1024) == expected


class TestProvenanceRecordValidation:
    def test_valid_statuses_accepted(self):
        for status in STATUS_VALUES:
            ProvenanceRecord(
                artifact="x", sha256="0" * 64, source="s", source_status=status,
                license="l", license_status=status, acquisition_method="m",
                verification_status=status,
            )

    def test_invalid_status_rejected(self):
        with pytest.raises(ValueError):
            ProvenanceRecord(
                artifact="x", sha256="0" * 64, source="s", source_status="MAYBE",
                license="l", license_status=UNKNOWN, acquisition_method="m",
            )


class TestProvenanceRegistry:
    def test_save_and_reload_json(self, tmp_path):
        registry = ProvenanceRegistry()
        registry.add(ProvenanceRecord(
            artifact="a.pt", sha256="0" * 64, source="s", source_status=UNKNOWN,
            license="l", license_status=UNKNOWN, acquisition_method="m",
        ))
        path = tmp_path / "registry.json"
        registry.save(path)
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data[0]["artifact"] == "a.pt"


class TestMlmodelStringExtraction:
    def test_extracts_embedded_printable_strings(self, tmp_path):
        p = tmp_path / "fake.mlmodel"
        p.write_bytes(b"\x00\x00" + b"hello world marker" + b"\x01\x02" + b"AGPL-3.0 License text")
        strings = extract_mlmodel_strings(p)
        assert find_marker(strings, contains="hello world") is not None
        assert find_marker(strings, contains="AGPL-3.0") is not None

    def test_no_marker_found_returns_none(self, tmp_path):
        p = tmp_path / "empty.mlmodel"
        p.write_bytes(b"\x00\x01\x02\x03")
        assert find_marker(extract_mlmodel_strings(p), contains="anything") is None


class TestPtCheckpointMetadataHonesty:
    def test_missing_file_returns_empty_never_raises(self, tmp_path):
        assert extract_pt_checkpoint_metadata(tmp_path / "does_not_exist.pt") == {}

    def test_non_checkpoint_file_returns_empty_not_fabricated(self, tmp_path):
        p = tmp_path / "not_a_checkpoint.pt"
        p.write_bytes(b"not a real torch checkpoint")
        assert extract_pt_checkpoint_metadata(p) == {}


class TestExp0006CheckpointRegistry:
    def test_registry_builds_against_real_repo_files(self):
        """Runs against the ACTUAL local repo artifacts -- no network, no
        fabrication. If the files aren't present in this environment, the
        registry is simply shorter (never fabricated placeholders)."""
        registry = build_exp0006_checkpoint_registry(REPO_ROOT)
        for record in registry.records:
            assert len(record.sha256) == 64
            assert record.verification_status in STATUS_VALUES
            assert record.source_status in STATUS_VALUES
            assert record.license_status in STATUS_VALUES

    def test_pt_checkpoint_never_fabricates_from_filename(self):
        """The registry's yolov8m-oiv7.pt record must derive its metadata
        from the checkpoint's OWN embedded dict, not from the filename --
        confirmed by checking the notes field cites real embedded keys."""
        registry = build_exp0006_checkpoint_registry(REPO_ROOT)
        pt_records = [r for r in registry.records if r.artifact.endswith("yolov8m-oiv7.pt")]
        if not pt_records:
            pytest.skip("benchmark/models/yolov8m-oiv7.pt not present in this environment")
        record = pt_records[0]
        assert "train_args" in record.notes or record.verification_status == UNKNOWN
