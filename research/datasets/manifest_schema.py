"""Future OmniSight-domain dataset manifest schema + deterministic
validators (EXP-0006 preparation, reports/phase_i/EXP0006_DATASET_PROTOCOL.md).

No real dataset is collected, staged, or referenced here. This module
defines the CONTRACT a future manifest must satisfy -- schema, leakage
checks, frozen-eval isolation, privacy-class validation, immutable
versioning -- entirely independent of any actual collected media.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

PRIVACY_CLASSES = (
    "STAGED_CONSENTED",         # explicitly consented participants, staged capture
    "INCIDENTAL_CONSENTED",     # incidental appearance, separately consented (e.g. posted-signage pilot)
    "PUBLIC_LICENSED",          # public/licensed source (e.g. an existing open dataset)
    "SYNTHETIC_NONIDENTIFIABLE",# synthetic/staged, no real identifiable person
    "RESTRICTED_PRIVATE",       # requires explicit human approval before any use
)

CONSENT_STATUSES = ("STAGED_CONSENTED", "INCIDENTAL_CONSENTED", "PROHIBITED")

SPLITS = ("train", "val", "test", "excluded")

ANNOTATION_STATUSES = ("UNANNOTATED", "FIRST_PASS", "ADJUDICATED", "REJECTED")

OVERLAP_STATUSES = ("NONE_DETECTED", "EXACT_MATCH", "NEAR_DUPLICATE_FLAGGED")

# A record must never contain an absolute personal filesystem path -- only
# logical/relative identifiers are trackable (Phase authorization section 26).
_PROHIBITED_PATH_RE = re.compile(r"[A-Za-z]:\\Users\\|/home/|/Users/")
_PROHIBITED_KEY_HINTS = ("api_key", "apikey", "secret", "token", "password")


class ManifestValidationError(ValueError):
    pass


@dataclass(frozen=True)
class MediaUnitRecord:
    """One future media unit's manifest entry. No field here is ever
    populated with real collected data by this module -- it is a schema
    only, exercised in tests with synthetic fixture values."""

    dataset_version: str
    session_id: str
    sequence_id: str
    frame_id: str
    source_type: str  # e.g. "omnisight-staged-capture"
    sha256: str
    perceptual_hash: Optional[str] = None
    capture_timestamp_policy: str = "DATE_ONLY"  # DATE_ONLY | NONE | FULL -- a policy, not raw EXIF
    environment_domain: str = ""  # e.g. "indoor_low_light"
    device: str = ""
    camera_configuration: Optional[str] = None
    annotation_status: str = "UNANNOTATED"
    annotator_ids: tuple = field(default_factory=tuple)  # pseudonymous ids only
    adjudication_status: str = "NONE"
    consent_status: str = "PROHIBITED"
    privacy_class: str = "RESTRICTED_PRIVATE"
    licensing_status: str = "UNKNOWN"
    split: str = "excluded"
    exclusion_reason: Optional[str] = None
    baseline_eval_overlap_status: str = "NONE_DETECTED"

    def to_dict(self) -> dict:
        return asdict(self)


def validate_media_unit(record: MediaUnitRecord) -> list:
    """Returns a list of issue strings (empty = valid). Never raises for a
    single bad field -- callers decide whether to reject or just log."""
    issues = []
    if record.privacy_class not in PRIVACY_CLASSES:
        issues.append(f"privacy_class {record.privacy_class!r} not in {PRIVACY_CLASSES}")
    if record.consent_status not in CONSENT_STATUSES:
        issues.append(f"consent_status {record.consent_status!r} not in {CONSENT_STATUSES}")
    if record.split not in SPLITS:
        issues.append(f"split {record.split!r} not in {SPLITS}")
    if record.annotation_status not in ANNOTATION_STATUSES:
        issues.append(f"annotation_status {record.annotation_status!r} not in {ANNOTATION_STATUSES}")
    if record.baseline_eval_overlap_status not in OVERLAP_STATUSES:
        issues.append(f"baseline_eval_overlap_status {record.baseline_eval_overlap_status!r} not in {OVERLAP_STATUSES}")
    if len(record.sha256) != 64:
        issues.append(f"sha256 must be a 64-hex-char string, got {record.sha256!r}")
    if record.consent_status == "PROHIBITED" and record.split != "excluded":
        issues.append("consent_status=PROHIBITED must never be assigned any split other than 'excluded'")
    if record.baseline_eval_overlap_status != "NONE_DETECTED" and record.split != "excluded":
        issues.append(
            f"baseline_eval_overlap_status={record.baseline_eval_overlap_status!r} requires split='excluded' "
            "(fail-closed default pending human review, per EXP-0006 dataset protocol section 10)"
        )
    for value in (record.session_id, record.sequence_id, record.frame_id, record.device, record.environment_domain):
        if _PROHIBITED_PATH_RE.search(value or ""):
            issues.append(f"field contains a prohibited absolute personal path: {value!r}")
    for annotator_id in record.annotator_ids:
        if any(hint in annotator_id.lower() for hint in _PROHIBITED_KEY_HINTS):
            issues.append(f"annotator_id looks like a secret, not a pseudonymous id: {annotator_id!r}")
    return issues


# ---------------------------------------------------------------------------
# Session/sequence-aware split-leakage detection (section 9)
# ---------------------------------------------------------------------------


def detect_session_split_leakage(records: list) -> list:
    """Returns a list of session_ids that appear in more than one of
    {train, val, test} (excluded may coexist with anything -- excluding a
    frame is never leakage). Near-adjacent frames from one video/session
    must never be split across train and evaluation."""
    session_splits: dict = {}
    for r in records:
        if r.split == "excluded":
            continue
        session_splits.setdefault(r.session_id, set()).add(r.split)
    return [sid for sid, splits in session_splits.items() if len(splits) > 1]


# ---------------------------------------------------------------------------
# Frozen-evaluation-set isolation (section 10)
# ---------------------------------------------------------------------------


def check_exact_overlap_with_frozen_eval(candidate_hashes: dict, frozen_eval_hashes: set) -> dict:
    """candidate_hashes: {record_id: sha256}. Returns {record_id: sha256}
    for every exact SHA-256 match against the frozen evaluation set --
    default handling (per section 10) is EXCLUSION, never silent inclusion."""
    return {rid: h for rid, h in candidate_hashes.items() if h in frozen_eval_hashes}


def hamming_distance_hex(a: str, b: str) -> int:
    """Hamming distance between two equal-length hex-encoded bit strings
    (e.g. perceptual hashes). Deterministic, dependency-free -- does not
    itself compute a perceptual hash from an image (no images exist to
    hash in this task); it is the comparison PRIMITIVE a future pipeline
    would use once real perceptual hashes exist."""
    if len(a) != len(b):
        raise ValueError(f"hamming_distance_hex requires equal-length hex strings, got {len(a)} vs {len(b)}")
    int_a, int_b = int(a, 16), int(b, 16)
    return bin(int_a ^ int_b).count("1")


def flag_near_duplicates(candidate_phashes: dict, reference_phashes: dict, threshold: int) -> list:
    """Returns [(candidate_id, reference_id, distance), ...] for every pair
    whose Hamming distance is <= threshold. NEVER auto-excludes -- this is
    explicitly a candidate-flagging function for human/manual review, per
    section 22: perceptual hashing is not perfect semantic-duplicate
    detection, and `threshold` is itself a preregistered parameter that
    requires future validation, not a value this function chooses."""
    flags = []
    for cid, cphash in candidate_phashes.items():
        for rid, rphash in reference_phashes.items():
            distance = hamming_distance_hex(cphash, rphash)
            if distance <= threshold:
                flags.append((cid, rid, distance))
    return flags


# ---------------------------------------------------------------------------
# Immutable dataset versioning (section 23)
# ---------------------------------------------------------------------------


def dataset_version_hash(*, manifest_hash: str, annotation_guideline_hash: str, split_assignment_hash: str, provenance_summary_hash: str) -> str:
    """Combines the four component hashes into one canonical version hash.
    Changing ANY input after freeze must produce a NEW dataset version --
    never silently alters the old one (this function has no persistence of
    its own; callers are responsible for never overwriting a prior
    version's stored hash)."""
    payload = json.dumps(
        {
            "manifest_hash": manifest_hash,
            "annotation_guideline_hash": annotation_guideline_hash,
            "split_assignment_hash": split_assignment_hash,
            "provenance_summary_hash": provenance_summary_hash,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def contains_prohibited_path(value: str) -> bool:
    return bool(_PROHIBITED_PATH_RE.search(value or ""))
