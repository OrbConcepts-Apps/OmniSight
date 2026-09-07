"""Phase J -- generic checkpoint/resume compatibility contract.

Defines what a Runner must report about a checkpoint, and the compatibility
check that must pass before JobManager ever calls Runner.resume(). No real
neural-network checkpoint format is implied here -- this is purely the
metadata envelope future stochastic-training runners must populate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CheckpointRef:
    path: str
    step: int
    seed: int
    config_hash: str
    dataset_manifest_hash: str
    code_commit_hash: str
    created_at: str
    experiment_id: str = ""

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "step": self.step,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "code_commit_hash": self.code_commit_hash,
            "created_at": self.created_at,
            "experiment_id": self.experiment_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointRef":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


class IncompatibleCheckpointError(RuntimeError):
    """Raised when a resume is refused. `reasons` lists every mismatch
    found (not just the first) so a human reviewing a blocked resume sees
    the complete picture in one shot."""

    def __init__(self, reasons: list):
        self.reasons = list(reasons)
        super().__init__("incompatible checkpoint, resume refused: " + "; ".join(self.reasons))


def check_resume_compatibility(
    *,
    checkpoint: Optional[CheckpointRef],
    expected_config_hash: str,
    expected_dataset_manifest_hash: str,
    expected_code_commit_hash: str,
    expected_experiment_id: str,
    expected_seed: int,
) -> None:
    """Raises IncompatibleCheckpointError unless every identity field
    matches exactly. Per Phase J authorization section 11, at minimum
    reject resume when: config hash differs, dataset manifest hash differs,
    code/experiment identity differs, seed identity is inconsistent, or the
    checkpoint is missing/corrupt. Never silently starts over and calls it
    a resume -- the caller must treat any raise here as "launch fresh
    instead", explicitly, never implicitly."""
    if checkpoint is None:
        raise IncompatibleCheckpointError(["checkpoint is missing"])

    reasons = []
    if checkpoint.config_hash != expected_config_hash:
        reasons.append(
            f"config_hash mismatch: checkpoint={checkpoint.config_hash!r} "
            f"expected={expected_config_hash!r}"
        )
    if checkpoint.dataset_manifest_hash != expected_dataset_manifest_hash:
        reasons.append(
            f"dataset_manifest_hash mismatch: checkpoint={checkpoint.dataset_manifest_hash!r} "
            f"expected={expected_dataset_manifest_hash!r}"
        )
    if checkpoint.code_commit_hash != expected_code_commit_hash:
        reasons.append(
            f"code_commit_hash mismatch: checkpoint={checkpoint.code_commit_hash!r} "
            f"expected={expected_code_commit_hash!r}"
        )
    if checkpoint.experiment_id != expected_experiment_id:
        reasons.append(
            f"experiment_id mismatch: checkpoint={checkpoint.experiment_id!r} "
            f"expected={expected_experiment_id!r}"
        )
    if checkpoint.seed != expected_seed:
        reasons.append(f"seed mismatch: checkpoint={checkpoint.seed!r} expected={expected_seed!r}")

    if reasons:
        raise IncompatibleCheckpointError(reasons)
