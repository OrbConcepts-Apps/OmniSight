"""Phase J -- output-directory collision/ownership safety.

Per Phase J authorization section 14: before long-running execution,
validate the output directory belongs to the job, never silently overwrite
another job's/experiment's artifacts, and keep partial runs preserved.
"""

from __future__ import annotations

import json
from pathlib import Path

MARKER_NAME = ".omnilab_job_owner"


class OutputDirCollisionError(RuntimeError):
    """Raised when a job's output directory already exists and is owned by
    a DIFFERENT job -- refuses to reuse/overwrite it."""


def reserve_job_output_dir(job_id: str, out_dir: Path) -> Path:
    """Stamps `out_dir` (the EXACT directory the caller wants to use --
    typically `base_dir/job_id`, but passed explicitly so two different
    job_ids that would otherwise collide on the same path are still
    caught) with an ownership marker. Raises OutputDirCollisionError if the
    directory already exists and is owned by a different job_id (or has no
    marker at all -- an unrecognized existing directory is treated as
    someone else's, never silently adopted). Safe/idempotent to call again
    for the SAME job_id (e.g. after a crash-restart) -- it does not clear
    existing contents."""
    out_dir = Path(out_dir)
    marker = out_dir / MARKER_NAME
    if out_dir.exists():
        if not marker.exists():
            raise OutputDirCollisionError(
                f"{out_dir} already exists with no OmniLab ownership marker -- refusing to "
                f"treat it as {job_id!r}'s output directory (would risk overwriting "
                "unrelated/pre-existing artifacts)."
            )
        owner = json.loads(marker.read_text(encoding="utf-8")).get("job_id")
        if owner != job_id:
            raise OutputDirCollisionError(
                f"{out_dir} is already owned by {owner!r}, not {job_id!r} -- refusing to reuse "
                "another job's output directory."
            )
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=False)
    marker.write_text(json.dumps({"job_id": job_id}), encoding="utf-8")
    return out_dir
