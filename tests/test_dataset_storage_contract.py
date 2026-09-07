"""EXP-0006 dataset storage-contract tests. No real media created --
verifies the repo's existing .gitignore/git-tracking invariants that the
future OmniSight-domain dataset protocol depends on."""

from __future__ import annotations

import subprocess

from research.config import REPO_ROOT

_MEDIA_EXTENSIONS = (".jpg", ".jpeg", ".png", ".mp4", ".mov", ".avi", ".webm", ".heic")


class TestRawDataNeverTracked:
    def test_data_raw_is_gitignored(self, tmp_path):
        """data/raw/ (where any future OmniSight-domain capture would live)
        must be gitignored -- confirmed against a synthetic path under it,
        never assuming a real capture exists."""
        probe = REPO_ROOT / "data" / "raw" / "omnisight_domain" / "probe.jpg"
        out = subprocess.run(
            ["git", "check-ignore", "--quiet", str(probe)],
            cwd=REPO_ROOT,
        )
        assert out.returncode == 0, "data/raw/ must be gitignored (see .gitignore)"

    def test_no_media_files_currently_tracked_under_data_raw(self):
        out = subprocess.run(
            ["git", "ls-files", "--", "data/raw/"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout
        tracked = [line for line in out.splitlines() if line.strip()]
        media = [f for f in tracked if f.lower().endswith(_MEDIA_EXTENSIONS)]
        assert media == [], f"raw media files must never be tracked in git: {media}"

    def test_manifests_directory_convention_exists(self):
        """A future OmniSight-domain manifest belongs alongside the
        existing tracked data/manifests/eval_manifest.jsonl convention
        (JSONL manifests ARE tracked; only raw media and .sqlite caches
        under data/manifests/ are gitignored)."""
        assert (REPO_ROOT / "data" / "manifests" / "eval_manifest.jsonl").exists()
