"""OMNISIGHT-PILOT-001 ethics/institutional-review status tests
(research/datasets/ethics_review.py). No real determination recorded."""

from __future__ import annotations

from research.datasets.ethics_review import (
    APPROVED_OR_EXEMPT,
    BLOCKED_PENDING_REVIEW,
    CURRENT_ETHICS_STATUS,
    NOT_ASSESSED,
    NOT_REQUIRED_BY_INSTITUTION,
    is_ethics_status_valid,
    is_field_clearance_granted,
)


class TestDefaultStatus:
    def test_current_status_defaults_to_not_assessed(self):
        """Never self-set to approved/exempt."""
        assert CURRENT_ETHICS_STATUS == NOT_ASSESSED

    def test_not_assessed_is_not_cleared(self):
        assert is_field_clearance_granted(NOT_ASSESSED) is False

    def test_blocked_pending_review_is_not_cleared(self):
        assert is_field_clearance_granted(BLOCKED_PENDING_REVIEW) is False


class TestClearedStatuses:
    def test_not_required_by_institution_is_cleared(self):
        assert is_field_clearance_granted(NOT_REQUIRED_BY_INSTITUTION) is True

    def test_approved_or_exempt_is_cleared(self):
        assert is_field_clearance_granted(APPROVED_OR_EXEMPT) is True


class TestValidation:
    def test_unknown_status_invalid(self):
        assert is_ethics_status_valid("MADE_UP_STATUS") is False
        assert is_field_clearance_granted("MADE_UP_STATUS") is False

    def test_all_four_statuses_valid(self):
        for status in (NOT_ASSESSED, NOT_REQUIRED_BY_INSTITUTION, APPROVED_OR_EXEMPT, BLOCKED_PENDING_REVIEW):
            assert is_ethics_status_valid(status) is True
