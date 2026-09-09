"""OMNISIGHT-PILOT-001 ethics/institutional-review status -- DISTINCT from
the software `staged_pilot_collection_approved` flag. Software approval
means "the collection-admission gate will not itself refuse this request
on scope grounds"; ethics/institutional-review status means "a human has
actually recorded the applicable institutional/mentor/ethics determination
for real human-subject capture." Both must clear before any real
participant is recorded -- neither implies the other.

This module never invents, assumes, or self-sets an approved/exempt
determination. The default is always the most conservative value.
"""

from __future__ import annotations

NOT_ASSESSED = "NOT_ASSESSED"
NOT_REQUIRED_BY_INSTITUTION = "NOT_REQUIRED_BY_INSTITUTION"
APPROVED_OR_EXEMPT = "APPROVED_OR_EXEMPT"
BLOCKED_PENDING_REVIEW = "BLOCKED_PENDING_REVIEW"

ETHICS_STATUSES = (NOT_ASSESSED, NOT_REQUIRED_BY_INSTITUTION, APPROVED_OR_EXEMPT, BLOCKED_PENDING_REVIEW)

# The only two statuses that clear the ethics/institutional-review gate for
# real participant recording. NOT_ASSESSED (the default) and
# BLOCKED_PENDING_REVIEW both correctly keep field work blocked.
CLEARED_STATUSES = (NOT_REQUIRED_BY_INSTITUTION, APPROVED_OR_EXEMPT)

# The module's own current determination. Hardcoded to the most
# conservative value -- this file NEVER sets it to a cleared status itself;
# only a human operator recording a real determination (by editing this
# constant, or a future equivalent config the human controls) can change
# it. Never touched by any automated code path in this codebase.
CURRENT_ETHICS_STATUS = NOT_ASSESSED


def is_ethics_status_valid(status: str) -> bool:
    return status in ETHICS_STATUSES


def is_field_clearance_granted(status: str) -> bool:
    """True only for the two statuses a human can affirmatively record as
    already satisfying institutional requirements. NOT_ASSESSED (never
    looked at) and BLOCKED_PENDING_REVIEW (looked at, not yet cleared)
    both correctly return False -- the default is always fail-closed."""
    return status in CLEARED_STATUSES
