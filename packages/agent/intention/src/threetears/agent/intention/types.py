"""Intention value types -- the status lifecycle enum.

The status is a *constrained value set*, not a transition state machine
(design §6.5 "Lifecycle -- validated enum, NO state machine"). Nothing
guards a transition; the enum only bounds which values a ``status``
column may hold. A transition FSM would be gold-plating.

:data:`INTENTION_STATUS_VALUES` is the single source of truth for the
enum members. The schema Column's ``enum_type`` tuple imports it, and
the migration-chain test asserts the live PG enum carries exactly these
values -- so a drift between the DSL declaration and the hand-written
``CREATE TYPE`` DDL surfaces as a test failure rather than at runtime.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "INTENTION_STATUS_VALUES",
    "IntentionStatus",
]


class IntentionStatus(str, Enum):
    """Lifecycle status for a standing want.

    Conventional flow (not enforced): ``OPEN`` on log; ``OPEN -> ASKED``
    when the agent surfaces it via the deliberation wake; ``ASKED ->
    GRANTED`` / ``ASKED -> DROPPED`` when the user responds in the
    presence UI. Subclasses :class:`str` so a member compares equal to
    its wire value and serialises transparently.
    """

    OPEN = "open"
    ASKED = "asked"
    GRANTED = "granted"
    DROPPED = "dropped"


# declaration-order tuple of the wire values -- the schema enum_type and
# the migration ``CREATE TYPE`` share this ordering.
INTENTION_STATUS_VALUES: tuple[str, ...] = tuple(status.value for status in IntentionStatus)
