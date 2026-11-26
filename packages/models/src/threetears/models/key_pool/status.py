"""What a key pool reports about its keys: to a health endpoint (``status()``, KP-K2), and in the error
it raises when none can serve (KP-I1).

Every field is the operator's vocabulary: a key's label, its state and until when, and its money. No
secret and no fingerprint is ever in one (KP-B1).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

__all__ = ["KeyPoolStatus", "KeyState", "KeyStatus"]


class KeyState(enum.StrEnum):
    """Why a key can or can't serve, worst first after :attr:`OK`."""

    #: the key can serve
    OK = "ok"
    #: the provider said the key's (or its account's) credit, spend limit or quota is used up; it ends
    #: at the reset the provider named, else when the billing period rolls over, and at once when the
    #: label's secret changes or an operator resets it
    EXHAUSTED = "exhausted"
    #: the provider no longer accepts the key; it ends when the label's secret changes or an operator
    #: resets it
    REVOKED = "revoked"
    #: the key may not use the model (or any model); it ends as :attr:`REVOKED` does
    FORBIDDEN = "forbidden"
    #: the key's circuit breaker is open in this process
    TRIPPED = "tripped"
    #: the key's room for this period is below the pool's ``min_headroom``, or too small for the call
    NO_ROOM = "no_room"
    #: only in :class:`~threetears.models.key_pool.errors.NoKeyAvailableError`: the call already tried
    #: this key, and each key is tried at most once per call
    TRIED = "tried"


@dataclass(frozen=True, slots=True)
class KeyStatus:
    """One key, as the pool sees it at one instant.

    :ivar label: the key's label
    :ivar state: whether it can serve, and if not, the worst reason why
    :ivar until: when ``state`` ends, where it ends at a known time (a breaker; an exhausted key's
        reset, or the period's rollover for an exhausted key the provider named none for and for one
        with no room); ``None`` otherwise
    :ivar spent: money settled this period, orphaned reservations included
    :ivar reserved: money held by calls in flight this period
    :ivar remaining: ``budget - spent - reserved``, never below zero
    :ivar budget: the key's budget for the period, as this process configures it
    :ivar active: whether it is the pool's active key
    :ivar forbidden_models: models the key is forbidden for, when it is forbidden per model
    """

    label: str
    state: KeyState
    until: datetime | None
    spent: Decimal
    reserved: Decimal
    remaining: Decimal
    budget: Decimal
    active: bool
    forbidden_models: tuple[str, ...] = ()

    def describe(self) -> str:
        """The state in a few words, as :class:`NoKeyAvailableError` names it.

        :return: ``exhausted``, ``revoked``, ``forbidden``, ``tripped until T``, ``no room: R of B``,
            ``tried`` or ``ok``
        :rtype: str
        """
        if self.state is KeyState.NO_ROOM:
            return f"no room: {_money(self.remaining)} of {_money(self.budget)}"
        if self.state is KeyState.TRIPPED and self.until is not None:
            return f"{self.state} until {self.until.isoformat()}"
        return str(self.state)


@dataclass(frozen=True, slots=True)
class KeyPoolStatus:
    """A pool's keys at one instant (KP-K2).

    :ivar pool: the pool's name
    :ivar provider: the provider its keys are for
    :ivar period: the billing period's id
    :ivar active: the active key's label, ``None`` before the pool's first use
    :ivar keys: every key, in preference order
    """

    pool: str
    provider: str
    period: str
    active: str | None
    keys: tuple[KeyStatus, ...]


def _money(amount: Decimal) -> str:
    """``amount`` without trailing zeros or an exponent: ``0.4``, ``100``."""
    text = format(amount.normalize(), "f")
    return text if text != "-0" else "0"
