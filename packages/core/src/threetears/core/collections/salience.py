"""Salience decay on read, an option of a collection whose rows carry the salience substrate.

A table with the substrate columns (``salience``, ``last_decayed_at``, ``date_created``, and for memories
``evergreen``) stores each row's base salience and the time it was stamped. With decay on (a collection's
``salience_decay``, off by default), a reader never sees the stored base: it sees the effective salience

    ``clamp(floor, base * 0.5^((now - stamped_at) / half_life), 1)``

where ``stamped_at = COALESCE(last_decayed_at, date_created)``. Nothing sweeps the table: a write that changes a row's
salience (a reinforcement) stamps the row by its key, folding the decay so far into the new base. This is the value
the scheduled sweep it replaces (``apply_salience_decay``, removed by Pace's ruling 2026-11-11) gave at the same
instant, and it is cadence-free by construction: ``0.5^(a/h) * 0.5^(b/h) = 0.5^((a+b)/h)``, and the floor, once
reached, holds.

A ranking by effective salience is a ranking by ``ln(base) + ln(2)/half_life * epoch(stamped_at)`` (the same order
above the floor, ``now`` dropped as a constant), an expression an index can hold (:meth:`SalienceDecay.rank_sql`).

Off (``None``), every statement reads the stored ``salience`` as it always has.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

__all__ = ["SalienceDecay"]

#: what a row's salience is stamped by: its last stamp, else its creation
_STAMPED_AT = "COALESCE(last_decayed_at, date_created)"

#: the smallest salience the ranking takes a logarithm of (a stored 0 ranks last, not as minus infinity)
_LEAST_LOGGED = 1e-12


@dataclass(frozen=True)
class SalienceDecay:
    """a collection's salience decay on read.

    :ivar half_life: the time a row's salience takes to halve
    :ivar floor: the salience no row decays below
    :ivar pinned: a boolean column whose rows never decay (memories' ``evergreen``), or None
    """

    half_life: timedelta
    floor: float
    pinned: str | None = None

    def __post_init__(self) -> None:
        if self.half_life.total_seconds() <= 0:
            raise ValueError(f"a salience half-life is positive, got {self.half_life}")
        if not 0.0 <= self.floor <= 1.0:
            raise ValueError(f"a salience floor is within 0..1, got {self.floor}")

    def effective(self, base: float, stamped_at: datetime, now: datetime, *, pinned: bool = False) -> float:
        """a row's effective salience at ``now``.

        :param base: its stored salience
        :ptype base: float
        :param stamped_at: when it was stamped (``last_decayed_at``, else ``date_created``)
        :ptype stamped_at: datetime
        :param now: the time read at
        :ptype now: datetime
        :param pinned: whether the row's pin column is set (it never decays)
        :ptype pinned: bool
        :return: the effective salience
        :rtype: float
        """
        if pinned:
            return base
        age = (now - stamped_at).total_seconds()
        return min(1.0, max(self.floor, base * math.pow(0.5, age / self.half_life.total_seconds())))

    def effective_sql(self) -> str:
        """the effective salience of the row read, as SQL over its own columns (PostgreSQL / YugabyteDB).

        The half-life and floor are this config's numbers, written as literals (never caller input).

        :return: the expression
        :rtype: str
        """
        decayed = (
            f"LEAST(1.0, GREATEST({float(self.floor)!r}, salience * power(0.5::double precision, "
            f"EXTRACT(EPOCH FROM (now() - {_STAMPED_AT})) / {self.half_life.total_seconds()!r})))"
        )
        return decayed if self.pinned is None else f"(CASE WHEN {self.pinned} THEN salience ELSE {decayed} END)"

    def rank_sql(self) -> str:
        """an expression that orders rows as their effective salience does above the floor, with no ``now`` in it,
        so an index can hold it: ``ln(base) + ln(2) / half_life * epoch(stamped_at)``.

        :return: the expression
        :rtype: str
        """
        rate = math.log(2.0) / self.half_life.total_seconds()
        return f"(ln(GREATEST(salience, {_LEAST_LOGGED!r})) + {rate!r} * EXTRACT(EPOCH FROM {_STAMPED_AT}))"

    def reinforced_sql(self, bump: str) -> str:
        """the ``SET`` of a reinforcement by key: the effective salience so far plus ``bump``, clamped to 1, stamped
        now, so the decay to now is folded into the new base.

        :param bump: the increment's bind (``$1``)
        :ptype bump: str
        :return: the assignments
        :rtype: str
        """
        return f"salience = LEAST(1.0, {self.effective_sql()} + {bump}), last_decayed_at = now()"
