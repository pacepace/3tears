"""Shared salience decay for agent primitives (memory, intention).

A single set-based decay pass reused across every table that carries the
salience substrate columns (``salience``, ``last_decayed_at``,
``evergreen``, ``date_created``). Each Collection keeps a thin
``decay_salience()`` method -- the single entry point for its table's SQL
-- that delegates here, so the decay formula lives in exactly one place
and ``agent/memory`` and ``agent/intention`` run identical, cadence-safe
decay.

Cadence safety: age is measured from ``last_decayed_at`` (the last decay
run), NOT last access, so total decay over a period is independent of how
often the job runs -- ``0.5^(a/h) * 0.5^(b/h) = 0.5^((a+b)/h)``. Rows never
decayed anchor on ``date_created``.

The pass NEVER deletes and NEVER drops below the floor: salience is clamped
to ``[floor, 1.0]``. ``evergreen`` rows are skipped entirely (the pin for
core identity facts / standing wants).
"""

from __future__ import annotations

from threetears.core.backends.protocol import L3Backend, parse_rowcount
from threetears.observe import get_logger

__all__ = ["apply_salience_decay"]

log = get_logger(__name__)


async def apply_salience_decay(
    pool: L3Backend,
    *,
    table: str,
    half_life_seconds: float,
    floor: float,
    skip_evergreen: bool = True,
) -> int:
    """Decay stored salience for every eligible row in ``table``.

    Applies ``salience := clamp(floor, salience * 0.5^(age / half_life), 1.0)``
    where ``age = now - COALESCE(last_decayed_at, date_created)``, and stamps
    ``last_decayed_at = now()``. Set-based, single round trip.

    ``table`` is a code-controlled identifier supplied by the calling
    Collection (never user input); it is interpolated as an SQL identifier
    while the tunable floor + half-life ride bound parameters.

    ``skip_evergreen`` controls the ``WHERE NOT evergreen`` guard: tables
    that carry the memory salience substrate's ``evergreen`` pin column
    (``memories``) leave it ``True`` so pinned rows never decay; tables
    without that column (``intentions`` -- a standing want has no pin
    concept, design §6.5) pass ``False`` so the pass touches every row.
    ``skip_evergreen`` is a code-controlled boolean, never user input.

    :param pool: L3 durable-store backend for the table's schema
    :ptype pool: L3Backend
    :param table: target table name (e.g. ``"memories"`` / ``"intentions"``)
    :ptype table: str
    :param half_life_seconds: decay half-life in seconds
    :ptype half_life_seconds: float
    :param floor: salience asymptote; salience never decays below this
    :ptype floor: float
    :param skip_evergreen: when ``True`` (default), exclude rows flagged
        ``evergreen``; set ``False`` for tables without that column
    :ptype skip_evergreen: bool
    :return: number of rows decayed
    :rtype: int
    """
    # cross-partition by design: salience decay is a global maintenance
    # sweep with no partition to scope to (it ages every row the pool holds).
    # the table name is a fixed literal from the Collection, so no partition
    # column predicate applies; the tunable floor + half-life ride params.
    where_clause = " WHERE NOT evergreen" if skip_evergreen else ""
    sql = (
        f"UPDATE {table} "
        "SET salience = LEAST(1.0, GREATEST($1, salience * power("
        "0.5::double precision, "
        "EXTRACT(EPOCH FROM (now() - COALESCE(last_decayed_at, date_created))) / $2))), "
        "    last_decayed_at = now()"
        f"{where_clause}"
    )
    status = await pool.execute(sql, floor, half_life_seconds)
    # framework-owned parser for the L3Backend command-tag border
    # ("UPDATE <n>"); resolves mock/empty statuses to 0.
    return parse_rowcount(status)
