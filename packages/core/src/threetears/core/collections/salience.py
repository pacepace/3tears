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

from collections.abc import Sequence

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
    returning_columns: Sequence[str] | None = None,
) -> int | list[tuple[object, ...]]:
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

    ``returning_columns`` closes the three-tier cache-coherence gap: a raw
    L3 decay leaves L1/L2 holding the pre-decay salience, and a later
    full-entity save from that stale cache would write it back. When the
    caller passes its primary-key column names (also code-controlled
    identifiers, never user input), the UPDATE runs ``RETURNING`` those
    columns and this function returns the affected pk tuples so the
    Collection can invalidate each cached row (via ``invalidate_cache``).
    When ``None`` (the legacy shape), it returns the decayed-row count.

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
    :param returning_columns: primary-key column names to ``RETURNING``;
        when set, the return is the list of affected pk tuples (for cache
        invalidation) instead of the row count
    :ptype returning_columns: Sequence[str] | None
    :return: decayed-row count, or the affected pk tuples when
        ``returning_columns`` is set
    :rtype: int | list[tuple[object, ...]]
    """
    # cross-partition by design: salience decay is a global maintenance
    # sweep with no partition to scope to (it ages every row the pool holds).
    # the table name is a fixed literal from the Collection, so no partition
    # column predicate applies; the tunable floor + half-life ride params.
    where_clause = " WHERE NOT evergreen" if skip_evergreen else ""
    set_clause = (
        "SET salience = LEAST(1.0, GREATEST($1, salience * power("
        "0.5::double precision, "
        "EXTRACT(EPOCH FROM (now() - COALESCE(last_decayed_at, date_created))) / $2))), "
        "    last_decayed_at = now()"
    )
    if returning_columns:
        # returning_columns are the Collection's pk identifiers (code-
        # controlled, never user input), joined like ``table`` above.
        cols = ", ".join(returning_columns)
        rows = await pool.fetch(
            f"UPDATE {table} {set_clause}{where_clause} RETURNING {cols}",
            floor,
            half_life_seconds,
        )
        return [tuple(row[c] for c in returning_columns) for row in rows]
    status = await pool.execute(
        f"UPDATE {table} {set_clause}{where_clause}",
        floor,
        half_life_seconds,
    )
    # framework-owned parser for the L3Backend command-tag border
    # ("UPDATE <n>"); resolves mock/empty statuses to 0.
    return parse_rowcount(status)
