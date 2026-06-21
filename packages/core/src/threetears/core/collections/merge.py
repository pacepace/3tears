"""user-merge repoint helpers.

merging a source user into a master user repoints every user-owned row to
the master. ``user_id`` (and the analogous ownership columns) are declared
``immutable=True`` on every owning table, so a repoint cannot ride the
collection upsert path (``DO UPDATE`` excludes immutable columns). these
helpers issue a raw scoped ``UPDATE`` against a caller-supplied
transaction connection and return the primary keys of the rows they moved
so the caller can invalidate the L1/L2 cache for them AFTER the
transaction commits (a bulk UPDATE bypasses per-entity invalidation).

the helpers are pure SQL against ``conn``; they do no caching and no
validation. they are the single sanctioned place a user-ownership column
is rewritten, used by the per-collection ``repoint_user`` methods across
the hub and the 3tears agent packages.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

__all__ = ["repoint_user_rows"]


async def repoint_user_rows(
    conn: Any,
    *,
    table: str,
    user_column: str,
    pk_columns: Sequence[str],
    from_user_id: UUID,
    to_user_id: UUID,
    touch_column: str | None = "date_updated",
    now: datetime | None = None,
) -> list[tuple[Any, ...]]:
    """repoint one ownership column from a source user to a master user.

    issues ``UPDATE {table} SET {user_column} = master[, {touch} = now]
    WHERE {user_column} = source RETURNING {pk_columns}`` on ``conn`` (a
    transaction connection). returns the primary-key tuples of the moved
    rows for post-commit cache invalidation.

    cache-bypass: this is a deliberate bulk ownership rewrite for a user
    merge; ``user_column`` is immutable so it cannot ride the collection
    upsert path, and the move is one transactional UPDATE. the caller
    invalidates the returned keys after commit.

    :param conn: asyncpg transaction connection (search_path set by caller)
    :ptype conn: Any
    :param table: unqualified table name to repoint
    :ptype table: str
    :param user_column: ownership column holding the user id
    :ptype user_column: str
    :param pk_columns: primary-key column names, in order
    :ptype pk_columns: Sequence[str]
    :param from_user_id: source user whose rows move
    :ptype from_user_id: UUID
    :param to_user_id: master user the rows move to
    :ptype to_user_id: UUID
    :param touch_column: optional ``date_updated``-style column to stamp;
        ``None`` to skip
    :ptype touch_column: str | None
    :param now: timestamp for ``touch_column``; defaults to ``now(UTC)``
    :ptype now: datetime | None
    :return: list of primary-key tuples for the repointed rows
    :rtype: list[tuple[Any, ...]]
    """
    set_parts = [f"{user_column} = $2"]
    params: list[Any] = [from_user_id, to_user_id]
    if touch_column is not None:
        params.append(now if now is not None else datetime.now(UTC))
        set_parts.append(f"{touch_column} = ${len(params)}")
    pk_list = ", ".join(pk_columns)
    sql = f"UPDATE {table} SET {', '.join(set_parts)} WHERE {user_column} = $1 RETURNING {pk_list}"
    rows = await conn.fetch(sql, *params)
    return [tuple(row[col] for col in pk_columns) for row in rows]
