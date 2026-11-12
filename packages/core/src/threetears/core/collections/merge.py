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

from threetears.core.keyed import update_keyed

__all__ = ["repoint_user_rows"]


async def repoint_user_rows(
    conn: Any,
    *,
    table: str,
    user_column: str,
    pk_columns: Sequence[str],
    from_user_id: UUID,
    to_user_id: UUID,
    agent_id: UUID | None = None,
    touch_column: str | None = "date_updated",
    now: datetime | None = None,
) -> list[tuple[Any, ...]]:
    """repoint one ownership column from a source user to a master user.

    issues ``UPDATE {table} SET {user_column} = master[, {touch} = now]
    WHERE {user_column} = source RETURNING {pk_columns}`` on ``conn`` (a
    transaction connection). returns the primary-key tuples of the moved
    rows for post-commit cache invalidation.

    in an agent's table (``agent_id`` given) the ``UPDATE`` pins ``agent_id =
    agent`` beside the owner: a table in an agent's schema is led by
    ``agent_id`` (its key and its owner index ``(agent_id, user_id)``), so
    pinning the agent the schema holds reads one user's rows through that
    index. the schema holds only that agent's rows, so it moves the rows the
    owner alone names.

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
    :param agent_id: the agent whose schema ``conn`` is bound to, for a table
        in an agent's schema; None for a platform table
    :ptype agent_id: UUID | None
    :param touch_column: optional ``date_updated``-style column to stamp;
        ``None`` to skip
    :ptype touch_column: str | None
    :param now: timestamp for ``touch_column``; defaults to ``now(UTC)``
    :ptype now: datetime | None
    :return: list of primary-key tuples for the repointed rows
    :rtype: list[tuple[Any, ...]]
    """
    values = _repointed_values(user_column, to_user_id, touch_column, now)
    by = {user_column: from_user_id} if agent_id is None else {"agent_id": agent_id, user_column: from_user_id}
    moved = await update_keyed(conn, table, values=values, by=by, returning=pk_columns)
    return [tuple(row[col] for col in pk_columns) for row in moved.rows]


def _repointed_values(
    user_column: str, to_user_id: UUID, touch_column: str | None, now: datetime | None
) -> dict[str, Any]:
    """the columns a repoint sets: the owner to the master, and the touch column stamped when there is one.

    :param user_column: ownership column holding the user id
    :ptype user_column: str
    :param to_user_id: master user the rows move to
    :ptype to_user_id: UUID
    :param touch_column: ``date_updated``-style column to stamp, or None
    :ptype touch_column: str | None
    :param now: timestamp for ``touch_column``; defaults to ``now(UTC)``
    :ptype now: datetime | None
    :return: ``{column: value}`` to set
    :rtype: dict[str, Any]
    """
    values: dict[str, Any] = {user_column: to_user_id}
    if touch_column is not None:
        values[touch_column] = now if now is not None else datetime.now(UTC)
    return values
