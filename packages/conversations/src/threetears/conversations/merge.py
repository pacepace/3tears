"""user-merge repoint for the ``conversations`` table.

owns the ``conversations``-table half of a user merge: when source user
S is merged into master user M, every conversation S owns is repointed to
M. the table lives in a per-agent schema (``agent_<hex>``); the caller
(the hub merge orchestrator) opens a connection, sets ``search_path`` to
the agent schema, and calls this inside that schema's transaction — the
same per-agent-schema pattern the broker uses to migrate agent schemas.

``conversations.user_id`` is ``immutable=True`` (the collection upsert
path refuses to write it), so the repoint cannot ride the collection save
path: it is a raw scoped UPDATE via
:func:`threetears.core.collections.repoint_user_rows`. the caller
invalidates the returned conversation keys after the transaction commits
(a bulk UPDATE bypasses per-entity cache invalidation) and reconciles the
master's ``conversation-owner`` RBAC group on the platform schema.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from threetears.core.collections import repoint_user_rows
from threetears.observe import get_logger

__all__ = ["repoint_user"]

log = get_logger(__name__)


async def repoint_user(
    conn: Any,
    *,
    from_user_id: UUID,
    to_user_id: UUID,
) -> list[tuple[Any, ...]]:
    """repoint every conversation owned by ``from_user_id`` to ``to_user_id``.

    issues one scoped ``UPDATE conversations SET user_id = master WHERE
    user_id = source`` against ``conn`` (a transaction connection whose
    ``search_path`` the caller has set to the target agent schema) and
    returns the ``(agent_id, conversation_id)`` keys of the moved rows
    for post-commit cache invalidation. idempotent: a re-run finds no
    rows still owned by the source and is a no-op.

    :param conn: asyncpg transaction connection bound to the agent schema
    :ptype conn: Any
    :param from_user_id: source user whose conversations move
    :ptype from_user_id: UUID
    :param to_user_id: master user the conversations move to
    :ptype to_user_id: UUID
    :return: ``(agent_id, conversation_id)`` keys of the repointed rows
    :rtype: list[tuple[Any, ...]]
    """
    moved = await repoint_user_rows(
        conn,
        table="conversations",
        user_column="user_id",
        pk_columns=["agent_id", "conversation_id"],
        from_user_id=from_user_id,
        to_user_id=to_user_id,
    )
    log.info(
        "repointed %d conversation(s) from user %s to %s",
        len(moved),
        from_user_id,
        to_user_id,
    )
    return moved
