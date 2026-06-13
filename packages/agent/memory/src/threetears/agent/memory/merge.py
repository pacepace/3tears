"""user-merge repoint for the memory tables.

owns the memory half of a user merge: when source user S is merged into
master user M, every memory (and its media / media_content /
memory_chunks children) that S owns is repointed to M. the tables live in
a per-agent schema (``agent_<hex>``); the caller (the hub merge
orchestrator) opens a connection, sets ``search_path`` to the agent
schema, and calls this inside that schema's transaction.

merge is one-way: M wins. the ``memories`` table carries a partial unique
index ``(agent_id, user_id, alias) WHERE alias IS NOT NULL`` — an alias is
a named lookup anchor, unambiguous per user per agent. if S and M each
anchored a memory with the SAME alias under one agent, repointing S's
``user_id`` to M would violate that index. the merge resolves the clash
by DELETING S's colliding memory (M's wins) BEFORE the repoint; the
delete cascades to that memory's media / media_content / memory_chunks via
``ON DELETE CASCADE``. S's NON-colliding memories repoint normally.

every owned ``user_id`` column is ``immutable=True`` (the collection
upsert path refuses to write it), so the repoint is a raw scoped UPDATE
via :func:`threetears.core.collections.repoint_user_rows`. the caller
invalidates the returned keys after commit and reconciles the master's
``memory-owner`` RBAC group on the platform schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from threetears.core.collections import repoint_user_rows
from threetears.observe import get_logger

__all__ = ["MemoryRepointResult", "repoint_user"]

log = get_logger(__name__)


@dataclass
class MemoryRepointResult:
    """primary keys touched by a memory-table user merge in one agent schema.

    ``alias_collisions_deleted`` are ``(agent_id, memory_id)`` keys of
    source memories hard-deleted because their alias collided with a
    master memory (children cascaded). the remaining lists are the keys
    repointed from source to master per table. the caller sums these for
    the merge audit event and invalidates the repointed keys post-commit.

    :param alias_collisions_deleted: deleted colliding source memory keys
    :ptype alias_collisions_deleted: list[tuple[Any, ...]]
    :param memories: repointed ``memories`` keys
    :ptype memories: list[tuple[Any, ...]]
    :param media: repointed ``media`` keys
    :ptype media: list[tuple[Any, ...]]
    :param media_content: repointed ``media_content`` keys
    :ptype media_content: list[tuple[Any, ...]]
    :param memory_chunks: repointed ``memory_chunks`` keys
    :ptype memory_chunks: list[tuple[Any, ...]]
    """

    alias_collisions_deleted: list[tuple[Any, ...]] = field(default_factory=list)
    memories: list[tuple[Any, ...]] = field(default_factory=list)
    media: list[tuple[Any, ...]] = field(default_factory=list)
    media_content: list[tuple[Any, ...]] = field(default_factory=list)
    memory_chunks: list[tuple[Any, ...]] = field(default_factory=list)


async def _delete_alias_collisions(
    conn: Any,
    *,
    from_user_id: UUID,
    to_user_id: UUID,
) -> list[tuple[Any, ...]]:
    """delete source memories whose alias collides with a master memory.

    cache-bypass: a merge collision resolution — the source's colliding
    memory loses to the master's (merge is one-way). the DELETE cascades
    to that memory's media / media_content / memory_chunks children. run
    BEFORE the repoint so the ``(agent_id, user_id, alias)`` unique index
    is not violated when the survivors flip to the master. idempotent: a
    re-run finds no source-owned colliding rows.

    :param conn: asyncpg transaction connection bound to the agent schema
    :ptype conn: Any
    :param from_user_id: source user whose colliding memories are deleted
    :ptype from_user_id: UUID
    :param to_user_id: master user whose memories win the alias
    :ptype to_user_id: UUID
    :return: ``(agent_id, memory_id)`` keys of the deleted memories
    :rtype: list[tuple[Any, ...]]
    """
    rows = await conn.fetch(
        "DELETE FROM memories m "
        "WHERE m.user_id = $1 AND m.alias IS NOT NULL "
        "AND EXISTS ("
        "    SELECT 1 FROM memories m2 "
        "    WHERE m2.user_id = $2 "
        "      AND m2.agent_id = m.agent_id "
        "      AND m2.alias = m.alias"
        ") "
        "RETURNING agent_id, memory_id",
        from_user_id,
        to_user_id,
    )
    return [(row["agent_id"], row["memory_id"]) for row in rows]


async def repoint_user(
    conn: Any,
    *,
    from_user_id: UUID,
    to_user_id: UUID,
) -> MemoryRepointResult:
    """repoint every memory owned by ``from_user_id`` to ``to_user_id``.

    resolves alias collisions first (master wins, source's colliding
    memory deleted with its children), then repoints ``user_id`` from
    source to master across ``memories``, ``media``, ``media_content``,
    and ``memory_chunks`` against ``conn`` (a transaction connection whose
    ``search_path`` the caller has set to the target agent schema).
    returns the keys touched per table for post-commit invalidation and
    the merge audit event. idempotent: a re-run finds no source-owned
    rows and is a no-op.

    the four tables each carry their own ``user_id`` (denormalized for
    per-table RBAC filtering), so each is repointed directly rather than
    relying on a parent join.

    :param conn: asyncpg transaction connection bound to the agent schema
    :ptype conn: Any
    :param from_user_id: source user whose memories move
    :ptype from_user_id: UUID
    :param to_user_id: master user the memories move to
    :ptype to_user_id: UUID
    :return: keys deleted + repointed per table
    :rtype: MemoryRepointResult
    """
    deleted = await _delete_alias_collisions(
        conn,
        from_user_id=from_user_id,
        to_user_id=to_user_id,
    )
    result = MemoryRepointResult(alias_collisions_deleted=deleted)
    result.memories = await repoint_user_rows(
        conn,
        table="memories",
        user_column="user_id",
        pk_columns=["agent_id", "memory_id"],
        from_user_id=from_user_id,
        to_user_id=to_user_id,
    )
    result.media = await repoint_user_rows(
        conn,
        table="media",
        user_column="user_id",
        pk_columns=["agent_id", "media_id"],
        from_user_id=from_user_id,
        to_user_id=to_user_id,
        touch_column=None,
    )
    result.media_content = await repoint_user_rows(
        conn,
        table="media_content",
        user_column="user_id",
        pk_columns=["agent_id", "content_id"],
        from_user_id=from_user_id,
        to_user_id=to_user_id,
        touch_column=None,
    )
    result.memory_chunks = await repoint_user_rows(
        conn,
        table="memory_chunks",
        user_column="user_id",
        pk_columns=["agent_id", "chunk_id"],
        from_user_id=from_user_id,
        to_user_id=to_user_id,
        touch_column=None,
    )
    log.info(
        "repointed memories from user %s to %s (deleted %d alias "
        "collision(s); moved %d memories, %d media, %d media_content, "
        "%d memory_chunks)",
        from_user_id,
        to_user_id,
        len(deleted),
        len(result.memories),
        len(result.media),
        len(result.media_content),
        len(result.memory_chunks),
    )
    return result
