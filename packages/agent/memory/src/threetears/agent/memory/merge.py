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
delete cascades to that memory's media / media_content / memory_chunks /
memory_consolidations via ``ON DELETE CASCADE``. S's NON-colliding
memories repoint normally.

a cascade returns nothing, and every child it removes may sit in some
pod's cache, served by id after L3 lost it. so the colliding memories and
their media are locked ``FOR UPDATE`` first -- no child can be added under
a locked parent until the transaction ends -- every child key the cascade
will remove is read, and only then are exactly the locked memories
deleted. the result names every one of them for the caller to evict.

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
    master memory; the ``alias_collision_*`` lists are the keys of the
    children that delete cascaded to. the remaining lists are the keys
    repointed from source to master per table. the caller sums these for
    the merge audit event and evicts every one of them post-commit.

    :param alias_collisions_deleted: deleted colliding source memory keys
    :ptype alias_collisions_deleted: list[tuple[Any, ...]]
    :param alias_collision_media: ``(agent_id, media_id)`` keys cascaded
        with the deleted memories
    :ptype alias_collision_media: list[tuple[Any, ...]]
    :param alias_collision_media_content: ``(agent_id, content_id)`` keys
        cascaded with that media
    :ptype alias_collision_media_content: list[tuple[Any, ...]]
    :param alias_collision_memory_chunks: ``(agent_id, chunk_id)`` keys
        cascaded with the deleted memories
    :ptype alias_collision_memory_chunks: list[tuple[Any, ...]]
    :param alias_collision_memory_consolidations: ``(agent_id,
        consolidated_memory_id, source_memory_id)`` edge keys cascaded with
        either endpoint
    :ptype alias_collision_memory_consolidations: list[tuple[Any, ...]]
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
    alias_collision_media: list[tuple[Any, ...]] = field(default_factory=list)
    alias_collision_media_content: list[tuple[Any, ...]] = field(default_factory=list)
    alias_collision_memory_chunks: list[tuple[Any, ...]] = field(default_factory=list)
    alias_collision_memory_consolidations: list[tuple[Any, ...]] = field(default_factory=list)
    memories: list[tuple[Any, ...]] = field(default_factory=list)
    media: list[tuple[Any, ...]] = field(default_factory=list)
    media_content: list[tuple[Any, ...]] = field(default_factory=list)
    memory_chunks: list[tuple[Any, ...]] = field(default_factory=list)


# the collision predicate: a source memory whose alias a master memory under the same agent
# already holds. ``$1`` is the source user, ``$2`` the master.
_COLLISION_PREDICATE = (
    "m.user_id = $1 AND m.alias IS NOT NULL "
    "AND EXISTS ("
    "    SELECT 1 FROM memories m2 "
    "    WHERE m2.user_id = $2 "
    "      AND m2.agent_id = m.agent_id "
    "      AND m2.alias = m.alias"
    ")"
)

# a set of composite keys bound as two parallel uuid arrays.
_KEY_SET = "(SELECT * FROM unnest({agents}::uuid[], {ids}::uuid[]))"


def _key_set(agents: str, ids: str) -> str:
    """render a composite-key set bound as two parallel uuid-array parameters.

    :param agents: the placeholder carrying the ``agent_id`` array (``$1``)
    :ptype agents: str
    :param ids: the placeholder carrying the matching id array (``$2``)
    :ptype ids: str
    :return: a subquery yielding the ``(agent_id, id)`` pairs
    :rtype: str
    """
    return _KEY_SET.format(agents=agents, ids=ids)


def _columns(keys: list[tuple[Any, ...]]) -> tuple[list[Any], list[Any]]:
    """split ``(agent_id, id)`` keys into the two parallel arrays :func:`_key_set` binds.

    :param keys: composite keys
    :ptype keys: list[tuple[Any, ...]]
    :return: the agent ids and the ids, index-aligned
    :rtype: tuple[list[Any], list[Any]]
    """
    return [key[0] for key in keys], [key[1] for key in keys]


async def _delete_alias_collisions(
    conn: Any,
    *,
    from_user_id: UUID,
    to_user_id: UUID,
) -> MemoryRepointResult:
    """delete source memories whose alias collides with a master memory, naming every row it removes.

    cache-bypass: a merge collision resolution -- the source's colliding
    memory loses to the master's (merge is one-way). run BEFORE the repoint
    so the ``(agent_id, user_id, alias)`` unique index is not violated when
    the survivors flip to the master. idempotent: a re-run finds no
    source-owned colliding rows.

    the DELETE cascades to the memories' media and that media's
    media_content, the memories' memory_chunks, and every
    memory_consolidations edge with either end on them -- and returns none
    of them. so the colliding memories are locked ``FOR UPDATE``, then their
    media, which blocks any new child referencing either (an FK insert takes
    a share lock on its parent) until the transaction ends; every child key
    is read under those locks; and the DELETE removes exactly the locked
    memories. the children read are then exactly the children cascaded.

    :param conn: asyncpg transaction connection bound to the agent schema
    :ptype conn: Any
    :param from_user_id: source user whose colliding memories are deleted
    :ptype from_user_id: UUID
    :param to_user_id: master user whose memories win the alias
    :ptype to_user_id: UUID
    :return: the deleted memories and every child their deletion cascaded to, in the
        ``alias_collision*`` fields; the repoint fields are empty
    :rtype: MemoryRepointResult
    """
    result = MemoryRepointResult()
    locked = await conn.fetch(
        f"SELECT m.agent_id, m.memory_id FROM memories m WHERE {_COLLISION_PREDICATE} FOR UPDATE OF m",
        from_user_id,
        to_user_id,
    )
    if not locked:
        return result
    memory_agents, memory_ids = _columns([(row["agent_id"], row["memory_id"]) for row in locked])
    memories = _key_set("$1", "$2")

    media_rows = await conn.fetch(
        f"SELECT agent_id, media_id FROM media WHERE (agent_id, memory_id) IN {memories} FOR UPDATE",
        memory_agents,
        memory_ids,
    )
    result.alias_collision_media = [(row["agent_id"], row["media_id"]) for row in media_rows]
    media_agents, media_ids = _columns(result.alias_collision_media)

    content_rows = await conn.fetch(
        f"SELECT agent_id, content_id FROM media_content WHERE (agent_id, media_id) IN {_key_set('$1', '$2')}",
        media_agents,
        media_ids,
    )
    result.alias_collision_media_content = [(row["agent_id"], row["content_id"]) for row in content_rows]

    chunk_rows = await conn.fetch(
        f"SELECT agent_id, chunk_id FROM memory_chunks WHERE (agent_id, memory_id) IN {memories}",
        memory_agents,
        memory_ids,
    )
    result.alias_collision_memory_chunks = [(row["agent_id"], row["chunk_id"]) for row in chunk_rows]

    edge_rows = await conn.fetch(
        "SELECT agent_id, consolidated_memory_id, source_memory_id FROM memory_consolidations WHERE "
        f"(agent_id, consolidated_memory_id) IN {memories} OR (agent_id, source_memory_id) IN {memories}",
        memory_agents,
        memory_ids,
    )
    result.alias_collision_memory_consolidations = [
        (row["agent_id"], row["consolidated_memory_id"], row["source_memory_id"]) for row in edge_rows
    ]

    deleted = await conn.fetch(
        f"DELETE FROM memories m WHERE {_COLLISION_PREDICATE} "
        f"AND (m.agent_id, m.memory_id) IN {_key_set('$3', '$4')} "
        "RETURNING agent_id, memory_id",
        from_user_id,
        to_user_id,
        memory_agents,
        memory_ids,
    )
    result.alias_collisions_deleted = [(row["agent_id"], row["memory_id"]) for row in deleted]
    return result


async def repoint_user(
    conn: Any,
    *,
    from_user_id: UUID,
    to_user_id: UUID,
) -> MemoryRepointResult:
    """repoint every memory owned by ``from_user_id`` to ``to_user_id``.

    resolves alias collisions first (master wins, source's colliding
    memory deleted with its children, every one of them named in the
    result), then repoints ``user_id`` from
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
    result = await _delete_alias_collisions(
        conn,
        from_user_id=from_user_id,
        to_user_id=to_user_id,
    )
    deleted = result.alias_collisions_deleted
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
