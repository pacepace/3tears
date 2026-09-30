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

the result groups every key by the table it belongs to
(:attr:`MemoryRepointResult.evict`), so the caller evicts with one loop
over that map and a table added to the cascade needs no edit on its side.

every owned ``user_id`` column is ``immutable=True`` (the collection
upsert path refuses to write it), so the repoint is a raw scoped UPDATE
via :func:`threetears.core.collections.repoint_user_rows`. the caller
invalidates every key in :attr:`MemoryRepointResult.evict` after commit and
reconciles the master's ``memory-owner`` RBAC group on the platform schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from threetears.core.collections import repoint_user_rows
from threetears.observe import get_logger

__all__ = ["MemoryRepointResult", "repoint_user"]

log = get_logger(__name__)


#: the tables the repoint moves from source to master: each carries its own ``user_id``.
_REPOINTED_TABLES: tuple[str, ...] = ("memories", "media", "media_content", "memory_chunks")

#: the tables the alias-collision delete removes rows from: the colliding memories, and every
#: table an ``ON DELETE CASCADE`` chain reaches from them. pinned against the declared schemas'
#: foreign keys by the merge tests, so a new cascade onto ``memories`` or ``media`` fails there
#: until it is named here and read below.
_REMOVED_TABLES: tuple[str, ...] = ("memories", "media", "media_content", "memory_chunks", "memory_consolidations")


def _per_table(tables: tuple[str, ...]) -> dict[str, list[tuple[Any, ...]]]:
    """an empty key list for each of ``tables``.

    :param tables: the table names
    :ptype tables: tuple[str, ...]
    :return: ``{table: []}`` for every table, in the given order
    :rtype: dict[str, list[tuple[Any, ...]]]
    """
    return {table: [] for table in tables}


@dataclass
class MemoryRepointResult:
    """primary keys touched by a memory-table user merge in one agent schema, grouped by table.

    the caller evicts every key in :attr:`evict` post-commit -- one loop, routed by table -- and
    sums the counts for the merge audit event.

    :param repointed: per table, the keys repointed from source to master: ``(agent_id, memory_id)``,
        ``(agent_id, media_id)``, ``(agent_id, content_id)``, ``(agent_id, chunk_id)``
    :ptype repointed: dict[str, list[tuple[Any, ...]]]
    :param removed: per table, the keys the alias-collision delete removed: under ``memories`` the
        colliding source memories deleted because the master holds their alias, under every other
        table the rows that delete cascaded to (``memory_consolidations`` keys are
        ``(agent_id, consolidated_memory_id, source_memory_id)``)
    :ptype removed: dict[str, list[tuple[Any, ...]]]
    """

    repointed: dict[str, list[tuple[Any, ...]]] = field(default_factory=lambda: _per_table(_REPOINTED_TABLES))
    removed: dict[str, list[tuple[Any, ...]]] = field(default_factory=lambda: _per_table(_REMOVED_TABLES))

    @property
    def evict(self) -> dict[str, list[tuple[Any, ...]]]:
        """every key the merge touched, grouped by table: what the caller evicts after commit.

        a row repointed is served by id with the old owner until evicted; a row removed is served by
        id after L3 lost it. both are evicted the same way, so they are one map.

        :return: ``{table: [key, ...]}``, the repointed keys of a table before its removed ones
        :rtype: dict[str, list[tuple[Any, ...]]]
        """
        merged: dict[str, list[tuple[Any, ...]]] = {}
        for table in (*self.repointed, *self.removed):
            merged[table] = [*self.repointed.get(table, []), *self.removed.get(table, [])]
        return merged


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
    :return: the deleted memories and every child their deletion cascaded to, in
        :attr:`MemoryRepointResult.removed`; nothing repointed yet
    :rtype: MemoryRepointResult
    """
    result = MemoryRepointResult()
    removed = result.removed
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
    removed["media"] = [(row["agent_id"], row["media_id"]) for row in media_rows]
    media_agents, media_ids = _columns(removed["media"])

    content_rows = await conn.fetch(
        f"SELECT agent_id, content_id FROM media_content WHERE (agent_id, media_id) IN {_key_set('$1', '$2')}",
        media_agents,
        media_ids,
    )
    removed["media_content"] = [(row["agent_id"], row["content_id"]) for row in content_rows]

    chunk_rows = await conn.fetch(
        f"SELECT agent_id, chunk_id FROM memory_chunks WHERE (agent_id, memory_id) IN {memories}",
        memory_agents,
        memory_ids,
    )
    removed["memory_chunks"] = [(row["agent_id"], row["chunk_id"]) for row in chunk_rows]

    edge_rows = await conn.fetch(
        "SELECT agent_id, consolidated_memory_id, source_memory_id FROM memory_consolidations WHERE "
        f"(agent_id, consolidated_memory_id) IN {memories} OR (agent_id, source_memory_id) IN {memories}",
        memory_agents,
        memory_ids,
    )
    removed["memory_consolidations"] = [
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
    removed["memories"] = [(row["agent_id"], row["memory_id"]) for row in deleted]
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
    returns the keys touched, grouped by table, for post-commit invalidation
    (:attr:`MemoryRepointResult.evict`) and the merge audit event. idempotent: a re-run finds no source-owned
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
    :return: keys removed + repointed, grouped by table
    :rtype: MemoryRepointResult
    """
    result = await _delete_alias_collisions(
        conn,
        from_user_id=from_user_id,
        to_user_id=to_user_id,
    )
    repointed = result.repointed
    repointed["memories"] = await repoint_user_rows(
        conn,
        table="memories",
        user_column="user_id",
        pk_columns=["agent_id", "memory_id"],
        from_user_id=from_user_id,
        to_user_id=to_user_id,
    )
    repointed["media"] = await repoint_user_rows(
        conn,
        table="media",
        user_column="user_id",
        pk_columns=["agent_id", "media_id"],
        from_user_id=from_user_id,
        to_user_id=to_user_id,
        touch_column=None,
    )
    repointed["media_content"] = await repoint_user_rows(
        conn,
        table="media_content",
        user_column="user_id",
        pk_columns=["agent_id", "content_id"],
        from_user_id=from_user_id,
        to_user_id=to_user_id,
        touch_column=None,
    )
    repointed["memory_chunks"] = await repoint_user_rows(
        conn,
        table="memory_chunks",
        user_column="user_id",
        pk_columns=["agent_id", "chunk_id"],
        from_user_id=from_user_id,
        to_user_id=to_user_id,
        touch_column=None,
    )
    removed = result.removed
    log.info(
        "repointed memories from user %s to %s (deleted %d alias collision(s), cascading to %d media, "
        "%d media_content, %d memory_chunks, %d memory_consolidations; moved %d memories, %d media, "
        "%d media_content, %d memory_chunks)",
        from_user_id,
        to_user_id,
        len(removed["memories"]),
        len(removed["media"]),
        len(removed["media_content"]),
        len(removed["memory_chunks"]),
        len(removed["memory_consolidations"]),
        len(repointed["memories"]),
        len(repointed["media"]),
        len(repointed["media_content"]),
        len(repointed["memory_chunks"]),
        extra={
            "extra_data": {
                "removed": {table: len(keys) for table, keys in removed.items()},
                "repointed": {table: len(keys) for table, keys in repointed.items()},
            }
        },
    )
    return result
