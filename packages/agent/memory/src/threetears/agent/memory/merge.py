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
from threetears.core.keyed import Fragment, KeyBatch, Lock, delete_keyed, read_by, read_keyed
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


async def lock_colliding_memories(conn: Any, *, from_user_id: UUID, to_user_id: UUID, agent_id: UUID) -> list[Any]:
    """lock the source's memories whose alias the master already holds under the agent, ``FOR UPDATE``.

    :param conn: asyncpg transaction connection bound to the agent schema
    :ptype conn: Any
    :param from_user_id: source user whose colliding memories are locked
    :ptype from_user_id: UUID
    :param to_user_id: master user whose memories win the alias
    :ptype to_user_id: UUID
    :param agent_id: the agent the schema holds
    :ptype agent_id: UUID
    :return: the locked memories' ``agent_id`` and ``memory_id``
    :rtype: list[Any]
    """
    # the collision guard: a source memory whose alias a master memory under the same agent already holds. Beside
    # the pins (the schema's agent and the source user) and ``alias IS NOT NULL`` (the partial index's predicate), a
    # correlated EXISTS bound to the master user and matched on the outer row's agent and alias: the source's named
    # memories are read through ix_memories_user_alias (agent_id, user_id, alias) WHERE alias IS NOT NULL, the
    # master's alias by a point of the same index (test_index_plans::test_memories_user_alias)
    return await read_by(
        conn,
        "memories",
        by={"agent_id": agent_id, "user_id": from_user_id},
        columns=["agent_id", "memory_id"],
        lock=Lock.UPDATE,
        where=[
            Fragment("t.alias IS NOT NULL"),
            Fragment(
                "EXISTS (SELECT 1 FROM memories m2 WHERE m2.user_id = $1"
                " AND m2.agent_id = t.agent_id AND m2.alias = t.alias)",
                (to_user_id,),
            ),
        ],
    )


async def delete_colliding_memories(
    conn: Any, *, from_user_id: UUID, to_user_id: UUID, agent_id: UUID, memory_ids: list[Any]
) -> list[dict[str, Any]]:
    """delete exactly the locked memories, the collision guard re-checked as the hand-written DELETE re-checked it.

    :param conn: asyncpg transaction connection bound to the agent schema
    :ptype conn: Any
    :param from_user_id: source user whose colliding memories are deleted
    :ptype from_user_id: UUID
    :param to_user_id: master user whose memories win the alias
    :ptype to_user_id: UUID
    :param agent_id: the agent the schema holds
    :ptype agent_id: UUID
    :param memory_ids: the memories :func:`lock_colliding_memories` locked
    :ptype memory_ids: list[Any]
    :return: the deleted memories' ``agent_id`` and ``memory_id``
    :rtype: list[dict[str, Any]]
    """
    # by the primary key's (agent_id, memory_id) under the source user, the guard as the lock wrote it: the hand-written
    # DELETE named the locked keys as an ``(agent_id, memory_id) IN unnest(...)`` tuple set beside the predicate
    deleted = await delete_keyed(
        conn,
        "memories",
        by={"agent_id": agent_id, "user_id": from_user_id},
        keys=KeyBatch(("memory_id",), [list(memory_ids)], casts=("uuid",)),
        where=[
            Fragment("memories.alias IS NOT NULL"),
            Fragment(
                "EXISTS (SELECT 1 FROM memories m2 WHERE m2.user_id = $1"
                " AND m2.agent_id = memories.agent_id AND m2.alias = memories.alias)",
                (to_user_id,),
            ),
        ],
        returning=["agent_id", "memory_id"],
    )
    return deleted.rows


async def _delete_alias_collisions(
    conn: Any,
    *,
    from_user_id: UUID,
    to_user_id: UUID,
    agent_id: UUID,
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
    :param agent_id: the agent the schema holds, pinned beside the users
    :ptype agent_id: UUID
    :return: the deleted memories and every child their deletion cascaded to, in
        :attr:`MemoryRepointResult.removed`; nothing repointed yet
    :rtype: MemoryRepointResult
    """
    result = MemoryRepointResult()
    removed = result.removed
    locked = await lock_colliding_memories(conn, from_user_id=from_user_id, to_user_id=to_user_id, agent_id=agent_id)
    if not locked:
        return result
    # every child read by key: the colliding memories (or their media) are all the schema's agent's, so each read pins
    # the agent and names the parents, through the index each table's cascade follows
    pinned = {"agent_id": agent_id}
    named = [row["memory_id"] for row in locked]

    media_rows = await read_keyed(
        conn,
        "media",
        keys=KeyBatch(("memory_id",), [named], casts=("uuid",)),
        by=pinned,
        columns=["agent_id", "media_id"],
        lock=Lock.UPDATE,
    )
    removed["media"] = [(row["agent_id"], row["media_id"]) for row in media_rows]

    content_rows = await read_keyed(
        conn,
        "media_content",
        keys=KeyBatch(("media_id",), [[media_id for _, media_id in removed["media"]]], casts=("uuid",)),
        by=pinned,
        columns=["agent_id", "content_id"],
    )
    removed["media_content"] = [(row["agent_id"], row["content_id"]) for row in content_rows]

    chunk_rows = await read_keyed(
        conn,
        "memory_chunks",
        keys=KeyBatch(("memory_id",), [named], casts=("uuid",)),
        by=pinned,
        columns=["agent_id", "chunk_id"],
    )
    removed["memory_chunks"] = [(row["agent_id"], row["chunk_id"]) for row in chunk_rows]

    # an edge with either end on a colliding memory: by the consolidated end (the key's lead), then by the source end
    # (idx_memory_consolidations_source); an edge between two colliding memories is named once
    edge_columns = ["agent_id", "consolidated_memory_id", "source_memory_id"]
    by_consolidated = await read_keyed(
        conn,
        "memory_consolidations",
        keys=KeyBatch(("consolidated_memory_id",), [named], casts=("uuid",)),
        by=pinned,
        columns=edge_columns,
    )
    by_source = await read_keyed(
        conn,
        "memory_consolidations",
        keys=KeyBatch(("source_memory_id",), [named], casts=("uuid",)),
        by=pinned,
        columns=edge_columns,
    )
    edges = {
        (row["agent_id"], row["consolidated_memory_id"], row["source_memory_id"]): None
        for row in (*by_consolidated, *by_source)
    }
    removed["memory_consolidations"] = list(edges)

    deleted = await delete_colliding_memories(
        conn, from_user_id=from_user_id, to_user_id=to_user_id, agent_id=agent_id, memory_ids=named
    )
    removed["memories"] = [(row["agent_id"], row["memory_id"]) for row in deleted]
    return result


async def repoint_user(
    conn: Any,
    *,
    from_user_id: UUID,
    to_user_id: UUID,
    agent_id: UUID,
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

    the caller names the agent whose schema ``conn`` is bound to as
    ``agent_id``: every statement pins ``agent_id`` beside ``user_id`` and
    reads the table's ``(agent_id, user_id)`` index. the schema holds only
    that agent's rows, so it moves every row the source user holds there.

    :param conn: asyncpg transaction connection bound to the agent schema
    :ptype conn: Any
    :param from_user_id: source user whose memories move
    :ptype from_user_id: UUID
    :param to_user_id: master user the memories move to
    :ptype to_user_id: UUID
    :param agent_id: the agent the schema holds
    :ptype agent_id: UUID
    :return: keys removed + repointed, grouped by table
    :rtype: MemoryRepointResult
    """
    result = await _delete_alias_collisions(
        conn,
        from_user_id=from_user_id,
        to_user_id=to_user_id,
        agent_id=agent_id,
    )
    repointed = result.repointed
    repointed["memories"] = await repoint_user_rows(
        conn,
        user_column="user_id",
        table="memories",
        agent_id=agent_id,
        pk_columns=["agent_id", "memory_id"],
        from_user_id=from_user_id,
        to_user_id=to_user_id,
    )
    repointed["media"] = await repoint_user_rows(
        conn,
        user_column="user_id",
        table="media",
        agent_id=agent_id,
        pk_columns=["agent_id", "media_id"],
        from_user_id=from_user_id,
        to_user_id=to_user_id,
        touch_column=None,
    )
    repointed["media_content"] = await repoint_user_rows(
        conn,
        user_column="user_id",
        table="media_content",
        agent_id=agent_id,
        pk_columns=["agent_id", "content_id"],
        from_user_id=from_user_id,
        to_user_id=to_user_id,
        touch_column=None,
    )
    repointed["memory_chunks"] = await repoint_user_rows(
        conn,
        user_column="user_id",
        table="memory_chunks",
        agent_id=agent_id,
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
