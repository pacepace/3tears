"""tests for memory-table user-merge repoint (alias collision + repoint)."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from threetears.agent.memory.merge import MemoryRepointResult, repoint_user


class _RoutingConn:
    """returns canned rows keyed by a substring of the SQL.

    records every statement in order so a test can assert the DELETE
    runs before the repointing UPDATEs.
    """

    def __init__(self, routes: dict[str, list[dict[str, Any]]]) -> None:
        self._routes = routes
        self.order: list[str] = []

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        for needle, rows in self._routes.items():
            if needle in sql:
                self.order.append(needle)
                return rows
        raise AssertionError(f"unexpected SQL: {sql}")


@pytest.mark.asyncio
async def test_deletes_alias_collisions_before_repoint() -> None:
    """colliding source memories are deleted, then survivors repoint."""
    agent = uuid4()
    from_id = uuid4()
    to_id = uuid4()
    colliding = uuid4()
    conn = _RoutingConn(
        {
            "DELETE FROM memories": [{"agent_id": agent, "memory_id": colliding}],
            "FROM memories m WHERE": [{"agent_id": agent, "memory_id": colliding}],
            "FROM media_content": [],
            "FROM media WHERE": [],
            "FROM memory_chunks WHERE": [],
            "FROM memory_consolidations": [],
            "UPDATE memories": [{"agent_id": agent, "memory_id": uuid4()}],
            "UPDATE media ": [{"agent_id": agent, "media_id": uuid4()}],
            "UPDATE media_content": [{"agent_id": agent, "content_id": uuid4()}],
            "UPDATE memory_chunks": [{"agent_id": agent, "chunk_id": uuid4()}],
        }
    )

    result = await repoint_user(conn, from_user_id=from_id, to_user_id=to_id)

    assert isinstance(result, MemoryRepointResult)
    # the DELETE must run before the memories repoint or the unique index
    # on (agent_id, user_id, alias) would reject the flipped survivor.
    # the DELETE is the first write: only the lookups that name what it will cascade to precede it.
    delete_at = conn.order.index("DELETE FROM memories")
    assert all(not step.startswith("UPDATE") for step in conn.order[:delete_at])
    assert conn.order.index("DELETE FROM memories") < conn.order.index(
        "UPDATE memories",
    )
    assert len(result.removed["memories"]) == 1
    assert len(result.repointed["memories"]) == 1
    assert len(result.repointed["media"]) == 1
    assert len(result.repointed["media_content"]) == 1
    assert len(result.repointed["memory_chunks"]) == 1


@pytest.mark.asyncio
async def test_child_tables_repoint_without_touch() -> None:
    """media / media_content / memory_chunks have no date_updated to stamp."""
    captured: list[tuple[str, tuple[Any, ...]]] = []

    class _CapturingConn:
        async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
            captured.append((sql, params))
            return []

    await repoint_user(
        _CapturingConn(),
        from_user_id=uuid4(),
        to_user_id=uuid4(),
    )

    by_table = {sql.split()[1] if sql.startswith("UPDATE") else "DELETE": params for sql, params in captured}
    # children: only $1 (where) + $2 (set) — no $3 touch parameter.
    assert len(by_table["media"]) == 2
    assert len(by_table["media_content"]) == 2
    assert len(by_table["memory_chunks"]) == 2
    # memories: $1, $2, plus $3 date_updated touch.
    assert len(by_table["memories"]) == 3


@pytest.mark.asyncio
async def test_alias_delete_is_scoped_to_source_and_master() -> None:
    """the collision DELETE filters source rows that clash with master."""
    captured: dict[str, str] = {}

    class _SqlConn:
        async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
            if sql.startswith("DELETE"):
                captured["delete"] = sql
            if "FROM memories m WHERE" in sql and sql.startswith("SELECT"):
                # one colliding memory, so the DELETE runs.
                return [{"agent_id": uuid4(), "memory_id": uuid4()}]
            return []

    from_id = uuid4()
    to_id = uuid4()
    await repoint_user(_SqlConn(), from_user_id=from_id, to_user_id=to_id)

    delete_sql = captured["delete"]
    assert "m.user_id = $1" in delete_sql  # source rows only
    assert "alias IS NOT NULL" in delete_sql
    assert "m2.user_id = $2" in delete_sql  # collide with master
    assert "m2.alias = m.alias" in delete_sql
    assert "m2.agent_id = m.agent_id" in delete_sql


@pytest.mark.asyncio
async def test_the_children_an_alias_collision_cascades_to_are_named() -> None:
    """the collision DELETE cascades to children it does not return; the result names them.

    The hub evicts every key the merge removed from every pod's cache. A child the cascade took
    and nobody named stayed cached on every pod that had read it, served by id after L3 lost it.
    """
    agent = uuid4()
    memory_id, media_id = uuid4(), uuid4()
    content_id, chunk_id, gist_id = uuid4(), uuid4(), uuid4()
    conn = _RoutingConn(
        {
            "DELETE FROM memories": [{"agent_id": agent, "memory_id": memory_id}],
            "FROM memories m WHERE": [{"agent_id": agent, "memory_id": memory_id}],
            "FROM media_content": [{"agent_id": agent, "content_id": content_id}],
            "FROM media WHERE": [{"agent_id": agent, "media_id": media_id}],
            "FROM memory_chunks WHERE": [{"agent_id": agent, "chunk_id": chunk_id}],
            "FROM memory_consolidations": [
                {"agent_id": agent, "consolidated_memory_id": gist_id, "source_memory_id": memory_id}
            ],
            "UPDATE memories": [],
            "UPDATE media ": [],
            "UPDATE media_content": [],
            "UPDATE memory_chunks": [],
        }
    )

    result = await repoint_user(conn, from_user_id=uuid4(), to_user_id=uuid4())

    assert result.removed == {
        "memories": [(agent, memory_id)],
        "media": [(agent, media_id)],
        "media_content": [(agent, content_id)],
        "memory_chunks": [(agent, chunk_id)],
        "memory_consolidations": [(agent, gist_id, memory_id)],
    }
    # every child is named before the DELETE takes it, inside the same transaction.
    delete_at = conn.order.index("DELETE FROM memories")
    for lookup in ("FROM media WHERE", "FROM media_content", "FROM memory_chunks WHERE", "FROM memory_consolidations"):
        assert conn.order.index(lookup) < delete_at, f"{lookup} ran after the DELETE had cascaded"


@pytest.mark.asyncio
async def test_evict_names_every_touched_key_by_table_in_one_map() -> None:
    """the caller evicts with one loop over ``evict``: repointed and removed keys, grouped by table."""
    agent = uuid4()
    deleted, cascaded_media, moved, moved_media = uuid4(), uuid4(), uuid4(), uuid4()
    conn = _RoutingConn(
        {
            "DELETE FROM memories": [{"agent_id": agent, "memory_id": deleted}],
            "FROM memories m WHERE": [{"agent_id": agent, "memory_id": deleted}],
            "FROM media_content": [],
            "FROM media WHERE": [{"agent_id": agent, "media_id": cascaded_media}],
            "FROM memory_chunks WHERE": [],
            "FROM memory_consolidations": [],
            "UPDATE memories": [{"agent_id": agent, "memory_id": moved}],
            "UPDATE media ": [{"agent_id": agent, "media_id": moved_media}],
            "UPDATE media_content": [],
            "UPDATE memory_chunks": [],
        }
    )

    result = await repoint_user(conn, from_user_id=uuid4(), to_user_id=uuid4())

    assert result.evict == {
        "memories": [(agent, moved), (agent, deleted)],
        "media": [(agent, moved_media), (agent, cascaded_media)],
        "media_content": [],
        "memory_chunks": [],
        "memory_consolidations": [],
    }


def _cascade_closure(root: str) -> set[str]:
    """every table an ``ON DELETE CASCADE`` chain reaches from ``root``, root included, per the declared schemas."""
    from threetears.agent.memory import collections as memory_collections
    from threetears.core.collections.schema_backed import SchemaBackedCollection

    schemas = [
        value.schema
        for value in vars(memory_collections).values()
        if isinstance(value, type) and issubclass(value, SchemaBackedCollection) and "schema" in vars(value)
    ]
    assert schemas, "found no memory table schema to read the cascade from"
    reached = {root}
    grew = True
    while grew:
        grew = False
        for schema in schemas:
            for fk in schema.foreign_keys:
                if fk.on_delete == "CASCADE" and fk.ref_table in reached and schema.name not in reached:
                    reached.add(schema.name)
                    grew = True
    return reached


def test_the_removed_tables_are_exactly_what_a_memory_delete_cascades_to() -> None:
    """a new ``ON DELETE CASCADE`` onto memories (or media) must be named by the merge, or this fails."""
    closure = _cascade_closure("memories")
    assert closure != {"memories"}, "read no cascade at all: the schema walk is broken, not the merge"
    assert set(MemoryRepointResult().removed) == closure


@pytest.mark.asyncio
async def test_the_summary_log_counts_what_the_cascade_removed(caplog: pytest.LogCaptureFixture) -> None:
    """an operator chasing a stale row after a merge can tell from the log whether the cascade removed it."""
    agent = uuid4()
    memory_id = uuid4()
    conn = _RoutingConn(
        {
            "DELETE FROM memories": [{"agent_id": agent, "memory_id": memory_id}],
            "FROM memories m WHERE": [{"agent_id": agent, "memory_id": memory_id}],
            "FROM media_content": [{"agent_id": agent, "content_id": uuid4()}],
            "FROM media WHERE": [{"agent_id": agent, "media_id": uuid4()}],
            "FROM memory_chunks WHERE": [
                {"agent_id": agent, "chunk_id": uuid4()},
                {"agent_id": agent, "chunk_id": uuid4()},
            ],
            "FROM memory_consolidations": [],
            "UPDATE memories": [],
            "UPDATE media ": [],
            "UPDATE media_content": [],
            "UPDATE memory_chunks": [],
        }
    )

    with caplog.at_level("INFO", logger="threetears.agent.memory.merge"):
        await repoint_user(conn, from_user_id=uuid4(), to_user_id=uuid4())

    [record] = [r for r in caplog.records if r.name == "threetears.agent.memory.merge" and r.levelname == "INFO"]
    assert record.extra_data["removed"] == {  # type: ignore[attr-defined]
        "memories": 1,
        "media": 1,
        "media_content": 1,
        "memory_chunks": 2,
        "memory_consolidations": 0,
    }
    assert "cascading to 1 media, 1 media_content, 2 memory_chunks, 0 memory_consolidations" in record.getMessage()
