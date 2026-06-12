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
    conn = _RoutingConn({
        "DELETE FROM memories": [{"agent_id": agent, "memory_id": uuid4()}],
        "UPDATE memories": [{"agent_id": agent, "memory_id": uuid4()}],
        "UPDATE media ": [{"agent_id": agent, "media_id": uuid4()}],
        "UPDATE media_content": [{"agent_id": agent, "content_id": uuid4()}],
        "UPDATE memory_chunks": [{"agent_id": agent, "chunk_id": uuid4()}],
    })

    result = await repoint_user(conn, from_user_id=from_id, to_user_id=to_id)

    assert isinstance(result, MemoryRepointResult)
    # the DELETE must run before the memories repoint or the unique index
    # on (agent_id, user_id, alias) would reject the flipped survivor.
    assert conn.order[0] == "DELETE FROM memories"
    assert conn.order.index("DELETE FROM memories") < conn.order.index(
        "UPDATE memories",
    )
    assert len(result.alias_collisions_deleted) == 1
    assert len(result.memories) == 1
    assert len(result.media) == 1
    assert len(result.media_content) == 1
    assert len(result.memory_chunks) == 1


@pytest.mark.asyncio
async def test_child_tables_repoint_without_touch() -> None:
    """media / media_content / memory_chunks have no date_updated to stamp."""
    captured: list[tuple[str, tuple[Any, ...]]] = []

    class _CapturingConn:
        async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
            captured.append((sql, params))
            return []

    await repoint_user(
        _CapturingConn(), from_user_id=uuid4(), to_user_id=uuid4(),
    )

    by_table = {sql.split()[1] if sql.startswith("UPDATE") else "DELETE": params
                for sql, params in captured}
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
