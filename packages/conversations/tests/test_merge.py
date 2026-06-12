"""tests for conversation user-merge repoint."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from threetears.conversations.merge import repoint_user


class _RecordingConn:
    """records UPDATE statements, returns canned RETURNING rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        self.calls.append((sql, params))
        return self._rows


@pytest.mark.asyncio
async def test_repoints_conversations_user_id() -> None:
    """repoint issues a scoped UPDATE and returns the moved composite keys."""
    agent_id = uuid4()
    conv_a = uuid4()
    conv_b = uuid4()
    from_id = uuid4()
    to_id = uuid4()
    conn = _RecordingConn([
        {"agent_id": agent_id, "conversation_id": conv_a},
        {"agent_id": agent_id, "conversation_id": conv_b},
    ])

    moved = await repoint_user(conn, from_user_id=from_id, to_user_id=to_id)

    sql, params = conn.calls[0]
    assert "UPDATE conversations SET user_id = $2" in sql
    assert "WHERE user_id = $1" in sql
    assert "RETURNING agent_id, conversation_id" in sql
    assert params[0] == from_id
    assert params[1] == to_id
    assert moved == [(agent_id, conv_a), (agent_id, conv_b)]


@pytest.mark.asyncio
async def test_no_rows_is_noop() -> None:
    """a source user with no conversations returns an empty list."""
    conn = _RecordingConn([])
    moved = await repoint_user(
        conn, from_user_id=uuid4(), to_user_id=uuid4(),
    )
    assert moved == []
