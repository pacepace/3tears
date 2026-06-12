"""tests for the user-merge repoint helper."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from threetears.core.collections.merge import repoint_user_rows


class _RecordingConn:
    """records the executed UPDATE and returns canned RETURNING rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.sql: str | None = None
        self.params: tuple[Any, ...] = ()

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        self.sql = sql
        self.params = params
        return self._rows


@pytest.mark.asyncio
async def test_builds_scoped_update_with_touch() -> None:
    """the helper updates the user column where it matches the source."""
    from_id = uuid4()
    to_id = uuid4()
    pk1 = uuid4()
    conn = _RecordingConn([{"id": pk1}])

    result = await repoint_user_rows(
        conn,
        table="channel_identities",
        user_column="user_id",
        pk_columns=["id"],
        from_user_id=from_id,
        to_user_id=to_id,
    )

    assert "UPDATE channel_identities SET user_id = $2" in conn.sql
    assert "date_updated = $3" in conn.sql
    assert "WHERE user_id = $1" in conn.sql
    assert "RETURNING id" in conn.sql
    # $1 source, $2 master, $3 touch timestamp
    assert conn.params[0] == from_id
    assert conn.params[1] == to_id
    assert len(conn.params) == 3
    assert result == [(pk1,)]


@pytest.mark.asyncio
async def test_composite_pk_returns_tuples() -> None:
    """composite-pk tables return the full key tuple per row."""
    conn = _RecordingConn([
        {"agent_id": "a1", "id": "c1"},
        {"agent_id": "a2", "id": "c2"},
    ])

    result = await repoint_user_rows(
        conn,
        table="conversations",
        user_column="user_id",
        pk_columns=["agent_id", "id"],
        from_user_id=uuid4(),
        to_user_id=uuid4(),
    )

    assert "RETURNING agent_id, id" in conn.sql
    assert result == [("a1", "c1"), ("a2", "c2")]


@pytest.mark.asyncio
async def test_no_touch_column_omits_set() -> None:
    """touch_column=None issues only the user-column SET (two params)."""
    conn = _RecordingConn([])

    await repoint_user_rows(
        conn,
        table="gateway_usage",
        user_column="user_id",
        pk_columns=["id"],
        from_user_id=uuid4(),
        to_user_id=uuid4(),
        touch_column=None,
    )

    assert "date_updated" not in conn.sql
    assert conn.sql.count("$") == 2  # $1 where, $2 set, no touch param
    assert len(conn.params) == 2


@pytest.mark.asyncio
async def test_custom_user_column() -> None:
    """a non-default ownership column (e.g. member_id) is honored."""
    conn = _RecordingConn([])

    await repoint_user_rows(
        conn,
        table="promotion_requests",
        user_column="requested_by_user_id",
        pk_columns=["id"],
        from_user_id=uuid4(),
        to_user_id=uuid4(),
        touch_column=None,
    )

    assert "SET requested_by_user_id = $2" in conn.sql
    assert "WHERE requested_by_user_id = $1" in conn.sql
