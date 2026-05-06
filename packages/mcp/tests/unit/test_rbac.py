"""unit tests for :class:`threetears.mcp.rbac.McpToolGrantCollection`.

every test substitutes the L3 pool / save / find / delete plumbing
with mocks; round-trip behaviour (real Postgres + L1 cache) is
covered by the per-product integration tests in Chunks B and C.
this surface verifies the framework's add/remove/load-all contract
in isolation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from threetears.mcp.rbac import McpToolGrantCollection, McpToolGrantEntity


def _make_collection() -> tuple[McpToolGrantCollection, Any]:
    """build a McpToolGrantCollection with a mock L3 pool attached.

    bypasses the real ``__init__`` so we don't have to stand up a
    full CollectionRegistry for unit tests; the parent
    SchemaBackedCollection's behaviours are tested elsewhere.
    """
    coll = McpToolGrantCollection.__new__(McpToolGrantCollection)
    coll.l3_pool = MagicMock()
    coll.l3_pool.fetch = AsyncMock(return_value=[])
    return coll, coll.l3_pool


class TestAddGrant:
    """``add_grant`` constructs the right row dict and saves it.

    BaseCollection wires private state in ``__init__`` that the
    bypass-init pattern can't reach; instead of constructing the
    entity through it, we mock ``entity_class`` and inspect the
    dict that the framework would build.
    """

    @pytest.mark.asyncio
    async def test_add_grant_constructs_entity_with_passed_fields(self) -> None:
        """save_entity is called with an entity built from the passed fields."""
        coll, _ = _make_collection()
        coll.save_entity = AsyncMock()  # type: ignore[method-assign]

        # capture the dict the framework constructs to feed the entity.
        captured: list[dict[str, Any]] = []

        class _StubEntity:
            def __init__(self, data: dict[str, Any], **_kwargs: Any) -> None:
                captured.append(data)
                self._data = data

            def __getattr__(self, name: str) -> Any:
                return self._data[name]

        # patch the entity_class property at the class level since it's a property.
        original_property = McpToolGrantCollection.entity_class
        McpToolGrantCollection.entity_class = property(  # type: ignore[assignment]
            lambda self: _StubEntity,
        )
        try:
            principal_id = uuid4()
            await coll.add_grant(
                principal_type="user",
                principal_id=principal_id,
                tool_name="list_conversations",
                permission="metallm.conversations.read",
            )
        finally:
            McpToolGrantCollection.entity_class = original_property  # type: ignore[assignment]

        coll.save_entity.assert_awaited_once()
        assert len(captured) == 1
        row = captured[0]
        assert row["principal_type"] == "user"
        assert row["principal_id"] == principal_id
        assert row["tool_name"] == "list_conversations"
        assert row["permission"] == "metallm.conversations.read"
        assert isinstance(row["grant_id"], UUID)
        # date_created is set to a UTC-aware timestamp.
        assert row["date_created"].tzinfo is not None

    @pytest.mark.asyncio
    async def test_add_grant_each_call_has_unique_grant_id(self) -> None:
        """consecutive calls produce distinct grant_ids (uuid4-derived)."""
        coll, _ = _make_collection()
        coll.save_entity = AsyncMock()  # type: ignore[method-assign]

        captured: list[dict[str, Any]] = []

        class _StubEntity:
            def __init__(self, data: dict[str, Any], **_kwargs: Any) -> None:
                captured.append(data)

        original_property = McpToolGrantCollection.entity_class
        McpToolGrantCollection.entity_class = property(  # type: ignore[assignment]
            lambda self: _StubEntity,
        )
        try:
            principal_id = uuid4()
            await coll.add_grant(
                principal_type="user", principal_id=principal_id,
                tool_name="t", permission="p",
            )
            await coll.add_grant(
                principal_type="user", principal_id=principal_id,
                tool_name="t", permission="p",
            )
        finally:
            McpToolGrantCollection.entity_class = original_property  # type: ignore[assignment]

        assert captured[0]["grant_id"] != captured[1]["grant_id"]


class TestRemoveGrant:
    """``remove_grant`` lookups + delete + missing-row handling.

    Uses the public :meth:`BaseCollection.get` + :meth:`BaseCollection.delete`
    extension seams (the prior ``find_by_id`` / ``delete_entity`` shorthand
    does not exist on the canonical BaseCollection surface).
    """

    @pytest.mark.asyncio
    async def test_remove_grant_existing_row_returns_true(self) -> None:
        """a found entity is deleted; True returned."""
        coll, _ = _make_collection()
        existing = MagicMock()
        coll.get = AsyncMock(return_value=existing)  # type: ignore[method-assign]
        coll.delete = AsyncMock(return_value=True)  # type: ignore[method-assign]

        target_id = uuid4()
        result = await coll.remove_grant(target_id)

        assert result is True
        coll.delete.assert_awaited_once_with(target_id)

    @pytest.mark.asyncio
    async def test_remove_grant_missing_row_returns_false(self) -> None:
        """no matching row -> False, no delete attempted."""
        coll, _ = _make_collection()
        coll.get = AsyncMock(return_value=None)  # type: ignore[method-assign]
        coll.delete = AsyncMock()  # type: ignore[method-assign]

        result = await coll.remove_grant(uuid4())

        assert result is False
        coll.delete.assert_not_called()


class TestLoadAllGrants:
    """``load_all_grants`` reads via l3_pool; surfaces RuntimeError when pool absent."""

    @pytest.mark.asyncio
    async def test_load_all_grants_returns_dict_rows(self) -> None:
        """asyncpg-shape rows are returned as dicts (one per row)."""
        coll, pool = _make_collection()
        principal_id = uuid4()
        grant_id = uuid4()
        # asyncpg.Record-like: dict() coerces to dict; MagicMock won't,
        # so we use plain dicts and the dict(row) call passes through.
        pool.fetch = AsyncMock(return_value=[
            {
                "grant_id": grant_id,
                "principal_type": "user",
                "principal_id": principal_id,
                "tool_name": "list_conversations",
                "permission": "metallm.conversations.read",
                "date_created": "2026-05-05T00:00:00+00:00",
            },
        ])

        rows = await coll.load_all_grants()

        assert len(rows) == 1
        assert rows[0]["grant_id"] == grant_id
        assert rows[0]["principal_id"] == principal_id
        assert rows[0]["permission"] == "metallm.conversations.read"

    @pytest.mark.asyncio
    async def test_load_all_grants_empty_when_no_rows(self) -> None:
        """empty result returns empty list, not None."""
        coll, pool = _make_collection()
        pool.fetch = AsyncMock(return_value=[])
        rows = await coll.load_all_grants()
        assert rows == []

    @pytest.mark.asyncio
    async def test_load_all_grants_raises_when_l3_pool_unconfigured(self) -> None:
        """unset l3_pool fails closed with a clear RuntimeError."""
        coll = McpToolGrantCollection.__new__(McpToolGrantCollection)
        coll.l3_pool = None  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="L3 pool"):
            await coll.load_all_grants()
