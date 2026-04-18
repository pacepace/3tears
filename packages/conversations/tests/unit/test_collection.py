"""
unit tests for :class:`ConversationsCollection`.

exercises the L3 SQL paths (``_fetch_from_postgres``,
``_save_to_postgres``, ``_delete_from_postgres``) via an asyncpg mock,
plus the serialize / deserialize round-trip used by L2.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID

from uuid import uuid7

from threetears.conversations.collection import ConversationsCollection
from threetears.conversations.entity import Conversation


def _sample_data() -> dict[str, Any]:
    """
    build a fully populated conversation row dict.

    :return: sample row dict
    :rtype: dict[str, Any]
    """
    now = datetime.now(UTC)
    return {
        "id": uuid7(),
        "agent_id": uuid7(),
        "customer_id": uuid7(),
        "user_id": uuid7(),
        "channel_type": "slack",
        "conversation_ref": "C1234567890",
        "status": "active",
        "summary": "initial summary",
        "date_created": now,
        "date_updated": now,
        "date_last_message": now,
        "metadata": {"source": "test"},
    }


def _make_pg_mock(store: dict[str, dict[str, Any]] | None = None) -> AsyncMock:
    """
    build a mock asyncpg pool backed by an in-memory dict.

    :param store: optional existing row dict
    :ptype store: dict[str, dict[str, Any]] | None
    :return: async-mock pool
    :rtype: AsyncMock
    """
    if store is None:
        store = {}
    pg = AsyncMock()

    async def _fetchrow(query: str, *args: object) -> dict[str, Any] | None:
        """
        simulate ``SELECT * FROM conversations WHERE id = $1``.

        :param query: SQL text (ignored by the mock)
        :ptype query: str
        :param args: positional params
        :ptype args: object
        :return: stored row dict or ``None``
        :rtype: dict[str, Any] | None
        """
        entity_id = args[0] if args else None
        return store.get(str(entity_id))

    async def _execute(query: str, *args: object) -> str:
        """
        simulate INSERT/UPDATE/DELETE against the in-memory store.

        :param query: SQL text
        :ptype query: str
        :param args: positional params
        :ptype args: object
        :return: status string compatible with asyncpg
        :rtype: str
        """
        result: str
        if "INSERT" in query:
            keys = [
                "id",
                "agent_id",
                "customer_id",
                "user_id",
                "channel_type",
                "conversation_ref",
                "status",
                "summary",
                "date_created",
                "date_updated",
                "date_last_message",
                "metadata",
            ]
            row = dict(zip(keys, args, strict=False))
            store[str(row["id"])] = row
            result = "INSERT 0 1"
            return result
        if "UPDATE" in query:
            entity_id = str(args[0])
            existing = store.get(entity_id)
            if existing is None:
                result = "UPDATE 0"
                return result
            existing["status"] = args[1]
            existing["summary"] = args[2]
            existing["date_updated"] = args[3]
            existing["date_last_message"] = args[4]
            existing["metadata"] = args[5]
            result = "UPDATE 1"
            return result
        if "DELETE" in query:
            entity_id = str(args[0])
            if entity_id in store:
                del store[entity_id]
            result = "DELETE 1"
            return result
        result = "OK"
        return result

    pg.fetchrow = AsyncMock(side_effect=_fetchrow)
    pg.execute = AsyncMock(side_effect=_execute)
    pg.fetch = AsyncMock(return_value=[])
    return pg


class TestFetchFromPostgres:
    """tests for L3 read path."""

    async def test_fetch_returns_row_dict(self) -> None:
        """_fetch_from_postgres returns the row as a dict when present."""
        data = _sample_data()
        store: dict[str, dict[str, Any]] = {str(data["id"]): data}
        pg = _make_pg_mock(store)
        collection = ConversationsCollection.__new__(ConversationsCollection)
        collection._postgres_pool = pg  # type: ignore[attr-defined]

        result = await collection._fetch_from_postgres(data["id"])

        assert result == data

    async def test_fetch_returns_none_when_missing(self) -> None:
        """_fetch_from_postgres returns None on a miss."""
        pg = _make_pg_mock({})
        collection = ConversationsCollection.__new__(ConversationsCollection)
        collection._postgres_pool = pg  # type: ignore[attr-defined]

        result = await collection._fetch_from_postgres(uuid7())

        assert result is None


class TestSaveToPostgres:
    """tests for L3 write path."""

    async def test_insert_new_row(self) -> None:
        """_save_to_postgres insert path stores a row and reports 1 affected."""
        data = _sample_data()
        l3: dict[str, dict[str, Any]] = {}
        pg = _make_pg_mock(l3)
        collection = ConversationsCollection.__new__(ConversationsCollection)
        collection._postgres_pool = pg  # type: ignore[attr-defined]

        affected = await collection._save_to_postgres(data)

        assert affected == 1
        assert str(data["id"]) in l3

    async def test_update_existing_row_with_timestamp(self) -> None:
        """update path respects optimistic concurrency timestamp."""
        data = _sample_data()
        l3: dict[str, dict[str, Any]] = {str(data["id"]): data}
        pg = _make_pg_mock(l3)
        collection = ConversationsCollection.__new__(ConversationsCollection)
        collection._postgres_pool = pg  # type: ignore[attr-defined]

        updated = dict(data)
        updated["status"] = "closed"
        affected = await collection._save_to_postgres(
            updated, original_timestamp=data["date_updated"]
        )

        assert affected == 1
        assert l3[str(data["id"])]["status"] == "closed"


class TestDeleteFromPostgres:
    """tests for L3 delete path."""

    async def test_delete_removes_row(self) -> None:
        """_delete_from_postgres removes the row from the backing store."""
        data = _sample_data()
        l3: dict[str, dict[str, Any]] = {str(data["id"]): data}
        pg = _make_pg_mock(l3)
        collection = ConversationsCollection.__new__(ConversationsCollection)
        collection._postgres_pool = pg  # type: ignore[attr-defined]

        await collection._delete_from_postgres(data["id"])

        assert str(data["id"]) not in l3


class TestSerializationRoundTrip:
    """tests for L2 JSON serialize + deserialize."""

    async def test_round_trip_preserves_scalar_fields(self) -> None:
        """serialize + deserialize preserves strings and scalars."""
        data = _sample_data()
        collection = ConversationsCollection.__new__(ConversationsCollection)

        payload = collection._serialize(data)
        restored = collection._deserialize(payload)

        assert restored["channel_type"] == data["channel_type"]
        assert restored["status"] == data["status"]
        assert restored["summary"] == data["summary"]

    async def test_round_trip_preserves_uuids(self) -> None:
        """deserialize reconstructs UUID-typed identity fields."""
        data = _sample_data()
        collection = ConversationsCollection.__new__(ConversationsCollection)

        payload = collection._serialize(data)
        restored = collection._deserialize(payload)

        assert isinstance(restored["agent_id"], UUID)
        assert isinstance(restored["customer_id"], UUID)
        assert isinstance(restored["user_id"], UUID)
        assert restored["agent_id"] == data["agent_id"]

    async def test_round_trip_preserves_datetimes(self) -> None:
        """deserialize reconstructs datetime fields from ISO-8601 text."""
        data = _sample_data()
        collection = ConversationsCollection.__new__(ConversationsCollection)

        payload = collection._serialize(data)
        restored = collection._deserialize(payload)

        assert isinstance(restored["date_created"], datetime)
        assert restored["date_created"] == data["date_created"]


class TestTableAndEntityClass:
    """tests for static class properties used by BaseCollection."""

    async def test_table_name_is_conversations(self) -> None:
        """collection.table_name returns the canonical table name."""
        collection = ConversationsCollection.__new__(ConversationsCollection)
        assert collection.table_name == "conversations"

    async def test_entity_class_is_conversation(self) -> None:
        """collection.entity_class returns :class:`Conversation`."""
        collection = ConversationsCollection.__new__(ConversationsCollection)
        assert collection.entity_class is Conversation
