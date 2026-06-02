"""
unit tests for :class:`ConversationsCollection`.

exercises the L3 SQL paths (``fetch_from_postgres``,
``save_to_postgres``, ``delete_from_postgres``) via an asyncpg mock,
plus the serialize / deserialize round-trip used by L2.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
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
        "conversation_id": uuid7(),
        "agent_id": uuid7(),
        "customer_id": uuid7(),
        "user_id": uuid7(),
        "channel_type": "slack",
        "conversation_ref": "C1234567890",
        "name": None,
        "status": "active",
        "summary": "initial summary",
        "date_created": now,
        "date_updated": now,
        "date_last_message": now,
        "metadata": {"source": "test"},
        "message_count": 0,
        # v006: per-row FTS language column.
        "language": "english",
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
        simulate ``SELECT * FROM conversations WHERE agent_id = $1 AND conversation_id = $2``.

        composite-pk fetch: arg[0] is agent_id, arg[1] is conversation_id.

        :param query: SQL text (ignored by the mock)
        :ptype query: str
        :param args: positional params
        :ptype args: object
        :return: stored row dict or ``None``
        :rtype: dict[str, Any] | None
        """
        entity_id = args[1] if len(args) > 1 else None
        return store.get(str(entity_id))

    async def _execute(query: str, *args: object) -> str:
        """
        simulate INSERT/UPDATE/DELETE against the in-memory store.

        composite-pk schema column order (matches SchemaBackedCollection
        generator): agent_id, conversation_id, customer_id, user_id,
        channel_type, conversation_ref, name, status, summary,
        date_created, date_updated, date_last_message, metadata,
        message_count.

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
                "agent_id",
                "conversation_id",
                "customer_id",
                "user_id",
                "channel_type",
                "conversation_ref",
                "name",
                "status",
                "summary",
                "date_created",
                "date_updated",
                "date_last_message",
                "metadata",
                "message_count",
                # v006: per-row FTS language column.
                "language",
            ]
            row = dict(zip(keys, args, strict=False))
            store[str(row["conversation_id"])] = row
            result = "INSERT 0 1"
            return result
        if "UPDATE" in query:
            # CAS UPDATE pk-first: $1=agent_id, $2=conversation_id, then mutables, then CAS fence.
            entity_id = str(args[1])
            existing = store.get(entity_id)
            if existing is None:
                result = "UPDATE 0"
                return result
            existing["name"] = args[2]
            existing["status"] = args[3]
            existing["summary"] = args[4]
            existing["date_updated"] = args[5]
            existing["date_last_message"] = args[6]
            existing["metadata"] = args[7]
            existing["message_count"] = args[8]
            # v006: language is a mutable column (changing it
            # re-tokenizes the search_vector via the trigger).
            existing["language"] = args[9]
            result = "UPDATE 1"
            return result
        if "DELETE" in query:
            entity_id = str(args[1])
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
        """fetch_from_postgres returns the row as a dict when present."""
        data = _sample_data()
        store: dict[str, dict[str, Any]] = {str(data["conversation_id"]): data}
        pg = _make_pg_mock(store)
        collection = ConversationsCollection.__new__(ConversationsCollection)
        collection.l3_pool = pg

        result = await collection.fetch_from_postgres((data["agent_id"], data["conversation_id"]))

        assert result == data

    async def test_fetch_returns_none_when_missing(self) -> None:
        """fetch_from_postgres returns None on a miss."""
        pg = _make_pg_mock({})
        collection = ConversationsCollection.__new__(ConversationsCollection)
        collection.l3_pool = pg

        result = await collection.fetch_from_postgres((uuid7(), uuid7()))

        assert result is None


class TestSaveToPostgres:
    """tests for L3 write path."""

    async def test_insert_new_row(self) -> None:
        """save_to_postgres insert path stores a row and reports 1 affected."""
        data = _sample_data()
        l3: dict[str, dict[str, Any]] = {}
        pg = _make_pg_mock(l3)
        collection = ConversationsCollection.__new__(ConversationsCollection)
        collection.l3_pool = pg

        affected = await collection.save_to_postgres(data)

        assert affected == 1
        assert str(data["conversation_id"]) in l3

    async def test_update_existing_row_with_timestamp(self) -> None:
        """update path respects optimistic concurrency timestamp."""
        data = _sample_data()
        l3: dict[str, dict[str, Any]] = {str(data["conversation_id"]): data}
        pg = _make_pg_mock(l3)
        collection = ConversationsCollection.__new__(ConversationsCollection)
        collection.l3_pool = pg

        updated = dict(data)
        updated["status"] = "closed"
        affected = await collection.save_to_postgres(updated, original_timestamp=data["date_updated"])

        assert affected == 1
        assert l3[str(data["conversation_id"])]["status"] == "closed"


class TestDeleteFromPostgres:
    """tests for L3 delete path."""

    async def test_delete_removes_row(self) -> None:
        """delete_from_postgres removes the row from the backing store."""
        data = _sample_data()
        l3: dict[str, dict[str, Any]] = {str(data["conversation_id"]): data}
        pg = _make_pg_mock(l3)
        collection = ConversationsCollection.__new__(ConversationsCollection)
        collection.l3_pool = pg

        await collection.delete_from_postgres((data["agent_id"], data["conversation_id"]))

        assert str(data["conversation_id"]) not in l3


class TestSerializationRoundTrip:
    """tests for L2 JSON serialize + deserialize."""

    async def test_round_trip_preserves_scalar_fields(self) -> None:
        """serialize + deserialize preserves strings and scalars."""
        data = _sample_data()
        collection = ConversationsCollection.__new__(ConversationsCollection)

        payload = collection.serialize(data)
        restored = collection.deserialize(payload)

        assert restored["channel_type"] == data["channel_type"]
        assert restored["status"] == data["status"]
        assert restored["summary"] == data["summary"]

    async def test_round_trip_preserves_uuids(self) -> None:
        """deserialize reconstructs UUID-typed identity fields."""
        data = _sample_data()
        collection = ConversationsCollection.__new__(ConversationsCollection)

        payload = collection.serialize(data)
        restored = collection.deserialize(payload)

        assert isinstance(restored["agent_id"], UUID)
        assert isinstance(restored["customer_id"], UUID)
        assert isinstance(restored["user_id"], UUID)
        assert restored["agent_id"] == data["agent_id"]

    async def test_round_trip_preserves_datetimes(self) -> None:
        """deserialize reconstructs datetime fields from ISO-8601 text."""
        data = _sample_data()
        collection = ConversationsCollection.__new__(ConversationsCollection)

        payload = collection.serialize(data)
        restored = collection.deserialize(payload)

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


class TestSearch:
    """tests for :meth:`ConversationsCollection.search` FTS contract.

    Mocks the L3 ``fetch`` to assert the SQL skeleton stays aligned with
    the v005 migration's column / index / trigger shape. A real
    end-to-end FTS exercise lives in the integration suite under
    ``tests/integration/`` because it requires a live postgres + the
    migration runner.
    """

    async def test_empty_query_returns_empty_list(self) -> None:
        """An empty / whitespace query short-circuits before hitting the
        pool. Without this guard, postgres FTS' empty-query semantics
        would match every row."""
        pg = _make_pg_mock()
        collection = ConversationsCollection.__new__(ConversationsCollection)
        collection.l3_pool = pg

        result_empty = await collection.search(
            agent_id=uuid7(),
            user_id=uuid7(),
            query="",
        )
        result_whitespace = await collection.search(
            agent_id=uuid7(),
            user_id=uuid7(),
            query="   ",
        )
        assert result_empty == []
        assert result_whitespace == []
        pg.fetch.assert_not_called()

    async def test_query_uses_websearch_to_tsquery_with_user_scope(self) -> None:
        """The SQL must use ``websearch_to_tsquery`` (not plainto / to)
        and pin ``user_id`` at the WHERE clause so cross-user leakage is
        impossible at the SQL boundary."""
        pg = _make_pg_mock()
        pg.fetch = AsyncMock(return_value=[])
        collection = ConversationsCollection.__new__(ConversationsCollection)
        collection.l3_pool = pg

        agent_id = uuid7()
        user_id = uuid7()
        await collection.search(
            agent_id=agent_id,
            user_id=user_id,
            query="kerning argument",
            limit=10,
            offset=0,
        )

        pg.fetch.assert_called_once()
        sql, *params = pg.fetch.call_args.args
        assert "websearch_to_tsquery" in sql
        assert "ts_rank_cd" in sql
        assert "search_vector @@" in sql
        assert "agent_id = $1" in sql
        assert "user_id = $2" in sql
        # v006: the FTS config is passed as the $6 bind param cast to
        # regconfig so the trigger and the query agree on language.
        assert "$6::regconfig" in sql
        # status != 'closed' excluded by default (shifted to $7 now
        # that $6 carries the language).
        assert "status != $7" in sql
        # Bound params (positional): (agent_id, user_id, query, limit,
        # offset, query_language, "closed")
        assert params[0] == agent_id
        assert params[1] == user_id
        assert params[2] == "kerning argument"
        assert params[3] == 10
        assert params[4] == 0
        assert params[5] == "english"  # default query_language
        assert params[6] == "closed"

    async def test_include_closed_widens_scope(self) -> None:
        """``include_closed=True`` drops the status filter from the SQL."""
        pg = _make_pg_mock()
        pg.fetch = AsyncMock(return_value=[])
        collection = ConversationsCollection.__new__(ConversationsCollection)
        collection.l3_pool = pg

        await collection.search(
            agent_id=uuid7(),
            user_id=uuid7(),
            query="anything",
            include_closed=True,
        )

        sql, *params = pg.fetch.call_args.args
        assert "status !=" not in sql
        # 6 bound params: agent_id, user_id, query, limit, offset, query_language.
        # No "closed" sentinel (include_closed widens the scope).
        assert len(params) == 6
        assert params[5] == "english"

    async def test_query_language_passed_through_to_sql(self) -> None:
        """v006: callers in polyglot products pass query_language so
        the tsquery tokenizer matches the conversation's stored
        tsvector tokenization. The parameter flows through to the
        $6 bind."""
        pg = _make_pg_mock()
        pg.fetch = AsyncMock(return_value=[])
        collection = ConversationsCollection.__new__(ConversationsCollection)
        collection.l3_pool = pg

        await collection.search(
            agent_id=uuid7(),
            user_id=uuid7(),
            query="trabajo",
            query_language="spanish",
        )

        sql, *params = pg.fetch.call_args.args
        # Language flows through as a bind, not interpolated -- a typo
        # surfaces as a postgres error, never a wrong-tokenization
        # silent match.
        assert "$6::regconfig" in sql
        assert params[5] == "spanish"

    async def test_select_projects_rank_for_ordering(self) -> None:
        """The SELECT projects ``ts_rank_cd(...) AS rank`` so the ORDER
        BY can sort by relevance. ``rank`` is stripped before entity
        construction (exercised in the integration test that runs the
        full migration); this unit test just pins the SQL shape so a
        future refactor doesn't break the relevance-ordered semantics."""
        pg = _make_pg_mock()
        pg.fetch = AsyncMock(return_value=[])
        collection = ConversationsCollection.__new__(ConversationsCollection)
        collection.l3_pool = pg

        await collection.search(
            agent_id=uuid7(),
            user_id=uuid7(),
            query="kerning",
        )

        sql, *_ = pg.fetch.call_args.args
        assert "ts_rank_cd(search_vector," in sql
        assert "AS rank" in sql
        assert "ORDER BY rank DESC" in sql

    async def test_date_bounds_default_field_is_created(self) -> None:
        """``date_after`` / ``date_before`` default to filtering
        ``date_created`` as appended binds, leaving the fixed $1..$6 slots
        and the $4/$5 LIMIT/OFFSET binds intact."""
        pg = _make_pg_mock()
        pg.fetch = AsyncMock(return_value=[])
        collection = ConversationsCollection.__new__(ConversationsCollection)
        collection.l3_pool = pg

        after = datetime(2026, 5, 1, tzinfo=UTC)
        before = datetime(2026, 5, 31, tzinfo=UTC)
        await collection.search(
            agent_id=uuid7(),
            user_id=uuid7(),
            query="moon colony",
            date_after=after,
            date_before=before,
        )

        sql, *params = pg.fetch.call_args.args
        # Default exclude-closed keeps status at $7; the date bounds
        # append from $8.
        assert "status != $7" in sql
        assert "date_created >= $8" in sql
        assert "date_created <= $9" in sql
        # LIMIT/OFFSET stay positionally bound to $4/$5.
        assert "LIMIT $4 OFFSET $5" in sql
        assert params[7] == after
        assert params[8] == before

    async def test_date_field_updated_filters_date_updated(self) -> None:
        """``date_field='updated'`` filters the ``date_updated`` column."""
        pg = _make_pg_mock()
        pg.fetch = AsyncMock(return_value=[])
        collection = ConversationsCollection.__new__(ConversationsCollection)
        collection.l3_pool = pg

        after = datetime(2026, 5, 1, tzinfo=UTC)
        await collection.search(
            agent_id=uuid7(),
            user_id=uuid7(),
            query="x",
            date_field="updated",
            date_after=after,
        )

        sql, *_ = pg.fetch.call_args.args
        assert "date_updated >= $8" in sql
        assert "date_created >=" not in sql

    async def test_invalid_date_field_raises(self) -> None:
        """An unknown ``date_field`` raises before touching the DB, so a
        column expression can never reach the SQL."""
        pg = _make_pg_mock()
        pg.fetch = AsyncMock(return_value=[])
        collection = ConversationsCollection.__new__(ConversationsCollection)
        collection.l3_pool = pg

        with pytest.raises(ValueError, match="date_field must be one of"):
            await collection.search(
                agent_id=uuid7(),
                user_id=uuid7(),
                query="x",
                date_field="date_created; DROP TABLE conversations",
                date_after=datetime(2026, 5, 1, tzinfo=UTC),
            )
        pg.fetch.assert_not_called()

    async def test_only_lower_bound_emits_single_predicate(self) -> None:
        """Passing only ``date_after`` emits just the lower-bound
        predicate; ``include_closed=True`` frees $7 for the date bind."""
        pg = _make_pg_mock()
        pg.fetch = AsyncMock(return_value=[])
        collection = ConversationsCollection.__new__(ConversationsCollection)
        collection.l3_pool = pg

        after = datetime(2026, 5, 1, tzinfo=UTC)
        await collection.search(
            agent_id=uuid7(),
            user_id=uuid7(),
            query="x",
            include_closed=True,
            date_after=after,
        )

        sql, *params = pg.fetch.call_args.args
        assert "status !=" not in sql
        assert "date_created >= $7" in sql
        assert "date_created <=" not in sql
        assert params[6] == after

    async def test_no_date_bounds_omit_predicates(self) -> None:
        """No date filters -> no ``date_created`` range predicates."""
        pg = _make_pg_mock()
        pg.fetch = AsyncMock(return_value=[])
        collection = ConversationsCollection.__new__(ConversationsCollection)
        collection.l3_pool = pg

        await collection.search(agent_id=uuid7(), user_id=uuid7(), query="x")

        sql, *_ = pg.fetch.call_args.args
        assert "date_created >=" not in sql
        assert "date_created <=" not in sql
