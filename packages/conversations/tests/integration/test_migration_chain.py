"""integration test: the conversations migration chain against a real Postgres.

pins v010 on a live schema: a schema migrated to v009 carries the v005 GIN
index over ``search_vector``, and v010 removes it while the column, its
trigger, and the ``ConversationsCollection.search`` predicate keep working.
"""

from __future__ import annotations

from uuid import uuid4

import asyncpg
import pytest

from threetears.conversations.migrations import drop_search_vector_gin_index
from threetears.conversations.migrations import register as register_conversations
from threetears.core.data.gin import gin_filter
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore

pytestmark = pytest.mark.integration

_DROPPED_INDEX = "idx_conversations_search_vector"


async def _index_exists(conn: asyncpg.Connection, schema: str, index_name: str) -> bool:
    """return whether ``schema.index_name`` exists in pg_indexes.

    :param conn: live asyncpg connection
    :ptype conn: asyncpg.Connection
    :param schema: schema to check
    :ptype schema: str
    :param index_name: index to check
    :ptype index_name: str
    :return: True if the index exists
    :rtype: bool
    """
    row = await conn.fetchrow(
        "SELECT 1 FROM pg_indexes WHERE schemaname = $1 AND indexname = $2",
        schema,
        index_name,
    )
    return row is not None


class TestSearchVectorGinIndexDropped:
    """v010 drops ``idx_conversations_search_vector`` and leaves full-text search working."""

    async def test_v010_drops_the_index_and_search_still_filters(self, pg_schema: tuple[str, str]) -> None:
        """the index exists at v009, is gone after v010, and the search predicate still matches.

        the v009 checkpoint is the non-vacuity guard: it proves the name
        asserted absent afterwards really existed on an upgraded schema.

        :param pg_schema: (url, schema) tuple
        :ptype pg_schema: tuple[str, str]
        """
        url, schema = pg_schema
        runner = MigrationRunner()
        register_conversations(runner)
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = AsyncpgStore(conn)
            await runner.apply_for_agent_schema(store, target=9)  # type: ignore[arg-type]
            assert await _index_exists(conn, schema, _DROPPED_INDEX)

            assert await runner.apply_for_agent_schema(store) == 1  # type: ignore[arg-type]
            assert not await _index_exists(conn, schema, _DROPPED_INDEX)
            # the btree scope indexes the filter narrows through are untouched.
            assert await _index_exists(conn, schema, "idx_conv_user")

            agent_id = uuid4()
            user_id = uuid4()
            await conn.execute(
                "INSERT INTO conversations (agent_id, conversation_id, customer_id, user_id, "
                "channel_type, status, name, date_created, date_updated, message_count) "
                "VALUES ($1, $2, $3, $4, 'web', 'active', 'quarterly budget review', now(), now(), 0)",
                agent_id,
                uuid4(),
                uuid4(),
                user_id,
            )
            # the trigger still maintains the column, and the gin_filter-wrapped
            # predicate the collection issues still matches, OR included.
            predicate = gin_filter("search_vector @@ websearch_to_tsquery('english', $3)")
            count = await conn.fetchval(
                f"SELECT count(*) FROM conversations WHERE agent_id = $1 AND user_id = $2 AND {predicate}",
                agent_id,
                user_id,
                "budget or forecast",
            )
            assert count == 1

            # replay is a no-op, and the body tolerates an index already gone.
            assert await runner.apply_for_agent_schema(store) == 0  # type: ignore[arg-type]
            await drop_search_vector_gin_index(store)  # type: ignore[arg-type]
        finally:
            await conn.close()
