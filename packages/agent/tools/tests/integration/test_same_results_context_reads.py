"""the context-items reads moved onto the keyed helpers give the rows the statements they replaced gave.

``ContextItemCollection.find_by_conversation`` (a conversation's rows, by the primary key's lead), ``count_results``
(its tool results, by ``ix_context_items_type``) and ``evict_lru``'s pick (the least recently read tool results, a
page) each once wrote its statement, an optional agent written ``($n::uuid IS NULL OR agent_id = $n)``. Each is
compared, through :func:`~threetears.core.testing.same_results.assert_same_results`, with the statement it replaced:
empty, one row, many, NULL agents, ties in the order, a tenant (conversation) boundary, a page boundary and one past
it; with no agent named and with one. The method runs as the collection runs it, the helper's rows taken as it
answered them (``evict_lru`` stopped there, before its deletes).

Runs on PostgreSQL and YugabyteDB (the session's ``db_container``), a schema of its own.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest
import pytest_asyncio

from threetears.agent.tools import collections as tools_collections
from threetears.agent.tools.collections import ContextItemCollection
from threetears.agent.tools.migrations import register as register_tools
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.core.testing.same_results import Seeded, assert_same_results

from packages.agent.tools.tests.testing_utils import make_nats_mock

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

NS = uuid.UUID("5d1c2f8e-0b7a-4c55-9a51-2d7e3c9b4f10")
CONVERSATIONS = (uuid.uuid5(NS, "conversation-0"), uuid.uuid5(NS, "conversation-1"))
AGENTS = (uuid.uuid5(NS, "agent-0"), uuid.uuid5(NS, "agent-1"))
T0 = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
PAGE = 6


class _Store:
    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        return str(await self._conn.execute(sql, *params))

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        return [dict(r) for r in await self._conn.fetch(sql, *params)]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """a fresh schema holding context_items, its conversation foreign key dropped (the rows' conversations are
    the seeds' tenants, not conversation rows)."""
    schema = f"ctx_same_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    await connection.execute(f'CREATE SCHEMA "{schema}"')
    await connection.execute(f'SET search_path TO "{schema}", public')
    runner = MigrationRunner()
    register_conversations(runner)
    register_tools(runner)
    await runner.apply_for_agent_schema(_Store(connection))
    for (name,) in await connection.fetch(
        "SELECT conname FROM pg_constraint WHERE conrelid = 'context_items'::regclass AND contype = 'f'"
    ):
        await connection.execute(f'ALTER TABLE context_items DROP CONSTRAINT "{name}"')
    try:
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


def _row(seed: Seeded) -> Mapping[str, Any]:
    """a row of its tenant's conversation: two in three a tool result, the rest variables; its agent one of two, or
    none for a NULL seed; read and written at its tie."""
    return {
        "conversation_id": CONVERSATIONS[seed.tenant],
        "context_id": uuid.uuid5(NS, f"context-{seed.index}"),
        "context_type": "variable" if seed.index % 3 == 2 else "tool_result",
        "key": f"k{seed.index}",
        "short_desc": "s",
        "long_desc": "",
        "content": f"c{seed.index}",
        "date_accessed": T0 + timedelta(minutes=seed.tie),
        "date_created": T0 + timedelta(minutes=seed.tie),
        "date_updated": T0,
        "agent_id": None if seed.null else AGENTS[seed.index % 2],
    }


class _Stop(Exception):
    """raised by the spy once it holds the helper's rows, so the method goes no further."""


def _collection(c: Any) -> ContextItemCollection:
    registry = CollectionRegistry()
    registry.configure(l3_pool=c)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return ContextItemCollection(registry, config, nats_client=make_nats_mock())


def _spied(monkeypatch: pytest.MonkeyPatch, name: str, *, stop: bool = False) -> list[Any]:
    """the module's ``name`` helper wrapped: each call's rows kept (a page's ``.rows``), then ``_Stop`` when asked."""
    kept: list[Any] = []
    real = getattr(tools_collections, name)

    async def spy(*args: Any, **kwargs: Any) -> Any:
        answered = await real(*args, **kwargs)
        kept.append(getattr(answered, "rows", answered))
        if stop:
            raise _Stop
        return answered

    monkeypatch.setattr(tools_collections, name, spy)
    return kept


_AGENT_CHOICES = {"no agent named": None, "one agent": AGENTS[0]}


@pytest.mark.parametrize("agent", sorted(_AGENT_CHOICES))
async def test_find_by_conversation_gives_the_hand_written_rows(
    conn: asyncpg.Connection, monkeypatch: pytest.MonkeyPatch, agent: str
) -> None:
    """Flips if find_by_conversation's keyed read returns a row the statement did not (another conversation's,
    another agent's when one is named), misses one (a NULL agent's when none is named), or orders them otherwise."""
    agent_id = _AGENT_CHOICES[agent]
    kept = _spied(monkeypatch, "read_by")

    async def helper(c: Any) -> Any:
        kept.clear()
        await _collection(c).find_by_conversation(CONVERSATIONS[0], agent_id)
        (rows,) = kept
        return rows

    await assert_same_results(
        conn,
        table="context_items",
        row=_row,
        old_sql="SELECT * FROM context_items WHERE conversation_id = $1 AND ($2::uuid IS NULL OR agent_id = $2)"
        " ORDER BY date_created ASC",
        params=[CONVERSATIONS[0], agent_id],
        helper=helper,
        order_by=["date_created"],
        page_size=PAGE,
    )


@pytest.mark.parametrize("agent", sorted(_AGENT_CHOICES))
async def test_count_results_gives_the_hand_written_count(conn: asyncpg.Connection, agent: str) -> None:
    """Flips if count_results counts a row the statement did not (a variable, another conversation's, another
    agent's) or misses one."""
    agent_id = _AGENT_CHOICES[agent]

    async def helper(c: Any) -> Any:
        return [{"cnt": await _collection(c).count_results(CONVERSATIONS[0], agent_id)}]

    await assert_same_results(
        conn,
        table="context_items",
        row=_row,
        old_sql="SELECT COUNT(*) AS cnt FROM context_items WHERE conversation_id = $1 AND context_type = 'tool_result'"
        " AND ($2::uuid IS NULL OR agent_id = $2)",
        params=[CONVERSATIONS[0], agent_id],
        helper=helper,
        page_size=PAGE,
    )


@pytest.mark.parametrize("agent", sorted(_AGENT_CHOICES))
async def test_evict_lru_picks_the_hand_written_rows(
    conn: asyncpg.Connection, monkeypatch: pytest.MonkeyPatch, agent: str
) -> None:
    """Flips if evict_lru's page of least recently read tool results holds a row the statement's ``LIMIT`` did not
    (a variable, another conversation's or agent's, a more recently read one) or misses one; of the rows tied at
    the limit, only how many (the statement left which to the engine)."""
    agent_id = _AGENT_CHOICES[agent]
    kept = _spied(monkeypatch, "keyset_page", stop=True)

    async def helper(c: Any) -> Any:
        kept.clear()
        with pytest.raises(_Stop):
            # a limit of 1 evicts all but one: the page asked for is every counted row but one
            await _collection(c).evict_lru(CONVERSATIONS[0], 1, agent_id)
        (rows,) = kept
        return rows

    async def old_limit(c: Any) -> int:
        counted = await c.fetchval(
            "SELECT COUNT(*) FROM context_items WHERE conversation_id = $1 AND context_type = 'tool_result'"
            " AND ($2::uuid IS NULL OR agent_id = $2)",
            CONVERSATIONS[0],
            agent_id,
        )
        return int(counted) - 1

    # the statement's LIMIT is what evict_lru hands it: the count past the limit of 1; the seeded cases are fixed,
    # so it is bound in SQL from the same count (the date_accessed it orders by selected too, to compare ties)
    await assert_same_results(
        conn,
        table="context_items",
        row=_row,
        old_sql="SELECT context_id, date_accessed FROM context_items WHERE conversation_id = $1"
        " AND context_type = 'tool_result' AND ($3::uuid IS NULL OR agent_id = $3) ORDER BY date_accessed ASC"
        " LIMIT GREATEST((SELECT COUNT(*) FROM context_items WHERE conversation_id = $1"
        " AND context_type = 'tool_result' AND ($3::uuid IS NULL OR agent_id = $3)) - $2, 0)",
        params=[CONVERSATIONS[0], 1, agent_id],
        helper=_skipping_short(helper, old_limit),
        columns=["context_id", "date_accessed"],
        order_by=["date_accessed"],
        page_size=PAGE,
        limited=True,
    )


def _skipping_short(helper: Any, old_limit: Any) -> Any:
    """``helper``, or no rows where evict_lru would not have read at all (nothing past the limit)."""

    async def run(c: Any) -> Any:
        if await old_limit(c) <= 0:
            return []
        return await helper(c)

    return run
