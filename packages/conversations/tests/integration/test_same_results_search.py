"""integration: the conversations a full-text search finds are the ones the statement it replaced found, in its order.

``ConversationsCollection.search`` sent ``SELECT *, ts_rank_cd(...) AS rank FROM conversations WHERE agent_id = $1 AND
user_id = $2 AND <the match> [AND status != 'closed'] [AND <date bounds>] ORDER BY rank DESC, date_updated DESC LIMIT
$4 OFFSET $5`` by hand; it now reads through ``read_nearest`` with a ``TextRank`` (its ties on the rank by
``date_updated`` descending, then the key), the first ``limit + offset`` rows with the offset sliced off. Compared,
through :func:`~threetears.core.testing.same_results.assert_same_results`, with that statement: names matching once,
twice and not at all (distinct ranks), a NULL name, closed conversations, another user's and another agent's, dates
on both sides of the bounds, a page cut inside the rows and an offset past some; OR / NOT queries; both date fields;
another language. Rows tied on (rank, date_updated) share a name, so (name, date_updated) is their sort value; a
tie block is a whole run of three seeds (every filter here keeps or drops a run whole), so an offset of three never
cuts one. Runs on PostgreSQL and YugabyteDB (the session's ``db_container``), a schema of its own.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest
import pytest_asyncio

from threetears.conversations.collection import ConversationsCollection
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.core.testing.same_results import Seeded, assert_same_results

from .conftest import AsyncpgStore

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

NS = uuid.UUID("5c1f7e20-8a43-4b9d-9e62-3f0a7d4c2b18")
AGENTS = (uuid.uuid5(NS, "agent-0"), uuid.uuid5(NS, "agent-1"))
USER = uuid.uuid5(NS, "user")
OTHER_USER = uuid.uuid5(NS, "other-user")
T0 = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)

#: a seed's name by its tie: no match, one match, two (a higher rank), and the same again for "bread"
_NAMES = ("banana bread", "apple pie", "apple apple tart", "fresh bread apple")


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def conn(pg_url: str) -> AsyncIterator[asyncpg.Connection]:
    """a fresh schema holding the conversations table."""
    schema = f"conv_search_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(pg_url)
    await connection.execute(f'CREATE SCHEMA "{schema}"')
    await connection.execute(f'SET search_path TO "{schema}", public')
    runner = MigrationRunner()
    register_conversations(runner)
    await runner.apply_for_agent_schema(AsyncpgStore(connection))
    try:
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


def _row(seed: Seeded) -> Mapping[str, Any]:
    """a conversation of its tenant's agent. Everything but its key is a function of its tie, so the rows tied on
    (rank, date_updated) are one run of seeds, kept or dropped whole: its name from :data:`_NAMES` (NULL in the NULLs
    case), every fifth tie another user's, every fourth closed, updated at the tie's minute and created before it."""
    return {
        "agent_id": AGENTS[seed.tenant],
        "conversation_id": uuid.uuid5(NS, f"conversation-{seed.index}"),
        "customer_id": uuid.uuid5(NS, "customer"),
        "user_id": OTHER_USER if seed.tie % 5 == 2 else USER,
        "channel_type": "web",
        "status": "closed" if seed.tie % 4 == 3 else "active",
        "name": None if seed.null else _NAMES[seed.tie % len(_NAMES)],
        "date_created": T0 - timedelta(minutes=seed.tie),
        "date_updated": T0 + timedelta(minutes=seed.tie),
        "message_count": 0,
    }


def _old_statement(
    query: str,
    *,
    limit: int,
    offset: int,
    include_closed: bool,
    language: str,
    date_field: str,
    date_after: datetime | None,
    date_before: datetime | None,
) -> tuple[str, list[Any]]:
    """the statement ``search`` wrote by hand before, and its binds, built as it built them."""
    params: list[Any] = [AGENTS[0], USER, query, limit, offset, language]
    predicates = [
        "agent_id = $1",
        "user_id = $2",
        "(search_vector @@ websearch_to_tsquery($6::regconfig, $3)) IS TRUE",
    ]
    if not include_closed:
        predicates.append(f"status != ${len(params) + 1}")
        params.append("closed")
    column = {"created": "date_created", "updated": "date_updated"}[date_field]
    if date_after is not None:
        predicates.append(f"{column} >= ${len(params) + 1}")
        params.append(date_after)
    if date_before is not None:
        predicates.append(f"{column} <= ${len(params) + 1}")
        params.append(date_before)
    sql = (
        "SELECT *, ts_rank_cd(search_vector, websearch_to_tsquery($6::regconfig, $3)) AS rank "
        f"FROM conversations WHERE {' AND '.join(predicates)} ORDER BY rank DESC, date_updated DESC LIMIT $4 OFFSET $5"
    )
    return sql, params


_CASES = [
    pytest.param({"query": "apple"}, id="one-word"),
    pytest.param({"query": "apple OR bread", "limit": 4}, id="or-page-cut"),
    pytest.param({"query": "apple -tart", "limit": 3, "offset": 3}, id="not-offset"),
    pytest.param({"query": "apple", "include_closed": True, "offset": 3}, id="closed-included-offset"),
    pytest.param(
        {"query": "bread OR apple", "date_field": "updated", "date_after": T0 + timedelta(minutes=3)},
        id="updated-after",
    ),
    pytest.param(
        {
            "query": "apple",
            "date_after": T0 - timedelta(minutes=9),
            "date_before": T0 - timedelta(minutes=2),
            "include_closed": True,
        },
        id="created-window",
    ),
    pytest.param({"query": "apples OR pie", "language": "simple"}, id="simple-language"),
]


@pytest.mark.parametrize("case", _CASES)
async def test_the_search_s_conversations_are_the_statement_s(conn: asyncpg.Connection, case: dict[str, Any]) -> None:
    """Flips if a search finds other conversations than the hand-written statement did, or ranks them in another
    order: another user's or agent's, a closed one, one outside the dates, a row the offset skipped, a tie order
    reversed."""
    options: dict[str, Any] = {
        "limit": 20,
        "offset": 0,
        "include_closed": False,
        "language": "english",
        "date_field": "created",
        "date_after": None,
        "date_before": None,
        **case,
    }
    query = options.pop("query")
    old_sql, params = _old_statement(query, **options)

    async def helper(c: asyncpg.Connection) -> list[dict[str, Any]]:
        registry = CollectionRegistry()
        registry.configure(l3_pool=c)
        collection = ConversationsCollection(registry, DefaultCoreConfig(), c)
        found = await collection.search(
            AGENTS[0],
            USER,
            query,
            limit=options["limit"],
            offset=options["offset"],
            include_closed=options["include_closed"],
            query_language=options["language"],
            date_field=options["date_field"],
            date_after=options["date_after"],
            date_before=options["date_before"],
        )
        return [entity.to_dict() for entity in found]

    answered = await assert_same_results(
        conn,
        table="conversations",
        row=_row,
        old_sql=old_sql,
        params=params,
        helper=helper,
        columns=["conversation_id", "agent_id", "user_id", "name", "status", "date_updated", "date_created"],
        order_by=["name", "date_updated"],
        limited=True,
    )
    # the comparison means something only where the statement found rows
    assert any(answered), "the seeds match no row of this search"
