"""the intention reads moved onto the keyed helpers give the rows the statements they replaced gave.

``find_by_user`` (a person's every want, salience-ranked), ``find_open_for_deliberation`` (their open wants outside
the cooldown, ranked alike) and ``find_similar_for_dedup`` (their open wants nearest a vector) each once wrote its
statement; they now send ``read_by`` with the rank as a sort expression (by ``ix_intentions_agent_user`` /
``idx_intentions_open_ranked``) and ``read_nearest``. Each is compared, through
:func:`~threetears.core.testing.same_results.assert_same_results`, with the statement it replaced: empty, one row,
many, NULLs, ties in the order, a tenant (agent) boundary, a page boundary and one past it; with decay on read off and
on. Another person's wants, other statuses, unembedded wants and recently surfaced ones are seeded in.

Runs on PostgreSQL and YugabyteDB (the session's ``db_container``), a schema of its own.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest
import pytest_asyncio

from threetears.agent.intention import collections as intention_collections
from threetears.agent.intention.collections import IntentionsCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.salience import SalienceDecay
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeNatsClient
from threetears.core.testing.same_results import Seeded, assert_same_results

from .conftest import AsyncpgStore, build_runner

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

NS = uuid.UUID("8b2e6d1f-4c3a-4f5b-8e71-0a9c2d4e6f80")
AGENTS = (uuid.uuid5(NS, "agent-0"), uuid.uuid5(NS, "agent-1"))
USER = uuid.uuid5(NS, "user")
OTHER = uuid.uuid5(NS, "other-user")
T0 = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
DIM = 1024
PAGE = 6
_DECAY = SalienceDecay(half_life=timedelta(days=60), floor=0.1)
_STATUSES = ("open", "open", "asked", "open", "dropped")
_COLUMNS = (
    "intention_id, agent_id, customer_id, user_id, status, content, embedding::text AS embedding, salience, "
    "last_decayed_at, last_surfaced_at, source_memory_id, source_conversation_id, date_created, date_updated"
)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def conn(pg_url: str) -> AsyncIterator[asyncpg.Connection]:
    schema = f"int_rank_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(pg_url)
    await connection.execute('CREATE EXTENSION IF NOT EXISTS "vector"')
    await connection.execute(f'CREATE SCHEMA "{schema}"')
    await connection.execute(f'SET search_path TO "{schema}", public')
    await build_runner().apply_for_agent_schema(AsyncpgStore(connection))
    vector_schema = await connection.fetchval(
        "SELECT n.nspname FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace WHERE t.typname = 'vector'"
    )
    await connection.set_type_codec("vector", schema=vector_schema, encoder=str, decoder=str, format="text")
    try:
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


def _intention(seed: Seeded) -> Mapping[str, Any]:
    """a want of its tenant's agent: every fourth another person's; its status in turn; its salience and stamps
    from its tie (so ties in the rank tie whole); unembedded and never surfaced for a NULL seed."""
    index = seed.index
    return {
        "intention_id": uuid.uuid5(NS, f"intention-{index}"),
        "agent_id": AGENTS[seed.tenant],
        "user_id": OTHER if index % 4 == 3 else USER,
        "status": _STATUSES[index % len(_STATUSES)],
        "content": f"want {index}",
        "embedding": None if seed.null else "[" + ",".join(["1"] * (DIM - 1) + [str(seed.tie)]) + "]",
        "salience": round(0.1 + 0.07 * (seed.tie % 12), 4),
        "last_decayed_at": None if seed.null else T0 - timedelta(days=seed.tie),
        "last_surfaced_at": None if seed.null else T0 - timedelta(days=seed.tie % 9),
        "date_created": T0 - timedelta(days=2 * seed.tie),
    }


def _collection(c: Any) -> IntentionsCollection:
    registry = CollectionRegistry()
    registry.configure(l3_pool=c)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return IntentionsCollection(registry, config, nats_client=FakeNatsClient())


def _spied(monkeypatch: pytest.MonkeyPatch, name: str) -> list[Any]:
    kept: list[Any] = []
    real = getattr(intention_collections, name)

    async def spy(*args: Any, **kwargs: Any) -> Any:
        rows = await real(*args, **kwargs)
        kept.append(rows)
        return rows

    monkeypatch.setattr(intention_collections, name, spy)
    return kept


def _rank(decay: SalienceDecay | None) -> str:
    return "salience" if decay is None else decay.rank_sql()


_DECAYS = {"decay off": None, "decay on": _DECAY}


@pytest.mark.parametrize("decay", sorted(_DECAYS))
async def test_find_by_user_gives_the_hand_written_rows(
    conn: asyncpg.Connection, monkeypatch: pytest.MonkeyPatch, decay: str
) -> None:
    """Flips if find_by_user returns a want the statement did not (another person's or agent's), misses one (any
    status), or ranks them otherwise."""
    monkeypatch.setattr(IntentionsCollection, "salience_decay", _DECAYS[decay])
    kept = _spied(monkeypatch, "read_by")

    async def helper(c: Any) -> Any:
        kept.clear()
        await _collection(c).find_by_user(USER, agent_id=AGENTS[0])
        (rows,) = kept
        return rows

    await assert_same_results(
        conn,
        table="intentions",
        row=_intention,
        old_sql=f"SELECT {_COLUMNS}, {_rank(_DECAYS[decay])} AS r FROM intentions WHERE agent_id = $1 AND user_id = $2"
        f" ORDER BY {_rank(_DECAYS[decay])} DESC, date_created DESC",
        params=[AGENTS[0], USER],
        helper=_ranked(helper, _DECAYS[decay]),
        order_by=["r", "date_created"],
        page_size=PAGE,
    )


@pytest.mark.parametrize("decay", sorted(_DECAYS))
async def test_find_open_for_deliberation_gives_the_hand_written_rows(
    conn: asyncpg.Connection, monkeypatch: pytest.MonkeyPatch, decay: str
) -> None:
    """Flips if find_open_for_deliberation returns a want the statement did not (not open, surfaced inside the
    cooldown, another person's or agent's), misses one (never surfaced), or ranks them otherwise."""
    monkeypatch.setattr(IntentionsCollection, "salience_decay", _DECAYS[decay])
    kept = _spied(monkeypatch, "read_by")
    cutoff = T0 - timedelta(days=4)

    async def helper(c: Any) -> Any:
        kept.clear()
        await _collection(c).find_open_for_deliberation(USER, agent_id=AGENTS[0], cooldown_cutoff=cutoff)
        (rows,) = kept
        return rows

    await assert_same_results(
        conn,
        table="intentions",
        row=_intention,
        old_sql=f"SELECT {_COLUMNS}, {_rank(_DECAYS[decay])} AS r FROM intentions WHERE agent_id = $1 AND user_id = $2"
        " AND status = 'open' AND (last_surfaced_at IS NULL OR last_surfaced_at < $3)"
        f" ORDER BY {_rank(_DECAYS[decay])} DESC, date_created DESC",
        params=[AGENTS[0], USER, cutoff],
        helper=_ranked(helper, _DECAYS[decay]),
        order_by=["r", "date_created"],
        page_size=PAGE,
    )


def _ranked(helper: Any, decay: SalienceDecay | None) -> Any:
    """the helper's rows, each beside its rank as the database computes it (the order the statement compared by)."""

    async def run(c: Any) -> Any:
        rows = [dict(row) for row in await helper(c)]
        ranks = {
            r["intention_id"]: r["r"]
            for r in await c.fetch(f"SELECT intention_id, {_rank(decay)} AS r FROM intentions")
        }
        return [{**row, "r": ranks[row["intention_id"]]} for row in rows]

    return run


async def test_find_similar_for_dedup_gives_the_hand_written_rows(
    conn: asyncpg.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if find_similar_for_dedup's ranked read returns a want the statement did not (not open, unembedded,
    another person's or agent's), misses one, or ranks them otherwise; of the rows tied at the limit, only how many."""
    kept = _spied(monkeypatch, "read_nearest")
    query = [1.0] * (DIM - 1) + [0.0]

    async def helper(c: Any) -> Any:
        kept.clear()
        await _collection(c).find_similar_for_dedup(
            user_id=USER, agent_id=AGENTS[0], embedding=query, top_k=PAGE, threshold=-1.0
        )
        (rows,) = kept
        return rows

    await assert_same_results(
        conn,
        table="intentions",
        row=_intention,
        old_sql="SELECT intention_id, content, salience, "
        "1 - (embedding OPERATOR(public.<=>) $1::text::public.vector) AS similarity FROM intentions "
        "WHERE agent_id = $2 AND user_id = $3 AND embedding IS NOT NULL AND status = 'open' "
        "ORDER BY embedding OPERATOR(public.<=>) $1::text::public.vector LIMIT $4",
        params=[json.dumps(query), AGENTS[0], USER, PAGE],
        helper=helper,
        order_by=["similarity"],
        page_size=PAGE,
        limited=True,
    )
