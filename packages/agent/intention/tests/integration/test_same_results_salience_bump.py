"""``IntentionsCollection.bump_salience``, moved onto ``update_keyed``, writes what its statement wrote.

The reinforcement once wrote ``UPDATE intentions SET <reinforced> WHERE agent_id = $2 AND intention_id = ANY($3)``;
it now hands ``update_keyed`` its values (the stored salience plus the bump with decay on read off;
``SalienceDecay.reinforced_values`` with it on). Compared, through
:func:`~threetears.core.testing.same_results.assert_same_results` with ``state_sql``, with that statement: the table
after, salience exactly (``NUMERIC(5,4)``), the decay stamp as whether it was set, every same_results case.

Runs on PostgreSQL and YugabyteDB (the session's ``db_container``), a schema of its own.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import asyncpg
import pytest
import pytest_asyncio

from threetears.agent.intention.collections import IntentionsCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.salience import SalienceDecay
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeNatsClient
from threetears.core.testing.same_results import Seeded, assert_same_results

from .conftest import AsyncpgStore, build_runner

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

NS = uuid.UUID("5d0f4a0e-2a8e-4c7a-9a35-6f1e0b7c9d21")
AGENTS = (uuid.uuid5(NS, "agent-0"), uuid.uuid5(NS, "agent-1"))
T0 = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
_DECAY = SalienceDecay(half_life=timedelta(days=60), floor=0.1)

#: the table after: salience exactly, the decay stamp as whether this write set it (the seeds' are weeks old)
_STATE = "SELECT intention_id, salience, (last_decayed_at > now() - interval '1 day') AS decay_stamped FROM intentions"


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def conn(pg_url: str) -> AsyncIterator[asyncpg.Connection]:
    """a fresh schema holding the intention tables."""
    schema = f"int_same_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(pg_url)
    await connection.execute('CREATE EXTENSION IF NOT EXISTS "vector"')
    await connection.execute(f'CREATE SCHEMA "{schema}"')
    await connection.execute(f'SET search_path TO "{schema}", public')
    await build_runner().apply_for_agent_schema(AsyncpgStore(connection))
    try:
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


class _Serial:
    """one connection as the pool the collection writes through, each statement in turn."""

    def __init__(self, conn: Any) -> None:
        self.conn = conn
        self.lock = asyncio.Lock()

    async def fetch(self, sql: str, *args: Any) -> Any:
        async with self.lock:
            return await self.conn.fetch(sql, *args)

    async def execute(self, sql: str, *args: Any) -> Any:
        async with self.lock:
            return await self.conn.execute(sql, *args)


def _intention(seed: Seeded) -> Mapping[str, Any]:
    index = seed.index
    return {
        "intention_id": uuid.uuid5(NS, f"intention-{index}"),
        "agent_id": AGENTS[seed.tenant],
        "content": f"want {index}",
        "salience": round(0.1 + 0.07 * (index % 12), 4),
        "last_decayed_at": None if seed.null else T0 - timedelta(days=index),
        "date_created": T0 - timedelta(days=2 * index),
    }


@pytest.mark.parametrize("decay", [None, _DECAY], ids=["decay off", "decay on"])
async def test_bump_salience_writes_what_its_statement_wrote(
    conn: asyncpg.Connection, monkeypatch: pytest.MonkeyPatch, decay: SalienceDecay | None
) -> None:
    """Flips if bump_salience, with decay on read off or on, sets another salience (the stored one plus the bump,
    or the decayed one plus the bump, clamped to 1), bumps another agent's want or one not named, or leaves the decay
    stamp unset with decay on."""
    monkeypatch.setattr(IntentionsCollection, "salience_decay", decay)
    ids = [uuid.uuid5(NS, f"intention-{index}") for index in (0, 1, 2, 5, 9, 30)]
    folded = "salience" if decay is None else decay.effective_sql()
    stamp = "" if decay is None else ", last_decayed_at = now()"

    async def helper(c: Any) -> list[dict[str, Any]]:
        registry = CollectionRegistry()
        registry.configure(l3_pool=cast(Any, _Serial(c)), kv_key_scope="salience-principal")
        config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
        intentions = IntentionsCollection(registry, config, nats_client=FakeNatsClient())
        await intentions.bump_salience(ids, agent_id=AGENTS[0], access_bump=0.35)
        return []

    await assert_same_results(
        conn,
        table="intentions",
        row=_intention,
        old_sql=(
            f"UPDATE intentions SET salience = LEAST(1.0, {folded} + $1){stamp} "
            "WHERE agent_id = $2 AND intention_id = ANY($3::uuid[])"
        ),
        params=[0.35, AGENTS[0], ids],
        helper=helper,
        state_sql=_STATE,
        page_size=6,
    )
