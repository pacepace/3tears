"""``IdentityVersionsCollection.resolve_active``, moved onto ``keyset_page``, gives the row its statement gave.

The prompt-assembly read once sent ``SELECT ... WHERE agent_id = $1 AND customer_id IS NOT DISTINCT FROM $2 AND
user_id IS NOT DISTINCT FROM $3 AND block_key = $4 AND variant = $5 AND status = 'active' ORDER BY date_created DESC``
by ``fetchrow``: its first row. It now reads the first row of a page by those pins (a None grain pinned ``IS NULL``),
newest first, the key breaking a tie the statement left to the engine. Compared, through
:func:`~threetears.core.testing.same_results.assert_same_results` (``limited``: of rows tied at the first, only the
count and sort values), with the statement under ``LIMIT 1``: with both grains named and with a NULL one, other
blocks, variants, statuses and agents seeded in.

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

from threetears.agent.identity.collections import IdentityVersionsCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.same_results import Seeded, assert_same_results

from .conftest import AsyncpgStore, build_runner

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

NS = uuid.UUID("2c4e6a8b-1d3f-4a5c-9e7b-3f5a7c9e1b20")
AGENTS = (uuid.uuid5(NS, "agent-0"), uuid.uuid5(NS, "agent-1"))
USER = uuid.uuid5(NS, "user")
T0 = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
PAGE = 6
_STATUSES = ("active", "active", "superseded", "active", "proposed")
_COLUMNS = (
    "version_id, agent_id, customer_id, user_id, block_key, variant, content, rationale, content_hash, "
    "parent_version_id, status, proposer_agent_id, consenter_user_id, date_created, date_updated"
)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def conn(pg_url: str) -> AsyncIterator[asyncpg.Connection]:
    schema = f"ident_same_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(pg_url)
    await connection.execute(f'CREATE SCHEMA "{schema}"')
    await connection.execute(f'SET search_path TO "{schema}", public')
    await build_runner().apply_for_agent_schema(AsyncpgStore(connection))
    try:
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


def _version(seed: Seeded) -> Mapping[str, Any]:
    """a version of its tenant's agent, no customer (the NULL grain, where many rows may be active), the caller's
    user or none for a NULL seed; its status in turn, every fifth another block and every seventh another variant."""
    index = seed.index
    return {
        "version_id": uuid.uuid5(NS, f"version-{index}"),
        "agent_id": AGENTS[seed.tenant],
        "customer_id": None,
        "user_id": None if seed.null else USER,
        "block_key": "presence" if index % 5 == 4 else "personality",
        "variant": "v2" if index % 7 == 6 else "",
        "content": f"content {index}",
        "content_hash": f"h{index}",
        "status": _STATUSES[index % len(_STATUSES)],
        "date_created": T0 + timedelta(minutes=seed.tie),
    }


@pytest.mark.parametrize("user", ["named", "none"])
async def test_resolve_active_gives_the_hand_written_row(conn: asyncpg.Connection, user: str) -> None:
    """Flips if resolve_active answers a row the statement's first row was not (an inactive one, another block's,
    variant's, agent's or user's; an older one), or none where it had one."""
    user_id = USER if user == "named" else None

    async def helper(c: Any) -> Any:
        registry = CollectionRegistry()
        registry.configure(l3_pool=c)
        config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
        found = await IdentityVersionsCollection(registry=registry, config=config).resolve_active(
            agent_id=AGENTS[0], customer_id=None, user_id=user_id, block_key="personality", variant=""
        )
        return [] if found is None else [{"version_id": found.version_id, "date_created": found.date_created}]

    await assert_same_results(
        conn,
        table="identity_versions",
        row=_version,
        old_sql=f"SELECT {_COLUMNS} FROM identity_versions WHERE agent_id = $1 AND customer_id IS NOT DISTINCT FROM $2"
        " AND user_id IS NOT DISTINCT FROM $3 AND block_key = $4 AND variant = $5 AND status = 'active'"
        " ORDER BY date_created DESC LIMIT 1",
        params=[AGENTS[0], None, user_id, "personality", ""],
        helper=helper,
        columns=["version_id", "date_created"],
        order_by=["date_created"],
        page_size=PAGE,
        limited=True,
    )
