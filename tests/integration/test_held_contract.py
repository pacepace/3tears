"""integration: each package's held CONTRACT step drops what it names, on PostgreSQL and YugabyteDB, and replays clean.

The session schema (``conftest.py``) is migrated with every package's held contract step
(``register(..., held_contract=True)``), the shape the release after this one gives every deployment. These hold the
steps to what they say: the indexes no read uses are gone, wake's fire key is validated, and running each step again
changes nothing. The plan tests in this directory run on the same schema, so each read they prove is proven with the
dropped indexes gone.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from threetears.agent.intention.migrations.v002_contract_last_surfaced_index_dropped import drop_last_surfaced_index
from threetears.agent.skills.migrations.v007_contract_replaced_index_dropped import drop_replaced_conversation_index
from threetears.agent.tools.migrations.v006_contract_lru_index_dropped import drop_lru_index
from threetears.agent.wake.migrations.v015_contract_unread_indexes_dropped import (
    drop_unread_indexes_and_validate_fire_key,
)
from threetears.conversations.migrations.v014_contract_unread_indexes_dropped import drop_unread_conversation_indexes
from threetears.mcp.migrations.v002_contract_lookup_indexes_dropped import drop_lookup_indexes
from threetears.scrape.migrations import v015_contract_unread_indexes_dropped

__all__: list[str] = []

pytestmark = pytest.mark.integration

#: every index a held contract step drops, by its table
_DROPPED: dict[str, tuple[str, ...]] = {
    "agent_skill_invocations": ("idx_skill_invocations_conv",),
    "agent_wake_schedules": ("idx_wake_schedules_conv_status",),
    "webhook_subscriptions": ("idx_webhook_subs_conv",),
    "wake_fires": ("idx_wake_fires_started_conversation",),
    "conversations": ("idx_conv_customer", "idx_conv_parent"),
    "context_items": ("ix_context_items_lru",),
    "intentions": ("idx_intentions_last_surfaced",),
    "mcp_tool_grants": ("idx_mcp_tool_grants_principal", "idx_mcp_tool_grants_tool"),
    "scrape_extractions": ("scrape_extractions_target_id", "scrape_extractions_retrieved_at"),
    "scrape_target_health": ("scrape_target_health_circuit_state",),
}

#: each held contract step
_STEPS: tuple[Callable[[Any], Awaitable[None]], ...] = (
    drop_replaced_conversation_index,
    drop_unread_indexes_and_validate_fire_key,
    drop_unread_conversation_indexes,
    drop_lru_index,
    drop_last_surfaced_index,
    drop_lookup_indexes,
    v015_contract_unread_indexes_dropped,
)


class _Store:
    """a migration step's store over one connection."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def execute(self, sql: str, *args: Any) -> str:
        return str(await self._conn.execute(sql, *args))

    async def query(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        return [dict(row) for row in await self._conn.fetch(sql, *args)]


@pytest.fixture(params=["conn", "yugabyte_conn"])
def engine_conn(request: pytest.FixtureRequest) -> Any:
    """each engine's connection (``conftest.py``): the run's engine is proven, the other deselected."""
    return request.getfixturevalue(request.param)


async def _indexes(conn: Any) -> dict[str, set[str]]:
    rows = await conn.fetch(
        "SELECT tablename, indexname FROM pg_indexes WHERE schemaname = current_schema() AND tablename = ANY($1)",
        list(_DROPPED),
    )
    found: dict[str, set[str]] = {table: set() for table in _DROPPED}
    for row in rows:
        found[row["tablename"]].add(row["indexname"])
    return found


async def _fire_key_validated(conn: Any) -> bool:
    return bool(
        await conn.fetchval(
            "SELECT c.convalidated FROM pg_constraint c JOIN pg_class t ON t.oid = c.conrelid"
            " WHERE c.conname = 'wake_fires_schedule_conversation_fk' AND t.relnamespace = current_schema()::regnamespace"
        )
    )


async def test_the_held_contract_steps_drop_what_they_name_and_replay_clean(engine_conn: Any) -> None:
    """Flips if a held contract step leaves an index it drops, leaves wake's fire key unvalidated, or fails or
    changes anything when run again."""
    before = await _indexes(engine_conn)
    for table, dropped in _DROPPED.items():
        assert not set(dropped) & before[table], (table, sorted(before[table]))
        # the table is there, with the indexes the step keeps: the step dropped what it names, not the table
        assert before[table], table
    assert await _fire_key_validated(engine_conn)
    for step in _STEPS:
        await step(_Store(engine_conn))
    assert await _indexes(engine_conn) == before
    assert await _fire_key_validated(engine_conn)
