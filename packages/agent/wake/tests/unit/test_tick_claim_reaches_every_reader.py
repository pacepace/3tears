"""a tick's claim of a due schedule reaches every replica that cached the schedule.

The claim advances ``next_fire_at``, stamps ``last_fired_at``, and expires a one-shot, and it
evicts the row from every cache tier and broadcasts the eviction. The tick used to build its own
registry per pass with no NATS client, so the eviction reached no other replica: each kept
serving the pre-claim row, and the schedule tools there read it -- a fired one-shot still shown
active -- and saved it back.

The tick now runs on the collections its host process built once, on the registry that carries
the NATS client and runs the invalidation listener. The test drives the real engine and asserts
on the PUBLISH and on a second replica reading L3 again.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import MetaData
from threetears.agent.skills.tables import agent_skills_table
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CacheInvalidationMessage, CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeNatsClient

from threetears.agent.wake.collections import WakeFireCollection, WakeScheduleCollection
from threetears.agent.wake.tables import (
    agent_wake_schedules_table,
    wake_fires_table,
    webhook_subscriptions_table,
)
from threetears.agent.wake.tick import wake_tick_job
from threetears.agent.wake.types import WakeDispatchResult, WakeTrigger

_SCOPE = "wake-tick-principal"


def _metadata() -> MetaData:
    metadata = MetaData()
    agent_skills_table(metadata)
    agent_wake_schedules_table(metadata)
    wake_fires_table(metadata)
    webhook_subscriptions_table(metadata)
    return metadata


# parity-with: asyncpg.Pool (fetch / fetchrow / fetchval / execute -- the seam one tick drives)
class _Store:
    """an L3 holding one due one-shot; the claim expires it."""

    def __init__(self, row: dict[str, Any]) -> None:
        self.row = row
        self.reads = 0

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        assert "FROM agent_wake_schedules" in sql, sql
        self.reads += 1
        return dict(self.row) if args[1] == self.row["schedule_id"] else None

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        del args
        if "FROM agent_wake_schedules" in sql and self.row["status"] == "active":
            return [dict(self.row)]
        return []

    async def fetchval(self, sql: str, *args: Any) -> Any:
        assert sql.startswith("UPDATE agent_wake_schedules"), sql
        computed_next_fire, now, new_status = args[0], args[1], args[2]
        self.row.update(next_fire_at=computed_next_fire, last_fired_at=now, date_updated=now, status=new_status)
        return self.row["schedule_id"]

    async def execute(self, sql: str, *args: Any) -> str:
        del sql, args
        return "UPDATE 1"


async def _replica(nats: FakeNatsClient, store: _Store) -> CollectionRegistry:
    l1 = SQLiteBackend(db_name=f"wake_tick_{uuid.uuid4().hex[:8]}")
    l1.initialize(_metadata())
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l2_client=nats, l3_pool=store, kv_key_scope=_SCOPE)  # type: ignore[arg-type]
    await registry.start_invalidation_listener(nats)  # type: ignore[arg-type]
    return registry


def _config() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


def _due_one_shot(conv: UUID, sid: UUID, due_at: datetime) -> dict[str, Any]:
    return {
        "conversation_id": conv,
        "schedule_id": sid,
        "user_id": uuid.uuid4(),
        "agent_id": uuid.uuid4(),
        "skill_id": None,
        "schedule_type": "one_shot_at",
        "schedule_config": {"fire_at_iso": due_at.isoformat()},
        "task_prompt": "check in",
        "execution_mode": "inline",
        "status": "active",
        "next_fire_at": due_at,
        "last_fired_at": None,
        "name": "check-in",
        "missed_fire_policy": "coalesce",
        "context_from_schedule_id": None,
        "include_conversation_history": True,
        "date_created": due_at - timedelta(hours=1),
        "date_updated": due_at - timedelta(hours=1),
    }


@pytest.mark.asyncio
async def test_a_claim_evicts_the_schedule_on_every_replica() -> None:
    nats = FakeNatsClient()
    conv, sid = uuid.uuid4(), uuid.uuid4()
    store = _Store(_due_one_shot(conv, sid, datetime.now(UTC) - timedelta(minutes=1)))

    reader = WakeScheduleCollection(registry=await _replica(nats, store), config=_config())
    cached = await reader.get((conv, sid))
    assert cached is not None and cached.status == "active"

    ticking = await _replica(nats, store)
    schedules = WakeScheduleCollection(registry=ticking, config=_config())
    fires = WakeFireCollection(registry=ticking, config=_config())

    async def _fired(trigger: WakeTrigger, fire_id: UUID, pool: Any) -> WakeDispatchResult:
        del trigger, fire_id, pool
        return WakeDispatchResult(status="fired", output_text="ok", latency_ms=3)

    await wake_tick_job(store, None, _fired, schedules=schedules, fires=fires)

    assert store.row["status"] == "expired", "the tick never claimed the due one-shot"
    assert any(
        isinstance(message, CacheInvalidationMessage)
        and message.table == "agent_wake_schedules"
        and message.ids == [str(conv), str(sid)]
        for message in nats.published
    ), "the claim broadcast no invalidation"
    reads_before = store.reads
    fresh = await reader.get((conv, sid))
    assert store.reads == reads_before + 1, "the other replica answered from a cache L3 no longer agrees with"
    assert fresh is not None and fresh.status == "expired"


@pytest.mark.asyncio
async def test_a_tick_given_a_nats_client_refuses_schedules_that_cannot_broadcast() -> None:
    """a tick running under the cross-pod lock on a registry with no client claims rows no replica hears of.

    Its nats_client proves the process has a bus; schedules built on a registry without one would
    evict only locally, and every other replica would keep the pre-claim row. Refused before the
    pass claims anything.
    """
    conv, sid = uuid.uuid4(), uuid.uuid4()
    store = _Store(_due_one_shot(conv, sid, datetime.now(UTC) - timedelta(minutes=1)))
    l1 = SQLiteBackend(db_name=f"wake_tick_bare_{uuid.uuid4().hex[:8]}")
    l1.initialize(_metadata())
    bare = CollectionRegistry()
    bare.configure(l1_backend=l1, l3_pool=store)  # type: ignore[arg-type]
    schedules = WakeScheduleCollection(registry=bare, config=_config())
    fires = WakeFireCollection(registry=bare, config=_config())

    async def _fired(trigger: WakeTrigger, fire_id: UUID, pool: Any) -> WakeDispatchResult:
        del trigger, fire_id, pool
        return WakeDispatchResult(status="fired", output_text="ok", latency_ms=3)

    with pytest.raises(ValueError, match="no NATS client"):
        await wake_tick_job(store, FakeNatsClient(), _fired, schedules=schedules, fires=fires)

    assert store.row["status"] == "active", "the refused tick claimed the schedule anyway"
