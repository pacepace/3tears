"""a targeted L3 UPDATE on a wake row leaves no cache tier, on any replica, serving the old row.

The schedule and subscription collections change rows with targeted UPDATEs that bypass
``save_entity``. Each such row is also read by primary key through ``get`` -- the schedule tools
read before every edit, pause, resume and delete; the subscription tools likewise -- and ``get``
answers from L1, then L2, before L3. A targeted UPDATE that left those tiers alone left every
replica that had read the row serving the old one, and the tools' edit path then wrote that old
row back over L3 with ``save_entity``: a paused schedule resumed, an expired one-shot re-armed,
a rotated webhook secret restored.

Each test reads a row on one replica, changes it on another, and checks both that the reader goes
back to L3 and that the change was broadcast. The broadcast is asserted directly: a re-read on the
writing replica alone passes against a fix that evicts nothing beyond its own process.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import MetaData
from threetears.agent.skills.tables import agent_skills_table
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import CallerTransaction
from threetears.core.collections.registry import CacheInvalidationMessage, CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeNatsClient

from threetears.agent.wake.collections import WakeScheduleCollection, WebhookSubscriptionCollection
from threetears.agent.wake.protected import delete_protected, update_protected
from threetears.agent.wake.rate_limit import resume_schedule_serialized
from threetears.agent.wake.tables import (
    agent_wake_schedules_table,
    wake_fires_table,
    webhook_subscriptions_table,
)
from threetears.agent.wake.tick import _WakeDueSchedule

_SCOPE = "wake-cache-principal"


def _metadata() -> MetaData:
    metadata = MetaData()
    agent_skills_table(metadata)
    agent_wake_schedules_table(metadata)
    wake_fires_table(metadata)
    webhook_subscriptions_table(metadata)
    return metadata


# parity-with: asyncpg.Connection (the transaction + execute seam the serialized resume drives)
class _Conn:
    """one connection onto :class:`_Store`, whose transaction applies on exit."""

    def __init__(self, store: _Store) -> None:
        self.store = store

    def transaction(self) -> _Conn:
        return self

    async def __aenter__(self) -> _Conn:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def execute(self, sql: str, *args: Any) -> str:
        return await self.store.execute(sql, *args)

    async def fetchval(self, sql: str, *args: Any) -> Any:
        return await self.store.fetchval(sql, *args)


# parity-with: asyncpg.Pool (fetchrow / execute / fetchval / fetch / acquire)
class _Store:
    """an L3 holding one row per key; a write applies whatever the test staged for it.

    The tests are about the cache tiers, not the SQL, so a write does not interpret its
    statement: it replaces each staged key's row with the row the test says L3 holds after
    the UPDATE. ``reads`` counts primary-key reads, which is how a test tells an answer
    from L3 from an answer out of a cache.
    """

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.staged: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.reads = 0

    def _apply(self) -> None:
        self.rows.update(self.staged)
        self.staged.clear()

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        table = "webhook_subscriptions" if "FROM webhook_subscriptions" in sql else "agent_wake_schedules"
        self.reads += 1
        # match on whichever columns the WHERE names: the pk for ``get``, the agent and bare id for
        # ``find_for_agent``.
        where = {col: args[int(pos) - 1] for col, pos in re.findall(r"(\w+) = \$(\d+)", sql.split("WHERE", 1)[1])}
        found: dict[str, Any] | None = None
        for (row_table, _, _), row in self.rows.items():
            if row_table == table and all(row.get(col) == value for col, value in where.items()):
                found = dict(row)
        return found

    async def execute(self, sql: str, *args: Any) -> str:
        del sql, args
        self._apply()
        return "UPDATE 1"

    async def fetchval(self, sql: str, *args: Any) -> Any:
        del args
        if "COUNT" in sql.upper():
            return 0
        self._apply()
        return uuid.uuid4()

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        del sql, args
        return []

    def acquire(self) -> _Conn:
        return _Conn(self)


async def _replica(nats: FakeNatsClient, store: _Store) -> CollectionRegistry:
    l1 = SQLiteBackend(db_name=f"wake_cache_{uuid.uuid4().hex[:8]}")
    l1.initialize(_metadata())
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l2_client=nats, l3_pool=store, kv_key_scope=_SCOPE)  # type: ignore[arg-type]
    await registry.start_invalidation_listener(nats)  # type: ignore[arg-type]
    return registry


def _config() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


def _schedule_row(conv: UUID, sid: UUID, **changes: Any) -> dict[str, Any]:
    now = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
    row: dict[str, Any] = {
        "conversation_id": conv,
        "schedule_id": sid,
        "user_id": uuid.uuid4(),
        "agent_id": uuid.uuid4(),
        "skill_id": None,
        "schedule_type": "every_n_hours",
        "schedule_config": {"n": 3},
        "task_prompt": "check in",
        "execution_mode": "inline",
        "status": "active",
        "next_fire_at": now + timedelta(hours=3),
        "last_fired_at": None,
        "name": "check-in",
        "missed_fire_policy": "coalesce",
        "context_from_schedule_id": None,
        "include_conversation_history": True,
        "date_created": now,
        "date_updated": now,
    }
    row.update(changes)
    return row


def _subscription_row(conv: UUID, sub: UUID, **changes: Any) -> dict[str, Any]:
    now = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
    row: dict[str, Any] = {
        "conversation_id": conv,
        "subscription_id": sub,
        "user_id": uuid.uuid4(),
        "agent_id": uuid.uuid4(),
        "default_skill_id": None,
        "name": "deploys",
        "secret_ciphertext": b"old-secret",
        "allowed_source_pattern": None,
        "execution_mode": "inline",
        "task_prompt_template": None,
        "verification_scheme": "generic_hmac_sha256",
        "status": "active",
        "rate_limit_per_minute": None,
        "last_fired_at": None,
        "date_created": now,
        "date_updated": now,
    }
    row.update(changes)
    return row


def _broadcast_for(nats: FakeNatsClient, table: str, *ids: UUID) -> bool:
    wanted = [str(i) for i in ids]
    return any(
        isinstance(message, CacheInvalidationMessage) and message.table == table and message.ids == wanted
        for message in nats.published
    )


_ScheduleWrite = Callable[[WakeScheduleCollection, UUID, UUID], Awaitable[Any]]

_LATER = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
_EXPECTED = datetime(2026, 9, 30, 15, 0, tzinfo=UTC)

_SCHEDULE_WRITES: dict[str, tuple[_ScheduleWrite, dict[str, Any]]] = {
    "pause": (lambda c, conv, sid: c.pause(conv, sid), {"status": "paused", "next_fire_at": None}),
    "resume": (
        lambda c, conv, sid: c.resume(conv, sid, next_fire_at=_LATER),
        {"status": "active", "next_fire_at": _LATER},
    ),
    "update_next_fire_at": (
        lambda c, conv, sid: c.update_next_fire_at(conv, sid, next_fire_at=_LATER),
        {"next_fire_at": _LATER},
    ),
    "update_next_fire_at_and_last": (
        lambda c, conv, sid: c.update_next_fire_at(conv, sid, next_fire_at=_LATER, last_fired_at=_EXPECTED),
        {"next_fire_at": _LATER, "last_fired_at": _EXPECTED},
    ),
    "mark_expired": (lambda c, conv, sid: c.mark_expired(conv, sid), {"status": "expired", "next_fire_at": None}),
    "claim_and_reschedule": (
        lambda c, conv, sid: c.claim_and_reschedule(
            conversation_id=conv,
            schedule_id=sid,
            expected_next_fire=_EXPECTED,
            computed_next_fire=None,
            new_status="expired",
            now=_EXPECTED,
        ),
        {"status": "expired", "next_fire_at": None, "last_fired_at": _EXPECTED},
    ),
}


class TestAScheduleUpdateReachesEveryReader:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("write", sorted(_SCHEDULE_WRITES))
    async def test_a_replica_that_read_the_row_reads_l3_after_the_update(self, write: str) -> None:
        nats = FakeNatsClient()
        store = _Store()
        conv, sid = uuid.uuid4(), uuid.uuid4()
        store.rows[("agent_wake_schedules", str(conv), str(sid))] = _schedule_row(conv, sid)
        reader = WakeScheduleCollection(registry=await _replica(nats, store), config=_config())
        writer = WakeScheduleCollection(registry=await _replica(nats, store), config=_config())

        cached = await reader.get((conv, sid))
        assert cached is not None and cached.status == "active"
        reads_before = store.reads

        perform, changes = _SCHEDULE_WRITES[write]
        store.staged[("agent_wake_schedules", str(conv), str(sid))] = _schedule_row(conv, sid, **changes)
        await perform(writer, conv, sid)

        assert _broadcast_for(nats, "agent_wake_schedules", conv, sid), f"{write} broadcast no invalidation"
        fresh = await reader.get((conv, sid))
        assert store.reads == reads_before + 1, f"{write}: the reader answered from a cache L3 no longer agrees with"
        assert fresh is not None
        assert fresh.status == changes.get("status", "active")

    @pytest.mark.asyncio
    async def test_a_lost_claim_changes_nothing_and_broadcasts_nothing(self) -> None:
        """the CAS missed, so L3 still holds the row every cache holds; there is nothing to evict."""
        nats = FakeNatsClient()
        store = _Store()
        conv, sid = uuid.uuid4(), uuid.uuid4()
        store.rows[("agent_wake_schedules", str(conv), str(sid))] = _schedule_row(conv, sid)
        writer = WakeScheduleCollection(registry=await _replica(nats, store), config=_config())

        async def _missed(sql: str, *args: Any) -> Any:
            del sql, args
            return None

        store.fetchval = _missed  # type: ignore[method-assign]
        claimed = await writer.claim_and_reschedule(
            conversation_id=conv,
            schedule_id=sid,
            expected_next_fire=_EXPECTED,
            computed_next_fire=_LATER,
            new_status="active",
            now=_EXPECTED,
        )

        assert claimed is False
        assert not _broadcast_for(nats, "agent_wake_schedules", conv, sid)

    @pytest.mark.asyncio
    async def test_an_update_that_raises_still_evicts(self) -> None:
        """a write whose outcome is unknown may have reached L3; the caches must not outlive it."""
        nats = FakeNatsClient()
        store = _Store()
        conv, sid = uuid.uuid4(), uuid.uuid4()
        store.rows[("agent_wake_schedules", str(conv), str(sid))] = _schedule_row(conv, sid)
        writer = WakeScheduleCollection(registry=await _replica(nats, store), config=_config())

        async def _lost(sql: str, *args: Any) -> str:
            del sql, args
            raise ConnectionError("connection lost after the statement was sent")

        store.execute = _lost  # type: ignore[method-assign]
        with pytest.raises(ConnectionError):
            await writer.pause(conv, sid)

        assert _broadcast_for(nats, "agent_wake_schedules", conv, sid)


class TestASerializedResumeSettlesAfterItsTransaction:
    @pytest.mark.asyncio
    async def test_the_cap_checked_resume_reaches_every_reader(self) -> None:
        nats = FakeNatsClient()
        store = _Store()
        conv, sid = uuid.uuid4(), uuid.uuid4()
        store.rows[("agent_wake_schedules", str(conv), str(sid))] = _schedule_row(
            conv, sid, status="paused", next_fire_at=None
        )
        reader = WakeScheduleCollection(registry=await _replica(nats, store), config=_config())
        writer = WakeScheduleCollection(registry=await _replica(nats, store), config=_config())
        cached = await reader.get((conv, sid))
        assert cached is not None and cached.status == "paused"

        store.staged[("agent_wake_schedules", str(conv), str(sid))] = _schedule_row(conv, sid, next_fire_at=_LATER)
        await resume_schedule_serialized(
            collection=writer,
            agent_id=uuid.uuid4(),
            conversation_id=conv,
            schedule_id=sid,
            next_fire_at=_LATER,
            cap=10,
            pool=store,
        )

        assert _broadcast_for(nats, "agent_wake_schedules", conv, sid)
        fresh = await reader.get((conv, sid))
        assert fresh is not None and fresh.status == "active"

    @pytest.mark.asyncio
    async def test_a_resume_on_a_bare_connection_is_refused(self) -> None:
        """a caller's transaction is the only place the row can be settled once it is final."""
        store = _Store()
        collection = WakeScheduleCollection(registry=await _replica(FakeNatsClient(), store), config=_config())
        with pytest.raises(ValueError, match="CallerTransaction"):
            await collection.resume(uuid.uuid4(), uuid.uuid4(), next_fire_at=_LATER, conn=_Conn(store))

    @pytest.mark.asyncio
    async def test_a_resume_joined_to_a_callers_transaction_waits_for_it(self) -> None:
        """nothing is evicted before the commit: a reader in between would re-cache the old row."""
        nats = FakeNatsClient()
        store = _Store()
        conv, sid = uuid.uuid4(), uuid.uuid4()
        collection = WakeScheduleCollection(registry=await _replica(nats, store), config=_config())
        conn = _Conn(store)
        async with CallerTransaction(conn):
            await collection.resume(conv, sid, next_fire_at=_LATER, conn=conn)
            assert not _broadcast_for(nats, "agent_wake_schedules", conv, sid)
        assert _broadcast_for(nats, "agent_wake_schedules", conv, sid)


class TestAProtectedWakeChangeSettlesAfterItsTransaction:
    """0.57.0's two doors onto a protected wake write L3 past ``save_entity``, so they settle too."""

    @pytest.mark.asyncio
    async def test_a_protected_schedule_change_reaches_every_reader(self) -> None:
        nats = FakeNatsClient()
        store = _Store()
        conv, sid, agent_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        key = ("agent_wake_schedules", str(conv), str(sid))
        store.rows[key] = _schedule_row(conv, sid, agent_id=agent_id, protected=True)
        reader = WakeScheduleCollection(registry=await _replica(nats, store), config=_config())
        writer = WakeScheduleCollection(registry=await _replica(nats, store), config=_config())
        assert await reader.get((conv, sid)) is not None
        reads_before = store.reads

        store.staged[key] = _schedule_row(conv, sid, agent_id=agent_id, protected=True, schedule_config={"n": 6})
        await update_protected(collection=writer, agent_id=agent_id, schedule_id=sid, schedule_config={"n": 6})

        assert _broadcast_for(nats, "agent_wake_schedules", conv, sid), "update_protected broadcast no invalidation"
        fresh = await reader.get((conv, sid))
        assert store.reads > reads_before + 1, "the reader answered from a cache L3 no longer agrees with"
        assert fresh is not None and fresh.schedule_config == {"n": 6}

    @pytest.mark.asyncio
    async def test_a_protected_delete_on_its_own_transaction_reaches_every_reader(self) -> None:
        nats = FakeNatsClient()
        store = _Store()
        conv, sid, agent_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        store.rows[("agent_wake_schedules", str(conv), str(sid))] = _schedule_row(
            conv, sid, agent_id=agent_id, protected=True
        )
        collection = WakeScheduleCollection(registry=await _replica(nats, store), config=_config())

        await delete_protected(collection=collection, agent_id=agent_id, schedule_id=sid)

        assert _broadcast_for(nats, "agent_wake_schedules", conv, sid)

    @pytest.mark.asyncio
    async def test_a_protected_delete_joined_to_the_agent_deletion_waits_for_it(self) -> None:
        """evicting before the agent deletion commits lets a reader re-cache the row L3 still holds."""
        nats = FakeNatsClient()
        store = _Store()
        conv, sid, agent_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        store.rows[("agent_wake_schedules", str(conv), str(sid))] = _schedule_row(
            conv, sid, agent_id=agent_id, protected=True
        )
        collection = WakeScheduleCollection(registry=await _replica(nats, store), config=_config())
        conn = _Conn(store)
        async with CallerTransaction(conn):
            await delete_protected(collection=collection, agent_id=agent_id, schedule_id=sid, conn=conn)
            assert not _broadcast_for(nats, "agent_wake_schedules", conv, sid)
        assert _broadcast_for(nats, "agent_wake_schedules", conv, sid)

    @pytest.mark.asyncio
    async def test_a_protected_delete_on_a_bare_connection_is_refused(self) -> None:
        store = _Store()
        conv, sid, agent_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        store.rows[("agent_wake_schedules", str(conv), str(sid))] = _schedule_row(
            conv, sid, agent_id=agent_id, protected=True
        )
        collection = WakeScheduleCollection(registry=await _replica(FakeNatsClient(), store), config=_config())
        with pytest.raises(ValueError, match="CallerTransaction"):
            await delete_protected(collection=collection, agent_id=agent_id, schedule_id=sid, conn=_Conn(store))


_SubscriptionWrite = Callable[[WebhookSubscriptionCollection, UUID, UUID], Awaitable[Any]]

_SUBSCRIPTION_WRITES: dict[str, tuple[_SubscriptionWrite, dict[str, Any]]] = {
    "rotate_secret": (
        lambda c, conv, sub: c.rotate_secret(conv, sub, new_ciphertext=b"new-secret"),
        {"secret_ciphertext": b"new-secret"},
    ),
    "pause": (lambda c, conv, sub: c.pause(conv, sub), {"status": "paused"}),
    "resume": (lambda c, conv, sub: c.resume(conv, sub), {"status": "active"}),
    "record_fire": (lambda c, conv, sub: c.record_fire(conv, sub, fired_at=_EXPECTED), {"last_fired_at": _EXPECTED}),
}


class TestASubscriptionUpdateReachesEveryReader:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("write", sorted(_SUBSCRIPTION_WRITES))
    async def test_a_replica_that_read_the_row_reads_l3_after_the_update(self, write: str) -> None:
        nats = FakeNatsClient()
        store = _Store()
        conv, sub = uuid.uuid4(), uuid.uuid4()
        store.rows[("webhook_subscriptions", str(conv), str(sub))] = _subscription_row(conv, sub)
        reader = WebhookSubscriptionCollection(registry=await _replica(nats, store), config=_config())
        writer = WebhookSubscriptionCollection(registry=await _replica(nats, store), config=_config())

        assert await reader.get((conv, sub)) is not None
        reads_before = store.reads

        perform, changes = _SUBSCRIPTION_WRITES[write]
        store.staged[("webhook_subscriptions", str(conv), str(sub))] = _subscription_row(conv, sub, **changes)
        await perform(writer, conv, sub)

        assert _broadcast_for(nats, "webhook_subscriptions", conv, sub), f"{write} broadcast no invalidation"
        fresh = await reader.get((conv, sub))
        assert store.reads == reads_before + 1, f"{write}: the reader answered from a cache L3 no longer agrees with"
        assert fresh is not None

    @pytest.mark.asyncio
    async def test_a_rotated_secret_is_not_restored_by_a_later_edit(self) -> None:
        """the security case: the edit path saves the whole row it read, secret included."""
        nats = FakeNatsClient()
        store = _Store()
        conv, sub = uuid.uuid4(), uuid.uuid4()
        store.rows[("webhook_subscriptions", str(conv), str(sub))] = _subscription_row(conv, sub)
        editor = WebhookSubscriptionCollection(registry=await _replica(nats, store), config=_config())
        rotator = WebhookSubscriptionCollection(registry=await _replica(nats, store), config=_config())
        assert await editor.get((conv, sub)) is not None

        store.staged[("webhook_subscriptions", str(conv), str(sub))] = _subscription_row(
            conv, sub, secret_ciphertext=b"new-secret"
        )
        await rotator.rotate_secret(conv, sub, new_ciphertext=b"new-secret")

        entity = await editor.get((conv, sub))
        assert entity is not None
        assert bytes(entity.secret_ciphertext) == b"new-secret"


class TestADueScheduleOutlivesAnEviction:
    @pytest.mark.asyncio
    async def test_the_tick_reads_what_it_listed_after_the_row_is_evicted(self) -> None:
        """the tick reads a due row's fields AFTER it claims the row, and the claim evicts it.

        A due schedule that read its fields through the entity's L1 proxy would lose them to its
        own claim -- or to any other replica's write to the row broadcast while the tick ran.
        """
        store = _Store()
        conv, sid = uuid.uuid4(), uuid.uuid4()
        collection = WakeScheduleCollection(registry=await _replica(FakeNatsClient(), store), config=_config())
        row = _schedule_row(conv, sid)
        due = _WakeDueSchedule(collection.entity_class(row, is_new=False, collection=collection))

        collection.evict_from_cache_sync((conv, sid))

        assert due.job_id == sid
        assert due.name == "check-in"
        assert due.payload["task_prompt"] == "check in"
