"""Integration test: composite-pk lookup round-trips through the Collections.

Verifies that each Collection's ``save_entity`` + ``get`` round-trip
preserves typed fields and that the domain methods on each Collection
return the expected shapes.

This exercises the BaseCollection contract directly: no L1 backend
configured, no NATS client; the L3 pool path is the single source of
truth for the round-trip.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import asyncpg
import pytest
from uuid_utils import uuid7

from threetears.agent.skills.migrations import register as register_skills
from threetears.agent.wake.collections import (
    WakeFireCollection,
    WakeScheduleCollection,
    WebhookSubscriptionCollection,
)
from threetears.agent.wake.entities import (
    WakeFireEntity,
    WakeScheduleEntity,
    WebhookSubscriptionEntity,
)
from threetears.agent.wake.migrations import register as register_wake
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration


def _new_uuid() -> UUID:
    """Return a fresh UUIDv7 cast to stdlib ``UUID``."""
    return UUID(str(uuid7()))


async def _apply_schema(url: str, schema: str) -> asyncpg.Pool:
    """Apply migrations and return a pool bound to the schema."""
    setup_conn = await asyncpg.connect(url)
    try:
        await setup_conn.execute(f'SET search_path TO "{schema}", public')
        runner = MigrationRunner()
        register_conversations(runner)
        register_skills(runner)
        register_wake(runner)
        store = AsyncpgStore(setup_conn)
        await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
    finally:
        await setup_conn.close()
    pool = await asyncpg.create_pool(
        url,
        min_size=1,
        max_size=4,
        server_settings={"search_path": f"{schema}, public"},
        init=init_connection,
    )
    assert pool is not None
    return pool


def _build_schedule_collection(pool: asyncpg.Pool) -> WakeScheduleCollection:
    """Build a Collection bound to the pool with no L1 / L2 wiring."""
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return WakeScheduleCollection(registry=registry, config=cfg)


def _build_fire_collection(pool: asyncpg.Pool) -> WakeFireCollection:
    """Build a fire Collection sharing the same pool."""
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return WakeFireCollection(registry=registry, config=cfg)


def _build_subscription_collection(pool: asyncpg.Pool) -> WebhookSubscriptionCollection:
    """Build a subscription Collection sharing the same pool."""
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return WebhookSubscriptionCollection(registry=registry, config=cfg)


class TestScheduleRoundTrip:
    """``save_entity`` + ``get`` preserves every typed field."""

    async def test_round_trip_through_collection(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """Save a schedule, read it back via composite pk, fields match."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            coll = _build_schedule_collection(pool)
            conv = _new_uuid()
            sched = _new_uuid()
            user = _new_uuid()
            agent = _new_uuid()
            now = datetime.now(UTC)
            data: dict[str, Any] = {
                "conversation_id": conv,
                "schedule_id": sched,
                "user_id": user,
                "agent_id": agent,
                "schedule_type": "cron",
                "schedule_config": {"expr": "0 9 * * *"},
                "task_prompt": "Morning brief",
                "execution_mode": "inline",
                "status": "active",
                "next_fire_at": now,
                "name": "morning-brief",
                "missed_fire_policy": "coalesce",
                "date_created": now,
                "date_updated": now,
            }
            entity = coll.create(data)
            await coll.save_entity(entity)

            fetched = await coll.get((conv, sched))
            assert fetched is not None
            assert isinstance(fetched, WakeScheduleEntity)
            assert fetched.schedule_id == sched
            assert fetched.conversation_id == conv
            assert fetched.user_id == user
            assert fetched.agent_id == agent
            assert fetched.schedule_type == "cron"
            assert fetched.schedule_config == {"expr": "0 9 * * *"}
            assert fetched.task_prompt == "Morning brief"
            assert fetched.missed_fire_policy == "coalesce"
            assert fetched.status == "active"
        finally:
            await pool.close()

    async def test_pause_and_resume_round_trip(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """``pause`` then ``resume`` flips status + next_fire_at."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            coll = _build_schedule_collection(pool)
            conv = _new_uuid()
            sched = _new_uuid()
            now = datetime.now(UTC)
            entity = coll.create(
                {
                    "conversation_id": conv,
                    "schedule_id": sched,
                    "user_id": _new_uuid(),
                    "agent_id": _new_uuid(),
                    "schedule_type": "interval",
                    "schedule_config": {"seconds": 60},
                    "next_fire_at": now,
                    "date_created": now,
                    "date_updated": now,
                },
            )
            await coll.save_entity(entity)

            await coll.pause(conv, sched)
            paused = await coll.get((conv, sched))
            assert paused is not None
            assert paused.status == "paused"
            assert paused.next_fire_at is None

            future = now + timedelta(seconds=60)
            await coll.resume(conv, sched, next_fire_at=future)
            resumed = await coll.get((conv, sched))
            assert resumed is not None
            assert resumed.status == "active"
            assert resumed.next_fire_at == future
        finally:
            await pool.close()

    async def test_list_due_for_tick_returns_only_active_due(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """``list_due_for_tick`` filters by status='active' AND next_fire_at <= now."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            coll = _build_schedule_collection(pool)
            base = datetime.now(UTC)
            conv = _new_uuid()
            # past-due active
            due = coll.create(
                {
                    "conversation_id": conv,
                    "schedule_id": _new_uuid(),
                    "user_id": _new_uuid(),
                    "agent_id": _new_uuid(),
                    "schedule_type": "interval",
                    "next_fire_at": base - timedelta(seconds=10),
                    "date_created": base,
                    "date_updated": base,
                },
            )
            await coll.save_entity(due)
            # future active
            future = coll.create(
                {
                    "conversation_id": conv,
                    "schedule_id": _new_uuid(),
                    "user_id": _new_uuid(),
                    "agent_id": _new_uuid(),
                    "schedule_type": "interval",
                    "next_fire_at": base + timedelta(minutes=5),
                    "date_created": base,
                    "date_updated": base,
                },
            )
            await coll.save_entity(future)
            # paused
            paused = coll.create(
                {
                    "conversation_id": conv,
                    "schedule_id": _new_uuid(),
                    "user_id": _new_uuid(),
                    "agent_id": _new_uuid(),
                    "schedule_type": "interval",
                    "status": "paused",
                    "next_fire_at": None,
                    "date_created": base,
                    "date_updated": base,
                },
            )
            await coll.save_entity(paused)

            results = await coll.list_due_for_tick(base)
            ids = {row.schedule_id for row in results}
            assert due.schedule_id in ids
            assert future.schedule_id not in ids
            assert paused.schedule_id not in ids
        finally:
            await pool.close()

    async def test_count_active_for_conversation(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """``count_active_for_conversation`` counts active rows only."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            coll = _build_schedule_collection(pool)
            conv = _new_uuid()
            base = datetime.now(UTC)
            for _ in range(3):
                entity = coll.create(
                    {
                        "conversation_id": conv,
                        "schedule_id": _new_uuid(),
                        "user_id": _new_uuid(),
                        "agent_id": _new_uuid(),
                        "schedule_type": "interval",
                        "date_created": base,
                        "date_updated": base,
                    },
                )
                await coll.save_entity(entity)
            # one paused (should not count)
            paused = coll.create(
                {
                    "conversation_id": conv,
                    "schedule_id": _new_uuid(),
                    "user_id": _new_uuid(),
                    "agent_id": _new_uuid(),
                    "schedule_type": "interval",
                    "status": "paused",
                    "date_created": base,
                    "date_updated": base,
                },
            )
            await coll.save_entity(paused)

            assert await coll.count_active_for_conversation(conv) == 3
        finally:
            await pool.close()


class TestFireRoundTrip:
    """Fire Collection: record + get + list_for_schedule."""

    async def test_record_and_get_round_trip(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """``record`` persists a fire; ``get`` reads it back via tuple pk."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            sched_coll = _build_schedule_collection(pool)
            fire_coll = _build_fire_collection(pool)

            conv = _new_uuid()
            sched = _new_uuid()
            now = datetime.now(UTC)

            schedule = sched_coll.create(
                {
                    "conversation_id": conv,
                    "schedule_id": sched,
                    "user_id": _new_uuid(),
                    "agent_id": _new_uuid(),
                    "schedule_type": "daily_at",
                    "date_created": now,
                    "date_updated": now,
                },
            )
            await sched_coll.save_entity(schedule)

            fire_id = _new_uuid()
            fire = fire_coll.create(
                {
                    "conversation_id": conv,
                    "fire_id": fire_id,
                    "schedule_id": sched,
                    "actual_fired_at": now,
                    "status": "fired",
                    "display_suppressed": False,
                    "output_text": "ok",
                    "latency_ms": 50,
                    "date_created": now,
                },
            )
            await fire_coll.record(conv, fire)

            fetched = await fire_coll.get((conv, fire_id))
            assert fetched is not None
            assert isinstance(fetched, WakeFireEntity)
            assert fetched.fire_id == fire_id
            assert fetched.schedule_id == sched
            assert fetched.status == "fired"
            assert fetched.output_text == "ok"
        finally:
            await pool.close()

    async def test_record_rejects_mismatched_conversation_id(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """``record`` raises ValueError if the partition contract is violated."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            fire_coll = _build_fire_collection(pool)
            entity = fire_coll.create(
                {
                    "conversation_id": _new_uuid(),
                    "fire_id": _new_uuid(),
                    "schedule_id": _new_uuid(),
                    "actual_fired_at": datetime.now(UTC),
                    "status": "fired",
                },
            )
            with pytest.raises(ValueError):
                await fire_coll.record(_new_uuid(), entity)
        finally:
            await pool.close()

    async def test_latest_for_schedule(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """``latest_for_schedule`` returns the most recent fire only."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            sched_coll = _build_schedule_collection(pool)
            fire_coll = _build_fire_collection(pool)
            conv = _new_uuid()
            sched = _new_uuid()
            now = datetime.now(UTC)

            schedule = sched_coll.create(
                {
                    "conversation_id": conv,
                    "schedule_id": sched,
                    "user_id": _new_uuid(),
                    "agent_id": _new_uuid(),
                    "schedule_type": "interval",
                    "date_created": now,
                    "date_updated": now,
                },
            )
            await sched_coll.save_entity(schedule)

            for i in range(3):
                fire = fire_coll.create(
                    {
                        "conversation_id": conv,
                        "fire_id": _new_uuid(),
                        "schedule_id": sched,
                        "actual_fired_at": now + timedelta(seconds=i),
                        "status": "fired",
                    },
                )
                await fire_coll.record(conv, fire)
            latest = await fire_coll.latest_for_schedule(conv, sched)
            assert latest is not None
            assert latest.actual_fired_at == now + timedelta(seconds=2)
        finally:
            await pool.close()

    async def test_count_in_window(self, pg_schema: tuple[str, str]) -> None:
        """``count_in_window`` returns fires since the supplied instant."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            sched_coll = _build_schedule_collection(pool)
            fire_coll = _build_fire_collection(pool)
            conv = _new_uuid()
            sched = _new_uuid()
            now = datetime.now(UTC)

            await sched_coll.save_entity(
                sched_coll.create(
                    {
                        "conversation_id": conv,
                        "schedule_id": sched,
                        "user_id": _new_uuid(),
                        "agent_id": _new_uuid(),
                        "schedule_type": "interval",
                        "date_created": now,
                        "date_updated": now,
                    },
                ),
            )
            for i in range(5):
                await fire_coll.record(
                    conv,
                    fire_coll.create(
                        {
                            "conversation_id": conv,
                            "fire_id": _new_uuid(),
                            "schedule_id": sched,
                            "actual_fired_at": now - timedelta(minutes=i),
                            "status": "fired",
                        },
                    ),
                )
            # only 3 fires within the last 2-minute window (i=0, 1, 2)
            window_start = now - timedelta(minutes=2, seconds=30)
            assert await fire_coll.count_in_window(conv, since=window_start) == 3
        finally:
            await pool.close()


class TestSubscriptionRoundTrip:
    """Subscription Collection round-trip + domain methods."""

    async def test_round_trip(self, pg_schema: tuple[str, str]) -> None:
        """save + get preserves every typed field."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            coll = _build_subscription_collection(pool)
            conv = _new_uuid()
            sub = _new_uuid()
            now = datetime.now(UTC)
            await coll.save_entity(
                coll.create(
                    {
                        "conversation_id": conv,
                        "subscription_id": sub,
                        "user_id": _new_uuid(),
                        "agent_id": _new_uuid(),
                        "name": "github-events",
                        "secret_ciphertext": b"\xab\xcd",
                        "task_prompt_template": "Handle {{event.action}}",
                        "rate_limit_per_minute": 30,
                        "date_created": now,
                        "date_updated": now,
                    },
                ),
            )
            fetched = await coll.get((conv, sub))
            assert fetched is not None
            assert isinstance(fetched, WebhookSubscriptionEntity)
            assert fetched.subscription_id == sub
            assert fetched.secret_ciphertext == b"\xab\xcd"
            assert fetched.task_prompt_template == "Handle {{event.action}}"
            assert fetched.rate_limit_per_minute == 30
            assert fetched.verification_scheme == "generic_hmac_sha256"
            assert fetched.status == "active"
        finally:
            await pool.close()

    async def test_find_by_id_cross_partition(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """``find_by_id`` resolves a subscription with no conversation context."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            coll = _build_subscription_collection(pool)
            conv = _new_uuid()
            sub = _new_uuid()
            now = datetime.now(UTC)
            await coll.save_entity(
                coll.create(
                    {
                        "conversation_id": conv,
                        "subscription_id": sub,
                        "user_id": _new_uuid(),
                        "agent_id": _new_uuid(),
                        "secret_ciphertext": b"\x01",
                        "date_created": now,
                        "date_updated": now,
                    },
                ),
            )
            found = await coll.find_by_id(sub)
            assert found is not None
            assert found.conversation_id == conv
            assert found.subscription_id == sub
        finally:
            await pool.close()

    async def test_rotate_secret(self, pg_schema: tuple[str, str]) -> None:
        """``rotate_secret`` overwrites the ciphertext."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            coll = _build_subscription_collection(pool)
            conv = _new_uuid()
            sub = _new_uuid()
            now = datetime.now(UTC)
            await coll.save_entity(
                coll.create(
                    {
                        "conversation_id": conv,
                        "subscription_id": sub,
                        "user_id": _new_uuid(),
                        "agent_id": _new_uuid(),
                        "secret_ciphertext": b"\x01",
                        "date_created": now,
                        "date_updated": now,
                    },
                ),
            )
            await coll.rotate_secret(conv, sub, new_ciphertext=b"\x02\x03")
            after = await coll.get((conv, sub))
            assert after is not None
            assert after.secret_ciphertext == b"\x02\x03"
        finally:
            await pool.close()

    async def test_pause_resume_subscription(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """``pause`` / ``resume`` flips status."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            coll = _build_subscription_collection(pool)
            conv = _new_uuid()
            sub = _new_uuid()
            now = datetime.now(UTC)
            await coll.save_entity(
                coll.create(
                    {
                        "conversation_id": conv,
                        "subscription_id": sub,
                        "user_id": _new_uuid(),
                        "agent_id": _new_uuid(),
                        "secret_ciphertext": b"\x01",
                        "date_created": now,
                        "date_updated": now,
                    },
                ),
            )
            await coll.pause(conv, sub)
            paused = await coll.get((conv, sub))
            assert paused is not None
            assert paused.status == "paused"
            await coll.resume(conv, sub)
            resumed = await coll.get((conv, sub))
            assert resumed is not None
            assert resumed.status == "active"
        finally:
            await pool.close()

    async def test_record_fire_stamps_last_fired_at(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """``record_fire`` updates the denormalised ``last_fired_at``."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            coll = _build_subscription_collection(pool)
            conv = _new_uuid()
            sub = _new_uuid()
            now = datetime.now(UTC)
            await coll.save_entity(
                coll.create(
                    {
                        "conversation_id": conv,
                        "subscription_id": sub,
                        "user_id": _new_uuid(),
                        "agent_id": _new_uuid(),
                        "secret_ciphertext": b"\x01",
                        "date_created": now,
                        "date_updated": now,
                    },
                ),
            )
            await coll.record_fire(conv, sub, fired_at=now)
            after = await coll.get((conv, sub))
            assert after is not None
            assert after.last_fired_at == now
        finally:
            await pool.close()
