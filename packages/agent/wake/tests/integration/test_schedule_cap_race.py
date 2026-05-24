"""Integration test: the active-schedule cap is race-proof.

Closes the check-then-insert TOCTOU race on the per-conversation
active-schedule cap (PLACEMENT §1.9). Before the fix, two create paths
(the agent ``wake_schedule_create`` tool and the metallm REST router)
each did a non-atomic ``count -> insert``: N concurrent creates could
all read a count under the cap and then all insert, blowing past it.

:func:`threetears.agent.wake.rate_limit.create_schedule_serialized`
serializes count + insert per conversation under a transaction-scoped
``pg_advisory_xact_lock``. This test fires 12 concurrent creates at one
conversation with cap 10 and asserts EXACTLY 10 active schedules land
(the excess 2 rejected with :class:`ScheduleCapExceeded`) -- proving the
race is closed against a REAL Postgres backend (the lock has teeth only
against the DB, not a mock).

Two further tests pin the serial contract: the 11th create against a
full conversation rejects, and creates for DIFFERENT conversations do
not contend (distinct advisory-lock keys), so a busy conversation never
starves an unrelated one.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID

import asyncpg
import pytest
from uuid_utils import uuid7

from threetears.agent.skills.migrations import register as register_skills
from threetears.agent.wake.collections import WakeScheduleCollection
from threetears.agent.wake.entities import WakeScheduleEntity
from threetears.agent.wake.rate_limit import (
    ScheduleCapExceeded,
    create_schedule_serialized,
)
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from threetears.agent.wake.migrations import register as register_wake

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration


_CAP = 10


def _new_uuid() -> UUID:
    return UUID(str(uuid7()))


async def _apply_schema(url: str, schema: str, *, max_size: int) -> asyncpg.Pool:
    """Apply the conversations + skills + wake migrations, return a pool.

    ``max_size`` is bumped well past the cap so N concurrent
    ``create_schedule_serialized`` calls each get their own connection --
    otherwise the test would measure pool-connection contention instead
    of the advisory-lock serialization under test.
    """
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
        min_size=2,
        max_size=max_size,
        server_settings={"search_path": f"{schema}, public"},
        init=init_connection,
    )
    assert pool is not None
    return pool


def _build_collection(pool: asyncpg.Pool) -> WakeScheduleCollection:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return WakeScheduleCollection(registry=registry, config=cfg)


def _build_entity(
    collection: WakeScheduleCollection,
    *,
    conversation_id: UUID,
    user_id: UUID,
    agent_id: UUID,
) -> WakeScheduleEntity:
    now = datetime.now(UTC)
    return collection.create(
        {
            "schedule_id": _new_uuid(),
            "conversation_id": conversation_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "skill_id": None,
            "schedule_type": "interval",
            "schedule_config": {"seconds": 1800},
            "task_prompt": None,
            "execution_mode": "inline",
            "status": "active",
            "next_fire_at": now + timedelta(minutes=30),
            "last_fired_at": None,
            "name": None,
            "missed_fire_policy": "coalesce",
            "context_from_schedule_id": None,
            "delivery_target": "conversation",
            "delivery_config": {},
            "date_created": now,
            "date_updated": now,
        },
    )


async def _count_active(pool: asyncpg.Pool, conversation_id: UUID) -> int:
    value = await pool.fetchval(
        "SELECT COUNT(*) FROM agent_wake_schedules WHERE conversation_id = $1 AND status = 'active'",
        conversation_id,
    )
    return int(value or 0)


class TestActiveScheduleCapRace:
    """N concurrent creates against a capped conversation land exactly cap."""

    async def test_twelve_concurrent_creates_yield_exactly_cap(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema, max_size=20)
        try:
            conversation_id = _new_uuid()
            user_id = _new_uuid()
            agent_id = _new_uuid()
            collection = _build_collection(pool)

            n_attempts = 12  # cap is 10 -> 2 must be rejected

            async def _attempt() -> str:
                entity = _build_entity(
                    collection,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    agent_id=agent_id,
                )
                try:
                    await create_schedule_serialized(
                        collection=collection,
                        entity=entity,
                        conversation_id=conversation_id,
                        cap=_CAP,
                        pool=pool,
                    )
                except ScheduleCapExceeded:
                    return "rejected"
                return "created"

            results = await asyncio.gather(*[_attempt() for _ in range(n_attempts)])

            created = results.count("created")
            rejected = results.count("rejected")

            # Exactly the cap was created; the excess was rejected. The
            # advisory lock makes count+insert atomic per conversation, so
            # the cap holds EXACTLY even under 12-way concurrency.
            assert created == _CAP, f"expected {_CAP} created, got {created}"
            assert rejected == n_attempts - _CAP

            # The DB agrees: exactly cap active rows, never more.
            assert await _count_active(pool, conversation_id) == _CAP
        finally:
            await pool.close()

    async def test_eleventh_create_rejects_when_full(self, pg_schema: tuple[str, str]) -> None:
        """Serially: cap creates succeed, the (cap+1)th raises."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema, max_size=6)
        try:
            conversation_id = _new_uuid()
            user_id = _new_uuid()
            agent_id = _new_uuid()
            collection = _build_collection(pool)

            for _ in range(_CAP):
                entity = _build_entity(
                    collection,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    agent_id=agent_id,
                )
                await create_schedule_serialized(
                    collection=collection,
                    entity=entity,
                    conversation_id=conversation_id,
                    cap=_CAP,
                    pool=pool,
                )

            assert await _count_active(pool, conversation_id) == _CAP

            overflow = _build_entity(
                collection,
                conversation_id=conversation_id,
                user_id=user_id,
                agent_id=agent_id,
            )
            with pytest.raises(ScheduleCapExceeded) as exc_info:
                await create_schedule_serialized(
                    collection=collection,
                    entity=overflow,
                    conversation_id=conversation_id,
                    cap=_CAP,
                    pool=pool,
                )
            # The typed error carries enough context for either consumer
            # to render its own surface message without re-counting.
            assert exc_info.value.cap == _CAP
            assert exc_info.value.count == _CAP
            assert exc_info.value.conversation_id == conversation_id

            # The overflow row was NOT persisted (raise is before/instead
            # of the insert).
            assert await _count_active(pool, conversation_id) == _CAP
        finally:
            await pool.close()

    async def test_distinct_conversations_do_not_contend(self, pg_schema: tuple[str, str]) -> None:
        """Two full conversations each accept their own cap independently.

        The advisory-lock key is per-conversation, so a busy conversation
        never starves an unrelated one: 12 concurrent creates split across
        TWO conversations (cap 10 each) land 10 + 10 = 20, with the excess
        per conversation rejected.
        """
        url, schema = pg_schema
        pool = await _apply_schema(url, schema, max_size=20)
        try:
            conv_a = _new_uuid()
            conv_b = _new_uuid()
            user_id = _new_uuid()
            agent_id = _new_uuid()
            collection = _build_collection(pool)

            async def _attempt(conversation_id: UUID) -> str:
                entity = _build_entity(
                    collection,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    agent_id=agent_id,
                )
                try:
                    await create_schedule_serialized(
                        collection=collection,
                        entity=entity,
                        conversation_id=conversation_id,
                        cap=_CAP,
                        pool=pool,
                    )
                except ScheduleCapExceeded:
                    return "rejected"
                return "created"

            # 12 attempts per conversation, both running concurrently.
            tasks = [_attempt(conv_a) for _ in range(12)] + [_attempt(conv_b) for _ in range(12)]
            await asyncio.gather(*tasks)

            assert await _count_active(pool, conv_a) == _CAP
            assert await _count_active(pool, conv_b) == _CAP
        finally:
            await pool.close()
