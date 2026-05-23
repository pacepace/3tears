"""Integration test: DB-side CHECK constraints fire as expected.

Covers:

- ``agent_wake_schedules`` enum CHECK constraints reject invalid
  ``execution_mode`` / ``status`` / ``missed_fire_policy`` /
  ``delivery_target`` values.
- ``wake_fires_status_check`` rejects values outside the eight-value
  enum (including verifying ``'yielded'`` is accepted, per the
  wake-yield revision).
- ``wake_fires_one_source_check`` rejects "both sources" inserts.
  "Neither source" is now permitted because the subscription-side FK
  is ``ON DELETE SET NULL`` (audit history outlives a subscription
  delete; the resulting row has both source fields NULL).
- ``webhook_subscriptions`` enum CHECK constraints reject invalid
  ``execution_mode`` / ``delivery_target`` / ``verification_scheme`` /
  ``status``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import asyncpg
import pytest
from uuid_utils import uuid7

from threetears.agent.skills.migrations import register as register_skills
from threetears.agent.wake.migrations import register as register_wake
from threetears.conversations.migrations import register as register_conversations
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration


def _new_uuid() -> UUID:
    """Return a fresh UUIDv7 cast to stdlib ``UUID``."""
    return UUID(str(uuid7()))


async def _apply(conn: asyncpg.Connection, schema: str) -> None:
    """Apply conversations + skills + wake migrations."""
    await conn.execute(f'SET search_path TO "{schema}", public')
    runner = MigrationRunner()
    register_conversations(runner)
    register_skills(runner)
    register_wake(runner)
    store = AsyncpgStore(conn)
    await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]


class TestScheduleCheckConstraints:
    """CHECK constraints on ``agent_wake_schedules``."""

    async def test_execution_mode_rejected(self, pg_schema: tuple[str, str]) -> None:
        """An out-of-enum ``execution_mode`` raises ``CheckViolationError``."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            with pytest.raises(asyncpg.exceptions.CheckViolationError):
                await conn.execute(
                    "INSERT INTO agent_wake_schedules "
                    "(conversation_id, schedule_id, user_id, agent_id, "
                    " schedule_type, execution_mode) "
                    "VALUES ($1, $2, $3, $4, $5, $6)",
                    _new_uuid(),
                    _new_uuid(),
                    _new_uuid(),
                    _new_uuid(),
                    "daily_at",
                    "background",
                )
        finally:
            await conn.close()

    async def test_status_rejected(self, pg_schema: tuple[str, str]) -> None:
        """An out-of-enum ``status`` raises ``CheckViolationError``."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            with pytest.raises(asyncpg.exceptions.CheckViolationError):
                await conn.execute(
                    "INSERT INTO agent_wake_schedules "
                    "(conversation_id, schedule_id, user_id, agent_id, "
                    " schedule_type, status) "
                    "VALUES ($1, $2, $3, $4, $5, $6)",
                    _new_uuid(),
                    _new_uuid(),
                    _new_uuid(),
                    _new_uuid(),
                    "daily_at",
                    "frozen",
                )
        finally:
            await conn.close()

    async def test_missed_fire_policy_rejected(self, pg_schema: tuple[str, str]) -> None:
        """An out-of-enum ``missed_fire_policy`` raises ``CheckViolationError``."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            with pytest.raises(asyncpg.exceptions.CheckViolationError):
                await conn.execute(
                    "INSERT INTO agent_wake_schedules "
                    "(conversation_id, schedule_id, user_id, agent_id, "
                    " schedule_type, missed_fire_policy) "
                    "VALUES ($1, $2, $3, $4, $5, $6)",
                    _new_uuid(),
                    _new_uuid(),
                    _new_uuid(),
                    _new_uuid(),
                    "daily_at",
                    "burst",
                )
        finally:
            await conn.close()

    async def test_delivery_target_rejected(self, pg_schema: tuple[str, str]) -> None:
        """An out-of-enum ``delivery_target`` raises ``CheckViolationError``."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            with pytest.raises(asyncpg.exceptions.CheckViolationError):
                await conn.execute(
                    "INSERT INTO agent_wake_schedules "
                    "(conversation_id, schedule_id, user_id, agent_id, "
                    " schedule_type, delivery_target) "
                    "VALUES ($1, $2, $3, $4, $5, $6)",
                    _new_uuid(),
                    _new_uuid(),
                    _new_uuid(),
                    _new_uuid(),
                    "daily_at",
                    "slack",
                )
        finally:
            await conn.close()


class TestFireConstraints:
    """CHECK constraints on ``wake_fires``."""

    async def _seed_schedule(
        self,
        conn: asyncpg.Connection,
        conv_id: UUID,
    ) -> UUID:
        """Seed one schedule for FK satisfaction; return its id."""
        schedule_id = _new_uuid()
        await conn.execute(
            "INSERT INTO agent_wake_schedules "
            "(conversation_id, schedule_id, user_id, agent_id, schedule_type) "
            "VALUES ($1, $2, $3, $4, $5)",
            conv_id,
            schedule_id,
            _new_uuid(),
            _new_uuid(),
            "daily_at",
        )
        return schedule_id

    async def test_one_source_check_accepts_neither(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """A fire row with NEITHER source is accepted (post-FK-SET-NULL state).

        The constraint is mutually-exclusive (NOT both), not strict XOR,
        because the subscription-side FK is ``ON DELETE SET NULL``: a
        webhook fire whose subscription is later deleted ends up with
        both source fields NULL. That state must remain queryable so
        audit history outlives source deletes.
        """
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            await conn.execute(
                "INSERT INTO wake_fires "
                "(conversation_id, fire_id, schedule_id, "
                " webhook_subscription_id, actual_fired_at, status) "
                "VALUES ($1, $2, NULL, NULL, $3, $4)",
                _new_uuid(),
                _new_uuid(),
                datetime.now(UTC),
                "fired",
            )
        finally:
            await conn.close()

    async def test_one_source_check_rejects_both(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """A fire row with BOTH schedule_id AND subscription_id fails."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            conv = _new_uuid()
            schedule_id = await self._seed_schedule(conn, conv)
            # Seed a subscription to provide a real subscription_id.
            sub_id = _new_uuid()
            await conn.execute(
                "INSERT INTO webhook_subscriptions "
                "(conversation_id, subscription_id, user_id, agent_id, "
                " secret_ciphertext) "
                "VALUES ($1, $2, $3, $4, $5)",
                conv,
                sub_id,
                _new_uuid(),
                _new_uuid(),
                b"\x00",
            )
            with pytest.raises(asyncpg.exceptions.CheckViolationError):
                await conn.execute(
                    "INSERT INTO wake_fires "
                    "(conversation_id, fire_id, schedule_id, "
                    " webhook_subscription_id, actual_fired_at, status) "
                    "VALUES ($1, $2, $3, $4, $5, $6)",
                    conv,
                    _new_uuid(),
                    schedule_id,
                    sub_id,
                    datetime.now(UTC),
                    "fired",
                )
        finally:
            await conn.close()

    async def test_yielded_status_accepted(self, pg_schema: tuple[str, str]) -> None:
        """The wake-yield revision's ``'yielded'`` status is accepted."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            conv = _new_uuid()
            schedule_id = await self._seed_schedule(conn, conv)
            await conn.execute(
                "INSERT INTO wake_fires "
                "(conversation_id, fire_id, schedule_id, actual_fired_at, status) "
                "VALUES ($1, $2, $3, $4, $5)",
                conv,
                _new_uuid(),
                schedule_id,
                datetime.now(UTC),
                "yielded",
            )
        finally:
            await conn.close()

    async def test_dispatching_status_accepted(self, pg_schema: tuple[str, str]) -> None:
        """The v004 ``'dispatching'`` in-flight placeholder is accepted."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            conv = _new_uuid()
            schedule_id = await self._seed_schedule(conn, conv)
            await conn.execute(
                "INSERT INTO wake_fires "
                "(conversation_id, fire_id, schedule_id, actual_fired_at, status) "
                "VALUES ($1, $2, $3, $4, $5)",
                conv,
                _new_uuid(),
                schedule_id,
                datetime.now(UTC),
                "dispatching",
            )
        finally:
            await conn.close()

    async def test_status_rejected_outside_enum(self, pg_schema: tuple[str, str]) -> None:
        """A status value outside the eight-value enum fails."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            conv = _new_uuid()
            schedule_id = await self._seed_schedule(conn, conv)
            with pytest.raises(asyncpg.exceptions.CheckViolationError):
                await conn.execute(
                    "INSERT INTO wake_fires "
                    "(conversation_id, fire_id, schedule_id, actual_fired_at, status) "
                    "VALUES ($1, $2, $3, $4, $5)",
                    conv,
                    _new_uuid(),
                    schedule_id,
                    datetime.now(UTC),
                    "garbage",
                )
        finally:
            await conn.close()


class TestSubscriptionCheckConstraints:
    """CHECK constraints on ``webhook_subscriptions``."""

    async def test_status_rejected(self, pg_schema: tuple[str, str]) -> None:
        """An out-of-enum ``status`` fails."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            with pytest.raises(asyncpg.exceptions.CheckViolationError):
                await conn.execute(
                    "INSERT INTO webhook_subscriptions "
                    "(conversation_id, subscription_id, user_id, agent_id, "
                    " secret_ciphertext, status) "
                    "VALUES ($1, $2, $3, $4, $5, $6)",
                    _new_uuid(),
                    _new_uuid(),
                    _new_uuid(),
                    _new_uuid(),
                    b"\x00",
                    "expired",
                )
        finally:
            await conn.close()

    async def test_verification_scheme_rejected(self, pg_schema: tuple[str, str]) -> None:
        """A future scheme not yet in the enum fails."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            with pytest.raises(asyncpg.exceptions.CheckViolationError):
                await conn.execute(
                    "INSERT INTO webhook_subscriptions "
                    "(conversation_id, subscription_id, user_id, agent_id, "
                    " secret_ciphertext, verification_scheme) "
                    "VALUES ($1, $2, $3, $4, $5, $6)",
                    _new_uuid(),
                    _new_uuid(),
                    _new_uuid(),
                    _new_uuid(),
                    b"\x00",
                    "slack_signing",
                )
        finally:
            await conn.close()
