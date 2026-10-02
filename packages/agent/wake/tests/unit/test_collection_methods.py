"""Unit tests for how the agent-wake collections write a new row.

The Collection classes are wired to a real Postgres pool in
integration tests; the unit suite drives each collection's
``save_to_store`` for a NEW row against a connection that records the
statement, and checks:

- every value is bound at the position of its column in the INSERT's own
  column list, so positional asyncpg parameters stay aligned with the SQL
  placeholders;
- the upsert carries the expected conflict target and update set;
- defaults applied for omitted columns mirror the schema's DEFAULT
  semantics;
- class attributes (``primary_key_column``, ``partition_column``)
  declare the documented contract.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from uuid_utils import uuid7

from threetears.agent.wake.collections import (
    WakeFireCollection,
    WakeScheduleCollection,
    WebhookSubscriptionCollection,
)
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

#: the shape of every new-row write: the insert, its placeholders, and the upsert tail.
_UPSERT = re.compile(
    r"^INSERT INTO (?P<table>\w+) \((?P<columns>[^)]*)\) VALUES \((?P<placeholders>[^)]*)\) "
    r"ON CONFLICT \((?P<conflict>[^)]*)\) DO UPDATE SET (?P<updates>.+)$"
)


def _new_uuid() -> UUID:
    """Return a fresh UUIDv7 cast to stdlib ``UUID``."""
    return UUID(str(uuid7()))


# parity-exempt: records the one asyncpg Connection.execute call save_to_store makes on a new row; nothing else is reached
class _RecordingConnection:
    """an asyncpg-compatible connection that records each statement and its parameters."""

    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, sql: str, *params: Any) -> str:
        self.statements.append((sql, params))
        return "INSERT 0 1"


class _Insert:
    """one recorded new-row write, read back by column name."""

    def __init__(self, sql: str, params: tuple[Any, ...]) -> None:
        match = _UPSERT.match(sql)
        assert match is not None, f"not an upsert: {sql}"
        self.sql = sql
        self.table = match["table"]
        self.columns = tuple(c.strip() for c in match["columns"].split(","))
        self.placeholders = tuple(p.strip() for p in match["placeholders"].split(","))
        self.conflict = tuple(c.strip() for c in match["conflict"].split(","))
        self.updates = tuple(u.strip() for u in match["updates"].split(","))
        self.params = params

    def __getitem__(self, column: str) -> Any:
        return self.params[self.columns.index(column)]


def _config() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


async def _insert(collection_class: type[Any], data: dict[str, Any]) -> _Insert:
    """save ``data`` as a new row through ``collection_class`` and return the recorded write.

    :param collection_class: the collection to write through
    :ptype collection_class: type[Any]
    :param data: the row
    :ptype data: dict[str, Any]
    :return: the one statement the save executed
    :rtype: _Insert
    """
    connection = _RecordingConnection()
    collection = collection_class(registry=CollectionRegistry(), config=_config())
    affected = await collection.save_to_store(data, conn=connection)
    assert affected == 1
    [(sql, params)] = connection.statements
    return _Insert(sql, params)


def _schedule_row(**overrides: Any) -> dict[str, Any]:
    """the minimum new schedule row, with columns replaced per test."""
    now = datetime.now(UTC)
    row: dict[str, Any] = {
        "conversation_id": _new_uuid(),
        "schedule_id": _new_uuid(),
        "user_id": _new_uuid(),
        "agent_id": _new_uuid(),
        "schedule_type": "daily_at",
        "date_created": now,
        "date_updated": now,
    }
    row.update(overrides)
    return row


def _subscription_row(**overrides: Any) -> dict[str, Any]:
    """the minimum new subscription row, with columns replaced per test."""
    now = datetime.now(UTC)
    row: dict[str, Any] = {
        "conversation_id": _new_uuid(),
        "subscription_id": _new_uuid(),
        "user_id": _new_uuid(),
        "agent_id": _new_uuid(),
        "secret_ciphertext": b"\x01",
        "date_created": now,
        "date_updated": now,
    }
    row.update(overrides)
    return row


class TestScheduleInsert:
    """A new schedule binds every column at its position and applies defaults."""

    async def test_full_row_round_trip(self) -> None:
        """Every column in the dict is bound at its declared position."""
        conv = _new_uuid()
        sched = _new_uuid()
        user = _new_uuid()
        agent = _new_uuid()
        skill = _new_uuid()
        now = datetime.now(UTC)
        data = {
            "conversation_id": conv,
            "schedule_id": sched,
            "user_id": user,
            "agent_id": agent,
            "skill_id": skill,
            "schedule_type": "cron",
            "schedule_config": {"expr": "*/5 * * * *"},
            "task_prompt": "Check status",
            "execution_mode": "inline",
            "status": "active",
            "next_fire_at": now,
            "last_fired_at": None,
            "name": "status-check",
            "missed_fire_policy": "coalesce",
            "context_from_schedule_id": None,
            "date_created": now,
            "date_updated": now,
        }
        insert = await _insert(WakeScheduleCollection, data)
        assert insert.table == "agent_wake_schedules"
        assert len(insert.params) == len(insert.columns) == len(insert.placeholders)
        for column, value in data.items():
            assert insert[column] == value, column

    async def test_defaults_applied_for_omitted_columns(self) -> None:
        """Missing ``status`` / ``missed_fire_policy`` / etc. get defaults."""
        insert = await _insert(WakeScheduleCollection, _schedule_row())
        assert insert["status"] == "active"
        assert insert["execution_mode"] == "spawn"
        assert insert["protected"] is False
        assert insert["missed_fire_policy"] == "coalesce"
        assert insert["schedule_config"] == {}
        assert insert["skill_id"] is None

    async def test_null_jsonb_coerced_to_empty_dict(self) -> None:
        """Explicit ``schedule_config=None`` writes ``{}`` to satisfy NOT NULL."""
        insert = await _insert(WakeScheduleCollection, _schedule_row(schedule_type="cron", schedule_config=None))
        assert insert["schedule_config"] == {}

    async def test_a_missing_protected_flag_binds_false(self) -> None:
        """a cached row with no flag (written before v007, or never read back) upserts as unprotected."""
        insert = await _insert(WakeScheduleCollection, _schedule_row(schedule_type="interval", protected=None))
        assert insert["protected"] is False


class TestFireInsert:
    """A new fire binds every column at its position."""

    async def test_full_row_round_trip(self) -> None:
        """Every fire column maps positionally."""
        conv = _new_uuid()
        fire_id = _new_uuid()
        schedule = _new_uuid()
        now = datetime.now(UTC)
        data = {
            "conversation_id": conv,
            "fire_id": fire_id,
            "schedule_id": schedule,
            "webhook_subscription_id": None,
            "scheduled_fire_at": now,
            "actual_fired_at": now,
            "status": "fired",
            "display_suppressed": False,
            "output_text": "ok",
            "latency_ms": 100,
            "error": None,
            "date_created": now,
        }
        insert = await _insert(WakeFireCollection, data)
        assert insert.table == "wake_fires"
        assert len(insert.params) == len(insert.columns) == len(insert.placeholders)
        for column, value in data.items():
            assert insert[column] == value, column

    async def test_display_suppressed_defaults_to_false(self) -> None:
        """Omitted ``display_suppressed`` defaults to ``False``."""
        data = {
            "conversation_id": _new_uuid(),
            "fire_id": _new_uuid(),
            "schedule_id": _new_uuid(),
            "actual_fired_at": datetime.now(UTC),
            "status": "fired",
        }
        insert = await _insert(WakeFireCollection, data)
        assert insert["display_suppressed"] is False


class TestSubscriptionInsert:
    """A new subscription binds every column at its position and applies defaults."""

    async def test_full_row_round_trip(self) -> None:
        """Every subscription column is bound at its declared position."""
        conv = _new_uuid()
        sub = _new_uuid()
        user = _new_uuid()
        agent = _new_uuid()
        default_skill = _new_uuid()
        now = datetime.now(UTC)
        data = {
            "conversation_id": conv,
            "subscription_id": sub,
            "user_id": user,
            "agent_id": agent,
            "default_skill_id": default_skill,
            "name": "github",
            "secret_ciphertext": b"\x00\xff",
            "allowed_source_pattern": None,
            "execution_mode": "inline",
            "task_prompt_template": "Investigate {{event}}",
            "verification_scheme": "generic_hmac_sha256",
            "status": "active",
            "rate_limit_per_minute": 60,
            "last_fired_at": None,
            "date_created": now,
            "date_updated": now,
        }
        insert = await _insert(WebhookSubscriptionCollection, data)
        assert insert.table == "webhook_subscriptions"
        assert len(insert.params) == len(insert.columns) == len(insert.placeholders)
        for column, value in data.items():
            assert insert[column] == value, column

    async def test_defaults_applied_for_omitted_columns(self) -> None:
        """Missing enums fall back to schema defaults."""
        insert = await _insert(WebhookSubscriptionCollection, _subscription_row())
        assert insert["execution_mode"] == "spawn"
        assert insert["verification_scheme"] == "generic_hmac_sha256"
        assert insert["status"] == "active"

    async def test_secret_ciphertext_coerced_to_bytes(self) -> None:
        """A bytearray input is normalised to ``bytes``."""
        insert = await _insert(
            WebhookSubscriptionCollection, _subscription_row(secret_ciphertext=bytearray(b"\xab\xcd"))
        )
        value = insert["secret_ciphertext"]
        assert isinstance(value, bytes)
        assert value == b"\xab\xcd"


class TestUpsertShape:
    """Each new-row write is a positional upsert on the table's composite pk."""

    async def test_placeholders_are_positional_in_column_order(self) -> None:
        """``$1..$n`` bind the columns in the order the INSERT lists them."""
        for collection_class, row in (
            (WakeScheduleCollection, _schedule_row()),
            (WebhookSubscriptionCollection, _subscription_row()),
        ):
            insert = await _insert(collection_class, row)
            assert insert.placeholders == tuple(f"${i + 1}" for i in range(len(insert.columns)))

    async def test_agent_wake_schedules_upsert_targets_composite_pk(self) -> None:
        """The schedule upsert conflict-targets the composite pk, and never rewrites what is fixed at insert."""
        insert = await _insert(WakeScheduleCollection, _schedule_row())
        assert insert.conflict == ("conversation_id", "schedule_id")
        expected = [
            c for c in insert.columns if c not in {"conversation_id", "schedule_id", "date_created", "protected"}
        ]
        assert insert.updates == tuple(f"{c} = EXCLUDED.{c}" for c in expected)

    async def test_wake_fires_upsert_targets_composite_pk(self) -> None:
        """The fire upsert conflict-targets the composite pk and fixes up only the finalize columns."""
        data = {
            "conversation_id": _new_uuid(),
            "fire_id": _new_uuid(),
            "schedule_id": _new_uuid(),
            "actual_fired_at": datetime.now(UTC),
            "status": "fired",
        }
        insert = await _insert(WakeFireCollection, data)
        assert insert.conflict == ("conversation_id", "fire_id")
        assert insert.updates == tuple(
            f"{c} = EXCLUDED.{c}" for c in ("status", "display_suppressed", "output_text", "latency_ms", "error")
        )

    async def test_webhook_subscriptions_upsert_targets_composite_pk(self) -> None:
        """The subscription upsert conflict-targets the composite pk."""
        insert = await _insert(WebhookSubscriptionCollection, _subscription_row())
        assert insert.conflict == ("conversation_id", "subscription_id")
        expected = [c for c in insert.columns if c not in {"conversation_id", "subscription_id", "date_created"}]
        assert insert.updates == tuple(f"{c} = EXCLUDED.{c}" for c in expected)


class TestCollectionClassAttributes:
    """Class attributes match the spec's documented contract."""

    def test_schedule_collection_partition_column(self) -> None:
        """``partition_column`` is ``conversation_id``."""
        assert WakeScheduleCollection.partition_column == "conversation_id"

    def test_schedule_collection_primary_key(self) -> None:
        """Composite PK is ``(conversation_id, schedule_id)``."""
        assert WakeScheduleCollection.primary_key_column == (
            "conversation_id",
            "schedule_id",
        )

    def test_fire_collection_partition_column(self) -> None:
        """``partition_column`` is ``conversation_id``."""
        assert WakeFireCollection.partition_column == "conversation_id"

    def test_fire_collection_primary_key(self) -> None:
        """Composite PK is ``(conversation_id, fire_id)``."""
        assert WakeFireCollection.primary_key_column == ("conversation_id", "fire_id")

    def test_subscription_collection_partition_column(self) -> None:
        """``partition_column`` is ``conversation_id``."""
        assert WebhookSubscriptionCollection.partition_column == "conversation_id"

    def test_subscription_collection_primary_key(self) -> None:
        """Composite PK is ``(conversation_id, subscription_id)``."""
        assert WebhookSubscriptionCollection.primary_key_column == (
            "conversation_id",
            "subscription_id",
        )
