"""Unit tests for pure-logic helpers on the agent-wake collections.

The Collection classes are wired to a real Postgres pool in
integration tests; the unit suite exercises:

- ``_schedule_insert_params`` / ``_fire_insert_params`` /
  ``_subscription_insert_params`` projections preserve declared column
  order so positional asyncpg parameters stay aligned with the SQL
  placeholders.
- ``_build_upsert_sql`` emits a syntactically valid upsert with the
  expected conflict target.
- Class attributes (``primary_key_column``, ``partition_column``)
  declare the documented contract.
- Defaults applied by the value-for-column helpers mirror the schema's
  DEFAULT semantics.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from uuid_utils import uuid7

from threetears.agent.wake.collections import (
    _AGENT_WAKE_SCHEDULES_UPSERT_SQL,
    _FIRE_INSERT_COLUMNS,
    _SCHEDULE_INSERT_COLUMNS,
    _SUBSCRIPTION_INSERT_COLUMNS,
    _WAKE_FIRES_UPSERT_SQL,
    _WEBHOOK_SUBSCRIPTIONS_UPSERT_SQL,
    WakeFireCollection,
    WakeScheduleCollection,
    WebhookSubscriptionCollection,
    _build_upsert_sql,
    _fire_insert_params,
    _schedule_insert_params,
    _subscription_insert_params,
)


def _new_uuid() -> UUID:
    """Return a fresh UUIDv7 cast to stdlib ``UUID``."""
    return UUID(str(uuid7()))


class TestScheduleInsertParams:
    """``_schedule_insert_params`` preserves column order + applies defaults."""

    def test_full_row_round_trip(self) -> None:
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
        params = _schedule_insert_params(data)
        assert len(params) == len(_SCHEDULE_INSERT_COLUMNS)
        conv_idx = _SCHEDULE_INSERT_COLUMNS.index("conversation_id")
        skill_idx = _SCHEDULE_INSERT_COLUMNS.index("skill_id")
        policy_idx = _SCHEDULE_INSERT_COLUMNS.index("missed_fire_policy")
        config_idx = _SCHEDULE_INSERT_COLUMNS.index("schedule_config")
        assert params[conv_idx] == conv
        assert params[skill_idx] == skill
        assert params[policy_idx] == "coalesce"
        assert params[config_idx] == {"expr": "*/5 * * * *"}

    def test_defaults_applied_for_omitted_columns(self) -> None:
        """Missing ``status`` / ``missed_fire_policy`` / etc. get defaults."""
        data = {
            "conversation_id": _new_uuid(),
            "schedule_id": _new_uuid(),
            "user_id": _new_uuid(),
            "agent_id": _new_uuid(),
            "schedule_type": "daily_at",
            "date_created": datetime.now(UTC),
            "date_updated": datetime.now(UTC),
        }
        params = _schedule_insert_params(data)
        assert params[_SCHEDULE_INSERT_COLUMNS.index("status")] == "active"
        assert params[_SCHEDULE_INSERT_COLUMNS.index("execution_mode")] == "inline"
        assert params[_SCHEDULE_INSERT_COLUMNS.index("missed_fire_policy")] == "coalesce"
        assert params[_SCHEDULE_INSERT_COLUMNS.index("schedule_config")] == {}
        assert params[_SCHEDULE_INSERT_COLUMNS.index("skill_id")] is None

    def test_null_jsonb_coerced_to_empty_dict(self) -> None:
        """Explicit ``schedule_config=None`` writes ``{}`` to satisfy NOT NULL."""
        data = {
            "conversation_id": _new_uuid(),
            "schedule_id": _new_uuid(),
            "user_id": _new_uuid(),
            "agent_id": _new_uuid(),
            "schedule_type": "cron",
            "schedule_config": None,
            "date_created": datetime.now(UTC),
            "date_updated": datetime.now(UTC),
        }
        params = _schedule_insert_params(data)
        assert params[_SCHEDULE_INSERT_COLUMNS.index("schedule_config")] == {}


class TestFireInsertParams:
    """``_fire_insert_params`` preserves column order."""

    def test_full_row_round_trip(self) -> None:
        """All 12 fire columns map positionally."""
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
        params = _fire_insert_params(data)
        assert len(params) == len(_FIRE_INSERT_COLUMNS)
        assert params[_FIRE_INSERT_COLUMNS.index("conversation_id")] == conv
        assert params[_FIRE_INSERT_COLUMNS.index("status")] == "fired"
        assert params[_FIRE_INSERT_COLUMNS.index("display_suppressed")] is False

    def test_display_suppressed_defaults_to_false(self) -> None:
        """Omitted ``display_suppressed`` defaults to ``False``."""
        data = {
            "conversation_id": _new_uuid(),
            "fire_id": _new_uuid(),
            "schedule_id": _new_uuid(),
            "actual_fired_at": datetime.now(UTC),
            "status": "fired",
        }
        params = _fire_insert_params(data)
        assert params[_FIRE_INSERT_COLUMNS.index("display_suppressed")] is False


class TestSubscriptionInsertParams:
    """``_subscription_insert_params`` preserves column order + defaults."""

    def test_full_row_round_trip(self) -> None:
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
        params = _subscription_insert_params(data)
        assert len(params) == len(_SUBSCRIPTION_INSERT_COLUMNS)
        assert params[_SUBSCRIPTION_INSERT_COLUMNS.index("conversation_id")] == conv
        assert params[_SUBSCRIPTION_INSERT_COLUMNS.index("default_skill_id")] == default_skill
        assert params[_SUBSCRIPTION_INSERT_COLUMNS.index("secret_ciphertext")] == b"\x00\xff"
        assert params[_SUBSCRIPTION_INSERT_COLUMNS.index("verification_scheme")] == "generic_hmac_sha256"

    def test_defaults_applied_for_omitted_columns(self) -> None:
        """Missing enums fall back to schema defaults."""
        data = {
            "conversation_id": _new_uuid(),
            "subscription_id": _new_uuid(),
            "user_id": _new_uuid(),
            "agent_id": _new_uuid(),
            "secret_ciphertext": b"\x01",
            "date_created": datetime.now(UTC),
            "date_updated": datetime.now(UTC),
        }
        params = _subscription_insert_params(data)
        assert params[_SUBSCRIPTION_INSERT_COLUMNS.index("execution_mode")] == "inline"
        assert params[_SUBSCRIPTION_INSERT_COLUMNS.index("verification_scheme")] == "generic_hmac_sha256"
        assert params[_SUBSCRIPTION_INSERT_COLUMNS.index("status")] == "active"

    def test_secret_ciphertext_coerced_to_bytes(self) -> None:
        """A bytearray input is normalised to ``bytes``."""
        data = {
            "conversation_id": _new_uuid(),
            "subscription_id": _new_uuid(),
            "user_id": _new_uuid(),
            "agent_id": _new_uuid(),
            "secret_ciphertext": bytearray(b"\xab\xcd"),
            "date_created": datetime.now(UTC),
            "date_updated": datetime.now(UTC),
        }
        params = _subscription_insert_params(data)
        value = params[_SUBSCRIPTION_INSERT_COLUMNS.index("secret_ciphertext")]
        assert isinstance(value, bytes)
        assert value == b"\xab\xcd"


class TestBuildUpsertSql:
    """``_build_upsert_sql`` emits SQL with the expected shape."""

    def test_upsert_includes_conflict_clause(self) -> None:
        """Generated SQL carries ``ON CONFLICT (pk_cols) DO UPDATE SET ...``."""
        sql = _build_upsert_sql(
            "demo",
            ("a", "b", "c"),
            ("b", "c"),
            ("a",),
        )
        assert sql.startswith("INSERT INTO demo (a, b, c) VALUES ($1, $2, $3) ")
        assert "ON CONFLICT (a) DO UPDATE SET" in sql
        assert "b = EXCLUDED.b" in sql
        assert "c = EXCLUDED.c" in sql

    def test_agent_wake_schedules_upsert_targets_composite_pk(self) -> None:
        """The module-level schedule upsert conflict-targets the composite pk."""
        assert "ON CONFLICT (conversation_id, schedule_id)" in _AGENT_WAKE_SCHEDULES_UPSERT_SQL

    def test_wake_fires_upsert_targets_composite_pk(self) -> None:
        """The module-level fire upsert conflict-targets the composite pk."""
        assert "ON CONFLICT (conversation_id, fire_id)" in _WAKE_FIRES_UPSERT_SQL

    def test_webhook_subscriptions_upsert_targets_composite_pk(self) -> None:
        """The module-level subscription upsert conflict-targets the composite pk."""
        assert "ON CONFLICT (conversation_id, subscription_id)" in _WEBHOOK_SUBSCRIPTIONS_UPSERT_SQL


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
