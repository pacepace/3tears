"""a save of a wake row lands only on the row it was read from.

The schedule and webhook tools read a row (``find_for_agent``, or ``get``), change the fields the model asked for,
and save the WHOLE row back. Between the read and the save the row can change underneath them:
the tick claims a schedule and expires a one-shot, another replica pauses it, a webhook fire
stamps ``last_fired_at``, a rotation replaces the secret. A save that ignored the version it read
wrote every one of those changes away -- an expired one-shot re-armed, a paused schedule resumed,
a rotated secret restored.

Each test reads a row, changes it in L3 behind the reader's back (as another writer would), and
checks that the stale save is refused and that L3 keeps the other writer's row. The fake L3
interprets the statements the collection sends, so a save that is not fenced overwrites the row
exactly as the real upsert does.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import MetaData
from threetears.agent.skills.tables import agent_skills_table
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.exceptions import ConcurrentModificationError

from threetears.agent.wake.collections import WakeScheduleCollection, WebhookSubscriptionCollection
from threetears.agent.wake.tables import (
    agent_wake_schedules_table,
    wake_fires_table,
    webhook_subscriptions_table,
)
from threetears.agent.wake.tools.schedule_tools import load_wake_schedule_update_tool
from threetears.agent.wake.tools.webhook_tools import load_webhook_subscription_update_tool

_READ_AT = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
_CHANGED_AT = _READ_AT + timedelta(seconds=5)

_PK_COLUMN = {"agent_wake_schedules": "schedule_id", "webhook_subscriptions": "subscription_id"}


def _metadata() -> MetaData:
    metadata = MetaData()
    agent_skills_table(metadata)
    agent_wake_schedules_table(metadata)
    wake_fires_table(metadata)
    webhook_subscriptions_table(metadata)
    return metadata


# parity-with: asyncpg.Pool (fetchrow / execute -- the seam a by-pk read and a save drive)
class _Store:
    """an L3 that runs the collection's upsert and fenced UPDATE the way Postgres would.

    An ``INSERT ... ON CONFLICT DO UPDATE`` replaces the row whatever it holds. An ``UPDATE`` sets
    its ``SET`` columns only on a row matching every ``WHERE`` column, and answers the count.
    """

    def __init__(self) -> None:
        self.rows: dict[tuple[str, UUID, UUID], dict[str, Any]] = {}
        # another writer's change, applied once, right after the next read answers
        self.after_next_read: dict[str, Any] | None = None

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        table = "webhook_subscriptions" if "FROM webhook_subscriptions" in sql else "agent_wake_schedules"
        where = {col: args[int(pos) - 1] for col, pos in re.findall(r"(\w+) = \$(\d+)", sql.split("WHERE", 1)[1])}
        found: dict[str, Any] | None = None
        for (row_table, _, _), row in self.rows.items():
            if row_table == table and all(row.get(col) == value for col, value in where.items()):
                found = row
        answer = None if found is None else dict(found)
        if found is not None and self.after_next_read is not None:
            found.update(self.after_next_read)
            self.after_next_read = None
        return answer

    async def execute(self, sql: str, *args: Any) -> str:
        insert = re.match(r"INSERT INTO (\w+) \(([^)]*)\)", sql)
        if insert is not None:
            table = insert.group(1)
            columns = [c.strip() for c in insert.group(2).split(",")]
            row = dict(zip(columns, args, strict=True))
            self.rows[(table, row["conversation_id"], row[_PK_COLUMN[table]])] = row
            return "INSERT 0 1"
        update = re.match(r"UPDATE (\w+) SET (.*) WHERE (.*)", sql)
        assert update is not None, f"unexpected statement: {sql}"
        table, set_part, where_part = update.groups()
        where = {col: args[int(pos) - 1] for col, pos in re.findall(r"(\w+) = \$(\d+)", where_part)}
        row = self.rows.get((table, where["conversation_id"], where[_PK_COLUMN[table]]))
        if row is None or any(row.get(col) != value for col, value in where.items()):
            return "UPDATE 0"
        for col, pos in re.findall(r"(\w+) = \$(\d+)", set_part):
            row[col] = args[int(pos) - 1]
        return "UPDATE 1"


def _collections(store: _Store) -> tuple[WakeScheduleCollection, WebhookSubscriptionCollection]:
    l1 = SQLiteBackend(db_name=f"wake_fence_{uuid.uuid4().hex[:8]}")
    l1.initialize(_metadata())
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l3_pool=store)  # type: ignore[arg-type]
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return (
        WakeScheduleCollection(registry=registry, config=config),
        WebhookSubscriptionCollection(registry=registry, config=config),
    )


def _schedule_row(conv: UUID, sid: UUID, user_id: UUID, agent_id: UUID) -> dict[str, Any]:
    return {
        "conversation_id": conv,
        "schedule_id": sid,
        "user_id": user_id,
        "agent_id": agent_id,
        "skill_id": None,
        "schedule_type": "one_shot_at",
        "schedule_config": {"fire_at_iso": "2026-09-30T15:00:00+00:00"},
        "task_prompt": "check in",
        "execution_mode": "inline",
        "status": "active",
        "next_fire_at": datetime(2026, 9, 30, 15, 0, tzinfo=UTC),
        "last_fired_at": None,
        "name": "check-in",
        "missed_fire_policy": "coalesce",
        "context_from_schedule_id": None,
        "include_conversation_history": True,
        "date_created": _READ_AT,
        "date_updated": _READ_AT,
    }


def _subscription_row(conv: UUID, sub: UUID, user_id: UUID, agent_id: UUID) -> dict[str, Any]:
    return {
        "conversation_id": conv,
        "subscription_id": sub,
        "user_id": user_id,
        "agent_id": agent_id,
        "default_skill_id": None,
        "name": "deploys",
        "secret_ciphertext": b"old-secret",
        "allowed_source_pattern": None,
        "execution_mode": "inline",
        "task_prompt_template": "deploy: {{event.ref}}",
        "verification_scheme": "generic_hmac_sha256",
        "status": "active",
        "rate_limit_per_minute": None,
        "last_fired_at": None,
        "date_created": _READ_AT,
        "date_updated": _READ_AT,
    }


# parity-with: threetears.agent.wake.tools.schedule_tools.WakeRegistryClient
class _Registry:
    """permits nothing and names nothing; the edits under test attach no skill."""

    async def acl_permits_skill(self, *, user_id: UUID, agent_id: UUID, skill_id: UUID) -> bool:
        del user_id, agent_id, skill_id
        return False

    async def skill_name_for_id(self, *, user_id: UUID, agent_id: UUID, skill_id: UUID) -> str | None:
        del user_id, agent_id, skill_id
        return None


class TestAScheduleSaveIsFencedOnTheRowItRead:
    @pytest.mark.asyncio
    async def test_a_save_after_the_tick_expired_the_row_is_refused(self) -> None:
        store = _Store()
        schedules, _ = _collections(store)
        conv, sid = uuid.uuid4(), uuid.uuid4()
        key = ("agent_wake_schedules", conv, sid)
        store.rows[key] = _schedule_row(conv, sid, uuid.uuid4(), uuid.uuid4())

        entity = await schedules.get((conv, sid))
        assert entity is not None and entity.status == "active"

        # the tick claims the one-shot and expires it, on another replica.
        store.rows[key].update(status="expired", next_fire_at=None, last_fired_at=_CHANGED_AT, date_updated=_CHANGED_AT)

        entity.name = "renamed"
        with pytest.raises(ConcurrentModificationError):
            await schedules.save_entity(entity)

        assert store.rows[key]["status"] == "expired", "the stale save re-armed the expired one-shot"
        assert store.rows[key]["next_fire_at"] is None
        assert store.rows[key]["name"] == "check-in"

    @pytest.mark.asyncio
    async def test_a_save_of_the_row_as_read_lands(self) -> None:
        """the positive control: an unchanged row takes the save."""
        store = _Store()
        schedules, _ = _collections(store)
        conv, sid = uuid.uuid4(), uuid.uuid4()
        key = ("agent_wake_schedules", conv, sid)
        store.rows[key] = _schedule_row(conv, sid, uuid.uuid4(), uuid.uuid4())

        entity = await schedules.get((conv, sid))
        assert entity is not None
        entity.name = "renamed"
        await schedules.save_entity(entity)

        assert store.rows[key]["name"] == "renamed"
        assert store.rows[key]["date_updated"] > _READ_AT

    @pytest.mark.asyncio
    async def test_a_save_of_a_row_deleted_since_the_read_does_not_recreate_it(self) -> None:
        store = _Store()
        schedules, _ = _collections(store)
        conv, sid = uuid.uuid4(), uuid.uuid4()
        key = ("agent_wake_schedules", conv, sid)
        store.rows[key] = _schedule_row(conv, sid, uuid.uuid4(), uuid.uuid4())

        entity = await schedules.get((conv, sid))
        assert entity is not None
        del store.rows[key]

        entity.name = "renamed"
        with pytest.raises(ConcurrentModificationError):
            await schedules.save_entity(entity)
        assert key not in store.rows

    @pytest.mark.asyncio
    async def test_the_update_tool_tells_the_model_the_schedule_changed(self) -> None:
        store = _Store()
        schedules, _ = _collections(store)
        conv, sid, user_id, agent_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        key = ("agent_wake_schedules", conv, sid)
        store.rows[key] = _schedule_row(conv, sid, user_id, agent_id)
        # another replica pauses the schedule between the tool's read and its save.
        store.after_next_read = {"status": "paused", "next_fire_at": None, "date_updated": _CHANGED_AT}

        tool = load_wake_schedule_update_tool(
            user_id=user_id,
            agent_id=agent_id,
            schedules_collection=schedules,
            registry=_Registry(),
        )[0]
        result = await tool.ainvoke({"schedule_id": str(sid), "name": "renamed"})

        assert result.startswith("[TOOL ERROR] wake_schedule_update: the schedule changed"), result
        assert "wake_schedule_list" in result
        assert store.rows[key]["status"] == "paused", "the stale edit resumed the paused schedule"
        assert store.rows[key]["name"] == "check-in"


class TestASubscriptionSaveIsFencedOnTheRowItRead:
    @pytest.mark.asyncio
    async def test_a_save_after_a_rotation_does_not_restore_the_old_secret(self) -> None:
        store = _Store()
        _, subscriptions = _collections(store)
        conv, sub = uuid.uuid4(), uuid.uuid4()
        key = ("webhook_subscriptions", conv, sub)
        store.rows[key] = _subscription_row(conv, sub, uuid.uuid4(), uuid.uuid4())

        entity = await subscriptions.get((conv, sub))
        assert entity is not None
        store.rows[key].update(secret_ciphertext=b"new-secret", date_updated=_CHANGED_AT)

        entity.name = "renamed"
        with pytest.raises(ConcurrentModificationError):
            await subscriptions.save_entity(entity)
        assert store.rows[key]["secret_ciphertext"] == b"new-secret", "the stale save restored the rotated secret"

    @pytest.mark.asyncio
    async def test_the_update_tool_tells_the_model_the_subscription_changed(self) -> None:
        store = _Store()
        _, subscriptions = _collections(store)
        conv, sub, user_id, agent_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        key = ("webhook_subscriptions", conv, sub)
        store.rows[key] = _subscription_row(conv, sub, user_id, agent_id)
        # a webhook fire on another replica stamps last_fired_at between the tool's read and its save.
        store.after_next_read = {"last_fired_at": _CHANGED_AT, "date_updated": _CHANGED_AT}

        tool = load_webhook_subscription_update_tool(
            user_id=user_id,
            agent_id=agent_id,
            subscriptions_collection=subscriptions,
            registry=_Registry(),
        )[0]
        result = await tool.ainvoke({"subscription_id": str(sub), "name": "renamed"})

        assert result.startswith("[TOOL ERROR] webhook_subscription_update: the subscription changed"), result
        assert "webhook_subscription_list" in result
        assert store.rows[key]["last_fired_at"] == _CHANGED_AT
        assert store.rows[key]["name"] == "deploys"
