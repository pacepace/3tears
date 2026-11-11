"""every wake statement converted to a keyed helper gives the rows, and leaves the table, the hand-written one did.

Each case freezes the statement the collection sent before the conversion (``old_sql``, its parameters) and calls the
converted collection method on the same connection, through
:func:`~threetears.core.testing.same_results.assert_same_results`: empty, one row, many, NULLs, ties in the sort,
a tenant boundary, a page boundary and one past it. A write is compared by what it returns and by the table it
leaves (``state_sql``); ``date_updated`` is compared as "stamped by this statement" (``= now()``), since each side
runs in its own transaction and ``now()`` is that transaction's start.

Runs on PostgreSQL and YugabyteDB (the session's ``db_container``), a schema of its own.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest

from threetears.agent.skills.migrations import register as register_skills
from threetears.agent.wake.collections import (
    WakeFireCollection,
    WakeScheduleCollection,
    WebhookSubscriptionCollection,
)
from threetears.agent.wake.migrations import register as register_wake
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.core.testing.same_results import Seeded, assert_same_results
from threetears.scheduled_jobs.collections import REAPED_DISPATCH_ERROR

from .conftest import AsyncpgStore

pytestmark = pytest.mark.integration

_NS = uuid.UUID("6b1d6c55-3a8e-4f5e-9a59-0f3f9b2c7e11")
_CONVS = (uuid.uuid5(_NS, "conv-0"), uuid.uuid5(_NS, "conv-1"))
_AGENTS = (uuid.uuid5(_NS, "agent-0"), uuid.uuid5(_NS, "agent-1"))
_USER = uuid.uuid5(_NS, "user")
_T0 = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
_LATER = _T0 + timedelta(days=2)
_STATUSES = ("active", "active", "paused", "expired")
#: the page a listing case sits its boundary at; a multiple of the ties' run of three, so a LIMIT never cuts a tie
_PAGE = 6


def _id(kind: str, index: int) -> uuid.UUID:
    return uuid.uuid5(_NS, f"{kind}-{index}")


_SCHEDULE_COLUMNS = (
    "conversation_id",
    "schedule_id",
    "user_id",
    "agent_id",
    "skill_id",
    "schedule_type",
    "schedule_config",
    "task_prompt",
    "execution_mode",
    "status",
    "next_fire_at",
    "last_fired_at",
    "name",
    "missed_fire_policy",
    "context_from_schedule_id",
    "include_conversation_history",
    "protected",
    "date_created",
    "date_updated",
)
_SCHEDULE_SELECT = ", ".join(_SCHEDULE_COLUMNS)
_SCHEDULE_UPDATE = tuple(
    c for c in _SCHEDULE_COLUMNS if c not in {"conversation_id", "schedule_id", "date_created", "protected"}
)
_FIRE_COLUMNS = (
    "conversation_id",
    "fire_id",
    "schedule_id",
    "webhook_subscription_id",
    "scheduled_fire_at",
    "actual_fired_at",
    "status",
    "display_suppressed",
    "output_text",
    "latency_ms",
    "error",
    "started_conversation_id",
    "date_created",
    "fire_source",
    "requested_by",
)
_FIRE_SELECT = ", ".join(_FIRE_COLUMNS)
_SUBSCRIPTION_COLUMNS = (
    "conversation_id",
    "subscription_id",
    "user_id",
    "agent_id",
    "default_skill_id",
    "name",
    "secret_ciphertext",
    "allowed_source_pattern",
    "execution_mode",
    "task_prompt_template",
    "verification_scheme",
    "status",
    "rate_limit_per_minute",
    "last_fired_at",
    "date_created",
    "date_updated",
)
_SUBSCRIPTION_UPDATE = tuple(
    c for c in _SUBSCRIPTION_COLUMNS if c not in {"conversation_id", "subscription_id", "date_created"}
)


def _state(table: str, columns: Sequence[str]) -> str:
    """the read of a table after a write: every column, ``date_updated`` as whether this statement stamped it."""
    kept = [c for c in columns if c != "date_updated"]
    stamped = ", date_updated = now() AS stamped_now" if "date_updated" in columns else ""
    return f"SELECT {', '.join(kept)}{stamped} FROM {table}"  # noqa: S608 - trusted test identifiers


def _fenced_update_sql(table: str, update_cols: Sequence[str], pk_cols: Sequence[str]) -> str:
    """the statement the removed ``_build_fenced_update_sql`` rendered, frozen."""
    set_clause = ", ".join(f"{c} = ${len(pk_cols) + i + 1}" for i, c in enumerate(update_cols))
    where_clause = " AND ".join(f"{c} = ${i + 1}" for i, c in enumerate(pk_cols))
    fence = len(pk_cols) + len(update_cols) + 1
    return f"UPDATE {table} SET {set_clause} WHERE {where_clause} AND date_updated = ${fence}"


def _schedule(seed: Seeded) -> Mapping[str, Any]:
    null = seed.null
    return {
        "conversation_id": _CONVS[seed.tenant],
        "schedule_id": _id("schedule", seed.index),
        "user_id": _USER,
        "agent_id": _AGENTS[seed.tenant],
        "skill_id": None,
        "schedule_type": "every_n_hours",
        "schedule_config": json.dumps({"n": seed.index}),
        "task_prompt": None if null else f"prompt {seed.index}",
        "execution_mode": "inline",
        "status": _STATUSES[seed.index % 4],
        "next_fire_at": None if null else _T0 + timedelta(minutes=seed.tie),
        "last_fired_at": None if null else _T0 - timedelta(minutes=seed.index),
        "name": None if null else f"wake {seed.index}",
        "missed_fire_policy": "coalesce",
        "context_from_schedule_id": None,
        "include_conversation_history": True,
        "protected": False,
        "date_created": _T0 - timedelta(hours=seed.tie),
        "date_updated": _T0 - timedelta(hours=seed.tie),
    }


# each tenant's parent schedule and subscription, seeded once beside the fires (their foreign keys)
_PARENT_SCHEDULES = (_id("parent-schedule", 0), _id("parent-schedule", 1))
_PARENT_SUBSCRIPTIONS = (_id("parent-subscription", 0), _id("parent-subscription", 1))


def _fire(seed: Seeded) -> Mapping[str, Any]:
    null = seed.null
    from_schedule = seed.index % 2 == 0
    return {
        "conversation_id": _CONVS[seed.tenant],
        "fire_id": _id("fire", seed.index),
        "schedule_id": _PARENT_SCHEDULES[seed.tenant] if from_schedule else None,
        "webhook_subscription_id": None if from_schedule else _PARENT_SUBSCRIPTIONS[seed.tenant],
        "scheduled_fire_at": None if null else _T0 + timedelta(minutes=seed.tie),
        "actual_fired_at": _T0 + timedelta(minutes=seed.tie),
        "status": "dispatching" if seed.index % 3 == 0 else "fired",
        "display_suppressed": False,
        "output_text": None if null else f"out {seed.index}",
        "latency_ms": None if null else seed.index,
        "error": None,
        "started_conversation_id": None,
        "date_created": _T0 + timedelta(minutes=seed.tie),
        "fire_source": "scheduled_tick" if from_schedule else "webhook",
        "requested_by": None,
    }


def _subscription(seed: Seeded) -> Mapping[str, Any]:
    null = seed.null
    return {
        "conversation_id": _CONVS[seed.tenant],
        "subscription_id": _id("subscription", seed.index),
        "user_id": _USER,
        "agent_id": _AGENTS[seed.tenant],
        "default_skill_id": None,
        "name": f"hook {seed.index}",
        "secret_ciphertext": f"secret {seed.index}".encode(),
        "allowed_source_pattern": None if null else "10.0.0.0/8",
        "execution_mode": "inline",
        "task_prompt_template": None if null else "event: {{event}}",
        "verification_scheme": "generic_hmac_sha256",
        "status": "active" if seed.index % 3 else "paused",
        "rate_limit_per_minute": None if null else 10,
        "last_fired_at": None if null else _T0 - timedelta(minutes=seed.index),
        "date_created": _T0 - timedelta(hours=seed.tie),
        "date_updated": _T0 - timedelta(hours=seed.tie),
    }


_REASONS = ("rate_limited", "bad_signature", "paused")
_REFUSAL_COLUMNS = ("reason", "status_code", "window_start", "refusals", "last_refused_at", "detail")


def _refusal(seed: Seeded) -> Mapping[str, Any]:
    return {
        "conversation_id": _CONVS[seed.tenant],
        "subscription_id": _PARENT_SUBSCRIPTIONS[seed.tenant],
        "reason": _REASONS[seed.index % 3],
        "window_start": _T0 + timedelta(minutes=seed.index),
        "status_code": 429,
        "refusals": seed.index + 1,
        "last_refused_at": _T0 + timedelta(minutes=seed.tie, seconds=30),
        "detail": None if seed.null else f"refused {seed.index}",
    }


# the refusal record_refusal writes in passing (its INSERT, not a converted statement): left out of the table compared
_REFUSED_AT = _T0 + timedelta(minutes=40)
_REFUSALS_BUT_THE_NEW_ONE = (
    f"SELECT {', '.join(_REFUSAL_COLUMNS)}, subscription_id FROM webhook_refusals "  # noqa: S608 - test identifiers
    f"WHERE NOT (reason = 'rate_limited' AND window_start = '{_REFUSED_AT.isoformat()}'::timestamptz)"
)


def _config() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


def _registry(conn: Any) -> CollectionRegistry:
    registry = CollectionRegistry()
    registry.configure(l3_pool=conn)
    return registry


def _schedules(conn: Any) -> WakeScheduleCollection:
    return WakeScheduleCollection(registry=_registry(conn), config=_config())


def _fires(conn: Any) -> WakeFireCollection:
    return WakeFireCollection(registry=_registry(conn), config=_config())


def _subscriptions(conn: Any) -> WebhookSubscriptionCollection:
    return WebhookSubscriptionCollection(registry=_registry(conn), config=_config())


def _rows(entities: Sequence[Any]) -> list[dict[str, Any]]:
    return [entity.to_dict() for entity in entities]


def _wrote(call: Callable[[Any], Awaitable[Any]]) -> Callable[[Any], Awaitable[list[dict[str, Any]]]]:
    """a write that returns nothing, as the helper side of a write comparison."""

    async def helper(conn: Any) -> list[dict[str, Any]]:
        await call(conn)
        return []

    return helper


# the row each by-key case names: schedule 3 of tenant 0 in every case that seeds it
_SID = _id("schedule", 3)
_FID = _id("fire", 3)
_SUB = _id("subscription", 3)
# the date_updated schedule 3 / subscription 3 carry when seeded without ties (the fence a save read)
_FENCE = _T0 - timedelta(hours=3)


def _saved_schedule() -> dict[str, Any]:
    row = dict(_schedule(Seeded(index=3, case="saved", tenant=0, null=False, tie=3)))
    row.update(
        name="renamed", status="paused", next_fire_at=None, schedule_config=json.dumps({"n": 99}), date_updated=_LATER
    )
    return row


def _saved_subscription() -> dict[str, Any]:
    row = dict(_subscription(Seeded(index=3, case="saved", tenant=0, null=False, tie=3)))
    row.update(name="renamed", status="paused", secret_ciphertext=b"rotated", date_updated=_LATER)
    return row


def _schedule_params(row: Mapping[str, Any]) -> list[Any]:
    # schedule_config as its JSON text on both sides: this connection has no jsonb codec (a pool's init sets one)
    return [row[c] for c in ("conversation_id", "schedule_id", *_SCHEDULE_UPDATE)]


def _subscription_params(row: Mapping[str, Any]) -> list[Any]:
    return [row[c] for c in ("conversation_id", "subscription_id", *_SUBSCRIPTION_UPDATE)]


@dataclass(frozen=True)
class _Site:
    """one converted statement: the old text and its parameters, and the converted call."""

    table: str
    row: Callable[[Seeded], Mapping[str, Any]]
    old_sql: str
    params: Sequence[Any]
    helper: Callable[[Any], Awaitable[Sequence[Mapping[str, Any]]]]
    columns: Sequence[str] | None = None
    order_by: Sequence[str] | None = None
    state_sql: str | None = None
    #: the fires' parent schedules and subscriptions are seeded first
    parents: bool = False


async def _one(found: Awaitable[Any]) -> list[dict[str, Any]]:
    row = await found
    return [] if row is None else [dict(row)]


_SITES: dict[str, _Site] = {
    "schedule fetch_from_store": _Site(
        "agent_wake_schedules",
        _schedule,
        f"SELECT {_SCHEDULE_SELECT} FROM agent_wake_schedules WHERE conversation_id = $1 AND schedule_id = $2",
        [_CONVS[0], _SID],
        lambda c: _one(_schedules(c).fetch_from_store((_CONVS[0], _SID))),
    ),
    "schedule find_for_agent": _Site(
        "agent_wake_schedules",
        _schedule,
        f"SELECT {_SCHEDULE_SELECT} FROM agent_wake_schedules WHERE agent_id = $1 AND schedule_id = $2",
        [_AGENTS[0], _SID],
        lambda c: _one(_found_for_agent(c)),
        columns=_SCHEDULE_COLUMNS,
    ),
    "schedule list_due_for_tick": _Site(
        "agent_wake_schedules",
        _schedule,
        f"SELECT {_SCHEDULE_SELECT} FROM agent_wake_schedules "
        "WHERE status = 'active' AND next_fire_at IS NOT NULL AND next_fire_at <= $1 "
        "ORDER BY next_fire_at ASC LIMIT $2",
        [_T0 + timedelta(minutes=8), _PAGE],
        lambda c: _listed(_schedules(c).list_due_for_tick(_T0 + timedelta(minutes=8), limit=_PAGE)),
        columns=_SCHEDULE_COLUMNS,
        order_by=["next_fire_at"],
    ),
    "schedule list_active_for_conversation": _Site(
        "agent_wake_schedules",
        _schedule,
        f"SELECT {_SCHEDULE_SELECT} FROM agent_wake_schedules "
        "WHERE conversation_id = $1 AND status = 'active' ORDER BY next_fire_at ASC NULLS LAST",
        [_CONVS[0]],
        lambda c: _listed(_schedules(c).list_active_for_conversation(_CONVS[0])),
        columns=_SCHEDULE_COLUMNS,
        order_by=["next_fire_at"],
    ),
    "schedule list_for_conversation": _Site(
        "agent_wake_schedules",
        _schedule,
        f"SELECT {_SCHEDULE_SELECT} FROM agent_wake_schedules WHERE conversation_id = $1 ORDER BY date_created DESC",
        [_CONVS[0]],
        lambda c: _listed(_schedules(c).list_for_conversation(_CONVS[0])),
        columns=_SCHEDULE_COLUMNS,
        order_by=["date_created"],
    ),
    "schedule count_active_for_conversation": _Site(
        "agent_wake_schedules",
        _schedule,
        "SELECT COUNT(*) FROM agent_wake_schedules WHERE conversation_id = $1 AND status = 'active'",
        [_CONVS[0]],
        lambda c: _counted(_schedules(c).count_active_for_conversation(_CONVS[0])),
    ),
    "schedule list_for_agent": _Site(
        "agent_wake_schedules",
        _schedule,
        f"SELECT {_SCHEDULE_SELECT} FROM agent_wake_schedules "
        "WHERE agent_id = $1 AND status = ANY($2::text[]) "
        "ORDER BY next_fire_at ASC NULLS LAST, date_created ASC",
        [_AGENTS[0], ["active", "paused"]],
        lambda c: _listed(_schedules(c).list_for_agent(_AGENTS[0])),
        columns=_SCHEDULE_COLUMNS,
        order_by=["next_fire_at", "date_created"],
    ),
    "schedule update_next_fire_at": _Site(
        "agent_wake_schedules",
        _schedule,
        "UPDATE agent_wake_schedules SET next_fire_at = $3, date_updated = now() "
        "WHERE conversation_id = $1 AND schedule_id = $2",
        [_CONVS[0], _SID, _LATER],
        _wrote(lambda c: _schedules(c).update_next_fire_at(_CONVS[0], _SID, next_fire_at=_LATER)),
        state_sql=_state("agent_wake_schedules", _SCHEDULE_COLUMNS),
    ),
    "schedule update_next_fire_at with last_fired_at": _Site(
        "agent_wake_schedules",
        _schedule,
        "UPDATE agent_wake_schedules SET next_fire_at = $3, last_fired_at = $4, date_updated = now() "
        "WHERE conversation_id = $1 AND schedule_id = $2",
        [_CONVS[0], _SID, _LATER, _T0],
        _wrote(lambda c: _schedules(c).update_next_fire_at(_CONVS[0], _SID, next_fire_at=_LATER, last_fired_at=_T0)),
        state_sql=_state("agent_wake_schedules", _SCHEDULE_COLUMNS),
    ),
    "schedule pause": _Site(
        "agent_wake_schedules",
        _schedule,
        "UPDATE agent_wake_schedules SET status = 'paused', next_fire_at = NULL, date_updated = now() "
        "WHERE conversation_id = $1 AND schedule_id = $2 AND status = 'active'",
        [_CONVS[0], _SID],
        _wrote(lambda c: _schedules(c).pause(_CONVS[0], _SID)),
        state_sql=_state("agent_wake_schedules", _SCHEDULE_COLUMNS),
    ),
    "schedule resume": _Site(
        "agent_wake_schedules",
        _schedule,
        "UPDATE agent_wake_schedules SET status = 'active', next_fire_at = $3, date_updated = now() "
        "WHERE conversation_id = $1 AND schedule_id = $2 AND status != 'expired'",
        [_CONVS[0], _id("schedule", 2), _LATER],
        _wrote(lambda c: _schedules(c).resume(_CONVS[0], _id("schedule", 2), next_fire_at=_LATER)),
        state_sql=_state("agent_wake_schedules", _SCHEDULE_COLUMNS),
    ),
    "schedule resume of an expired one": _Site(
        "agent_wake_schedules",
        _schedule,
        "UPDATE agent_wake_schedules SET status = 'active', next_fire_at = $3, date_updated = now() "
        "WHERE conversation_id = $1 AND schedule_id = $2 AND status != 'expired'",
        [_CONVS[0], _SID, _LATER],
        _wrote(lambda c: _schedules(c).resume(_CONVS[0], _SID, next_fire_at=_LATER)),
        state_sql=_state("agent_wake_schedules", _SCHEDULE_COLUMNS),
    ),
    "schedule claim_and_reschedule": _Site(
        "agent_wake_schedules",
        _schedule,
        "UPDATE agent_wake_schedules "
        "SET next_fire_at = $1, last_fired_at = $2, date_updated = $2, status = $3 "
        "WHERE conversation_id = $4 AND schedule_id = $5 AND next_fire_at = $6 "
        "RETURNING schedule_id",
        [_LATER, _T0, "active", _CONVS[0], _id("schedule", 1), _T0 + timedelta(minutes=1)],
        lambda c: _claimed(c, _id("schedule", 1), _T0 + timedelta(minutes=1)),
        state_sql=_state("agent_wake_schedules", _SCHEDULE_COLUMNS),
    ),
    "schedule mark_expired": _Site(
        "agent_wake_schedules",
        _schedule,
        "UPDATE agent_wake_schedules SET status = 'expired', next_fire_at = NULL, date_updated = now() "
        "WHERE conversation_id = $1 AND schedule_id = $2",
        [_CONVS[0], _id("schedule", 1)],
        _wrote(lambda c: _schedules(c).mark_expired(_CONVS[0], _id("schedule", 1))),
        state_sql=_state("agent_wake_schedules", _SCHEDULE_COLUMNS),
    ),
    "schedule delete_from_store": _Site(
        "agent_wake_schedules",
        _schedule,
        "DELETE FROM agent_wake_schedules WHERE conversation_id = $1 AND schedule_id = $2",
        [_CONVS[0], _SID],
        _wrote(lambda c: _schedules(c).delete_from_store((_CONVS[0], _SID))),
        state_sql=_state("agent_wake_schedules", _SCHEDULE_COLUMNS),
    ),
    "schedule fenced save": _Site(
        "agent_wake_schedules",
        _schedule,
        _fenced_update_sql("agent_wake_schedules", _SCHEDULE_UPDATE, ("conversation_id", "schedule_id")),
        [*_schedule_params(_saved_schedule()), _FENCE],
        _wrote(lambda c: _schedules(c).save_to_store(_saved_schedule(), _FENCE)),
        state_sql=_state("agent_wake_schedules", _SCHEDULE_COLUMNS),
    ),
    "fire fetch_from_store": _Site(
        "wake_fires",
        _fire,
        f"SELECT {', '.join(_FIRE_COLUMNS)} FROM wake_fires WHERE conversation_id = $1 AND fire_id = $2",
        [_CONVS[0], _FID],
        lambda c: _one(_fires(c).fetch_from_store((_CONVS[0], _FID))),
        parents=True,
    ),
    "fire delete_from_store": _Site(
        "wake_fires",
        _fire,
        "DELETE FROM wake_fires WHERE conversation_id = $1 AND fire_id = $2",
        [_CONVS[0], _FID],
        _wrote(lambda c: _fires(c).delete_from_store((_CONVS[0], _FID))),
        state_sql=_state("wake_fires", _FIRE_COLUMNS),
        parents=True,
    ),
    "fire list_for_schedule": _Site(
        "wake_fires",
        _fire,
        f"SELECT {_FIRE_SELECT} FROM wake_fires WHERE conversation_id = $1 AND schedule_id = $2 "
        "ORDER BY actual_fired_at DESC LIMIT $3",
        [_CONVS[0], _PARENT_SCHEDULES[0], _PAGE],
        lambda c: _listed(_fires(c).list_for_schedule(_CONVS[0], _PARENT_SCHEDULES[0], limit=_PAGE)),
        columns=_FIRE_COLUMNS,
        order_by=["actual_fired_at"],
        parents=True,
    ),
    "fire list_for_subscription": _Site(
        "wake_fires",
        _fire,
        f"SELECT {_FIRE_SELECT} FROM wake_fires WHERE conversation_id = $1 AND webhook_subscription_id = $2 "
        "ORDER BY actual_fired_at DESC LIMIT $3",
        [_CONVS[0], _PARENT_SUBSCRIPTIONS[0], _PAGE],
        lambda c: _listed(_fires(c).list_for_subscription(_CONVS[0], _PARENT_SUBSCRIPTIONS[0], limit=_PAGE)),
        columns=_FIRE_COLUMNS,
        order_by=["actual_fired_at"],
        parents=True,
    ),
    "fire list_for_conversation": _Site(
        "wake_fires",
        _fire,
        f"SELECT {_FIRE_SELECT} FROM wake_fires WHERE conversation_id = $1 ORDER BY actual_fired_at DESC LIMIT $2",
        [_CONVS[0], _PAGE],
        lambda c: _listed(_fires(c).list_for_conversation(_CONVS[0], limit=_PAGE)),
        columns=_FIRE_COLUMNS,
        order_by=["actual_fired_at"],
        parents=True,
    ),
    "fire latest_for_schedule": _Site(
        "wake_fires",
        _fire,
        f"SELECT {_FIRE_SELECT} FROM wake_fires WHERE conversation_id = $1 AND schedule_id = $2 "
        "ORDER BY actual_fired_at DESC LIMIT 1",
        [_CONVS[0], _PARENT_SCHEDULES[0]],
        lambda c: _one(_latest(c, None)),
        columns=_FIRE_COLUMNS,
        parents=True,
    ),
    "fire latest_for_schedule of statuses": _Site(
        "wake_fires",
        _fire,
        f"SELECT {_FIRE_SELECT} FROM wake_fires WHERE conversation_id = $1 AND schedule_id = $2 "
        "AND status = ANY($3::text[]) ORDER BY actual_fired_at DESC LIMIT 1",
        [_CONVS[0], _PARENT_SCHEDULES[0], ["fired"]],
        lambda c: _one(_latest(c, ("fired",))),
        columns=_FIRE_COLUMNS,
        parents=True,
    ),
    "fire link_started_conversation": _Site(
        "wake_fires",
        _fire,
        "UPDATE wake_fires SET started_conversation_id = $3 "
        "WHERE conversation_id = $1 AND fire_id = $2 AND status = 'dispatching' RETURNING fire_id",
        [_CONVS[0], _id("fire", 3), _CONVS[1]],
        lambda c: _linked(c, _id("fire", 3)),
        state_sql=_state("wake_fires", _FIRE_COLUMNS),
        parents=True,
    ),
    "fire finalize_success": _Site(
        "wake_fires",
        _fire,
        "UPDATE wake_fires SET status = $1, output_text = $2, latency_ms = $3, display_suppressed = $4 "
        "WHERE conversation_id = $5 AND fire_id = $6 AND status = 'dispatching'",
        ["fired", "done", 42, True, _CONVS[0], _id("fire", 3)],
        _wrote(
            lambda c: _fires(c).finalize_success(
                _CONVS[0], _id("fire", 3), status="fired", output_text="done", latency_ms=42, display_suppressed=True
            )
        ),
        state_sql=_state("wake_fires", _FIRE_COLUMNS),
        parents=True,
    ),
    "fire finalize_failed": _Site(
        "wake_fires",
        _fire,
        "UPDATE wake_fires SET status = 'failed', error = $1, latency_ms = $2 "
        "WHERE conversation_id = $3 AND fire_id = $4 AND status = 'dispatching'",
        ["boom", 7, _CONVS[0], _id("fire", 3)],
        _wrote(lambda c: _fires(c).finalize_failed(_CONVS[0], _id("fire", 3), error="boom", latency_ms=7)),
        state_sql=_state("wake_fires", _FIRE_COLUMNS),
        parents=True,
    ),
    "fire reap_stale_dispatching": _Site(
        "wake_fires",
        _fire,
        "UPDATE wake_fires SET status = 'failed', error = $1 "
        "WHERE status = 'dispatching' AND actual_fired_at < $2 "
        "RETURNING conversation_id, fire_id, started_conversation_id",
        [REAPED_DISPATCH_ERROR, _T0 + timedelta(minutes=7)],
        lambda c: _reaped(
            _fires(c).reap_stale_dispatching(_T0 + timedelta(minutes=9), older_than=timedelta(minutes=2))
        ),
        state_sql=_state("wake_fires", _FIRE_COLUMNS),
        parents=True,
    ),
    "fire reap_fires": _Site(
        "wake_fires",
        _fire,
        "UPDATE wake_fires SET status = 'failed', error = $1 "
        "WHERE status = 'dispatching' AND fire_id = ANY($2::uuid[]) "
        "RETURNING conversation_id, fire_id, started_conversation_id",
        [REAPED_DISPATCH_ERROR, [_id("fire", i) for i in (0, 1, 3, 6, 9, 13)]],
        lambda c: _reaped(_fires(c).reap_fires([_id("fire", i) for i in (0, 1, 3, 6, 9, 13)])),
        state_sql=_state("wake_fires", _FIRE_COLUMNS),
        parents=True,
    ),
    "fire count_in_window": _Site(
        "wake_fires",
        _fire,
        "SELECT COUNT(*) FROM wake_fires WHERE conversation_id = $1 AND actual_fired_at >= $2",
        [_CONVS[0], _T0 + timedelta(minutes=3)],
        lambda c: _counted(_fires(c).count_in_window(_CONVS[0], since=_T0 + timedelta(minutes=3))),
        parents=True,
    ),
    "fire count_in_window of a subscription": _Site(
        "wake_fires",
        _fire,
        "SELECT COUNT(*) FROM wake_fires "
        "WHERE conversation_id = $1 AND actual_fired_at >= $2 AND webhook_subscription_id = $3",
        [_CONVS[0], _T0 + timedelta(minutes=3), _PARENT_SUBSCRIPTIONS[0]],
        lambda c: _counted(
            _fires(c).count_in_window(
                _CONVS[0], since=_T0 + timedelta(minutes=3), webhook_subscription_id=_PARENT_SUBSCRIPTIONS[0]
            )
        ),
        parents=True,
    ),
    "subscription find_by_id": _Site(
        "webhook_subscriptions",
        _subscription,
        f"SELECT {', '.join(_SUBSCRIPTION_COLUMNS)} FROM webhook_subscriptions WHERE subscription_id = $1",
        [_SUB],
        lambda c: _one(_entity(_subscriptions(c).find_by_id(_SUB))),
        columns=_SUBSCRIPTION_COLUMNS,
    ),
    "subscription find_for_agent": _Site(
        "webhook_subscriptions",
        _subscription,
        f"SELECT {', '.join(_SUBSCRIPTION_COLUMNS)} FROM webhook_subscriptions WHERE agent_id = $1 AND subscription_id = $2",
        [_AGENTS[0], _SUB],
        lambda c: _one(_entity(_subscriptions(c).find_for_agent(_AGENTS[0], _SUB))),
        columns=_SUBSCRIPTION_COLUMNS,
    ),
    "subscription list_for_agent": _Site(
        "webhook_subscriptions",
        _subscription,
        f"SELECT {', '.join(_SUBSCRIPTION_COLUMNS)} FROM webhook_subscriptions WHERE agent_id = $1 ORDER BY date_created DESC",
        [_AGENTS[0]],
        lambda c: _listed(_subscriptions(c).list_for_agent(_AGENTS[0])),
        columns=_SUBSCRIPTION_COLUMNS,
        order_by=["date_created"],
    ),
    "subscription list_for_conversation": _Site(
        "webhook_subscriptions",
        _subscription,
        f"SELECT {', '.join(_SUBSCRIPTION_COLUMNS)} FROM webhook_subscriptions WHERE conversation_id = $1 "
        "ORDER BY date_created DESC",
        [_CONVS[0]],
        lambda c: _listed(_subscriptions(c).list_for_conversation(_CONVS[0])),
        columns=_SUBSCRIPTION_COLUMNS,
        order_by=["date_created"],
    ),
    "subscription rotate_secret": _Site(
        "webhook_subscriptions",
        _subscription,
        "UPDATE webhook_subscriptions SET secret_ciphertext = $3, date_updated = now() "
        "WHERE conversation_id = $1 AND subscription_id = $2",
        [_CONVS[0], _SUB, b"rotated"],
        _wrote(lambda c: _subscriptions(c).rotate_secret(_CONVS[0], _SUB, new_ciphertext=b"rotated")),
        state_sql=_state("webhook_subscriptions", _SUBSCRIPTION_COLUMNS),
    ),
    "subscription pause": _Site(
        "webhook_subscriptions",
        _subscription,
        "UPDATE webhook_subscriptions SET status = 'paused', date_updated = now() "
        "WHERE conversation_id = $1 AND subscription_id = $2",
        [_CONVS[0], _id("subscription", 1)],
        _wrote(lambda c: _subscriptions(c).pause(_CONVS[0], _id("subscription", 1))),
        state_sql=_state("webhook_subscriptions", _SUBSCRIPTION_COLUMNS),
    ),
    "subscription resume": _Site(
        "webhook_subscriptions",
        _subscription,
        "UPDATE webhook_subscriptions SET status = 'active', date_updated = now() "
        "WHERE conversation_id = $1 AND subscription_id = $2",
        [_CONVS[0], _SUB],
        _wrote(lambda c: _subscriptions(c).resume(_CONVS[0], _SUB)),
        state_sql=_state("webhook_subscriptions", _SUBSCRIPTION_COLUMNS),
    ),
    "subscription record_fire": _Site(
        "webhook_subscriptions",
        _subscription,
        "UPDATE webhook_subscriptions SET last_fired_at = $3, date_updated = now() "
        "WHERE conversation_id = $1 AND subscription_id = $2",
        [_CONVS[0], _SUB, _LATER],
        _wrote(lambda c: _subscriptions(c).record_fire(_CONVS[0], _SUB, fired_at=_LATER)),
        state_sql=_state("webhook_subscriptions", _SUBSCRIPTION_COLUMNS),
    ),
    "refusal list_refusals": _Site(
        "webhook_refusals",
        _refusal,
        f"SELECT {', '.join(_REFUSAL_COLUMNS)} FROM webhook_refusals WHERE subscription_id = $1 "
        "ORDER BY last_refused_at DESC LIMIT $2",
        [_PARENT_SUBSCRIPTIONS[0], _PAGE],
        lambda c: _subscriptions(c).list_refusals(_PARENT_SUBSCRIPTIONS[0], limit=_PAGE),
        order_by=["last_refused_at"],
        parents=True,
    ),
    "refusal record_refusal's prune": _Site(
        "webhook_refusals",
        _refusal,
        "DELETE FROM webhook_refusals WHERE subscription_id = $1 AND last_refused_at < $2",
        [_PARENT_SUBSCRIPTIONS[0], _T0 + timedelta(minutes=4)],
        _wrote(
            lambda c: _subscriptions(c).record_refusal(
                _CONVS[0],
                _PARENT_SUBSCRIPTIONS[0],
                reason="rate_limited",
                status_code=429,
                at=_REFUSED_AT,
                keep=_REFUSED_AT - (_T0 + timedelta(minutes=4)),
            )
        ),
        state_sql=_REFUSALS_BUT_THE_NEW_ONE,
        parents=True,
    ),
    "subscription fetch_from_store": _Site(
        "webhook_subscriptions",
        _subscription,
        f"SELECT {', '.join(_SUBSCRIPTION_COLUMNS)} FROM webhook_subscriptions "
        "WHERE conversation_id = $1 AND subscription_id = $2",
        [_CONVS[0], _SUB],
        lambda c: _one(_subscriptions(c).fetch_from_store((_CONVS[0], _SUB))),
    ),
    "subscription delete_from_store": _Site(
        "webhook_subscriptions",
        _subscription,
        "DELETE FROM webhook_subscriptions WHERE conversation_id = $1 AND subscription_id = $2",
        [_CONVS[0], _SUB],
        _wrote(lambda c: _subscriptions(c).delete_from_store((_CONVS[0], _SUB))),
        state_sql=_state("webhook_subscriptions", _SUBSCRIPTION_COLUMNS),
    ),
    "subscription fenced save": _Site(
        "webhook_subscriptions",
        _subscription,
        _fenced_update_sql("webhook_subscriptions", _SUBSCRIPTION_UPDATE, ("conversation_id", "subscription_id")),
        [*_subscription_params(_saved_subscription()), _FENCE],
        _wrote(lambda c: _subscriptions(c).save_to_store(_saved_subscription(), _FENCE)),
        state_sql=_state("webhook_subscriptions", _SUBSCRIPTION_COLUMNS),
    ),
}


async def _latest(conn: Any, statuses: tuple[str, ...] | None) -> dict[str, Any] | None:
    entity = await _fires(conn).latest_for_schedule(_CONVS[0], _PARENT_SCHEDULES[0], statuses=statuses)
    return None if entity is None else entity.to_dict()


async def _linked(conn: Any, fire_id: uuid.UUID) -> list[dict[str, Any]]:
    linked = await _fires(conn).link_started_conversation(_CONVS[0], fire_id, _CONVS[1], conn=conn)
    return [{"fire_id": fire_id}] if linked else []


async def _reaped(found: Awaitable[Sequence[Any]]) -> list[dict[str, Any]]:
    return [
        {
            "conversation_id": fire.conversation_id,
            "fire_id": fire.fire_id,
            "started_conversation_id": fire.started_conversation_id,
        }
        for fire in await found
    ]


async def _seed_parents(conn: asyncpg.Connection) -> None:
    """each tenant's parent schedule and subscription, the fires' foreign keys point at."""
    for tenant in (0, 1):
        schedule = dict(_schedule(Seeded(index=900 + tenant, case="parent", tenant=tenant, null=False, tie=0)))
        schedule["schedule_id"] = _PARENT_SCHEDULES[tenant]
        subscription = dict(_subscription(Seeded(index=900 + tenant, case="parent", tenant=tenant, null=False, tie=0)))
        subscription["subscription_id"] = _PARENT_SUBSCRIPTIONS[tenant]
        for table, row in (("agent_wake_schedules", schedule), ("webhook_subscriptions", subscription)):
            marks = ", ".join(f"${i}" for i in range(1, len(row) + 1))
            await conn.execute(f"INSERT INTO {table} ({', '.join(row)}) VALUES ({marks})", *row.values())  # noqa: S608


async def _found_for_agent(conn: Any) -> dict[str, Any] | None:
    entity = await _schedules(conn).find_for_agent(_AGENTS[0], _SID)
    return None if entity is None else entity.to_dict()


async def _entity(found: Awaitable[Any]) -> dict[str, Any] | None:
    entity = await found
    return None if entity is None else entity.to_dict()


async def _listed(found: Awaitable[Sequence[Any]]) -> list[dict[str, Any]]:
    return _rows(await found)


async def _counted(found: Awaitable[int]) -> list[dict[str, Any]]:
    return [{"count": await found}]


async def _claimed(conn: Any, schedule_id: uuid.UUID, expected: datetime) -> list[dict[str, Any]]:
    won = await _schedules(conn).claim_and_reschedule(
        conversation_id=_CONVS[0],
        schedule_id=schedule_id,
        expected_next_fire=expected,
        computed_next_fire=_LATER,
        new_status="active",
        now=_T0,
    )
    return [{"schedule_id": schedule_id}] if won else []


@pytest.fixture
async def conn(pg_schema: tuple[str, str]) -> AsyncIterator[asyncpg.Connection]:
    url, schema = pg_schema
    connection = await asyncpg.connect(url)
    try:
        await connection.execute(f'SET search_path TO "{schema}", public')
        runner = MigrationRunner()
        register_conversations(runner)
        register_skills(runner)
        register_wake(runner)
        await runner.apply_for_agent_schema(AsyncpgStore(connection))
        yield connection
    finally:
        await connection.close()


@pytest.mark.parametrize("site", list(_SITES), ids=list(_SITES))
async def test_a_converted_wake_statement_gives_the_old_statements_rows(conn: asyncpg.Connection, site: str) -> None:
    """Flips if a converted wake read answers other rows (or in another order), or a converted write leaves the
    table otherwise, than the hand-written statement it replaced, on PostgreSQL or YugabyteDB."""
    case = _SITES[site]
    if case.parents:
        await _seed_parents(conn)
    await assert_same_results(
        conn,
        table=case.table,
        row=case.row,
        old_sql=case.old_sql,
        params=case.params,
        helper=case.helper,
        columns=case.columns,
        order_by=case.order_by,
        page_size=_PAGE,
        state_sql=case.state_sql,
    )
