"""Agent-wake collections -- three-tier CRUD for wake schedules, fires,
webhook subscriptions.

The package subclasses :class:`BaseCollection` directly rather than
:class:`SchemaBackedCollection` because several columns
(``schedule_config`` JSONB on schedules,
``secret_ciphertext`` BYTEA on subscriptions, etc.) require type
handling beyond the schema-driven CRUD generator's built-in tags. The
hand-rolled SQL keeps the contract local; asyncpg's native codecs
round-trip ``dict[str, Any] <-> JSONB`` and ``bytes <-> BYTEA``
cleanly.

All three Collections declare ``partition_column = "conversation_id"``
as a class attribute so consumers (and the workspace partition-column
enforcement walker in
``packages/core/tests/enforcement/test_partition_column_enforcement.py``)
can confirm the partition contract by introspection.

Method contracts mirror ``docs/agent-wake/shard-01-schema-and-
collections.md`` "Public API" section (with the 2026-05-19 revision
deltas applied). Multi-row scans (``list_due_for_tick``,
``list_for_conversation``, ``list_for_schedule``) carry explicit
``# cache-bypass:`` annotations because they are not primary-key
addressable and would not benefit from L1 row caching -- the row
cache still serves ``get((conv_id, id))`` calls uniformly.

**Every targeted UPDATE on a schedule or subscription evicts the row from
every cache tier, on every replica, once L3 has it.** Those rows are read by
primary key through ``get`` -- the schedule and webhook tools read before
every edit, pause, resume and delete -- and ``get`` answers from L1, then L2,
before L3. An UPDATE that bypassed ``save_entity`` and left the tiers alone
left every replica that had read the row serving the old one, and the tools'
edit path then saved that old row back over L3: a paused schedule resumed, a
tick-expired one-shot re-armed, a rotated webhook secret restored. The
eviction runs however the UPDATE ended, since one that raised may still have
reached L3. An UPDATE joined to a caller's transaction is settled by the
:class:`~threetears.core.collections.CallerTransaction` when it ends instead. Both
are :meth:`~threetears.core.collections.BaseCollection.bypassing_write`, the core
owner of the rule.

**A multi-row scan caches nothing.** ``list_*``, ``find_by_id`` and
``latest_for_schedule`` read L3 outside the per-key fence ``get`` reads under. The entities
they return are loaded entities, which hold their own rows and write no cache tier
(:class:`~threetears.core.entities.base.BaseEntity`), so a concurrent write cannot leave L1
stale behind them and an eviction cannot empty them under a caller still reading them.

**Fire rows are the exception, and why is a claim about their readers.** No
code reads a ``wake_fires`` row by primary key:
:meth:`WakeFireCollection.latest_for_schedule`,
:meth:`WakeFireCollection.list_for_schedule`,
:meth:`WakeFireCollection.list_for_conversation` and
:meth:`WakeFireCollection.count_in_window` all read L3, so a fire UPDATE has
no cached copy anyone is served. A future by-pk reader of fires must add the
eviction these UPDATEs do not do.

``schedule_config`` JSONB shape per ``schedule_type`` (validation
lives in the agent-tools shard, not at the DB layer):

- ``daily_at``: ``{"hour": 14, "minute": 0, "tz":
  "America/Los_Angeles"}``
- ``every_n_hours``: ``{"n": 3}``
- ``random_within_window``: ``{"start_hour": 9, "end_hour": 21,
  "tz": "America/Los_Angeles", "fires_per_day": 1}``
- ``one_shot_at``: ``{"fire_at_iso": "2026-05-25T14:00:00+00:00"}``
- ``cron``: ``{"expr": "0 */3 * * *"}``
- ``relative_delay``: ``{"delay": "30m"}``
- ``interval``: ``{"seconds": 1800}``
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any, ClassVar
from uuid import UUID

from threetears.core.keyed import (
    AnyOf,
    Fragment,
    KeyBatch,
    Sort,
    counts_by,
    delete_keyed,
    keyset_page,
    keyset_rows,
    read_keyed,
    update_keyed,
)
from threetears.agent.wake.entities import (
    WakeFireEntity,
    WakeScheduleEntity,
    WebhookSubscriptionEntity,
)
from threetears.agent.wake.types import ReapedFire
from threetears.core.collections.base import BaseCollection
from threetears.core.serialization import (
    deserialize_from_json,
    serialize_to_json,
)
from threetears.observe import get_logger
from threetears.scheduled_jobs.collections import REAPED_DISPATCH_ERROR
from threetears.core.tenancy import TENANT_BY_SCHEMA

__all__ = [
    "REFUSALS_KEPT_FOR",
    "WEBHOOK_REFUSAL_REASONS",
    "WakeFireCollection",
    "WakeScheduleCollection",
    "WebhookSubscriptionCollection",
]


log = get_logger(__name__)

#: Why a webhook delivery to a known subscription is refused, as ``webhook_refusals.reason``
#: keeps it (CHECK-pinned in v010). A delivery to an unknown subscription has no row to hang on.
WEBHOOK_REFUSAL_REASONS: frozenset[str] = frozenset(
    {
        "paused",
        "missing_signature",
        "bad_signature",
        "unknown_scheme",
        "source_not_allowed",
        "rate_limited",
        "bad_payload",
        "bad_template",
    }
)

#: How long a subscription's refused deliveries are kept.
REFUSALS_KEPT_FOR = timedelta(days=30)


# Field-type hints used when L2 cache rounds a row through JSON. The
# helpers in ``threetears.core.serialization.deserialize_from_json``
# dispatch on these to rehydrate UUID / datetime / dict / bytes values
# back to their native Python types after a NATS KV pull.
_SCHEDULE_FIELD_TYPES: dict[str, Any] = {
    "schedule_id": UUID,
    "conversation_id": UUID,
    "user_id": UUID,
    "agent_id": UUID,
    "skill_id": UUID | None,
    "schedule_type": str,
    "schedule_config": dict,
    "task_prompt": str | None,
    "execution_mode": str,
    "status": str,
    "next_fire_at": datetime | None,
    "last_fired_at": datetime | None,
    "name": str | None,
    "missed_fire_policy": str,
    "context_from_schedule_id": UUID | None,
    "include_conversation_history": bool,
    "protected": bool,
    "date_created": datetime,
    "date_updated": datetime,
}


_FIRE_FIELD_TYPES: dict[str, Any] = {
    "fire_id": UUID,
    "conversation_id": UUID,
    "schedule_id": UUID | None,
    "webhook_subscription_id": UUID | None,
    "scheduled_fire_at": datetime | None,
    "actual_fired_at": datetime,
    "status": str,
    "display_suppressed": bool,
    "output_text": str | None,
    "latency_ms": int | None,
    "error": str | None,
    "started_conversation_id": UUID | None,
    "date_created": datetime,
    "fire_source": str,
    "requested_by": str | None,
}


_SUBSCRIPTION_FIELD_TYPES: dict[str, Any] = {
    "subscription_id": UUID,
    "conversation_id": UUID,
    "user_id": UUID,
    "agent_id": UUID,
    "default_skill_id": UUID | None,
    "name": str | None,
    "secret_ciphertext": bytes,
    "allowed_source_pattern": str | None,
    "execution_mode": str,
    "task_prompt_template": str | None,
    "verification_scheme": str,
    "status": str,
    "rate_limit_per_minute": int | None,
    "last_fired_at": datetime | None,
    "date_created": datetime,
    "date_updated": datetime,
}


# columns the Collection emits on INSERT, in positional-param order.
_SCHEDULE_INSERT_COLUMNS: tuple[str, ...] = (
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


# columns updated on ON CONFLICT (partition + pk excluded;
# ``date_created`` immutable; ``protected`` is set once, at insert, and the
# table's trigger refuses any change to it).
_SCHEDULE_UPDATE_COLUMNS: tuple[str, ...] = tuple(
    c for c in _SCHEDULE_INSERT_COLUMNS if c not in {"conversation_id", "schedule_id", "date_created", "protected"}
)


_FIRE_INSERT_COLUMNS: tuple[str, ...] = (
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


# fires are immutable post-finalize; the update set covers only the
# error / output fixups that the dispatcher applies if it captures
# post-LLM data after the initial row write (rare; the row is
# typically inserted with the terminal state in one shot).
_FIRE_UPDATE_COLUMNS: tuple[str, ...] = (
    "status",
    "display_suppressed",
    "output_text",
    "latency_ms",
    "error",
)


_SUBSCRIPTION_INSERT_COLUMNS: tuple[str, ...] = (
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


_SUBSCRIPTION_UPDATE_COLUMNS: tuple[str, ...] = tuple(
    c for c in _SUBSCRIPTION_INSERT_COLUMNS if c not in {"conversation_id", "subscription_id", "date_created"}
)


def _build_upsert_sql(
    table: str,
    insert_cols: Sequence[str],
    update_cols: Sequence[str],
    pk_cols: Sequence[str],
) -> str:
    """Build a parameterised INSERT ... ON CONFLICT DO UPDATE statement.

    Positional parameters bind to ``insert_cols`` in declared order.

    :param table: table name
    :ptype table: str
    :param insert_cols: columns to write on INSERT (in positional-param
        order)
    :ptype insert_cols: Sequence[str]
    :param update_cols: columns to update on conflict
    :ptype update_cols: Sequence[str]
    :param pk_cols: conflict target columns
    :ptype pk_cols: Sequence[str]
    :return: SQL string ready for ``execute()``
    :rtype: str
    """
    placeholders = ", ".join(f"${i + 1}" for i in range(len(insert_cols)))
    column_list = ", ".join(insert_cols)
    pk_list = ", ".join(pk_cols)
    set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
    return (
        f"INSERT INTO {table} ({column_list}) VALUES ({placeholders}) "
        f"ON CONFLICT ({pk_list}) DO UPDATE SET {set_clause}"
    )


async def _fenced_update(
    target: Any,
    table: str,
    data: dict[str, Any],
    value_for: Any,
    update_cols: Sequence[str],
    pk_cols: Sequence[str],
    original_timestamp: Any,
) -> int:
    """update a row's ``update_cols`` only where it still carries the ``date_updated`` it was read with.

    A save built from a row read before another writer changed it -- or deleted it -- updates nothing rather than
    writing that change away, and never re-inserts a deleted row.

    :param target: the pool or connection
    :ptype target: Any
    :param table: the table
    :ptype table: str
    :param data: the row
    :ptype data: dict[str, Any]
    :param value_for: ``(column, data) -> value`` as the save binds it
    :ptype value_for: Any
    :param update_cols: columns the save writes
    :ptype update_cols: Sequence[str]
    :param pk_cols: primary-key columns
    :ptype pk_cols: Sequence[str]
    :param original_timestamp: the fence
    :ptype original_timestamp: Any
    :return: rows updated (1, or 0 when the row changed or went since it was read)
    :rtype: int
    """
    written = await update_keyed(
        target,
        table,
        values={column: value_for(column, data) for column in update_cols},
        by={**{column: value_for(column, data) for column in pk_cols}, "date_updated": original_timestamp},
    )
    return written.count


_AGENT_WAKE_SCHEDULES_UPSERT_SQL = _build_upsert_sql(
    "agent_wake_schedules",
    _SCHEDULE_INSERT_COLUMNS,
    _SCHEDULE_UPDATE_COLUMNS,
    ("conversation_id", "schedule_id"),
)


_SCHEDULE_PK_COLUMNS: tuple[str, ...] = ("conversation_id", "schedule_id")


_WAKE_FIRES_UPSERT_SQL = _build_upsert_sql(
    "wake_fires",
    _FIRE_INSERT_COLUMNS,
    _FIRE_UPDATE_COLUMNS,
    ("conversation_id", "fire_id"),
)


_WEBHOOK_SUBSCRIPTIONS_UPSERT_SQL = _build_upsert_sql(
    "webhook_subscriptions",
    _SUBSCRIPTION_INSERT_COLUMNS,
    _SUBSCRIPTION_UPDATE_COLUMNS,
    ("conversation_id", "subscription_id"),
)


_SUBSCRIPTION_PK_COLUMNS: tuple[str, ...] = ("conversation_id", "subscription_id")


_SCHEDULES = "agent_wake_schedules"
_FIRES = "wake_fires"
_SUBSCRIPTIONS = "webhook_subscriptions"
_REFUSALS = "webhook_refusals"

#: a SET value of the statement's own clock
_NOW = Fragment("now()")

#: fires newest first, as ``idx_wake_fires_*_time`` serves them (a page with no cursor: the index's column alone)
_NEWEST_FIRST = (Sort("actual_fired_at", cast="timestamptz", descending=True),)

#: subscriptions newest first, the id breaking a tie so a listing pages
_NEWEST_SUBSCRIPTIONS_FIRST = (
    Sort("date_created", cast="timestamptz", descending=True),
    Sort("subscription_id", cast="uuid"),
)

#: what a reaped fire answers back
_REAPED_COLUMNS = ("conversation_id", "fire_id", "started_conversation_id")


async def _read_one(
    pool: Any, table: str, columns: Sequence[str], key: str, value: Any, by: dict[str, Any]
) -> dict[str, Any] | None:
    """the one row of ``table`` whose ``key`` is ``value`` (within ``by``), or None.

    :param pool: the pool
    :ptype pool: Any
    :param table: the table
    :ptype table: str
    :param columns: the columns read
    :ptype columns: Sequence[str]
    :param key: the unique key column
    :ptype key: str
    :param value: its value
    :ptype value: Any
    :param by: the partition (or owner) pinned beside it
    :ptype by: dict[str, Any]
    :return: the row, or None
    :rtype: dict[str, Any] | None
    """
    rows = await read_keyed(pool, table, keys=KeyBatch((key,), [(value,)], ("uuid",)), columns=columns, by=by)
    return rows[0] if rows else None


class WakeScheduleCollection(BaseCollection[WakeScheduleEntity]):
    """Three-tier collection for :class:`WakeScheduleEntity`.

    Composite primary key ``(conversation_id, schedule_id)``;
    partition column ``conversation_id``. ``UNIQUE (schedule_id)``
    standalone constraint lets ``wake_fires.schedule_id`` reference the
    bare id.

    Domain methods cover the wake-tick lifecycle:

    - :meth:`list_due_for_tick` -- finds schedules ready to fire.
    - :meth:`list_active_for_conversation` -- conversation-scoped enum.
    - :meth:`claim_and_reschedule` -- atomic dispatch claim primitive
      (shard 02 fills in the body; shard 01 declares the signature).
    - :meth:`pause` / :meth:`resume` -- status flips.
    - :meth:`mark_expired` -- one-shot terminal transition.
    """

    tenancy = TENANT_BY_SCHEMA

    primary_key_column: str | tuple[str, ...] = (
        "conversation_id",
        "schedule_id",
    )

    partition_column: ClassVar[str] = "conversation_id"

    @property
    def table_name(self) -> str:
        """Return the L3 table name."""
        return "agent_wake_schedules"

    @property
    def entity_class(self) -> type[WakeScheduleEntity]:
        """Return the entity class."""
        return WakeScheduleEntity

    # --- BaseCollection contract ---

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """Fetch a row by composite pk.

        :param entity_id: tuple ``(conversation_id, schedule_id)``
        :ptype entity_id: Any
        :return: row dict on hit, ``None`` on miss
        :rtype: dict[str, Any] | None
        """
        if self.l3_pool is None:
            return None
        conv_id, schedule_id = self.normalize_pk(entity_id)
        return await _read_one(
            self.l3_pool, _SCHEDULES, _SCHEDULE_INSERT_COLUMNS, "schedule_id", schedule_id, {"conversation_id": conv_id}
        )

    async def save_to_store(
        self,
        data: dict[str, Any],
        original_timestamp: Any = None,
        *,
        conn: Any = None,
    ) -> int:
        """Insert a new schedule row, or update an existing one only as it was read.

        With ``original_timestamp`` -- the ``date_updated`` the row carried when it was read, which
        :meth:`BaseCollection.save_entity` passes for every entity read from a tier -- the write is
        an update-only statement fenced on that value. The schedule tools save the whole row they
        read, and between that read and this write the tick can claim and expire the row, another
        replica can pause or resume it, or it can be deleted; each of those moves
        ``date_updated`` or removes the row, so the save updates nothing and ``save_entity``
        raises :class:`~threetears.core.exceptions.ConcurrentModificationError` instead of
        writing the change away. Without it (a new row) the unfenced upsert runs.

        :param data: row dict keyed by column name; must carry both pk
            columns and every non-nullable column
        :ptype data: dict[str, Any]
        :param original_timestamp: the ``date_updated`` the row was read with, or ``None`` for an
            insert
        :ptype original_timestamp: Any
        :param conn: optional asyncpg-compatible connection
        :ptype conn: Any
        :return: rows affected (1 on success, 0 when the row changed or was deleted since it was
            read)
        :rtype: int
        """
        target = conn if conn is not None else self.l3_pool
        if target is None:
            return 0
        if original_timestamp is None:
            await target.execute(_AGENT_WAKE_SCHEDULES_UPSERT_SQL, *_schedule_insert_params(data))
            return 1
        return await _fenced_update(
            target,
            _SCHEDULES,
            data,
            _schedule_value_for_column,
            _SCHEDULE_UPDATE_COLUMNS,
            _SCHEDULE_PK_COLUMNS,
            original_timestamp,
        )

    async def delete_from_store(self, entity_id: Any) -> None:
        """Delete a schedule row by composite pk.

        ``ON DELETE CASCADE`` on ``wake_fires.schedule_id`` ensures
        every fire history row for the deleted schedule is removed in
        the same transaction.

        :param entity_id: tuple ``(conversation_id, schedule_id)``
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return None
        conv_id, schedule_id = self.normalize_pk(entity_id)
        await delete_keyed(self.l3_pool, _SCHEDULES, by={"conversation_id": conv_id, "schedule_id": schedule_id})
        return None

    def serialize(self, data: dict[str, Any]) -> bytes:
        """Encode a row dict for L2 (NATS KV) storage."""
        return serialize_to_json(data)

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """Decode L2-cached bytes back to a row dict."""
        return deserialize_from_json(data, _SCHEDULE_FIELD_TYPES)

    # --- Domain methods ---

    async def list_due_for_tick(
        self,
        now: datetime,
        *,
        limit: int = 200,
    ) -> list[WakeScheduleEntity]:
        """Return active schedules whose ``next_fire_at <= now``.

        Used by the tick engine (shard 02) to scope each tick's work.
        Cross-conversation scan; the partition-column enforcement
        walker exempts the literal because the query is a global
        platform-level enumeration of "what fires next" -- explicit
        ``__SPANS_PARTITIONS__`` marker keeps the gate happy.

        :param now: tick instant; rows with ``next_fire_at <= now`` are
            returned
        :ptype now: datetime
        :param limit: per-tick cap (defaults to 200; shard 02 will
            paginate larger backlogs)
        :ptype limit: int
        :return: list of due schedules ordered by ``next_fire_at`` ASC
        :rtype: list[WakeScheduleEntity]
        """
        if self.l3_pool is None:
            return []
        # spans-partitions: the tick engine enumerates ready schedules
        # across every conversation; the partition predicate cannot
        # apply here by construction. Same shape as agent-tools'
        # cross-partition LRU scan.
        # one page in the due index's order (idx_wake_schedules_next_fire_asc, status = 'active'): no cursor follows
        # it, so the sort is the index's column alone, as the index serves it
        page = await keyset_page(
            self.l3_pool,
            _SCHEDULES,
            columns=_SCHEDULE_INSERT_COLUMNS,
            by={"status": "active"},
            order=[Sort("next_fire_at", cast="timestamptz")],
            limit=limit,
            where=[Fragment('t."next_fire_at" IS NOT NULL AND t."next_fire_at" <= $1', (now,))],
        )
        rows = page.rows
        return [self.entity_class(dict(row), is_new=False, collection=self) for row in rows]

    async def list_active_for_conversation(
        self,
        conversation_id: UUID,
    ) -> list[WakeScheduleEntity]:
        """Return active schedules for a conversation.

        Powers per-conv cap enforcement (PLACEMENT §1.9) plus the
        "what is this conversation waking on" admin view. PostgreSQL reads
        the ``idx_wake_schedules_conv_active`` partial index; YugabyteDB the
        conversation's own primary key prefix (v013's docstring says why).

        :param conversation_id: partition column
        :ptype conversation_id: UUID
        :return: list of active schedules
        :rtype: list[WakeScheduleEntity]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: multi-row scan by conversation_id is not pk-
        # addressable; L1 row cache cannot serve.
        rows = await keyset_rows(
            self.l3_pool,
            _SCHEDULES,
            columns=_SCHEDULE_INSERT_COLUMNS,
            by={"conversation_id": conversation_id, "status": "active"},
            order=[Sort("next_fire_at", cast="timestamptz", nullable=True), Sort("schedule_id", cast="uuid")],
        )
        return [self.entity_class(dict(row), is_new=False, collection=self) for row in rows]

    async def list_for_conversation(
        self,
        conversation_id: UUID,
    ) -> list[WakeScheduleEntity]:
        """Return every schedule for a conversation regardless of status.

        Used by REST list endpoints + the frontend schedule manager.

        :param conversation_id: partition column
        :ptype conversation_id: UUID
        :return: list of schedules ordered by ``date_created`` DESC
        :rtype: list[WakeScheduleEntity]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: multi-row scan.
        rows = await keyset_rows(
            self.l3_pool,
            _SCHEDULES,
            columns=_SCHEDULE_INSERT_COLUMNS,
            by={"conversation_id": conversation_id},
            order=[Sort("date_created", cast="timestamptz", descending=True), Sort("schedule_id", cast="uuid")],
        )
        return [self.entity_class(dict(row), is_new=False, collection=self) for row in rows]

    async def count_active_for_conversation(
        self,
        conversation_id: UUID,
    ) -> int:
        """Return the count of active schedules for a conversation.

        Powers the per-conv active-schedule cap enforcement (PLACEMENT
        §1.9, default cap = 10 via ``WakeConfig``). Faster than
        ``len(await list_active_for_conversation(...))`` because the
        DB returns an aggregate.

        :param conversation_id: partition column
        :ptype conversation_id: UUID
        :return: integer count of active schedules
        :rtype: int
        """
        if self.l3_pool is None:
            return 0
        # cache-bypass: aggregate COUNT not pk-addressable.
        counted = await counts_by(
            self.l3_pool,
            _SCHEDULES,
            column="conversation_id",
            values=AnyOf([conversation_id], "uuid"),
            by={"status": "active"},
        )
        return counted.get(conversation_id, 0)

    async def find_for_agent(
        self,
        agent_id: UUID,
        schedule_id: UUID,
    ) -> WakeScheduleEntity | None:
        """Return one of an agent's schedules, in whichever conversation it lives.

        An agent's wakes live in its wake conversations, and a tool running in
        any conversation of that agent acts on them. The lookup is by the
        agent and the bare id, so another agent's schedule is not found: its
        existence does not leak.

        :param agent_id: the agent that must own the schedule
        :ptype agent_id: UUID
        :param schedule_id: the schedule
        :ptype schedule_id: UUID
        :return: the schedule, or ``None`` when the agent has no such schedule
        :rtype: WakeScheduleEntity | None
        """
        if self.l3_pool is None:
            return None
        # spans-partitions: an agent's wakes live in several conversations and
        # the caller names the wake, not its conversation; the standalone
        # UNIQUE (schedule_id) makes the lookup exact and the agent_id
        # predicate keeps it to the caller's own. conversation_id is read off
        # the row.
        row = await _read_one(
            self.l3_pool, _SCHEDULES, _SCHEDULE_INSERT_COLUMNS, "schedule_id", schedule_id, {"agent_id": agent_id}
        )
        if row is None:
            return None
        return WakeScheduleEntity(row, is_new=False, collection=self)

    async def list_for_agent(
        self,
        agent_id: UUID,
        *,
        include_paused: bool = True,
    ) -> list[WakeScheduleEntity]:
        """Return an agent's schedules across all its conversations.

        Expired one-shots are left out: they never fire again and would bury
        the live ones. Hits ``idx_wake_schedules_agent_status``.

        :param agent_id: the agent
        :ptype agent_id: UUID
        :param include_paused: include paused schedules
        :ptype include_paused: bool
        :return: schedules, the next to fire first
        :rtype: list[WakeScheduleEntity]
        """
        if self.l3_pool is None:
            return []
        statuses = ["active", "paused"] if include_paused else ["active"]
        # spans-partitions: the agent's listing covers every conversation it
        # holds wakes in; agent_id is the scope. conversation_id is read off
        # each row.
        rows = await keyset_rows(
            self.l3_pool,
            _SCHEDULES,
            columns=_SCHEDULE_INSERT_COLUMNS,
            by={"agent_id": agent_id, "status": AnyOf(statuses)},
            order=[
                Sort("next_fire_at", cast="timestamptz", nullable=True),
                Sort("date_created", cast="timestamptz"),
                Sort("schedule_id", cast="uuid"),
            ],
        )
        return [WakeScheduleEntity(dict(row), is_new=False, collection=self) for row in rows]

    async def update_next_fire_at(
        self,
        conversation_id: UUID,
        schedule_id: UUID,
        *,
        next_fire_at: datetime | None,
        last_fired_at: datetime | None = None,
    ) -> None:
        """Update the schedule's ``next_fire_at`` (+ optional last_fired_at).

        Called by the tick engine (shard 02) after computing the next
        fire instant. ``next_fire_at=None`` parks the schedule (used
        for one-shot terminal transitions before ``mark_expired`` is
        chained).

        :param conversation_id: partition column
        :ptype conversation_id: UUID
        :param schedule_id: target schedule
        :ptype schedule_id: UUID
        :param next_fire_at: new fire time (or ``None`` to park)
        :ptype next_fire_at: datetime | None
        :param last_fired_at: optional last-fired stamp to record
        :ptype last_fired_at: datetime | None
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return None
        async with self.bypassing_write((conversation_id, schedule_id)):
            if last_fired_at is None:
                # cache-bypass: targeted UPDATE on the next_fire_at column; the row is evicted
                # from every tier once it lands.
                await update_keyed(
                    self.l3_pool,
                    _SCHEDULES,
                    values={"next_fire_at": next_fire_at, "date_updated": _NOW},
                    by={"conversation_id": conversation_id, "schedule_id": schedule_id},
                )
            else:
                # cache-bypass: targeted UPDATE with both timestamps; the row is evicted from
                # every tier once it lands.
                await update_keyed(
                    self.l3_pool,
                    _SCHEDULES,
                    values={"next_fire_at": next_fire_at, "last_fired_at": last_fired_at, "date_updated": _NOW},
                    by={"conversation_id": conversation_id, "schedule_id": schedule_id},
                )
        return None

    async def pause(
        self,
        conversation_id: UUID,
        schedule_id: UUID,
    ) -> None:
        """Flip ``status`` to ``'paused'`` and clear ``next_fire_at``.

        Idempotent: replaying on an already-paused schedule is a no-op.

        :param conversation_id: partition column
        :ptype conversation_id: UUID
        :param schedule_id: target schedule
        :ptype schedule_id: UUID
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return None
        async with self.bypassing_write((conversation_id, schedule_id)):
            # cache-bypass: targeted UPDATE on status column; the row is evicted from every
            # tier once it lands.
            await update_keyed(
                self.l3_pool,
                _SCHEDULES,
                values={"status": "paused", "next_fire_at": None, "date_updated": _NOW},
                by={"conversation_id": conversation_id, "schedule_id": schedule_id, "status": "active"},
            )
        return None

    async def resume(
        self,
        conversation_id: UUID,
        schedule_id: UUID,
        *,
        next_fire_at: datetime,
        conn: Any = None,
    ) -> None:
        """Flip ``status`` to ``'active'`` and set ``next_fire_at``.

        Caller supplies the recomputed ``next_fire_at`` (the platform
        does not own scheduling math;
        ``threetears.scheduled_jobs.compute_next_fire_at`` provides it).
        Idempotent: replay on an already-active schedule
        overwrites the timestamp.

        ``conn`` lets the cap-serialized resume path
        (:func:`threetears.agent.wake.rate_limit.resume_schedule_serialized`)
        bind the UPDATE to the same transaction that holds the
        per-conversation advisory lock + active-count, so the
        re-activation cap holds atomically. That transaction must be
        opened through :class:`~threetears.core.collections.CallerTransaction`,
        which evicts the row from every cache tier once it ends: the row is
        not final until then. When ``conn`` is ``None`` the UPDATE runs on
        the pool (the direct, non-cap path) and the row is evicted as soon
        as it lands.

        :param conversation_id: partition column
        :ptype conversation_id: UUID
        :param schedule_id: target schedule
        :ptype schedule_id: UUID
        :param next_fire_at: when the resumed schedule should next fire
        :ptype next_fire_at: datetime
        :param conn: optional asyncpg-compatible connection (binds the
            UPDATE to a caller-owned transaction opened by
            :class:`~threetears.core.collections.CallerTransaction`)
        :ptype conn: Any
        :return: nothing
        :rtype: None
        :raises ValueError: if ``conn`` is given and its transaction was not
            opened by :class:`~threetears.core.collections.CallerTransaction`
        """
        target = conn if conn is not None else self.l3_pool
        if target is None:
            return None
        # joined to ``conn``, the row is enrolled in the caller's transaction before the write and
        # settled when it ends; on the pool it is evicted once the UPDATE lands, however it ended.
        async with self.bypassing_write((conversation_id, schedule_id), conn=conn):
            # cache-bypass: targeted UPDATE flipping paused -> active; the row is evicted from
            # every tier once it is final.
            await update_keyed(
                target,
                _SCHEDULES,
                values={"status": "active", "next_fire_at": next_fire_at, "date_updated": _NOW},
                by={"conversation_id": conversation_id, "schedule_id": schedule_id},
                where=[Fragment(""""status" != 'expired'""")],
            )
        return None

    async def claim_and_reschedule(
        self,
        *,
        conversation_id: UUID,
        schedule_id: UUID,
        expected_next_fire: datetime,
        computed_next_fire: datetime | None,
        new_status: str,
        now: datetime,
    ) -> bool:
        """Atomically claim a due schedule and advance its ``next_fire_at``.

        Optimistic-CAS UPDATE: the predicate ``next_fire_at =
        expected_next_fire`` ensures exactly one tick wins when two
        pods briefly disagree about the cross-pod lock (or, more
        commonly, when the same pod runs overlapping tick bodies).
        Returns ``True`` on a successful claim, ``False`` when another
        tick has already advanced the row.

        The UPDATE also stamps ``last_fired_at = now`` and rewrites
        ``status`` (the caller computes ``'expired'`` for terminal
        one-shots; ``'active'`` otherwise).

        Takes ``conversation_id`` (partition column) so the SQL
        predicate carries it -- pinned by
        ``test_partition_column_enforcement.py``.

        :param conversation_id: partition column
        :ptype conversation_id: UUID
        :param schedule_id: target schedule
        :ptype schedule_id: UUID
        :param expected_next_fire: the ``next_fire_at`` value the
            caller observed when it picked up the schedule; the CAS
            predicate
        :ptype expected_next_fire: datetime
        :param computed_next_fire: the new ``next_fire_at`` (may be
            ``None`` for terminal one-shots)
        :ptype computed_next_fire: datetime | None
        :param new_status: the new ``status`` value
        :ptype new_status: str
        :param now: tick instant; written as ``last_fired_at`` and
            ``date_updated``
        :ptype now: datetime
        :return: ``True`` on successful claim, ``False`` on CAS miss
        :rtype: bool
        """
        if self.l3_pool is None:
            return False
        # A won claim evicts the row from every tier: nothing reads a cached row back into place
        # otherwise, and the schedule tools save the row they read -- a stale one re-arms an
        # expired one-shot. A lost claim changed nothing and evicts nothing; one that raised may
        # have landed, so it evicts.
        async with self.bypassing_write((conversation_id, schedule_id)) as write:
            # cache-bypass: atomic CAS UPDATE; the row is evicted from every tier once it lands.
            claim = await update_keyed(
                self.l3_pool,
                _SCHEDULES,
                values={
                    "next_fire_at": computed_next_fire,
                    "last_fired_at": now,
                    "date_updated": now,
                    "status": new_status,
                },
                by={"conversation_id": conversation_id, "schedule_id": schedule_id, "next_fire_at": expected_next_fire},
                returning=["schedule_id"],
            )
            if not claim.count:
                write.unchanged()
        return bool(claim.count)

    async def mark_expired(
        self,
        conversation_id: UUID,
        schedule_id: UUID,
    ) -> None:
        """Flip ``status`` to ``'expired'`` (one-shot terminal transition).

        Used when a ``one_shot_at`` schedule's single fire completes.
        ``expired`` schedules cannot be resumed (per PLACEMENT §3.5
        lifecycle); a new schedule must be created.

        :param conversation_id: partition column
        :ptype conversation_id: UUID
        :param schedule_id: target schedule
        :ptype schedule_id: UUID
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return None
        async with self.bypassing_write((conversation_id, schedule_id)):
            # cache-bypass: targeted UPDATE on status column; the row is evicted from every
            # tier once it lands.
            await update_keyed(
                self.l3_pool,
                _SCHEDULES,
                values={"status": "expired", "next_fire_at": None, "date_updated": _NOW},
                by={"conversation_id": conversation_id, "schedule_id": schedule_id},
            )
        return None


class WakeFireCollection(BaseCollection[WakeFireEntity]):
    """Three-tier collection for :class:`WakeFireEntity`.

    Composite primary key ``(conversation_id, fire_id)``; partition
    column ``conversation_id``. Fires are append-mostly: the typical
    write path inserts the row with the terminal status in one shot;
    ``save_entity`` is allowed but the upsert path's update columns
    are limited to the status / error / output_text fixups.

    The ``schedule_id`` FK uses the cross-package standalone
    ``UNIQUE (schedule_id)`` constraint on ``agent_wake_schedules``;
    deleting a schedule cascades and removes the fire history.
    ``webhook_subscription_id`` FK is added by v003 retro-add and is
    ``ON DELETE SET NULL`` so webhook subscription deletes leave the
    history visible.
    """

    tenancy = TENANT_BY_SCHEMA

    primary_key_column: str | tuple[str, ...] = (
        "conversation_id",
        "fire_id",
    )

    partition_column: ClassVar[str] = "conversation_id"

    @property
    def table_name(self) -> str:
        """Return the L3 table name."""
        return "wake_fires"

    @property
    def entity_class(self) -> type[WakeFireEntity]:
        """Return the entity class."""
        return WakeFireEntity

    # --- BaseCollection contract ---

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """Fetch a row by composite pk."""
        if self.l3_pool is None:
            return None
        conv_id, fire_id = self.normalize_pk(entity_id)
        return await _read_one(
            self.l3_pool, _FIRES, _FIRE_INSERT_COLUMNS, "fire_id", fire_id, {"conversation_id": conv_id}
        )

    async def save_to_store(
        self,
        data: dict[str, Any],
        original_timestamp: Any = None,
        *,
        conn: Any = None,
    ) -> int:
        """Upsert a fire row.

        :param data: row dict keyed by column name
        :ptype data: dict[str, Any]
        :param original_timestamp: always ``None``: ``wake_fires`` has no ``date_updated``
            column, so an entity read from it carries no fence. Fires are written once as new
            rows and finalized by targeted UPDATEs, never re-saved from a read.
        :ptype original_timestamp: Any
        :param conn: optional asyncpg-compatible connection
        :ptype conn: Any
        :return: rows affected
        :rtype: int
        """
        del original_timestamp
        params = _fire_insert_params(data)
        target = conn if conn is not None else self.l3_pool
        if target is None:
            return 0
        await target.execute(_WAKE_FIRES_UPSERT_SQL, *params)
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        """Delete a fire row by composite pk."""
        if self.l3_pool is None:
            return None
        conv_id, fire_id = self.normalize_pk(entity_id)
        await delete_keyed(self.l3_pool, _FIRES, by={"conversation_id": conv_id, "fire_id": fire_id})
        return None

    def serialize(self, data: dict[str, Any]) -> bytes:
        """Encode a row dict for L2 (NATS KV) storage."""
        return serialize_to_json(data)

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """Decode L2-cached bytes back to a row dict."""
        return deserialize_from_json(data, _FIRE_FIELD_TYPES)

    # --- Domain methods ---

    async def record(
        self,
        conversation_id: UUID,
        fire: WakeFireEntity,
    ) -> None:
        """Persist a fire row.

        ``conversation_id`` is taken as an argument so the partition-
        column contract is visible on the call site; the value is
        already on the entity but the parameter is a contract reminder
        + guard against bugs that construct an entity with a stale
        ``conversation_id``.

        :param conversation_id: partition column expected on the entity
        :ptype conversation_id: UUID
        :param fire: fire entity to persist
        :ptype fire: WakeFireEntity
        :return: nothing
        :rtype: None
        :raises ValueError: when ``fire.conversation_id`` does not
            match the caller's ``conversation_id``
        """
        if fire.conversation_id != conversation_id:
            raise ValueError(
                f"record(): entity conversation_id {fire.conversation_id!r} does not match caller {conversation_id!r}",
            )
        await self.save_entity(fire)

    async def list_for_schedule(
        self,
        conversation_id: UUID,
        schedule_id: UUID,
        *,
        limit: int = 20,
    ) -> list[WakeFireEntity]:
        """List fires for a schedule, newest first.

        Powers the per-schedule history view + ``context_from`` lookup
        (shard 03's handler reads the latest fire's ``output_text``).
        Hits ``idx_wake_fires_schedule_time``.

        :param conversation_id: partition column
        :ptype conversation_id: UUID
        :param schedule_id: schedule whose history to read
        :ptype schedule_id: UUID
        :param limit: page size
        :ptype limit: int
        :return: list of fire entities ordered by ``actual_fired_at`` DESC
        :rtype: list[WakeFireEntity]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: per-schedule history scan.
        page = await keyset_page(
            self.l3_pool,
            _FIRES,
            columns=_FIRE_INSERT_COLUMNS,
            by={"conversation_id": conversation_id, "schedule_id": schedule_id},
            order=_NEWEST_FIRST,
            limit=limit,
        )
        rows = page.rows
        return [self.entity_class(dict(row), is_new=False, collection=self) for row in rows]

    async def list_for_subscription(
        self,
        conversation_id: UUID,
        webhook_subscription_id: UUID,
        *,
        limit: int = 20,
    ) -> list[WakeFireEntity]:
        """List the fires a webhook subscription started, newest first.

        The accepted deliveries of a subscription: each one is a fire row.
        Refused deliveries are not fires; they are read with
        :meth:`WebhookSubscriptionCollection.list_refusals`.

        :param conversation_id: partition column
        :ptype conversation_id: UUID
        :param webhook_subscription_id: the subscription whose fires to read
        :ptype webhook_subscription_id: UUID
        :param limit: page size
        :ptype limit: int
        :return: fire entities ordered by ``actual_fired_at`` DESC
        :rtype: list[WakeFireEntity]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: per-subscription history scan.
        page = await keyset_page(
            self.l3_pool,
            _FIRES,
            columns=_FIRE_INSERT_COLUMNS,
            by={"conversation_id": conversation_id, "webhook_subscription_id": webhook_subscription_id},
            order=_NEWEST_FIRST,
            limit=limit,
        )
        rows = page.rows
        return [self.entity_class(dict(row), is_new=False, collection=self) for row in rows]

    async def list_for_conversation(
        self,
        conversation_id: UUID,
        *,
        limit: int = 50,
    ) -> list[WakeFireEntity]:
        """List fires within a conversation, newest first.

        Powers the conversation-scoped "what fired here" admin view.
        Hits ``idx_wake_fires_conv_time``.

        :param conversation_id: partition column
        :ptype conversation_id: UUID
        :param limit: page size
        :ptype limit: int
        :return: list of fire entities ordered by ``actual_fired_at`` DESC
        :rtype: list[WakeFireEntity]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: per-conversation scan.
        page = await keyset_page(
            self.l3_pool,
            _FIRES,
            columns=_FIRE_INSERT_COLUMNS,
            by={"conversation_id": conversation_id},
            order=_NEWEST_FIRST,
            limit=limit,
        )
        rows = page.rows
        return [self.entity_class(dict(row), is_new=False, collection=self) for row in rows]

    async def latest_for_schedule(
        self,
        conversation_id: UUID,
        schedule_id: UUID,
        *,
        statuses: tuple[str, ...] | None = None,
    ) -> WakeFireEntity | None:
        """Return the most recent fire for a schedule, or ``None``.

        Shorthand for ``list_for_schedule(..., limit=1)[0]`` -- the
        ``context_from`` resolver in shard 03 calls this in a hot path,
        so a dedicated method avoids constructing a throwaway list.

        :param conversation_id: partition column
        :ptype conversation_id: UUID
        :param schedule_id: schedule whose latest fire to read
        :ptype schedule_id: UUID
        :param statuses: only fires that ended in one of these; ``None`` reads the latest whatever it was
        :ptype statuses: tuple[str, ...] | None
        :return: latest fire entity or ``None``
        :rtype: WakeFireEntity | None
        """
        if self.l3_pool is None:
            return None
        # cache-bypass: lookup by (conv_id, schedule_id) is not pk-
        # addressable for the row cache (the pk is (conv_id, fire_id)).
        by: dict[str, Any] = {"conversation_id": conversation_id, "schedule_id": schedule_id}
        if statuses is not None:
            by["status"] = AnyOf(list(statuses))
        page = await keyset_page(
            self.l3_pool, _FIRES, columns=_FIRE_INSERT_COLUMNS, by=by, order=_NEWEST_FIRST, limit=1
        )
        if not page.rows:
            return None
        return self.entity_class(dict(page.rows[0]), is_new=False, collection=self)

    async def create_dispatching(
        self,
        *,
        fire_id: UUID,
        schedule_id: UUID | None,
        webhook_subscription_id: UUID | None,
        conversation_id: UUID,
        scheduled_fire_at: datetime | None,
        actual_fired_at: datetime,
        fire_source: str,
        execution_mode: str,
        requested_by: str | None = None,
    ) -> None:
        """Insert an initial in-flight ``wake_fires`` row.

        Called by the tick body (shard 02) and the webhook receiver
        (shard 06) immediately after a fire claim succeeds. The row
        lands with ``status='dispatching'`` -- a distinct placeholder
        added in migration v004 so a half-completed row is queryable
        as in-flight rather than terminal. The dispatch callback
        (shard 03) finalizes via :meth:`finalize_success` /
        :meth:`finalize_failed` which overwrite to the real terminal
        status. The two-write pattern lets the audit trail capture
        the fact a fire was attempted even if the dispatcher crashes
        before producing output.

        Using ``'dispatching'`` rather than pre-claiming
        ``'fired'`` matters because the ``context_from`` resolver
        keys off ``status IN {'fired', 'fired_silent'}``: if a future
        refactor parallelises per-schedule dispatch inside the tick
        body (PLACEMENT.md §1.3 anti-pattern), a downstream wake with
        ``context_from = upstream`` would otherwise pick up an
        in-flight row's NULL ``output_text`` and silently produce an
        empty context block.

        ``fire_source`` is stored on the row (v009), with ``requested_by``
        for a fire run by hand. ``execution_mode`` is not a fire column (it
        lives on the schedule); it is accepted for callsite-symmetry with
        the trigger envelope.

        :param fire_id: pre-generated UUIDv7
        :ptype fire_id: UUID
        :param schedule_id: source schedule (NULL for webhook fires)
        :ptype schedule_id: UUID | None
        :param webhook_subscription_id: source webhook (NULL for
            scheduled fires)
        :ptype webhook_subscription_id: UUID | None
        :param conversation_id: partition column
        :ptype conversation_id: UUID
        :param scheduled_fire_at: the planned fire time (None for
            webhook fires; the schedule's ``next_fire_at`` from the
            claim transaction)
        :ptype scheduled_fire_at: datetime | None
        :param actual_fired_at: the actual tick instant
        :ptype actual_fired_at: datetime
        :param fire_source: what started the fire: ``scheduled_tick``, ``webhook`` or ``manual``
        :ptype fire_source: str
        :param execution_mode: execution mode (not stored on the fire row)
        :ptype execution_mode: str
        :param requested_by: who asked for a manual fire; ``None`` on every other fire
        :ptype requested_by: str | None
        :return: nothing
        :rtype: None
        """
        del execution_mode
        if self.l3_pool is None:
            return None
        # cache-bypass: INSERT of a fresh fire_id, which no cache can hold. No code reads a
        # fire row by primary key (see the module docstring), so no tier needs settling.
        await self.l3_pool.execute(
            "INSERT INTO wake_fires "
            "(conversation_id, fire_id, schedule_id, webhook_subscription_id, "
            " scheduled_fire_at, actual_fired_at, status, display_suppressed, fire_source, requested_by) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
            conversation_id,
            fire_id,
            schedule_id,
            webhook_subscription_id,
            scheduled_fire_at,
            actual_fired_at,
            "dispatching",
            False,
            fire_source,
            requested_by,
        )
        return None

    async def link_started_conversation(
        self,
        conversation_id: UUID,
        fire_id: UUID,
        started_conversation_id: UUID,
        *,
        conn: Any,
    ) -> bool:
        """Record the conversation a fire started.

        Runs on the connection, and inside the transaction, that created the
        conversation, so the conversation and its link commit together or not
        at all. It happens before the model runs: a fire the reaper later
        fails is still linked to what it started.

        :param conversation_id: partition column (the wake's conversation)
        :ptype conversation_id: UUID
        :param fire_id: the fire
        :ptype fire_id: UUID
        :param started_conversation_id: the conversation the fire started
        :ptype started_conversation_id: UUID
        :param conn: the transaction's connection
        :ptype conn: Any
        :return: ``True`` when the fire was still in flight and is now linked
        :rtype: bool
        """
        # cache-bypass: a targeted UPDATE inside the caller's transaction. No code reads a fire row
        # by primary key -- every fire reader queries L3 (see the module docstring) -- so no cached
        # copy of this row is ever served and none is evicted.
        linked = await update_keyed(
            conn,
            _FIRES,
            values={"started_conversation_id": started_conversation_id},
            by={"conversation_id": conversation_id, "fire_id": fire_id, "status": "dispatching"},
            returning=["fire_id"],
        )
        return bool(linked.count)

    async def finalize_success(
        self,
        conversation_id: UUID,
        fire_id: UUID,
        *,
        status: str = "fired",
        output_text: str | None = None,
        latency_ms: int | None = None,
        display_suppressed: bool = False,
    ) -> None:
        """Stamp a successful dispatch result onto the fire row.

        Called after the handler returns. Changes only a row still
        ``'dispatching'``: a fire the reaper already failed stays failed, so
        the fire row and whatever the consumer recorded when the reaper told
        it the fire was lost cannot disagree.

        Takes ``conversation_id`` first (partition column) so the SQL
        carries the partition predicate -- pinned by
        ``test_partition_column_enforcement.py``.

        :param conversation_id: partition column for ``wake_fires``
        :ptype conversation_id: UUID
        :param fire_id: target fire row
        :ptype fire_id: UUID
        :param status: terminal status -- one of ``'fired'``,
            ``'fired_silent'``, ``'yielded'``, ``'checked_quiet'``, or a ``'skipped_*'``
        :ptype status: str
        :param output_text: captured assistant output text
        :ptype output_text: str | None
        :param latency_ms: end-to-end fire latency
        :ptype latency_ms: int | None
        :param display_suppressed: whether visible display was
            suppressed (``[SILENT]`` path)
        :ptype display_suppressed: bool
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return None
        # cache-bypass: targeted UPDATE on the terminal-state columns. No code reads a fire row
        # by primary key -- every fire reader queries L3 (see the module docstring) -- so no
        # cached copy of this row is ever served and none is evicted.
        await update_keyed(
            self.l3_pool,
            _FIRES,
            values={
                "status": status,
                "output_text": output_text,
                "latency_ms": latency_ms,
                "display_suppressed": display_suppressed,
            },
            by={"conversation_id": conversation_id, "fire_id": fire_id, "status": "dispatching"},
        )
        return None

    async def finalize_failed(
        self,
        conversation_id: UUID,
        fire_id: UUID,
        *,
        error: str,
        latency_ms: int | None = None,
    ) -> None:
        """Stamp a failed-dispatch result onto the fire row.

        Called by the tick body when the dispatch callback raises -- the
        per-schedule try/except keeps one bad fire from poisoning the rest of
        the tick. Changes only a row still ``'dispatching'``, as
        :meth:`finalize_success` does.

        Takes ``conversation_id`` first (partition column) so the SQL
        carries the partition predicate -- pinned by
        ``test_partition_column_enforcement.py``.

        :param conversation_id: partition column for ``wake_fires``
        :ptype conversation_id: UUID
        :param fire_id: target fire row
        :ptype fire_id: UUID
        :param error: captured error message (truncated by the DB if
            needed)
        :ptype error: str
        :param latency_ms: latency up to the failure (may be ``None``
            if the failure happened pre-LLM)
        :ptype latency_ms: int | None
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return None
        # cache-bypass: targeted UPDATE on the failure columns. No code reads a fire row by
        # primary key -- every fire reader queries L3 (see the module docstring) -- so no cached
        # copy of this row is ever served and none is evicted.
        await update_keyed(
            self.l3_pool,
            _FIRES,
            values={"status": "failed", "error": error, "latency_ms": latency_ms},
            by={"conversation_id": conversation_id, "fire_id": fire_id, "status": "dispatching"},
        )
        return None

    async def reap_stale_dispatching(
        self,
        now: datetime,
        *,
        older_than: timedelta,
    ) -> list[ReapedFire]:
        """Reap ``'dispatching'`` ``wake_fires`` rows abandoned mid-dispatch.

        Satisfies :meth:`FireStore.reap_stale_dispatching
        <threetears.scheduled_jobs.protocols.FireStore.reap_stale_dispatching>`
        so the shared tick engine's per-tick sweep reclaims wake fires a
        crashed pod left in ``'dispatching'`` (the schedule already
        advanced, so the occurrence never re-fires). Cross-partition
        UPDATE to ``'failed'`` with the shared reaper marker.

        :param now: sweep instant; the cutoff is ``now - older_than``
        :ptype now: datetime
        :param older_than: minimum in-flight age before a row is reaped
        :ptype older_than: timedelta
        :return: each reaped fire, with the conversation it started (if any)
        :rtype: list[ReapedFire]
        """
        if self.l3_pool is None:
            return []
        cutoff = now - older_than
        # __SPANS_PARTITIONS__: reclaiming abandoned in-flight fires is a
        # global sweep across every conversation; the partition predicate
        # cannot apply by construction. ``conversation_id`` is named in the
        # RETURNING clause so the partition-column enforcement walker sees
        # it as a static literal.
        # cache-bypass: bulk terminal-state UPDATE across partitions. No code reads a fire row
        # by primary key -- every fire reader queries L3 (see the module docstring) -- so no
        # cached copy of a reaped row is ever served and none is evicted.
        reaped = await update_keyed(
            self.l3_pool,
            _FIRES,
            values={"status": "failed", "error": REAPED_DISPATCH_ERROR},
            by={"status": "dispatching"},
            where=[Fragment('"actual_fired_at" < $1', (cutoff,))],
            returning=_REAPED_COLUMNS,
        )
        rows = reaped.rows
        return [
            ReapedFire(
                conversation_id=row["conversation_id"],
                fire_id=row["fire_id"],
                started_conversation_id=row["started_conversation_id"],
            )
            for row in rows
        ]

    async def reap_fires(self, fire_ids: Sequence[UUID]) -> list[ReapedFire]:
        """Fail the named fires that are still ``'dispatching'``: their dispatcher is known to be gone.

        :meth:`reap_stale_dispatching` reaps by age, because it cannot know who
        was dispatching a fire. A consumer that does know (it keeps a record of
        which process holds each fire, and that process stopped beating) names
        the fires here and does not wait out the age. Same terminal state and
        marker as the age sweep, and the same guard: a fire that finished is
        left as it ended.

        :param fire_ids: the fires whose dispatcher is gone
        :ptype fire_ids: Sequence[UUID]
        :return: each fire this call failed, with the conversation it started (if any)
        :rtype: list[ReapedFire]
        """
        if self.l3_pool is None or not fire_ids:
            return []
        # __SPANS_PARTITIONS__: the caller names fires across conversations;
        # ``conversation_id`` is named in the RETURNING clause so the
        # partition-column enforcement walker sees it as a static literal.
        # cache-bypass: see reap_stale_dispatching -- no code reads a fire row by primary key.
        reaped = await update_keyed(
            self.l3_pool,
            _FIRES,
            values={"status": "failed", "error": REAPED_DISPATCH_ERROR},
            keys=KeyBatch(("fire_id",), [(fire_id,) for fire_id in fire_ids], ("uuid",)),
            by={"status": "dispatching"},
            returning=_REAPED_COLUMNS,
        )
        rows = reaped.rows
        return [
            ReapedFire(
                conversation_id=row["conversation_id"],
                fire_id=row["fire_id"],
                started_conversation_id=row["started_conversation_id"],
            )
            for row in rows
        ]

    async def count_in_window(
        self,
        conversation_id: UUID,
        *,
        since: datetime,
        webhook_subscription_id: UUID | None = None,
    ) -> int:
        """Count fires for a conversation since ``since``.

        Used by the per-conv rate-limit primitive in shard 05. Hits
        ``idx_wake_fires_conv_time`` via the ``conversation_id``
        predicate; ``webhook_subscription_id``, when supplied, narrows
        the same scan to one webhook subscription server-side. That
        narrowing is the webhook receiver's per-subscription cap, which
        previously re-rolled this COUNT against a bare pool outside the
        Collection.

        :param conversation_id: partition column
        :ptype conversation_id: UUID
        :param since: lower bound on ``actual_fired_at``
        :ptype since: datetime
        :param webhook_subscription_id: when set, count only fires this
            webhook subscription produced
        :ptype webhook_subscription_id: UUID | None
        :return: count of fires in the window
        :rtype: int
        """
        if self.l3_pool is None:
            return 0
        # cache-bypass: aggregate COUNT not pk-addressable.
        by: dict[str, Any] = (
            {} if webhook_subscription_id is None else {"webhook_subscription_id": webhook_subscription_id}
        )
        counted = await counts_by(
            self.l3_pool,
            _FIRES,
            column="conversation_id",
            values=AnyOf([conversation_id], "uuid"),
            by=by,
            where=[Fragment('"actual_fired_at" >= $1', (since,))],
        )
        return counted.get(conversation_id, 0)


class WebhookSubscriptionCollection(BaseCollection[WebhookSubscriptionEntity]):
    """Three-tier collection for :class:`WebhookSubscriptionEntity`.

    Composite primary key ``(conversation_id, subscription_id)``;
    partition column ``conversation_id``. ``UNIQUE (subscription_id)``
    standalone constraint lets cross-package FKs (notably
    ``wake_fires.webhook_subscription_id``) reference the bare id.

    The collection does not own encryption: the consumer's
    ``EncryptionService`` is passed at the API surface (shard 04) and
    the resulting ``secret_ciphertext`` bytes land here unchanged.
    """

    tenancy = TENANT_BY_SCHEMA

    primary_key_column: str | tuple[str, ...] = (
        "conversation_id",
        "subscription_id",
    )

    partition_column: ClassVar[str] = "conversation_id"

    @property
    def table_name(self) -> str:
        """Return the L3 table name."""
        return "webhook_subscriptions"

    @property
    def entity_class(self) -> type[WebhookSubscriptionEntity]:
        """Return the entity class."""
        return WebhookSubscriptionEntity

    # --- BaseCollection contract ---

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """Fetch a row by composite pk."""
        if self.l3_pool is None:
            return None
        conv_id, subscription_id = self.normalize_pk(entity_id)
        return await _read_one(
            self.l3_pool,
            _SUBSCRIPTIONS,
            _SUBSCRIPTION_INSERT_COLUMNS,
            "subscription_id",
            subscription_id,
            {"conversation_id": conv_id},
        )

    async def save_to_store(
        self,
        data: dict[str, Any],
        original_timestamp: Any = None,
        *,
        conn: Any = None,
    ) -> int:
        """Insert a new subscription row, or update an existing one only as it was read.

        With ``original_timestamp`` the write is an update-only statement fenced on the
        ``date_updated`` the row was read with (see :meth:`WakeScheduleCollection.save_to_store`).
        The webhook tools save the whole row they read, secret included; a rotation, a pause, or
        a fire stamping ``last_fired_at`` between that read and this write moves ``date_updated``,
        so the save updates nothing and ``save_entity`` raises
        :class:`~threetears.core.exceptions.ConcurrentModificationError` rather than restoring
        the secret the rotation replaced. Without it (a new row) the unfenced upsert runs.

        :param data: row dict keyed by column name; must carry both pk
            columns and every non-nullable column (including
            ``secret_ciphertext`` bytes)
        :ptype data: dict[str, Any]
        :param original_timestamp: the ``date_updated`` the row was read with, or ``None`` for an
            insert
        :ptype original_timestamp: Any
        :param conn: optional asyncpg-compatible connection
        :ptype conn: Any
        :return: rows affected (1 on success, 0 when the row changed or was deleted since it was
            read)
        :rtype: int
        """
        target = conn if conn is not None else self.l3_pool
        if target is None:
            return 0
        if original_timestamp is None:
            await target.execute(_WEBHOOK_SUBSCRIPTIONS_UPSERT_SQL, *_subscription_insert_params(data))
            return 1
        return await _fenced_update(
            target,
            _SUBSCRIPTIONS,
            data,
            _subscription_value_for_column,
            _SUBSCRIPTION_UPDATE_COLUMNS,
            _SUBSCRIPTION_PK_COLUMNS,
            original_timestamp,
        )

    async def delete_from_store(self, entity_id: Any) -> None:
        """Delete a subscription row by composite pk.

        The FK on ``wake_fires.webhook_subscription_id`` is ``ON
        DELETE SET NULL`` -- deleting a subscription leaves the fire
        history visible but unbound from the source.
        """
        if self.l3_pool is None:
            return None
        conv_id, subscription_id = self.normalize_pk(entity_id)
        await delete_keyed(
            self.l3_pool, _SUBSCRIPTIONS, by={"conversation_id": conv_id, "subscription_id": subscription_id}
        )
        return None

    def serialize(self, data: dict[str, Any]) -> bytes:
        """Encode a row dict for L2 (NATS KV) storage."""
        return serialize_to_json(data)

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """Decode L2-cached bytes back to a row dict."""
        return deserialize_from_json(data, _SUBSCRIPTION_FIELD_TYPES)

    # --- Domain methods ---

    async def find_by_id(
        self,
        subscription_id: UUID,
    ) -> WebhookSubscriptionEntity | None:
        """Look up a subscription by bare id (cross-partition).

        The HTTP webhook receiver (shard 06) takes a path-param
        subscription_id with no conversation context; it hits the
        standalone ``UNIQUE (subscription_id)`` to locate the row
        before doing any per-conversation work. ``conversation_id``
        is read off the returned entity afterwards.

        :param subscription_id: bare id (matches the standalone UNIQUE)
        :ptype subscription_id: UUID
        :return: subscription entity or ``None``
        :rtype: WebhookSubscriptionEntity | None
        """
        if self.l3_pool is None:
            return None
        # spans-partitions: the receiver has no conversation context
        # until it resolves the subscription row. The UNIQUE constraint
        # makes the cross-partition lookup safe + fast.
        row = await _read_one(
            self.l3_pool, _SUBSCRIPTIONS, _SUBSCRIPTION_INSERT_COLUMNS, "subscription_id", subscription_id, {}
        )
        if row is None:
            return None
        return self.entity_class(dict(row), is_new=False, collection=self)

    async def find_for_agent(
        self,
        agent_id: UUID,
        subscription_id: UUID,
    ) -> WebhookSubscriptionEntity | None:
        """Return one of an agent's subscriptions, in whichever conversation it lives.

        The agent-scoped counterpart of :meth:`find_by_id`: another agent's
        subscription is not found, so its existence does not leak.

        :param agent_id: the agent that must own the subscription
        :ptype agent_id: UUID
        :param subscription_id: the subscription
        :ptype subscription_id: UUID
        :return: the subscription, or ``None``
        :rtype: WebhookSubscriptionEntity | None
        """
        if self.l3_pool is None:
            return None
        # spans-partitions: the caller names the subscription, not its
        # conversation; the standalone UNIQUE (subscription_id) makes the
        # lookup exact and agent_id keeps it to the caller's own.
        row = await _read_one(
            self.l3_pool,
            _SUBSCRIPTIONS,
            _SUBSCRIPTION_INSERT_COLUMNS,
            "subscription_id",
            subscription_id,
            {"agent_id": agent_id},
        )
        if row is None:
            return None
        return WebhookSubscriptionEntity(dict(row), is_new=False, collection=self)

    async def list_for_agent(
        self,
        agent_id: UUID,
    ) -> list[WebhookSubscriptionEntity]:
        """Return an agent's subscriptions across all its conversations.

        Hits ``idx_webhook_subs_agent``.

        :param agent_id: the agent
        :ptype agent_id: UUID
        :return: subscriptions ordered by ``date_created`` DESC
        :rtype: list[WebhookSubscriptionEntity]
        """
        if self.l3_pool is None:
            return []
        # spans-partitions: the agent's listing covers every conversation.
        rows = await keyset_rows(
            self.l3_pool,
            _SUBSCRIPTIONS,
            columns=_SUBSCRIPTION_INSERT_COLUMNS,
            by={"agent_id": agent_id},
            order=_NEWEST_SUBSCRIPTIONS_FIRST,
        )
        return [WebhookSubscriptionEntity(dict(row), is_new=False, collection=self) for row in rows]

    async def list_for_conversation(
        self,
        conversation_id: UUID,
    ) -> list[WebhookSubscriptionEntity]:
        """Return every subscription for a conversation.

        :param conversation_id: partition column
        :ptype conversation_id: UUID
        :return: list of subscriptions ordered by ``date_created`` DESC
        :rtype: list[WebhookSubscriptionEntity]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: per-conversation scan.
        rows = await keyset_rows(
            self.l3_pool,
            _SUBSCRIPTIONS,
            columns=_SUBSCRIPTION_INSERT_COLUMNS,
            by={"conversation_id": conversation_id},
            order=_NEWEST_SUBSCRIPTIONS_FIRST,
        )
        return [self.entity_class(dict(row), is_new=False, collection=self) for row in rows]

    async def rotate_secret(
        self,
        conversation_id: UUID,
        subscription_id: UUID,
        *,
        new_ciphertext: bytes,
    ) -> None:
        """Replace the stored ``secret_ciphertext`` (rotation).

        The agent-tools / REST layer in shard 04 supplies the new
        Fernet-encrypted ciphertext; the platform does not own
        encryption. Display-once semantics for the plaintext are the
        API-layer's responsibility.

        :param conversation_id: partition column
        :ptype conversation_id: UUID
        :param subscription_id: subscription to rotate
        :ptype subscription_id: UUID
        :param new_ciphertext: new Fernet-encrypted ciphertext
        :ptype new_ciphertext: bytes
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return None
        async with self.bypassing_write((conversation_id, subscription_id)):
            # cache-bypass: targeted UPDATE on a single bytes column; the row is evicted from
            # every tier once it lands. A cached pre-rotation row would otherwise be saved back
            # by the next edit, restoring the secret this replaced.
            await update_keyed(
                self.l3_pool,
                _SUBSCRIPTIONS,
                values={"secret_ciphertext": bytes(new_ciphertext), "date_updated": _NOW},
                by={"conversation_id": conversation_id, "subscription_id": subscription_id},
            )
        return None

    async def pause(
        self,
        conversation_id: UUID,
        subscription_id: UUID,
    ) -> None:
        """Flip ``status`` to ``'paused'``. Idempotent."""
        if self.l3_pool is None:
            return None
        async with self.bypassing_write((conversation_id, subscription_id)):
            # cache-bypass: targeted UPDATE on status; the row is evicted from every tier once
            # it lands.
            await update_keyed(
                self.l3_pool,
                _SUBSCRIPTIONS,
                values={"status": "paused", "date_updated": _NOW},
                by={"conversation_id": conversation_id, "subscription_id": subscription_id},
            )
        return None

    async def resume(
        self,
        conversation_id: UUID,
        subscription_id: UUID,
    ) -> None:
        """Flip ``status`` to ``'active'``. Idempotent."""
        if self.l3_pool is None:
            return None
        async with self.bypassing_write((conversation_id, subscription_id)):
            # cache-bypass: targeted UPDATE on status; the row is evicted from every tier once
            # it lands.
            await update_keyed(
                self.l3_pool,
                _SUBSCRIPTIONS,
                values={"status": "active", "date_updated": _NOW},
                by={"conversation_id": conversation_id, "subscription_id": subscription_id},
            )
        return None

    async def record_refusal(
        self,
        conversation_id: UUID,
        subscription_id: UUID,
        *,
        reason: str,
        status_code: int,
        at: datetime,
        detail: str | None = None,
        keep: timedelta = REFUSALS_KEPT_FOR,
    ) -> None:
        """Keep one refused delivery: counted into its subscription's row for this reason and minute.

        One statement inserts the minute's row or adds one to it, so a sender
        that is refused a thousand times a minute costs one row. Rows of the
        subscription older than ``keep`` are dropped in passing.

        :param conversation_id: partition column
        :ptype conversation_id: UUID
        :param subscription_id: the subscription the delivery was for
        :ptype subscription_id: UUID
        :param reason: one of :data:`WEBHOOK_REFUSAL_REASONS`
        :ptype reason: str
        :param status_code: the HTTP status the sender was answered with
        :ptype status_code: int
        :param at: when it was refused
        :ptype at: datetime
        :param detail: a short note of why, when the reason alone does not say
        :ptype detail: str | None
        :param keep: how long a subscription's refusals are kept
        :ptype keep: timedelta
        :return: nothing
        :rtype: None
        :raises ValueError: for a reason that is not one of :data:`WEBHOOK_REFUSAL_REASONS`
        """
        if reason not in WEBHOOK_REFUSAL_REASONS:
            raise ValueError(f"{reason!r} is not a webhook refusal reason")
        if self.l3_pool is None:
            return None
        # cache-bypass: webhook_refusals has no entity and no cached row; it is a counted log.
        await self.l3_pool.execute(
            "INSERT INTO webhook_refusals "
            "(conversation_id, subscription_id, reason, window_start, status_code, refusals, last_refused_at, detail) "
            "VALUES ($1, $2, $3, date_trunc('minute', $4::timestamptz), $5, 1, $4, $6) "
            "ON CONFLICT (subscription_id, reason, window_start) DO UPDATE "
            "SET refusals = webhook_refusals.refusals + 1, last_refused_at = EXCLUDED.last_refused_at, "
            "    status_code = EXCLUDED.status_code, detail = COALESCE(EXCLUDED.detail, webhook_refusals.detail)",
            conversation_id,
            subscription_id,
            reason,
            at,
            status_code,
            str(detail)[:500] if detail else None,
        )
        # cache-bypass: as above; the prune keeps the log bounded per subscription.
        await delete_keyed(
            self.l3_pool,
            _REFUSALS,
            by={"subscription_id": subscription_id},
            where=[Fragment('"last_refused_at" < $1', (at - keep,))],
        )
        return None

    async def list_refusals(
        self,
        subscription_id: UUID,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """A subscription's refused deliveries, newest first, one row per reason and minute.

        :param subscription_id: the subscription
        :ptype subscription_id: UUID
        :param limit: page size
        :ptype limit: int
        :return: rows with ``reason``, ``status_code``, ``window_start``, ``refusals``,
            ``last_refused_at`` and ``detail``
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: a counted log with no entity.
        page = await keyset_page(
            self.l3_pool,
            _REFUSALS,
            columns=("reason", "status_code", "window_start", "refusals", "last_refused_at", "detail"),
            by={"subscription_id": subscription_id},
            order=[Sort("last_refused_at", cast="timestamptz", descending=True)],
            limit=limit,
        )
        rows = page.rows
        return [dict(row) for row in rows]

    async def record_fire(
        self,
        conversation_id: UUID,
        subscription_id: UUID,
        *,
        fired_at: datetime,
    ) -> None:
        """Stamp ``last_fired_at`` after a successful webhook fire.

        Called by the dispatcher (shard 03) post-fire. Separate from
        ``record()`` on the fire collection because the
        ``last_fired_at`` denormalisation lets the per-subscription
        rate-limit query avoid a JOIN on ``wake_fires``.

        :param conversation_id: partition column
        :ptype conversation_id: UUID
        :param subscription_id: target subscription
        :ptype subscription_id: UUID
        :param fired_at: timestamp to record
        :ptype fired_at: datetime
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return None
        async with self.bypassing_write((conversation_id, subscription_id)):
            # cache-bypass: targeted UPDATE on the timestamp column; the row is evicted from
            # every tier once it lands.
            await update_keyed(
                self.l3_pool,
                _SUBSCRIPTIONS,
                values={"last_fired_at": fired_at, "date_updated": _NOW},
                by={"conversation_id": conversation_id, "subscription_id": subscription_id},
            )
        return None


# --- helpers -----------------------------------------------------------


def _schedule_insert_params(data: dict[str, Any]) -> list[Any]:
    """Project a row dict onto the schedule upsert's positional params.

    Missing values are coerced to the schema's DEFAULT-equivalent so
    every column carries an explicit bound value (the upsert path
    writes every column unconditionally; relying on server defaults
    here would change the bound-values arity and break the SQL).

    :param data: row dict keyed by column name
    :ptype data: dict[str, Any]
    :return: positional parameter list in ``_SCHEDULE_INSERT_COLUMNS``
        order
    :rtype: list[Any]
    """
    return [_schedule_value_for_column(col, data) for col in _SCHEDULE_INSERT_COLUMNS]


def _schedule_value_for_column(col: str, data: dict[str, Any]) -> Any:
    """Return ``data[col]`` with defaults applied for columns that have a
    DB-side default.

    :param col: column name
    :ptype col: str
    :param data: row dict
    :ptype data: dict[str, Any]
    :return: value to bind for this column
    :rtype: Any
    """
    if col in data:
        value = data[col]
        if col == "schedule_config" and value is None:
            return {}
        # an entity created or cached before v007 carries no flag; the column is NOT NULL
        if col == "protected" and value is None:
            return False
        return value
    if col == "schedule_config":
        return {}
    if col == "execution_mode":
        return "spawn"
    if col == "status":
        return "active"
    if col == "missed_fire_policy":
        return "coalesce"
    if col == "include_conversation_history":
        return True
    if col == "protected":
        return False
    return None


def _fire_insert_params(data: dict[str, Any]) -> list[Any]:
    """Project a row dict onto the fire upsert's positional params.

    :param data: row dict keyed by column name
    :ptype data: dict[str, Any]
    :return: positional parameter list in ``_FIRE_INSERT_COLUMNS``
        order
    :rtype: list[Any]
    """
    return [_fire_value_for_column(col, data) for col in _FIRE_INSERT_COLUMNS]


def _fire_value_for_column(col: str, data: dict[str, Any]) -> Any:
    """Return ``data[col]`` with defaults for omittable columns.

    :param col: column name
    :ptype col: str
    :param data: row dict
    :ptype data: dict[str, Any]
    :return: value to bind for this column
    :rtype: Any
    """
    if col in data:
        return data[col]
    if col == "display_suppressed":
        return False
    if col == "fire_source":
        # NOT NULL: a row built without one is what its source column says it is
        return "webhook" if data.get("webhook_subscription_id") is not None else "scheduled_tick"
    return None


def _subscription_insert_params(data: dict[str, Any]) -> list[Any]:
    """Project a row dict onto the subscription upsert's positional params.

    :param data: row dict keyed by column name
    :ptype data: dict[str, Any]
    :return: positional parameter list in
        ``_SUBSCRIPTION_INSERT_COLUMNS`` order
    :rtype: list[Any]
    """
    return [_subscription_value_for_column(col, data) for col in _SUBSCRIPTION_INSERT_COLUMNS]


def _subscription_value_for_column(col: str, data: dict[str, Any]) -> Any:
    """Return ``data[col]`` with defaults applied for omittable columns.

    :param col: column name
    :ptype col: str
    :param data: row dict
    :ptype data: dict[str, Any]
    :return: value to bind for this column
    :rtype: Any
    """
    if col in data:
        value = data[col]
        if col == "secret_ciphertext" and value is not None and not isinstance(value, bytes):
            return bytes(value)
        return value
    if col == "execution_mode":
        return "spawn"
    if col == "verification_scheme":
        return "generic_hmac_sha256"
    if col == "status":
        return "active"
    return None
