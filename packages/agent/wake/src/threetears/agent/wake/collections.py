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
from datetime import datetime
from typing import Any, ClassVar
from uuid import UUID

from threetears.agent.wake.entities import (
    WakeFireEntity,
    WakeScheduleEntity,
    WebhookSubscriptionEntity,
)
from threetears.core.collections.base import BaseCollection
from threetears.core.serialization import (
    deserialize_from_json,
    serialize_to_json,
)
from threetears.observe import get_logger

__all__ = [
    "WakeFireCollection",
    "WakeScheduleCollection",
    "WebhookSubscriptionCollection",
]


log = get_logger(__name__)


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
    "date_created": datetime,
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
    "date_created",
    "date_updated",
)


# columns updated on ON CONFLICT (partition + pk excluded;
# ``date_created`` immutable).
_SCHEDULE_UPDATE_COLUMNS: tuple[str, ...] = tuple(
    c for c in _SCHEDULE_INSERT_COLUMNS if c not in {"conversation_id", "schedule_id", "date_created"}
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
    "date_created",
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


_AGENT_WAKE_SCHEDULES_UPSERT_SQL = _build_upsert_sql(
    "agent_wake_schedules",
    _SCHEDULE_INSERT_COLUMNS,
    _SCHEDULE_UPDATE_COLUMNS,
    ("conversation_id", "schedule_id"),
)


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


_AGENT_WAKE_SCHEDULES_FETCH_SQL = (
    "SELECT conversation_id, schedule_id, user_id, agent_id, skill_id, "
    "schedule_type, schedule_config, task_prompt, execution_mode, status, "
    "next_fire_at, last_fired_at, name, missed_fire_policy, "
    "context_from_schedule_id, include_conversation_history, "
    "date_created, date_updated "
    "FROM agent_wake_schedules WHERE conversation_id = $1 AND schedule_id = $2"
)


_AGENT_WAKE_SCHEDULES_DELETE_SQL = "DELETE FROM agent_wake_schedules WHERE conversation_id = $1 AND schedule_id = $2"


_WAKE_FIRES_FETCH_SQL = (
    "SELECT conversation_id, fire_id, schedule_id, webhook_subscription_id, "
    "scheduled_fire_at, actual_fired_at, status, display_suppressed, "
    "output_text, latency_ms, error, date_created "
    "FROM wake_fires WHERE conversation_id = $1 AND fire_id = $2"
)


_WAKE_FIRES_DELETE_SQL = "DELETE FROM wake_fires WHERE conversation_id = $1 AND fire_id = $2"


_WEBHOOK_SUBSCRIPTIONS_FETCH_SQL = (
    "SELECT conversation_id, subscription_id, user_id, agent_id, "
    "default_skill_id, name, secret_ciphertext, allowed_source_pattern, "
    "execution_mode, task_prompt_template, "
    "verification_scheme, status, rate_limit_per_minute, last_fired_at, "
    "date_created, date_updated "
    "FROM webhook_subscriptions WHERE conversation_id = $1 AND subscription_id = $2"
)


_WEBHOOK_SUBSCRIPTIONS_DELETE_SQL = (
    "DELETE FROM webhook_subscriptions WHERE conversation_id = $1 AND subscription_id = $2"
)


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
        row = await self.l3_pool.fetchrow(
            _AGENT_WAKE_SCHEDULES_FETCH_SQL,
            conv_id,
            schedule_id,
        )
        if row is None:
            return None
        return dict(row)

    async def save_to_store(
        self,
        data: dict[str, Any],
        original_timestamp: Any = None,
        *,
        conn: Any = None,
    ) -> int:
        """Upsert a schedule row.

        :param data: row dict keyed by column name; must carry both pk
            columns and every non-nullable column
        :ptype data: dict[str, Any]
        :param original_timestamp: ignored (no CAS fence -- schedule
            updates are idempotent in the upsert path)
        :ptype original_timestamp: Any
        :param conn: optional asyncpg-compatible connection
        :ptype conn: Any
        :return: rows affected (1 on success)
        :rtype: int
        """
        del original_timestamp
        params = _schedule_insert_params(data)
        target = conn if conn is not None else self.l3_pool
        if target is None:
            return 0
        await target.execute(_AGENT_WAKE_SCHEDULES_UPSERT_SQL, *params)
        return 1

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
        await self.l3_pool.execute(
            _AGENT_WAKE_SCHEDULES_DELETE_SQL,
            conv_id,
            schedule_id,
        )
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
        rows = await self.l3_pool.fetch(
            "SELECT conversation_id, schedule_id, user_id, agent_id, skill_id, "
            "schedule_type, schedule_config, task_prompt, execution_mode, status, "
            "next_fire_at, last_fired_at, name, missed_fire_policy, "
            "context_from_schedule_id, include_conversation_history, "
            "date_created, date_updated "
            "FROM agent_wake_schedules "
            "WHERE status = 'active' AND next_fire_at IS NOT NULL AND next_fire_at <= $1 "
            "ORDER BY next_fire_at ASC LIMIT $2",
            now,
            limit,
        )
        return [WakeScheduleEntity(dict(row), is_new=False, collection=self) for row in rows]

    async def list_active_for_conversation(
        self,
        conversation_id: UUID,
    ) -> list[WakeScheduleEntity]:
        """Return active schedules for a conversation.

        Powers per-conv cap enforcement (PLACEMENT §1.9) plus the
        "what is this conversation waking on" admin view. Hits the
        ``idx_wake_schedules_conv_status`` partial index.

        :param conversation_id: partition column
        :ptype conversation_id: UUID
        :return: list of active schedules
        :rtype: list[WakeScheduleEntity]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: multi-row scan by conversation_id is not pk-
        # addressable; L1 row cache cannot serve.
        rows = await self.l3_pool.fetch(
            "SELECT conversation_id, schedule_id, user_id, agent_id, skill_id, "
            "schedule_type, schedule_config, task_prompt, execution_mode, status, "
            "next_fire_at, last_fired_at, name, missed_fire_policy, "
            "context_from_schedule_id, include_conversation_history, "
            "date_created, date_updated "
            "FROM agent_wake_schedules "
            "WHERE conversation_id = $1 AND status = 'active' "
            "ORDER BY next_fire_at ASC NULLS LAST",
            conversation_id,
        )
        return [WakeScheduleEntity(dict(row), is_new=False, collection=self) for row in rows]

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
        rows = await self.l3_pool.fetch(
            "SELECT conversation_id, schedule_id, user_id, agent_id, skill_id, "
            "schedule_type, schedule_config, task_prompt, execution_mode, status, "
            "next_fire_at, last_fired_at, name, missed_fire_policy, "
            "context_from_schedule_id, include_conversation_history, "
            "date_created, date_updated "
            "FROM agent_wake_schedules "
            "WHERE conversation_id = $1 "
            "ORDER BY date_created DESC",
            conversation_id,
        )
        return [WakeScheduleEntity(dict(row), is_new=False, collection=self) for row in rows]

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
        value = await self.l3_pool.fetchval(
            "SELECT COUNT(*) FROM agent_wake_schedules WHERE conversation_id = $1 AND status = 'active'",
            conversation_id,
        )
        return int(value or 0)

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
        if last_fired_at is None:
            # cache-bypass: targeted UPDATE on the next_fire_at column.
            await self.l3_pool.execute(
                "UPDATE agent_wake_schedules "
                "SET next_fire_at = $3, date_updated = now() "
                "WHERE conversation_id = $1 AND schedule_id = $2",
                conversation_id,
                schedule_id,
                next_fire_at,
            )
        else:
            # cache-bypass: targeted UPDATE with both timestamps.
            await self.l3_pool.execute(
                "UPDATE agent_wake_schedules "
                "SET next_fire_at = $3, last_fired_at = $4, date_updated = now() "
                "WHERE conversation_id = $1 AND schedule_id = $2",
                conversation_id,
                schedule_id,
                next_fire_at,
                last_fired_at,
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
        # cache-bypass: targeted UPDATE on status column.
        await self.l3_pool.execute(
            "UPDATE agent_wake_schedules "
            "SET status = 'paused', next_fire_at = NULL, date_updated = now() "
            "WHERE conversation_id = $1 AND schedule_id = $2 AND status = 'active'",
            conversation_id,
            schedule_id,
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
        does not own scheduling math; shard 02's ``_compute_next_fire_at``
        provides it). Idempotent: replay on an already-active schedule
        overwrites the timestamp.

        ``conn`` lets the cap-serialized resume path
        (:func:`threetears.agent.wake.rate_limit.resume_schedule_serialized`)
        bind the UPDATE to the same transaction that holds the
        per-conversation advisory lock + active-count, so the
        re-activation cap holds atomically. When ``conn`` is ``None`` the
        UPDATE runs on the pool (the direct, non-cap path).

        :param conversation_id: partition column
        :ptype conversation_id: UUID
        :param schedule_id: target schedule
        :ptype schedule_id: UUID
        :param next_fire_at: when the resumed schedule should next fire
        :ptype next_fire_at: datetime
        :param conn: optional asyncpg-compatible connection (binds the
            UPDATE to a caller-owned transaction)
        :ptype conn: Any
        :return: nothing
        :rtype: None
        """
        target = conn if conn is not None else self.l3_pool
        if target is None:
            return None
        # cache-bypass: targeted UPDATE flipping paused -> active.
        await target.execute(
            "UPDATE agent_wake_schedules "
            "SET status = 'active', next_fire_at = $3, date_updated = now() "
            "WHERE conversation_id = $1 AND schedule_id = $2 AND status != 'expired'",
            conversation_id,
            schedule_id,
            next_fire_at,
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
        # cache-bypass: atomic CAS UPDATE; the row cache is invalidated
        # naturally on the next read via the partition-aware fetch
        # path.
        claimed = await self.l3_pool.fetchval(
            "UPDATE agent_wake_schedules "
            "SET next_fire_at = $1, last_fired_at = $2, date_updated = $2, status = $3 "
            "WHERE conversation_id = $4 AND schedule_id = $5 AND next_fire_at = $6 "
            "RETURNING schedule_id",
            computed_next_fire,
            now,
            new_status,
            conversation_id,
            schedule_id,
            expected_next_fire,
        )
        return claimed is not None

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
        # cache-bypass: targeted UPDATE on status column.
        await self.l3_pool.execute(
            "UPDATE agent_wake_schedules "
            "SET status = 'expired', next_fire_at = NULL, date_updated = now() "
            "WHERE conversation_id = $1 AND schedule_id = $2",
            conversation_id,
            schedule_id,
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
        row = await self.l3_pool.fetchrow(
            _WAKE_FIRES_FETCH_SQL,
            conv_id,
            fire_id,
        )
        if row is None:
            return None
        return dict(row)

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
        :param original_timestamp: ignored
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
        await self.l3_pool.execute(
            _WAKE_FIRES_DELETE_SQL,
            conv_id,
            fire_id,
        )
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
        rows = await self.l3_pool.fetch(
            "SELECT conversation_id, fire_id, schedule_id, webhook_subscription_id, "
            "scheduled_fire_at, actual_fired_at, status, display_suppressed, "
            "output_text, latency_ms, error, date_created "
            "FROM wake_fires "
            "WHERE conversation_id = $1 AND schedule_id = $2 "
            "ORDER BY actual_fired_at DESC LIMIT $3",
            conversation_id,
            schedule_id,
            limit,
        )
        return [WakeFireEntity(dict(row), is_new=False, collection=self) for row in rows]

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
        rows = await self.l3_pool.fetch(
            "SELECT conversation_id, fire_id, schedule_id, webhook_subscription_id, "
            "scheduled_fire_at, actual_fired_at, status, display_suppressed, "
            "output_text, latency_ms, error, date_created "
            "FROM wake_fires "
            "WHERE conversation_id = $1 "
            "ORDER BY actual_fired_at DESC LIMIT $2",
            conversation_id,
            limit,
        )
        return [WakeFireEntity(dict(row), is_new=False, collection=self) for row in rows]

    async def latest_for_schedule(
        self,
        conversation_id: UUID,
        schedule_id: UUID,
    ) -> WakeFireEntity | None:
        """Return the most recent fire for a schedule, or ``None``.

        Shorthand for ``list_for_schedule(..., limit=1)[0]`` -- the
        ``context_from`` resolver in shard 03 calls this in a hot path,
        so a dedicated method avoids constructing a throwaway list.

        :param conversation_id: partition column
        :ptype conversation_id: UUID
        :param schedule_id: schedule whose latest fire to read
        :ptype schedule_id: UUID
        :return: latest fire entity or ``None``
        :rtype: WakeFireEntity | None
        """
        if self.l3_pool is None:
            return None
        # cache-bypass: lookup by (conv_id, schedule_id) is not pk-
        # addressable for the row cache (the pk is (conv_id, fire_id)).
        row = await self.l3_pool.fetchrow(
            "SELECT conversation_id, fire_id, schedule_id, webhook_subscription_id, "
            "scheduled_fire_at, actual_fired_at, status, display_suppressed, "
            "output_text, latency_ms, error, date_created "
            "FROM wake_fires "
            "WHERE conversation_id = $1 AND schedule_id = $2 "
            "ORDER BY actual_fired_at DESC LIMIT 1",
            conversation_id,
            schedule_id,
        )
        if row is None:
            return None
        return WakeFireEntity(dict(row), is_new=False, collection=self)

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

        ``execution_mode`` / ``fire_source`` are not stored on the v1
        ``wake_fires`` row (they live on the schedule); they are
        accepted here for callsite-symmetry with the trigger envelope
        and so the platform can promote them to fire columns later
        without churning every callsite.

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
        :param fire_source: source label (reserved for v2)
        :ptype fire_source: str
        :param execution_mode: execution mode (reserved for v2)
        :ptype execution_mode: str
        :return: nothing
        :rtype: None
        """
        del fire_source, execution_mode
        if self.l3_pool is None:
            return None
        # cache-bypass: write-path; the row cache is read-mostly and
        # invalidated naturally on the next fetch.
        await self.l3_pool.execute(
            "INSERT INTO wake_fires "
            "(conversation_id, fire_id, schedule_id, webhook_subscription_id, "
            " scheduled_fire_at, actual_fired_at, status, display_suppressed) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
            conversation_id,
            fire_id,
            schedule_id,
            webhook_subscription_id,
            scheduled_fire_at,
            actual_fired_at,
            "dispatching",
            False,
        )
        return None

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

        Called by the dispatch callback in shard 03 after the LLM /
        handler returns. Idempotent: replaying the same finalization
        on a row already in the terminal state overwrites with the
        same values.

        Takes ``conversation_id`` first (partition column) so the SQL
        carries the partition predicate -- pinned by
        ``test_partition_column_enforcement.py``.

        :param conversation_id: partition column for ``wake_fires``
        :ptype conversation_id: UUID
        :param fire_id: target fire row
        :ptype fire_id: UUID
        :param status: terminal status -- one of ``'fired'``,
            ``'fired_silent'``, ``'yielded'``, or a ``'skipped_*'``
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
        # cache-bypass: targeted UPDATE on the terminal-state columns.
        await self.l3_pool.execute(
            "UPDATE wake_fires "
            "SET status = $1, output_text = $2, latency_ms = $3, "
            "    display_suppressed = $4 "
            "WHERE conversation_id = $5 AND fire_id = $6",
            status,
            output_text,
            latency_ms,
            display_suppressed,
            conversation_id,
            fire_id,
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

        Called by the tick body in shard 02 when the dispatch callback
        raises -- the per-schedule try/except keeps one bad fire from
        poisoning the rest of the tick.

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
        # cache-bypass: targeted UPDATE on the failure columns.
        await self.l3_pool.execute(
            "UPDATE wake_fires SET status = 'failed', error = $1, latency_ms = $2 "
            "WHERE conversation_id = $3 AND fire_id = $4",
            error,
            latency_ms,
            conversation_id,
            fire_id,
        )
        return None

    async def count_in_window(
        self,
        conversation_id: UUID,
        *,
        since: datetime,
    ) -> int:
        """Count fires for a conversation since ``since``.

        Used by the per-conv rate-limit primitive in shard 05. Hits
        ``idx_wake_fires_conv_time``.

        :param conversation_id: partition column
        :ptype conversation_id: UUID
        :param since: lower bound on ``actual_fired_at``
        :ptype since: datetime
        :return: count of fires in the window
        :rtype: int
        """
        if self.l3_pool is None:
            return 0
        # cache-bypass: aggregate COUNT not pk-addressable.
        value = await self.l3_pool.fetchval(
            "SELECT COUNT(*) FROM wake_fires WHERE conversation_id = $1 AND actual_fired_at >= $2",
            conversation_id,
            since,
        )
        return int(value or 0)


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
        row = await self.l3_pool.fetchrow(
            _WEBHOOK_SUBSCRIPTIONS_FETCH_SQL,
            conv_id,
            subscription_id,
        )
        if row is None:
            return None
        return dict(row)

    async def save_to_store(
        self,
        data: dict[str, Any],
        original_timestamp: Any = None,
        *,
        conn: Any = None,
    ) -> int:
        """Upsert a subscription row.

        :param data: row dict keyed by column name; must carry both pk
            columns and every non-nullable column (including
            ``secret_ciphertext`` bytes)
        :ptype data: dict[str, Any]
        :param original_timestamp: ignored
        :ptype original_timestamp: Any
        :param conn: optional asyncpg-compatible connection
        :ptype conn: Any
        :return: rows affected
        :rtype: int
        """
        del original_timestamp
        params = _subscription_insert_params(data)
        target = conn if conn is not None else self.l3_pool
        if target is None:
            return 0
        await target.execute(_WEBHOOK_SUBSCRIPTIONS_UPSERT_SQL, *params)
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        """Delete a subscription row by composite pk.

        The FK on ``wake_fires.webhook_subscription_id`` is ``ON
        DELETE SET NULL`` -- deleting a subscription leaves the fire
        history visible but unbound from the source.
        """
        if self.l3_pool is None:
            return None
        conv_id, subscription_id = self.normalize_pk(entity_id)
        await self.l3_pool.execute(
            _WEBHOOK_SUBSCRIPTIONS_DELETE_SQL,
            conv_id,
            subscription_id,
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
        row = await self.l3_pool.fetchrow(
            "SELECT conversation_id, subscription_id, user_id, agent_id, "
            "default_skill_id, name, secret_ciphertext, allowed_source_pattern, "
            "execution_mode, task_prompt_template, "
            "verification_scheme, status, rate_limit_per_minute, last_fired_at, "
            "date_created, date_updated "
            "FROM webhook_subscriptions WHERE subscription_id = $1",
            subscription_id,
        )
        if row is None:
            return None
        return WebhookSubscriptionEntity(dict(row), is_new=False, collection=self)

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
        rows = await self.l3_pool.fetch(
            "SELECT conversation_id, subscription_id, user_id, agent_id, "
            "default_skill_id, name, secret_ciphertext, allowed_source_pattern, "
            "execution_mode, task_prompt_template, "
            "verification_scheme, status, rate_limit_per_minute, last_fired_at, "
            "date_created, date_updated "
            "FROM webhook_subscriptions WHERE conversation_id = $1 "
            "ORDER BY date_created DESC",
            conversation_id,
        )
        return [WebhookSubscriptionEntity(dict(row), is_new=False, collection=self) for row in rows]

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
        # cache-bypass: targeted UPDATE on a single bytes column.
        await self.l3_pool.execute(
            "UPDATE webhook_subscriptions "
            "SET secret_ciphertext = $3, date_updated = now() "
            "WHERE conversation_id = $1 AND subscription_id = $2",
            conversation_id,
            subscription_id,
            bytes(new_ciphertext),
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
        # cache-bypass: targeted UPDATE on status.
        await self.l3_pool.execute(
            "UPDATE webhook_subscriptions "
            "SET status = 'paused', date_updated = now() "
            "WHERE conversation_id = $1 AND subscription_id = $2",
            conversation_id,
            subscription_id,
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
        # cache-bypass: targeted UPDATE on status.
        await self.l3_pool.execute(
            "UPDATE webhook_subscriptions "
            "SET status = 'active', date_updated = now() "
            "WHERE conversation_id = $1 AND subscription_id = $2",
            conversation_id,
            subscription_id,
        )
        return None

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
        # cache-bypass: targeted UPDATE on the timestamp column.
        await self.l3_pool.execute(
            "UPDATE webhook_subscriptions "
            "SET last_fired_at = $3, date_updated = now() "
            "WHERE conversation_id = $1 AND subscription_id = $2",
            conversation_id,
            subscription_id,
            fired_at,
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
        return value
    if col == "schedule_config":
        return {}
    if col == "execution_mode":
        return "inline"
    if col == "status":
        return "active"
    if col == "missed_fire_policy":
        return "coalesce"
    if col == "include_conversation_history":
        return True
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
        return "inline"
    if col == "verification_scheme":
        return "generic_hmac_sha256"
    if col == "status":
        return "active"
    return None
