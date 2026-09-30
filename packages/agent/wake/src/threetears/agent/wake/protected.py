"""The two ways a protected wake changes: its schedule, and its deletion with its agent.

A protected wake (``agent_wake_schedules.protected``) cannot be deleted,
paused, expired or retyped. The table's trigger (wake v007) refuses those
from any code path, so the protection does not depend on every caller
remembering a guard. These two functions are the only doors, and each opens
the trigger's gate for its own transaction only:

- :func:`update_protected` changes the schedule's ``schedule_config`` (an
  interval's length, say), after validating it against the wake's type,
  and moves ``next_fire_at`` to match.
- :func:`delete_protected` deletes it. It exists for deleting the agent the
  wake belongs to, and for nothing else.

The gate is ``set_config('threetears.wake_protected_gate', ..., true)``,
which PostgreSQL resets when the transaction ends.

Each write runs in a :class:`~threetears.core.collections.CallerTransaction`
through :meth:`~threetears.core.collections.BaseCollection.bypassing_write`, so
the row is evicted from L1 and L2 on every replica once the transaction has
ended -- never before the commit, when a reader could re-cache the old row
from L3 with nothing left to evict it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from threetears.core.collections import CallerTransaction
from threetears.observe import get_logger
from threetears.scheduled_jobs import compute_next_fire_at

from threetears.agent.wake.collections import WakeScheduleCollection
from threetears.agent.wake.entities import WakeScheduleEntity
from threetears.agent.wake.migrations import PROTECTED_GATE_SETTING
from threetears.agent.wake.tools.validators import validate_schedule_config

__all__ = [
    "ProtectedWakeError",
    "delete_protected",
    "update_protected",
]

log = get_logger(__name__)

_OPEN_GATE_SQL = "SELECT set_config($1, $2, true)"

_UPDATE_SQL = (
    "UPDATE agent_wake_schedules SET schedule_config = $3, next_fire_at = $4, date_updated = now() "
    "WHERE conversation_id = $1 AND schedule_id = $2 AND protected"
)

_DELETE_SQL = "DELETE FROM agent_wake_schedules WHERE conversation_id = $1 AND schedule_id = $2 AND protected"


class ProtectedWakeError(ValueError):
    """The wake is not the agent's protected wake, or the change is not valid for it."""


async def update_protected(
    *,
    collection: WakeScheduleCollection,
    agent_id: UUID,
    schedule_id: UUID,
    schedule_config: dict[str, Any],
    now: datetime | None = None,
) -> WakeScheduleEntity:
    """Change a protected wake's schedule.

    Only ``schedule_config`` changes; the type, the status and everything
    else stay. The new config is validated against the wake's type before
    anything is written, and ``next_fire_at`` is recomputed from it. The
    cached row is invalidated so the next read sees the change.

    :param collection: three-tier wake-schedules collection
    :ptype collection: WakeScheduleCollection
    :param agent_id: the agent the wake must belong to
    :ptype agent_id: UUID
    :param schedule_id: the protected wake
    :ptype schedule_id: UUID
    :param schedule_config: the new config for the wake's existing type
    :ptype schedule_config: dict[str, Any]
    :param now: the instant ``next_fire_at`` is computed from (defaults to now)
    :ptype now: datetime | None
    :return: the wake as it now stands
    :rtype: WakeScheduleEntity
    :raises ProtectedWakeError: when the agent has no such protected wake, or
        the config is not valid for its type
    """
    entity = await _protected_wake(collection, agent_id, schedule_id)
    error = validate_schedule_config(entity.schedule_type, schedule_config)
    if error is not None:
        raise ProtectedWakeError(error)
    try:
        next_fire_at = compute_next_fire_at(
            entity.schedule_type,
            schedule_config,
            entity.missed_fire_policy,
            last_fired_at=entity.last_fired_at,
            now=now or datetime.now(UTC),
        )
    except ValueError as exc:
        raise ProtectedWakeError(f"schedule_config rejected by the reschedule engine: {exc}") from exc
    conversation_id = entity.conversation_id
    async with (
        _pool(collection).acquire() as conn,
        CallerTransaction(conn),
        collection.bypassing_write((conversation_id, schedule_id), conn=conn),
    ):
        await conn.execute(_OPEN_GATE_SQL, PROTECTED_GATE_SETTING, "update")
        # cache-bypass: the gated UPDATE must run on the transaction that opened the gate; the row
        # is evicted from every tier once that transaction ends.
        await conn.execute(_UPDATE_SQL, conversation_id, schedule_id, dict(schedule_config), next_fire_at)
    log.info(
        "protected wake schedule changed",
        extra={
            "extra_data": {
                "agent_id": str(agent_id),
                "schedule_id": str(schedule_id),  # convert at border: protected-wake-changed log extra_data field
                "schedule_type": entity.schedule_type,
                "next_fire_at": next_fire_at.isoformat() if next_fire_at else None,
            }
        },
    )
    updated = await collection.find_for_agent(agent_id, schedule_id)
    if updated is None:
        raise ProtectedWakeError(f"wake {schedule_id} vanished while it was being changed")
    return updated


async def delete_protected(
    *,
    collection: WakeScheduleCollection,
    agent_id: UUID,
    schedule_id: UUID,
    conn: Any = None,
) -> None:
    """Delete a protected wake, as part of deleting its agent.

    Pass the agent deletion's own connection as ``conn`` so the wake goes
    in that transaction; the gate then stays open until it ends. That
    transaction must be opened by
    :class:`~threetears.core.collections.CallerTransaction`, which evicts the
    row from every cache tier once it has committed or rolled back: an
    eviction before the commit lets a reader re-cache the row L3 still holds.
    Without ``conn`` the delete runs in a transaction of its own.

    :param collection: three-tier wake-schedules collection
    :ptype collection: WakeScheduleCollection
    :param agent_id: the agent being deleted
    :ptype agent_id: UUID
    :param schedule_id: its protected wake
    :ptype schedule_id: UUID
    :param conn: the agent deletion's connection, inside its transaction
    :ptype conn: Any
    :return: nothing
    :rtype: None
    :raises ProtectedWakeError: when the agent has no such protected wake
    :raises ValueError: when ``conn`` is given and its transaction was not opened by
        :class:`~threetears.core.collections.CallerTransaction`
    """
    entity = await _protected_wake(collection, agent_id, schedule_id)
    conversation_id = entity.conversation_id
    if conn is not None:
        async with collection.bypassing_write((conversation_id, schedule_id), conn=conn):
            await _delete(conn, conversation_id, schedule_id)
    else:
        async with (
            _pool(collection).acquire() as own,
            CallerTransaction(own),
            collection.bypassing_write((conversation_id, schedule_id), conn=own),
        ):
            await _delete(own, conversation_id, schedule_id)
    log.info(
        "protected wake deleted with its agent",
        extra={"extra_data": {"agent_id": str(agent_id), "schedule_id": str(schedule_id)}},
    )


async def _delete(conn: Any, conversation_id: UUID, schedule_id: UUID) -> None:
    """Open the delete gate and delete, on one connection.

    :param conn: a connection inside a transaction
    :ptype conn: Any
    :param conversation_id: the wake's conversation
    :ptype conversation_id: UUID
    :param schedule_id: the wake
    :ptype schedule_id: UUID
    """
    await conn.execute(_OPEN_GATE_SQL, PROTECTED_GATE_SETTING, "delete")
    # cache-bypass: the gated DELETE must run on the transaction that opened the gate; the caller
    # runs it inside bypassing_write, which evicts the row once that transaction ends.
    await conn.execute(_DELETE_SQL, conversation_id, schedule_id)


def _pool(collection: WakeScheduleCollection) -> Any:
    """The collection's database pool, which these writes cannot do without.

    :param collection: three-tier wake-schedules collection
    :ptype collection: WakeScheduleCollection
    :return: the pool
    :rtype: Any
    :raises ProtectedWakeError: when the collection has no database
    """
    pool = collection.l3_pool
    if pool is None:
        raise ProtectedWakeError("the wake collection has no database to write to")
    return pool


async def _protected_wake(
    collection: WakeScheduleCollection,
    agent_id: UUID,
    schedule_id: UUID,
) -> WakeScheduleEntity:
    """The agent's protected wake, or an error naming what is wrong.

    :param collection: three-tier wake-schedules collection
    :ptype collection: WakeScheduleCollection
    :param agent_id: the agent
    :ptype agent_id: UUID
    :param schedule_id: the wake
    :ptype schedule_id: UUID
    :return: the wake
    :rtype: WakeScheduleEntity
    :raises ProtectedWakeError: when it is not the agent's, or not protected
    """
    entity = await collection.find_for_agent(agent_id, schedule_id)
    if entity is None:
        raise ProtectedWakeError(f"agent {agent_id} has no wake {schedule_id}")
    if not entity.protected:
        raise ProtectedWakeError(f"wake {schedule_id} is not protected; change it through the ordinary tools")
    return entity
