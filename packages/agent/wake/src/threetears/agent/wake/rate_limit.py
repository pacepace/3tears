"""Wake fire limits (per wake + per agent, or the older per-conv + per-user) + active-schedule cap.

Both helpers are pure functions over the asyncpg pool + a
:class:`WakeConfig` instance the consumer supplies. Pre-computed
counts are NOT cached -- the rate-limit query is cheap (covered by the
``idx_wake_fires_conv_time`` index from shard 01) and a 60s NATS-KV
cache would add a class of staleness bugs without measurable benefit.

Used by:

- :func:`threetears.agent.wake.dispatch.dispatch_wake` (per-fire
  rate-limit check as step 1).
- :mod:`threetears.agent.wake.tools.schedule_tools` (active-schedule
  cap on ``wake_schedule_create``).
- :mod:`threetears.agent.wake.webhook_adapter` (defers to its own
  per-subscription per-minute logic; could lift to here in a future
  shard if a third consumer surfaces the same shape).

Spec ref: ``docs/agent-wake/shard-05-observability-and-models.md``
OBS-13 / OBS-14; PLACEMENT §1.9.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from threetears.observe import get_logger

from threetears.agent.wake.config import WakeConfig
from threetears.agent.wake.types import FireLimits

if TYPE_CHECKING:
    from threetears.agent.wake.collections import WakeScheduleCollection
    from threetears.agent.wake.entities import WakeScheduleEntity
    from threetears.agent.wake.types import WakeTrigger

__all__ = [
    "RATE_LIMIT_WINDOW_HOURS",
    "ScheduleCapExceeded",
    "RateLimitScope",
    "WakeConversationMaker",
    "check_active_schedule_cap",
    "check_fire_limits",
    "check_rate_limit",
    "create_schedule_serialized",
    "resume_schedule_serialized",
]


# An agent's unprotected active schedules, across all its conversations,
# counted INSIDE the per-agent advisory lock so the count and the insert
# serialize. The lock + count + insert MUST share one connection / txn for
# the cap to hold under concurrency. Protected wakes are not counted: they
# exist for the agent's own sake and must never crowd out, or be refused by,
# the cap.
_COUNT_ACTIVE_SQL = (
    "SELECT COUNT(*) FROM agent_wake_schedules WHERE agent_id = $1 AND status = 'active' AND NOT protected"
)


# The same count for the resume path, EXCLUDING the schedule being
# (re)activated, so re-resuming an already-active schedule cannot refuse
# itself at exactly cap: the cap counts OTHER active schedules, then this one
# makes it ``other + 1 <= cap`` iff ``other < cap``.
_COUNT_ACTIVE_EXCLUDING_SQL = (
    "SELECT COUNT(*) FROM agent_wake_schedules "
    "WHERE agent_id = $1 AND status = 'active' AND NOT protected AND schedule_id != $2"
)


# Per-agent advisory lock taken for the transaction's lifetime. ``hashtext``
# maps the agent_id text to an int4; the ``::bigint`` cast selects the
# single-argument ``pg_advisory_xact_lock(bigint)`` form. It auto-releases at
# COMMIT/ROLLBACK, so a crashed or rolled-back create never strands it.
_ADVISORY_XACT_LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtext($1)::bigint)"


# Fires that ran, silent ones included, for one schedule / one webhook
# subscription since a point in time. Both are keyed by the wake's own
# conversation, the fire rows' partition.
_COUNT_WAKE_FIRES_SQL = (
    "SELECT COUNT(*) FROM wake_fires "
    "WHERE conversation_id = $1 AND schedule_id = $2 AND actual_fired_at > $3 "
    "AND status IN ('fired', 'fired_silent')"
)
_COUNT_SUBSCRIPTION_FIRES_SQL = (
    "SELECT COUNT(*) FROM wake_fires "
    "WHERE conversation_id = $1 AND webhook_subscription_id = $2 AND actual_fired_at > $3 "
    "AND status IN ('fired', 'fired_silent')"
)

# Fires that ran for every wake and subscription of one agent, protected
# wakes left out. The agent's wakes live in several conversations, so the
# count joins the two source tables on the agent.
_COUNT_AGENT_FIRES_SQL = (
    "SELECT "
    "(SELECT COUNT(*) FROM wake_fires wf "
    " JOIN agent_wake_schedules ws ON wf.schedule_id = ws.schedule_id "
    " WHERE ws.agent_id = $1 AND NOT ws.protected AND wf.actual_fired_at > $2 "
    " AND wf.status IN ('fired', 'fired_silent')) "
    "+ "
    "(SELECT COUNT(*) FROM wake_fires wf "
    " JOIN webhook_subscriptions ws ON wf.webhook_subscription_id = ws.subscription_id "
    " WHERE ws.agent_id = $1 AND wf.actual_fired_at > $2 "
    " AND wf.status IN ('fired', 'fired_silent')) "
    "AS total"
)


# Which cap was hit. ``None`` means the fire may proceed.
# :func:`check_fire_limits` answers ``'wake'`` / ``'agent'``; the older
# :func:`check_rate_limit`, used when the consumer supplies no permit
# callback, answers ``'conv'`` / ``'user'``. ``'webhook'`` is the webhook
# receiver's own per-subscription-per-minute cap (enforced in
# :mod:`threetears.agent.wake.webhook_adapter`). The set stays bounded
# because it is a Prometheus label.
RateLimitScope = Literal["wake", "agent", "conv", "user", "webhook"]


log = get_logger(__name__)


# 24h window for the per-conv / per-user fire caps (PLACEMENT §1.9).
# The rate-limit query counts ``status='fired'`` rows in the trailing
# 24h. Rate-limited rows do NOT count toward the cap because their
# ``next_fire_at`` already advanced past the window when first
# throttled (OBS-14).
RATE_LIMIT_WINDOW_HOURS: int = 24


async def check_rate_limit(
    trigger: "WakeTrigger",
    pool: Any,
    config: WakeConfig,
) -> RateLimitScope | None:
    """Return the scope of the exceeded cap, or ``None`` if the fire may proceed.

    Counts only ``status='fired'`` rows in the trailing 24h. The
    per-user count covers BOTH schedule-source and webhook-source
    fires (UNION over the two source-tables) so a user can't dodge the
    cap by alternating subscription types.

    Returns ``'conv'`` when the per-conv cap is at-or-over (per-user
    query is skipped). Returns ``'user'`` when the per-conv cap is
    under but the per-user cap is at-or-over. Returns ``None`` (the
    happy path) when both are under cap.

    Emit-site logging + Prometheus counter increment are the
    CALLER's responsibility -- the helper logs the per-cap diagnostic
    here but the caller attaches the structured event +
    :meth:`WakeMetricsEmitter.inc_rate_limit_rejection` so the trigger
    context lands on the event row coherently.

    ``None`` pool returns ``None`` so unit tests without a DB still
    exercise the call path.

    :param trigger: fire envelope (carries ``conversation_id`` +
        ``user_id``)
    :ptype trigger: WakeTrigger
    :param pool: asyncpg-compatible pool (or ``None`` in unit tests)
    :ptype pool: Any
    :param config: consumer's :class:`WakeConfig` impl supplying caps
    :ptype config: WakeConfig
    :return: scope of exceeded cap (``'conv'`` | ``'user'``) or
        ``None`` if the fire may proceed
    :rtype: RateLimitScope | None
    """
    if pool is None:
        return None

    since = datetime.now(UTC) - timedelta(hours=RATE_LIMIT_WINDOW_HOURS)
    # cache-bypass: aggregate COUNT over a rolling 24h window is not pk-addressable and must
    # read committed state at fire time -- a cached count is a cap that silently over-fires.
    # WakeFireCollection.count_in_window is the nearest Collection method and does not fit:
    # it has no status predicate, and this cap counts only status='fired' rows (a
    # rate-limited row already advanced its next_fire_at past the window, OBS-14). The
    # helper takes a bare pool by contract -- the consumer owns the pool and no Collection
    # reaches this call path.
    conv_count = await pool.fetchval(
        "SELECT COUNT(*) FROM wake_fires WHERE conversation_id = $1 AND actual_fired_at > $2 AND status = 'fired'",
        trigger.conversation_id,
        since,
    )
    conv_count_int = int(conv_count or 0)
    if conv_count_int >= config.max_fires_per_conv_per_day:
        log.info(
            "rate-limit: per-conv cap exceeded",
            extra={
                "extra_data": {
                    "schedule_id": str(trigger.schedule_id) if trigger.schedule_id else None,
                    "conversation_id": str(trigger.conversation_id),
                    "user_id": str(trigger.user_id),
                    "fire_source": trigger.fire_source,
                    "count": conv_count_int,
                    "cap": config.max_fires_per_conv_per_day,
                    "window_hours": RATE_LIMIT_WINDOW_HOURS,
                }
            },
        )
        return "conv"

    # Per-user count covers both schedule-source and webhook-source
    # fires. The two subqueries union via ``+`` and each hits the
    # corresponding source-table's PK->wake_fires index.
    #
    # cache-bypass: a summed pair of JOINed aggregates spanning three tables
    # (wake_fires x agent_wake_schedules, wake_fires x webhook_subscriptions) is exactly the
    # JOIN shape the Collection API cannot express -- user_id lives on the two source tables,
    # not on wake_fires, so no per-table Collection call can produce this number without
    # fanning the join out into the application and losing the single-statement read.
    user_count = await pool.fetchval(
        "SELECT "
        "(SELECT COUNT(*) FROM wake_fires wf "
        " JOIN agent_wake_schedules ws ON wf.schedule_id = ws.schedule_id "
        " WHERE ws.user_id = $1 AND wf.actual_fired_at > $2 AND wf.status = 'fired') "
        "+ "
        "(SELECT COUNT(*) FROM wake_fires wf "
        " JOIN webhook_subscriptions ws ON wf.webhook_subscription_id = ws.subscription_id "
        " WHERE ws.user_id = $1 AND wf.actual_fired_at > $2 AND wf.status = 'fired') "
        "AS total",
        trigger.user_id,
        since,
    )
    user_count_int = int(user_count or 0)
    if user_count_int >= config.max_fires_per_user_per_day:
        log.info(
            "rate-limit: per-user cap exceeded",
            extra={
                "extra_data": {
                    "schedule_id": str(trigger.schedule_id) if trigger.schedule_id else None,
                    "conversation_id": str(trigger.conversation_id),
                    "user_id": str(trigger.user_id),
                    "fire_source": trigger.fire_source,
                    "count": user_count_int,
                    "cap": config.max_fires_per_user_per_day,
                    "window_hours": RATE_LIMIT_WINDOW_HOURS,
                }
            },
        )
        return "user"

    return None


async def check_fire_limits(
    trigger: "WakeTrigger",
    pool: Any,
    limits: FireLimits,
) -> RateLimitScope | None:
    """Return the scope of the exceeded fire limit, or ``None`` if the fire may proceed.

    The limits come from the consumer's :class:`~threetears.agent.wake.types.FirePermit`
    for this fire. Fires that ran count, silent ones included, over the
    trailing 24 hours:

    - ``'wake'``: the fires of this wake (its schedule, or its webhook
      subscription) against ``limits.per_wake``.
    - ``'agent'``: the fires of every wake and subscription of the agent,
      protected wakes left out, against ``limits.per_agent``.

    A protected wake is not counted at all: this returns ``None`` for it
    without a query. ``None`` pool returns ``None`` so unit tests without a
    DB still exercise the call path.

    :param trigger: the fire
    :ptype trigger: WakeTrigger
    :param pool: asyncpg-compatible pool (or ``None`` in unit tests)
    :ptype pool: Any
    :param limits: the agent's limits for this fire
    :ptype limits: FireLimits
    :return: ``'wake'`` or ``'agent'`` when that limit is reached, else ``None``
    :rtype: RateLimitScope | None
    """
    if pool is None or trigger.protected:
        return None
    since = datetime.now(UTC) - timedelta(hours=RATE_LIMIT_WINDOW_HOURS)
    # cache-bypass: aggregate COUNTs over a rolling window must read committed
    # state at fire time; a cached count is a limit that silently over-fires.
    if trigger.schedule_id is not None:
        wake_count = await pool.fetchval(_COUNT_WAKE_FIRES_SQL, trigger.conversation_id, trigger.schedule_id, since)
    else:
        wake_count = await pool.fetchval(
            _COUNT_SUBSCRIPTION_FIRES_SQL,
            trigger.conversation_id,
            trigger.webhook_subscription_id,
            since,
        )
    if int(wake_count or 0) >= limits.per_wake:
        _log_limit_reached(trigger, "wake", int(wake_count or 0), limits.per_wake)
        return "wake"
    agent_count = await pool.fetchval(_COUNT_AGENT_FIRES_SQL, trigger.agent_id, since)
    if int(agent_count or 0) >= limits.per_agent:
        _log_limit_reached(trigger, "agent", int(agent_count or 0), limits.per_agent)
        return "agent"
    return None


def _log_limit_reached(trigger: "WakeTrigger", scope: RateLimitScope, count: int, cap: int) -> None:
    """Log one reached fire limit with the fire's identity.

    :param trigger: the fire
    :ptype trigger: WakeTrigger
    :param scope: which limit
    :ptype scope: RateLimitScope
    :param count: fires counted
    :ptype count: int
    :param cap: the limit
    :ptype cap: int
    """
    log.info(
        "rate-limit: fire limit reached",
        extra={
            "extra_data": {
                "scope": scope,
                "schedule_id": str(trigger.schedule_id) if trigger.schedule_id else None,
                "agent_id": str(trigger.agent_id),
                "conversation_id": str(trigger.conversation_id),
                "fire_source": trigger.fire_source,
                "count": count,
                "cap": cap,
                "window_hours": RATE_LIMIT_WINDOW_HOURS,
            }
        },
    )


async def check_active_schedule_cap(
    *,
    agent_id: UUID,
    cap: int,
    pool: Any | None = None,
    count_func: Callable[[], Awaitable[int]] | None = None,
) -> bool:
    """Return ``True`` when the agent is under its active-schedule cap.

    Counts the agent's unprotected ``status='active'`` schedules across all
    its conversations and compares against ``cap`` (the consumer's
    :class:`WakeConfig` typically passes ``config.max_active_schedules_per_agent``).
    This is the advisory check; the enforcing one is
    :func:`create_schedule_serialized`, which counts under the agent's lock.

    ``count_func`` wins over ``pool`` when both are supplied. Supplying
    neither returns ``True`` (parallels the ``pool=None`` short-circuit on
    :func:`check_rate_limit`).

    :param agent_id: the agent under test
    :ptype agent_id: UUID
    :param cap: maximum allowed active schedules for the agent
    :ptype cap: int
    :param pool: asyncpg-compatible pool (alternative to ``count_func``)
    :ptype pool: Any | None
    :param count_func: async callable returning the active count
    :ptype count_func: Callable[[], Awaitable[int]] | None
    :return: ``True`` if a new schedule may be created
    :rtype: bool
    """
    if count_func is not None:
        count = int(await count_func())
    elif pool is not None:
        # cache-bypass: aggregate COUNT not pk-addressable, and a cap read from
        # cache is a cap that lets an extra schedule through.
        value = await pool.fetchval(_COUNT_ACTIVE_SQL, agent_id)
        count = int(value or 0)
    else:
        return True

    if count >= cap:
        log.info(
            "rate-limit: active-schedule cap exceeded",
            extra={"extra_data": {"agent_id": str(agent_id), "count": count, "cap": cap}},
        )
        return False
    return True


class ScheduleCapExceeded(Exception):
    """Raised by :func:`create_schedule_serialized` when the cap is hit.

    Carries the agent, the observed active count, and the cap so both
    consumers (the agent ``wake_schedule_create`` tool and the consumer's
    REST router) can render their own surface-appropriate error (a
    ``[TOOL ERROR]`` string vs. an HTTP 400) without re-counting.

    :ivar agent_id: agent whose cap was hit
    :ivar count: active-schedule count observed under the lock
    :ivar cap: the configured per-agent cap
    """

    def __init__(self, *, agent_id: UUID, count: int, cap: int) -> None:
        self.agent_id = agent_id
        self.count = count
        self.cap = cap
        super().__init__(
            f"active-schedule cap reached for agent {agent_id}: {count} >= {cap}",
        )


# Makes the wake conversation a new schedule lives in, on the create
# transaction's connection, and returns its id.
WakeConversationMaker = Callable[[Any], Awaitable[UUID]]


async def create_schedule_serialized(
    *,
    collection: WakeScheduleCollection,
    data: dict[str, Any],
    agent_id: UUID,
    cap: int,
    pool: Any,
    make_wake_conversation: WakeConversationMaker | None = None,
) -> WakeScheduleEntity:
    """Insert a wake schedule under a per-agent advisory lock.

    Closes the check-then-insert race on the active-schedule cap. Within a
    SINGLE transaction on one pooled connection:

    1. ``pg_advisory_xact_lock(hashtext(agent_id::text))`` -- serializes
       every concurrent create and resume for the SAME agent, whichever
       conversation each targets; different agents do not contend. The lock
       is transaction-scoped, so it releases on COMMIT/ROLLBACK.
    2. Count the agent's unprotected active schedules ON THE SAME
       connection, so the count reflects every create that already
       released the lock.
    3. Raise :class:`ScheduleCapExceeded` when ``count >= cap`` -- BEFORE
       anything is written, so the cap holds exactly. A protected schedule
       is not counted and is not refused.
    4. When ``make_wake_conversation`` is given, make the schedule's wake
       conversation on the same connection and put its id on the row, so a
       refused or failed insert leaves no empty conversation.
    5. ``collection.save_entity(entity, conn=conn)``.

    The caller owns validation (schedule config, skill ACL, ``context_from``)
    and builds ``data``; ``data["conversation_id"]`` is the wake
    conversation to create into, and is replaced when
    ``make_wake_conversation`` makes a new one.

    :param collection: three-tier wake-schedules collection
    :ptype collection: WakeScheduleCollection
    :param data: the new row, every column but a new wake conversation's id
    :ptype data: dict[str, Any]
    :param agent_id: the agent; the advisory-lock key and the cap's scope
    :ptype agent_id: UUID
    :param cap: maximum allowed active schedules for the agent
    :ptype cap: int
    :param pool: asyncpg-compatible pool exposing ``acquire()`` +
        per-connection ``transaction()`` / ``fetchval()`` / ``execute()``
    :ptype pool: Any
    :param make_wake_conversation: makes a new wake conversation on the
        connection and returns its id; ``None`` creates into
        ``data["conversation_id"]``
    :ptype make_wake_conversation: WakeConversationMaker | None
    :return: the persisted schedule
    :rtype: WakeScheduleEntity
    :raises ScheduleCapExceeded: when the agent is at/over cap
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _lock_agent(conn, agent_id)
            if not data.get("protected", False):
                await _refuse_at_cap(conn, _COUNT_ACTIVE_SQL, (agent_id,), agent_id=agent_id, cap=cap, action="create")
            row = dict(data)
            if make_wake_conversation is not None:
                row["conversation_id"] = await make_wake_conversation(conn)
            entity = collection.create(row)
            await collection.save_entity(entity, conn=conn)
    return entity


async def resume_schedule_serialized(
    *,
    collection: WakeScheduleCollection,
    agent_id: UUID,
    conversation_id: UUID,
    schedule_id: UUID,
    next_fire_at: datetime,
    cap: int,
    pool: Any,
) -> None:
    """Re-activate a schedule under the per-agent advisory lock.

    The mirror of :func:`create_schedule_serialized` for transitions INTO
    ``status='active'``. Without it a pause -> create-to-fill -> resume
    sequence could push the active count past the cap. Within a SINGLE
    transaction: take the agent's lock, count the agent's other unprotected
    active schedules (the target excluded, so re-resuming an active one
    cannot refuse itself), raise :class:`ScheduleCapExceeded` at cap, then
    flip the row with :meth:`WakeScheduleCollection.resume` on the same
    connection.

    The caller owns ``next_fire_at`` and the ownership check.

    :param collection: three-tier wake-schedules collection
    :ptype collection: WakeScheduleCollection
    :param agent_id: the agent; lock key and cap scope
    :ptype agent_id: UUID
    :param conversation_id: the schedule's conversation (partition column)
    :ptype conversation_id: UUID
    :param schedule_id: schedule being re-activated (excluded from count)
    :ptype schedule_id: UUID
    :param next_fire_at: recomputed next fire instant for the resumed row
    :ptype next_fire_at: datetime
    :param cap: maximum allowed active schedules for the agent
    :ptype cap: int
    :param pool: asyncpg-compatible pool
    :ptype pool: Any
    :return: nothing
    :rtype: None
    :raises ScheduleCapExceeded: when the agent is at/over cap
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            await _lock_agent(conn, agent_id)
            await _refuse_at_cap(
                conn,
                _COUNT_ACTIVE_EXCLUDING_SQL,
                (agent_id, schedule_id),
                agent_id=agent_id,
                cap=cap,
                action="resume",
            )
            await collection.resume(
                conversation_id,
                schedule_id,
                next_fire_at=next_fire_at,
                conn=conn,
            )
    # NOTE: like :meth:`WakeScheduleCollection.resume` / ``.pause``, the
    # flip is a cache-bypass L3 UPDATE -- the L1/L2 row cache is "read-
    # mostly, invalidated naturally on the next fetch" (the established
    # contract for these status transitions). We deliberately do NOT call
    # ``invalidate_cache`` here: an eviction would strip the row out from
    # under any LIVE entity proxy the caller still holds (the proxy reads
    # every field through L1 via ``get_field_sync``), turning a subsequent
    # ``entity.schedule_id`` read into ``None``. The REST caller
    # reflects the new ``status`` / ``next_fire_at`` onto its proxy via
    # field setters (which write through to L1) for the response; the
    # agent tool caller returns a string and re-reads on the next ``get``.


async def _lock_agent(conn: Any, agent_id: UUID) -> None:
    """Take the agent's transaction-scoped advisory lock.

    :param conn: the transaction's connection
    :ptype conn: Any
    :param agent_id: the agent
    :ptype agent_id: UUID
    """
    await conn.execute(
        _ADVISORY_XACT_LOCK_SQL, str(agent_id)
    )  # convert at border: pg_advisory_xact_lock(hashtext($1)) text arg


async def _refuse_at_cap(
    conn: Any,
    sql: str,
    args: tuple[Any, ...],
    *,
    agent_id: UUID,
    cap: int,
    action: str,
) -> None:
    """Count under the lock and raise when the agent is at its cap.

    :param conn: the transaction's connection, holding the agent's lock
    :ptype conn: Any
    :param sql: the count to run
    :ptype sql: str
    :param args: its arguments
    :ptype args: tuple[Any, ...]
    :param agent_id: the agent
    :ptype agent_id: UUID
    :param cap: the cap
    :ptype cap: int
    :param action: ``'create'`` or ``'resume'``, for the log line
    :ptype action: str
    :raises ScheduleCapExceeded: when ``count >= cap``
    """
    count = int(await conn.fetchval(sql, *args) or 0)
    if count >= cap:
        log.info(
            "rate-limit: active-schedule cap exceeded (serialized)",
            extra={"extra_data": {"agent_id": str(agent_id), "action": action, "count": count, "cap": cap}},
        )
        raise ScheduleCapExceeded(agent_id=agent_id, count=count, cap=cap)
