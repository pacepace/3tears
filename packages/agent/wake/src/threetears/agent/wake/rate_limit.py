"""Wake rate-limit helpers (per-conv + per-user) + active-schedule cap.

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

if TYPE_CHECKING:
    from threetears.agent.wake.collections import WakeScheduleCollection
    from threetears.agent.wake.entities import WakeScheduleEntity
    from threetears.agent.wake.types import WakeTrigger

__all__ = [
    "RATE_LIMIT_WINDOW_HOURS",
    "ScheduleCapExceeded",
    "RateLimitScope",
    "_check_active_schedule_cap",
    "_check_rate_limit",
    "create_schedule_serialized",
]


# Active-schedule COUNT executed INSIDE the per-conversation advisory
# lock (so the count + insert serialize). Identical predicate to
# :meth:`WakeScheduleCollection.count_active_for_conversation`, but bound
# to the caller's transaction connection rather than the collection's
# pool -- the lock + count + insert MUST share one connection / txn for
# the cap to hold under concurrency.
_COUNT_ACTIVE_SQL = "SELECT COUNT(*) FROM agent_wake_schedules WHERE conversation_id = $1 AND status = 'active'"


# Per-conversation advisory lock taken for the transaction's lifetime.
# ``hashtext`` maps the conversation_id text to an int4; the ``::bigint``
# cast selects the single-argument ``pg_advisory_xact_lock(bigint)`` form.
# The lock auto-releases at COMMIT/ROLLBACK (xact-scoped), so no explicit
# unlock is needed and a crashed/rolled-back create never strands the lock.
_ADVISORY_XACT_LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtext($1)::bigint)"


# Which cap was hit by :func:`_check_rate_limit`. ``None`` means
# the fire may proceed; ``'conv'`` / ``'user'`` identify the per-conv
# vs per-user cap so the caller can attach the scope to its log line
# + Prometheus label. ``'webhook'`` is reserved for the webhook-side
# per-subscription cap (enforced separately in
# :mod:`threetears.agent.wake.webhook_adapter`) so the scope label set
# stays bounded across both call sites.
RateLimitScope = Literal["conv", "user", "webhook"]


log = get_logger(__name__)


# 24h window for the per-conv / per-user fire caps (PLACEMENT §1.9).
# The rate-limit query counts ``status='fired'`` rows in the trailing
# 24h. Rate-limited rows do NOT count toward the cap because their
# ``next_fire_at`` already advanced past the window when first
# throttled (OBS-14).
RATE_LIMIT_WINDOW_HOURS: int = 24


async def _check_rate_limit(
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


async def _check_active_schedule_cap(
    *,
    conversation_id: UUID,
    cap: int,
    pool: Any | None = None,
    count_func: Callable[[], Awaitable[int]] | None = None,
) -> bool:
    """Return ``True`` when the conversation is under its active-schedule cap.

    Counts ``status='active'`` rows for ``conversation_id`` and
    compares against ``cap`` (the default lives at
    :data:`DEFAULT_MAX_SCHEDULES_PER_CONVERSATION` = 10 per PLACEMENT
    §1.9; the consumer's :class:`WakeConfig` impl typically passes
    ``config.max_schedules_per_conversation`` here). Returns ``False``
    when at-or-over the cap.

    The single source of truth for the cap-check semantics lives in
    this helper. ``wake_schedule_create`` calls it before INSERT (the
    primary enforcement point) so future surfaces (a cleanup task that
    re-asserts caps on user-initiated re-enables, say) share one rule.

    Two count paths are supported so both call shapes can route through
    the same enforcement:

    - ``count_func`` (preferred for the tools layer): an async callable
      returning the current count. Lets a caller that already owns a
      :class:`WakeScheduleCollection` share the collection's tested
      ``count_active_for_conversation`` method instead of duplicating
      the SQL.
    - ``pool`` (kept for direct-pool callers): runs the COUNT inline
      against the pool.

    When both are supplied, ``count_func`` wins. Supplying neither
    returns ``True`` (parallels the ``pool=None`` short-circuit on
    :func:`_check_rate_limit`).

    Why ``cap: int`` instead of ``config: WakeConfig``: callers that
    already have an integer cap in hand (the tool factory closes over
    ``max_schedules_per_conversation``) would otherwise have to build a
    throwaway :class:`WakeConfig` shim to satisfy a single attribute
    read. Taking the integer directly keeps the helper minimal; callers
    holding a full :class:`WakeConfig` pass
    ``config.max_schedules_per_conversation``.

    :param conversation_id: conversation under test
    :ptype conversation_id: UUID
    :param cap: maximum allowed active schedules for the conversation
    :ptype cap: int
    :param pool: asyncpg-compatible pool (alternative to ``count_func``)
    :ptype pool: Any | None
    :param count_func: async callable returning the active count
        (preferred when the caller already has a Collection handy)
    :ptype count_func: Callable[[], Awaitable[int]] | None
    :return: ``True`` if a new schedule may be created
    :rtype: bool
    """
    if count_func is not None:
        count = int(await count_func())
    elif pool is not None:
        value = await pool.fetchval(
            "SELECT COUNT(*) FROM agent_wake_schedules WHERE conversation_id = $1 AND status = 'active'",
            conversation_id,
        )
        count = int(value or 0)
    else:
        return True

    if count >= cap:
        log.info(
            "rate-limit: active-schedule cap exceeded",
            extra={
                "extra_data": {
                    "conversation_id": str(conversation_id),
                    "count": count,
                    "cap": cap,
                }
            },
        )
        return False
    return True


class ScheduleCapExceeded(Exception):
    """Raised by :func:`create_schedule_serialized` when the cap is hit.

    Carries the conversation, the observed active count, and the cap so
    both consumers (the agent ``wake_schedule_create`` tool and the
    metallm REST router) can render their own surface-appropriate error
    (a ``[TOOL ERROR]`` string vs. an HTTP 400) without re-counting.

    :ivar conversation_id: conversation whose cap was hit
    :ivar count: active-schedule count observed under the lock
    :ivar cap: the configured per-conversation cap
    """

    def __init__(self, *, conversation_id: UUID, count: int, cap: int) -> None:
        self.conversation_id = conversation_id
        self.count = count
        self.cap = cap
        super().__init__(
            f"active-schedule cap reached for conversation {conversation_id}: {count} >= {cap}",
        )


async def create_schedule_serialized(
    *,
    collection: WakeScheduleCollection,
    entity: WakeScheduleEntity,
    conversation_id: UUID,
    cap: int,
    pool: Any,
) -> None:
    """Insert a wake schedule under a per-conversation advisory lock.

    Closes the check-then-insert TOCTOU race on the active-schedule cap
    (PLACEMENT §1.9). Within a SINGLE transaction on one pooled
    connection:

    1. ``pg_advisory_xact_lock(hashtext(conversation_id::text))`` --
       serializes every concurrent create for the SAME conversation;
       creates for different conversations do not contend (distinct lock
       keys). The lock is transaction-scoped, so it releases on
       COMMIT/ROLLBACK with no explicit unlock.
    2. Re-count ``status='active'`` rows for the conversation ON THE
       SAME connection (so the count reflects rows committed by any
       create that already released the lock).
    3. Raise :class:`ScheduleCapExceeded` when ``count >= cap`` --
       BEFORE the insert, so the cap holds exactly.
    4. ``collection.save_entity(entity, conn=conn)`` -- the L3 INSERT
       binds to the locked transaction; the L1/L2/invalidation tiers run
       through the normal :meth:`save_entity` path.

    Two concurrent creates against a full conversation thus serialize:
    the first commits (releasing the lock), the second re-counts under
    the lock, sees the cap, and raises. The cap can never be exceeded by
    a race because the count + insert are atomic per conversation.

    The single ``COUNT`` runs inside the lock rather than relying on a DB
    trigger / CHECK constraint (which cannot express "count < cap" without
    a subquery in the constraint, unsupported by PostgreSQL). The advisory
    lock is the least-invasive race-proof primitive that needs no schema
    change.

    The caller owns entity construction (id, next_fire_at, skill ACL,
    context_from validation all happen before this call) so the helper
    stays agnostic of the create surface's validation order. Both the
    agent tool and the REST router route their persist step through here.

    :param collection: three-tier wake-schedules collection
    :ptype collection: WakeScheduleCollection
    :param entity: the fully-constructed (not-yet-persisted) schedule
    :ptype entity: WakeScheduleEntity
    :param conversation_id: partition column + advisory-lock key
    :ptype conversation_id: UUID
    :param cap: maximum allowed active schedules for the conversation
    :ptype cap: int
    :param pool: asyncpg-compatible pool exposing ``acquire()`` +
        per-connection ``transaction()`` / ``fetchval()`` / ``execute()``
    :ptype pool: Any
    :return: nothing
    :rtype: None
    :raises ScheduleCapExceeded: when the conversation is at/over cap
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(_ADVISORY_XACT_LOCK_SQL, str(conversation_id))
            value = await conn.fetchval(_COUNT_ACTIVE_SQL, conversation_id)
            count = int(value or 0)
            if count >= cap:
                log.info(
                    "rate-limit: active-schedule cap exceeded (serialized create)",
                    extra={
                        "extra_data": {
                            "conversation_id": str(conversation_id),
                            "count": count,
                            "cap": cap,
                        }
                    },
                )
                raise ScheduleCapExceeded(
                    conversation_id=conversation_id,
                    count=count,
                    cap=cap,
                )
            await collection.save_entity(entity, conn=conn)
