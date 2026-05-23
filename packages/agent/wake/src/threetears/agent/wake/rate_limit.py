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

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID

from threetears.observe import get_logger

from threetears.agent.wake.config import WakeConfig

if TYPE_CHECKING:
    from threetears.agent.wake.types import WakeTrigger

__all__ = [
    "RATE_LIMIT_WINDOW_HOURS",
    "_check_active_schedule_cap",
    "_check_rate_limit",
]


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
) -> bool:
    """Return ``True`` when both per-conv + per-user counts are under cap.

    Counts only ``status='fired'`` rows in the trailing 24h. The
    per-user count covers BOTH schedule-source and webhook-source
    fires (UNION over the two source-tables) so a user can't dodge the
    cap by alternating subscription types.

    Returns ``False`` when either cap is exceeded; emit-site logs +
    counter increment are the caller's responsibility (so the caller
    can attach trigger context to the log + scope='conv' / 'user' to
    the counter). ``None`` pool returns ``True`` so unit tests without
    a DB still exercise the call path.

    :param trigger: fire envelope (carries ``conversation_id`` +
        ``user_id``)
    :ptype trigger: WakeTrigger
    :param pool: asyncpg-compatible pool (or ``None`` in unit tests)
    :ptype pool: Any
    :param config: consumer's :class:`WakeConfig` impl supplying caps
    :ptype config: WakeConfig
    :return: ``True`` if the fire may proceed; ``False`` if either cap
        is at-or-over
    :rtype: bool
    """
    if pool is None:
        return True

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
        return False

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
        return False

    return True


async def _check_active_schedule_cap(
    *,
    conversation_id: UUID,
    pool: Any,
    config: WakeConfig,
) -> bool:
    """Return ``True`` when the conversation is under its active-schedule cap.

    Counts ``status='active'`` rows for ``conversation_id`` and
    compares against :attr:`WakeConfig.max_schedules_per_conversation`
    (default 10 per PLACEMENT §1.9). Returns ``False`` when at-or-over
    the cap.

    Used by ``wake_schedule_create`` before INSERT (the primary
    enforcement point) and is exposed here so future surfaces (a
    cleanup task that re-asserts caps on user-initiated re-enables,
    say) can share the rule. ``None`` pool returns ``True``.

    :param conversation_id: conversation under test
    :ptype conversation_id: UUID
    :param pool: asyncpg-compatible pool (or ``None`` in unit tests)
    :ptype pool: Any
    :param config: consumer's :class:`WakeConfig` impl supplying the cap
    :ptype config: WakeConfig
    :return: ``True`` if a new schedule may be created
    :rtype: bool
    """
    if pool is None:
        return True

    value = await pool.fetchval(
        "SELECT COUNT(*) FROM agent_wake_schedules WHERE conversation_id = $1 AND status = 'active'",
        conversation_id,
    )
    count = int(value or 0)
    if count >= config.max_schedules_per_conversation:
        log.info(
            "rate-limit: active-schedule cap exceeded",
            extra={
                "extra_data": {
                    "conversation_id": str(conversation_id),
                    "count": count,
                    "cap": config.max_schedules_per_conversation,
                }
            },
        )
        return False
    return True
