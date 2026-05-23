"""Literal types + enum constants for agent-wake.

These mirror the CHECK constraints on the L3 tables: changing the
allowed values here requires a matching migration.

Why ``Literal`` and not ``Enum``: callers (the rest of 3tears, plus
metallm consumers) pass these values through tool input schemas + JSON
boundaries where strings round-trip cleanly. ``Literal`` keeps the
runtime payload a plain ``str`` so JSON encoding doesn't need a custom
serializer and mypy still pins valid value sets at every call site.

Naming follows the agent-skills precedent
(:mod:`threetears.agent.skills.types`).
"""

from __future__ import annotations

from typing import Literal

__all__ = [
    "DeliveryTarget",
    "ExecutionMode",
    "FireStatus",
    "MissedFirePolicy",
    "ScheduleStatus",
    "ScheduleType",
    "VerificationScheme",
    "WebhookSubscriptionStatus",
]


# ``schedule_type`` column on ``agent_wake_schedules``. CHECK-pinned in
# the L3 schema. Each value maps to a specific ``schedule_config`` JSONB
# shape documented in :mod:`threetears.agent.wake.collections`. The
# ``'cron'`` and ``'relative_delay'`` values were added in the
# 2026-05-19 revision per PLACEMENT §1.5.
ScheduleType = Literal[
    "daily_at",
    "every_n_hours",
    "random_within_window",
    "one_shot_at",
    "cron",
    "relative_delay",
    "interval",
]


# ``status`` column on ``agent_wake_schedules``. ``'paused'`` is set by
# the agent / user via ``WakeScheduleCollection.pause``; ``'expired'``
# is set automatically for one-shot schedules after firing. CHECK-pinned
# in the L3 schema.
ScheduleStatus = Literal["active", "paused", "expired"]


# ``execution_mode`` column on both ``agent_wake_schedules`` and
# ``webhook_subscriptions``. ``'inline'`` (default) injects the wake
# turn into the originating conversation; ``'spawn'`` creates a new
# conversation per fire. CHECK-pinned in the L3 schema.
ExecutionMode = Literal["inline", "spawn"]


# ``missed_fire_policy`` column on ``agent_wake_schedules`` (added per
# PLACEMENT §1.7 in the 2026-05-19 revision). ``'coalesce'`` fires
# ONCE for a backlog of missed ticks (recomputed ``next_fire_at``
# forward); ``'catch_up'`` fires once per missed tick. CHECK-pinned in
# the L3 schema.
MissedFirePolicy = Literal["coalesce", "catch_up"]


# ``status`` column on ``wake_fires``. The full enum after the
# wake-yield revision (2026-05-19 evening, PLACEMENT §8.5.1) gained the
# ``'yielded'`` value. CHECK-pinned in the L3 schema.
#
# - ``'fired'`` -- wake executed end-to-end, visible assistant response.
# - ``'fired_silent'`` -- wake executed; assistant response started with
#   ``[SILENT]`` and was suppressed (PLACEMENT §1.4).
# - ``'yielded'`` -- agent voluntarily yielded mid-turn via
#   ``wake_yield`` because a user message was waiting (PLACEMENT §8.5.1).
# - ``'skipped_busy'`` -- per-conv lock held; deferred.
# - ``'skipped_rate_limit'`` -- per-conv / per-user cap exceeded.
# - ``'skipped_cap'`` -- per-conv active-schedule cap exceeded
#   (PLACEMENT §1.9).
# - ``'skipped_no_handler'`` -- product did not register a handler
#   callback at dispatch time.
# - ``'failed'`` -- exception raised during dispatch / handler.
FireStatus = Literal[
    "fired",
    "fired_silent",
    "yielded",
    "skipped_busy",
    "skipped_rate_limit",
    "skipped_cap",
    "skipped_no_handler",
    "failed",
]


# ``delivery_target`` column on both ``agent_wake_schedules`` and
# ``webhook_subscriptions``. ``'conversation'`` (default) injects the
# wake response into the originating conversation; ``'email'`` routes
# via a consumer-supplied ``DeliveryAdapter`` (PLACEMENT §1.17 / §3.4).
# CHECK-pinned in the L3 schema.
DeliveryTarget = Literal["conversation", "email"]


# ``verification_scheme`` column on ``webhook_subscriptions``. Only
# ``'generic_hmac_sha256'`` is recognised in v1; the enum is typed so
# adding ``'slack_signing'`` / ``'github_hmac'`` in a future shard
# requires only a CHECK migration + Literal update.
VerificationScheme = Literal["generic_hmac_sha256"]


# ``status`` column on ``webhook_subscriptions``. Symmetrical to
# ``ScheduleStatus`` minus the one-shot ``'expired'`` transition --
# webhooks are long-lived. CHECK-pinned in the L3 schema.
WebhookSubscriptionStatus = Literal["active", "paused"]
