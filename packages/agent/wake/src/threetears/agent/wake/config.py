"""Wake-side config Protocol + platform defaults.

Per PLACEMENT.md (2026-05-19 revision) the policy values live with the
consumer (its ``system_settings``) while the mechanism that
enforces them is platform. :class:`WakeConfig` declares the read-side
shape; the platform ships ``DEFAULT_*`` constants the consumer can
fall back to (or override per-deployment).

The consumer typically caches its concrete impl with a short TTL (~60s
suggested) so admin updates propagate within a tick. The Protocol is
pure-read; mutations go through whichever admin REST surface the
consumer ships.

Spec ref: ``docs/agent-wake/shard-05-observability-and-models.md``
requirements OBS-15 / OBS-16; PLACEMENT §1.9 / §1.15 / §3.5.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = [
    "DEFAULT_HTTP_ALLOWED_HOSTS",
    "DEFAULT_LOKI_NAMED_QUERIES",
    "DEFAULT_MAX_FIRES_PER_CONV_PER_DAY",
    "DEFAULT_MAX_FIRES_PER_USER_PER_DAY",
    "DEFAULT_MAX_SCHEDULES_PER_CONVERSATION",
    "DEFAULT_MAX_WEBHOOK_FIRES_PER_SUBSCRIPTION_PER_HOUR",
    "DEFAULT_POSTGRES_NAMED_QUERIES",
    "DEFAULT_WAKE_CONFIG",
    "WakeConfig",
]


# Per-conversation 24h fire cap (PLACEMENT §1.9). The conv-rate-limit
# query in :mod:`threetears.agent.wake.rate_limit` counts status='fired'
# rows in the trailing 24h window and rejects when the count is at or
# above this value.
DEFAULT_MAX_FIRES_PER_CONV_PER_DAY: int = 24


# Per-user 24h fire cap covering BOTH scheduled and webhook fires
# (PLACEMENT §1.9). The two source-tables are unioned in the rate-limit
# helper's per-user count.
DEFAULT_MAX_FIRES_PER_USER_PER_DAY: int = 100


# Per-webhook-subscription rate cap. The webhook receiver's per-minute
# window cap defaults to this divided down (60/min default) -- the
# subscription row can override via ``rate_limit_per_minute``. Kept as
# an hourly value so the WakeConfig surface stays consistent (per-hour
# units throughout the platform's caps).
DEFAULT_MAX_WEBHOOK_FIRES_PER_SUBSCRIPTION_PER_HOUR: int = 60


# Per-conversation active-schedule cap (PLACEMENT §1.9 / §3.5).
# Enforced at ``wake_schedule_create`` time and re-verifiable on a tick
# via :func:`threetears.agent.wake.rate_limit._check_active_schedule_cap`.
# Default = 10 locked 2026-05-19.
DEFAULT_MAX_SCHEDULES_PER_CONVERSATION: int = 10


# Empty platform-side default for the consumer's HTTP-allow-list. The
# pre-check tools (``http_get`` etc., now living in agent-tools per
# PLACEMENT §2.2) consult this on every fetch; an empty tuple means
# "no hosts allowed" -- a safe default that forces the consumer to opt
# specific hosts in.
DEFAULT_HTTP_ALLOWED_HOSTS: tuple[str, ...] = ()


# Empty platform-side defaults for the named-query registries the
# pre-check tools (``loki_query`` / ``postgres_query``) consult. The
# consumer wires its product-specific queries into these dicts.
DEFAULT_LOKI_NAMED_QUERIES: dict[str, str] = {}
DEFAULT_POSTGRES_NAMED_QUERIES: dict[str, str] = {}


@runtime_checkable
class WakeConfig(Protocol):
    """Read-side configuration the consumer supplies to dispatch_wake.

    Implementations typically read from the consumer's system_settings
    (e.g. ``users.config_*`` columns + the ``system_settings``
    table). Pure read protocol -- no mutation methods. Cached in the
    consumer (suggested ~60s TTL); admin updates propagate within a
    tick.

    Pre-check tool surfaces (``http_get`` / ``loki_query`` /
    ``postgres_query``) read ``http_allowed_hosts`` /
    ``loki_named_queries`` / ``postgres_named_queries`` via this
    Protocol so the platform stays SMTP- and Loki-deployment-agnostic.

    Every property has a corresponding ``DEFAULT_*`` constant at module
    scope so a consumer that wants the platform-baseline behaviour can
    delegate to the defaults from its own ``WakeConfig`` implementation.

    Spec ref: ``docs/agent-wake/shard-05-observability-and-models.md``
    OBS-15.

    :ivar max_fires_per_conv_per_day: trailing-24h cap on
        ``status='fired'`` rows per conversation
    :ivar max_fires_per_user_per_day: trailing-24h cap covering both
        scheduled and webhook fires per user (UNION over both
        source-tables in the rate-limit query)
    :ivar max_webhook_fires_per_subscription_per_hour: rolling-hour
        cap consumed by the webhook receiver (subscription-row override
        wins when present)
    :ivar max_schedules_per_conversation: count cap on rows with
        ``status='active'`` for a given conversation (enforced at
        create + verifiable on tick)
    :ivar http_allowed_hosts: tuple of FQDN patterns the
        ``http_get`` pre-check tool may target; empty tuple = no hosts
        allowed (safe default)
    :ivar loki_client: consumer-supplied async Loki client (must expose
        ``query_range(query, since, until, limit)``); ``None`` disables
        the ``loki_query`` pre-check tool
    :ivar loki_named_queries: name -> LogQL string registry for the
        ``loki_query`` pre-check tool; lookup-only (the agent passes a
        name, never raw LogQL)
    :ivar postgres_named_queries: name -> parameterised SQL registry
        for the ``postgres_query`` pre-check tool; lookup-only (the
        agent passes a name and bind parameters, never raw SQL)
    """

    @property
    def max_fires_per_conv_per_day(self) -> int: ...

    @property
    def max_fires_per_user_per_day(self) -> int: ...

    @property
    def max_webhook_fires_per_subscription_per_hour(self) -> int: ...

    @property
    def max_schedules_per_conversation(self) -> int: ...

    @property
    def http_allowed_hosts(self) -> tuple[str, ...]: ...

    @property
    def loki_client(self) -> Any | None: ...

    @property
    def loki_named_queries(self) -> dict[str, str]: ...

    @property
    def postgres_named_queries(self) -> dict[str, str]: ...


class _DefaultWakeConfig:
    """Concrete :class:`WakeConfig` returning only the platform defaults.

    Used as the fallback when a consumer (or a call site that pre-dates
    full wake-config plumbing) invokes :func:`dispatch_wake` /
    :func:`webhook_receive` without an explicit config. Every property
    delegates to the corresponding module-level ``DEFAULT_*`` constant
    so an operator changing a default sees it land everywhere at once.

    Consumers wanting per-deployment overrides supply their own
    :class:`WakeConfig` impl (typically reading ``system_settings`` or
    equivalent) -- this default exists so the rate-limit + cap helpers
    have a non-``None`` config to consult even when no consumer config
    has been registered.
    """

    @property
    def max_fires_per_conv_per_day(self) -> int:
        return DEFAULT_MAX_FIRES_PER_CONV_PER_DAY

    @property
    def max_fires_per_user_per_day(self) -> int:
        return DEFAULT_MAX_FIRES_PER_USER_PER_DAY

    @property
    def max_webhook_fires_per_subscription_per_hour(self) -> int:
        return DEFAULT_MAX_WEBHOOK_FIRES_PER_SUBSCRIPTION_PER_HOUR

    @property
    def max_schedules_per_conversation(self) -> int:
        return DEFAULT_MAX_SCHEDULES_PER_CONVERSATION

    @property
    def http_allowed_hosts(self) -> tuple[str, ...]:
        return DEFAULT_HTTP_ALLOWED_HOSTS

    @property
    def loki_client(self) -> Any | None:
        return None

    @property
    def loki_named_queries(self) -> dict[str, str]:
        return DEFAULT_LOKI_NAMED_QUERIES

    @property
    def postgres_named_queries(self) -> dict[str, str]:
        return DEFAULT_POSTGRES_NAMED_QUERIES


# Platform-default :class:`WakeConfig` singleton. Caps return the
# ``DEFAULT_*`` integer constants; the named-query + HTTP-allow-list
# surfaces return their empty defaults. Used by :func:`dispatch_wake`
# and :func:`webhook_receive` when no consumer-supplied config is
# wired -- which means the platform's rate-limit + cap invariants are
# always enforced, even when the consumer forgot to plumb a config.
DEFAULT_WAKE_CONFIG: WakeConfig = _DefaultWakeConfig()
