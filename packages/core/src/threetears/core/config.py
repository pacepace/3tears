"""Protocol-based configuration for 3tears core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
    "DEFAULT_BRIDGE_LOOP_START_TIMEOUT_SECONDS",
    "DEFAULT_EGRESS_HEALTH_TIMEOUT_SECONDS",
    "DEFAULT_HTTP_TIMEOUT_SECONDS",
    "DEFAULT_POOL_CONNECT_TIMEOUT_SECONDS",
    "DEFAULT_POOL_STARTUP_TIMEOUT_SECONDS",
    "POOL_START_RETRY_FIRST_DELAY_SECONDS",
    "POOL_START_RETRY_MAX_DELAY_SECONDS",
    "CoreConfig",
    "DefaultCoreConfig",
    "VALID_FLUSH_STRATEGIES",
]

VALID_FLUSH_STRATEGIES = frozenset({"ALWAYS", "ON_CHECKPOINT", "ON_SCHEDULE", "ON_SHUTDOWN"})

# default per-request timeout (seconds) for the core outbound HTTP transport
# (:class:`threetears.core.http_client.TracedHttpClient`). the default lives
# here, the designated core config layer, so the transport signature carries
# no hardcoded timeout literal.
DEFAULT_HTTP_TIMEOUT_SECONDS = 30.0

# budget for an egress health probe (:meth:`threetears.core.egress.EgressDriver.health`).
# deliberately shorter than DEFAULT_HTTP_TIMEOUT_SECONDS: this probe runs to answer "is the
# exit up", and a check that hangs as long as a real request tells an operator nothing they
# could not already see from the requests themselves.
DEFAULT_EGRESS_HEALTH_TIMEOUT_SECONDS = 10.0

# startup budget for a PostgreSQL pool (:func:`threetears.core.utils.pg_pool_kwargs
# .create_pool_with_startup_timeout`). the same wall clock as the NATS startup
# budget (``threetears.nats.DEFAULT_STARTUP_TIMEOUT``), pinned equal by a test in
# the nats package, which is the one that can import both. lives here, the core
# config layer, because core cannot import nats and a literal beside the pool
# helper was a timeout constant the hardcoded-timeout gate had to be widened to
# see.
DEFAULT_POOL_STARTUP_TIMEOUT_SECONDS = 30.0

# ceiling on ONE connect while a PostgreSQL pool starts (asyncpg's per-connect ``timeout``). the
# pool helper uses the smaller of this and a third of the startup budget, so a connect whose
# backend never answers costs at most a third of the budget and the rest is left for retries.
# asyncpg's own default is 60s -- twice the startup budget -- which is how one stalled backend
# used to fail a whole pool start with nothing retried. 10s matches what the hub L3 pool had
# already set by hand; a healthy connect, TLS and auth included, takes milliseconds.
DEFAULT_POOL_CONNECT_TIMEOUT_SECONDS = 10.0

# the pause between pool-start attempts: the first, and the ceiling it doubles up to. a pause
# never runs past the startup budget (``threetears.observe.retry_bounded`` clamps it).
POOL_START_RETRY_FIRST_DELAY_SECONDS = 0.5
POOL_START_RETRY_MAX_DELAY_SECONDS = 4.0

# how long the first caller of the sync-to-async bridge (:mod:`threetears.core._bridge`)
# waits for the new background loop to be running before reporting it failed to start. a
# healthy start takes milliseconds; the bound exists so a loop that never starts is an
# error naming itself, not a caller hung forever holding the lock every later caller needs.
DEFAULT_BRIDGE_LOOP_START_TIMEOUT_SECONDS = 30.0


@runtime_checkable
class CoreConfig(Protocol):
    """Protocol that any configuration object must satisfy."""

    collection_flush: str  # ALWAYS | ON_CHECKPOINT | ON_SCHEDULE | ON_SHUTDOWN
    collection_flush_interval: int  # seconds
    collection_flush_tables: str  # comma-separated table names


@dataclass
class DefaultCoreConfig:
    """Concrete default configuration."""

    collection_flush: str = "ON_CHECKPOINT"
    collection_flush_interval: int = 30
    collection_flush_tables: str = "messages,token_usage_logs"

    def __post_init__(self) -> None:
        if self.collection_flush not in VALID_FLUSH_STRATEGIES:
            raise ValueError(
                f"collection_flush must be one of {sorted(VALID_FLUSH_STRATEGIES)}, got {self.collection_flush!r}"
            )
