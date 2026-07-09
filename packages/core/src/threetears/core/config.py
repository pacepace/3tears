"""Protocol-based configuration for 3tears core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
    "DEFAULT_HTTP_TIMEOUT_SECONDS",
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
