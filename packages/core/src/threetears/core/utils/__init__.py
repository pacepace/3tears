"""threetears.core.utils — general-purpose helpers shared across packages.

contains stdlib-only utilities that every package can rely on without
pulling additional dependencies into core.
"""

from threetears.core.utils.atomic_write import atomic_write
from threetears.core.utils.pg_pool_kwargs import (
    DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS,
    DEFAULT_POOL_STARTUP_TIMEOUT_SECONDS,
    ENV_MAX_INACTIVE_LIFETIME,
    PoolStartupTimeoutError,
    create_pool_with_startup_timeout,
    get_pg_pool_kwargs,
    log_pool_created,
    redact_dsn,
)

__all__ = [
    "DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS",
    "DEFAULT_POOL_STARTUP_TIMEOUT_SECONDS",
    "ENV_MAX_INACTIVE_LIFETIME",
    "PoolStartupTimeoutError",
    "atomic_write",
    "create_pool_with_startup_timeout",
    "get_pg_pool_kwargs",
    "log_pool_created",
    "redact_dsn",
]
