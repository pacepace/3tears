"""threetears.core.utils — general-purpose helpers shared across packages.

helpers every package can rely on without pulling dependencies into core
beyond the ones core already declares (asyncpg, 3tears-observe).
"""

from threetears.core.utils.atomic_write import atomic_write, atomic_write_sync
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
from threetears.core.utils.yugabyte_rpc_timeout import (
    DEFAULT_MIN_SECONDS_BETWEEN_EXPIRIES,
    YUGABYTE_RPC_TIMEOUT_MESSAGE_PREFIX,
    YugabyteRpcTimeoutRecycler,
    is_yugabyte_rpc_timeout,
)

__all__ = [
    "DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS",
    "DEFAULT_MIN_SECONDS_BETWEEN_EXPIRIES",
    "DEFAULT_POOL_STARTUP_TIMEOUT_SECONDS",
    "ENV_MAX_INACTIVE_LIFETIME",
    "YUGABYTE_RPC_TIMEOUT_MESSAGE_PREFIX",
    "PoolStartupTimeoutError",
    "YugabyteRpcTimeoutRecycler",
    "atomic_write",
    "atomic_write_sync",
    "create_pool_with_startup_timeout",
    "get_pg_pool_kwargs",
    "is_yugabyte_rpc_timeout",
    "log_pool_created",
    "redact_dsn",
]
