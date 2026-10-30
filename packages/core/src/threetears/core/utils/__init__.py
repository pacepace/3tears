"""threetears.core.utils — general-purpose helpers shared across packages.

helpers every package can rely on without pulling dependencies into core
beyond the ones core already declares (asyncpg, 3tears-observe).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from threetears.core._extras import import_extras_module

# lazy public API (PEP 562), as in ``threetears.core``: importing this package costs only this file,
# and each name resolves its defining module on first access, so a caller that needs one light
# module here does not load the extras-backed ones beside it. the TYPE_CHECKING block carries the
# real imports for mypy and IDEs; ``_LAZY`` is the runtime equivalent.
if TYPE_CHECKING:
    from threetears.core.utils.atomic_write import atomic_write, atomic_write_sync
    from threetears.core.utils.local_time import parse_date_bound, render_local_time, resolve_zone
    from threetears.core.utils.pg_pool_kwargs import (
        DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS,
        DEFAULT_POOL_CONNECT_TIMEOUT_SECONDS,
        DEFAULT_POOL_STARTUP_TIMEOUT_SECONDS,
        ENV_MAX_INACTIVE_LIFETIME,
        PoolStartupTimeoutError,
        create_pool_with_startup_timeout,
        get_pg_pool_kwargs,
        log_pool_created,
        redact_dsn,
        resolve_pool_connect_timeout,
    )
    from threetears.core.utils.session_settings import session_settings_connection_class
    from threetears.core.utils.yugabyte_pool_recycler import (
        DEFAULT_MIN_SECONDS_BETWEEN_EXPIRIES,
        YUGABYTE_POOL_TRIGGERS,
        YUGABYTE_RPC_TIMEOUT,
        YUGABYTE_RPC_TIMEOUT_MESSAGE_PREFIX,
        YUGABYTE_STALE_TABLE_SHAPE,
        PoolExpiryTrigger,
        YugabytePoolRecycler,
        is_yugabyte_rpc_timeout,
        is_yugabyte_stale_table_shape,
    )

_LAZY: dict[str, tuple[str, str]] = {
    "atomic_write": ("threetears.core.utils.atomic_write", "atomic_write"),
    "atomic_write_sync": ("threetears.core.utils.atomic_write", "atomic_write_sync"),
    "parse_date_bound": ("threetears.core.utils.local_time", "parse_date_bound"),
    "render_local_time": ("threetears.core.utils.local_time", "render_local_time"),
    "resolve_zone": ("threetears.core.utils.local_time", "resolve_zone"),
    "DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS": (
        "threetears.core.utils.pg_pool_kwargs",
        "DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS",
    ),
    "DEFAULT_POOL_CONNECT_TIMEOUT_SECONDS": (
        "threetears.core.utils.pg_pool_kwargs",
        "DEFAULT_POOL_CONNECT_TIMEOUT_SECONDS",
    ),
    "DEFAULT_POOL_STARTUP_TIMEOUT_SECONDS": (
        "threetears.core.utils.pg_pool_kwargs",
        "DEFAULT_POOL_STARTUP_TIMEOUT_SECONDS",
    ),
    "ENV_MAX_INACTIVE_LIFETIME": ("threetears.core.utils.pg_pool_kwargs", "ENV_MAX_INACTIVE_LIFETIME"),
    "PoolStartupTimeoutError": ("threetears.core.utils.pg_pool_kwargs", "PoolStartupTimeoutError"),
    "create_pool_with_startup_timeout": ("threetears.core.utils.pg_pool_kwargs", "create_pool_with_startup_timeout"),
    "get_pg_pool_kwargs": ("threetears.core.utils.pg_pool_kwargs", "get_pg_pool_kwargs"),
    "log_pool_created": ("threetears.core.utils.pg_pool_kwargs", "log_pool_created"),
    "redact_dsn": ("threetears.core.utils.pg_pool_kwargs", "redact_dsn"),
    "resolve_pool_connect_timeout": ("threetears.core.utils.pg_pool_kwargs", "resolve_pool_connect_timeout"),
    "session_settings_connection_class": (
        "threetears.core.utils.session_settings",
        "session_settings_connection_class",
    ),
    "DEFAULT_MIN_SECONDS_BETWEEN_EXPIRIES": (
        "threetears.core.utils.yugabyte_pool_recycler",
        "DEFAULT_MIN_SECONDS_BETWEEN_EXPIRIES",
    ),
    "YUGABYTE_POOL_TRIGGERS": ("threetears.core.utils.yugabyte_pool_recycler", "YUGABYTE_POOL_TRIGGERS"),
    "YUGABYTE_RPC_TIMEOUT": ("threetears.core.utils.yugabyte_pool_recycler", "YUGABYTE_RPC_TIMEOUT"),
    "YUGABYTE_RPC_TIMEOUT_MESSAGE_PREFIX": (
        "threetears.core.utils.yugabyte_pool_recycler",
        "YUGABYTE_RPC_TIMEOUT_MESSAGE_PREFIX",
    ),
    "YUGABYTE_STALE_TABLE_SHAPE": ("threetears.core.utils.yugabyte_pool_recycler", "YUGABYTE_STALE_TABLE_SHAPE"),
    "PoolExpiryTrigger": ("threetears.core.utils.yugabyte_pool_recycler", "PoolExpiryTrigger"),
    "YugabytePoolRecycler": ("threetears.core.utils.yugabyte_pool_recycler", "YugabytePoolRecycler"),
    "is_yugabyte_rpc_timeout": ("threetears.core.utils.yugabyte_pool_recycler", "is_yugabyte_rpc_timeout"),
    "is_yugabyte_stale_table_shape": ("threetears.core.utils.yugabyte_pool_recycler", "is_yugabyte_stale_table_shape"),
}

__all__ = [
    "DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS",
    "DEFAULT_MIN_SECONDS_BETWEEN_EXPIRIES",
    "DEFAULT_POOL_CONNECT_TIMEOUT_SECONDS",
    "DEFAULT_POOL_STARTUP_TIMEOUT_SECONDS",
    "ENV_MAX_INACTIVE_LIFETIME",
    "YUGABYTE_POOL_TRIGGERS",
    "YUGABYTE_RPC_TIMEOUT",
    "YUGABYTE_RPC_TIMEOUT_MESSAGE_PREFIX",
    "YUGABYTE_STALE_TABLE_SHAPE",
    "PoolExpiryTrigger",
    "PoolStartupTimeoutError",
    "YugabytePoolRecycler",
    "atomic_write",
    "atomic_write_sync",
    "create_pool_with_startup_timeout",
    "get_pg_pool_kwargs",
    "is_yugabyte_rpc_timeout",
    "is_yugabyte_stale_table_shape",
    "log_pool_created",
    "redact_dsn",
    "parse_date_bound",
    "render_local_time",
    "resolve_pool_connect_timeout",
    "resolve_zone",
    "session_settings_connection_class",
]


def __getattr__(name: str) -> object:
    """resolve a public attribute from its defining module on first access.

    :param name: attribute name being resolved
    :ptype name: str
    :return: the resolved attribute (also cached in module globals so
        ``__getattr__`` fires at most once per name)
    :rtype: object
    :raises AttributeError: when ``name`` is not part of the public API
    :raises ImportError: when the defining module needs an extra that is not installed,
        naming the extra
    """
    entry = _LAZY.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = entry
    value: object = getattr(import_extras_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """include lazy attributes in ``dir()`` output.

    :return: sorted union of materialized globals and lazy names
    :rtype: list[str]
    """
    return sorted(set(globals()) | set(_LAZY))
