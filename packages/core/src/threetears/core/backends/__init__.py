"""L3 durable-tier backends + protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING

from threetears.core._extras import import_extras_module

# lazy public API (PEP 562), as in ``threetears.core``: importing this package costs only this file,
# and each name resolves its defining module on first access, so a caller that needs one light
# module here does not load the extras-backed ones beside it. the TYPE_CHECKING block carries the
# real imports for mypy and IDEs; ``_LAZY`` is the runtime equivalent.
if TYPE_CHECKING:
    from threetears.core.backends.broker_generation import BrokerGenerationSource, GenerationReader
    from threetears.core.backends.nats_proxy import NatsProxyL3Backend, check_reply_row_count
    from threetears.core.backends.protocol import (
        DurableStore,
        L3Backend,
        L3Reader,
        RowcountUnreadableError,
        parse_rowcount,
    )
    from threetears.core.backends.sql import SqlL3Backend, bound_request_connection

_LAZY: dict[str, tuple[str, str]] = {
    "BrokerGenerationSource": ("threetears.core.backends.broker_generation", "BrokerGenerationSource"),
    "GenerationReader": ("threetears.core.backends.broker_generation", "GenerationReader"),
    "NatsProxyL3Backend": ("threetears.core.backends.nats_proxy", "NatsProxyL3Backend"),
    "check_reply_row_count": ("threetears.core.backends.nats_proxy", "check_reply_row_count"),
    "DurableStore": ("threetears.core.backends.protocol", "DurableStore"),
    "L3Backend": ("threetears.core.backends.protocol", "L3Backend"),
    "L3Reader": ("threetears.core.backends.protocol", "L3Reader"),
    "RowcountUnreadableError": ("threetears.core.backends.protocol", "RowcountUnreadableError"),
    "parse_rowcount": ("threetears.core.backends.protocol", "parse_rowcount"),
    "SqlL3Backend": ("threetears.core.backends.sql", "SqlL3Backend"),
    "bound_request_connection": ("threetears.core.backends.sql", "bound_request_connection"),
}

__all__ = [
    "BrokerGenerationSource",
    "GenerationReader",
    "DurableStore",
    "L3Backend",
    "L3Reader",
    "NatsProxyL3Backend",
    "RowcountUnreadableError",
    "SqlL3Backend",
    "bound_request_connection",
    "check_reply_row_count",
    "parse_rowcount",
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
