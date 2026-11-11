"""the one spelling of a keyed read's filters, shared by every helper in :mod:`threetears.core.keyed`.

The filters, the names and the vocabulary they are written in are pure, and live in the dependency-free
:mod:`threetears.interfaces.keyed.filters` (re-exported here under the names every helper imports); what stays here
is the one question a filter asks of the connection it is sent on: whether it binds typed arrays natively.
"""

from __future__ import annotations

from typing import Any

import asyncpg
from asyncpg.pool import PoolConnectionProxy

from threetears.interfaces.keyed.filters import (
    IN_SUBQUERY,
    AnyOf,
    Derived,
    Fragment,
    Lock,
    Selected,
    batched_join_hint,
    by_conditions,
    narrowing,
    prefix_conditions,
    quoted_table,
    relation,
    renumbered,
    selected_column,
)

__all__ = [
    "IN_SUBQUERY",
    "AnyOf",
    "Derived",
    "Fragment",
    "Lock",
    "Selected",
    "batched_join_hint",
    "binds_native_arrays",
    "by_conditions",
    "prefix_conditions",
    "quoted_table",
    "selected_column",
    "relation",
    "narrowing",
    "renumbered",
]


def binds_native_arrays(conn: Any) -> bool:
    """whether ``conn`` binds a typed array natively (``$1::bigint[]`` from a list of ints): asyncpg's own
    connection, pool or pooled connection, or anything saying so (``binds_native_arrays = True``: a wrapper of one).
    The L3 rail carries its parameters as JSON, so a keyed read through it binds text arrays cast to the column's
    type (``$1::text[]::bigint[]``); identity, not truthiness, so a mock's auto-attribute never reads as yes.

    :param conn: the connection a keyed helper reads through
    :ptype conn: Any
    :return: whether its arrays bind natively
    :rtype: bool
    """
    if isinstance(conn, asyncpg.Connection | asyncpg.Pool | PoolConnectionProxy):
        return True
    return getattr(conn, "binds_native_arrays", None) is True
