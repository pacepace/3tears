"""what a keys-unknown record's bracket changed, read from the tables' own epochs; and the restore's budget.

The change-note ring this module once read retired with append-only L3 (``design-append-only-l3.md`` decision 3, Pace
2026-11-25): every table carries its epoch, so a bracket's keys are paged from the table's own versions (or stamped rows)
in the bracket's epoch range (:func:`~threetears.core.collections.epoch_reads.epoch_keys_page`). The names stay: the
hub's ``l3.noted_keys`` door, :class:`PoolNotedKeysReader` and the wire are the notes' until 0.95.

**Why apart from the keeper** (Pace 2026-11-15, pyarrow stays an optional extra of core): the log keeper
(:mod:`threetears.core.collections.log_keeper`) folds a restore as Arrow in a worker process, so it imports pyarrow and
NumPy at module top. Every process that follows the epoch log imports these readers; only the process that runs the
keeper needs the ``arrow`` extra. ``tests/enforcement/test_core_imports_without_arrow.py`` holds the line.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final

import asyncpg
from threetears.observe import contain, get_logger

from threetears.core.collections.epoch_reads import epoch_keys_page
from threetears.core.collections.noted_change import NotedChange
from threetears.core.collections.noted_keys import NotedKeysCursor, NotedKeysPage, NotedKeysReader

__all__ = [
    "RESTORE_BUDGET_RATIO",
    "NotedChange",
    "NotedKeysCursor",
    "NotedKeysPage",
    "NotedKeysReader",
    "PoolNotedKeysReader",
    "noted_keys_page",
]

log = get_logger(__name__)

#: the most of a change log's ``max_bytes`` a restore may fill, what the log already holds included: room left for the
#: writes that land while it runs (:mod:`threetears.core.collections.log_keeper`)
RESTORE_BUDGET_RATIO: Final = 0.75


async def noted_keys_page(
    conn: Any,
    *,
    notes_schema: str,
    table: str,
    tenant_token: str | None,
    noted_between: tuple[datetime, datetime],
    cursor: NotedKeysCursor | None,
    limit: int,
    max_bytes: int,
    every_tenant: bool = False,
) -> NotedKeysPage | str:
    """a page of the keys ``table`` changed in a commit's bracket, of ``tenant_token``'s rows, or why they cannot be
    read: the table's own versions (or stamped rows) in the bracket's epoch range
    (:func:`~threetears.core.collections.epoch_reads.epoch_keys_page`, the restore's read narrowed to the range).

    For the broker's answer to a copy follower holding a keys-unknown record whose commit named too many rows
    (:attr:`~threetears.core.collections.epoch_change.EpochChange.noted_between`), and for every
    :class:`~threetears.core.collections.noted_keys.NotedKeysReader` in this process. The read and the wire are the
    notes' until 0.95 (the request, the page, a cursor to go on from); only the cursor's place is the range's now.

    :param conn: a connection to the L3 database (the broker's, on the platform schema), or a pool
    :ptype conn: Any
    :param notes_schema: the platform schema: an undeclared table there is ``_all``'s, one in a pod's schema the pod's
    :ptype notes_schema: str
    :param table: the table's name
    :ptype table: str
    :param tenant_token: the tenant whose rows are read (the asker's schema's), ``None`` for ``_all``: never another's
    :ptype tenant_token: str | None
    :param noted_between: the commit's bracket, ``(from, until)``
    :ptype noted_between: tuple[datetime, datetime]
    :param cursor: where the last page ended; ``None`` for the first
    :ptype cursor: NotedKeysCursor | None
    :param limit: the most rows a page reads
    :ptype limit: int
    :param max_bytes: the most bytes of keys a page holds
    :ptype max_bytes: int
    :param every_tenant: read every tenant's rows of the table, ``tenant_token`` unused: for an operator's check of a
        table whose rows carry their tenant in their key (a backup's live check), never a tenant's own reader
    :ptype every_tenant: bool
    :return: the page; or ``"notes_unavailable"`` (the table carries no epoch) or ``"not_covered"`` (the cursor names
        a relation the database no longer holds)
    :rtype: NotedKeysPage | str
    """
    return await epoch_keys_page(
        conn,
        table=table,
        platform_schema=notes_schema,
        tenant_token=tenant_token,
        noted_between=noted_between,
        cursor=cursor,
        limit=limit,
        max_bytes=max_bytes,
        every_tenant=every_tenant,
    )


class PoolNotedKeysReader:
    """a :class:`NotedKeysReader` over a direct L3 pool: the notes read in this process, as the broker reads them.

    For a registry on a direct pool (a platform process: the hub's own registries, the gateway's), whose transport is
    no broker to ask (:meth:`~threetears.core.backends.nats_proxy.NatsProxyL3Backend.noted_keys`). Without one, such a
    registry stays behind on a table at any keys-unknown record carrying a notes bracket (a writer's commit too large
    to record, a reconciled lost write of more rows than one record names) until it restarts, the write's rows stale
    in it (2026-11-16). With one it pages the bracket's keys from the table's own epoch range (:func:`noted_keys_page`,
    an index range read a page at a time) and evicts exactly them: nothing dropped, nothing read whole.

    Whose notes: a tenant's (``tenant_token``, ``None`` for ``_all``), or every tenant's (``every_tenant``). A platform
    process caches every tenant's rows of the tables it follows, so it reads every tenant's notes: a key of another
    tenant the bracket names is evicted too and read again on demand, which is never wrong.

    A database role that may not read the table is answered ``"notes_unavailable"``, said at WARNING: the registry is
    then behind on the table, as it was with no reader, and its catch-up pass never fails on it.

    :param pool: the L3 pool (anything :func:`noted_keys_page` takes: a pool, or one connection)
    :ptype pool: Any
    :param notes_schema: the platform schema
    :ptype notes_schema: str
    :param tenant_token: whose notes are read, ``None`` for ``_all``; unused with ``every_tenant``
    :ptype tenant_token: str | None
    :param every_tenant: read every tenant's notes of the table
    :ptype every_tenant: bool
    """

    def __init__(
        self, pool: Any, *, notes_schema: str, tenant_token: str | None = None, every_tenant: bool = False
    ) -> None:
        self._pool = pool
        self._notes_schema = notes_schema
        self._tenant_token = tenant_token
        self._every_tenant = every_tenant

    async def noted_keys(
        self,
        table: str,
        *,
        noted_between: tuple[datetime, datetime],
        cursor: NotedKeysCursor | None,
        limit: int,
        max_bytes: int,
    ) -> NotedKeysPage | str:
        """a page of the keys ``table``'s notes name in ``noted_between`` (:meth:`NotedKeysReader.noted_keys`).

        :param table: the table, as its notes name it
        :ptype table: str
        :param noted_between: the commit's bracket, on the database's clock
        :ptype noted_between: tuple[datetime, datetime]
        :param cursor: where the last page ended; ``None`` for the first
        :ptype cursor: NotedKeysCursor | None
        :param limit: the most notes a page reads
        :ptype limit: int
        :param max_bytes: the most bytes of keys a page holds
        :ptype max_bytes: int
        :return: the page; or ``"notes_unavailable"`` (no epoch, or no grant to read it) or ``"not_covered"``
        :rtype: NotedKeysPage | str
        """
        try:
            return await noted_keys_page(
                self._pool,
                notes_schema=self._notes_schema,
                table=table,
                tenant_token=self._tenant_token,
                noted_between=noted_between,
                cursor=cursor,
                limit=limit,
                max_bytes=max_bytes,
                every_tenant=self._every_tenant,
            )
        except asyncpg.InsufficientPrivilegeError as exc:
            # NOSILENT: this role may not read the table: the registry is behind on the table, said here and there
            contain(exc, log=log, what="a direct pool's role may not read the change notes", extra={"table": table})
            return "notes_unavailable"
