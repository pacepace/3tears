"""a test's own database: made fresh, dropped after, and any its earlier runs left behind swept first.

A fixture that makes a database per run and drops it at teardown leaves it behind when the run is killed (a timeout,
an interrupted session): such databases pile up on the shared server. :func:`scratch_database` names each one by a
prefix and a random suffix, and at start drops every database of that prefix nothing is connected to, so a killed
run's leftover goes on the next run. A database some session still uses (a run in progress) is never touched: each
run holds a connection to its own database for as long as it uses it, and the sweep and the create run under one
advisory lock per prefix, so no sweep sees a database between its create and that connection.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import asyncpg

__all__ = ["scratch_database"]

#: a prefix: lower-case letters and underscores, ending in an underscore (``reports_it_``)
_PREFIX = re.compile(r"[a-z_]+_")

#: what a run's suffix is: eight hex digits
_SUFFIX = "[0-9a-f]{8}"

#: the prefix's databases nothing is connected to (``_`` escaped: the prefix is matched literally)
_IDLE = (
    "SELECT d.datname FROM pg_database d WHERE d.datname LIKE $1 "
    "AND NOT EXISTS (SELECT 1 FROM pg_stat_activity a WHERE a.datname = d.datname)"
)


def _database_dsn(dsn: str, name: str) -> str:
    """``dsn`` pointed at database ``name`` (its user, host, port and options kept).

    :param dsn: a ``postgresql://`` DSN
    :ptype dsn: str
    :param name: the database
    :ptype name: str
    :return: the DSN of that database
    :rtype: str
    """
    parts = urlsplit(dsn)
    return urlunsplit(parts._replace(path=f"/{name}"))


async def _sweep(admin: Any, prefix: str) -> list[str]:
    """drop every database of ``prefix`` and a run's suffix that nothing is connected to.

    :param admin: a connection to the server's admin database
    :ptype admin: Any
    :param prefix: the prefix
    :ptype prefix: str
    :return: the databases dropped
    :rtype: list[str]
    """
    pattern = re.compile(f"{re.escape(prefix)}{_SUFFIX}")
    rows = await admin.fetch(_IDLE, prefix.replace("_", "\\_") + "%")
    dropped: list[str] = []
    for row in rows:
        name = row["datname"]
        if pattern.fullmatch(name) is None:
            continue
        try:
            # no FORCE: a session that connected since the read keeps its database
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}"')
        except asyncpg.ObjectInUseError:
            continue
        dropped.append(name)
    return dropped


@asynccontextmanager
async def scratch_database(
    admin_dsn: str,
    prefix: str,
    *,
    connect: Callable[[str], Awaitable[Any]] = asyncpg.connect,
) -> AsyncIterator[str]:
    """a fresh database for one test run: yields its DSN, and drops it after (with every connection still open to it),
    whether the run passed or raised. Before it is made, the leftovers of earlier runs of the same prefix that nothing
    is connected to are dropped.

    :param admin_dsn: a DSN of the server's admin database, as a role that may create and drop databases
    :ptype admin_dsn: str
    :param prefix: the run's prefix, lower-case letters and underscores ending in ``_`` (``reports_it_``); the
        database is the prefix and eight hex digits
    :ptype prefix: str
    :param connect: opens a connection to a DSN (``asyncpg.connect``)
    :ptype connect: Callable[[str], Awaitable[Any]]
    :return: the database's DSN
    :rtype: AsyncIterator[str]
    :raises ValueError: when the prefix is not lower-case letters and underscores ending in ``_``
    """
    if _PREFIX.fullmatch(prefix) is None:
        raise ValueError(f"a scratch database prefix is lower-case letters and underscores ending in '_': {prefix!r}")
    name = f"{prefix}{uuid.uuid4().hex[:8]}"
    dsn = _database_dsn(admin_dsn, name)
    admin = await connect(admin_dsn)
    holder: Any = None
    try:
        # one sweep or create of this prefix at a time: a sweep never sees a run's database before it is held
        await admin.execute("SELECT pg_advisory_lock(hashtext($1))", prefix)
        try:
            await _sweep(admin, prefix)
            await admin.execute(f'CREATE DATABASE "{name}"')
            holder = await connect(dsn)
        finally:
            await admin.execute("SELECT pg_advisory_unlock(hashtext($1))", prefix)
    except BaseException:
        await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        await admin.close()
        raise
    await admin.close()
    try:
        yield dsn
    finally:
        await holder.close()
        admin = await connect(admin_dsn)
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        finally:
            await admin.close()
