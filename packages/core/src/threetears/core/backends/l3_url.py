"""one setting picks the L3: a ``postgresql://`` DSN, or a ``sqlite:///`` path.

A deployment that runs on one machine and one that runs against Postgres build the same collections; which L3 they
hold is the URL in their configuration::

    async with l3_store(config.l3_url) as store:
        registry.configure(l1_backend=SQLiteBackend("app"), l3_pool=store)
        ...

``sqlite:////var/lib/app/state.sqlite`` opens a :class:`~threetears.core.backends.sqlite_file.SqliteL3Backend` on
that absolute path; ``sqlite:///state.sqlite`` names a path relative to the working directory (SQLAlchemy's reading:
what follows the third slash is the path as written). ``postgresql://`` or ``postgres://`` starts an asyncpg pool the platform's way
(:func:`~threetears.core.utils.pg_pool_kwargs.create_pool_with_startup_timeout`, every connection initialised with
the platform's codecs, :func:`~threetears.core.collections.asyncpg_init.init_connection`) and wraps it in
:class:`~threetears.core.backends.sql.SqlL3Backend`. Leaving the block closes the file or the pool.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.backends.sqlite_file import SqliteL3Backend
from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.utils.pg_pool_kwargs import create_pool_with_startup_timeout

__all__ = ["l3_store"]

_POSTGRES_SCHEMES = frozenset({"postgresql", "postgres"})


def _sqlite_path(url: str) -> Path:
    """the file a ``sqlite:///`` URL names.

    :param url: the URL
    :ptype url: str
    :return: the path
    :rtype: Path
    :raises ValueError: when ``url`` is not a ``sqlite:///`` URL naming a path
    """
    parts = urlsplit(url)
    if parts.scheme != "sqlite" or parts.netloc or not parts.path or parts.path == "/":
        raise ValueError(f"an SQLite L3 is named sqlite:///<path>; got {url!r}")
    # what follows the third slash is the path as written: sqlite:///rel is relative, sqlite:////abs absolute
    return Path(unquote(parts.path)[1:])


@contextlib.asynccontextmanager
async def l3_store(
    url: str, *, pool_name: str = "l3", **pool_kwargs: Any
) -> AsyncIterator[SqlL3Backend | SqliteL3Backend]:
    """the L3 ``url`` names, open for the block and closed after it.

    :param url: ``sqlite:///<path>``, or a ``postgresql://`` DSN
    :ptype url: str
    :param pool_name: the pool's name in logs and errors (Postgres)
    :ptype pool_name: str
    :param pool_kwargs: what else :func:`create_pool_with_startup_timeout` takes (``min_size``, ``server_settings``
        ...); refused for SQLite, which has no pool
    :ptype pool_kwargs: Any
    :return: a :class:`SqliteL3Backend` or a :class:`SqlL3Backend`
    :rtype: AsyncIterator[SqlL3Backend | SqliteL3Backend]
    :raises ValueError: when the scheme is neither, or pool options are given for SQLite
    """
    scheme = urlsplit(url).scheme
    if scheme == "sqlite":
        if pool_kwargs:
            raise ValueError(f"an SQLite L3 has no pool; options {sorted(pool_kwargs)} do not apply")
        async with SqliteL3Backend(_sqlite_path(url)) as store:
            yield store
        return
    if scheme not in _POSTGRES_SCHEMES:
        raise ValueError(f"an L3 URL is sqlite:///<path> or postgresql://...; got the scheme {scheme!r}")
    pool_kwargs.setdefault("init", init_connection)
    pool = await create_pool_with_startup_timeout(url, pool_name=pool_name, **pool_kwargs)
    try:
        yield SqlL3Backend(pool)
    finally:
        await pool.close()
