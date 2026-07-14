"""Pluggable database dump/restore drivers, with Postgres/Yugabyte autodetection.

A driver knows one thing the engine doesn't: which command-line tool dumps and restores its
database, and with which flags. :class:`PostgresDriver` uses ``pg_dump``/``pg_restore`` (custom
archive format); :class:`YugabyteDriver` uses ``ysql_dump``/``ysqlsh`` (plain SQL) — Yugabyte
ships its own fork of the tools. The engine picks one by asking the database ``SELECT version()``:
Yugabyte stamps ``-YB-`` into its version string (the same tell scriob's ``is_yugabyte`` uses).

The argv builders and :func:`driver_for_version` are pure (unit-tested without a database); the
actual dump/restore streams through the shared subprocess plumbing in :mod:`threetears.backup.process`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from typing import ClassVar, Protocol, runtime_checkable

from threetears.backup.process import feed_stdin, stream_stdout

__all__ = ["DbDumpDriver", "PostgresDriver", "YugabyteDriver", "detect_driver", "driver_for_version"]

#: the marker Yugabyte stamps into ``version()`` (e.g. "... (YugabyteDB 2.20 ... -YB-...)").
_YUGABYTE_MARKER = "-YB-"


class DbDumpDriver(ABC):
    """Abstract dump/restore driver: declares the argv, inherits the streaming."""

    name: ClassVar[str]
    #: True when the dump format is already compressed (so the engine skips gzip).
    compressed: ClassVar[bool]

    @abstractmethod
    def dump_argv(self, dsn: str) -> list[str]:
        """Argv that dumps ``dsn`` to stdout."""

    @abstractmethod
    def restore_argv(self, dsn: str) -> list[str]:
        """Argv that restores into ``dsn`` from stdin."""

    def dump(self, dsn: str, *, env: Mapping[str, str] | None = None) -> AsyncIterator[bytes]:
        """Stream a dump of ``dsn`` as bytes."""
        return stream_stdout(self.dump_argv(dsn), env=env)

    async def restore(self, dsn: str, source: AsyncIterator[bytes], *, env: Mapping[str, str] | None = None) -> None:
        """Restore ``source`` (a dump stream) into ``dsn``."""
        await feed_stdin(self.restore_argv(dsn), source, env=env)


class PostgresDriver(DbDumpDriver):
    """Vanilla PostgreSQL via ``pg_dump`` (custom format) + ``pg_restore``."""

    name: ClassVar[str] = "postgres"
    compressed: ClassVar[bool] = True  # pg_dump custom format is zlib-compressed already

    def dump_argv(self, dsn: str) -> list[str]:
        return ["pg_dump", "--dbname", dsn, "--format=custom", "--no-owner", "--no-privileges"]

    def restore_argv(self, dsn: str) -> list[str]:
        # a fresh (empty) target — the verifier's temp db — so no --clean is needed; fail loudly.
        return ["pg_restore", "--dbname", dsn, "--no-owner", "--no-privileges", "--exit-on-error"]


class YugabyteDriver(DbDumpDriver):
    """YugabyteDB via ``ysql_dump`` (plain SQL) + ``ysqlsh``."""

    name: ClassVar[str] = "yugabyte"
    compressed: ClassVar[bool] = False  # ysql_dump emits plain SQL — gzip it

    def dump_argv(self, dsn: str) -> list[str]:
        return ["ysql_dump", "--dbname", dsn, "--no-owner", "--no-privileges"]

    def restore_argv(self, dsn: str) -> list[str]:
        # ysqlsh reads SQL from stdin; ON_ERROR_STOP makes a bad statement a non-zero exit.
        return ["ysqlsh", "--dbname", dsn, "--quiet", "--set", "ON_ERROR_STOP=1"]


def driver_for_version(version: str) -> DbDumpDriver:
    """Pick a driver from a ``version()`` string.

    :param version: the output of ``SELECT version()``.
    :return: a :class:`YugabyteDriver` if the string carries the Yugabyte marker, else Postgres.
    """
    return YugabyteDriver() if _YUGABYTE_MARKER in version else PostgresDriver()


@runtime_checkable
class _VersionSource(Protocol):
    async def fetchval(self, query: str) -> object: ...


async def detect_driver(conn: _VersionSource) -> DbDumpDriver:
    """Autodetect the driver by querying ``version()`` on an open connection.

    :param conn: anything with an async ``fetchval(query)`` (e.g. an asyncpg connection).
    :return: the driver matching the connected database engine.
    """
    version = await conn.fetchval("SELECT version()")
    return driver_for_version(str(version))
