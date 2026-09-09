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

import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from typing import ClassVar, Protocol, runtime_checkable

from threetears.backup.process import feed_stdin, stream_stdout

__all__ = [
    "DbDumpDriver",
    "PostgresDriver",
    "YugabyteDriver",
    "detect_driver",
    "driver_by_name",
    "driver_for_version",
]

#: the marker Yugabyte stamps into ``version()`` (e.g. "... (YugabyteDB 2.20 ... -YB-...)").
_YUGABYTE_MARKER = "-YB-"


class DbDumpDriver(ABC):
    """Abstract dump/restore driver: declares the argv, inherits the streaming."""

    name: ClassVar[str]
    #: True when the dump format is already compressed (so the engine skips gzip).
    compressed: ClassVar[bool]

    @abstractmethod
    def dump_argv(self, dsn: str, *, snapshot: str | None = None) -> list[str]:
        """Argv that dumps ``dsn`` to stdout, optionally under an exported snapshot.

        ``snapshot`` is the id returned by ``pg_export_snapshot()`` in another session, and it
        pins the dump to that session's instant. Without it the dump picks its own snapshot when
        it happens to start, which is a DIFFERENT instant from any inventory taken beside it --
        so the counts describe one database state and the bytes describe another.
        """

    @abstractmethod
    def restore_argv(self, dsn: str) -> list[str]:
        """Argv that restores into ``dsn`` from stdin."""

    def restore_pg_options(self, *, copy_rows_per_transaction: int | None = None) -> str:
        """libpq options the RESTORE session needs, as a ``PGOPTIONS`` fragment.

        Empty by default: a driver that needs nothing adds nothing, and the caller's environment
        is then passed through untouched.

        :param copy_rows_per_transaction: rows a bulk COPY may commit at once, or None for the
            server's own default
        :ptype copy_rows_per_transaction: int | None
        :return: a ``PGOPTIONS`` fragment, or the empty string
        :rtype: str
        """
        return ""

    def dump_globals_argv(self, dsn: str) -> list[str]:
        """Argv that dumps cluster GLOBALS (roles, grants, tablespaces) to stdout as plain SQL.

        Globals live outside any single database, so a per-database dump never captures them —
        a restored cluster without them has every table and none of the roles that own or may
        read them. Deliberately NOT abstract: existing driver subclasses predate cluster
        backups, and a new abstract method would break their construction. A driver that never
        overrides it refuses at call time instead.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement a globals dump")

    def restore_sql_argv(self, dsn: str) -> list[str]:
        """Argv that executes plain SQL from stdin against ``dsn`` (globals restore path)."""
        raise NotImplementedError(f"{type(self).__name__} does not implement SQL replay")

    def dump(
        self,
        dsn: str,
        *,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        snapshot: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream a dump of ``dsn`` as bytes (bounded by ``timeout`` seconds when given).

        The exported snapshot lives only as long as the transaction that made it, so a caller
        passing one must keep that transaction open until this stream is exhausted.
        """
        return stream_stdout(self.dump_argv(dsn, snapshot=snapshot), env=env, timeout=timeout)

    async def restore(
        self,
        dsn: str,
        source: AsyncIterator[bytes],
        *,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        copy_rows_per_transaction: int | None = None,
    ) -> None:
        """Restore ``source`` (a dump stream) into ``dsn`` (bounded by ``timeout`` seconds).

        :param copy_rows_per_transaction: how many rows a bulk COPY may commit at once. See
            :meth:`YugabyteDriver.restore_pg_options` for why an unbounded one is a problem.
        :ptype copy_rows_per_transaction: int | None
        """
        options = self.restore_pg_options(copy_rows_per_transaction=copy_rows_per_transaction)
        if options:
            # `feed_stdin` passes a mapping straight to the child as its WHOLE environment, so
            # building one from the fragment alone would drop PATH and the caller's PG* vars.
            # And PGOPTIONS is a space-separated list, so a caller that already set one keeps it.
            merged = dict(env) if env is not None else dict(os.environ)
            prior = merged.get("PGOPTIONS", "").strip()
            merged["PGOPTIONS"] = f"{prior} {options}".strip()
            env = merged
        await feed_stdin(self.restore_argv(dsn), source, env=env, timeout=timeout)


class PostgresDriver(DbDumpDriver):
    """Vanilla PostgreSQL via ``pg_dump`` (custom format) + ``pg_restore``."""

    name: ClassVar[str] = "postgres"
    compressed: ClassVar[bool] = True  # pg_dump custom format is zlib-compressed already

    def dump_argv(self, dsn: str, *, snapshot: str | None = None) -> list[str]:
        argv = ["pg_dump", "--dbname", dsn, "--format=custom", "--no-owner", "--no-privileges"]
        if snapshot is not None:
            argv.append(f"--snapshot={snapshot}")
        return argv

    def restore_argv(self, dsn: str) -> list[str]:
        # a fresh (empty) target — the verifier's temp db — so no --clean is needed; fail loudly.
        return ["pg_restore", "--dbname", dsn, "--no-owner", "--no-privileges", "--exit-on-error"]

    def dump_globals_argv(self, dsn: str) -> list[str]:
        return ["pg_dumpall", "--dbname", dsn, "--globals-only", "--no-role-passwords"]

    def restore_sql_argv(self, dsn: str) -> list[str]:
        return ["psql", "--dbname", dsn, "--quiet", "--set", "ON_ERROR_STOP=1"]


class YugabyteDriver(DbDumpDriver):
    """YugabyteDB via ``ysql_dump`` (plain SQL) + ``ysqlsh``."""

    name: ClassVar[str] = "yugabyte"
    compressed: ClassVar[bool] = False  # ysql_dump emits plain SQL — gzip it

    def dump_argv(self, dsn: str, *, snapshot: str | None = None) -> list[str]:
        argv = ["ysql_dump", "--dbname", dsn, "--no-owner", "--no-privileges"]
        if snapshot is not None:
            # Yugabyte's fork defaults serializable-deferrable ON, where upstream pg_dump makes
            # it opt-in, and Yugabyte then REFUSES `SET TRANSACTION SNAPSHOT` in a serializable
            # transaction: "cannot export/import snapshot in SERIALIZABLE Isolation Level".
            # Without this flag every dump on Yugabyte fails outright while the same code passes
            # against Postgres, so no Postgres-backed test can see it.
            #
            # The trade is deliberate. Serializable-deferrable waits for a view free of
            # serialization anomalies; importing the inventory's snapshot instead gives the dump
            # the exact instant its row counts describe. For a backup whose verification compares
            # against that inventory, the same instant is the property that matters.
            argv.append("--no-serializable-deferrable")
            argv.append(f"--snapshot={snapshot}")
        return argv

    def restore_argv(self, dsn: str) -> list[str]:
        # ysqlsh reads SQL from stdin; ON_ERROR_STOP makes a bad statement a non-zero exit.
        return ["ysqlsh", "--dbname", dsn, "--quiet", "--set", "ON_ERROR_STOP=1"]

    def restore_pg_options(self, *, copy_rows_per_transaction: int | None = None) -> str:
        """Bound how much one bulk COPY commits at a time.

        THE BUG THIS EXISTS FOR took every restore down on a real cluster, and the error it
        produced named none of this. Yugabyte batches ``COPY`` by
        ``yb_default_copy_from_rows_per_transaction`` -- a ROW COUNT, default 20000, with no
        regard for how big a row is. Measured on one live set: the LangGraph ``checkpoints``
        tables average 115 KB per row and peak near 196 KB, so the default asks the server to
        commit a 2.3 GB transaction. The tserver's whole inbound RPC read buffer is about
        365 MB (5% of a 6.8 GB hard limit, shared with every other caller), so it refuses:

            Service unavailable: Call rejected due to memory pressure:
            yb.tserver.TabletServerService.Write

        and the client sees only the follow-on, ``Predecessor request for N was not applied``,
        against whichever table it happened to reach. That is why it looked non-deterministic
        and why no server log named a cause.

        The right bound is BYTES. Yugabyte only offers rows, so the value has to be chosen
        assuming rows are large: at the 196 KB worst case observed, 100 rows is about 20 MB,
        which leaves the buffer room for everything else on the node. Measured on the same
        3 GB dump: 20000 and 1000 both failed, 100 restored cleanly in 812s.

        It costs time -- more transactions, more round trips -- and that is the trade. A slow
        restore that finishes beats a fast one that does not.

        :param copy_rows_per_transaction: rows one COPY may commit at once, or None to leave
            the server's own default in force
        :ptype copy_rows_per_transaction: int | None
        :return: the ``PGOPTIONS`` fragment, or empty when the default is being left alone
        :rtype: str
        """
        options = ""
        if copy_rows_per_transaction is not None and copy_rows_per_transaction > 0:
            options = f"-c yb_default_copy_from_rows_per_transaction={copy_rows_per_transaction}"
        return options

    def dump_globals_argv(self, dsn: str) -> list[str]:
        # Yugabyte ships its own dumpall fork; role passwords are deliberately excluded on both
        # drivers — a backup must not become a credential store.
        return ["ysql_dumpall", "--dbname", dsn, "--globals-only", "--no-role-passwords"]

    def restore_sql_argv(self, dsn: str) -> list[str]:
        return ["ysqlsh", "--dbname", dsn, "--quiet", "--set", "ON_ERROR_STOP=1"]


def driver_by_name(name: str) -> DbDumpDriver:
    """Construct the driver a manifest names.

    A restore must run the SAME driver that wrote the dump — the two formats are not
    interchangeable (custom-archive vs gzipped plain SQL) — so the manifest records the name and
    restores resolve it here rather than trusting whatever driver the restoring process happens
    to hold.

    :param name: a driver's ``name`` classvar, as stored in a manifest.
    :raises ValueError: on a name no driver claims.
    """
    for cls in (PostgresDriver, YugabyteDriver):
        if cls.name == name:
            return cls()
    raise ValueError(f"no dump driver named {name!r}")


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
