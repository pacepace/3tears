"""DuckDB's default database let down to one thread, by every module here that imports DuckDB.

``import duckdb`` opens a default in-memory database (the one ``duckdb.sql`` and ``duckdb.execute`` use) with a
thread a core, started as it loads: 15 on a 16-core runner, 63 on a 64-core one whatever its cpuset, idle in every
process that imports DuckDB, the fork server included (measured on the pool, DuckDB 1.5.6, 2026-11-25). Nothing here
uses that database: every query runs on a connection of its own, sized by its caller. Those threads counted against
a container's pid limit beside every test process's own (monad-1: 4096, 15 test processes and five fork servers at
3,148 threads, its test processes aborting in Arrow as a thread failed to start), and against a pod's.

Every module that imports DuckDB calls :func:`release_default_threads` right after (its own ``import duckdb`` stays,
so a process without the extra fails there, as it always did), so whichever imports it first lets the threads go.
"""

from __future__ import annotations

import duckdb

__all__ = ["release_default_threads"]


def release_default_threads() -> None:
    """set DuckDB's default database to one thread: its scheduler stops and joins the threads past one at once (15 to
    none on a 16-core runner, 2026-11-25). Cheap, and the same on every call.

    :return: nothing
    :rtype: None
    """
    duckdb.execute("SET threads = 1")
