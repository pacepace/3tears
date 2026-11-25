"""The declared queries and prepare hook the copy query pool's tests hand its workers: module-level, so the workers
import them by name."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any

#: the view the prepare hook makes: each state's votes
TOTALS_VIEW = "state_totals"


def views(connection: Any) -> None:
    """a product's views over the copy, made on each query's connection.

    :param connection: the query's connection
    :ptype connection: duckdb.DuckDBPyConnection
    :return: nothing
    :rtype: None
    """
    connection.execute(
        f"CREATE VIEW {TOTALS_VIEW} AS SELECT state, sum(votes)::BIGINT AS votes FROM results GROUP BY state"
    )


def gated(connection: Any, binds: Sequence[Any]) -> Any:
    """wait until a file exists (60 s at most), then run SQL: a query running as long as a test wants.

    :param connection: the query's connection
    :ptype connection: duckdb.DuckDBPyConnection
    :param binds: the file, the SQL, then the SQL's binds
    :ptype binds: Sequence[Any]
    :return: the SQL's answer
    :rtype: duckdb.DuckDBPyConnection
    """
    path, sql = binds[0], binds[1]
    deadline = time.monotonic() + 60
    while not os.path.exists(path) and time.monotonic() < deadline:
        time.sleep(0.01)
    return connection.execute(sql, list(binds[2:]))


def touch(connection: Any, binds: Sequence[Any]) -> Any:
    """make a file, then answer one row: a query that says it ran.

    :param connection: the query's connection
    :ptype connection: duckdb.DuckDBPyConnection
    :param binds: the file
    :ptype binds: Sequence[Any]
    :return: one row
    :rtype: duckdb.DuckDBPyConnection
    """
    with open(binds[0], "w"):
        pass
    return connection.execute("SELECT 1 AS one")


def dies(connection: Any, binds: Sequence[Any]) -> Any:
    """kill the worker running it.

    :param connection: the query's connection (not read)
    :ptype connection: duckdb.DuckDBPyConnection
    :param binds: none read
    :ptype binds: Sequence[Any]
    :return: never
    :rtype: Any
    """
    os._exit(9)
