"""a test's own schema, dropped at teardown even when YugabyteDB refuses the drop for a concurrent catalog write.

On YugabyteDB a ``DROP SCHEMA ... CASCADE`` takes the shared catalog: another session's DDL committed meanwhile (a
second test run on the same database, a migration) fails it with ``SerializationError`` ("conflict with concurrently
committed data", kConflict), which YugabyteDB asks the client to retry. A teardown that does not retry fails a test
that passed and leaves its schema behind.
"""

from __future__ import annotations

import asyncio
from typing import Any

import asyncpg

__all__ = ["drop_schema"]

#: the drops tried before the conflict is raised, and the wait before the first retry (doubled each time)
_ATTEMPTS = 6
_FIRST_WAIT = 0.25


async def drop_schema(conn: Any, schema: str, *, attempts: int = _ATTEMPTS, first_wait: float = _FIRST_WAIT) -> None:
    """drop ``schema`` and everything in it, retrying while YugabyteDB answers a concurrent catalog write with
    ``SerializationError``; any other error, or the conflict on the last attempt, is raised.

    :param conn: an asyncpg connection (or anything with its ``execute``), outside a transaction
    :ptype conn: Any
    :param schema: the schema, a TRUSTED identifier the test made
    :ptype schema: str
    :param attempts: the drops tried
    :ptype attempts: int
    :param first_wait: seconds before the first retry, doubled for each after
    :ptype first_wait: float
    :return: nothing
    :rtype: None
    :raises asyncpg.SerializationError: when every attempt conflicts
    """
    wait = first_wait
    for attempt in range(1, attempts + 1):
        try:
            await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            return
        except asyncpg.SerializationError:
            if attempt == attempts:
                raise
            await asyncio.sleep(wait)
            wait *= 2
