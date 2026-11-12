"""whether a login may read or write a table: one row asked for, never by key, by design.

A grant check (a warehouse reader's ``SELECT``, a writer's ``INSERT``) proves the permission by using it on the table
the profile names: it reads no data anyone uses and has no key to read by. :func:`access_probe_sql` writes the one
statement each check sends, and :func:`probe_access` sends it.

- **read**: ``SELECT 1 FROM <table> LIMIT 1``. Never ``LIMIT 0``: Redshift answers ``LIMIT 0`` without checking the
  grant (found live 2026-10-08: an ungranted table returned no rows at ``LIMIT 0`` and SQLSTATE 42501 at ``LIMIT 1``).
  An empty table still answers.
- **write**: ``INSERT INTO <table> SELECT * FROM <table> LIMIT 1``, rolled back: the insert is checked against the
  grant and nothing is kept.
"""

from __future__ import annotations

from typing import Any

__all__ = ["access_probe_sql", "probe_access"]


def access_probe_sql(table: str, *, write: bool = False) -> str:
    """the statement that proves a login may read (or write) ``table``.

    :param table: the table, ``schema.table`` as the login names it, a TRUSTED identifier (from a checked profile)
    :ptype table: str
    :param write: the write probe (an ``INSERT ... SELECT`` of one row, for the caller to roll back) instead of the read
    :ptype write: bool
    :return: the statement
    :rtype: str
    """
    if write:
        return f"INSERT INTO {table} SELECT * FROM {table} LIMIT 1"  # noqa: S608 - trusted identifier
    return f"SELECT 1 FROM {table} LIMIT 1"  # noqa: S608 - trusted identifier


async def probe_access(session: Any, table: str, *, write: bool = False) -> None:
    """use the login's read (or write) permission on ``table`` once; the driver's error is raised when it is refused.

    The write probe is rolled back: through the session's ``execute_rolled_back`` when it has one (a warehouse
    driver's), else inside a transaction the probe rolls back (asyncpg's ``transaction()``).

    :param session: the login's session: ``fetch`` for the read; ``execute_rolled_back`` or ``execute`` and
        ``transaction()`` for the write
    :ptype session: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param write: probe the write permission instead of the read
    :ptype write: bool
    :return: nothing
    :rtype: None
    """
    sql = access_probe_sql(table, write=write)
    if not write:
        await session.fetch(sql)
        return
    rolled_back = getattr(session, "execute_rolled_back", None)
    if rolled_back is not None:
        await rolled_back(sql)
        return
    transaction = session.transaction()
    await transaction.start()
    try:
        await session.execute(sql)
    finally:
        await transaction.rollback()
