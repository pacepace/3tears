"""
answers a fake migration session gives the database-wide DDL lock.

every locked :class:`~threetears.core.data.migrations.MigrationRunner` entry
point takes :func:`~threetears.core.data.migrations.database_ddl_lock` first,
which reads ``current_database()`` and then polls ``pg_try_advisory_lock`` and
releases with ``pg_advisory_unlock``. an in-memory session fake that answers
unknown queries with no rows makes the lock fail loudly (a real PostgreSQL
session always answers them), so a fake that stands in for one session has to
answer these three. :func:`uncontended_ddl_lock_rows` gives the answers of a
session alone in its database, for fakes that are not testing the lock itself.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "FAKE_DATABASE_NAME",
    "uncontended_ddl_lock_rows",
]

#: the database a fake session reports itself connected to
FAKE_DATABASE_NAME = "fake_database"


def uncontended_ddl_lock_rows(sql: str) -> list[dict[str, Any]] | None:
    """
    answer one of the DDL lock's statements as a session alone in its database would.

    :param sql: the statement a fake session's ``query`` received
    :ptype sql: str
    :return: the rows PostgreSQL would return -- the database name, a granted
        lock, a released lock -- or None when ``sql`` is not one of the lock's
        statements and the fake should answer it itself
    :rtype: list[dict[str, Any]] | None
    """
    normalized = " ".join(sql.split()).lower()
    rows: list[dict[str, Any]] | None = None
    if "current_database()" in normalized:
        rows = [{"database_name": FAKE_DATABASE_NAME}]
    elif "pg_try_advisory_lock(" in normalized:
        rows = [{"acquired": True}]
    elif "pg_advisory_unlock(" in normalized:
        rows = [{"released": True}]
    return rows
