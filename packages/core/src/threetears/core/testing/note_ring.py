"""the change-note ring a live test seeds, installed as the note keeper installs it.

A test that reads or writes change notes wants the ring the platform runs: its tables, the ring row, and indexes
the planner can use. :func:`install_note_ring` goes through :meth:`ChangeNoteKeeper.install
<threetears.core.collections.change_notes.ChangeNoteKeeper.install>`, which builds again any ring index a failed
build left invalid (on a shared YugabyteDB, another session's DDL was seen to leave every one so, and every read
of the notes then scanned its table), rather than running the ring's DDL statements itself.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from threetears.core.collections.change_notes import ChangeNoteKeeper

__all__ = ["install_note_ring"]


class _OneConnection:
    """one connection as the keeper's pool: ``fetch`` and ``execute`` on it, ``acquire`` handing it out."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def fetch(self, query: str, *args: Any) -> Any:
        return await self._conn.fetch(query, *args)

    async def execute(self, query: str, *args: Any) -> Any:
        return await self._conn.execute(query, *args)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Any]:
        yield self._conn


class _NoChangeLog:
    """the keeper's change log, which an install never reads."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"installing the ring reads no change log ({name})")


async def install_note_ring(conn: Any, schema: str) -> None:
    """install the change-note ring in ``schema`` as the keeper does, every index of it usable.

    :param conn: an asyncpg connection allowed DDL, outside a transaction, or a pool
    :ptype conn: Any
    :param schema: the schema to hold the ring, already created
    :ptype schema: str
    :return: nothing
    :rtype: None
    """
    pool = conn if hasattr(conn, "acquire") else _OneConnection(conn)
    await ChangeNoteKeeper(pool, notes_schema=schema, log=_NoChangeLog()).install()
