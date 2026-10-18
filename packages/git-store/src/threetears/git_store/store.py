"""``GitDurableStore``: 3tears' structured durable-store contract over a git working tree.

It satisfies :class:`~threetears.core.backends.DurableStore` (``fetch_one`` / ``upsert`` / ``delete``
/ ``scan``), :class:`~threetears.core.backends.protocol.BulkDurableStore` and
:class:`~threetears.core.backends.protocol.BulkDeletingStore`, so a
:class:`~threetears.core.collections.DurableStoreCollection` runs on it with L1 and L2 in front,
unchanged. A table is a set of files the repository's layout places; a row is one file.

- **Reads** come from the working tree, off the event loop. A read handed a change set
  (``conn=cs``) sees the set's own writes first.
- **Writes are never loose.** With ``conn`` a :class:`~threetears.git_store.change.ChangeSet`, the
  write joins it and lands in its one commit. With no ``conn``, the write is a commit of its own.
- **Compare-and-swap works**: a write carrying ``cas`` (the ``date_updated`` the writer read) lands
  only while the file still holds that version, and a row deleted since is a miss. A miss returns
  0, the protocol's "nothing written", and the collection raises
  :class:`~threetears.core.exceptions.ConcurrentModificationError`. A ``cas`` on a table with no
  version column raises; it is never silently ignored. A write that joins a change set is checked
  against the files when it joins (an early answer) and again by the writer when the set commits
  (the one that counts).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from threetears.git_store.change import Change, ChangeSet, OnConflict, same_version
from threetears.git_store.errors import ChangeConflictError, ConflictedRow
from threetears.git_store.git import CommitEntry
from threetears.git_store.layout import locate
from threetears.git_store.repo import GitRepo

__all__ = ["GitDurableStore"]


def _key_text(pk: Mapping[str, Any]) -> str:
    """a key as a commit message names it.

    :param pk: the key
    :ptype pk: Mapping[str, Any]
    :return: its values, joined with ``/``
    :rtype: str
    """
    return "/".join(str(value) for value in pk.values())


class GitDurableStore:
    """the durable tier of one git repository's tables.

    :ivar repo: the repository
    """

    def __init__(self, repo: GitRepo) -> None:
        """serve the tables of ``repo``'s layout.

        :param repo: the repository; open it before the store is used
        :ptype repo: GitRepo
        :return: None
        :rtype: None
        """
        self.repo = repo

    def _change_set(self, conn: Any) -> ChangeSet | None:
        """the change set a call was handed, refusing any other connection.

        :param conn: what the caller passed as ``conn``
        :ptype conn: Any
        :return: the change set, or ``None``
        :rtype: ChangeSet | None
        :raises ValueError: for a connection that is not a change set of this repository
        """
        if conn is None:
            return None
        if not isinstance(conn, ChangeSet) or conn.repo is not self.repo:
            raise ValueError(
                f"a git-store write joins a ChangeSet of repository {self.repo.config.key!r} "
                f"(GitRepo.change()); it was handed {type(conn).__name__}"
            )
        return conn

    # ------------------------------------------------------------------ reads

    async def fetch_one(self, table: str, pk: Mapping[str, Any], *, conn: Any = None) -> dict[str, Any] | None:
        """the row with key ``pk``, or ``None``.

        :param table: the table
        :ptype table: str
        :param pk: the key
        :ptype pk: Mapping[str, Any]
        :param conn: a change set whose own writes the read sees first, or ``None``
        :ptype conn: Any
        :return: the row
        :rtype: dict[str, Any] | None
        """
        change_set = self._change_set(conn)
        if change_set is not None:
            pending = change_set.pending(table, pk)
            if pending is not None:
                return None if pending.row is None else dict(pending.row)
        return await self.repo.read_row(table, pk)

    async def scan(self, table: str, filters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """every row matching the equality ``filters``; a filter on a path column narrows the files read.

        :param table: the table
        :ptype table: str
        :param filters: equality filters; ``None`` for every row
        :ptype filters: Mapping[str, Any] | None
        :return: the rows
        :rtype: list[dict[str, Any]]
        """
        return await self.repo.read_rows(table, filters)

    async def history(self, table: str, pk: Mapping[str, Any], *, limit: int = 50) -> list[CommitEntry]:
        """the commits that changed a row's file, newest first.

        :param table: the table
        :ptype table: str
        :param pk: the row's key
        :ptype pk: Mapping[str, Any]
        :param limit: the most to return
        :ptype limit: int
        :return: the commits; empty when the row has no file the layout can name
        :rtype: list[CommitEntry]
        """
        path = await asyncio.to_thread(locate, self.repo.layout, self.repo.config.work_tree, table, pk)
        if path is None:
            return []
        return await self.repo.history(path.as_posix(), limit=limit)

    async def read_at(self, rev: str, table: str, pk: Mapping[str, Any]) -> dict[str, Any] | None:
        """a row as a commit held it.

        :param rev: the commit
        :ptype rev: str
        :param table: the table
        :ptype table: str
        :param pk: the row's key
        :ptype pk: Mapping[str, Any]
        :return: the row, or ``None`` when its file did not exist there
        :rtype: dict[str, Any] | None
        """
        path = await asyncio.to_thread(locate, self.repo.layout, self.repo.config.work_tree, table, pk)
        if path is None:
            return None
        data = await self.repo.read_at(rev, path.as_posix())
        return None if data is None else self.repo.layout.decode(table, path, data)

    # ------------------------------------------------------------------ writes

    def _precheck(self, change: Change) -> ConflictedRow | None:
        """the early answer to a write joining a change set: would it apply to the files as they are? BLOCKING.

        :param change: the write
        :ptype change: Change
        :return: the conflict, or ``None`` when it would apply; an ``"exists"`` conflict under
            ``on_conflict="ignore"`` means "skip it"
        :rtype: ConflictedRow | None
        """
        stored = self.repo.read_local(change.table, change.pk)
        if change.cas is not None:
            column = self.repo.layout.version_column(change.table)
            if stored is None:
                return ConflictedRow(change.table, change.pk, "deleted", change.cas)
            if column is not None and not same_version(stored.get(column), change.cas):
                return ConflictedRow(change.table, change.pk, "changed", change.cas)
        if stored is not None and change.on_conflict in ("ignore", "raise"):
            return ConflictedRow(change.table, change.pk, "exists")
        return None

    def _refuse_unfenced(self, table: str, cas: datetime | None) -> None:
        """refuse a compare-and-swap on a table that keeps no version.

        :param table: the table
        :ptype table: str
        :param cas: the version the caller read
        :ptype cas: datetime | None
        :return: None
        :rtype: None
        :raises ValueError: when ``cas`` is given and the table has no version column
        """
        if cas is not None and self.repo.layout.version_column(table) is None:
            raise ValueError(
                f"{table} has no version column, so a compare-and-swap write cannot be honoured; "
                f"it is refused rather than written unfenced"
            )

    async def upsert(
        self,
        table: str,
        row: Mapping[str, Any],
        *,
        pk: Sequence[str] | None = None,
        on_conflict: str = "update",
        cas: datetime | None = None,
        conn: Any = None,
    ) -> int:
        """write ``row``: in the change set ``conn``, or as a commit of its own.

        :param table: the table
        :ptype table: str
        :param row: the whole row, key columns included
        :ptype row: Mapping[str, Any]
        :param pk: the key columns; the layout's when ``None``
        :ptype pk: Sequence[str] | None
        :param on_conflict: ``"update"``, ``"ignore"`` (0 when the row exists) or ``"raise"``
        :ptype on_conflict: str
        :param cas: the version the caller read; the write lands only while the file holds it
        :ptype cas: datetime | None
        :param conn: a :class:`ChangeSet` of this repository, or ``None``
        :ptype conn: Any
        :return: 1 written (or joined); 0 on a compare-and-swap miss or an ignored existing row
        :rtype: int
        :raises ChangeConflictError: under ``on_conflict="raise"`` when the row exists
        :raises ValueError: for ``cas`` on a table with no version column, an unknown
            ``on_conflict``, or a ``conn`` that is not this repository's change set
        """
        if on_conflict not in ("update", "ignore", "raise"):
            raise ValueError(f"on_conflict must be 'update', 'ignore' or 'raise', not {on_conflict!r}")
        policy: OnConflict = "update" if on_conflict == "update" else "ignore" if on_conflict == "ignore" else "raise"
        self._refuse_unfenced(table, cas)
        columns = tuple(pk) if pk is not None else self.repo.layout.pk_columns(table)
        key = {column: row[column] for column in columns}
        change = Change(table, key, dict(row), cas, policy)
        change_set = self._change_set(conn)
        if change_set is not None and change_set.pending(table, key) is not None:
            change_set.add(change)  # a second write of a row the set already checked
            return 1
        conflict = await asyncio.to_thread(self._precheck, change)
        if conflict is not None:
            if conflict.reason == "exists" and policy == "raise":
                raise ChangeConflictError([conflict])
            return 0
        if change_set is not None:
            change_set.add(change)
            return 1
        try:
            await self.repo.commit([change], f"{table} {_key_text(key)}: save")
        except ChangeConflictError as exc:
            if any(item.reason == "exists" for item in exc.rows):
                raise
            return 0
        return 1

    async def delete(self, table: str, pk: Mapping[str, Any], *, conn: Any = None) -> None:
        """remove a row's file: in the change set ``conn``, or as a commit of its own. A missing row is no error.

        :param table: the table
        :ptype table: str
        :param pk: the key
        :ptype pk: Mapping[str, Any]
        :param conn: a :class:`ChangeSet` of this repository, or ``None``
        :ptype conn: Any
        :return: None
        :rtype: None
        """
        change = Change(table, dict(pk), None)
        change_set = self._change_set(conn)
        if change_set is not None:
            change_set.add(change)
            return
        await self.repo.commit([change], f"{table} {_key_text(pk)}: delete")

    async def upsert_many(
        self,
        table: str,
        rows: Sequence[Mapping[str, Any]],
        *,
        max_rows: int,
        max_bytes: int,
        conn: Any = None,
    ) -> int:
        """write many rows of one table: in the change set ``conn``, or as ONE commit of their own.

        The batch bounds are for a transport that splits a write; a commit has no such limit, so
        they are accepted and not needed.

        :param table: the table
        :ptype table: str
        :param rows: the whole rows
        :ptype rows: Sequence[Mapping[str, Any]]
        :param max_rows: unused
        :ptype max_rows: int
        :param max_bytes: unused
        :ptype max_bytes: int
        :param conn: a :class:`ChangeSet` of this repository, or ``None``
        :ptype conn: Any
        :return: how many rows were written or joined
        :rtype: int
        """
        del max_rows, max_bytes
        columns = self.repo.layout.pk_columns(table)
        changes = [Change(table, {column: row[column] for column in columns}, dict(row)) for row in rows]
        change_set = self._change_set(conn)
        if change_set is not None:
            for change in changes:
                change_set.add(change)
        elif changes:
            await self.repo.commit(changes, f"{table}: save {len(changes)} rows")
        return len(changes)

    async def delete_many(
        self,
        table: str,
        keys: Sequence[Sequence[Any]],
        *,
        max_rows: int,
        conn: Any = None,
    ) -> int:
        """remove many rows of one table: in the change set ``conn``, or as ONE commit of their own.

        :param table: the table
        :ptype table: str
        :param keys: each row's key values, in the layout's key order
        :ptype keys: Sequence[Sequence[Any]]
        :param max_rows: unused
        :ptype max_rows: int
        :param conn: a :class:`ChangeSet` of this repository, or ``None``
        :ptype conn: Any
        :return: how many keys were named
        :rtype: int
        """
        del max_rows
        columns = self.repo.layout.pk_columns(table)
        changes = [Change(table, dict(zip(columns, values, strict=True)), None) for values in keys]
        change_set = self._change_set(conn)
        if change_set is not None:
            for change in changes:
                change_set.add(change)
        elif changes:
            await self.repo.commit(changes, f"{table}: delete {len(changes)} rows")
        return len(changes)
