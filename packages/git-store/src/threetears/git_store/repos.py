"""many repositories at once: one per entity group, one shared, or several, as a deployment says.

A product's data may live in one repository holding every entity, in one repository per entity
(a character, a story), or in several chosen ones. :class:`GitRepos` keeps the repositories a
process has open -- opening one on first use from a ``config_for`` the product supplies -- with one
broadcast listener among them. :class:`RoutedGitStore` is one durable store over all of them: a
``route`` says which repository a row lives in, from its table and key.

Each repository keeps its own writer, lease and commits: a change set belongs to one repository.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from threetears.git_store.advanced import AdvancedListener
from threetears.git_store.change import ChangeSet
from threetears.git_store.git import CommitEntry
from threetears.git_store.layout import Layout
from threetears.git_store.repo import GitRepo, GitRepoConfig, RowsChanged
from threetears.git_store.store import GitDurableStore

if TYPE_CHECKING:
    from threetears.nats import NatsClient

__all__ = ["GitRepos", "RepoRoute", "RoutedGitStore"]

#: which repository a row lives in, from its table and its key (or a scan's filters); ``None``
#: means "every open repository" (a scan with no filter on the routing column).
RepoRoute = Callable[[str, Mapping[str, Any]], str | None]


class GitRepos:
    """the repositories a process has open, opened on first use.

    :ivar layout: the layout every repository shares
    """

    def __init__(
        self,
        layout: Layout,
        *,
        nats: NatsClient | None,
        config_for: Callable[[str], GitRepoConfig] | None = None,
    ) -> None:
        """describe the set; nothing opens until a repository is asked for.

        :param layout: where each table's rows live, in every repository
        :ptype layout: Layout
        :param nats: the connected client every repository's writer uses; ``None`` for one process
        :ptype nats: NatsClient | None
        :param config_for: the settings of the repository with a given key, for repositories opened
            on first use (one per character: ``root / key``); ``None`` opens only those :meth:`add` names
        :ptype config_for: Callable[[str], GitRepoConfig] | None
        :return: None
        :rtype: None
        """
        self.layout = layout
        self._nats = nats
        self._config_for = config_for
        self._listener = None if nats is None else AdvancedListener(nats)
        self._repos: dict[str, GitRepo] = {}
        self._callbacks: list[RowsChanged] = []
        self._opening = asyncio.Lock()

    async def add(self, config: GitRepoConfig) -> GitRepo:
        """open a repository (or return it, when already open).

        :param config: its settings
        :ptype config: GitRepoConfig
        :return: the open repository
        :rtype: GitRepo
        """
        async with self._opening:
            existing = self._repos.get(config.key)
            if existing is not None:
                return existing
            repo = GitRepo(config, self.layout, nats=self._nats, listener=self._listener)
            for callback in self._callbacks:
                repo.on_rows_changed(callback)
            await repo.open()
            self._repos[config.key] = repo
            return repo

    async def repo(self, key: str) -> GitRepo:
        """the repository with ``key``, opened from ``config_for`` on first use.

        :param key: its key
        :ptype key: str
        :return: the open repository
        :rtype: GitRepo
        :raises KeyError: when it is not open and there is no ``config_for``
        """
        existing = self._repos.get(key)
        if existing is not None:
            return existing
        if self._config_for is None:
            raise KeyError(f"git-store repository {key!r} is not open and nothing says how to open it")
        return await self.add(self._config_for(key))

    def get(self, key: str) -> GitRepo | None:
        """the repository with ``key`` if it is open.

        :param key: its key
        :ptype key: str
        :return: the repository, or ``None``
        :rtype: GitRepo | None
        """
        return self._repos.get(key)

    def __iter__(self) -> Iterator[GitRepo]:
        """every open repository.

        :return: an iterator over them
        :rtype: Iterator[GitRepo]
        """
        return iter(list(self._repos.values()))

    def on_rows_changed(self, callback: RowsChanged) -> None:
        """be told of the rows every commit of every repository changes, those opened later included.

        :param callback: called with each commit's rows
        :ptype callback: RowsChanged
        :return: None
        :rtype: None
        """
        self._callbacks.append(callback)
        for repo in self._repos.values():
            repo.on_rows_changed(callback)

    async def aclose(self) -> None:
        """close every repository, giving up each writer lease.

        :return: None
        :rtype: None
        """
        for repo in list(self._repos.values()):
            await repo.aclose()
        self._repos.clear()


class RoutedGitStore:
    """one durable store over many repositories; each row goes to the repository ``route`` names.

    Satisfies the same store protocols as :class:`~threetears.git_store.store.GitDurableStore`. A
    change set handed as ``conn`` must belong to the row's repository. A scan whose filters do not
    name one repository reads every open one.

    :ivar repos: the repositories
    """

    def __init__(self, repos: GitRepos, route: RepoRoute) -> None:
        """route rows to repositories.

        :param repos: the repositories
        :ptype repos: GitRepos
        :param route: which repository a row lives in
        :ptype route: RepoRoute
        :return: None
        :rtype: None
        """
        self.repos = repos
        self._route = route
        self._stores: dict[str, GitDurableStore] = {}

    async def store(self, key: str) -> GitDurableStore:
        """the store of one repository, opening it on first use.

        :param key: the repository's key
        :ptype key: str
        :return: its store
        :rtype: GitDurableStore
        """
        repo = await self.repos.repo(key)
        store = self._stores.get(key)
        if store is None or store.repo is not repo:
            store = GitDurableStore(repo)
            self._stores[key] = store
        return store

    async def _store_for(self, table: str, values: Mapping[str, Any], conn: Any) -> GitDurableStore:
        """the store a row's repository has, checking a change set belongs to it.

        :param table: the table
        :ptype table: str
        :param values: the row's key or the whole row
        :ptype values: Mapping[str, Any]
        :param conn: the change set passed, or ``None``
        :ptype conn: Any
        :return: the store
        :rtype: GitDurableStore
        :raises ValueError: when the route names no repository, or the change set is another's
        """
        key = self._route(table, values)
        if key is None:
            raise ValueError(f"the route names no repository for a {table} row {dict(values)}")
        if isinstance(conn, ChangeSet) and conn.repo.config.key != key:
            raise ValueError(f"a {table} row lives in repository {key!r}; the change set is {conn.repo.config.key!r}'s")
        return await self.store(key)

    async def fetch_one(self, table: str, pk: Mapping[str, Any], *, conn: Any = None) -> dict[str, Any] | None:
        """the row with key ``pk``, from its repository.

        :param table: the table
        :ptype table: str
        :param pk: the key
        :ptype pk: Mapping[str, Any]
        :param conn: a change set of the row's repository, or ``None``
        :ptype conn: Any
        :return: the row, or ``None``
        :rtype: dict[str, Any] | None
        """
        store = await self._store_for(table, pk, conn)
        return await store.fetch_one(table, pk, conn=conn)

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
        """write ``row`` to its repository (see :meth:`GitDurableStore.upsert`).

        :param table: the table
        :ptype table: str
        :param row: the whole row
        :ptype row: Mapping[str, Any]
        :param pk: the key columns
        :ptype pk: Sequence[str] | None
        :param on_conflict: ``"update"``, ``"ignore"`` or ``"raise"``
        :ptype on_conflict: str
        :param cas: the version the caller read
        :ptype cas: datetime | None
        :param conn: a change set of the row's repository, or ``None``
        :ptype conn: Any
        :return: 1 written, 0 on a miss
        :rtype: int
        """
        store = await self._store_for(table, row, conn)
        return await store.upsert(table, row, pk=pk, on_conflict=on_conflict, cas=cas, conn=conn)

    async def delete(self, table: str, pk: Mapping[str, Any], *, conn: Any = None) -> None:
        """remove a row from its repository.

        :param table: the table
        :ptype table: str
        :param pk: the key
        :ptype pk: Mapping[str, Any]
        :param conn: a change set of the row's repository, or ``None``
        :ptype conn: Any
        :return: None
        :rtype: None
        """
        store = await self._store_for(table, pk, conn)
        await store.delete(table, pk, conn=conn)

    async def scan(self, table: str, filters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """every matching row: from the one repository the filters route to, else from every open one.

        :param table: the table
        :ptype table: str
        :param filters: equality filters
        :ptype filters: Mapping[str, Any] | None
        :return: the rows
        :rtype: list[dict[str, Any]]
        """
        key = self._route(table, dict(filters or {}))
        if key is not None:
            return await (await self.store(key)).scan(table, filters)
        rows: list[dict[str, Any]] = []
        for repo in self.repos:
            rows.extend(await (await self.store(repo.config.key)).scan(table, filters))
        return rows

    async def upsert_many(
        self,
        table: str,
        rows: Sequence[Mapping[str, Any]],
        *,
        max_rows: int,
        max_bytes: int,
        conn: Any = None,
    ) -> int:
        """write many rows, one commit per repository they route to (or into ``conn``).

        :param table: the table
        :ptype table: str
        :param rows: the whole rows
        :ptype rows: Sequence[Mapping[str, Any]]
        :param max_rows: passed on
        :ptype max_rows: int
        :param max_bytes: passed on
        :ptype max_bytes: int
        :param conn: a change set of the rows' one repository, or ``None``
        :ptype conn: Any
        :return: how many rows were written or joined
        :rtype: int
        """
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            key = self._route(table, row)
            if key is None:
                raise ValueError(f"the route names no repository for a {table} row")
            grouped.setdefault(key, []).append(row)
        written = 0
        for group in grouped.values():
            store = await self._store_for(table, group[0], conn)
            written += await store.upsert_many(table, group, max_rows=max_rows, max_bytes=max_bytes, conn=conn)
        return written

    async def delete_many(
        self,
        table: str,
        keys: Sequence[Sequence[Any]],
        *,
        max_rows: int,
        conn: Any = None,
    ) -> int:
        """remove many rows, one commit per repository they route to (or into ``conn``).

        :param table: the table
        :ptype table: str
        :param keys: each row's key values, in key order
        :ptype keys: Sequence[Sequence[Any]]
        :param max_rows: passed on
        :ptype max_rows: int
        :param conn: a change set of the rows' one repository, or ``None``
        :ptype conn: Any
        :return: how many keys were named
        :rtype: int
        """
        columns = self.repos.layout.pk_columns(table)
        grouped: dict[str, list[Sequence[Any]]] = {}
        for values in keys:
            key = self._route(table, dict(zip(columns, values, strict=True)))
            if key is None:
                raise ValueError(f"the route names no repository for a {table} key {tuple(values)}")
            grouped.setdefault(key, []).append(values)
        named = 0
        for group in grouped.values():
            store = await self._store_for(table, dict(zip(columns, group[0], strict=True)), conn)
            named += await store.delete_many(table, group, max_rows=max_rows, conn=conn)
        return named

    async def history(self, table: str, pk: Mapping[str, Any], *, limit: int = 50) -> list[CommitEntry]:
        """the commits that changed a row's file, newest first.

        :param table: the table
        :ptype table: str
        :param pk: the row's key
        :ptype pk: Mapping[str, Any]
        :param limit: the most to return
        :ptype limit: int
        :return: the commits
        :rtype: list[CommitEntry]
        """
        store = await self._store_for(table, pk, None)
        return await store.history(table, pk, limit=limit)
