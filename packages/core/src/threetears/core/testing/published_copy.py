"""a test's copy as a pod's backend keeps it: the copy's worker backend itself, or a DuckDB L1 that publishes each
applied state as shared Arrow.

:class:`SharedCopy` is the pod's own: a follower's backend over the copy's worker backend
(``collections.copy_stages.SharedCopyStages`` over ``collections.shared_copy.SharedCopyBackend``), each change applied
in its apply worker, each reload staged in its reload worker, with its query pool over the store it publishes in; a
test reads what the copy holds through :meth:`SharedCopy.rows` (the current version's parts, mapped: no DuckDB).

A copy follower has no cursor read of its copy (``design-copy-worker-backend.md``, chunk 3T-8): every read it makes is
a query worker's (``collections.copy_reads``), over the versions its backend publishes to a shared Arrow store with the
mark of the change each holds (``collections.shared_copy.MARK_TABLE``). A test drives the follower over the in-process
DuckDB it can inspect: :class:`PublishedCopy` stages and applies each change in it (its own staging and apply, called
through, so a test that patches them patches the copy) and, after every apply, publishes the applied state to a store
of its own, which its :attr:`PublishedCopy.queries` read. The publish reads the DuckDB in the test process: a fixture,
never a serving path.

Close each copy when done (:func:`close_published_copies` closes every one still open, of either kind; the collections
tests' conftest calls it after each test).
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Protocol

import pyarrow as pa
import pyarrow.ipc as ipc

from threetears.core.cache.duckdb import DuckDBBackend, StagedChange
from threetears.core.collections.copy_follower import FollowedTable
from threetears.core.collections.copy_parts import PartRule, part_rules
from threetears.core.collections.copy_queries import CopyQueries
from threetears.core.collections.copy_stages import SharedCopyStages
from threetears.core.collections.schema_backed import rail_arrow_types
from threetears.core.collections.shared_copy import MARK_TABLE, ApplyHooks, SharedCopyBackend, held_parts
from threetears.core.shared_arrow import Parts, SharedArrowStore, version_files
from threetears.core.sql_fragments import quote_identifier
from threetears.interfaces.keyed.rows import DeclaredKey
from threetears.observe import get_logger, remove_tree

__all__ = ["PublishedCopy", "SharedCopy", "close_published_copies", "copy_rows"]

log = get_logger(__name__)

#: the bytes a test's store may hold: every version a test publishes, with room for the follower's checkpoint jobs
_STORE_BYTES = 1 << 30


class _Closes(Protocol):
    """what :func:`close_published_copies` closes."""

    async def close(self) -> None:
        """close it."""
        ...


#: the copies made and not yet closed, in order: what :func:`close_published_copies` closes (a test helper's own list,
#: never reached by serving code)
_OPEN: list[_Closes] = []


class PublishedCopy:
    """a DuckDB L1 as a copy follower's backend: each change staged and applied in it, and each applied state then
    published as a version of a shared Arrow store with its mark, read by :attr:`queries`.

    :param backend: the DuckDB copy, its tables created
    :ptype backend: DuckDBBackend
    :param tables: the tables each version holds; None for every table the DuckDB holds (its stages and its mark
        table aside: every name starting ``__``)
    :ptype tables: Sequence[str] | None
    :param workers: the query workers
    :ptype workers: int
    """

    def __init__(self, backend: DuckDBBackend, tables: Sequence[str] | None = None, *, workers: int = 1) -> None:
        self.backend = backend
        self._tables = None if tables is None else tuple(tables)
        self._root = Path(tempfile.mkdtemp(prefix="published-copy-"))
        #: the store each applied state is published to
        self.store = SharedArrowStore(self._root / "shm", max_bytes=_STORE_BYTES)
        #: the follower's query pool over the store
        self.queries = CopyQueries(self.store, workers=workers)
        self._published = 0
        _OPEN.append(self)
        self.publish()

    def publish(self) -> None:
        """publish the state the DuckDB holds now, with the mark of the change it holds. Blocking (IO).

        :return: nothing
        :rtype: None
        """
        with self.backend.read_snapshot() as cursor:
            names = self._tables
            if names is None:
                found = cursor.execute(
                    "SELECT table_name FROM duckdb_tables() WHERE NOT starts_with(table_name, '__') ORDER BY 1"
                ).fetchall()
                names = tuple(str(row[0]) for row in found)
            tables: dict[str, Any] = {
                name: cursor.execute(f"SELECT * FROM {quote_identifier(name)}").fetch_arrow_table()  # noqa: S608
                for name in names
            }
            mark = self.backend.staged_mark(cursor)
        tables[MARK_TABLE] = pa.table({"mark": pa.array([mark], pa.int64())})
        self.store.publish(f"c-{self._published}", tables)
        self._published += 1

    def stage_rows(
        self,
        stage: str,
        table: str,
        rows: Any,
        key: Annotated[Sequence[str], DeclaredKey],
        *,
        unique_keys: bool = False,
    ) -> int:
        """stage rows in the DuckDB, as :meth:`~threetears.core.cache.duckdb.DuckDBBackend.stage_rows`.

        :param stage: the staging table
        :ptype stage: str
        :param table: the table the rows are for
        :ptype table: str
        :param rows: the rows
        :ptype rows: Any
        :param key: the table's key columns
        :ptype key: Annotated[Sequence[str], DeclaredKey]
        :param unique_keys: the rows hold each key once already
        :ptype unique_keys: bool
        :return: the rows staged
        :rtype: int
        """
        if unique_keys:
            return int(self.backend.stage_rows(stage, table, rows, key, unique_keys=True))
        # called as a follower calls it, so a test that patches the copy's stage_rows patches this call
        return int(self.backend.stage_rows(stage, table, rows, key))

    def stage_keys(
        self, stage: str, table: str, key: Sequence[str], keys: Any, *, rows_stage: str | None = None
    ) -> int:
        """stage keys in the DuckDB, as :meth:`~threetears.core.cache.duckdb.DuckDBBackend.stage_keys`.

        :param stage: the staging table
        :ptype stage: str
        :param table: the table the keys are of
        :ptype table: str
        :param key: the key columns, in order
        :ptype key: Sequence[str]
        :param keys: the keys
        :ptype keys: Any
        :param rows_stage: a staging table of ``table``'s rows to take these keys' rows out of; None for none
        :ptype rows_stage: str | None
        :return: the keys staged
        :rtype: int
        """
        return int(self.backend.stage_keys(stage, table, key, keys, rows_stage=rows_stage))

    def unstage_keys(self, stages: Sequence[str], table: str, key: Sequence[str], keys: Any) -> int:
        """take keys out of the DuckDB's staging tables, as :meth:`~threetears.core.cache.duckdb.DuckDBBackend.unstage_keys`.

        :param stages: the staging tables
        :ptype stages: Sequence[str]
        :param table: the table the stages are of
        :ptype table: str
        :param key: the key columns, in order
        :ptype key: Sequence[str]
        :param keys: the keys
        :ptype keys: Any
        :return: the keys given
        :rtype: int
        """
        return int(self.backend.unstage_keys(stages, table, key, keys))

    def drop_stage(self, *stages: str) -> None:
        """drop the DuckDB's staging tables, as :meth:`~threetears.core.cache.duckdb.DuckDBBackend.drop_stage`.

        :param stages: the staging tables
        :ptype stages: str
        """
        self.backend.drop_stage(*stages)

    def staged_stages(self) -> list[str]:
        """the DuckDB's staging tables now, as :meth:`~threetears.core.cache.duckdb.DuckDBBackend.staged_stages`.

        :return: their names
        :rtype: list[str]
        """
        return list(self.backend.staged_stages())

    async def apply_staged(self, changes: Sequence[StagedChange], *, mark: int | None = None) -> int:
        """apply the staged changes in the DuckDB (:meth:`~threetears.core.cache.duckdb.DuckDBBackend.apply_staged`), then publish the state it holds.

        :param changes: each table's change
        :ptype changes: Sequence[StagedChange]
        :param mark: the mark of the change applied; None leaves it as it was
        :ptype mark: int | None
        :return: the rows that went in
        :rtype: int
        """
        # the DuckDB's own apply and the publish, on a thread: the follower awaits a backend's apply on its loop
        rows = int(await asyncio.to_thread(self.backend.apply_staged, changes, mark=mark))
        await asyncio.to_thread(self.publish)
        return rows

    def schema_digest(self, table: str) -> str:
        """the DuckDB's digest of a table's columns, as :meth:`~threetears.core.cache.duckdb.DuckDBBackend.schema_digest`.

        :param table: the table
        :ptype table: str
        :return: the hex digest
        :rtype: str
        """
        return self.backend.schema_digest(table)

    async def close(self) -> None:
        """close the query pool and remove the store.

        :return: nothing
        :rtype: None
        """
        if self in _OPEN:
            _OPEN.remove(self)
        await self.queries.close()
        await asyncio.to_thread(remove_tree, self._root, log=log, what="a test's published copy store")


class SharedCopy:
    """the copy's worker backend as a follower's: :attr:`copy` (give it to the follower as its backend, and to its
    checkpoints), over a store of its own, read by :attr:`queries`. Its first version (every table empty, no mark) is
    published as it is made, as the backend publishes a first change's base (``SharedCopyStages.start``), so a test
    may make it outside the event loop.

    :param tables: the copied tables
    :ptype tables: Sequence[FollowedTable]
    :param workers: the query workers
    :ptype workers: int
    :param rules: each table's parts; None holds each as one part (under the size floor)
    :ptype rules: Mapping[str, PartRule] | None
    :param derived: tables the hooks derive, beside the followed ones: each one's columns and parts
    :ptype derived: Mapping[str, tuple[pa.Schema, PartRule]] | None
    :param hooks: the backend's hooks (``shared_copy.ApplyHooks``, ``WholeHooks``); None for none
    :ptype hooks: ApplyHooks | None
    :param preload: modules the apply worker holds from its start (the hooks')
    :ptype preload: Sequence[str]
    """

    def __init__(
        self,
        tables: Sequence[FollowedTable],
        *,
        workers: int = 1,
        rules: dict[str, PartRule] | None = None,
        derived: Mapping[str, tuple[pa.Schema, PartRule]] | None = None,
        hooks: ApplyHooks | None = None,
        preload: Sequence[str] = (),
    ) -> None:
        self._root = Path(tempfile.mkdtemp(prefix="shared-copy-"))
        #: the store the copy publishes in
        self.store = SharedArrowStore(self._root / "shm", max_bytes=_STORE_BYTES)
        schemas = {table.name: pa.schema(list(rail_arrow_types(table.columns).items())) for table in tables}
        held = dict(part_rules(tables, dict.fromkeys(schemas, 0)) if rules is None else rules)
        for name, (schema, rule) in (derived or {}).items():
            schemas[name] = schema
            held[name] = rule
        self.store.publish("c-0", {table: Parts(schema=schema, parts={}) for table, schema in schemas.items()})
        #: the follower's backend
        self.copy = SharedCopyStages(SharedCopyBackend(self.store, schemas, held, hooks=hooks, preload=preload))
        #: the follower's query pool over the store
        self.queries = CopyQueries(self.store, workers=workers)
        _OPEN.append(self)

    async def started(self) -> SharedCopy:
        """the copy's first version published (every table empty), when the store holds none (it does once made).

        :return: this copy
        :rtype: SharedCopy
        """
        await self.copy.start()
        return self

    def rows(self, table: str, columns: Sequence[str] | None = None) -> list[tuple[Any, ...]]:
        """what the current version holds of ``table``, sorted: its parts mapped and read in the test process (Arrow
        alone, no DuckDB). Blocking (IO).

        :param table: the table
        :ptype table: str
        :param columns: the columns, in order; None for every one
        :ptype columns: Sequence[str] | None
        :return: its rows
        :rtype: list[tuple[Any, ...]]
        """
        return copy_rows(self.store, table, columns)

    async def close(self) -> None:
        """close the copy's lanes and workers, the query pool, and remove the store.

        :return: nothing
        :rtype: None
        """
        if self in _OPEN:
            _OPEN.remove(self)
        await self.copy.close()
        await self.queries.close()
        await asyncio.to_thread(remove_tree, self._root, log=log, what="a test's shared copy store")


def copy_rows(store: SharedArrowStore, table: str, columns: Sequence[str] | None = None) -> list[tuple[Any, ...]]:
    """what ``store``'s current version holds of ``table``, sorted: its file or its parts mapped and read in the test
    process (Arrow alone, no DuckDB), the version leased meanwhile. Blocking (IO); a fixture's read, never a serving
    path.

    :param store: the copy's store
    :ptype store: SharedArrowStore
    :param table: the table
    :ptype table: str
    :param columns: the columns, in order; None for every one
    :ptype columns: Sequence[str] | None
    :return: its rows
    :rtype: list[tuple[Any, ...]]
    """
    with store.lease() as version:
        whole = version_files(store.root, version.name).get(f"{table}.arrow")
        paths = [whole.path] if whole is not None else list(held_parts(store.root, version.name, table).values())
        read: list[pa.Table] = []
        for path in paths:
            with pa.memory_map(str(path), "r") as source:
                read.append(ipc.open_file(source).read_all())
    if not read:
        return []
    held = pa.concat_tables(read)
    if columns is not None:
        held = held.select(list(columns))
    return sorted(zip(*(held.column(name).to_pylist() for name in held.column_names), strict=True))


async def close_published_copies() -> None:
    """close every copy made and not yet closed.

    :return: nothing
    :rtype: None
    """
    while _OPEN:
        await _OPEN[-1].close()
