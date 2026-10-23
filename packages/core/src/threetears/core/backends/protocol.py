"""L3 durable-tier backend protocols + the shared rowcount parser.

The L3 (durable) tier sits behind two abstraction levels, deliberately separated
so a non-SQL backend (e.g. a git working tree) can be a first-class L3:

- :class:`L3Backend` — the **low-level raw transport**: ``fetch`` / ``fetchrow`` /
  ``execute`` / ``execute_batch`` / ``acquire`` / ``transaction``, taking **raw SQL
  strings**. This is the ad-hoc-SQL escape hatch the ``l3_pool`` ivar advertises
  (keyset pagination, JOINs, bulk queries). It is *irreducibly SQL*; a git backend
  legitimately **does not implement it**. :class:`~...backends.nats_proxy.NatsProxyL3Backend`
  and ``SqlL3Backend`` conform.
- :class:`DurableStore` — the **high-level structured ops**: ``fetch_one`` / ``upsert``
  / ``delete`` / ``scan``, keyed by table + column dict + pk, **no SQL string**. This
  is the level the standard CRUD lifecycle uses. A SQL backend implements it by
  *generating* SQL; a ``GitL3Backend`` (scriob) implements it as file read / write /
  delete + commit. **This is the seam that makes a non-SQL durable backend possible.**

The two are independent: a backend MAY implement only :class:`DurableStore` (a git
backend), only :class:`L3Backend` (a raw-SQL pool with no structured layer), or both
(``SqlL3Backend``). The collection framework's standard CRUD path uses the structured
layer; code that needs raw SQL drops to :class:`L3Backend` explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final, Protocol, runtime_checkable
from uuid import UUID

__all__ = [
    "L3_RAIL_ROW_CAP",
    "LOAD_FILES_MAX_WAIT_SECONDS",
    "BulkDeletingStore",
    "BulkDurableStore",
    "CopyingStore",
    "DurableStore",
    "FileLoad",
    "KeyLedReadingStore",
    "L3Backend",
    "L3Reader",
    "LoadFile",
    "LoadResult",
    "LoadingStore",
    "OrderedDurableStore",
    "RowcountUnreadableError",
    "TruncatingStore",
    "parse_rowcount",
]


class L3Reader(Protocol):
    """the read half of :class:`L3Backend` a whole-table copy needs: parameterized SELECTs.

    Every :class:`L3Backend` is one. The copies (:mod:`threetears.core.collections.complete_copy`,
    the scoped snapshot) take this narrower seam so a test double need answer only the reads they
    make, while mypy still refuses an object that cannot run SQL at all (a :class:`DurableStore`).
    """

    async def fetch(self, query: str, /, *params: Any) -> list[dict[str, Any]]:
        """Run a SELECT and return all rows as dicts (empty list on no rows)."""
        ...

    async def fetchrow(self, query: str, /, *params: Any) -> Any:
        """Run a SELECT and return the first row dict, or ``None``."""
        ...


#: the most rows the L3 rail (the hub's broker, behind ``NatsProxyL3Backend``) answers one
#: statement. It cuts there and does not say so, so every reader that might pass it pages or reads
#: in halves against this number: the one owner of it. A transport states its own cap as
#: ``rows_per_statement``, ``None`` for one that never cuts. One that states nothing -- a direct
#: asyncpg pool, which cuts nothing but does not say so -- is taken to be the rail, which costs a
#: needless split at worst and never a row.
L3_RAIL_ROW_CAP: Final = 1000

#: the longest a pod waits for one ``l3.load_files`` reply, in seconds, however many rows its files
#: hold. NATS request/reply is at most once: a broker that restarts mid-load loses the reply, and a
#: wait that grew with the load (each transaction's bound times the transactions) held the caller
#: for a day at the defaults. A load longer than this is sent as several requests, one per
#: partition of the export. A load's URLs are presigned to outlive it
#: (:data:`~threetears.datasources.export_read.DEFAULT_PRESIGN_SECONDS` is longer), so the broker
#: never meets an expired file inside a wait the pod still keeps.
LOAD_FILES_MAX_WAIT_SECONDS: Final = 3600


@runtime_checkable
class L3Backend(Protocol):
    """Raw-SQL transport for the L3 durable tier (mirrors the asyncpg pool surface).

    Methods take **parameterized SQL strings**. This is the ad-hoc escape hatch — a
    backend that cannot run SQL (a git working tree) does not implement it and instead
    satisfies :class:`DurableStore`. ``NatsProxyL3Backend`` and ``SqlL3Backend`` conform;
    conformance is asserted via :func:`isinstance` against this ``runtime_checkable``
    protocol. Errors surface as the data layer's typed unavailable error, never silently.

    The read methods accept an optional ``customer_scope``: the broker-isolation clamp
    for customer-scoped (Class-B) reads on the ``system.platform.rbac`` carve-out
    (broker-isolation-task-01). ``NatsProxyL3Backend`` routes it to the broker; the
    ``SqlL3Backend`` wrapper forwards it to a namespace-aware transport and drops it for
    a bare asyncpg pool. ``None`` (the default) ships no scope.
    """

    async def fetch(
        self, query: str, *params: Any, namespace: str | None = None, customer_scope: UUID | None = None
    ) -> list[dict[str, Any]]:
        """Run a SELECT and return all rows as dicts (empty list on no rows)."""
        ...

    async def fetchrow(
        self, query: str, *params: Any, namespace: str | None = None, customer_scope: UUID | None = None
    ) -> dict[str, Any] | None:
        """Run a SELECT and return the first row dict, or ``None``."""
        ...

    async def fetchval(
        self, query: str, *params: Any, namespace: str | None = None, customer_scope: UUID | None = None
    ) -> Any:
        """Run a SELECT and return the first column of the first row (scalar), or ``None``."""
        ...

    async def execute(self, query: str, *params: Any, namespace: str | None = None) -> str:
        """Run an INSERT/UPDATE/DELETE; return the asyncpg-shape status tag (``"UPDATE 1"``).

        The tag is a string (not a bare int) so callers can :func:`parse_rowcount` it;
        returning an int historically crashed ``.split()`` callers in production.
        """
        ...

    async def execute_batch(
        self, queries: list[dict[str, Any]], *, namespace: str | None = None, transaction: bool = True
    ) -> list[Any]:
        """Run a batch of ``{operation, query, params}`` dicts, atomically when ``transaction``."""
        ...

    def acquire(self) -> Any:
        """Return an asyncpg-Pool-style ``acquire()`` async context manager (a pooled connection)."""
        ...

    def transaction(self, namespace: str | None = None) -> Any:
        """Return a pool-level transaction async context manager (``acquire`` + ``conn.transaction``)."""
        ...


@runtime_checkable
class DurableStore(Protocol):
    """Structured, **SQL-free** durable-tier operations keyed by table + columns + pk.

    This is the seam the collection CRUD lifecycle uses and the one a non-SQL backend
    implements: a SQL backend generates SQL from the table/columns; a ``GitL3Backend``
    reads/writes/deletes the entity's file in a working tree and stages it. No method
    takes or returns a SQL string. ``pk`` is a column→value mapping (single- or
    composite-pk); a row is a column→value mapping. Conformance is asserted via
    :func:`isinstance` against this ``runtime_checkable`` protocol.
    """

    async def fetch_one(self, table: str, pk: Mapping[str, Any], *, conn: Any = None) -> dict[str, Any] | None:
        """Fetch the row whose primary key equals ``pk`` (column→value), or ``None`` on miss.

        :param conn: optional backend-specific transaction handle the read binds to
            instead of the backend's own store; ``None`` uses the backend's default
            handle. A non-transactional backend ignores it.
        :ptype conn: Any
        """
        ...

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
        """Insert-or-update ``row`` keyed by the ``pk`` columns; return rows affected.

        :param pk: the pk column names. Optional for a backend that already knows the pk
            for ``table`` (e.g. a schema-aware SQL backend); required otherwise.
        :ptype pk: Sequence[str] | None
        :param on_conflict: ``"update"`` (upsert), ``"ignore"`` (no-op on conflict), or
            ``"raise"`` (insert only — conflict is an error).
        :ptype on_conflict: str
        :param cas: optimistic-lock fence — the pre-modification ``date_updated`` the
            update must still match; a mismatch yields **0 rows affected** (the caller
            raises ``ConcurrentModificationError``). ``None`` for inserts. A non-``None``
            value says the caller READ the row as existing, so the write must be
            update-only: a row deleted since that read (an erasure, a retire) is a
            mismatch too, and must yield 0 rows rather than be re-inserted.
        :ptype cas: datetime | None
        :param conn: optional backend-specific transaction handle the write binds to so
            it commits atomically with the caller's other operations; ``None`` uses the
            backend's default handle. A non-transactional backend ignores it.
        :ptype conn: Any
        :return: rows affected — ``1`` on success, ``0`` on a CAS/optimistic-lock miss.
        :rtype: int
        """
        ...

    async def delete(self, table: str, pk: Mapping[str, Any], *, conn: Any = None) -> None:
        """Delete the row whose primary key equals ``pk``; a missing row is not an error.

        :param conn: optional backend-specific transaction handle the delete binds to;
            ``None`` uses the backend's default handle. A non-transactional backend
            ignores it.
        :ptype conn: Any
        """
        ...

    async def scan(self, table: str, filters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Return every row matching the equality ``filters`` (all rows when ``None``/empty)."""
        ...


@runtime_checkable
class BulkDurableStore(Protocol):
    """A durable store that saves many rows of one table at once.

    The seam :meth:`~threetears.core.collections.schema_backed.SchemaBackedCollection.save_rows`
    writes through. A store without it is written a row at a time through
    :meth:`DurableStore.upsert`; a SQL store issues multi-row upserts. No method takes SQL.
    """

    async def upsert_many(
        self,
        table: str,
        rows: Sequence[Mapping[str, Any]],
        *,
        max_rows: int,
        max_bytes: int,
        conn: Any = None,
    ) -> int:
        """Insert-or-update every row of ``rows`` in ``table``; return how many were written.

        :param rows: the rows, each keyed by column, every row supplying the same columns
        :ptype rows: Sequence[Mapping[str, Any]]
        :param max_rows: the most rows one write may carry (a statement's share of a timeout)
        :ptype max_rows: int
        :param max_bytes: the most bytes one write's whole request may come to (a message's share
            of a bus)
        :ptype max_bytes: int
        :param conn: the caller's transaction handle the writes bind to; ``None`` uses the
            backend's own. A failing write fails that transaction whole
        :ptype conn: Any
        :return: rows written
        :rtype: int
        """
        ...


@runtime_checkable
class CopyingStore(Protocol):
    """A durable store that bulk-loads new rows of one table with ``COPY``, insert-only.

    The seam :meth:`~threetears.core.collections.schema_backed.SchemaBackedCollection.save_rows`
    writes through when asked to copy. A store without it is written a row at a time through
    :meth:`DurableStore.upsert` with ``on_conflict="raise"``. No method takes SQL.
    """

    async def copy_many(
        self,
        table: str,
        rows: Sequence[Mapping[str, Any]],
        *,
        max_rows: int,
        max_bytes: int,
        conn: Any,
    ) -> int:
        """Insert every row of ``rows`` into ``table``; a key already held fails the write.

        :param rows: the rows, each keyed by column, every row supplying the same columns
        :ptype rows: Sequence[Mapping[str, Any]]
        :param max_rows: the most rows one copy may carry
        :ptype max_rows: int
        :param max_bytes: the most bytes one copy's whole request may come to (a message's share of
            a bus)
        :ptype max_bytes: int
        :param conn: the caller's transaction handle the copies bind to. A failing copy fails that
            transaction whole
        :ptype conn: Any
        :return: rows written
        :rtype: int
        """
        ...


@runtime_checkable
class TruncatingStore(Protocol):
    """A durable store that empties one table at once, outside any transaction.

    The seam :meth:`~threetears.core.collections.schema_backed.SchemaBackedCollection.truncate`
    empties through. ``TRUNCATE`` leaves no row behind to tombstone, where deleting every row
    leaves one per row; it is not transactional and runs alone. No method takes SQL.
    """

    async def truncate(self, table: str) -> None:
        """Remove every row of ``table``; an empty table is not an error.

        :param table: the table
        :ptype table: str
        :return: nothing
        :rtype: None
        """
        ...


@dataclass(frozen=True)
class LoadFile:
    """One file a table is loaded from: where to fetch it, and how many rows it holds.

    The url is one the loader fetches as it is, with no credential of its own: a presigned GET of a
    warehouse export's parquet file
    (:func:`~threetears.datasources.export_read.presigned_export_files`). The row count is the
    export's own count of the file, which the loader holds the file to.

    Which URLs a loader fetches is :class:`LoadingStore`'s contract: on the L3 rail only an
    ``https`` URL of a file of the caller's own recorded export, and it must stay valid for the
    load's whole wait (:data:`LOAD_FILES_MAX_WAIT_SECONDS`).

    :ivar url: the file's URL
    :ivar rows: the rows the file holds
    """

    url: str
    rows: int

    def __post_init__(self) -> None:
        """refuse a file with no URL or a row count that is not a count.

        :return: nothing
        :rtype: None
        :raises ValueError: when ``url`` is empty or ``rows`` is not a non-negative int
        """
        if not self.url:
            raise ValueError("a LoadFile needs a url")
        if isinstance(self.rows, bool) or not isinstance(self.rows, int) or self.rows < 0:
            raise ValueError(f"a LoadFile's rows must be a non-negative int, got {self.rows!r}")


@dataclass(frozen=True)
class FileLoad:
    """How many rows one file of a load wrote.

    :ivar index: the file's position in the files the load was given
    :ivar rows_loaded: the rows written from it
    """

    index: int
    rows_loaded: int


@dataclass(frozen=True)
class LoadResult:
    """What a load from files wrote: every row, and each file's share.

    :ivar row_count: the rows written, every file's together
    :ivar files: each file's rows, in the order the files were given
    """

    row_count: int
    files: tuple[FileLoad, ...]


@runtime_checkable
class LoadingStore(Protocol):
    """A durable store that loads one table from files, outside any transaction of the caller's.

    The seam :meth:`~threetears.core.collections.schema_backed.SchemaBackedCollection.load_files`
    loads through. The store fetches each file itself and copies its rows in, committing every
    ``rows_per_transaction`` rows: on the L3 rail the broker does both (the ``l3.load_files``
    request), so the rows never cross the bus. No method takes SQL.

    **Which URLs are fetched.** The broker fetches with the hub's network position on a pod's word,
    so a URL a pod chose freely would make the hub a proxy for any request (SSRF: an internal
    service, the cloud metadata endpoint, another pod's export). The broker therefore fetches only
    a presigned ``https`` URL (no user info, no port but 443, no fragment) on an S3 host of the
    caller's datasource's export bucket, whose key lies under the prefix of an export the hub
    recorded running for that caller; it follows no redirect, and refuses the whole request,
    before fetching anything, when any URL is not one. The hub enforces it
    (``aibots.hub.broker.export_files`` in bluelabsio/14-eng-ai-bot). A direct pool's load
    (:meth:`~threetears.core.backends.sql.SqlL3Backend.load_files`) runs in a process that already
    holds the database's credentials and admits more: ``https``, ``http`` on this machine (a local
    MinIO) and ``file://`` (a :class:`~threetears.object_store.FilesystemObjectStore`'s presign).

    **How long.** A pod waits for one load at most :data:`LOAD_FILES_MAX_WAIT_SECONDS`; a caller
    loading more than that holds splits the load into several calls (one per partition), and
    presigns each call's URLs just before it, valid for longer than the wait.
    """

    async def load_files(
        self,
        table: str,
        files: Sequence[LoadFile],
        *,
        columns: Sequence[str],
        constants: Mapping[str, Any],
        rows_per_transaction: int,
        timeout_ms: int,
    ) -> LoadResult:
        """Insert every row of ``files`` into ``table``, insert-only; a key already held fails the load.

        :param table: the table
        :ptype table: str
        :param files: the parquet files, each holding the columns ``columns`` names
        :ptype files: Sequence[LoadFile]
        :param columns: the table's columns each file supplies
        :ptype columns: Sequence[str]
        :param constants: column -> the value every row takes for it, for a column no file holds
            (a part's own key, a load's stamp)
        :ptype constants: Mapping[str, Any]
        :param rows_per_transaction: the most rows one committed transaction writes; at least one
        :ptype rows_per_transaction: int
        :param timeout_ms: the most time one transaction may take, in milliseconds
        :ptype timeout_ms: int
        :return: the rows written, every file's and each file's
        :rtype: LoadResult
        """
        ...


@runtime_checkable
class BulkDeletingStore(Protocol):
    """A durable store that deletes many rows of one table at once.

    The seam :meth:`~threetears.core.collections.schema_backed.SchemaBackedCollection.delete_rows`
    deletes through. A store without it is asked a key at a time through :meth:`DurableStore.delete`;
    a SQL store issues multi-row deletes. No method takes SQL.
    """

    async def delete_many(
        self,
        table: str,
        keys: Sequence[Sequence[Any]],
        *,
        max_rows: int,
        conn: Any = None,
    ) -> int:
        """Delete every row of ``table`` whose primary key is one of ``keys``; return how many keys were named.

        :param table: the table
        :ptype table: str
        :param keys: each row's key values, in the schema's key order
        :ptype keys: Sequence[Sequence[Any]]
        :param max_rows: the most keys one write may name
        :ptype max_rows: int
        :param conn: the caller's transaction handle the deletes bind to; ``None`` uses the
            backend's own. A failing delete fails that transaction whole
        :ptype conn: Any
        :return: keys named (a key no row holds is not an error)
        :rtype: int
        """
        ...


@runtime_checkable
class KeyLedReadingStore(Protocol):
    """A durable store that reads the rows many leading-key values hold, every statement led by the key.

    The seam :meth:`~threetears.core.collections.schema_backed.SchemaBackedCollection.read_rows_led_by`
    reads through. A store without it is scanned a value at a time through :meth:`DurableStore.scan`.
    No method takes SQL.
    """

    async def fetch_led_by(
        self,
        table: str,
        values: Sequence[Any],
        *,
        columns: Sequence[str],
        max_values: int,
        conn: Any = None,
    ) -> list[dict[str, Any]]:
        """Read ``columns`` of every row of ``table`` whose leading key column is one of ``values``.

        Each statement is led by the key and names at most ``max_values`` values. The store knows
        how many rows its transport answers a statement (:data:`L3_RAIL_ROW_CAP` on the rail) and
        reads so that a cut loses no row: an answer that reaches the cap is read again in halves,
        and one value holding that many rows is paged by the rest of its key.

        :param table: the table
        :ptype table: str
        :param values: the leading key column's values, each named once
        :ptype values: Sequence[Any]
        :param columns: the columns to read, the key's among them
        :ptype columns: Sequence[str]
        :param max_values: the most values one statement names; at least one
        :ptype max_values: int
        :param conn: the caller's connection the reads run on; ``None`` uses the backend's own
        :ptype conn: Any
        :return: the rows
        :rtype: list[dict[str, Any]]
        """
        ...


@runtime_checkable
class OrderedDurableStore(Protocol):
    """A durable store that can write a row only over one holding an older compare-and-swap order.

    Separate from :class:`DurableStore` so a backend that cannot express the conditional write --
    a git working tree, today -- still satisfies the structured seam and simply is not offered
    compare-and-swap persistence: ``BaseCollection.l2_cas_mutate`` refuses such a collection
    before touching L2 rather than persisting unfenced. ``SqlL3Backend`` conforms.
    """

    async def upsert_ordered(self, table: str, row: Mapping[str, Any], *, conn: Any = None) -> int:
        """Insert ``row``, or update the stored row only when its order is strictly older.

        The order is the row's ``l2_epoch`` / ``l2_revision`` pair
        (:mod:`threetears.core.collections.l2_order`), compared as a pair; a stored row whose
        pair holds a ``NULL`` is older than every order.

        :param table: the table, whose schema the backend knows
        :ptype table: str
        :param row: the row, order columns included
        :ptype row: Mapping[str, Any]
        :param conn: optional backend-specific transaction handle the write binds to
        :ptype conn: Any
        :return: ``1`` when written, ``0`` when the stored order is newer or equal
        :rtype: int
        """
        ...


class RowcountUnreadableError(ValueError):
    """a counting command's status tag carries no row count that can be read.

    The statement RAN -- the tag is its answer -- but how many rows it touched is unknown. Not
    "no rows": a fenced update that "lost", a delete that "found nothing" are decisions a caller
    makes on 0, and this is not 0. A :class:`ValueError` so a caller already treating a bad tag as
    a value error keeps doing so.
    """


#: command tags that end in a rows-affected count (PostgreSQL ``CommandComplete``).
_COUNTED_COMMANDS: frozenset[str] = frozenset(
    {"INSERT", "UPDATE", "DELETE", "MERGE", "SELECT", "COPY", "MOVE", "FETCH"}
)


def parse_rowcount(status: Any) -> int:
    """Parse an asyncpg command-tag (``"INSERT 0 1"`` / ``"UPDATE 1"`` / ``"DELETE 2"``) to a count.

    The framework-owned parser for the :meth:`L3Backend.execute` status tag. Empty /
    falsy / non-string values (mock pools sometimes return ``None`` or ``""``) resolve
    to ``0``, and so does the tag of a command that reports no row count at all
    (``"CREATE TABLE"``, ``"BEGIN"``): it affected no rows. Consolidates the duplicated
    ``int(result.split()[-1])`` idiom across product code (the ``DurableStore`` structured
    ops return an int rowcount natively, so this is only needed at the raw-SQL
    :class:`L3Backend` border).

    A tag for a command that DOES report a count (``INSERT``, ``UPDATE``, ``DELETE``,
    ``MERGE``, ``SELECT``, ``COPY``, ``MOVE``, ``FETCH``) whose count does not parse is
    refused rather than read as ``0``: "no rows affected" is an answer callers act on
    (a conditional update that "lost", a delete that "found nothing"), and a tag this
    cannot read is not that answer.

    :param status: the asyncpg status tag (or any value).
    :ptype status: Any
    :return: the rows-affected count.
    :rtype: int
    :raises RowcountUnreadableError: a counting command's tag carries no readable count.
    """
    if not status or not isinstance(status, str):
        return 0
    parts = status.split()
    if not parts:
        return 0
    if parts[-1].isascii() and parts[-1].isdigit():
        return int(parts[-1])
    if parts[0].upper() in _COUNTED_COMMANDS:
        raise RowcountUnreadableError(f"command tag {status!r} reports a row count this cannot read")
    return 0
