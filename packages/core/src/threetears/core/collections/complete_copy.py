"""An L1 that is a complete copy of its L3 table, proven complete before anything reads it.

**Why.** An L1 is ordinarily a cache: it holds the rows someone asked for, and a miss falls
through to L3. A query that aggregates over the L1 -- a total, a ranking, a count -- reads only
what the cache happens to hold, and an aggregate over some of the rows is wrong without saying
so. A collection whose L1 answers such queries (an analytic table in a ``DuckDBBackend``) must
hold every row, and must be known to.

**Warming.** :meth:`CompleteCopy.warm` reads the whole L3 table by its key, a page at a time
(the L3 rail answers at most a thousand rows per statement and does not say when it cut, so a
read that did not page would be silently short), and replaces the L1 table with exactly those
rows in one transaction, so a row L3 no longer holds leaves the copy too.

**The proof.** The table is counted and fingerprinted in L3 (:mod:`threetears.core.fingerprint`)
over every column of every row, before the first page and again after the last. The copy is
proven only when the two readings agree (no row was added, removed or changed in any value during
the read), the rows read number the count, and the L1 then holds that many rows with the same keys
(their fingerprint over the keys as L1 stores them; the values are the ones read, written as they
were read). :meth:`CompleteCopy.require` returns the proof, or raises :class:`IncompleteCopyError`
saying why, and a query over the L1 calls it first.

**Any change to the L1 voids the proof.** Every row the collection writes into its L1 (a write
through, a pull-through, a new entity) and every row that leaves it (a write this process
settled, a peer's invalidation broadcast) reaches the copy through the collection's change
listener, and the copy is refused until it is warmed again; one that lands during a warm fails
that warm. So a reader is never given a copy with a row missing or a value older than L3's, at
the cost of answering "not ready" between a write and the next warm.

**One per collection.** A copy listens to its collection from construction; build one for each
collection and keep it.

**Double-buffered copies.** A copy kept in the collection's own L1 is refused from the first write
until the next warm, and a warm rewrites it in place. Where readers must keep answering while the
tables change (a refresh that writes some rows every few minutes), :class:`BufferedCopies` keeps the
copies apart from the collections' L1: it builds a whole new set beside the live one, in a fresh
backend, proves every table of it, and only then makes it the live set (:class:`CopyGeneration`). A
reader takes the live generation once and reads it to the end; a swap never changes a generation a
reader holds, and a generation is never half built.

**One state of all the tables.** Each table's proof says it held still while it was read, not that
the tables agree with each other: a writer could change one table between the reads of two others.
So a build asks the writer's own record (``settled``, a seqlock: a stamp naming the last committed
write, or :class:`Unsettled` while one is in progress) before the first table and after the last, and keeps the
set only when the two agree.
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, Generic, Protocol, TypeVar, runtime_checkable

from threetears.observe import get_logger

from threetears.core.sql_fragments import equality_conditions, quote_identifier
from threetears.core.fingerprint import KeyFingerprint, key_fingerprint, postgres_fingerprint_sql

if TYPE_CHECKING:
    from threetears.core.backends.protocol import L3Reader

__all__ = [
    "DEFAULT_PAGE_SIZE",
    "BufferedCopies",
    "CompleteCopy",
    "CopyGeneration",
    "CopyProof",
    "IncompleteCopyError",
    "Unsettled",
    "WholeTableL1",
    "copy_table",
    "l3_fingerprint",
    "read_l3_rows",
]

log = get_logger(__name__)

#: rows per page: under the L3 rail's thousand-row answer, with one row to spare that says more follow
DEFAULT_PAGE_SIZE: Final = 999

#: what a copy that has not been warmed says
_NEVER_WARMED: Final = "it has not been warmed"


@runtime_checkable
class WholeTableL1(Protocol):
    """an L1 that can hold a whole table and say which keys it holds, as ``DuckDBBackend`` can."""

    def replace_all(self, table: str, rows: Sequence[Mapping[str, Any]], primary_key: str | tuple[str, ...]) -> int:
        """make ``table`` hold exactly ``rows``."""
        ...

    def stored_keys(self, table: str, key: Sequence[str]) -> list[tuple[Any, ...]]:
        """every row's key as stored."""
        ...

    def column_types(self, table: str) -> Mapping[str, str]:
        """each column's type code."""
        ...

    def serialize_value(self, value: Any, col_type: str) -> Any:
        """a value in its stored form."""
        ...

    def reset(self) -> None:
        """close it and drop everything it holds."""
        ...


class IncompleteCopyError(RuntimeError):
    """the L1 copy is not proven to hold every row of its table, so nothing may read it."""


@dataclass(frozen=True)
class CopyProof:
    """that an L1 copy held every row of its table when it was warmed.

    :ivar table: the table
    :ivar row_count: the rows L3 and the copy held
    :ivar digest: the L3 fingerprint's digest over every row's every column when the copy was taken
    :ivar proven_at: when the copy was proven
    """

    table: str
    row_count: int
    digest: str
    proven_at: datetime


class CompleteCopy:
    """a collection's L1 as a complete copy of its L3 table.

    :param collection: the collection; its L1 must replace a table whole (``replace_all``, which
        ``DuckDBBackend`` has) and it must have an L3
    :ptype collection: BaseCollection
    :param page_size: rows per L3 page
    :ptype page_size: int
    :raises ValueError: when the collection's L1 cannot replace a table whole, or the page size
        is not positive
    """

    def __init__(self, collection: Any, *, page_size: int = DEFAULT_PAGE_SIZE) -> None:
        l1 = collection.l1_backend
        if not isinstance(l1, WholeTableL1):
            raise ValueError(
                f"{collection.table_name}: a complete copy needs an L1 that can replace a table whole "
                f"(replace_all and stored_keys, as DuckDBBackend has); this one is {type(l1).__name__}"
            )
        if page_size < 1:
            raise ValueError(f"page_size must be positive, got {page_size}")
        self._collection = collection
        self._l1 = l1
        self._table: str = collection.table_name
        self._key: tuple[str, ...] = tuple(collection.primary_key_columns)
        self._page_size = page_size
        self._proof: CopyProof | None = None
        self._why_not = _NEVER_WARMED
        #: rises with every change to the L1, so a warm can tell one landed while it ran
        self._changes = 0
        collection.add_l1_change_listener(self._changed)

    @property
    def table(self) -> str:
        """the table this copies.

        :return: its name
        :rtype: str
        """
        return self._table

    @property
    def proof(self) -> CopyProof | None:
        """the proof, while the copy is complete; None before a warm and after a row left it.

        :return: the proof
        :rtype: CopyProof | None
        """
        return self._proof

    def require(self) -> CopyProof:
        """the proof that the copy is complete, for a caller about to read it.

        :return: the proof
        :rtype: CopyProof
        :raises IncompleteCopyError: when the copy is not proven complete, saying why
        """
        if self._proof is None:
            raise IncompleteCopyError(f"{self._table}: not proven complete: {self._why_not}")
        return self._proof

    async def warm(self) -> CopyProof:
        """read the whole L3 table, make the L1 hold exactly its rows, and prove it.

        :return: the proof
        :rtype: CopyProof
        :raises IncompleteCopyError: when the table changed (a row added, removed or changed in any
            value) while it was read, the rows read do not number its count, the L1 does not then
            hold them, or a row changed in the L1 meanwhile; the copy stays refused
        :raises Exception: what reading L3 raised; the copy stays refused, saying so
        """
        self._proof = None
        self._why_not = "it is being warmed"
        try:
            proof = await self._warm()
        except IncompleteCopyError:
            raise
        except BaseException as exc:  # prawduct:allow prawduct/broad-except -- recorded as why the copy is refused, then re-raised unchanged
            self._why_not = f"the last warm failed: {type(exc).__name__}: {exc}"
            raise
        return proof

    async def _warm(self) -> CopyProof:
        """the body of :meth:`warm`.

        :return: the proof
        :rtype: CopyProof
        :raises IncompleteCopyError: when the copy cannot be proven
        """
        changes = self._changes
        try:
            proof = await copy_table(
                self._collection.required_l3_pool, self._table, self._key, self._l1, page_size=self._page_size
            )
        except IncompleteCopyError as exc:
            self._why_not = str(exc).removeprefix(f"{self._table}: not proven complete: ")
            raise
        if self._changes != changes:
            self._why_not = "a row changed in L1 while the copy was warmed"
            log.warning("complete copy refused", extra={"extra_data": {"table": self._table, "reason": self._why_not}})
            raise IncompleteCopyError(f"{self._table}: not proven complete: {self._why_not}")
        self._proof = proof
        return proof

    def _changed(self, entity_id: Any) -> None:
        """a row changed in, or left, the L1: the copy is no longer known to be complete.

        :param entity_id: the row's key
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        self._changes += 1
        if self._proof is not None:
            log.info("complete copy voided by a change to its L1", extra={"extra_data": {"table": self._table}})
        self._proof = None
        self._why_not = "a row changed after it was copied; it must be warmed again"


def _filtered(where: Mapping[str, Any] | None, first: int) -> tuple[str, list[Any]]:
    """the conditions keeping only ``where``'s rows, their placeholders numbered from ``first``.

    :param where: equality filters, column -> value; columns are TRUSTED identifiers
    :ptype where: Mapping[str, Any] | None
    :param first: the first placeholder's number
    :ptype first: int
    :return: the conditions joined by ``AND`` (empty without filters), and their values
    :rtype: tuple[str, list[Any]]
    """
    return equality_conditions(where, first=first)


async def l3_fingerprint(
    l3: L3Reader, table: str, columns: Sequence[str], *, where: Mapping[str, Any] | None = None
) -> KeyFingerprint:
    """count an L3 table (or the part ``where`` names) and fingerprint ``columns`` of every row, in one statement.

    :param l3: the L3 backend
    :ptype l3: L3Reader
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param columns: the columns the digest covers, TRUSTED identifiers
    :ptype columns: Sequence[str]
    :param where: equality filters naming the rows; every row when None
    :ptype where: Mapping[str, Any] | None
    :return: the fingerprint
    :rtype: KeyFingerprint
    """
    conditions, values = _filtered(where, 1)
    filters = f" WHERE {conditions}" if conditions else ""
    row = await l3.fetchrow(
        postgres_fingerprint_sql(quote_identifier(table), [quote_identifier(c) for c in columns], filters), *values
    )
    return KeyFingerprint(row_count=int(row["row_count"]), digest=str(row["digest"]))


async def read_l3_rows(
    l3: L3Reader,
    table: str,
    columns: Sequence[str],
    key: Sequence[str],
    *,
    where: Mapping[str, Any] | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> list[dict[str, Any]]:
    """every row of an L3 table (or the part ``where`` names), in key order, a page at a time.

    The L3 rail answers at most a thousand rows a statement and does not say when it cut, so a read
    that did not page would be silently short: this asks for one row more than a page and pages on
    the key until a page comes back without it.

    **A whole-table read, not a bounded one.** Each page is ``ORDER BY`` the whole key, which a
    btree key answers in order but a YugabyteDB hash-sharded key answers by reading and sorting
    every row the filters leave, once a page. Fine for a table copied whole that is small enough
    to copy; for the rows some leading-key values hold, read
    :meth:`~threetears.core.collections.schema_backed.SchemaBackedCollection.read_rows_led_by`,
    every statement of which the key leads.

    :param l3: the L3 backend
    :ptype l3: L3Reader
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param columns: the columns to read, TRUSTED identifiers
    :ptype columns: Sequence[str]
    :param key: the table's key, TRUSTED identifiers; unique, so paging steps past no row
    :ptype key: Sequence[str]
    :param where: equality filters naming the rows; every row when None
    :ptype where: Mapping[str, Any] | None
    :param page_size: rows per page; under the rail's thousand
    :ptype page_size: int
    :return: the rows
    :rtype: list[dict[str, Any]]
    """
    selected = ", ".join(quote_identifier(c) for c in columns)
    order = ", ".join(quote_identifier(c) for c in key)
    head = f"SELECT {selected} FROM {quote_identifier(table)}"  # noqa: S608 - trusted identifiers
    tail = f" ORDER BY {order} LIMIT {page_size + 1}"
    filters, values = _filtered(where, 1)
    rows: list[dict[str, Any]] = []
    cursor: Sequence[Any] | None = None
    more = True
    while more:
        conditions = [filters] if filters else []
        params = list(values)
        if cursor is not None:
            marks = ", ".join(f"${len(values) + index}" for index in range(1, len(key) + 1))
            conditions.append(f"({order}) > ({marks})")
            params += list(cursor)
        where_sql = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        page = await l3.fetch(f"{head}{where_sql}{tail}", *params)
        kept = [dict(row) for row in page[:page_size]]
        rows.extend(kept)
        more = len(page) > page_size and bool(kept)
        if more:
            cursor = tuple(kept[-1][column] for column in key)
    return rows


def _held_differs(
    target: WholeTableL1, table: str, key: Sequence[str], rows: list[dict[str, Any]], count: int
) -> str | None:
    """why ``target`` does not hold exactly the rows read, or None when it does.

    :param target: the L1 the rows were written to
    :ptype target: WholeTableL1
    :param table: the table
    :ptype table: str
    :param key: the table's key
    :ptype key: Sequence[str]
    :param rows: the rows read
    :ptype rows: list[dict[str, Any]]
    :param count: the table's count
    :ptype count: int
    :return: the reason, or None
    :rtype: str | None
    """
    types = target.column_types(table)
    read = key_fingerprint(tuple(target.serialize_value(row[c], types.get(c, "VARCHAR")) for c in key) for row in rows)
    held = key_fingerprint(target.stored_keys(table, key))
    reason: str | None = None
    if held.row_count != count:
        reason = f"the L1 holds {held.row_count} rows where the table counts {count}"
    elif held != read:
        reason = "the L1's keys are not the keys read"
    return reason


async def copy_table(
    l3: L3Reader, table: str, key: Sequence[str], target: WholeTableL1, *, page_size: int = DEFAULT_PAGE_SIZE
) -> CopyProof:
    """read an L3 table whole into ``target`` and prove the copy, or raise.

    The table is fingerprinted over every column ``target`` holds for it, before the first page and
    after the last; the copy is proven when the two agree, the rows read number the count, and
    ``target`` then holds exactly those keys.

    :param l3: the L3 backend
    :ptype l3: L3Reader
    :param table: the table
    :ptype table: str
    :param key: its key
    :ptype key: Sequence[str]
    :param target: the L1 to hold the copy; its table's columns are the columns read
    :ptype target: WholeTableL1
    :param page_size: rows per L3 page
    :ptype page_size: int
    :return: the proof
    :rtype: CopyProof
    :raises IncompleteCopyError: when the table changed while it was read, the rows read do not
        number its count, or ``target`` does not then hold them
    """
    started = datetime.now(UTC)
    columns = tuple(target.column_types(table))
    before = await l3_fingerprint(l3, table, columns)
    rows = await read_l3_rows(l3, table, columns, key, page_size=page_size)
    after = await l3_fingerprint(l3, table, columns)
    failure: str | None = None
    if after != before:
        failure = f"the table changed while it was read ({before} before, {after} after)"
    elif len(rows) != before.row_count:
        failure = f"{len(rows)} rows were read where the table counts {before.row_count}"
    if failure is None:
        target.replace_all(table, rows, tuple(key))
        failure = _held_differs(target, table, key, rows, before.row_count)
    if failure is not None:
        log.warning("complete copy refused", extra={"extra_data": {"table": table, "reason": failure}})
        raise IncompleteCopyError(f"{table}: not proven complete: {failure}")
    proof = CopyProof(table=table, row_count=before.row_count, digest=before.digest, proven_at=datetime.now(UTC))
    log.info(
        "complete copy proven",
        extra={
            "extra_data": {
                "table": table,
                "rows": before.row_count,
                "seconds": round((proof.proven_at - started).total_seconds(), 2),
            }
        },
    )
    return proof


StampT = TypeVar("StampT")


@dataclass(frozen=True)
class Unsettled:
    """the writer's record while a write is in progress: no stamp to build against, and why.

    :ivar reason: what the writer says of the write (which one, how long it has run)
    """

    reason: str


@dataclass(frozen=True)
class CopyGeneration(Generic[StampT]):
    """one complete set of copies: every table, proven, in a backend nothing writes to again.

    :ivar backend: the L1 holding every table's copy
    :ivar proofs: each table's proof, by table
    :ivar stamp: the writer's record when the set was taken (what ``settled`` answered before the
        first table and again after the last)
    :ivar built_at: when the set was proven
    """

    backend: Any
    proofs: Mapping[str, CopyProof]
    stamp: StampT
    built_at: datetime


class BufferedCopies(Generic[StampT]):
    """complete copies of several L3 tables, built beside the live set and swapped in whole.

    :param collections: the tables' collections, for their names, keys and L3
    :ptype collections: Sequence[BaseCollection]
    :param new_backend: makes an empty L1 holding every table (``DuckDBBackend`` initialized with
        their metadata); a build copies into a new one each time
    :ptype new_backend: Callable[[], WholeTableL1]
    :param settled: the writer's record: a stamp naming the last committed write, or
        :class:`Unsettled` (saying why) while one is in progress; equal stamps mean no write committed
        between them (``ScopeEpochs.settled``)
    :ptype settled: Callable[[], Awaitable[StampT | Unsettled]]
    :param page_size: rows per L3 page
    :ptype page_size: int
    """

    def __init__(
        self,
        collections: Sequence[Any],
        new_backend: Callable[[], WholeTableL1],
        settled: Callable[[], Awaitable[StampT | Unsettled]],
        *,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> None:
        self._tables = tuple(
            (collection.table_name, tuple(collection.primary_key_columns), collection) for collection in collections
        )
        self._new_backend = new_backend
        self._settled = settled
        self._page_size = page_size
        self._current: CopyGeneration[StampT] | None = None
        self._building = asyncio.Lock()

    @property
    def current(self) -> CopyGeneration[StampT] | None:
        """the live set; None before the first build has succeeded.

        :return: the live generation
        :rtype: CopyGeneration | None
        """
        return self._current

    def require(self) -> CopyGeneration[StampT]:
        """the live set, for a reader about to read it; hold it for the whole read.

        :return: the live generation
        :rtype: CopyGeneration
        :raises IncompleteCopyError: before the first build has succeeded
        """
        if self._current is None:
            raise IncompleteCopyError("no complete copy has been built yet")
        return self._current

    async def build(self) -> CopyGeneration[StampT]:
        """build a new set beside the live one and make it the live set once every table is proven.

        :return: the new live generation
        :rtype: CopyGeneration
        :raises IncompleteCopyError: when a write is in progress, a write committed while the set was
            built, or a table could not be proven; the live set stays as it was
        :raises Exception: what reading L3 raised; the live set stays as it was
        """
        async with self._building:
            return await self._build()

    async def build_if_behind(self) -> CopyGeneration[StampT]:
        """build a new set only when the writer has committed since the live set was taken.

        While a write is in progress there is nothing newer to build, so the live set is answered as
        it is (it is still one complete state, the last committed one).

        :return: the live generation, new or not
        :rtype: CopyGeneration
        :raises IncompleteCopyError: when there is no live set and none can be built yet (a write in
            progress), or a build was needed and could not be proven
        """
        async with self._building:
            generation = self._current
            stamp = await self._settled()
            if generation is None or (not isinstance(stamp, Unsettled) and stamp != generation.stamp):
                generation = await self._build()
            elif isinstance(stamp, Unsettled):
                log.info(
                    "complete copies not rebuilt: a write is in progress; the live copies stay",
                    extra={"extra_data": {"live": str(generation.stamp), "write": stamp.reason}},
                )
            return generation

    async def _build(self) -> CopyGeneration[StampT]:
        """the body of :meth:`build`, under the build lock.

        :return: the new live generation
        :rtype: CopyGeneration
        :raises IncompleteCopyError: when the set cannot be proven one state of every table
        """
        started = datetime.now(UTC)
        before = await self._settled()
        if isinstance(before, Unsettled):
            raise IncompleteCopyError(f"not built: a write is in progress ({before.reason}); the live copies stay")
        backend = self._new_backend()
        proofs: dict[str, CopyProof] = {}
        try:
            for table, key, collection in self._tables:
                proofs[table] = await copy_table(
                    collection.required_l3_pool, table, key, backend, page_size=self._page_size
                )
            after = await self._settled()
        except (
            BaseException
        ):  # prawduct:allow prawduct/broad-except -- closes the half-filled backend, then re-raises unchanged
            backend.reset()
            raise
        if after != before:
            backend.reset()
            log.warning(
                "complete copies refused: a write committed while they were built",
                extra={"extra_data": {"before": str(before), "after": str(after)}},
            )
            raise IncompleteCopyError(
                f"not built: the tables changed while the copies were built ({before} before, {after} after); "
                "the live copies stay"
            )
        generation = CopyGeneration(backend=backend, proofs=proofs, stamp=before, built_at=datetime.now(UTC))
        superseded = self._current
        # one assignment: a reader takes the old set or the new, never part of either
        self._current = generation
        if superseded is not None:
            # freed when its last reader lets go; said then, so a set a reader keeps alive shows
            weakref.finalize(
                superseded.backend,
                log.info,
                "superseded complete copies released",
                extra={"extra_data": {"stamp": str(superseded.stamp)}},
            )
            log.info("complete copies superseded", extra={"extra_data": {"stamp": str(superseded.stamp)}})
        log.info(
            "complete copies swapped in",
            extra={
                "extra_data": {
                    "stamp": str(before),
                    "rows": {table: proof.row_count for table, proof in proofs.items()},
                    "seconds": round((generation.built_at - started).total_seconds(), 2),
                }
            },
        )
        return generation
