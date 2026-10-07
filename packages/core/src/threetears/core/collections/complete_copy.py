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
before the first page and again after the last. The copy is proven only when the two readings
agree (nothing was written during the read), the rows read number the count, and the L1 then
holds that many rows with the same keys (their fingerprint over the keys as L1 stores them).
:meth:`CompleteCopy.require` returns the proof, or raises :class:`IncompleteCopyError`, and a
query over the L1 calls it first.

**A row leaving L1 voids the proof.** Every eviction from the collection's L1 -- a write this
process settled, a peer's invalidation broadcast -- reaches :meth:`CompleteCopy` through the
collection's eviction listener, and the copy is refused until it is warmed again. One that lands
during a warm fails that warm. So a reader is never given a copy with a row missing, at the cost
of answering "not ready" between a write and the next warm.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

from threetears.observe import get_logger

from threetears.core.cache.base import quote_identifier
from threetears.core.fingerprint import KeyFingerprint, key_fingerprint, postgres_fingerprint_sql

__all__ = ["DEFAULT_PAGE_SIZE", "CompleteCopy", "CopyProof", "IncompleteCopyError"]

log = get_logger(__name__)

#: rows per page: under the L3 rail's thousand-row answer, with one row to spare that says more follow
DEFAULT_PAGE_SIZE: Final = 999

#: what a copy that has not been warmed says
_NEVER_WARMED: Final = "it has not been warmed"


class IncompleteCopyError(RuntimeError):
    """the L1 copy is not proven to hold every row of its table, so nothing may read it."""


@dataclass(frozen=True)
class CopyProof:
    """that an L1 copy held every row of its table when it was warmed.

    :ivar table: the table
    :ivar row_count: the rows L3 and the copy held
    :ivar digest: the L3 fingerprint's key digest when the copy was taken
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
        if l1 is None or not hasattr(l1, "replace_all") or not hasattr(l1, "stored_keys"):
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
        #: rises with every eviction, so a warm can tell one landed while it ran
        self._evictions = 0
        collection.add_l1_eviction_listener(self._evicted)

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
        :raises IncompleteCopyError: when the table changed while it was read, the rows read do not
            number its count, the L1 does not then hold them, or a row left the L1 meanwhile; the
            copy stays refused
        """
        self._proof = None
        self._why_not = "it is being warmed"
        evictions = self._evictions
        started = datetime.now(UTC)
        before = await self._l3_fingerprint()
        rows = await self._read_all()
        after = await self._l3_fingerprint()
        failure: str | None = None
        if after != before:
            failure = f"the table changed while it was read ({before} before, {after} after)"
        elif len(rows) != before.row_count:
            failure = f"{len(rows)} rows were read where the table counts {before.row_count}"
        if failure is None:
            self._l1.replace_all(self._table, rows, self._key)
            failure = self._l1_differs(rows, before.row_count)
        if failure is None and self._evictions != evictions:
            failure = "a row changed in L3 while the copy was warmed"
        if failure is not None:
            self._why_not = failure
            log.warning("complete copy refused", extra={"extra_data": {"table": self._table, "reason": failure}})
            raise IncompleteCopyError(f"{self._table}: not proven complete: {failure}")
        self._proof = CopyProof(
            table=self._table, row_count=before.row_count, digest=before.digest, proven_at=datetime.now(UTC)
        )
        log.info(
            "complete copy proven",
            extra={
                "extra_data": {
                    "table": self._table,
                    "rows": before.row_count,
                    "seconds": round((self._proof.proven_at - started).total_seconds(), 2),
                }
            },
        )
        return self._proof

    def _evicted(self, entity_id: Any) -> None:
        """a row left the L1: the copy is no longer known to be complete.

        :param entity_id: the row's key
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        self._evictions += 1
        if self._proof is not None:
            log.info("complete copy voided by an eviction", extra={"extra_data": {"table": self._table}})
        self._proof = None
        self._why_not = "a row changed in L3 after it was copied; it must be warmed again"

    async def _l3_fingerprint(self) -> KeyFingerprint:
        """count and fingerprint the L3 table in one statement.

        :return: the fingerprint
        :rtype: KeyFingerprint
        """
        row = await self._collection.required_l3_pool.fetchrow(
            postgres_fingerprint_sql(quote_identifier(self._table), [quote_identifier(c) for c in self._key])
        )
        return KeyFingerprint(row_count=int(row["row_count"]), digest=str(row["digest"]))

    async def _read_all(self) -> list[dict[str, Any]]:
        """every row of the L3 table, in key order, a page at a time.

        :return: the rows
        :rtype: list[dict[str, Any]]
        """
        l3 = self._collection.required_l3_pool
        columns = ", ".join(quote_identifier(c) for c in self._l1.column_types(self._table))
        order = ", ".join(quote_identifier(c) for c in self._key)
        head = f"SELECT {columns} FROM {quote_identifier(self._table)}"  # noqa: S608 - trusted identifiers
        tail = f" ORDER BY {order} LIMIT {self._page_size + 1}"
        rows: list[dict[str, Any]] = []
        cursor: Sequence[Any] | None = None
        more = True
        while more:
            if cursor is None:
                page = await l3.fetch(head + tail)
            else:
                marks = ", ".join(f"${index}" for index in range(1, len(self._key) + 1))
                page = await l3.fetch(f"{head} WHERE ({order}) > ({marks}){tail}", *cursor)
            kept = [dict(row) for row in page[: self._page_size]]
            rows.extend(kept)
            more = len(page) > self._page_size and bool(kept)
            if more:
                cursor = tuple(kept[-1][column] for column in self._key)
        return rows

    def _l1_differs(self, rows: list[dict[str, Any]], count: int) -> str | None:
        """why the L1 does not hold exactly the rows read, or None when it does.

        :param rows: the rows read
        :ptype rows: list[dict[str, Any]]
        :param count: the table's count
        :ptype count: int
        :return: the reason, or None
        :rtype: str | None
        """
        types = self._l1.column_types(self._table)
        read = key_fingerprint(
            tuple(self._l1.serialize_value(row[c], types.get(c, "VARCHAR")) for c in self._key) for row in rows
        )
        held = key_fingerprint(self._l1.stored_keys(self._table, self._key))
        reason: str | None = None
        if held.row_count != count:
            reason = f"the L1 holds {held.row_count} rows where the table counts {count}"
        elif held != read:
            reason = "the L1's keys are not the keys read"
        return reason
