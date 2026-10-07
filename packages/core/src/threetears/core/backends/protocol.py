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
from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

__all__ = [
    "BulkDurableStore",
    "DurableStore",
    "L3Backend",
    "OrderedDurableStore",
    "parse_rowcount",
]


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
        :param max_bytes: the most bytes of values one write may carry (a message's share of a bus)
        :ptype max_bytes: int
        :param conn: the caller's transaction handle the writes bind to; ``None`` uses the
            backend's own. A failing write fails that transaction whole
        :ptype conn: Any
        :return: rows written
        :rtype: int
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


def parse_rowcount(status: Any) -> int:
    """Parse an asyncpg command-tag (``"INSERT 0 1"`` / ``"UPDATE 1"`` / ``"DELETE 2"``) to a count.

    The framework-owned parser for the :meth:`L3Backend.execute` status tag. Empty /
    falsy / non-string values (mock pools sometimes return ``None`` or ``""``) resolve
    to ``0``. Consolidates the duplicated ``int(result.split()[-1])`` idiom across
    product code (the ``DurableStore`` structured ops return an int rowcount natively, so
    this is only needed at the raw-SQL :class:`L3Backend` border).

    :param status: the asyncpg status tag (or any value).
    :ptype status: Any
    :return: the rows-affected count; ``0`` when unparseable.
    :rtype: int
    """
    if not status or not isinstance(status, str):
        return 0
    parts = status.split()
    if not parts:
        return 0
    try:
        return int(parts[-1])
    except ValueError:
        return 0
