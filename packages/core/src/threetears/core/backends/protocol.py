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

__all__ = [
    "DurableStore",
    "L3Backend",
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
    """

    async def fetch(self, query: str, *params: Any, namespace: str | None = None) -> list[dict[str, Any]]:
        """Run a SELECT and return all rows as dicts (empty list on no rows)."""
        ...

    async def fetchrow(self, query: str, *params: Any, namespace: str | None = None) -> dict[str, Any] | None:
        """Run a SELECT and return the first row dict, or ``None``."""
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

    async def fetch_one(self, table: str, pk: Mapping[str, Any]) -> dict[str, Any] | None:
        """Fetch the row whose primary key equals ``pk`` (column→value), or ``None`` on miss."""
        ...

    async def upsert(
        self,
        table: str,
        row: Mapping[str, Any],
        *,
        pk: Sequence[str],
        on_conflict: str = "update",
        cas: datetime | None = None,
    ) -> int:
        """Insert-or-update ``row`` keyed by the ``pk`` columns; return rows affected.

        :param on_conflict: ``"update"`` (upsert), ``"ignore"`` (no-op on conflict), or
            ``"raise"`` (insert only — conflict is an error).
        :ptype on_conflict: str
        :param cas: optimistic-lock fence — the pre-modification ``date_updated`` the
            update must still match; a mismatch yields **0 rows affected** (the caller
            raises ``ConcurrentModificationError``). ``None`` for inserts.
        :ptype cas: datetime | None
        :return: rows affected — ``1`` on success, ``0`` on a CAS/optimistic-lock miss.
        :rtype: int
        """
        ...

    async def delete(self, table: str, pk: Mapping[str, Any]) -> None:
        """Delete the row whose primary key equals ``pk``; a missing row is not an error."""
        ...

    async def scan(self, table: str, filters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Return every row matching the equality ``filters`` (all rows when ``None``/empty)."""
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
