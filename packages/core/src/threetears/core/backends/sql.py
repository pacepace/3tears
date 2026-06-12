"""``SqlL3Backend`` — the default L3 backend over an asyncpg-shaped pool.

Implements **both** L3-tier protocols:

- :class:`~threetears.core.backends.protocol.L3Backend` — the raw-SQL transport, by
  delegating to the wrapped pool (``fetch`` / ``fetchrow`` / ``execute`` / ``acquire`` /
  ``transaction``; ``execute_batch`` runs the batch in one pool transaction).
- :class:`~threetears.core.backends.protocol.DurableStore` — the structured ops, by
  *generating* parameterized SQL (``fetch_one`` → SELECT, ``upsert`` →
  INSERT…ON CONFLICT, ``delete`` → DELETE, ``scan`` → SELECT…WHERE). This is the
  reference SQL implementation of the same structured contract a ``GitL3Backend``
  satisfies with file read/write/delete + commit.

The wrapped pool is the previously-untyped ``l3_pool`` (a bare asyncpg ``Pool`` or the
``NatsProxyL3Backend``); ``SqlL3Backend`` is the named, protocol-conformant wrapper the
``L3Backend``-retyped registry/base resolve to.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from threetears.core.backends.protocol import parse_rowcount

__all__ = ["SqlL3Backend"]

_ON_CONFLICT_VALUES = frozenset({"update", "ignore", "raise"})


def _quote_ident(name: str) -> str:
    """Double-quote a SQL identifier, rejecting embedded quotes (no injection via column names).

    :param name: a table or column identifier.
    :ptype name: str
    :return: the double-quoted identifier.
    :rtype: str
    :raises ValueError: if ``name`` contains a double-quote.
    """
    if '"' in name:
        raise ValueError(f"illegal SQL identifier {name!r}")
    return f'"{name}"'


class SqlL3Backend:
    """An :class:`L3Backend` + :class:`DurableStore` over an asyncpg-shaped pool.

    :param pool: the underlying durable handle — an asyncpg ``Pool`` (or anything with
        the same ``fetch`` / ``fetchrow`` / ``execute`` / ``acquire`` surface, e.g. the
        ``NatsProxyL3Backend``). Lifecycle (open/close) is owned by the caller.
    :ptype pool: Any
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    # ── L3Backend: raw-SQL transport (delegate to the pool) ─────────────────────────
    async def fetch(self, query: str, *params: Any, namespace: str | None = None) -> list[dict[str, Any]]:
        """Run a SELECT and return all rows as dicts (delegates to the pool)."""
        rows = await self._pool.fetch(query, *params)
        return [dict(r) for r in rows]

    async def fetchrow(self, query: str, *params: Any, namespace: str | None = None) -> dict[str, Any] | None:
        """Run a SELECT and return the first row dict, or ``None`` (delegates to the pool)."""
        row = await self._pool.fetchrow(query, *params)
        return dict(row) if row is not None else None

    async def execute(self, query: str, *params: Any, namespace: str | None = None) -> str:
        """Run an INSERT/UPDATE/DELETE; return the asyncpg command-tag string (``"UPDATE 1"``)."""
        result = await self._pool.execute(query, *params)
        return result if isinstance(result, str) else ""

    async def execute_batch(
        self, queries: list[dict[str, Any]], *, namespace: str | None = None, transaction: bool = True
    ) -> list[Any]:
        """Run ``{query, params}`` dicts; atomically in one pool transaction when ``transaction``."""
        if not transaction:
            results: list[Any] = []
            for q in queries:
                results.append(await self._pool.execute(q["query"], *q.get("params", [])))
            return results
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                out: list[Any] = []
                for q in queries:
                    out.append(await conn.execute(q["query"], *q.get("params", [])))
                return out

    def acquire(self) -> Any:
        """Return the pool's ``acquire()`` async context manager (a pooled connection)."""
        return self._pool.acquire()

    def transaction(self, namespace: str | None = None) -> Any:
        """Return a pool-level transaction async context manager (``acquire`` + ``conn.transaction``)."""
        if hasattr(self._pool, "transaction"):
            return self._pool.transaction(namespace=namespace)
        return _PoolTransactionCM(self._pool)

    # ── DurableStore: structured ops (generate SQL) ─────────────────────────────────
    async def fetch_one(self, table: str, pk: Mapping[str, Any]) -> dict[str, Any] | None:
        """Fetch the row whose primary key equals ``pk`` via a generated SELECT, or ``None``."""
        where, params = self._where(pk, start=1)
        sql = f"SELECT * FROM {_quote_ident(table)} WHERE {where}"
        return await self.fetchrow(sql, *params)

    async def upsert(
        self,
        table: str,
        row: Mapping[str, Any],
        *,
        pk: Sequence[str],
        on_conflict: str = "update",
        cas: datetime | None = None,
    ) -> int:
        """Insert-or-update ``row`` via a generated INSERT…ON CONFLICT; return rows affected.

        ``on_conflict`` selects the conflict clause (``"update"``/``"ignore"``/``"raise"``);
        ``cas`` fences the update on the prior ``date_updated`` (a mismatch → 0 rows).
        """
        if on_conflict not in _ON_CONFLICT_VALUES:
            raise ValueError(f"on_conflict must be one of {sorted(_ON_CONFLICT_VALUES)!r}, got {on_conflict!r}")
        cols = list(row.keys())
        placeholders = ", ".join(f"${i}" for i in range(1, len(cols) + 1))
        col_sql = ", ".join(_quote_ident(c) for c in cols)
        pk_sql = ", ".join(_quote_ident(c) for c in pk)
        insert = f"INSERT INTO {_quote_ident(table)} ({col_sql}) VALUES ({placeholders})"
        params: list[Any] = [row[c] for c in cols]

        if on_conflict == "raise":
            return parse_rowcount(await self.execute(insert, *params))
        if on_conflict == "ignore":
            return parse_rowcount(await self.execute(f"{insert} ON CONFLICT ({pk_sql}) DO NOTHING", *params))

        # "update": upsert the mutable (non-pk) columns. Optionally fence on the CAS value.
        mutable = [c for c in cols if c not in set(pk)]
        set_sql = (
            ", ".join(f"{_quote_ident(c)} = EXCLUDED.{_quote_ident(c)}" for c in mutable) or pk_sql + " = " + pk_sql
        )
        sql = f"{insert} ON CONFLICT ({pk_sql}) DO UPDATE SET {set_sql}"
        if cas is not None and "date_updated" in row:
            params.append(cas)
            sql += f" WHERE {_quote_ident(table)}.{_quote_ident('date_updated')} = ${len(params)}"
        return parse_rowcount(await self.execute(sql, *params))

    async def delete(self, table: str, pk: Mapping[str, Any]) -> None:
        """Delete the row whose primary key equals ``pk`` via a generated DELETE (missing is not an error)."""
        where, params = self._where(pk, start=1)
        await self.execute(f"DELETE FROM {_quote_ident(table)} WHERE {where}", *params)

    async def scan(self, table: str, filters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Return rows matching the equality ``filters`` via a generated SELECT (all rows when empty)."""
        if not filters:
            return await self.fetch(f"SELECT * FROM {_quote_ident(table)}")
        where, params = self._where(filters, start=1)
        return await self.fetch(f"SELECT * FROM {_quote_ident(table)} WHERE {where}", *params)

    @staticmethod
    def _where(cols: Mapping[str, Any], *, start: int) -> tuple[str, list[Any]]:
        """Build a ``col = $n AND …`` clause + ordered params from a column→value mapping."""
        parts: list[str] = []
        params: list[Any] = []
        for i, (col, val) in enumerate(cols.items(), start=start):
            parts.append(f"{_quote_ident(col)} = ${i}")
            params.append(val)
        return " AND ".join(parts), params


class _PoolTransactionCM:
    """``acquire()`` + ``conn.transaction()`` for a bare pool without a ``transaction()`` helper."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self._acquire_cm: Any = None
        self._conn: Any = None
        self._tx: Any = None

    async def __aenter__(self) -> Any:
        self._acquire_cm = self._pool.acquire()
        self._conn = await self._acquire_cm.__aenter__()
        self._tx = self._conn.transaction()
        await self._tx.__aenter__()
        return self._conn

    async def __aexit__(self, *exc: Any) -> None:
        await self._tx.__aexit__(*exc)
        await self._acquire_cm.__aexit__(*exc)
