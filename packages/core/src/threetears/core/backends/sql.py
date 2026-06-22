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

**Schema awareness.** A :class:`~threetears.core.collections.schema_backed.SchemaBackedCollection`
registers its :class:`TableSchema` via :meth:`SqlL3Backend.register_schema`. Once a schema
is registered for a table, the structured ops generate **schema-aware** SQL (declared-
columns-only SELECT with ``VECTOR``/``TSVECTOR`` ``::text`` projection casts, ``::jsonb`` /
``::vector`` write casts, composite-PK WHERE, server-default column dropping, CAS fencing,
``on_conflict`` modes) + read/write codec — byte-identical to the hand-rolled collection
path it replaces. When **no** schema is registered for a table, the generic ``SELECT *`` /
INSERT path is used (the contract a non-collection caller of :class:`DurableStore` relies on).

The wrapped pool is the previously-untyped ``l3_pool`` (a bare asyncpg ``Pool`` or the
``NatsProxyL3Backend``); ``SqlL3Backend`` is the named, protocol-conformant wrapper the
``L3Backend``-retyped registry/base resolve to.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

import threetears.core.backends.schema_sql as schema_sql
from threetears.core.backends.protocol import parse_rowcount

if TYPE_CHECKING:
    from threetears.core.collections.schema_backed import TableSchema

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
        # table name -> TableSchema. when a schema is registered for a table the
        # structured DurableStore ops generate schema-aware SQL + run the per-column
        # read/write codec; otherwise the generic SELECT */INSERT path is used.
        self._schemas: dict[str, TableSchema] = {}
        # capability sniff (mirrors the ``hasattr(self._pool, "transaction")`` check in
        # ``transaction`` below): a namespace-aware transport such as
        # ``NatsProxyL3Backend`` carries ``accepts_scoped_reads = True`` and routes
        # ``namespace`` + ``customer_scope`` to the broker, so the raw-SQL transport
        # methods forward those kwargs instead of dropping them. NOT an isinstance gate
        # -- NatsProxyL3Backend omits ``fetchval`` so it fails the L3Backend protocol
        # check, which would silently leave forwarding off. a bare asyncpg pool lacks
        # the marker, so its behaviour is unchanged (kwargs dropped).
        #
        # identity (``is True``), not truthiness: the marker is the literal ``True`` on the
        # transport class. ``bool(getattr(...))`` would be fooled by a ``MagicMock`` pool,
        # whose auto-created attribute is a truthy mock object, flipping forwarding on and
        # pushing ``namespace=`` into a mock signature that rejects it.
        self._scope_aware: bool = getattr(pool, "accepts_scoped_reads", False) is True

    def __getattr__(self, name: str) -> Any:
        """Delegate any unlisted attribute to the wrapped pool (the raw-SQL escape hatch).

        ``L3Backend`` formalizes the common transport surface (``fetch`` / ``fetchrow`` /
        ``execute`` / ``execute_batch`` / ``acquire`` / ``transaction``), but the
        ``l3_pool`` ivar is also the documented ad-hoc-SQL escape hatch — product code
        reaches for the full asyncpg pool surface through it (e.g. ``fetchval``,
        ``copy_records_to_table``). Forwarding the long tail keeps the wrapper a
        transparent superset of the pool it wraps, so wrapping never removes a method a
        caller already relied on. Invoked only for attributes not found normally
        (``_pool`` / ``_schemas`` and every defined method resolve before this).

        :param name: the missing attribute name.
        :ptype name: str
        :return: the corresponding attribute on the wrapped pool.
        :rtype: Any
        :raises AttributeError: if the wrapped pool has no such attribute.
        """
        return getattr(self._pool, name)

    def register_schema(self, table: str, schema: TableSchema) -> None:
        """Register a :class:`TableSchema` so the structured ops emit schema-aware SQL for ``table``.

        Idempotent — re-registering the same table overwrites the prior schema (a
        Collection registers its schema on construction; reconstruction is harmless).

        :param table: target table name (matches ``schema.name``).
        :ptype table: str
        :param schema: the table schema driving SQL generation + value coercion.
        :ptype schema: TableSchema
        :return: nothing
        :rtype: None
        """
        self._schemas[table] = schema

    # ── L3Backend: raw-SQL transport (delegate to the pool) ─────────────────────────
    # ``namespace`` + extra ``**kwargs`` (e.g. NatsProxy's ``customer_scope``) are
    # forwarded to the pool ONLY when it declared ``accepts_scoped_reads`` (see
    # __init__): a namespace-aware transport routes them to the broker; a bare asyncpg
    # pool would TypeError on them, so they are dropped for it (the historical path).
    # generic by design -- the wrapper forwards opaque kwargs and stays ignorant of
    # NATS-specific concepts like ``customer_scope``.
    async def fetch(
        self, query: str, *params: Any, namespace: str | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Run a SELECT and return all rows as dicts (delegates to the pool)."""
        if self._scope_aware:
            rows = await self._pool.fetch(query, *params, namespace=namespace, **kwargs)
        else:
            rows = await self._pool.fetch(query, *params)
        return [dict(r) for r in rows]

    async def fetchrow(
        self, query: str, *params: Any, namespace: str | None = None, **kwargs: Any
    ) -> dict[str, Any] | None:
        """Run a SELECT and return the first row dict, or ``None`` (delegates to the pool)."""
        if self._scope_aware:
            row = await self._pool.fetchrow(query, *params, namespace=namespace, **kwargs)
        else:
            row = await self._pool.fetchrow(query, *params)
        return dict(row) if row is not None else None

    async def fetchval(self, query: str, *params: Any, namespace: str | None = None, **kwargs: Any) -> Any:
        """Run a SELECT and return the first column of the first row (scalar) (delegates to the pool)."""
        if self._scope_aware:
            return await self._pool.fetchval(query, *params, namespace=namespace, **kwargs)
        return await self._pool.fetchval(query, *params)

    async def execute(self, query: str, *params: Any, namespace: str | None = None, **kwargs: Any) -> str:
        """Run an INSERT/UPDATE/DELETE; return the asyncpg command-tag string (``"UPDATE 1"``)."""
        if self._scope_aware:
            result = await self._pool.execute(query, *params, namespace=namespace, **kwargs)
        else:
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
    async def fetch_one(self, table: str, pk: Mapping[str, Any], *, conn: Any = None) -> dict[str, Any] | None:
        """Fetch the row whose primary key equals ``pk`` via a generated SELECT, or ``None``.

        When a schema is registered for ``table`` the SELECT projects the schema's
        declared columns (``VECTOR``/``TSVECTOR`` cast ``::text``), the pk values are
        write-coerced, and the row is read-coerced — byte-identical to the collection's
        old ``fetch_from_store``. Otherwise a generic ``SELECT *`` is issued.
        """
        schema = self._schemas.get(table)
        if schema is None:
            where, params = self._where(pk, start=1)
            sql = f"SELECT * FROM {_quote_ident(table)} WHERE {where}"
            return await self._fetchrow(sql, *params, conn=conn)
        coerced: list[Any] = [
            schema_sql.normalize_write_value(schema.column(name), pk[name]) for name in schema.pk_columns
        ]
        row = await self._fetchrow(schema_sql.build_fetch_sql(schema), *coerced, conn=conn)
        if row is None:
            return None
        return schema_sql.coerce_row(schema, dict(row))

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
        """Insert-or-update ``row`` via a generated INSERT…ON CONFLICT; return rows affected.

        When a schema is registered for ``table`` the INSERT / CAS-UPDATE SQL + params are
        generated from the schema (``on_conflict`` modes, server-default dropping, CAS
        fencing, ``::jsonb`` / ``::vector`` casts) — byte-identical to the collection's old
        ``save_to_store``; ``pk`` / ``on_conflict`` arguments are taken from the schema.
        Otherwise the generic INSERT path runs, honoring the ``pk`` / ``on_conflict`` / ``cas``
        arguments directly.

        :param conn: optional caller-supplied connection (a transaction handle) the write
            binds to instead of the pool, so it commits atomically with the caller's
            other operations. ``None`` uses the wrapped pool.
        :ptype conn: Any
        """
        schema = self._schemas.get(table)
        if schema is not None:
            return await self._upsert_schema(schema, dict(row), cas=cas, conn=conn)

        if pk is None:
            raise ValueError("upsert without a registered schema requires the pk argument")
        if on_conflict not in _ON_CONFLICT_VALUES:
            raise ValueError(f"on_conflict must be one of {sorted(_ON_CONFLICT_VALUES)!r}, got {on_conflict!r}")
        cols = list(row.keys())
        placeholders = ", ".join(f"${i}" for i in range(1, len(cols) + 1))
        col_sql = ", ".join(_quote_ident(c) for c in cols)
        pk_sql = ", ".join(_quote_ident(c) for c in pk)
        insert = f"INSERT INTO {_quote_ident(table)} ({col_sql}) VALUES ({placeholders})"
        params: list[Any] = [row[c] for c in cols]

        if on_conflict == "raise":
            return parse_rowcount(await self._execute(insert, *params, conn=conn))
        if on_conflict == "ignore":
            return parse_rowcount(
                await self._execute(f"{insert} ON CONFLICT ({pk_sql}) DO NOTHING", *params, conn=conn)
            )

        # "update": upsert the mutable (non-pk) columns. Optionally fence on the CAS value.
        mutable = [c for c in cols if c not in set(pk)]
        set_sql = (
            ", ".join(f"{_quote_ident(c)} = EXCLUDED.{_quote_ident(c)}" for c in mutable) or pk_sql + " = " + pk_sql
        )
        sql = f"{insert} ON CONFLICT ({pk_sql}) DO UPDATE SET {set_sql}"
        if cas is not None and "date_updated" in row:
            params.append(cas)
            sql += f" WHERE {_quote_ident(table)}.{_quote_ident('date_updated')} = ${len(params)}"
        return parse_rowcount(await self._execute(sql, *params, conn=conn))

    async def _upsert_schema(
        self,
        schema: TableSchema,
        data: dict[str, Any],
        *,
        cas: datetime | None,
        conn: Any,
    ) -> int:
        """Schema-aware upsert: byte-identical to the collection's old ``save_to_store``."""
        cas_eligible = schema.cas_column is not None and cas is not None and schema.on_conflict == "update"
        if cas_eligible:
            assert cas is not None  # narrowed by cas_eligible
            sql = schema_sql.build_cas_update_sql(schema, data)
            params = schema_sql.build_cas_params(schema, data, cas)
        else:
            sql = schema_sql.build_insert_sql(schema, data)
            params = schema_sql.build_insert_params(schema, data)
        return parse_rowcount(await self._execute(sql, *params, conn=conn))

    async def delete(self, table: str, pk: Mapping[str, Any], *, conn: Any = None) -> None:
        """Delete the row whose primary key equals ``pk`` via a generated DELETE (missing is not an error).

        When a schema is registered for ``table`` the pk values are write-coerced and the
        DELETE projects the declared composite pk — byte-identical to the collection's old
        ``delete_from_store``. Otherwise a generic DELETE keyed on the ``pk`` mapping runs.
        """
        schema = self._schemas.get(table)
        if schema is None:
            where, params = self._where(pk, start=1)
            await self._execute(f"DELETE FROM {_quote_ident(table)} WHERE {where}", *params, conn=conn)
            return
        coerced: list[Any] = [
            schema_sql.normalize_write_value(schema.column(name), pk[name]) for name in schema.pk_columns
        ]
        await self._execute(schema_sql.build_delete_sql(schema), *coerced, conn=conn)

    async def scan(self, table: str, filters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Return rows matching the equality ``filters`` via a generated SELECT (all rows when empty)."""
        if not filters:
            return await self.fetch(f"SELECT * FROM {_quote_ident(table)}")
        where, params = self._where(filters, start=1)
        return await self.fetch(f"SELECT * FROM {_quote_ident(table)} WHERE {where}", *params)

    # ── connection-aware execution helpers ──────────────────────────────────────────
    async def _execute(self, query: str, *params: Any, conn: Any = None) -> str:
        """Run a write on ``conn`` when supplied (transactional override), else the pool."""
        if conn is not None:
            result = await conn.execute(query, *params)
            return result if isinstance(result, str) else ""
        return await self.execute(query, *params)

    async def _fetchrow(self, query: str, *params: Any, conn: Any = None) -> dict[str, Any] | None:
        """Fetch one row on ``conn`` when supplied (transactional override), else the pool."""
        if conn is not None:
            row = await conn.fetchrow(query, *params)
            return dict(row) if row is not None else None
        return await self.fetchrow(query, *params)

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
