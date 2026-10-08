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

import json
import os
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from threetears.observe import get_logger

import threetears.core.backends.schema_sql as schema_sql
from threetears.core.backends.protocol import L3_RAIL_ROW_CAP, parse_rowcount
from threetears.core.keyset import read_keyset_pages
from threetears.core.sql_fragments import as_written

if TYPE_CHECKING:
    from threetears.core.collections.schema_backed import TableSchema

__all__ = ["SqlL3Backend", "bound_request_connection"]

_logger = get_logger(__name__)

_ON_CONFLICT_VALUES = frozenset({"update", "ignore", "raise"})

#: the asyncpg module whose release path raises the masking AttributeError.
_ASYNCPG_POOL_FILE = os.path.join("asyncpg", "pool.py")


def _is_asyncpg_release_race(exc: AttributeError) -> bool:
    """Whether ``exc`` is asyncpg's pool-release race masking the query's own error.

    asyncpg 0.31.0's ``PoolConnectionHolder.release`` waits for an in-flight cancellation
    after a query fails. If the server closes the connection during that wait, the holder's
    connection is cleared, and release then calls ``reset`` on ``None``. The resulting
    ``AttributeError`` escapes ``pool.acquire()``'s exit and replaces the error the query
    actually raised, which survives only as ``__context__``. Recognised by all three: raised
    inside asyncpg's ``pool.py``, on a ``None`` object, with an exception already in flight.

    :param exc: the AttributeError a pool call raised
    :ptype exc: AttributeError
    :return: whether it is the release race rather than a real attribute error
    :rtype: bool
    """
    tb = exc.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    raised_in_pool = tb is not None and tb.tb_frame.f_code.co_filename.endswith(_ASYNCPG_POOL_FILE)
    return raised_in_pool and exc.obj is None and exc.__context__ is not None


@contextmanager
def _unmask_asyncpg_release_race() -> Iterator[None]:
    """Re-raise the query's own error when asyncpg's release race has masked it.

    Any other ``AttributeError`` propagates unchanged.

    :return: context manager wrapping one pool call
    :rtype: Iterator[None]
    :raises BaseException: the exception the query raised, in place of the masking one
    """
    try:
        yield
    except AttributeError as exc:
        original = exc.__context__
        if not _is_asyncpg_release_race(exc) or original is None:
            raise
        _logger.warning(
            "asyncpg pool release raced a server-side close and masked the query's own "
            "error; raising the original %s instead of the release AttributeError",
            type(original).__name__,
        )
        raise original from None


# ── request-scoped L3 connection (per-request transaction; RLS session-GUC support) ─────────
# When a connection is bound here, every ``SqlL3Backend`` op in the async context runs ITS
# statements on that one connection -- so a hub request can open ONE transaction, set a
# txn-local session GUC (e.g. RLS's ``app.customer_ids``) on it, and have every collection
# read/write in the request share that transaction + GUC, instead of each statement taking a
# fresh autocommit connection from the pool. UNBOUND (the default) the behaviour is unchanged:
# ops go straight to the pool exactly as before. The binder owns the connection lifecycle +
# the transaction; ``SqlL3Backend`` only READS this var.
_request_l3_conn: ContextVar[Any | None] = ContextVar("threetears_request_l3_conn", default=None)


@asynccontextmanager
async def bound_request_connection(conn: Any) -> AsyncIterator[Any]:
    """Bind ``conn`` as the request-scoped L3 connection for the enclosing async context.

    Within the context, every :class:`SqlL3Backend` ``fetch`` / ``execute`` / ``transaction`` /
    ``acquire`` runs on ``conn`` (one transaction). The CALLER owns ``conn`` -- acquire it, open
    its transaction, set any session/txn GUCs, then enter this context; this helper only sets +
    resets the context variable. Nested binds restore the previous binding on exit.

    :param conn: an asyncpg-shaped connection already inside a transaction.
    :ptype conn: Any
    :return: async iterator yielding ``conn``.
    :rtype: AsyncIterator[Any]
    """
    token = _request_l3_conn.set(conn)
    try:
        yield conn
    finally:
        _request_l3_conn.reset(token)


class _BoundConnCM:
    """Async-CM yielding an already-bound request connection WITHOUT acquiring/releasing it.

    Returned by :meth:`SqlL3Backend.acquire` / :meth:`SqlL3Backend.transaction` when a request
    connection is bound, so explicit-connection / transactional code reuses the one request
    transaction instead of escaping to a fresh pooled connection (which would miss the request's
    session GUC and, under RLS ``FORCE``, see zero rows). Enter/exit are no-ops: the binder owns
    the connection's lifecycle and transaction.
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, *exc: object) -> None:
        return None


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


def _key_led_groups(
    schema: TableSchema, keys: list[tuple[Any, ...]]
) -> tuple[int, list[tuple[tuple[Any, ...], list[Any]]]]:
    """group whole keys for key-led deletes: one key column varies in an array, the rest are fixed.

    The varying column is the one leaving the fewest groups, the leading column on a tie, among the
    key columns with an array form. Each group is the fixed columns' values, in key order, and the
    varying column's values, in the order the keys came. Keys are told apart by their values' JSON
    form, so a fixed jsonb key column (a dict, which cannot key a dict) groups like any other.

    :param schema: the table's schema
    :ptype schema: TableSchema
    :param keys: whole keys, write-coerced, in the schema's key order
    :ptype keys: list[tuple[Any, ...]]
    :return: the varying column's index in the key, and the groups
    :rtype: tuple[int, list[tuple[tuple[Any, ...], list[Any]]]]
    :raises ValueError: when no key column has an array form
    """
    best: tuple[int, list[tuple[tuple[Any, ...], list[Any]]]] | None = None
    for index, name in enumerate(schema.pk_columns):
        if schema_sql.key_array_type(schema, name) is None:
            continue
        groups: dict[str, tuple[tuple[Any, ...], list[Any]]] = {}
        for key in keys:
            fixed = key[:index] + key[index + 1 :]
            token = json.dumps(list(fixed), default=schema_sql.json_default, sort_keys=True)
            groups.setdefault(token, (fixed, []))[1].append(key[index])
        if best is None or len(groups) < len(best[1]):
            best = (index, list(groups.values()))
    if best is None:
        raise ValueError(f"{schema.name}: no key column of {schema.pk_columns} has an array form to delete by")
    return best


def _every_key_column_has_an_array(schema: TableSchema) -> bool:
    """whether every key column can be named as an array, so whole keys can go one array a column.

    :param schema: the table's schema
    :ptype schema: TableSchema
    :return: True when each has an array form
    :rtype: bool
    """
    return all(schema_sql.key_array_type(schema, name) is not None for name in schema.pk_columns)


@dataclass
class _LedRead:
    """one key-led read: its statement, the transport's cap, and a tally of what it took.

    :param schema: the table's schema
    :ptype schema: TableSchema
    :param sql: the batch statement (:func:`~threetears.core.backends.schema_sql.build_led_by_select_sql`)
    :ptype sql: str
    :param columns: the columns read, the key's among them
    :ptype columns: tuple[str, ...]
    :param row_cap: the most rows the transport answers a statement, or None when it never cuts
    :ptype row_cap: int | None
    :param reader: what the statements run on: the caller's connection, or the backend
    :ptype reader: Any
    """

    schema: TableSchema
    sql: str
    columns: tuple[str, ...]
    row_cap: int | None
    reader: Any
    statements: int = 0
    splits: int = 0
    paged_values: int = 0

    async def batch(self, values: list[Any]) -> list[dict[str, Any]]:
        """the rows a batch of leading-key values holds: in halves while an answer reaches the cap, then by pages.

        :param values: the batch's leading-key values, write-coerced
        :ptype values: list[Any]
        :return: the rows
        :rtype: list[dict[str, Any]]
        """
        rows = [dict(row) for row in await self.reader.fetch(self.sql, values)]
        self.statements += 1
        if self.row_cap is None or len(rows) < self.row_cap:
            return rows
        if len(values) > 1:
            self.splits += 1
            half = len(values) // 2
            return await self.batch(values[:half]) + await self.batch(values[half:])
        return await self._paged(values[0], self.row_cap)

    async def _paged(self, value: Any, row_cap: int) -> list[dict[str, Any]]:
        """every row one leading-key value holds, paged by the rest of its key under the cap.

        :param value: the leading-key value, write-coerced
        :ptype value: Any
        :param row_cap: the transport's cap
        :ptype row_cap: int
        :return: the rows
        :rtype: list[dict[str, Any]]
        """
        lead, *rest = self.schema.pk_columns
        self.paged_values += 1
        select_from = f"SELECT {schema_sql.build_select_column_list(self.schema, self.columns)} FROM {self.schema.name}"
        read = await read_keyset_pages(
            self.reader, select_from, rest, where={lead: value}, page_size=row_cap - 1, quote=as_written
        )
        self.statements += read.statements
        return read.rows


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
    def _target(self) -> tuple[Any, bool]:
        """The handle to run a statement on, plus whether it forwards scoped-read kwargs.

        A request-scoped bound connection (a raw conn whose ``search_path`` + session GUCs the
        binder already set) takes precedence over the pool and is NOT scope-aware (``namespace``
        + ``customer_scope`` kwargs dropped -- routing is already pinned on the connection).
        UNBOUND, the pool is used with its declared scope-awareness -- the historical path,
        byte-for-byte unchanged.
        """
        bound = _request_l3_conn.get()
        if bound is not None:
            return bound, False
        return self._pool, self._scope_aware

    async def fetch(
        self, query: str, *params: Any, namespace: str | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Run a SELECT and return all rows as dicts (bound request conn, else the pool)."""
        target, scope_aware = self._target()
        with _unmask_asyncpg_release_race():
            if scope_aware:
                rows = await target.fetch(query, *params, namespace=namespace, **kwargs)
            else:
                rows = await target.fetch(query, *params)
        return [dict(r) for r in rows]

    async def fetchrow(
        self, query: str, *params: Any, namespace: str | None = None, **kwargs: Any
    ) -> dict[str, Any] | None:
        """Run a SELECT and return the first row dict, or ``None`` (bound request conn, else pool)."""
        target, scope_aware = self._target()
        with _unmask_asyncpg_release_race():
            if scope_aware:
                row = await target.fetchrow(query, *params, namespace=namespace, **kwargs)
            else:
                row = await target.fetchrow(query, *params)
        return dict(row) if row is not None else None

    async def fetchval(self, query: str, *params: Any, namespace: str | None = None, **kwargs: Any) -> Any:
        """Run a SELECT and return the first column of the first row (bound request conn, else pool)."""
        target, scope_aware = self._target()
        with _unmask_asyncpg_release_race():
            if scope_aware:
                value = await target.fetchval(query, *params, namespace=namespace, **kwargs)
            else:
                value = await target.fetchval(query, *params)
        return value

    async def execute(self, query: str, *params: Any, namespace: str | None = None, **kwargs: Any) -> str:
        """Run an INSERT/UPDATE/DELETE; return the command-tag (bound request conn, else pool)."""
        target, scope_aware = self._target()
        with _unmask_asyncpg_release_race():
            if scope_aware:
                result = await target.execute(query, *params, namespace=namespace, **kwargs)
            else:
                result = await target.execute(query, *params)
        return result if isinstance(result, str) else ""

    async def execute_batch(
        self, queries: list[dict[str, Any]], *, namespace: str | None = None, transaction: bool = True
    ) -> list[Any]:
        """Run ``{query, params}`` dicts atomically in one transaction when ``transaction``."""
        bound = _request_l3_conn.get()
        if bound is not None:
            # already inside the request transaction; run every statement on the bound conn so
            # the batch shares the request's transaction + session GUC (no nested transaction).
            on_conn: list[Any] = []
            for q in queries:
                on_conn.append(await bound.execute(q["query"], *q.get("params", [])))
            return on_conn
        out: list[Any] = []
        with _unmask_asyncpg_release_race():
            if not transaction:
                for q in queries:
                    out.append(await self._pool.execute(q["query"], *q.get("params", [])))
            else:
                async with self._pool.acquire() as conn, conn.transaction():
                    for q in queries:
                        out.append(await conn.execute(q["query"], *q.get("params", [])))
        return out

    def acquire(self) -> Any:
        """Return an ``acquire()`` async-CM: the bound request conn when set, else a pooled one."""
        bound = _request_l3_conn.get()
        if bound is not None:
            return _BoundConnCM(bound)
        return self._pool.acquire()

    def transaction(self, namespace: str | None = None) -> Any:
        """Return a transaction async-CM: the bound request conn when set (reuse the request

        transaction, no nesting), else a fresh pool-level transaction.
        """
        bound = _request_l3_conn.get()
        if bound is not None:
            return _BoundConnCM(bound)
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

        # "update": upsert the mutable (non-pk) columns.
        mutable = [c for c in cols if c not in set(pk)]
        if cas is not None and "date_updated" in row:
            # a fence value means the caller read the row as EXISTING, so this is an
            # update and never an insert. an upsert fenced in its DO UPDATE branch would
            # re-create a row deleted since that read, because a missing row reaches the
            # INSERT and the fence never runs.
            return await self._update_fenced(table, row, pk=pk, mutable=mutable, cas=cas, conn=conn)
        set_sql = (
            ", ".join(f"{_quote_ident(c)} = EXCLUDED.{_quote_ident(c)}" for c in mutable) or pk_sql + " = " + pk_sql
        )
        sql = f"{insert} ON CONFLICT ({pk_sql}) DO UPDATE SET {set_sql}"
        return parse_rowcount(await self._execute(sql, *params, conn=conn))

    async def upsert_many(
        self,
        table: str,
        rows: Sequence[Mapping[str, Any]],
        *,
        max_rows: int,
        max_bytes: int,
        conn: Any = None,
    ) -> int:
        """Insert-or-update many rows of a schema-registered table in multi-row statements.

        The column list is derived once, from the rows (:func:`schema_sql.insert_columns_for_data`,
        which leaves out a server-default column a row omits), and both each statement's SQL and
        every row's parameters are built from it, so the two cannot disagree. Rows that disagree on
        which server-default columns they supply are refused: one statement names one column list.
        Batches split by :func:`schema_sql.bulk_batches`. On ``conn`` a failing statement fails the
        caller's transaction, so every batch rolls back with it.

        :param table: the table; its schema must be registered (:meth:`register_schema`)
        :ptype table: str
        :param rows: the rows, each keyed by column
        :ptype rows: Sequence[Mapping[str, Any]]
        :param max_rows: the most rows one statement writes
        :ptype max_rows: int
        :param max_bytes: the most JSON bytes of parameters one statement carries
        :ptype max_bytes: int
        :param conn: the caller's transaction handle; ``None`` uses the pool
        :ptype conn: Any
        :return: rows written
        :rtype: int
        :raises ValueError: when no schema is registered for ``table``, or the rows disagree on
            which columns they supply
        """
        schema = self._schemas.get(table)
        if schema is None:
            raise ValueError(f"upsert_many needs the schema of {table!r} registered")
        written = 0
        if rows:
            columns = schema_sql.insert_columns_for_data(schema, dict(rows[0]))
            names = [c.name for c in columns]
            params: list[list[Any]] = []
            for row in rows:
                data = dict(row)
                if [c.name for c in schema_sql.insert_columns_for_data(schema, data)] != names:
                    raise ValueError(
                        f"{table}: rows of one bulk upsert must supply the same columns; "
                        f"a server-default column is supplied by some rows and not others"
                    )
                params.append(schema_sql.build_insert_params(schema, data))
            for batch in schema_sql.bulk_batches(
                params, max_rows=max_rows, max_bytes=max_bytes, max_params=schema_sql.MAX_STATEMENT_PARAMS
            ):
                sql = schema_sql.build_bulk_insert_sql(schema, rows=len(batch), columns=columns)
                await self._execute(sql, *(value for row in batch for value in row), conn=conn)
                written += len(batch)
        return written

    @property
    def rows_per_statement(self) -> int | None:
        """the most rows the wrapped transport answers one statement, or None when it never cuts.

        The transport says so as its own ``rows_per_statement`` (``NatsProxyL3Backend``: the rail's
        :data:`~threetears.core.backends.protocol.L3_RAIL_ROW_CAP`). One that says nothing, or
        says it in a form that is not a count (a mock's auto-attribute), is taken to be the rail:
        a needless split at worst, never a row lost to a cut nobody looked for.

        :return: the cap, or None
        :rtype: int | None
        """
        stated = getattr(self._pool, "rows_per_statement", L3_RAIL_ROW_CAP)
        if stated is None:
            return None
        if isinstance(stated, int) and not isinstance(stated, bool):
            return stated
        return L3_RAIL_ROW_CAP

    async def delete_many(
        self,
        table: str,
        keys: Sequence[Sequence[Any]],
        *,
        max_rows: int,
        conn: Any = None,
    ) -> int:
        """Delete the rows of a schema-registered table named by ``keys``, every statement led by the key.

        Two key-led forms, whichever needs fewer statements (the first on a tie):

        - the keys grouped by all but one key column, the varying one, chosen to make the fewest
          groups (the leading column on a tie), each group ``max_rows`` keys a statement as
          ``DELETE ... WHERE <varying> = ANY($1) AND <rest of the key> = ...``
          (:func:`~threetears.core.backends.schema_sql.build_key_led_delete_sql`): one statement for
          a generation retired, or one race's geographies;
        - whole keys, ``max_rows`` a statement, one array a key column
          (:func:`~threetears.core.backends.schema_sql.build_whole_key_delete_sql`): for keys that
          share no value, which grouped would take a statement each.

        Either way a statement names whole keys, so a hash-sharded table is looked up at each one
        and never read whole: at most ``ceil(len(keys) / max_rows)`` statements when every key
        column has an array form. On ``conn`` a failing statement fails the caller's transaction,
        so every batch rolls back with it.

        :param table: the table; its schema must be registered (:meth:`register_schema`)
        :ptype table: str
        :param keys: each row's key values, in the schema's key order
        :ptype keys: Sequence[Sequence[Any]]
        :param max_rows: the most keys one statement names; at least one
        :ptype max_rows: int
        :param conn: the caller's transaction handle; ``None`` uses the pool
        :ptype conn: Any
        :return: keys named
        :rtype: int
        :raises ValueError: when no schema is registered for ``table``, ``max_rows`` is under one,
            a key is not as wide as the table's key, or no key column has an array form
        """
        schema = self._schemas.get(table)
        if schema is None:
            raise ValueError(f"delete_many needs the schema of {table!r} registered")
        if max_rows < 1:
            raise ValueError(f"{table}: max_rows must be at least 1, got {max_rows}")
        key_columns = [schema.column(name) for name in schema.pk_columns]
        named: list[tuple[Any, ...]] = []
        for key in keys:
            if len(key) != len(key_columns):
                raise ValueError(
                    f"{table}: a key to delete has {len(key)} values where the table's key has {len(key_columns)}"
                )
            named.append(
                tuple(
                    schema_sql.normalize_write_value(column, value)
                    for column, value in zip(key_columns, key, strict=True)
                )
            )
        if not named:
            return 0
        varying, groups = _key_led_groups(schema, named)
        grouped = sum(-(-len(values) // max_rows) for _, values in groups)
        whole = -(-len(named) // max_rows)
        statements = 0
        if len(key_columns) > 1 and whole < grouped and _every_key_column_has_an_array(schema):
            sql = schema_sql.build_whole_key_delete_sql(schema)
            for start in range(0, len(named), max_rows):
                batch = named[start : start + max_rows]
                await self._execute(sql, *(list(column) for column in zip(*batch, strict=True)), conn=conn)
                statements += 1
        else:
            sql = schema_sql.build_key_led_delete_sql(schema, varying=schema.pk_columns[varying])
            for fixed, values in groups:
                for start in range(0, len(values), max_rows):
                    await self._execute(sql, values[start : start + max_rows], *fixed, conn=conn)
                    statements += 1
        _logger.debug(
            "key-led delete",
            extra={"extra_data": {"table": table, "keys": len(named), "statements": statements}},
        )
        return len(named)

    async def fetch_led_by(
        self,
        table: str,
        values: Sequence[Any],
        *,
        columns: Sequence[str],
        max_values: int,
        conn: Any = None,
    ) -> list[dict[str, Any]]:
        """Read ``columns`` of the rows whose leading key column is one of ``values``, every statement led by the key.

        ``max_values`` values a statement, ``WHERE <lead> = ANY($1)``. A transport that cuts answers
        (:attr:`rows_per_statement`) may cut one without saying so, so an answer that reaches the
        cap is read again in halves, and a single value that alone reaches it is paged by the rest
        of its key through :func:`~threetears.core.keyset.read_keyset_pages`
        (``WHERE <lead> = $1 AND (<rest>) > (...) ORDER BY <rest> LIMIT``), which is the key's own
        order inside one hash bucket. No statement reads or sorts the whole table. A read that had
        to split or page says so once at INFO, with its statement count: the sign that
        ``max_values`` is too many for the table.

        :param table: the table; its schema must be registered (:meth:`register_schema`)
        :ptype table: str
        :param values: the leading key column's values, each named once
        :ptype values: Sequence[Any]
        :param columns: the columns to read, the key's among them
        :ptype columns: Sequence[str]
        :param max_values: the most values one statement names; at least one
        :ptype max_values: int
        :param conn: the caller's connection; ``None`` uses the pool
        :ptype conn: Any
        :return: the rows, read-coerced
        :rtype: list[dict[str, Any]]
        :raises ValueError: when no schema is registered for ``table``, ``max_values`` is under
            one, or the transport's row cap is under two
        """
        schema = self._schemas.get(table)
        if schema is None:
            raise ValueError(f"fetch_led_by needs the schema of {table!r} registered")
        if max_values < 1:
            raise ValueError(f"{table}: max_values must be at least 1, got {max_values}")
        row_cap = self.rows_per_statement
        if row_cap is not None and row_cap < 2:
            raise ValueError(f"{table}: a row cap of {row_cap} leaves no room to tell a full answer from a cut one")
        lead = schema.column(schema.pk_columns[0])
        normalized = [schema_sql.normalize_write_value(lead, value) for value in values]
        sql = schema_sql.build_led_by_select_sql(schema, columns)
        read = _LedRead(
            schema=schema, sql=sql, columns=tuple(columns), row_cap=row_cap, reader=conn if conn is not None else self
        )
        rows: list[dict[str, Any]] = []
        for start in range(0, len(normalized), max_values):
            rows += await read.batch(normalized[start : start + max_values])
        if read.splits or read.paged_values:
            _logger.info(
                "key-led read split or paged to stay under the transport's row cap",
                extra={
                    "extra_data": {
                        "table": table,
                        "values": len(normalized),
                        "rows": len(rows),
                        "statements": read.statements,
                        "splits": read.splits,
                        "paged_values": read.paged_values,
                        "row_cap": row_cap,
                    }
                },
            )
        return [schema_sql.coerce_row(schema, row) for row in rows]

    async def _update_fenced(
        self,
        table: str,
        row: Mapping[str, Any],
        *,
        pk: Sequence[str],
        mutable: Sequence[str],
        cas: datetime,
        conn: Any,
    ) -> int:
        """Update one existing row only while its ``date_updated`` still equals ``cas``.

        The generic (schema-less) half of the update-only rule: 0 rows when the row
        changed OR no longer exists, never an insert.

        :param table: table name
        :ptype table: str
        :param row: the row, pk columns included
        :ptype row: Mapping[str, Any]
        :param pk: the pk column names
        :ptype pk: Sequence[str]
        :param mutable: the non-pk columns to write
        :ptype mutable: Sequence[str]
        :param cas: the ``date_updated`` value the caller read
        :ptype cas: datetime
        :param conn: optional caller-supplied connection
        :ptype conn: Any
        :return: rows affected, 0 on a fence miss or a vanished row
        :rtype: int
        """
        params: list[Any] = [row[c] for c in mutable]
        set_sql = ", ".join(f"{_quote_ident(c)} = ${i + 1}" for i, c in enumerate(mutable))
        where_parts: list[str] = []
        for c in pk:
            params.append(row[c])
            where_parts.append(f"{_quote_ident(c)} = ${len(params)}")
        params.append(cas)
        where_parts.append(f"{_quote_ident('date_updated')} = ${len(params)}")
        sql = f"UPDATE {_quote_ident(table)} SET {set_sql} WHERE {' AND '.join(where_parts)}"
        return parse_rowcount(await self._execute(sql, *params, conn=conn))

    async def _upsert_schema(
        self,
        schema: TableSchema,
        data: dict[str, Any],
        *,
        cas: datetime | None,
        conn: Any,
    ) -> int:
        """Schema-aware upsert: byte-identical to the collection's old ``save_to_store``.

        Three generated shapes, selected by the schema and by whether the caller
        read a version (a non-``None`` ``cas``):

        * ``cas_column`` set, ``on_conflict='update'`` and a non-``None`` ``cas``
          -- the fenced ``UPDATE ... WHERE pk AND <cas> = $N``, on EVERY such
          schema, ``cas_null_safe`` included. a fence value means the caller read
          the row as existing, so the write is update-only: a row deleted since
          that read (a respondent erasure, a relocation retire) affects 0 rows and
          the caller gets ``ConcurrentModificationError``. the NULL-safe upsert
          used to serve this case too, and because a missing row reaches its
          INSERT with the fence never evaluated, a save racing a delete
          re-created the row the delete had just removed.
        * ``cas_null_safe=True`` and ``cas=None`` -- ONE statement:
          ``INSERT ... ON CONFLICT (pk) DO UPDATE SET ... WHERE t.<cas> IS NOT
          DISTINCT FROM $N``. ``cas=None`` is FENCE-ELIGIBLE here, which is the
          whole point: a derived (non-random) primary key means two concurrent
          FIRST writers compute the same id, and NULL-safe equality is what lets
          the loser affect 0 rows instead of silently overwriting the winner.
        * everything else -- the unfenced ``INSERT ... ON CONFLICT`` from
          ``build_insert_sql``. Unchanged, and still what ``cas=None`` selects on
          every schema that has not opted in.

        ``cas_null_safe`` is read through ``getattr`` because this module reads the
        schema duck-typed (see the ``schema_sql`` module docstring); a non-
        ``TableSchema`` descriptor without the attribute keeps the old behaviour.
        """
        cas_declared = schema.cas_column is not None and schema.on_conflict == "update"
        if cas_declared and cas is not None:
            sql = schema_sql.build_cas_update_sql(schema, data)
            params = schema_sql.build_cas_params(schema, data, cas)
        elif cas_declared and getattr(schema, "cas_null_safe", False):
            sql = schema_sql.build_cas_upsert_sql(schema, data)
            params = schema_sql.build_cas_upsert_params(schema, data, cas)
        else:
            sql = schema_sql.build_insert_sql(schema, data)
            params = schema_sql.build_insert_params(schema, data)
        return parse_rowcount(await self._execute(sql, *params, conn=conn))

    async def upsert_ordered(self, table: str, row: Mapping[str, Any], *, conn: Any = None) -> int:
        """Insert ``row``, or update the stored row only when its compare-and-swap order is older.

        :class:`~threetears.core.backends.protocol.OrderedDurableStore`'s one operation. The SQL
        is generated from the registered schema by
        :func:`~threetears.core.backends.schema_sql.build_ordered_upsert_sql`, so the table must
        have one: without it neither the key nor the mutable columns are known.

        :param table: the table, whose schema is registered
        :ptype table: str
        :param row: the row, ``l2_epoch`` / ``l2_revision`` included
        :ptype row: Mapping[str, Any]
        :param conn: optional caller-supplied connection (a transaction handle) the write binds to
        :ptype conn: Any
        :return: ``1`` when written, ``0`` when the stored order is newer or equal
        :rtype: int
        :raises ValueError: when no schema is registered for ``table``
        """
        schema = self._schemas.get(table)
        if schema is None:
            raise ValueError(f"upsert_ordered on {table!r} needs its schema registered; none is")
        data = dict(row)
        sql = schema_sql.build_ordered_upsert_sql(schema, data)
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
