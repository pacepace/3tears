"""NATS-based L3 backend proxy for agent pods.

drop-in replacement for direct asyncpg pool. routes all L3 queries
through Hub's L3 Broker via NATS request-reply. collections use
this transparently without knowing queries are proxied.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid7

from threetears.core.exceptions import DataLayerUnavailableError
from threetears.core.namespaces import PLURAL_PREFIX_AGENT, build_namespace_name
from threetears.observe import get_logger

__all__ = ["NatsProxyL3Backend"]

_logger = get_logger(__name__)


def _serialize_param(value: Any) -> Any:
    """serialize parameter value for NATS transport.

    converts UUID, datetime, Decimal to string representations.
    other types passed through unchanged.

    :param value: parameter value to serialize
    :ptype value: Any
    :return: serialized value suitable for JSON transport
    :rtype: Any
    """
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bytes):
        return "\\x" + value.hex()
    return value


def _deserialize_row(row: dict[str, Any]) -> dict[str, Any]:
    """deserialize row values returned from L3 broker proxy.

    three conversions run per row:

    - hex-encoded byte strings (``"\\xHEX"`` prefix) are restored to
      :class:`bytes`; the broker's :func:`_serialize_row` produces
      this shape for ``BYTEA`` columns
    - ``date_*`` columns arriving as ISO-8601 strings are rehydrated
      to timezone-aware :class:`datetime` instances. asyncpg returns
      ``TIMESTAMP`` values as native datetimes on direct connections,
      but when the broker round-trips them through JSON they collapse
      to strings. agent-side entity classes call ``.isoformat()`` on
      ``date_*`` fields expecting datetime objects; the normalization
      keeps their code path unchanged regardless of the backing pool
    - every other value passes through unchanged

    the date rehydration defers on parse failure (leaves value as a
    string) so columns happening to start with ``date_`` but holding
    non-ISO content do not corrupt on the wire.

    :param row: row dictionary from broker response
    :ptype row: dict[str, Any]
    :return: row with bytes + datetime values restored
    :rtype: dict[str, Any]
    """
    result: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, str) and value.startswith("\\x"):
            result[key] = bytes.fromhex(value[2:])
        elif isinstance(value, str) and key.startswith("date_") and value:
            result[key] = _parse_iso_datetime(value)
        else:
            result[key] = value
    return result


def _parse_iso_datetime(raw: str) -> Any:
    """parse an ISO-8601 string to a timezone-aware datetime.

    :func:`datetime.fromisoformat` accepts both timezone-aware
    (``"...+00:00"``, ``"...Z"``) and naive (``"..."``) forms. naive
    inputs are attached to UTC defensively so callers that expect
    ``.tzinfo is not None`` do not blow up; aware inputs pass through
    with their original offset. on parse failure the raw string is
    returned so the caller sees the payload they shipped rather than
    a silent :class:`None`.

    :param raw: candidate ISO-8601 string
    :ptype raw: str
    :return: parsed datetime or the original string on failure
    :rtype: Any
    """
    candidate = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return raw
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _format_execute_tag(operation: str, row_count: Any) -> str:
    """build an asyncpg-shape status tag for a given operation + row count.

    asyncpg conventions:

    - ``INSERT``: ``"INSERT {oid} {count}"`` with ``oid`` always 0 on
      modern postgres; we emit ``"INSERT 0 {count}"`` so splits land
      the count at index -1
    - ``UPDATE`` / ``DELETE`` / ``UPSERT``: ``"{VERB} {count}"``
    - ``SELECT`` (rare for execute): ``"SELECT {count}"``

    ``row_count`` may be ``None`` when the broker did not populate it
    (DDL, transactional no-ops). that's treated as zero so the tag
    remains parseable by every caller.

    :param operation: detected op verb (select/insert/update/delete/upsert)
    :ptype operation: str
    :param row_count: affected row count from the broker response
    :ptype row_count: Any
    :return: asyncpg-compatible status tag
    :rtype: str
    """
    count = 0 if row_count is None else int(row_count)
    verb = operation.upper()
    if verb == "INSERT":
        return f"INSERT 0 {count}"
    if verb == "UPSERT":
        # postgres never emits "UPSERT"; asyncpg reports INSERT for
        # ON CONFLICT DO UPDATE. preserve that convention.
        return f"INSERT 0 {count}"
    return f"{verb} {count}"


def _detect_operation(query: str) -> str:
    """detect SQL operation type from query string.

    :param query: SQL query string
    :ptype query: str
    :return: operation type (select, insert, update, delete, upsert)
    :rtype: str
    """
    stripped = query.strip().upper()
    if stripped.startswith("SELECT"):
        return "select"
    if stripped.startswith("INSERT"):
        if "ON CONFLICT" in stripped:
            return "upsert"
        return "insert"
    if stripped.startswith("UPDATE"):
        return "update"
    if stripped.startswith("DELETE"):
        return "delete"
    return "select"


class NatsProxyL3Backend:
    """L3 backend that routes queries through NATS-based broker.

    drop-in replacement for direct asyncpg pool. collections use
    this without knowing queries are proxied through Hub.

    :ivar ns: NATS subject namespace prefix; collaborator proxies
        (``_ProxyConnection``, ``_ProxyTransaction``) read this to
        compose tx subject names
    :ivar agent_id: agent UUID string used for ACL validation on the
        broker side; collaborator proxies include it in every tx
        envelope
    :ivar default_namespace: default logical namespace applied to
        queries that did not supply one
    :ivar timeout_ms: query-side timeout threaded into outgoing
        envelopes and used to compute the NATS-level response timeout

    :param nats_client: connected NATS client
    :ptype nats_client: Any
    :param namespace_prefix: NATS subject namespace prefix
    :ptype namespace_prefix: str
    :param agent_id: agent UUID string for ACL validation
    :ptype agent_id: str
    :param default_namespace: default namespace for queries
    :ptype default_namespace: str | None
    :param timeout_ms: default query timeout in milliseconds
    :ptype timeout_ms: int
    """

    def __init__(
        self,
        nats_client: Any,
        namespace_prefix: str,
        agent_id: str,
        default_namespace: str | None = None,
        timeout_ms: int | None = None,
    ) -> None:
        """initialize NatsProxyL3Backend.

        :param nats_client: connected NATS client
        :ptype nats_client: Any
        :param namespace_prefix: NATS subject namespace prefix
        :ptype namespace_prefix: str
        :param agent_id: agent UUID string for ACL validation
        :ptype agent_id: str
        :param default_namespace: default namespace for queries
        :ptype default_namespace: str | None
        :param timeout_ms: default query timeout in milliseconds.
            sourced from THREETEARS_NATS_PROXY_TIMEOUT_MS env var if not provided (default 5000).
        :ptype timeout_ms: int | None
        """
        self._nc = nats_client
        self.ns = namespace_prefix
        self.agent_id = agent_id
        self.default_namespace = default_namespace or build_namespace_name(
            PLURAL_PREFIX_AGENT, agent_id
        )
        if timeout_ms is not None:
            self.timeout_ms = timeout_ms
        else:
            import os

            raw = os.environ.get("THREETEARS_NATS_PROXY_TIMEOUT_MS")
            self.timeout_ms = int(raw) if raw is not None else 5000

    async def fetch(
        self,
        query: str,
        *params: Any,
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        """execute SELECT query and return all rows.

        :param query: parameterized SQL query
        :ptype query: str
        :param params: query parameter values
        :ptype params: Any
        :param namespace: target namespace (default: agent's private namespace)
        :ptype namespace: str | None
        :return: list of row dictionaries
        :rtype: list[dict[str, Any]]
        :raises DataLayerUnavailableError: if broker returns error
        """
        response = await self._send_query(query, list(params), "select", namespace)
        rows: list[dict[str, Any]] = response.get("rows", [])
        result = [_deserialize_row(row) for row in rows]
        return result

    async def fetchrow(
        self,
        query: str,
        *params: Any,
        namespace: str | None = None,
    ) -> dict[str, Any] | None:
        """execute SELECT query and return first row.

        :param query: parameterized SQL query
        :ptype query: str
        :param params: query parameter values
        :ptype params: Any
        :param namespace: target namespace
        :ptype namespace: str | None
        :return: first row dictionary or None
        :rtype: dict[str, Any] | None
        :raises DataLayerUnavailableError: if broker returns error
        """
        rows = await self.fetch(query, *params, namespace=namespace)
        result: dict[str, Any] | None = rows[0] if rows else None
        return result

    async def execute(
        self,
        query: str,
        *params: Any,
        namespace: str | None = None,
    ) -> str:
        """execute INSERT/UPDATE/DELETE and return an asyncpg-shape tag.

        asyncpg's :meth:`Connection.execute` returns a status string
        of the shape ``"INSERT 0 1"`` / ``"UPDATE 3"`` /
        ``"DELETE 2"``. collection implementations in agent-workspace,
        agent-tools, and agent-memory all parse the final integer via
        ``int(result.split()[-1])``. the previous proxy implementation
        returned a bare ``int``, which silently crashed every one of
        those callers with ``'int' has no attribute 'split'`` the
        moment a real workspace_create / context-item write landed in
        production. the fix aligns the return shape with asyncpg so
        collections behave identically whether the pool is a real
        asyncpg pool or this proxy.

        :param query: parameterized SQL query
        :ptype query: str
        :param params: query parameter values
        :ptype params: Any
        :param namespace: target namespace
        :ptype namespace: str | None
        :return: asyncpg-style status tag (e.g. ``"UPDATE 3"``)
        :rtype: str
        :raises DataLayerUnavailableError: if broker returns error
        """
        operation = _detect_operation(query)
        response = await self._send_query(query, list(params), operation, namespace)
        return _format_execute_tag(operation, response.get("row_count"))

    async def execute_batch(
        self,
        queries: list[dict[str, Any]],
        *,
        namespace: str | None = None,
        transaction: bool = True,
    ) -> list[Any]:
        """execute batch of queries with optional transaction.

        :param queries: list of query dicts with operation, query, params keys
        :ptype queries: list[dict[str, Any]]
        :param namespace: target namespace
        :ptype namespace: str | None
        :param transaction: whether to execute atomically
        :ptype transaction: bool
        :return: list of results per query
        :rtype: list[Any]
        :raises DataLayerUnavailableError: if broker returns error
        """
        ns = namespace or self.default_namespace
        payload = {
            "correlation_id": str(uuid7()),
            "agent_id": self.agent_id,
            "namespace": ns,
            "queries": [
                {
                    "operation": q.get("operation", _detect_operation(q["query"])),
                    "query": q["query"],
                    "params": [_serialize_param(p) for p in q.get("params", [])],
                }
                for q in queries
            ],
            "transaction": transaction,
        }
        subject = f"{self.ns}.l3.batch"
        response = await self.nats_request(subject, payload)

        if not response.get("success", False):
            raise DataLayerUnavailableError(
                f"batch query failed: {response.get('error_code', 'UNKNOWN')}: "
                f"{response.get('error_message', 'no details')}"
            )

        results: list[Any] = response.get("results", [])
        return results

    async def _send_query(
        self,
        query: str,
        params: list[Any],
        operation: str,
        namespace: str | None,
    ) -> dict[str, Any]:
        """build and send single query request to broker.

        :param query: SQL query string
        :ptype query: str
        :param params: query parameters
        :ptype params: list[Any]
        :param operation: SQL operation type
        :ptype operation: str
        :param namespace: target namespace
        :ptype namespace: str | None
        :return: parsed response dict
        :rtype: dict[str, Any]
        :raises DataLayerUnavailableError: if broker returns error
        """
        ns = namespace or self.default_namespace
        payload = {
            "correlation_id": str(uuid7()),
            "agent_id": self.agent_id,
            "namespace": ns,
            "operation": operation,
            "query": query,
            "params": [_serialize_param(p) for p in params],
            "timeout_ms": self.timeout_ms,
        }
        subject = f"{self.ns}.l3.query"
        response = await self.nats_request(subject, payload)

        if not response.get("success", False):
            raise DataLayerUnavailableError(
                f"L3 query failed: {response.get('error_code', 'UNKNOWN')}: "
                f"{response.get('error_message', 'no details')}"
            )

        return response

    async def nats_request(
        self,
        subject: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """send raw NATS request and parse JSON response.

        public entry point for the backend's collaborator proxy objects
        (:class:`_ProxyConnection`, :class:`_ProxyTransaction`) so they
        do not have to reach for the underlying ``_nc`` client. owns
        envelope serialization, timeout computation, and error wrapping.

        :param subject: NATS subject on which to publish the request
        :ptype subject: str
        :param payload: request payload dict, will be JSON-encoded with
            ``default=str`` so UUID, datetime, Decimal values serialize
        :ptype payload: dict[str, Any]
        :return: parsed JSON response dict from broker
        :rtype: dict[str, Any]
        :raises DataLayerUnavailableError: if NATS request times out,
            the broker returns malformed JSON, or the client is closed
        """
        payload_bytes = json.dumps(payload, default=str).encode("utf-8")
        nats_timeout = (self.timeout_ms / 1000) + 2

        try:
            reply = await self._nc.request(subject, payload_bytes, timeout=nats_timeout)
            result: dict[str, Any] = json.loads(reply.data)
        except Exception as exc:
            raise DataLayerUnavailableError(f"NATS request failed: subject={subject}: {exc}") from exc

        return result

    def acquire(self) -> "_ProxyAcquireCM":
        """return an asyncpg-Pool-style ``acquire()`` context manager.

        emitted to make the proxy transparently usable by code that
        expects an asyncpg :class:`Pool`. the returned context manager
        yields a :class:`_ProxyConnection` on ``__aenter__`` and
        rolls-back-and-releases any open transaction on
        ``__aexit__``. outside a transaction the connection is a thin
        façade over the backend's existing ``execute`` / ``fetch`` /
        ``fetchrow`` round-trips (each call one request/response);
        inside a transaction every op is routed through the broker-
        minted ``tx_id`` so reads see prior writes within the same
        transaction.

        typical usage mirrors asyncpg::

            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(SQL1, ...)
                    row = await conn.fetchrow(SQL2, ...)
                    await conn.execute(SQL3, ...)

        :return: async context manager yielding a proxy connection
        :rtype: _ProxyAcquireCM
        """
        return _ProxyAcquireCM(self)

    def transaction(self, namespace: str | None = None) -> "_ProxyPoolTransactionCM":
        """return a pool-level transaction context manager (WS-ACL-06).

        combines :meth:`acquire` + ``conn.transaction(namespace=...)``
        into a single ``async with`` so workspace tools can write::

            async with pool.transaction(namespace=workspace.namespace_name):
                await pool.execute(...)

        without the two-level ``acquire()`` + ``conn.transaction()``
        dance. the yielded object is a :class:`_ProxyConnection` pinned
        to the broker-minted ``tx_id``; inside the ``async with`` every
        ``execute`` / ``fetch`` / ``fetchrow`` call the caller issues
        against THIS returned connection routes through the session.

        ``namespace`` is declared once at begin and bound to the session
        on the broker; statement-level ``namespace=`` overrides are
        rejected (:meth:`_ProxyConnection.execute` etc. raise
        ``ValueError`` when called with a non-None ``namespace`` inside
        a tx). this is intentional: the namespace is a property of the
        transaction, not of any single statement, and silently ignoring
        a mid-tx override would let a bug in a tool quietly land writes
        on the wrong schema.

        ``namespace=None`` (default) preserves today's behavior: the
        session binds to the caller's default agent namespace. explicit
        ``namespace=<str>`` is the WS-ACL-06 cross-agent path.

        :param namespace: logical namespace name the session binds to;
            ``None`` means use the caller's default agent namespace
        :ptype namespace: str | None
        :return: async context manager yielding the pinned proxy connection
        :rtype: _ProxyPoolTransactionCM
        """
        return _ProxyPoolTransactionCM(self, namespace)


# ---------------------------------------------------------------------------
# asyncpg-like acquire / transaction shim
# ---------------------------------------------------------------------------
#
# the workspace-tool family was written against :class:`asyncpg.Pool`
# semantics (``async with pool.acquire() as conn: async with
# conn.transaction():``) with read-modify-write patterns that cannot
# be expressed through ``execute_batch``. we don't want to move every
# tool's transactional logic to the hub side, so the proxy keeps
# asyncpg's shape and routes the tx lifecycle through broker-side
# stateful sessions on ``l3.tx.*`` subjects. the broker pins a real
# asyncpg connection for the duration of the tx window, so the tools
# keep working unchanged while all traffic still flows through NATS.


class _ProxyAcquireCM:
    """async context manager returned by :meth:`NatsProxyL3Backend.acquire`.

    symmetric with :class:`asyncpg.pool.PoolAcquireContext`. yields a
    :class:`_ProxyConnection` on enter and, on exit, rolls back any
    transaction the caller left open (defensive: well-behaved callers
    exit the transaction explicitly, but an exception inside the body
    or a forgotten ``commit`` would otherwise leak a pool slot on the
    broker).

    :param backend: backend instance to route NATS traffic through
    :ptype backend: NatsProxyL3Backend
    """

    def __init__(self, backend: "NatsProxyL3Backend") -> None:
        """capture the backend for lazy connection construction.

        :param backend: backend instance
        :ptype backend: NatsProxyL3Backend
        :return: None
        :rtype: None
        """
        self._backend = backend
        self._connection: _ProxyConnection | None = None

    async def __aenter__(self) -> "_ProxyConnection":
        """return a fresh proxy connection.

        :return: proxy connection
        :rtype: _ProxyConnection
        """
        self._connection = _ProxyConnection(self._backend)
        return self._connection

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        """force-rollback any dangling transaction left by the caller.

        exists purely as a safety net: well-behaved callers exit their
        ``async with conn.transaction():`` before this hook fires, at
        which point :attr:`_ProxyConnection.tx_id` is None and the
        method is a no-op. if the caller forgot to close the tx the
        broker's sweeper would eventually roll it back anyway, but
        calling rollback here returns the pool slot immediately.

        :param exc_type: exception type (unused)
        :ptype exc_type: Any
        :param exc_val: exception value (unused)
        :ptype exc_val: Any
        :param exc_tb: exception traceback (unused)
        :ptype exc_tb: Any
        :return: None
        :rtype: None
        """
        del exc_type, exc_val, exc_tb
        if self._connection is not None and self._connection.tx_id is not None:
            try:
                await self._connection.abort_tx()
            except Exception as abort_exc:
                _logger.warning(
                    "proxy connection close: dangling tx rollback failed: %s",
                    abort_exc,
                )


class _ProxyConnection:
    """asyncpg-Connection-shaped proxy over the NATS L3 backend.

    outside a transaction, every call round-trips to ``l3.query``
    exactly as the backend's top-level ``execute`` / ``fetch`` /
    ``fetchrow`` do today. inside a transaction (caller has entered
    :meth:`transaction`), ops route through the session's ``tx_id``
    via the ``l3.tx.*`` subjects so the broker can honor
    read-modify-write semantics on a single pinned asyncpg connection.

    :param backend: backend instance for NATS routing
    :ptype backend: NatsProxyL3Backend
    """

    def __init__(self, backend: "NatsProxyL3Backend") -> None:
        """capture the backend and initialize tx state.

        :param backend: backend instance
        :ptype backend: NatsProxyL3Backend
        :return: None
        :rtype: None
        """
        self._backend = backend
        # set to the broker-minted tx_id while a transaction is open;
        # reset to None on commit or rollback.
        self.tx_id: UUID | None = None

    def transaction(
        self,
        namespace: str | None = None,
    ) -> "_ProxyTransaction":
        """return a ``conn.transaction()``-shaped context manager.

        mirrors :meth:`asyncpg.Connection.transaction`. nested calls
        are rejected at ``__aenter__`` so callers who unintentionally
        open two transactions on one connection fail loudly instead
        of silently sharing tx state.

        WS-ACL-06: ``namespace=<str>`` threads a logical namespace
        through to the broker's ``tx.begin`` payload. the broker
        resolves the namespace once and binds ``SET search_path`` for
        the session. every subsequent ``tx.execute`` / ``tx.fetch`` /
        ``tx.fetchrow`` on this connection implicitly honors the
        session's namespace -- per-statement ``namespace=`` overrides
        inside an open tx are REJECTED (see
        :meth:`_ProxyConnection.execute` and siblings). ``namespace=None``
        preserves today's behavior: the session binds to the caller's
        default agent namespace.

        :param namespace: logical namespace name the session binds to;
            ``None`` means use the caller's default agent namespace
        :ptype namespace: str | None
        :return: transaction context manager
        :rtype: _ProxyTransaction
        """
        return _ProxyTransaction(self, self._backend, namespace)

    async def execute(
        self,
        query: str,
        *params: Any,
        namespace: str | None = None,
    ) -> str:
        """execute DML; route through tx_id when inside a transaction.

        matches :meth:`asyncpg.Connection.execute` and
        :meth:`NatsProxyL3Backend.execute` by returning the asyncpg-
        shape status tag (``"UPDATE 3"`` / ``"INSERT 0 1"`` etc.).
        collection implementations parse this via
        ``int(result.split()[-1])`` and would crash on a bare int
        return.

        :param query: parameterized SQL
        :ptype query: str
        :param params: query parameters
        :ptype params: Any
        :param namespace: override namespace for outside-tx calls;
            inside an open transaction a non-None value is REJECTED
            (the session's namespace is bound once at tx.begin)
        :ptype namespace: str | None
        :return: asyncpg-shape status tag
        :rtype: str
        :raises DataLayerUnavailableError: on broker error
        :raises ValueError: when called with ``namespace`` inside an
            open transaction
        """
        if self.tx_id is None:
            return await self._backend.execute(
                query,
                *params,
                namespace=namespace,
            )
        if namespace is not None:
            raise ValueError(
                "per-statement namespace= is not allowed inside an open "
                "transaction; the namespace is declared at "
                "conn.transaction(namespace=...) and bound on the broker "
                "for the lifetime of the session",
            )
        payload: dict[str, Any] = {
            "correlation_id": str(uuid7()),
            "agent_id": self._backend.agent_id,
            "tx_id": str(self.tx_id),
            "query": query,
            "params": [_serialize_param(p) for p in params],
        }
        subject = f"{self._backend.ns}.l3.tx.execute"
        response = await self._backend.nats_request(subject, payload)
        if not response.get("success", False):
            raise DataLayerUnavailableError(
                f"tx.execute failed: {response.get('error_code', 'UNKNOWN')}: "
                f"{response.get('error_message', 'no details')}",
            )
        return _format_execute_tag(
            _detect_operation(query),
            response.get("row_count"),
        )

    async def fetchrow(
        self,
        query: str,
        *params: Any,
        namespace: str | None = None,
    ) -> dict[str, Any] | None:
        """single-row SELECT; routed through tx_id when inside a tx.

        :param query: parameterized SELECT
        :ptype query: str
        :param params: query parameters
        :ptype params: Any
        :param namespace: override namespace for outside-tx calls;
            inside an open transaction a non-None value is REJECTED
            (the session's namespace is bound once at tx.begin)
        :ptype namespace: str | None
        :return: first row dict or None
        :rtype: dict[str, Any] | None
        :raises ValueError: when called with ``namespace`` inside an
            open transaction
        """
        if self.tx_id is None:
            outside_row = await self._backend.fetchrow(
                query,
                *params,
                namespace=namespace,
            )
            return outside_row
        if namespace is not None:
            raise ValueError(
                "per-statement namespace= is not allowed inside an open "
                "transaction; the namespace is declared at "
                "conn.transaction(namespace=...) and bound on the broker "
                "for the lifetime of the session",
            )
        payload: dict[str, Any] = {
            "correlation_id": str(uuid7()),
            "agent_id": self._backend.agent_id,
            "tx_id": str(self.tx_id),
            "query": query,
            "params": [_serialize_param(p) for p in params],
        }
        subject = f"{self._backend.ns}.l3.tx.fetchrow"
        response = await self._backend.nats_request(subject, payload)
        if not response.get("success", False):
            raise DataLayerUnavailableError(
                f"tx.fetchrow failed: {response.get('error_code', 'UNKNOWN')}: "
                f"{response.get('error_message', 'no details')}",
            )
        row = response.get("row")
        if row is None:
            return None
        return _deserialize_row(row)

    async def fetch(
        self,
        query: str,
        *params: Any,
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        """multi-row SELECT; routed through tx_id when inside a tx.

        :param query: parameterized SELECT
        :ptype query: str
        :param params: query parameters
        :ptype params: Any
        :param namespace: override namespace for outside-tx calls;
            inside an open transaction a non-None value is REJECTED
            (the session's namespace is bound once at tx.begin)
        :ptype namespace: str | None
        :return: row dicts
        :rtype: list[dict[str, Any]]
        :raises ValueError: when called with ``namespace`` inside an
            open transaction
        """
        if self.tx_id is None:
            outside_rows = await self._backend.fetch(
                query,
                *params,
                namespace=namespace,
            )
            return outside_rows
        if namespace is not None:
            raise ValueError(
                "per-statement namespace= is not allowed inside an open "
                "transaction; the namespace is declared at "
                "conn.transaction(namespace=...) and bound on the broker "
                "for the lifetime of the session",
            )
        payload: dict[str, Any] = {
            "correlation_id": str(uuid7()),
            "agent_id": self._backend.agent_id,
            "tx_id": str(self.tx_id),
            "query": query,
            "params": [_serialize_param(p) for p in params],
        }
        subject = f"{self._backend.ns}.l3.tx.fetch"
        response = await self._backend.nats_request(subject, payload)
        if not response.get("success", False):
            raise DataLayerUnavailableError(
                f"tx.fetch failed: {response.get('error_code', 'UNKNOWN')}: "
                f"{response.get('error_message', 'no details')}",
            )
        raw_rows: list[dict[str, Any]] = response.get("rows", [])
        return [_deserialize_row(r) for r in raw_rows]

    async def abort_tx(self) -> None:
        """send ``tx.rollback`` and clear the local tx_id.

        invoked by :class:`_ProxyTransaction.__aexit__` on an exception
        path and by :class:`_ProxyAcquireCM.__aexit__` as a safety net
        when a caller forgets to end the transaction. idempotent on
        double-call so the two exit hooks can both fire without
        hitting the broker twice. name is public because the safety-net
        exit hook lives in a sibling class and must call this method
        without crossing the private-boundary contract.

        :return: None
        :rtype: None
        """
        tx_id = self.tx_id
        if tx_id is None:
            return
        payload: dict[str, Any] = {
            "correlation_id": str(uuid7()),
            "agent_id": self._backend.agent_id,
            "tx_id": str(tx_id),
        }
        subject = f"{self._backend.ns}.l3.tx.rollback"
        try:
            await self._backend.nats_request(subject, payload)
        finally:
            # clear the tx_id even on network failure so subsequent
            # ops on this connection route through the outside-tx
            # path instead of trying to speak to a dead session.
            self.tx_id = None


class _ProxyTransaction:
    """asyncpg-Transaction-shaped context manager.

    on ``__aenter__`` requests ``l3.tx.begin`` and pins the returned
    ``tx_id`` on the connection. on clean ``__aexit__`` sends
    ``l3.tx.commit``; on exception sends ``l3.tx.rollback``. nested
    transactions raise because the broker does not implement
    savepoints yet and tools should not pretend they do.

    :param connection: owning proxy connection
    :ptype connection: _ProxyConnection
    :param backend: backend instance for NATS routing
    :ptype backend: NatsProxyL3Backend
    :param namespace: optional logical namespace name; when supplied
        the tx.begin payload carries this value verbatim so the broker
        binds ``SET search_path`` to the namespace's (possibly remote)
        schema, enabling cross-agent workspace operations per WS-ACL-06.
        when ``None`` (today's default) the backend's default namespace
        is used, preserving pre-WS-ACL-06 behavior for non-workspace
        callers.
    :ptype namespace: str | None
    """

    def __init__(
        self,
        connection: "_ProxyConnection",
        backend: "NatsProxyL3Backend",
        namespace: str | None = None,
    ) -> None:
        """capture the owning connection and backend.

        :param connection: owning connection
        :ptype connection: _ProxyConnection
        :param backend: backend instance
        :ptype backend: NatsProxyL3Backend
        :param namespace: logical namespace the session binds to, or
            ``None`` for the backend default
        :ptype namespace: str | None
        :return: None
        :rtype: None
        """
        self._connection = connection
        self._backend = backend
        self._namespace = namespace

    async def __aenter__(self) -> "_ProxyTransaction":
        """send ``l3.tx.begin`` and pin the returned tx_id on the connection.

        :return: self
        :rtype: _ProxyTransaction
        :raises DataLayerUnavailableError: on broker error
        :raises RuntimeError: when nested on an already-open tx
        """
        if self._connection.tx_id is not None:
            raise RuntimeError(
                "NatsProxyL3Backend does not support nested transactions; "
                "commit or rollback the outer tx before starting another",
            )
        # WS-ACL-06: when the caller specified namespace= on
        # .transaction(), ship that value so the broker binds the
        # session's search_path to the namespace's schema (possibly a
        # remote agent's schema under a grant). absent an explicit
        # value we preserve today's behavior and bind to the caller's
        # default agent namespace.
        effective_namespace = (
            self._namespace
            if self._namespace is not None
            else self._backend.default_namespace
        )
        payload: dict[str, Any] = {
            "correlation_id": str(uuid7()),
            "agent_id": self._backend.agent_id,
            "namespace": effective_namespace,
            "statement_timeout_ms": self._backend.timeout_ms,
        }
        subject = f"{self._backend.ns}.l3.tx.begin"
        response = await self._backend.nats_request(subject, payload)
        if not response.get("success", False):
            raise DataLayerUnavailableError(
                f"tx.begin failed: {response.get('error_code', 'UNKNOWN')}: "
                f"{response.get('error_message', 'no details')}",
            )
        raw_tx_id = response.get("tx_id")
        if not isinstance(raw_tx_id, str):
            raise DataLayerUnavailableError(
                "tx.begin response missing tx_id",
            )
        self._connection.tx_id = UUID(raw_tx_id)
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        """commit on clean exit, rollback on exception.

        always clears :attr:`_ProxyConnection.tx_id` so subsequent
        ops on the same connection fall back to the outside-tx path.

        :param exc_type: exception type or None
        :ptype exc_type: Any
        :param exc_val: exception value or None
        :ptype exc_val: Any
        :param exc_tb: exception traceback or None
        :ptype exc_tb: Any
        :return: None
        :rtype: None
        :raises DataLayerUnavailableError: when commit/rollback fails
        """
        tx_id = self._connection.tx_id
        if tx_id is None:
            return
        payload: dict[str, Any] = {
            "correlation_id": str(uuid7()),
            "agent_id": self._backend.agent_id,
            "tx_id": str(tx_id),
        }
        action = "rollback" if exc_type is not None else "commit"
        subject = f"{self._backend.ns}.l3.tx.{action}"
        try:
            response = await self._backend.nats_request(subject, payload)
            if not response.get("success", False):
                # swallow the failure on the rollback path (we already
                # have an exception in flight) but surface it on the
                # commit path so the caller learns the DB did not
                # persist their work.
                if action == "commit":
                    raise DataLayerUnavailableError(
                        f"tx.commit failed: "
                        f"{response.get('error_code', 'UNKNOWN')}: "
                        f"{response.get('error_message', 'no details')}",
                    )
                _logger.warning(
                    "proxy tx.rollback reported failure: %s",
                    response.get("error_message"),
                )
        finally:
            self._connection.tx_id = None


class _ProxyPoolTransactionCM:
    """pool-level ``async with`` wrapper combining acquire + transaction.

    returned by :meth:`NatsProxyL3Backend.transaction`. compresses the
    two-level ``async with pool.acquire() as conn: async with
    conn.transaction(namespace=...):`` dance into a single
    ``async with pool.transaction(namespace=...) as conn:`` so
    workspace tools can express a cross-agent transactional write in
    one step.

    the yielded value is the :class:`_ProxyConnection` pinned to the
    broker-minted ``tx_id``; every ``execute`` / ``fetch`` /
    ``fetchrow`` call the caller issues against THIS connection routes
    through the session.

    cleanup on exit:

    1. if the transaction body raised, the inner
       :class:`_ProxyTransaction` sends rollback on its own
       ``__aexit__``; this wrapper then runs the outer
       :class:`_ProxyAcquireCM` exit which is a no-op because
       ``tx_id`` has already been cleared.
    2. if the body ran cleanly the inner commit fires, the acquire
       wrapper exits cleanly, and the connection is considered
       released.

    :param backend: backend instance for NATS routing
    :ptype backend: NatsProxyL3Backend
    :param namespace: logical namespace the session binds to
    :ptype namespace: str | None
    """

    def __init__(
        self,
        backend: "NatsProxyL3Backend",
        namespace: str | None,
    ) -> None:
        """capture the backend + namespace for deferred acquire at enter.

        :param backend: backend instance
        :ptype backend: NatsProxyL3Backend
        :param namespace: logical namespace the session binds to
        :ptype namespace: str | None
        :return: None
        :rtype: None
        """
        self._backend = backend
        self._namespace = namespace
        self._acquire_cm: _ProxyAcquireCM | None = None
        self._tx_cm: _ProxyTransaction | None = None

    async def __aenter__(self) -> "_ProxyConnection":
        """acquire a connection, begin a tx, and return the pinned conn.

        :return: proxy connection with ``tx_id`` set
        :rtype: _ProxyConnection
        :raises DataLayerUnavailableError: on broker error
        """
        self._acquire_cm = _ProxyAcquireCM(self._backend)
        conn = await self._acquire_cm.__aenter__()
        self._tx_cm = conn.transaction(namespace=self._namespace)
        await self._tx_cm.__aenter__()
        return conn

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        """commit/rollback the tx, then run the acquire safety-net.

        :param exc_type: exception type or None
        :ptype exc_type: Any
        :param exc_val: exception value or None
        :ptype exc_val: Any
        :param exc_tb: exception traceback or None
        :ptype exc_tb: Any
        :return: None
        :rtype: None
        """
        try:
            if self._tx_cm is not None:
                await self._tx_cm.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            if self._acquire_cm is not None:
                await self._acquire_cm.__aexit__(exc_type, exc_val, exc_tb)
