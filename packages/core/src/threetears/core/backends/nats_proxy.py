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
from threetears.observe import get_logger

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
        elif (
            isinstance(value, str)
            and key.startswith("date_")
            and value
        ):
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
        self._ns = namespace_prefix
        self._agent_id = agent_id
        self._default_namespace = default_namespace or f"agent.{agent_id}"
        if timeout_ms is not None:
            self._timeout_ms = timeout_ms
        else:
            import os
            raw = os.environ.get("THREETEARS_NATS_PROXY_TIMEOUT_MS")
            self._timeout_ms = int(raw) if raw is not None else 5000

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
        ns = namespace or self._default_namespace
        payload = {
            "correlation_id": str(uuid7()),
            "agent_id": self._agent_id,
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
        subject = f"{self._ns}.l3.batch"
        response = await self._nats_request(subject, payload)

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
        ns = namespace or self._default_namespace
        payload = {
            "correlation_id": str(uuid7()),
            "agent_id": self._agent_id,
            "namespace": ns,
            "operation": operation,
            "query": query,
            "params": [_serialize_param(p) for p in params],
            "timeout_ms": self._timeout_ms,
        }
        subject = f"{self._ns}.l3.query"
        response = await self._nats_request(subject, payload)

        if not response.get("success", False):
            raise DataLayerUnavailableError(
                f"L3 query failed: {response.get('error_code', 'UNKNOWN')}: "
                f"{response.get('error_message', 'no details')}"
            )

        return response

    async def _nats_request(
        self,
        subject: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """send NATS request and parse JSON response.

        :param subject: NATS subject
        :ptype subject: str
        :param payload: request payload dict
        :ptype payload: dict[str, Any]
        :return: parsed response dict
        :rtype: dict[str, Any]
        :raises DataLayerUnavailableError: if NATS request fails
        """
        payload_bytes = json.dumps(payload, default=str).encode("utf-8")
        nats_timeout = (self._timeout_ms / 1000) + 2

        try:
            reply = await self._nc.request(subject, payload_bytes, timeout=nats_timeout)
            result: dict[str, Any] = json.loads(reply.data)
        except Exception as exc:
            raise DataLayerUnavailableError(
                f"NATS request failed: subject={subject}: {exc}"
            ) from exc

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
                await self._connection._abort_tx()
            except Exception as abort_exc:
                log.warning(
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

    def transaction(self) -> "_ProxyTransaction":
        """return a ``conn.transaction()``-shaped context manager.

        mirrors :meth:`asyncpg.Connection.transaction`. nested calls
        are rejected at ``__aenter__`` so callers who unintentionally
        open two transactions on one connection fail loudly instead
        of silently sharing tx state.

        :return: transaction context manager
        :rtype: _ProxyTransaction
        """
        return _ProxyTransaction(self, self._backend)

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
        :param namespace: override namespace for outside-tx calls
            (ignored inside a tx since the session is bound at begin)
        :ptype namespace: str | None
        :return: asyncpg-shape status tag
        :rtype: str
        :raises DataLayerUnavailableError: on broker error
        """
        if self.tx_id is None:
            return await self._backend.execute(
                query, *params, namespace=namespace,
            )
        payload: dict[str, Any] = {
            "correlation_id": str(uuid7()),
            "agent_id": self._backend._agent_id,
            "tx_id": str(self.tx_id),
            "query": query,
            "params": [_serialize_param(p) for p in params],
        }
        subject = f"{self._backend._ns}.l3.tx.execute"
        response = await self._backend._nats_request(subject, payload)
        if not response.get("success", False):
            raise DataLayerUnavailableError(
                f"tx.execute failed: {response.get('error_code', 'UNKNOWN')}: "
                f"{response.get('error_message', 'no details')}",
            )
        return _format_execute_tag(
            _detect_operation(query), response.get("row_count"),
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
        :param namespace: override namespace for outside-tx calls
        :ptype namespace: str | None
        :return: first row dict or None
        :rtype: dict[str, Any] | None
        """
        if self.tx_id is None:
            outside_row = await self._backend.fetchrow(
                query, *params, namespace=namespace,
            )
            return outside_row
        payload: dict[str, Any] = {
            "correlation_id": str(uuid7()),
            "agent_id": self._backend._agent_id,
            "tx_id": str(self.tx_id),
            "query": query,
            "params": [_serialize_param(p) for p in params],
        }
        subject = f"{self._backend._ns}.l3.tx.fetchrow"
        response = await self._backend._nats_request(subject, payload)
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
        :param namespace: override namespace for outside-tx calls
        :ptype namespace: str | None
        :return: row dicts
        :rtype: list[dict[str, Any]]
        """
        if self.tx_id is None:
            outside_rows = await self._backend.fetch(
                query, *params, namespace=namespace,
            )
            return outside_rows
        payload: dict[str, Any] = {
            "correlation_id": str(uuid7()),
            "agent_id": self._backend._agent_id,
            "tx_id": str(self.tx_id),
            "query": query,
            "params": [_serialize_param(p) for p in params],
        }
        subject = f"{self._backend._ns}.l3.tx.fetch"
        response = await self._backend._nats_request(subject, payload)
        if not response.get("success", False):
            raise DataLayerUnavailableError(
                f"tx.fetch failed: {response.get('error_code', 'UNKNOWN')}: "
                f"{response.get('error_message', 'no details')}",
            )
        raw_rows: list[dict[str, Any]] = response.get("rows", [])
        return [_deserialize_row(r) for r in raw_rows]

    async def _abort_tx(self) -> None:
        """send ``tx.rollback`` and clear the local tx_id.

        invoked by :class:`_ProxyTransaction.__aexit__` on an exception
        path and by :class:`_ProxyAcquireCM.__aexit__` as a safety net
        when a caller forgets to end the transaction. idempotent on
        double-call so the two exit hooks can both fire without
        hitting the broker twice.

        :return: None
        :rtype: None
        """
        tx_id = self.tx_id
        if tx_id is None:
            return
        payload: dict[str, Any] = {
            "correlation_id": str(uuid7()),
            "agent_id": self._backend._agent_id,
            "tx_id": str(tx_id),
        }
        subject = f"{self._backend._ns}.l3.tx.rollback"
        try:
            await self._backend._nats_request(subject, payload)
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
    """

    def __init__(
        self,
        connection: "_ProxyConnection",
        backend: "NatsProxyL3Backend",
    ) -> None:
        """capture the owning connection and backend.

        :param connection: owning connection
        :ptype connection: _ProxyConnection
        :param backend: backend instance
        :ptype backend: NatsProxyL3Backend
        :return: None
        :rtype: None
        """
        self._connection = connection
        self._backend = backend

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
        payload: dict[str, Any] = {
            "correlation_id": str(uuid7()),
            "agent_id": self._backend._agent_id,
            "namespace": self._backend._default_namespace,
            "statement_timeout_ms": self._backend._timeout_ms,
        }
        subject = f"{self._backend._ns}.l3.tx.begin"
        response = await self._backend._nats_request(subject, payload)
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
            "agent_id": self._backend._agent_id,
            "tx_id": str(tx_id),
        }
        action = "rollback" if exc_type is not None else "commit"
        subject = f"{self._backend._ns}.l3.tx.{action}"
        try:
            response = await self._backend._nats_request(subject, payload)
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
                log.warning(
                    "proxy tx.rollback reported failure: %s",
                    response.get("error_message"),
                )
        finally:
            self._connection.tx_id = None
