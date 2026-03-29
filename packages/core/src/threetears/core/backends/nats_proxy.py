"""NATS-based L3 backend proxy for agent pods.

drop-in replacement for direct asyncpg pool. routes all L3 queries
through Hub's L3 Broker via NATS request-reply. collections use
this transparently without knowing queries are proxied.
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid7

from threetears.core.exceptions import DataLayerUnavailableError
from threetears.core.logging import get_logger

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
        timeout_ms: int = 5000,
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
        :param timeout_ms: default query timeout in milliseconds
        :ptype timeout_ms: int
        """
        self._nc = nats_client
        self._ns = namespace_prefix
        self._agent_id = agent_id
        self._default_namespace = default_namespace or f"agent.{agent_id}"
        self._timeout_ms = timeout_ms

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
        return rows

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
    ) -> int:
        """execute INSERT/UPDATE/DELETE query and return rows affected.

        :param query: parameterized SQL query
        :ptype query: str
        :param params: query parameter values
        :ptype params: Any
        :param namespace: target namespace
        :ptype namespace: str | None
        :return: number of rows affected
        :rtype: int
        :raises DataLayerUnavailableError: if broker returns error
        """
        operation = _detect_operation(query)
        response = await self._send_query(query, list(params), operation, namespace)
        return response.get("row_count", 0) or 0

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
