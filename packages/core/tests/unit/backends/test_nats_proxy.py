"""Unit tests for NatsProxyL3Backend -- all NATS interactions are mocked."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from threetears.core.backends.nats_proxy import (
    NatsProxyL3Backend,
    _detect_operation,
    _serialize_param,
)
from threetears.core.exceptions import DataLayerUnavailableError


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------


def _make_reply(data: dict) -> MagicMock:  # type: ignore[type-arg]
    """build mock NATS reply message.

    :param data: response payload dict
    :ptype data: dict
    :return: mock reply with .data attribute
    :rtype: MagicMock
    """
    reply = MagicMock()
    reply.data = json.dumps(data).encode("utf-8")
    return reply


def _make_proxy(mock_nc: MagicMock) -> NatsProxyL3Backend:
    """build NatsProxyL3Backend with mock NATS client.

    :param mock_nc: mock NATS client
    :ptype mock_nc: MagicMock
    :return: configured proxy backend
    :rtype: NatsProxyL3Backend
    """
    return NatsProxyL3Backend(
        nats_client=mock_nc,
        namespace_prefix="test",
        agent_id="agent-123",
    )


# ------------------------------------------------------------------
# _serialize_param
# ------------------------------------------------------------------


class TestSerializeParam:
    def test_uuid(self) -> None:
        uid = UUID("12345678-1234-5678-1234-567812345678")
        assert _serialize_param(uid) == "12345678-1234-5678-1234-567812345678"

    def test_datetime(self) -> None:
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        assert _serialize_param(dt) == dt.isoformat()

    def test_decimal(self) -> None:
        d = Decimal("3.14")
        assert _serialize_param(d) == "3.14"

    def test_bytes(self) -> None:
        b = b"\xde\xad\xbe\xef"
        assert _serialize_param(b) == "deadbeef"

    def test_string_passthrough(self) -> None:
        assert _serialize_param("hello") == "hello"

    def test_int_passthrough(self) -> None:
        assert _serialize_param(42) == 42

    def test_none_passthrough(self) -> None:
        assert _serialize_param(None) is None


# ------------------------------------------------------------------
# _detect_operation
# ------------------------------------------------------------------


class TestDetectOperation:
    def test_select(self) -> None:
        assert _detect_operation("SELECT * FROM foo") == "select"

    def test_select_with_whitespace(self) -> None:
        assert _detect_operation("  SELECT * FROM foo") == "select"

    def test_insert(self) -> None:
        assert _detect_operation("INSERT INTO foo (a) VALUES ($1)") == "insert"

    def test_insert_on_conflict_upsert(self) -> None:
        assert _detect_operation(
            "INSERT INTO foo (a) VALUES ($1) ON CONFLICT (a) DO UPDATE SET a = $1"
        ) == "upsert"

    def test_update(self) -> None:
        assert _detect_operation("UPDATE foo SET a = $1 WHERE id = $2") == "update"

    def test_delete(self) -> None:
        assert _detect_operation("DELETE FROM foo WHERE id = $1") == "delete"

    def test_unknown_defaults_to_select(self) -> None:
        assert _detect_operation("WITH cte AS (SELECT 1)") == "select"


# ------------------------------------------------------------------
# default namespace
# ------------------------------------------------------------------


class TestDefaultNamespace:
    def test_default_namespace_from_agent_id(self) -> None:
        proxy = NatsProxyL3Backend(
            nats_client=MagicMock(),
            namespace_prefix="ns",
            agent_id="abc-def",
        )
        assert proxy._default_namespace == "agent.abc-def"

    def test_custom_namespace_override(self) -> None:
        proxy = NatsProxyL3Backend(
            nats_client=MagicMock(),
            namespace_prefix="ns",
            agent_id="abc-def",
            default_namespace="custom.namespace",
        )
        assert proxy._default_namespace == "custom.namespace"


# ------------------------------------------------------------------
# fetch
# ------------------------------------------------------------------


class TestFetch:
    @pytest.mark.asyncio
    async def test_fetch_returns_rows(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply({
                "success": True,
                "rows": [{"id": "abc"}, {"id": "def"}],
                "row_count": None,
                "duration_ms": 5,
            })
        )
        proxy = _make_proxy(mock_nc)

        rows = await proxy.fetch("SELECT * FROM foo WHERE id = $1", "abc")

        assert rows == [{"id": "abc"}, {"id": "def"}]
        mock_nc.request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_empty_rows(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply({
                "success": True,
                "rows": [],
                "row_count": None,
                "duration_ms": 2,
            })
        )
        proxy = _make_proxy(mock_nc)

        rows = await proxy.fetch("SELECT * FROM foo WHERE id = $1", "missing")

        assert rows == []

    @pytest.mark.asyncio
    async def test_fetch_subject_built_correctly(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply({
                "success": True,
                "rows": [],
                "row_count": None,
                "duration_ms": 1,
            })
        )
        proxy = _make_proxy(mock_nc)

        await proxy.fetch("SELECT 1")

        call_args = mock_nc.request.call_args
        subject = call_args[0][0]
        assert subject == "test.l3.query"

    @pytest.mark.asyncio
    async def test_fetch_with_custom_namespace(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply({
                "success": True,
                "rows": [{"x": 1}],
                "row_count": None,
                "duration_ms": 1,
            })
        )
        proxy = _make_proxy(mock_nc)

        rows = await proxy.fetch("SELECT 1", namespace="custom.ns")

        assert rows == [{"x": 1}]
        call_args = mock_nc.request.call_args
        payload = json.loads(call_args[0][1])
        assert payload["namespace"] == "custom.ns"


# ------------------------------------------------------------------
# fetchrow
# ------------------------------------------------------------------


class TestFetchrow:
    @pytest.mark.asyncio
    async def test_fetchrow_returns_first_row(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply({
                "success": True,
                "rows": [{"id": "first"}, {"id": "second"}],
                "row_count": None,
                "duration_ms": 3,
            })
        )
        proxy = _make_proxy(mock_nc)

        row = await proxy.fetchrow("SELECT * FROM foo LIMIT 1")

        assert row == {"id": "first"}

    @pytest.mark.asyncio
    async def test_fetchrow_returns_none_for_empty(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply({
                "success": True,
                "rows": [],
                "row_count": None,
                "duration_ms": 1,
            })
        )
        proxy = _make_proxy(mock_nc)

        row = await proxy.fetchrow("SELECT * FROM foo WHERE id = $1", "missing")

        assert row is None


# ------------------------------------------------------------------
# execute
# ------------------------------------------------------------------


class TestExecute:
    @pytest.mark.asyncio
    async def test_execute_returns_row_count(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply({
                "success": True,
                "rows": [],
                "row_count": 3,
                "duration_ms": 7,
            })
        )
        proxy = _make_proxy(mock_nc)

        count = await proxy.execute("UPDATE foo SET bar = $1", "baz")

        assert count == 3

    @pytest.mark.asyncio
    async def test_execute_returns_zero_for_null_row_count(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply({
                "success": True,
                "rows": [],
                "row_count": None,
                "duration_ms": 2,
            })
        )
        proxy = _make_proxy(mock_nc)

        count = await proxy.execute("DELETE FROM foo WHERE id = $1", "x")

        assert count == 0

    @pytest.mark.asyncio
    async def test_execute_detects_insert_operation(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply({
                "success": True,
                "rows": [],
                "row_count": 1,
                "duration_ms": 3,
            })
        )
        proxy = _make_proxy(mock_nc)

        await proxy.execute("INSERT INTO foo (a) VALUES ($1)", "val")

        call_args = mock_nc.request.call_args
        payload = json.loads(call_args[0][1])
        assert payload["operation"] == "insert"


# ------------------------------------------------------------------
# execute_batch
# ------------------------------------------------------------------


class TestExecuteBatch:
    @pytest.mark.asyncio
    async def test_execute_batch_returns_results(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply({
                "success": True,
                "results": [
                    {"rows": [{"id": "a"}], "row_count": None},
                    {"rows": [], "row_count": 1},
                ],
                "duration_ms": 10,
            })
        )
        proxy = _make_proxy(mock_nc)

        results = await proxy.execute_batch([
            {"query": "SELECT * FROM foo", "params": []},
            {"query": "INSERT INTO bar (a) VALUES ($1)", "params": ["val"]},
        ])

        assert len(results) == 2
        call_args = mock_nc.request.call_args
        subject = call_args[0][0]
        assert subject == "test.l3.batch"

    @pytest.mark.asyncio
    async def test_execute_batch_error_raises(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply({
                "success": False,
                "error_code": "TRANSACTION_FAILED",
                "error_message": "deadlock detected",
            })
        )
        proxy = _make_proxy(mock_nc)

        with pytest.raises(DataLayerUnavailableError, match="batch query failed"):
            await proxy.execute_batch([
                {"query": "UPDATE foo SET a = 1", "params": []},
            ])

    @pytest.mark.asyncio
    async def test_execute_batch_auto_detects_operation(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply({
                "success": True,
                "results": [{"rows": [], "row_count": 1}],
                "duration_ms": 2,
            })
        )
        proxy = _make_proxy(mock_nc)

        await proxy.execute_batch([
            {"query": "DELETE FROM foo WHERE id = $1", "params": ["abc"]},
        ])

        call_args = mock_nc.request.call_args
        payload = json.loads(call_args[0][1])
        assert payload["queries"][0]["operation"] == "delete"


# ------------------------------------------------------------------
# error handling
# ------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_acl_denied_raises_data_layer_unavailable(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply({
                "success": False,
                "error_code": "NAMESPACE_ACCESS_DENIED",
                "error_message": "agent not authorized for this namespace",
            })
        )
        proxy = _make_proxy(mock_nc)

        with pytest.raises(DataLayerUnavailableError, match="NAMESPACE_ACCESS_DENIED"):
            await proxy.fetch("SELECT * FROM foo")

    @pytest.mark.asyncio
    async def test_nats_timeout_raises_data_layer_unavailable(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(side_effect=TimeoutError("request timed out"))
        proxy = _make_proxy(mock_nc)

        with pytest.raises(DataLayerUnavailableError, match="NATS request failed"):
            await proxy.fetch("SELECT * FROM foo")

    @pytest.mark.asyncio
    async def test_nats_connection_error_raises_data_layer_unavailable(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(side_effect=ConnectionError("no route to host"))
        proxy = _make_proxy(mock_nc)

        with pytest.raises(DataLayerUnavailableError, match="NATS request failed"):
            await proxy.execute("INSERT INTO foo (a) VALUES ($1)", "val")


# ------------------------------------------------------------------
# payload verification
# ------------------------------------------------------------------


class TestPayloadFormat:
    @pytest.mark.asyncio
    async def test_query_payload_contains_required_fields(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply({
                "success": True,
                "rows": [],
                "row_count": None,
                "duration_ms": 1,
            })
        )
        proxy = _make_proxy(mock_nc)

        await proxy.fetch("SELECT * FROM foo WHERE id = $1", "abc")

        call_args = mock_nc.request.call_args
        payload = json.loads(call_args[0][1])
        assert "correlation_id" in payload
        assert payload["agent_id"] == "agent-123"
        assert payload["namespace"] == "agent.agent-123"
        assert payload["operation"] == "select"
        assert payload["query"] == "SELECT * FROM foo WHERE id = $1"
        assert payload["params"] == ["abc"]
        assert payload["timeout_ms"] == 5000

    @pytest.mark.asyncio
    async def test_params_are_serialized(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply({
                "success": True,
                "rows": [],
                "row_count": 1,
                "duration_ms": 1,
            })
        )
        proxy = _make_proxy(mock_nc)
        uid = UUID("12345678-1234-5678-1234-567812345678")
        dt = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)

        await proxy.execute(
            "INSERT INTO foo (id, date_created) VALUES ($1, $2)", uid, dt,
        )

        call_args = mock_nc.request.call_args
        payload = json.loads(call_args[0][1])
        assert payload["params"][0] == str(uid)
        assert payload["params"][1] == dt.isoformat()

    @pytest.mark.asyncio
    async def test_batch_payload_contains_required_fields(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply({
                "success": True,
                "results": [],
                "duration_ms": 1,
            })
        )
        proxy = _make_proxy(mock_nc)

        await proxy.execute_batch(
            [{"query": "SELECT 1", "params": []}],
            transaction=False,
        )

        call_args = mock_nc.request.call_args
        payload = json.loads(call_args[0][1])
        assert "correlation_id" in payload
        assert payload["agent_id"] == "agent-123"
        assert payload["namespace"] == "agent.agent-123"
        assert payload["transaction"] is False
        assert len(payload["queries"]) == 1

    @pytest.mark.asyncio
    async def test_custom_timeout_ms(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply({
                "success": True,
                "rows": [],
                "row_count": None,
                "duration_ms": 1,
            })
        )
        proxy = NatsProxyL3Backend(
            nats_client=mock_nc,
            namespace_prefix="test",
            agent_id="agent-123",
            timeout_ms=10000,
        )

        await proxy.fetch("SELECT 1")

        call_args = mock_nc.request.call_args
        payload = json.loads(call_args[0][1])
        assert payload["timeout_ms"] == 10000
        nats_timeout = call_args[1].get("timeout") or call_args[0][2]
        assert nats_timeout == 12.0
