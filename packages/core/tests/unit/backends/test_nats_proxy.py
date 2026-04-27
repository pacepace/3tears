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
    _deserialize_row,
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
        assert _serialize_param(b) == "\\xdeadbeef"

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
        assert _detect_operation("INSERT INTO foo (a) VALUES ($1) ON CONFLICT (a) DO UPDATE SET a = $1") == "upsert"

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
        assert proxy.default_namespace == "agents.abc-def"

    def test_custom_namespace_override(self) -> None:
        proxy = NatsProxyL3Backend(
            nats_client=MagicMock(),
            namespace_prefix="ns",
            agent_id="abc-def",
            default_namespace="custom.namespace",
        )
        assert proxy.default_namespace == "custom.namespace"


# ------------------------------------------------------------------
# fetch
# ------------------------------------------------------------------


class TestFetch:
    @pytest.mark.asyncio
    async def test_fetch_returns_rows(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply(
                {
                    "success": True,
                    "rows": [{"id": "abc"}, {"id": "def"}],
                    "row_count": None,
                    "duration_ms": 5,
                }
            )
        )
        proxy = _make_proxy(mock_nc)

        rows = await proxy.fetch("SELECT * FROM foo WHERE id = $1", "abc")

        assert rows == [{"id": "abc"}, {"id": "def"}]
        mock_nc.request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_empty_rows(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply(
                {
                    "success": True,
                    "rows": [],
                    "row_count": None,
                    "duration_ms": 2,
                }
            )
        )
        proxy = _make_proxy(mock_nc)

        rows = await proxy.fetch("SELECT * FROM foo WHERE id = $1", "missing")

        assert rows == []

    @pytest.mark.asyncio
    async def test_fetch_subject_built_correctly(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply(
                {
                    "success": True,
                    "rows": [],
                    "row_count": None,
                    "duration_ms": 1,
                }
            )
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
            return_value=_make_reply(
                {
                    "success": True,
                    "rows": [{"x": 1}],
                    "row_count": None,
                    "duration_ms": 1,
                }
            )
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
            return_value=_make_reply(
                {
                    "success": True,
                    "rows": [{"id": "first"}, {"id": "second"}],
                    "row_count": None,
                    "duration_ms": 3,
                }
            )
        )
        proxy = _make_proxy(mock_nc)

        row = await proxy.fetchrow("SELECT * FROM foo LIMIT 1")

        assert row == {"id": "first"}

    @pytest.mark.asyncio
    async def test_fetchrow_returns_none_for_empty(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply(
                {
                    "success": True,
                    "rows": [],
                    "row_count": None,
                    "duration_ms": 1,
                }
            )
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
            return_value=_make_reply(
                {
                    "success": True,
                    "rows": [],
                    "row_count": 3,
                    "duration_ms": 7,
                }
            )
        )
        proxy = _make_proxy(mock_nc)

        tag = await proxy.execute("UPDATE foo SET bar = $1", "baz")

        # asyncpg-shape tag; collections parse via int(tag.split()[-1]).
        assert tag == "UPDATE 3"
        assert int(tag.split()[-1]) == 3

    @pytest.mark.asyncio
    async def test_execute_returns_zero_for_null_row_count(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply(
                {
                    "success": True,
                    "rows": [],
                    "row_count": None,
                    "duration_ms": 2,
                }
            )
        )
        proxy = _make_proxy(mock_nc)

        tag = await proxy.execute("DELETE FROM foo WHERE id = $1", "x")

        # null row_count from the broker collapses to zero on the wire.
        assert tag == "DELETE 0"
        assert int(tag.split()[-1]) == 0

    @pytest.mark.asyncio
    async def test_execute_detects_insert_operation(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply(
                {
                    "success": True,
                    "rows": [],
                    "row_count": 1,
                    "duration_ms": 3,
                }
            )
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
            return_value=_make_reply(
                {
                    "success": True,
                    "results": [
                        {"rows": [{"id": "a"}], "row_count": None},
                        {"rows": [], "row_count": 1},
                    ],
                    "duration_ms": 10,
                }
            )
        )
        proxy = _make_proxy(mock_nc)

        results = await proxy.execute_batch(
            [
                {"query": "SELECT * FROM foo", "params": []},
                {"query": "INSERT INTO bar (a) VALUES ($1)", "params": ["val"]},
            ]
        )

        assert len(results) == 2
        call_args = mock_nc.request.call_args
        subject = call_args[0][0]
        assert subject == "test.l3.batch"

    @pytest.mark.asyncio
    async def test_execute_batch_error_raises(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply(
                {
                    "success": False,
                    "error_code": "TRANSACTION_FAILED",
                    "error_message": "deadlock detected",
                }
            )
        )
        proxy = _make_proxy(mock_nc)

        with pytest.raises(DataLayerUnavailableError, match="batch query failed"):
            await proxy.execute_batch(
                [
                    {"query": "UPDATE foo SET a = 1", "params": []},
                ]
            )

    @pytest.mark.asyncio
    async def test_execute_batch_auto_detects_operation(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply(
                {
                    "success": True,
                    "results": [{"rows": [], "row_count": 1}],
                    "duration_ms": 2,
                }
            )
        )
        proxy = _make_proxy(mock_nc)

        await proxy.execute_batch(
            [
                {"query": "DELETE FROM foo WHERE id = $1", "params": ["abc"]},
            ]
        )

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
            return_value=_make_reply(
                {
                    "success": False,
                    "error_code": "NAMESPACE_ACCESS_DENIED",
                    "error_message": "agent not authorized for this namespace",
                }
            )
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
            return_value=_make_reply(
                {
                    "success": True,
                    "rows": [],
                    "row_count": None,
                    "duration_ms": 1,
                }
            )
        )
        proxy = _make_proxy(mock_nc)

        await proxy.fetch("SELECT * FROM foo WHERE id = $1", "abc")

        call_args = mock_nc.request.call_args
        payload = json.loads(call_args[0][1])
        assert "correlation_id" in payload
        assert payload["agent_id"] == "agent-123"
        assert payload["namespace"] == "agents.agent-123"
        assert payload["operation"] == "select"
        assert payload["query"] == "SELECT * FROM foo WHERE id = $1"
        assert payload["params"] == ["abc"]
        assert payload["timeout_ms"] == 5000

    @pytest.mark.asyncio
    async def test_params_are_serialized(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply(
                {
                    "success": True,
                    "rows": [],
                    "row_count": 1,
                    "duration_ms": 1,
                }
            )
        )
        proxy = _make_proxy(mock_nc)
        uid = UUID("12345678-1234-5678-1234-567812345678")
        dt = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)

        await proxy.execute(
            "INSERT INTO foo (id, date_created) VALUES ($1, $2)",
            uid,
            dt,
        )

        call_args = mock_nc.request.call_args
        payload = json.loads(call_args[0][1])
        assert payload["params"][0] == str(uid)
        assert payload["params"][1] == dt.isoformat()

    @pytest.mark.asyncio
    async def test_batch_payload_contains_required_fields(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply(
                {
                    "success": True,
                    "results": [],
                    "duration_ms": 1,
                }
            )
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
        assert payload["namespace"] == "agents.agent-123"
        assert payload["transaction"] is False
        assert len(payload["queries"]) == 1

    @pytest.mark.asyncio
    async def test_custom_timeout_ms(self) -> None:
        mock_nc = MagicMock()
        mock_nc.request = AsyncMock(
            return_value=_make_reply(
                {
                    "success": True,
                    "rows": [],
                    "row_count": None,
                    "duration_ms": 1,
                }
            )
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


# -- row deserialization tests (bytes round-trip) --


class TestDeserializeRow:
    """tests for _deserialize_row hex bytes decoding."""

    def test_hex_string_decoded_to_bytes(self) -> None:
        """\\x-prefixed hex strings are decoded back to bytes."""
        row = {"id": "abc", "blob": "\\x800102ff"}
        result = _deserialize_row(row)
        assert result["blob"] == b"\x80\x01\x02\xff"
        assert result["id"] == "abc"

    def test_non_hex_strings_unchanged(self) -> None:
        """regular strings without \\x prefix pass through unchanged."""
        row = {"name": "hello", "type": "msgpack"}
        result = _deserialize_row(row)
        assert result == row

    def test_non_string_values_unchanged(self) -> None:
        """int, None, bool values pass through unchanged."""
        row = {"count": 42, "active": True, "nullable": None}
        result = _deserialize_row(row)
        assert result == row

    def test_empty_hex_decoded_to_empty_bytes(self) -> None:
        """\\x with no hex digits decodes to empty bytes."""
        row = {"empty": "\\x"}
        result = _deserialize_row(row)
        assert result["empty"] == b""

    def test_checkpoint_blob_round_trip(self) -> None:
        """binary checkpoint data survives hex encode/decode round-trip."""
        original_blob = bytes(range(256))
        hex_encoded = "\\x" + original_blob.hex()
        row = {"checkpoint": hex_encoded, "metadata_": hex_encoded, "type": "msgpack"}
        result = _deserialize_row(row)
        assert result["checkpoint"] == original_blob
        assert result["metadata_"] == original_blob
        assert result["type"] == "msgpack"


class TestFetchDeserializesBytes:
    """tests that fetch() deserializes hex-encoded bytes in rows."""

    @pytest.mark.asyncio
    async def test_fetch_decodes_hex_bytes_in_rows(self) -> None:
        """fetch returns rows with bytes restored from hex encoding."""
        blob = b"\x80\x01\x02"
        hex_blob = "\\x" + blob.hex()
        mock_nc = AsyncMock()
        response_data = {
            "success": True,
            "rows": [
                {"thread_id": "t1", "checkpoint": hex_blob, "type": "msgpack"},
            ],
            "duration_ms": 5,
        }
        mock_reply = MagicMock()
        mock_reply.data = json.dumps(response_data).encode()
        mock_nc.request = AsyncMock(return_value=mock_reply)

        proxy = NatsProxyL3Backend(
            nats_client=mock_nc,
            namespace_prefix="aibots",
            agent_id="agent-test",
        )

        rows = await proxy.fetch("SELECT thread_id, checkpoint, type FROM checkpoints")
        assert len(rows) == 1
        assert rows[0]["checkpoint"] == blob
        assert rows[0]["type"] == "msgpack"
        assert rows[0]["thread_id"] == "t1"

    @pytest.mark.asyncio
    async def test_fetchrow_decodes_hex_bytes(self) -> None:
        """fetchrow returns single row with bytes restored from hex encoding."""
        blob = b"\xde\xad\xbe\xef"
        hex_blob = "\\x" + blob.hex()
        mock_nc = AsyncMock()
        response_data = {
            "success": True,
            "rows": [
                {"id": "r1", "data": hex_blob},
            ],
            "duration_ms": 3,
        }
        mock_reply = MagicMock()
        mock_reply.data = json.dumps(response_data).encode()
        mock_nc.request = AsyncMock(return_value=mock_reply)

        proxy = NatsProxyL3Backend(
            nats_client=mock_nc,
            namespace_prefix="aibots",
            agent_id="agent-test",
        )

        row = await proxy.fetchrow("SELECT id, data FROM test WHERE id = $1", "r1")
        assert row is not None
        assert row["data"] == blob
        assert row["id"] == "r1"


# ------------------------------------------------------------------
# acquire() + transaction() shim
# ------------------------------------------------------------------


class _ScriptedReplyPlan:
    """drive mock ``nc.request`` with a scripted list of responses.

    each call pops the next plan entry, checks the subject matches,
    optionally records the payload for later assertion, and returns the
    pre-built reply. lets tests assert both "the right subject was
    called in the right order" and "the right payload was shipped"
    without hand-rolling stateful mocks per test.
    """

    def __init__(self, entries: list[tuple[str, dict]]) -> None:
        """capture ``(subject, response_dict)`` pairs in order.

        :param entries: list of ``(expected_subject, response_dict)``
        :ptype entries: list[tuple[str, dict]]
        :return: None
        :rtype: None
        """
        self._entries = list(entries)
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, subject: str, payload: bytes, timeout: float = 0):
        """mock coroutine for ``nc.request``.

        :param subject: subject argument passed by the proxy
        :ptype subject: str
        :param payload: raw bytes payload
        :ptype payload: bytes
        :param timeout: ignored
        :ptype timeout: float
        :return: pre-built reply
        :rtype: MagicMock
        :raises AssertionError: when a subject is called out of order
        """
        del timeout
        assert self._entries, f"no reply scripted for {subject}"
        expected_subject, response = self._entries.pop(0)
        assert subject == expected_subject, f"expected {expected_subject!r}, got {subject!r}"
        self.calls.append((subject, json.loads(payload.decode("utf-8"))))
        return _make_reply(response)


@pytest.mark.asyncio
async def test_acquire_outside_tx_routes_to_l3_query() -> None:
    """a proxy connection outside a transaction uses l3.query/l3.batch path."""
    plan = _ScriptedReplyPlan(
        [
            ("test.l3.query", {"success": True, "rows": [{"n": 1}]}),
        ]
    )
    mock_nc = MagicMock()
    mock_nc.request = plan
    proxy = _make_proxy(mock_nc)

    async with proxy.acquire() as conn:
        rows = await conn.fetch("SELECT 1 AS n")

    assert rows == [{"n": 1}]
    assert plan.calls[0][0] == "test.l3.query"


@pytest.mark.asyncio
async def test_transaction_happy_path_commits() -> None:
    """entering + clean-exiting a transaction calls begin + commit."""
    tx_id = "019d9a00-0000-7000-8000-000000000000"
    plan = _ScriptedReplyPlan(
        [
            ("test.l3.tx.begin", {"success": True, "tx_id": tx_id}),
            ("test.l3.tx.execute", {"success": True, "row_count": 1}),
            ("test.l3.tx.commit", {"success": True}),
        ]
    )
    mock_nc = MagicMock()
    mock_nc.request = plan
    proxy = _make_proxy(mock_nc)

    async with proxy.acquire() as conn:
        async with conn.transaction():
            tag = await conn.execute(
                "INSERT INTO t(x) VALUES ($1)",
                42,
            )
            # inside-tx execute returns the same asyncpg-shape tag as
            # outside-tx; collections that split on whitespace and
            # parse the trailing integer work unchanged.
            assert tag == "INSERT 0 1"
            assert int(tag.split()[-1]) == 1

    subjects = [c[0] for c in plan.calls]
    assert subjects == [
        "test.l3.tx.begin",
        "test.l3.tx.execute",
        "test.l3.tx.commit",
    ]


@pytest.mark.asyncio
async def test_transaction_exception_rolls_back() -> None:
    """an exception inside the transaction body sends rollback, not commit."""
    tx_id = "019d9a00-0000-7000-8000-000000000001"
    plan = _ScriptedReplyPlan(
        [
            ("test.l3.tx.begin", {"success": True, "tx_id": tx_id}),
            ("test.l3.tx.rollback", {"success": True}),
        ]
    )
    mock_nc = MagicMock()
    mock_nc.request = plan
    proxy = _make_proxy(mock_nc)

    class _Boom(RuntimeError):
        pass

    with pytest.raises(_Boom):
        async with proxy.acquire() as conn:
            async with conn.transaction():
                raise _Boom("body crashed")

    subjects = [c[0] for c in plan.calls]
    assert subjects == ["test.l3.tx.begin", "test.l3.tx.rollback"]


@pytest.mark.asyncio
async def test_transaction_fetchrow_routes_through_tx_id() -> None:
    """fetchrow inside a tx goes to tx.fetchrow, carries tx_id."""
    tx_id = "019d9a00-0000-7000-8000-000000000002"
    plan = _ScriptedReplyPlan(
        [
            ("test.l3.tx.begin", {"success": True, "tx_id": tx_id}),
            ("test.l3.tx.fetchrow", {"success": True, "row": {"v": 7}}),
            ("test.l3.tx.commit", {"success": True}),
        ]
    )
    mock_nc = MagicMock()
    mock_nc.request = plan
    proxy = _make_proxy(mock_nc)

    async with proxy.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow("SELECT MAX(v) AS v FROM t")

    assert row == {"v": 7}
    fetch_call = plan.calls[1]
    assert fetch_call[1]["tx_id"] == tx_id


@pytest.mark.asyncio
async def test_transaction_begin_failure_raises() -> None:
    """a broker error on tx.begin surfaces as DataLayerUnavailableError."""
    plan = _ScriptedReplyPlan(
        [
            (
                "test.l3.tx.begin",
                {
                    "success": False,
                    "error_code": "NAMESPACE_ACCESS_DENIED",
                    "error_message": "no write grant",
                },
            ),
        ]
    )
    mock_nc = MagicMock()
    mock_nc.request = plan
    proxy = _make_proxy(mock_nc)

    with pytest.raises(DataLayerUnavailableError) as excinfo:
        async with proxy.acquire() as conn:
            async with conn.transaction():
                pass

    assert "NAMESPACE_ACCESS_DENIED" in str(excinfo.value)


@pytest.mark.asyncio
async def test_nested_transaction_rejected() -> None:
    """opening a transaction while one is already active raises."""
    tx_id = "019d9a00-0000-7000-8000-000000000003"
    plan = _ScriptedReplyPlan(
        [
            ("test.l3.tx.begin", {"success": True, "tx_id": tx_id}),
            ("test.l3.tx.rollback", {"success": True}),
        ]
    )
    mock_nc = MagicMock()
    mock_nc.request = plan
    proxy = _make_proxy(mock_nc)

    with pytest.raises(RuntimeError, match="nested transactions"):
        async with proxy.acquire() as conn:
            async with conn.transaction():
                async with conn.transaction():  # <- should raise
                    pass


@pytest.mark.asyncio
async def test_dangling_tx_rolled_back_on_connection_close() -> None:
    """forgetting to exit a transaction triggers a safety-net rollback.

    asyncpg Pool.acquire as a context manager does NOT auto-commit
    dangling transactions; the proxy's acquire() provides the same
    safety net by rolling back whatever tx_id is pinned on the
    connection at __aexit__ time. pool connection release happens
    broker-side as part of the rollback.
    """
    tx_id = "019d9a00-0000-7000-8000-000000000004"
    plan = _ScriptedReplyPlan(
        [
            ("test.l3.tx.begin", {"success": True, "tx_id": tx_id}),
            # no commit, but acquire-CM will send tx.rollback on exit.
            ("test.l3.tx.rollback", {"success": True}),
        ]
    )
    mock_nc = MagicMock()
    mock_nc.request = plan
    proxy = _make_proxy(mock_nc)

    async with proxy.acquire() as conn:
        tx_ctx = conn.transaction()
        await tx_ctx.__aenter__()
        # deliberately skip __aexit__ so the acquire-CM rescues us.

    subjects = [c[0] for c in plan.calls]
    assert subjects[-1] == "test.l3.tx.rollback"


@pytest.mark.asyncio
async def test_transaction_commit_failure_raises() -> None:
    """a broker error on tx.commit surfaces as DataLayerUnavailableError.

    distinct from rollback-on-error: commit failure means the DB
    did not persist the caller's work, so the exception needs to
    reach the caller instead of being swallowed like rollback errors.
    """
    tx_id = "019d9a00-0000-7000-8000-000000000005"
    plan = _ScriptedReplyPlan(
        [
            ("test.l3.tx.begin", {"success": True, "tx_id": tx_id}),
            (
                "test.l3.tx.commit",
                {
                    "success": False,
                    "error_code": "TX_COMMIT_FAILED",
                    "error_message": "serialization failure",
                },
            ),
        ]
    )
    mock_nc = MagicMock()
    mock_nc.request = plan
    proxy = _make_proxy(mock_nc)

    with pytest.raises(DataLayerUnavailableError) as excinfo:
        async with proxy.acquire() as conn:
            async with conn.transaction():
                pass

    assert "TX_COMMIT_FAILED" in str(excinfo.value)


# ------------------------------------------------------------------
# _deserialize_row datetime rehydration
# ------------------------------------------------------------------


class TestDeserializeRowDatetimes:
    """pins the proxy's date_* -> datetime normalization.

    broker JSON-encodes TIMESTAMP columns to iso strings; entities on
    the agent side call ``.isoformat()`` on those fields expecting
    native datetime objects. without this rehydration every fs_list /
    workspace_list / memory read blows up with ``'str' has no
    attribute 'isoformat'``.
    """

    def test_aware_iso_string_rehydrates_to_datetime(self) -> None:
        """explicit offsets round-trip without tz changes."""
        from threetears.core.backends.nats_proxy import _deserialize_row

        row = {"date_updated": "2026-04-17T12:34:56+00:00"}
        out = _deserialize_row(row)
        assert isinstance(out["date_updated"], datetime)
        assert out["date_updated"].tzinfo is not None
        assert out["date_updated"].isoformat() == "2026-04-17T12:34:56+00:00"

    def test_zulu_suffix_rehydrates_to_utc_datetime(self) -> None:
        """``...Z`` trailing marker normalizes to +00:00."""
        from threetears.core.backends.nats_proxy import _deserialize_row

        row = {"date_created": "2026-04-17T12:34:56Z"}
        out = _deserialize_row(row)
        assert isinstance(out["date_created"], datetime)
        assert out["date_created"].tzinfo is not None

    def test_naive_iso_string_gets_utc(self) -> None:
        """naive iso strings gain UTC defensively so tz checks pass."""
        from threetears.core.backends.nats_proxy import _deserialize_row

        row = {"date_created": "2026-04-17T12:34:56"}
        out = _deserialize_row(row)
        assert isinstance(out["date_created"], datetime)
        assert out["date_created"].tzinfo is not None

    def test_non_date_columns_left_alone(self) -> None:
        """the rehydrator only touches ``date_*`` column names."""
        from threetears.core.backends.nats_proxy import _deserialize_row

        row = {"name": "2026-04-17T12:34:56", "id": "some-uuid"}
        out = _deserialize_row(row)
        assert out["name"] == "2026-04-17T12:34:56"
        assert out["id"] == "some-uuid"

    def test_unparseable_date_string_passes_through(self) -> None:
        """malformed date_* values survive so the caller sees them."""
        from threetears.core.backends.nats_proxy import _deserialize_row

        row = {"date_created": "not an iso stamp"}
        out = _deserialize_row(row)
        assert out["date_created"] == "not an iso stamp"

    def test_empty_date_string_not_parsed(self) -> None:
        """empty string does not produce a surprise datetime."""
        from threetears.core.backends.nats_proxy import _deserialize_row

        row = {"date_created": ""}
        out = _deserialize_row(row)
        assert out["date_created"] == ""
