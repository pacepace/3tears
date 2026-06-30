"""Tests for the producer-side object-store streaming helper.

Covers the produce seam: a tool streams a large result to the object store
that rides the per-call :class:`ToolCallScope`, gets back an
:class:`ObjectHandle`, and the key is built from the VERIFIED ``customer_id``.
Fail-closed paths (no scope / no store / no customer / no owning context) are
exercised explicitly -- a producing tool must never silently emit an unscoped
or untenanted object.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import AsyncIterator
from uuid import UUID

import pytest

from threetears.agent.tools.call_scope import ToolCallScope, enter_call_scope
from threetears.agent.tools.context_envelope import CallContext
from threetears.agent.tools.produce import (
    ProduceObjectError,
    stream_result_to_object_store,
)
from threetears.agent.tools.server import CallRequest, ToolServer

_CUSTOMER = UUID("06a41d51-a6d5-7824-8000-29ab66754fc0")
_CONVERSATION = UUID("019f1900-0000-7000-8000-000000000001")
_AGENT = UUID("019f1900-0000-7000-8000-000000000002")
_ENGAGEMENT = UUID("019f1900-0000-7000-8000-000000000003")
_CREATED = datetime(2026, 6, 30, 14, 5, 0, tzinfo=UTC)


# parity-with: threetears.media.contracts.ObjectStore
class _FakeStore:
    """Records the single ``put`` a produce call makes; ignores reads."""

    def __init__(self) -> None:
        self.puts: list[dict[str, object]] = []

    async def put(
        self,
        key: str,
        body: AsyncIterator[bytes],
        *,
        content_type: str,
        size: int | None = None,
    ) -> None:
        collected = b"".join([chunk async for chunk in body])
        self.puts.append({"key": key, "body": collected, "content_type": content_type, "size": size})

    # the remaining ObjectStore methods are unused by the producer path.
    def open_read(self, key: str) -> AsyncIterator[bytes]:  # pragma: no cover
        raise NotImplementedError

    async def delete(self, key: str) -> None:  # pragma: no cover
        raise NotImplementedError

    async def delete_many(self, keys: list[str]) -> None:  # pragma: no cover
        raise NotImplementedError

    def list_keys(self, prefix: str | None = None) -> AsyncIterator[str]:  # pragma: no cover
        raise NotImplementedError

    async def presigned_get_url(self, key: str, *, expires_in: int = 300) -> str:  # pragma: no cover
        raise NotImplementedError


async def _abytes(*chunks: bytes) -> AsyncIterator[bytes]:
    """Yield ``chunks`` as an async byte stream."""
    for chunk in chunks:
        yield chunk


def _scope(store: object | None, context: CallContext) -> ToolCallScope:
    """Build a call scope carrying ``store`` + ``context``."""
    return ToolCallScope(context=context, object_store=store)  # type: ignore[arg-type]


async def test_streams_to_store_and_returns_handle() -> None:
    """Helper streams the bytes and returns a handle describing the object."""
    store = _FakeStore()
    context = CallContext(customer_id=_CUSTOMER, conversation_id=_CONVERSATION)
    async with enter_call_scope(_scope(store, context)):
        handle = await stream_result_to_object_store(
            _abytes(b"hello ", b"world"),
            filename="scan.xml",
            content_type="application/xml",
            category="scans",
            summary="3 open ports",
            created=_CREATED,
        )
    assert len(store.puts) == 1
    put = store.puts[0]
    assert put["body"] == b"hello world"
    assert put["content_type"] == "application/xml"
    # no hint passed -> the advisory size forwarded to the store is None.
    assert put["size"] is None
    assert handle.s3_key == put["key"]
    assert handle.mime_type == "application/xml"
    # size on the handle is the ACTUAL streamed byte count ("hello world" = 11).
    assert handle.size_bytes == 11
    assert handle.summary == "3 open ports"
    # key is built from the VERIFIED customer + the conversation scope label.
    assert handle.s3_key.startswith(f"{_CUSTOMER}/conversation-{_CONVERSATION}/scans/")
    assert handle.s3_key.endswith(f"/{handle.object_id}/scan.xml")
    assert "2026/06/30" in handle.s3_key


async def test_handle_size_is_counted_not_the_hint() -> None:
    """A wrong size_hint never lands on the handle; the streamed count does."""
    store = _FakeStore()
    context = CallContext(customer_id=_CUSTOMER, conversation_id=_CONVERSATION)
    async with enter_call_scope(_scope(store, context)):
        handle = await stream_result_to_object_store(
            _abytes(b"ab", b"cde"),  # 5 bytes actually streamed
            filename="f.bin",
            content_type="application/octet-stream",
            category="evidence",
            size_hint=999,  # deliberately wrong
            created=_CREATED,
        )
    # the advisory hint is forwarded to the store verbatim ...
    assert store.puts[0]["size"] == 999
    # ... but the handle records the TRUE streamed length, not the lie.
    assert handle.size_bytes == 5
    assert store.puts[0]["body"] == b"abcde"


async def test_scope_label_prefers_engagement() -> None:
    """An engagement-bound call keys under ``engagement-<id>``."""
    store = _FakeStore()
    context = CallContext(
        customer_id=_CUSTOMER,
        conversation_id=_CONVERSATION,
        engagement_id=_ENGAGEMENT,
    )
    async with enter_call_scope(_scope(store, context)):
        handle = await stream_result_to_object_store(
            _abytes(b"x"),
            filename="f.bin",
            content_type="application/octet-stream",
            category="evidence",
            created=_CREATED,
        )
    assert handle.s3_key.startswith(f"{_CUSTOMER}/engagement-{_ENGAGEMENT}/evidence/")


async def test_scope_label_falls_back_to_agent() -> None:
    """With no engagement/conversation, the agent id is the owning scope."""
    store = _FakeStore()
    context = CallContext(customer_id=_CUSTOMER, agent_id=_AGENT)
    async with enter_call_scope(_scope(store, context)):
        handle = await stream_result_to_object_store(
            _abytes(b"x"),
            filename="f.bin",
            content_type="application/octet-stream",
            category="evidence",
            created=_CREATED,
        )
    assert handle.s3_key.startswith(f"{_CUSTOMER}/agent-{_AGENT}/evidence/")


async def test_fail_closed_outside_scope() -> None:
    """Called outside a call scope, the helper refuses (no ambient store)."""
    with pytest.raises(ProduceObjectError, match="outside a ToolServer call scope"):
        await stream_result_to_object_store(
            _abytes(b"x"),
            filename="f.bin",
            content_type="application/octet-stream",
            category="evidence",
        )


async def test_fail_closed_when_no_store_wired() -> None:
    """A scope with no object store refuses rather than dropping bytes."""
    context = CallContext(customer_id=_CUSTOMER, conversation_id=_CONVERSATION)
    async with enter_call_scope(_scope(None, context)):
        with pytest.raises(ProduceObjectError, match="no object store"):
            await stream_result_to_object_store(
                _abytes(b"x"),
                filename="f.bin",
                content_type="application/octet-stream",
                category="evidence",
            )


async def test_fail_closed_when_no_verified_customer() -> None:
    """Without a verified customer_id the helper refuses (no tenant prefix)."""
    store = _FakeStore()
    context = CallContext(conversation_id=_CONVERSATION)  # no customer_id
    async with enter_call_scope(_scope(store, context)):
        with pytest.raises(ProduceObjectError, match="no verified customer_id"):
            await stream_result_to_object_store(
                _abytes(b"x"),
                filename="f.bin",
                content_type="application/octet-stream",
                category="evidence",
            )
    assert store.puts == []


async def test_fail_closed_when_no_owning_context() -> None:
    """A verified customer but no engagement/conversation/agent still refuses."""
    store = _FakeStore()
    context = CallContext(customer_id=_CUSTOMER)
    async with enter_call_scope(_scope(store, context)):
        with pytest.raises(ProduceObjectError, match="engagement, conversation, or agent"):
            await stream_result_to_object_store(
                _abytes(b"x"),
                filename="f.bin",
                content_type="application/octet-stream",
                category="evidence",
            )
    assert store.puts == []


async def test_tool_server_wires_store_into_scope() -> None:
    """The ToolServer installs its pod-level store on every per-call scope."""
    store = _FakeStore()
    server = ToolServer(
        nats_url="nats://localhost:4222",
        namespace_collection=None,
        object_store=store,  # type: ignore[arg-type]
    )
    request = CallRequest(
        tool_name="t",
        tool_version="1.0.0",
        arguments={},
        context=CallContext(customer_id=_CUSTOMER, conversation_id=_CONVERSATION),
    )
    scope = await server._build_call_scope(request)  # noqa: SLF001 -- wiring seam: server propagates its store to the per-call scope
    assert scope.object_store is store


async def test_tool_server_default_scope_has_no_store() -> None:
    """A server wired without a store yields scopes with object_store=None."""
    server = ToolServer(nats_url="nats://localhost:4222", namespace_collection=None)
    request = CallRequest(tool_name="t", tool_version="1.0.0", arguments={})
    scope = await server._build_call_scope(request)  # noqa: SLF001 -- wiring seam: default server yields a storeless scope
    assert scope.object_store is None
