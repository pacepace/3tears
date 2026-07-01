"""Tests for the consumer-side object-store helpers.

Covers the consume seam: a tool reads a stored object's bytes (or presigns a
delivery URL) through the object store that rides the per-call
:class:`ToolCallScope`, with the key asserted to belong to the VERIFIED
``customer_id``. Fail-closed paths (no scope / no store / no customer /
cross-tenant key) are exercised explicitly -- a consuming tool must never read
or deliver an object outside its tenant.
"""

from __future__ import annotations

from typing import AsyncIterator
from uuid import UUID

import pytest

from threetears.agent.tools.call_scope import ToolCallScope, enter_call_scope
from threetears.agent.tools.consume import (
    ConsumeObjectError,
    open_object_stream,
    presigned_object_url,
    resolve_object,
)
from threetears.agent.tools.context_envelope import CallContext
from threetears.agent.tools.object_resolver import ResolveObjectError
from threetears.agent.tools.server import CallRequest, ToolServer
from threetears.media.contracts import ObjectHandle, ObjectListing

_CUSTOMER = UUID("06a41d51-a6d5-7824-8000-29ab66754fc0")
_OTHER_CUSTOMER = UUID("06a41d51-a6d5-7824-8000-2222aaaa2222")
_CONVERSATION = UUID("019f1900-0000-7000-8000-000000000001")
_OBJECT = UUID("019f1924-1a31-72d3-81b4-855415bd34ba")
_TOKEN = "hub.identity.token.value"

# a key under the verified customer's prefix (scope-first layout, keys.py).
_OWNED_KEY = f"{_CUSTOMER}/conversation-{_CONVERSATION}/reports/2026/06/30/{_OBJECT}/report.md"
# the same object shape but under a DIFFERENT customer's prefix.
_FOREIGN_KEY = f"{_OTHER_CUSTOMER}/conversation-{_CONVERSATION}/reports/2026/06/30/{_OBJECT}/report.md"


# parity-with: threetears.media.contracts.ObjectStore
class _FakeReadStore:
    """Serves reads + presigns for the consume path; records the calls."""

    def __init__(
        self, *, chunks: tuple[bytes, ...] = (b"hello ", b"world"), url: str = "https://signed.example/obj"
    ) -> None:
        self._chunks = chunks
        self._url = url
        self.opened: list[str] = []
        self.presigned: list[tuple[str, int]] = []

    def open_read(self, key: str) -> AsyncIterator[bytes]:
        # record eagerly (at call), then hand back the byte stream.
        self.opened.append(key)
        return self._stream()

    async def _stream(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def presigned_get_url(self, key: str, *, expires_in: int = 300) -> str:
        self.presigned.append((key, expires_in))
        return self._url

    # the remaining ObjectStore methods are unused by the consumer path.
    async def put(  # pragma: no cover
        self, key: str, body: AsyncIterator[bytes], *, content_type: str, size: int | None = None
    ) -> None:
        raise NotImplementedError

    async def delete(self, key: str) -> None:  # pragma: no cover
        raise NotImplementedError

    async def delete_many(self, keys: list[str]) -> None:  # pragma: no cover
        raise NotImplementedError

    def list_keys(self, prefix: str | None = None) -> AsyncIterator[str]:  # pragma: no cover
        raise NotImplementedError

    def list_entries(self, prefix: str | None = None) -> AsyncIterator[ObjectListing]:  # pragma: no cover
        raise NotImplementedError


def _scope(store: object | None, context: CallContext, *, resolver: object | None = None) -> ToolCallScope:
    """Build a call scope carrying ``store`` + ``context`` (+ optional resolver)."""
    return ToolCallScope(context=context, object_store=store, object_resolver=resolver)  # type: ignore[arg-type]


# parity-with: threetears.agent.tools.object_resolver.ObjectResolver
class _FakeResolver:
    """Returns a fixed handle for resolve, or raises; records the call args."""

    def __init__(self, *, handle: ObjectHandle | None = None, error: Exception | None = None) -> None:
        self._handle = handle
        self._error = error
        self.calls: list[tuple[UUID, UUID, str]] = []

    async def resolve(self, object_id: UUID, *, customer_id: UUID, identity_token: str) -> ObjectHandle:
        self.calls.append((object_id, customer_id, identity_token))
        if self._error is not None:
            raise self._error
        assert self._handle is not None
        return self._handle


async def _collect(stream: AsyncIterator[bytes]) -> bytes:
    """Drain an async byte stream into one buffer."""
    return b"".join([chunk async for chunk in stream])


async def test_open_object_stream_yields_bytes() -> None:
    """The helper streams the stored object's bytes for an owned key."""
    store = _FakeReadStore(chunks=(b"hello ", b"world"))
    context = CallContext(customer_id=_CUSTOMER, conversation_id=_CONVERSATION)
    async with enter_call_scope(_scope(store, context)):
        collected = await _collect(open_object_stream(_OWNED_KEY))
    assert collected == b"hello world"
    assert store.opened == [_OWNED_KEY]


async def test_presigned_object_url_returns_url() -> None:
    """The deliver helper presigns an owned key and returns the URL."""
    store = _FakeReadStore(url="https://signed.example/report")
    context = CallContext(customer_id=_CUSTOMER, conversation_id=_CONVERSATION)
    async with enter_call_scope(_scope(store, context)):
        url = await presigned_object_url(_OWNED_KEY, expires_in=120)
    assert url == "https://signed.example/report"
    # the ttl is forwarded verbatim to the store.
    assert store.presigned == [(_OWNED_KEY, 120)]


async def test_presigned_url_default_ttl() -> None:
    """Omitting expires_in forwards the helper's default ttl."""
    store = _FakeReadStore()
    context = CallContext(customer_id=_CUSTOMER, conversation_id=_CONVERSATION)
    async with enter_call_scope(_scope(store, context)):
        await presigned_object_url(_OWNED_KEY)
    assert store.presigned == [(_OWNED_KEY, 300)]


async def test_cross_tenant_key_refused_eagerly_for_stream() -> None:
    """A key under a foreign customer prefix is refused -- and eagerly.

    ``open_object_stream`` validates at call time, so a cross-tenant key raises
    before the iterator is created (the store is never touched).
    """
    store = _FakeReadStore()
    context = CallContext(customer_id=_CUSTOMER, conversation_id=_CONVERSATION)
    async with enter_call_scope(_scope(store, context)):
        with pytest.raises(ConsumeObjectError, match="not owned by the verified customer"):
            open_object_stream(_FOREIGN_KEY)  # no `async for` -- must raise here
    assert store.opened == []


async def test_cross_tenant_key_refused_for_presign() -> None:
    """The deliver path applies the same tenant assertion."""
    store = _FakeReadStore()
    context = CallContext(customer_id=_CUSTOMER, conversation_id=_CONVERSATION)
    async with enter_call_scope(_scope(store, context)):
        with pytest.raises(ConsumeObjectError, match="not owned by the verified customer"):
            await presigned_object_url(_FOREIGN_KEY)
    assert store.presigned == []


async def test_fail_closed_outside_scope() -> None:
    """Called outside a call scope, the helpers refuse (no ambient store)."""
    with pytest.raises(ConsumeObjectError, match="outside a ToolServer call scope"):
        open_object_stream(_OWNED_KEY)
    with pytest.raises(ConsumeObjectError, match="outside a ToolServer call scope"):
        await presigned_object_url(_OWNED_KEY)


async def test_fail_closed_when_no_store_wired() -> None:
    """A scope with no object store refuses rather than reading nothing."""
    context = CallContext(customer_id=_CUSTOMER, conversation_id=_CONVERSATION)
    async with enter_call_scope(_scope(None, context)):
        with pytest.raises(ConsumeObjectError, match="no object store"):
            open_object_stream(_OWNED_KEY)


async def test_fail_closed_when_no_verified_customer() -> None:
    """Without a verified customer_id the helper refuses (no tenant to check)."""
    store = _FakeReadStore()
    context = CallContext(conversation_id=_CONVERSATION)  # no customer_id
    async with enter_call_scope(_scope(store, context)):
        with pytest.raises(ConsumeObjectError, match="no verified customer_id"):
            open_object_stream(_OWNED_KEY)
    assert store.opened == []


async def test_prefix_check_is_boundary_exact() -> None:
    """A key whose first segment merely starts-with the customer id is refused.

    The assertion is on the ``<customer_id>/`` prefix, not a bare
    ``startswith(customer_id)`` -- so a sibling customer whose id shares a
    textual prefix cannot slip through.
    """
    store = _FakeReadStore()
    context = CallContext(customer_id=_CUSTOMER, conversation_id=_CONVERSATION)
    # a different id that begins with the verified id's text but is its own tenant.
    look_alike_key = f"{_CUSTOMER}deadbeef/conversation-{_CONVERSATION}/reports/x/report.md"
    async with enter_call_scope(_scope(store, context)):
        with pytest.raises(ConsumeObjectError, match="not owned by the verified customer"):
            open_object_stream(look_alike_key)
    assert store.opened == []


async def test_resolve_object_returns_handle_from_resolver() -> None:
    """resolve_object passes the verified customer + identity_token to the resolver."""
    handle = ObjectHandle(object_id=_OBJECT, s3_key=_OWNED_KEY, mime_type="text/markdown", size_bytes=5)
    resolver = _FakeResolver(handle=handle)
    context = CallContext(customer_id=_CUSTOMER, conversation_id=_CONVERSATION, identity_token=_TOKEN)
    async with enter_call_scope(_scope(None, context, resolver=resolver)):
        got = await resolve_object(_OBJECT)
    assert got is handle
    # the verified customer + the identity_token are threaded from the scope.
    assert resolver.calls == [(_OBJECT, _CUSTOMER, _TOKEN)]


async def test_resolve_object_fail_closed_outside_scope() -> None:
    """Called outside a call scope, resolve_object refuses."""
    with pytest.raises(ConsumeObjectError, match="outside a ToolServer call scope"):
        await resolve_object(_OBJECT)


async def test_resolve_object_fail_closed_no_resolver() -> None:
    """A scope with no resolver wired refuses rather than resolving nothing."""
    context = CallContext(customer_id=_CUSTOMER, identity_token=_TOKEN)
    async with enter_call_scope(_scope(None, context, resolver=None)):
        with pytest.raises(ConsumeObjectError, match="no object resolver"):
            await resolve_object(_OBJECT)


async def test_resolve_object_fail_closed_no_customer() -> None:
    """Without a verified customer_id resolve_object refuses (and never calls out)."""
    resolver = _FakeResolver(handle=None)
    context = CallContext(identity_token=_TOKEN)  # no customer_id
    async with enter_call_scope(_scope(None, context, resolver=resolver)):
        with pytest.raises(ConsumeObjectError, match="no verified customer_id"):
            await resolve_object(_OBJECT)
    assert resolver.calls == []


async def test_resolve_object_fail_closed_no_identity_token() -> None:
    """Without an identity_token resolve_object cannot authenticate -> refuses."""
    resolver = _FakeResolver(handle=None)
    context = CallContext(customer_id=_CUSTOMER)  # no identity_token
    async with enter_call_scope(_scope(None, context, resolver=resolver)):
        with pytest.raises(ConsumeObjectError, match="no identity_token"):
            await resolve_object(_OBJECT)
    assert resolver.calls == []


async def test_resolve_object_propagates_resolver_error() -> None:
    """A hub rejection surfaced by the resolver propagates unchanged."""
    resolver = _FakeResolver(error=ResolveObjectError("object resolve rejected: OBJECT_NOT_FOUND: nope"))
    context = CallContext(customer_id=_CUSTOMER, identity_token=_TOKEN)
    async with enter_call_scope(_scope(None, context, resolver=resolver)):
        with pytest.raises(ResolveObjectError, match="OBJECT_NOT_FOUND"):
            await resolve_object(_OBJECT)


async def test_tool_server_wires_injected_resolver_into_scope() -> None:
    """An injected resolver flows onto every per-call scope (like the store)."""
    resolver = _FakeResolver(handle=None)
    server = ToolServer(
        nats_url="nats://localhost:4222",
        namespace_collection=None,
        object_resolver=resolver,  # type: ignore[arg-type]
    )
    request = CallRequest(
        tool_name="t",
        tool_version="1.0.0",
        arguments={},
        context=CallContext(customer_id=_CUSTOMER),
    )
    scope = await server._build_call_scope(request)  # noqa: SLF001 -- wiring seam: server propagates its resolver to the per-call scope
    assert scope.object_resolver is resolver
