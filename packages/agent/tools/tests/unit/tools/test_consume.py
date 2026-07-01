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
)
from threetears.agent.tools.context_envelope import CallContext
from threetears.media.contracts import ObjectListing

_CUSTOMER = UUID("06a41d51-a6d5-7824-8000-29ab66754fc0")
_OTHER_CUSTOMER = UUID("06a41d51-a6d5-7824-8000-2222aaaa2222")
_CONVERSATION = UUID("019f1900-0000-7000-8000-000000000001")
_OBJECT = UUID("019f1924-1a31-72d3-81b4-855415bd34ba")

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


def _scope(store: object | None, context: CallContext) -> ToolCallScope:
    """Build a call scope carrying ``store`` + ``context``."""
    return ToolCallScope(context=context, object_store=store)  # type: ignore[arg-type]


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
