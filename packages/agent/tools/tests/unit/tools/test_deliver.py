"""Tests for the general deliver tool (Path-2 P2.3 first consumer).

Covers: resolve an object id -> presigned download URL, tenant-safely; and the
fail-closed refusals (missing/malformed id, hub rejection, a resolved key not
owned by the verified customer, no object store to presign with).
"""

from __future__ import annotations

from typing import AsyncIterator
from uuid import UUID

import pytest

from threetears.agent.tools.call_scope import ToolCallScope, enter_call_scope
from threetears.agent.tools.context_envelope import CallContext
from threetears.agent.tools.deliver import DeliverObjectTool
from threetears.agent.tools.object_resolver import ResolveObjectError
from threetears.media.contracts import ObjectHandle, ObjectListing

_CUSTOMER = UUID("06a41d51-a6d5-7824-8000-29ab66754fc0")
_OBJECT = UUID("019f1924-1a31-72d3-81b4-855415bd34ba")
_OWNED_KEY = f"{_CUSTOMER}/conversation-x/reports/2026/06/30/{_OBJECT}/report.md"
_FOREIGN_KEY = "06a41d51-a6d5-7824-8000-2222aaaa2222/conversation-x/reports/report.md"
_TOKEN = "hub.identity.token.value"
_URL = "https://signed.example/report.md"


# parity-with: threetears.agent.tools.object_resolver.ObjectResolver
class _FakeResolver:
    """Returns a fixed handle, or raises."""

    def __init__(self, *, handle: ObjectHandle | None = None, error: Exception | None = None) -> None:
        self._handle = handle
        self._error = error

    async def resolve(self, object_id: UUID, *, customer_id: UUID, identity_token: str) -> ObjectHandle:
        if self._error is not None:
            raise self._error
        assert self._handle is not None
        return self._handle


# parity-with: threetears.media.contracts.ObjectStore
class _FakeStore:
    """Serves presigns for the deliver path; records the calls."""

    def __init__(self, *, url: str = _URL) -> None:
        self._url = url
        self.presigned: list[tuple[str, int]] = []

    async def presigned_get_url(self, key: str, *, expires_in: int = 300) -> str:
        self.presigned.append((key, expires_in))
        return self._url

    # the remaining ObjectStore methods are unused by the deliver path.
    async def put(  # pragma: no cover
        self, key: str, body: AsyncIterator[bytes], *, content_type: str, size: int | None = None
    ) -> None:
        raise NotImplementedError

    def open_read(self, key: str) -> AsyncIterator[bytes]:  # pragma: no cover
        raise NotImplementedError

    async def delete(self, key: str) -> None:  # pragma: no cover
        raise NotImplementedError

    async def delete_many(self, keys: list[str]) -> None:  # pragma: no cover
        raise NotImplementedError

    def list_keys(self, prefix: str | None = None) -> AsyncIterator[str]:  # pragma: no cover
        raise NotImplementedError

    def list_entries(self, prefix: str | None = None) -> AsyncIterator[ObjectListing]:  # pragma: no cover
        raise NotImplementedError


def _handle(s3_key: str = _OWNED_KEY) -> ObjectHandle:
    """Build a resolved handle carrying ``s3_key``."""
    return ObjectHandle(object_id=_OBJECT, s3_key=s3_key, mime_type="text/markdown", size_bytes=42)


def _scope(*, resolver: object | None = None, store: object | None = None) -> ToolCallScope:
    """Build a call scope carrying a resolver + store + a verified identity."""
    context = CallContext(customer_id=_CUSTOMER, identity_token=_TOKEN)
    return ToolCallScope(context=context, object_resolver=resolver, object_store=store)  # type: ignore[arg-type]


async def test_delivers_presigned_url_for_owned_object() -> None:
    """A resolvable, owned object is delivered as a presigned URL."""
    store = _FakeStore()
    async with enter_call_scope(_scope(resolver=_FakeResolver(handle=_handle()), store=store)):
        result = await DeliverObjectTool().execute(object_id=str(_OBJECT))
    assert result.success is True
    assert _URL in result.content
    assert result.metadata["delivery_url"] == _URL
    assert result.metadata["object_id"] == str(_OBJECT)
    assert result.metadata["size_bytes"] == 42
    # presigned with the resolved (owned) key + the default TTL.
    assert store.presigned == [(_OWNED_KEY, 3600)]
    # no key material leaks: the raw s3_key is never in the metadata or content
    # (only the presigned URL, which is the intended deliverable to the owner).
    assert "s3_key" not in result.metadata
    assert _OWNED_KEY not in result.content


async def test_whitespace_object_id_fails_closed() -> None:
    """A whitespace-only object_id -> a clean refusal (never a lookup)."""
    async with enter_call_scope(_scope(resolver=_FakeResolver(handle=_handle()), store=_FakeStore())):
        result = await DeliverObjectTool().execute(object_id="   ")
    assert result.success is False
    assert "object_id" in (result.error or "")


async def test_missing_object_id_fails_closed() -> None:
    """No object_id -> a clean refusal."""
    async with enter_call_scope(_scope(resolver=_FakeResolver(handle=_handle()), store=_FakeStore())):
        result = await DeliverObjectTool().execute()
    assert result.success is False
    assert "object_id" in (result.error or "")


async def test_malformed_object_id_fails_closed() -> None:
    """A non-UUID object_id -> a clean refusal (never a lookup)."""
    async with enter_call_scope(_scope(resolver=_FakeResolver(handle=_handle()), store=_FakeStore())):
        result = await DeliverObjectTool().execute(object_id="not-a-uuid")
    assert result.success is False
    assert "not a valid id" in (result.error or "")


async def test_hub_rejection_fails_closed() -> None:
    """A hub resolve rejection surfaces as a refusal, not a crash."""
    resolver = _FakeResolver(error=ResolveObjectError("object resolve rejected: OBJECT_NOT_FOUND: nope"))
    async with enter_call_scope(_scope(resolver=resolver, store=_FakeStore())):
        result = await DeliverObjectTool().execute(object_id=str(_OBJECT))
    assert result.success is False
    assert "could not deliver" in (result.error or "")
    assert "OBJECT_NOT_FOUND" in (result.error or "")


async def test_cross_tenant_resolved_key_fails_closed() -> None:
    """A resolved key under a foreign prefix is refused (resolve_object DiD)."""
    resolver = _FakeResolver(handle=_handle(s3_key=_FOREIGN_KEY))
    async with enter_call_scope(_scope(resolver=resolver, store=_FakeStore())):
        result = await DeliverObjectTool().execute(object_id=str(_OBJECT))
    assert result.success is False
    assert "refused" in (result.error or "")


async def test_no_store_fails_closed() -> None:
    """Resolve succeeds, but with no object store to presign -> refusal."""
    async with enter_call_scope(_scope(resolver=_FakeResolver(handle=_handle()), store=None)):
        result = await DeliverObjectTool().execute(object_id=str(_OBJECT))
    assert result.success is False
    assert "cannot deliver" in (result.error or "")


async def test_ttl_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """DELIVER_PRESIGN_TTL_SECONDS overrides the presign validity."""
    monkeypatch.setenv("DELIVER_PRESIGN_TTL_SECONDS", "120")
    store = _FakeStore()
    async with enter_call_scope(_scope(resolver=_FakeResolver(handle=_handle()), store=store)):
        await DeliverObjectTool().execute(object_id=str(_OBJECT))
    assert store.presigned == [(_OWNED_KEY, 120)]


def test_mcp_schema_advertises_object_id() -> None:
    """The MCP schema requires object_id + uses the constructor name."""
    tool = DeliverObjectTool(name="pentest.deliver_object")
    schema = tool.mcp_schema()
    assert schema.name == "pentest.deliver_object"
    assert schema.input_schema["required"] == ["object_id"]
