"""Tests for the pod-side hub object resolver.

Covers :class:`HubObjectResolver`: it forwards the invoking agent's
``identity_token`` (never agent_id/session_id) to the hub resolve subject,
returns a handle carrying the resolved key, caches immutable mappings under the
VERIFIED ``(customer_id, object_id)``, and fails closed on transport error, a
hub error reply, or a success reply missing the key.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from threetears.agent.tools.object_resolver import (
    HubObjectResolver,
    ObjectResolveRequestModel,
    ObjectResolveResponseModel,
    ResolveObjectError,
)
from threetears.nats import RequestError

_CUSTOMER = UUID("06a41d51-a6d5-7824-8000-29ab66754fc0")
_OTHER_CUSTOMER = UUID("06a41d51-a6d5-7824-8000-2222aaaa2222")
_OBJECT = UUID("019f1924-1a31-72d3-81b4-855415bd34ba")
_OBJECT2 = UUID("019f1924-1a31-72d3-81b4-999999999999")
_KEY = f"{_CUSTOMER}/conversation-x/reports/2026/06/30/{_OBJECT}/report.md"
_TOKEN = "hub.identity.token.value"


# a recording double for threetears.nats.NatsClient; named ``_Recording*`` (not
# ``_Fake*``) following test_tool_server.py's _RecordingNatsClient, so the
# fake-parity enforcement -- which targets Fake<Name> -- does not require a
# full NatsClient mirror for a shim that only serves .request().
class _RecordingNats:
    """Records resolve requests + returns queued responses (or raises)."""

    def __init__(
        self,
        *,
        responses: object = None,
        error: Exception | None = None,
    ) -> None:
        # ``responses`` is either one response model reused every call, or a
        # list popped left-to-right so a test can script successive replies.
        self._responses = responses
        self._error = error
        self.requests: list[ObjectResolveRequestModel] = []

    async def request(self, *, subject: object, message: object, response_type: object, timeout: object) -> object:
        assert isinstance(message, ObjectResolveRequestModel)
        self.requests.append(message)
        if self._error is not None:
            raise self._error
        if isinstance(self._responses, list):
            return self._responses.pop(0)
        return self._responses


def _ok(s3_key: str = _KEY, mime: str = "text/markdown", size: int = 42) -> ObjectResolveResponseModel:
    """Build a success resolve reply."""
    return ObjectResolveResponseModel(success=True, s3_key=s3_key, mime_type=mime, size_bytes=size)


def _err(code: str = "OBJECT_NOT_FOUND", message: str = "no object") -> ObjectResolveResponseModel:
    """Build an error resolve reply."""
    return ObjectResolveResponseModel(success=False, error_code=code, error_message=message)


async def test_resolve_returns_handle_and_forwards_token() -> None:
    """A success reply yields a handle; the request carries the identity_token."""
    nc = _RecordingNats(responses=_ok())
    resolver = HubObjectResolver(nc, request_timeout_seconds=5.0)  # type: ignore[arg-type]
    handle = await resolver.resolve(_OBJECT, customer_id=_CUSTOMER, identity_token=_TOKEN)
    assert handle.object_id == _OBJECT
    assert handle.s3_key == _KEY
    assert handle.mime_type == "text/markdown"
    assert handle.size_bytes == 42
    # resolve returns only storage-locating fields.
    assert handle.summary is None
    assert handle.category is None
    # the request forwarded the identity_token as the caller proof -- no
    # self-asserted agent_id / session_id / customer_id.
    sent = nc.requests[0]
    assert sent.identity_token == _TOKEN
    assert sent.object_id == _OBJECT
    assert "agent_id" not in sent.model_dump()
    assert "customer_id" not in sent.model_dump()


async def test_resolve_caches_immutable_mapping() -> None:
    """A second resolve of the same (customer, object) is served from cache."""
    nc = _RecordingNats(responses=[_ok()])  # exactly ONE reply available
    resolver = HubObjectResolver(nc, request_timeout_seconds=5.0)  # type: ignore[arg-type]
    first = await resolver.resolve(_OBJECT, customer_id=_CUSTOMER, identity_token=_TOKEN)
    second = await resolver.resolve(_OBJECT, customer_id=_CUSTOMER, identity_token=_TOKEN)
    assert first is second
    assert len(nc.requests) == 1  # the second call never reached the hub


async def test_cache_is_tenant_scoped() -> None:
    """The same object id under a different customer is a distinct entry."""
    own_key = f"{_CUSTOMER}/a/report.md"
    foreign_key = f"{_OTHER_CUSTOMER}/a/report.md"
    nc = _RecordingNats(responses=[_ok(s3_key=own_key), _ok(s3_key=foreign_key)])
    resolver = HubObjectResolver(nc, request_timeout_seconds=5.0)  # type: ignore[arg-type]
    mine = await resolver.resolve(_OBJECT, customer_id=_CUSTOMER, identity_token=_TOKEN)
    theirs = await resolver.resolve(_OBJECT, customer_id=_OTHER_CUSTOMER, identity_token=_TOKEN)
    assert mine.s3_key == own_key
    assert theirs.s3_key == foreign_key
    assert len(nc.requests) == 2


async def test_cache_evicts_oldest_when_full() -> None:
    """With cache_max=1, a new mapping evicts the previous one (FIFO)."""
    nc = _RecordingNats(
        responses=[
            _ok(s3_key=f"{_CUSTOMER}/one"),
            _ok(s3_key=f"{_CUSTOMER}/two"),
            _ok(s3_key=f"{_CUSTOMER}/one-again"),
        ]
    )
    resolver = HubObjectResolver(nc, request_timeout_seconds=5.0, cache_max=1)  # type: ignore[arg-type]
    await resolver.resolve(_OBJECT, customer_id=_CUSTOMER, identity_token=_TOKEN)  # caches OBJECT
    await resolver.resolve(_OBJECT2, customer_id=_CUSTOMER, identity_token=_TOKEN)  # evicts OBJECT
    # OBJECT was evicted, so re-resolving it hits the hub again (3rd reply).
    again = await resolver.resolve(_OBJECT, customer_id=_CUSTOMER, identity_token=_TOKEN)
    assert again.s3_key == f"{_CUSTOMER}/one-again"
    assert len(nc.requests) == 3


async def test_fail_closed_on_transport_error() -> None:
    """A NATS RequestError surfaces as ResolveObjectError (fail-closed)."""
    nc = _RecordingNats(error=RequestError("no responders"))
    resolver = HubObjectResolver(nc, request_timeout_seconds=5.0)  # type: ignore[arg-type]
    with pytest.raises(ResolveObjectError, match="request failed"):
        await resolver.resolve(_OBJECT, customer_id=_CUSTOMER, identity_token=_TOKEN)


async def test_fail_closed_on_hub_error_reply() -> None:
    """A hub error reply raises with the hub's reason and caches nothing."""
    nc = _RecordingNats(responses=[_err(code="IDENTITY_UNVERIFIED", message="no session"), _ok()])
    resolver = HubObjectResolver(nc, request_timeout_seconds=5.0)  # type: ignore[arg-type]
    with pytest.raises(ResolveObjectError, match="IDENTITY_UNVERIFIED"):
        await resolver.resolve(_OBJECT, customer_id=_CUSTOMER, identity_token=_TOKEN)
    # the rejection was NOT cached: a later resolve retries the hub + succeeds.
    handle = await resolver.resolve(_OBJECT, customer_id=_CUSTOMER, identity_token=_TOKEN)
    assert handle.s3_key == _KEY
    assert len(nc.requests) == 2


async def test_fail_closed_on_success_without_key() -> None:
    """success=True but no s3_key is malformed -> fail closed, not a bad read."""
    nc = _RecordingNats(responses=ObjectResolveResponseModel(success=True, s3_key=None))
    resolver = HubObjectResolver(nc, request_timeout_seconds=5.0)  # type: ignore[arg-type]
    with pytest.raises(ResolveObjectError, match="no s3_key"):
        await resolver.resolve(_OBJECT, customer_id=_CUSTOMER, identity_token=_TOKEN)
