"""tests for the pre-call LimitGuard seam on CallProxy (gu-task-06).

the guard is a spend gate that lives in ``CallProxy._dispatch_call`` after the pop
check and before catalog routing. it MIRRORS the ``AgentToolAuthorizer`` injection
shape (required constructor arg, async method, allow-all/deny-all doubles) but returns
a typed :class:`LimitDecision` verdict rather than a bool so the dispatcher can set the
right ``error_code``.

the money path FAILS OPEN (Fork-2): a guard that RAISES serves the call (loud warn);
only a returned ``LimitDecision(allowed=False)`` hard-denies. contrast the fail-CLOSED
identity/pop/authorizer gates.

this module also covers the second injected seam: the ``EndpointUsageEmitter`` slot,
invoked post-call fire-and-forget where both the inbound request args and the outbound
response content are in hand.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from threetears.agent.tools.context_envelope import CallContext

from threetears.nats import IncomingMessage, set_default_namespace
from threetears.registry.auth import (
    INSUFFICIENT_CREDITS,
    LIMIT_EXCEEDED,
    AllowAllLimitGuard,
    DenyAllLimitGuard,
    LimitDecision,
)
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint
from threetears.registry.proxy import ProxyCallResponse

from ._dispatch_auth import make_authed_request, make_proxy

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    """default namespace so :class:`Subjects` builders are deterministic."""
    set_default_namespace("test")


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class _DenyLimitGuard:
    """guard that always denies with a caller-chosen code."""

    def __init__(self, error_code: str) -> None:
        self._error_code = error_code

    async def check(
        self,
        agent_id: str,
        user_id: str | None,
        customer_id: str | None,
        tool_name: str,
        tool_version: str,
    ) -> LimitDecision:
        return LimitDecision(allowed=False, error_code=self._error_code)


class _RaisingLimitGuard:
    """guard whose ``check`` raises -- exercises the fail-open dispatch branch."""

    async def check(
        self,
        agent_id: str,
        user_id: str | None,
        customer_id: str | None,
        tool_name: str,
        tool_version: str,
    ) -> LimitDecision:
        raise RuntimeError("counter backend unreachable")


class _RecordingUsageEmitter:
    """usage emitter that records the (request, response) it is handed post-call."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []

    async def emit(self, request: Any, response: Any) -> None:
        self.calls.append((request, response))


class _RaisingUsageEmitter:
    """usage emitter whose ``emit`` raises -- must never affect the reply."""

    async def emit(self, request: Any, response: Any) -> None:
        raise RuntimeError("usage sink unreachable")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_entry(pod_id: str = "pod-001") -> CatalogEntry:
    """build a catalog entry with one available endpoint."""
    endpoint = ToolEndpoint(pod_id=pod_id, status="available")
    return CatalogEntry(
        tool_name="threetears.calculator",
        tool_version="1.0.0",
        full_name="threetears.calculator@1.0.0",
        description="test tool",
        input_schema={"type": "object", "properties": {}},
        endpoints=[endpoint],
    )


def _make_nats_msg(data: bytes, reply: str | None = "reply.subject") -> IncomingMessage:
    """wrapper-shaped envelope for the dispatch path."""
    return IncomingMessage(data=data, reply_subject=reply, subject="test.tools.call")


def _tool_reply(success: bool = True, content: str = "result: 4") -> bytes:
    """bytes a tool-pod endpoint would return via ``request_raw``."""
    response = ProxyCallResponse(
        success=success,
        content=content,
        context=CallContext(),
    )
    return response.model_dump_json().encode("utf-8")


class _StubNats:
    """minimal async NATS double recording forwarded calls + published replies."""

    def __init__(self, reply_bytes: bytes) -> None:
        self._reply_bytes = reply_bytes
        self.request_raw_calls: list[dict[str, Any]] = []
        self.published: list[ProxyCallResponse] = []

    async def request_raw(self, **kwargs: Any) -> bytes:
        self.request_raw_calls.append(kwargs)
        return self._reply_bytes

    async def subscribe(self, **kwargs: Any) -> None:
        return None

    async def publish_reply(self, *, reply_subject: str, message: ProxyCallResponse) -> None:
        self.published.append(message)


async def _dispatch(proxy: Any, request: Any) -> None:
    """handle one call end-to-end and drain the spawned dispatch task.

    :meth:`CallProxy.handle_call` schedules the work on a background task, so the
    test drives it and then drains via the public :meth:`CallProxy.stop` (which
    gathers all in-flight tasks) before asserting on the recorded effects.
    """
    await proxy.handle_call(_make_nats_msg(request.model_dump_json().encode("utf-8")))
    await proxy.stop()


# ---------------------------------------------------------------------------
# error-code constants
# ---------------------------------------------------------------------------


async def test_error_code_constants_exist() -> None:
    assert INSUFFICIENT_CREDITS == "INSUFFICIENT_CREDITS"
    assert LIMIT_EXCEEDED == "LIMIT_EXCEEDED"


# ---------------------------------------------------------------------------
# doubles
# ---------------------------------------------------------------------------


async def test_allow_all_guard_allows() -> None:
    decision = await AllowAllLimitGuard().check("a", None, "c", "tool", "1.0.0")
    assert decision.allowed is True
    assert decision.error_code is None


async def test_deny_all_guard_denies_with_insufficient_credits() -> None:
    decision = await DenyAllLimitGuard().check("a", None, "c", "tool", "1.0.0")
    assert decision.allowed is False
    assert decision.error_code == INSUFFICIENT_CREDITS


# ---------------------------------------------------------------------------
# deny path -- verdict hard-denies, route not reached
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", [INSUFFICIENT_CREDITS, LIMIT_EXCEEDED])
async def test_deny_verdict_rejects_without_forwarding(code: str) -> None:
    catalog = ToolCatalog()
    await catalog.register(_make_entry())
    proxy = make_proxy(catalog, namespace="test", limit_guard=_DenyLimitGuard(code))
    nc = _StubNats(_tool_reply())
    await proxy.start(nc)

    request = make_authed_request()
    await _dispatch(proxy, request)

    # the deny verdict short-circuits: the tool pod was NEVER forwarded to.
    assert nc.request_raw_calls == []
    assert len(nc.published) == 1
    response = nc.published[0]
    assert response.success is False
    assert response.error_code == code


# ---------------------------------------------------------------------------
# fail-open path -- a raising guard SERVES the call (Fork-2)
# ---------------------------------------------------------------------------


async def test_raising_guard_serves_fail_open(caplog: pytest.LogCaptureFixture) -> None:
    catalog = ToolCatalog()
    await catalog.register(_make_entry())
    proxy = make_proxy(catalog, namespace="test", limit_guard=_RaisingLimitGuard())
    nc = _StubNats(_tool_reply(success=True))
    await proxy.start(nc)

    request = make_authed_request()
    with caplog.at_level(logging.WARNING):
        await _dispatch(proxy, request)

    # a guard exception must NOT surface as a deny: the call is forwarded + served.
    assert len(nc.request_raw_calls) == 1
    assert len(nc.published) == 1
    response = nc.published[0]
    assert response.success is True
    assert response.error_code is None
    assert any("fail-open" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# usage emitter -- post-call fire-and-forget seam
# ---------------------------------------------------------------------------


async def test_usage_emitter_invoked_post_call() -> None:
    catalog = ToolCatalog()
    await catalog.register(_make_entry())
    emitter = _RecordingUsageEmitter()
    proxy = make_proxy(
        catalog,
        namespace="test",
        limit_guard=AllowAllLimitGuard(),
        usage_emitter=emitter,
    )
    nc = _StubNats(_tool_reply(success=True))
    await proxy.start(nc)

    request = make_authed_request()
    await _dispatch(proxy, request)

    assert len(emitter.calls) == 1
    emitted_request, emitted_response = emitter.calls[0]
    assert emitted_request.tool_name == request.tool_name
    assert emitted_response.success is True


async def test_usage_emit_failure_never_affects_reply(caplog: pytest.LogCaptureFixture) -> None:
    catalog = ToolCatalog()
    await catalog.register(_make_entry())
    proxy = make_proxy(
        catalog,
        namespace="test",
        limit_guard=AllowAllLimitGuard(),
        usage_emitter=_RaisingUsageEmitter(),
    )
    nc = _StubNats(_tool_reply(success=True))
    await proxy.start(nc)

    request = make_authed_request()
    with caplog.at_level(logging.WARNING):
        await _dispatch(proxy, request)

    # the emit blew up but the caller still got its successful reply.
    assert len(nc.published) == 1
    assert nc.published[0].success is True


async def test_no_emitter_is_a_no_op() -> None:
    catalog = ToolCatalog()
    await catalog.register(_make_entry())
    proxy = make_proxy(catalog, namespace="test", limit_guard=AllowAllLimitGuard())
    nc = _StubNats(_tool_reply(success=True))
    await proxy.start(nc)

    request = make_authed_request()
    await _dispatch(proxy, request)

    assert len(nc.published) == 1
    assert nc.published[0].success is True
