"""Tests for the engagement re-authorization helper.

Covers :func:`resolve_engagement_scope`: it reads the ``engagement_id`` from the
call CONTEXT (never a tool arg), forwards the verified customer + identity_token to
the pod's engagement resolver, asserts the echoed customer matches this call's
verified customer, and refuses an empty scope. Every fail-closed path is exercised
explicitly -- an engagement-bound tool must never authorize against a missing,
cross-tenant, or empty scope.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from threetears.agent.tools.call_scope import ToolCallScope, enter_call_scope
from threetears.agent.tools.context_envelope import CallContext
from threetears.agent.tools.engagement import (
    EngagementScopeUnavailableError,
    resolve_engagement_scope,
)
from threetears.agent.tools.engagement_resolver import (
    EngagementScope,
    ResolveEngagementScopeError,
    ScopeTarget,
)
from threetears.agent.tools.server import CallRequest, ToolServer

_CUSTOMER = UUID("06a41d51-a6d5-7824-8000-29ab66754fc0")
_OTHER_CUSTOMER = UUID("06a41d51-a6d5-7824-8000-2222aaaa2222")
_ENGAGEMENT = UUID("019f1924-1a31-72d3-81b4-855415bd34ba")
_CONVERSATION = UUID("019f1900-0000-7000-8000-000000000001")
_TOKEN = "hub.identity.token.value"

_TARGETS = (ScopeTarget(target_type="ip", value="10.23.70.16", label="primary"),)


# parity-with: threetears.agent.tools.engagement_resolver.EngagementScopeResolver
class _FakeResolver:
    """Returns a fixed scope for resolve, or raises; records the call args."""

    def __init__(self, *, scope: EngagementScope | None = None, error: Exception | None = None) -> None:
        self._scope = scope
        self._error = error
        self.calls: list[tuple[UUID, UUID, str]] = []

    async def resolve(self, engagement_id: UUID, *, customer_id: UUID, identity_token: str) -> EngagementScope:
        self.calls.append((engagement_id, customer_id, identity_token))
        if self._error is not None:
            raise self._error
        assert self._scope is not None
        return self._scope


def _scope_obj(
    resolver: object | None,
    context: CallContext,
) -> ToolCallScope:
    """Build a call scope carrying ``resolver`` + ``context``."""
    return ToolCallScope(context=context, engagement_resolver=resolver)  # type: ignore[arg-type]


def _resolved(customer: UUID = _CUSTOMER, targets: tuple[ScopeTarget, ...] = _TARGETS) -> EngagementScope:
    """Build a resolved engagement scope."""
    return EngagementScope(engagement_id=_ENGAGEMENT, customer_id=customer, targets=targets)


async def test_resolve_scope_threads_context_and_returns_scope() -> None:
    """The helper passes the context engagement_id + verified customer + token to the resolver."""
    resolved = _resolved()
    resolver = _FakeResolver(scope=resolved)
    context = CallContext(
        customer_id=_CUSTOMER,
        conversation_id=_CONVERSATION,
        identity_token=_TOKEN,
        engagement_id=_ENGAGEMENT,
    )
    async with enter_call_scope(_scope_obj(resolver, context)):
        got = await resolve_engagement_scope()
    assert got is resolved
    # the engagement id came from the CONTEXT, and the verified customer + token
    # were threaded from the scope -- no tool argument sets any of them.
    assert resolver.calls == [(_ENGAGEMENT, _CUSTOMER, _TOKEN)]


async def test_fail_closed_outside_scope() -> None:
    """Called outside a call scope, the helper refuses (no ambient resolver)."""
    with pytest.raises(EngagementScopeUnavailableError, match="outside a ToolServer call scope"):
        await resolve_engagement_scope()


async def test_no_engagement_id_resolves_customer_default() -> None:
    """Without an engagement_id on the context, the helper resolves the customer's DEFAULT scope.

    Rather than refuse locally, it asks the hub to resolve the customer's single active
    engagement (resolver called with ``engagement_id=None``). The hub returns those targets
    or refuses (zero / multiple active); the fail-closed guarantee lives in the resolver +
    empty-scope path, not a local no-id refusal.
    """
    resolved = _resolved()
    resolver = _FakeResolver(scope=resolved)
    context = CallContext(customer_id=_CUSTOMER, identity_token=_TOKEN)  # no engagement_id
    async with enter_call_scope(_scope_obj(resolver, context)):
        got = await resolve_engagement_scope()
    assert got is resolved
    # the resolver was asked for the DEFAULT scope: engagement_id is None.
    assert resolver.calls == [(None, _CUSTOMER, _TOKEN)]


async def test_fail_closed_when_no_resolver_wired() -> None:
    """A scope with no resolver wired refuses rather than resolving nothing."""
    context = CallContext(customer_id=_CUSTOMER, identity_token=_TOKEN, engagement_id=_ENGAGEMENT)
    async with enter_call_scope(_scope_obj(None, context)):
        with pytest.raises(EngagementScopeUnavailableError, match="no engagement resolver"):
            await resolve_engagement_scope()


async def test_fail_closed_when_no_verified_customer() -> None:
    """Without a verified customer_id the helper refuses (and never calls out)."""
    resolver = _FakeResolver(scope=_resolved())
    context = CallContext(identity_token=_TOKEN, engagement_id=_ENGAGEMENT)  # no customer_id
    async with enter_call_scope(_scope_obj(resolver, context)):
        with pytest.raises(EngagementScopeUnavailableError, match="no verified customer_id"):
            await resolve_engagement_scope()
    assert resolver.calls == []


async def test_fail_closed_when_no_identity_token() -> None:
    """Without an identity_token the helper cannot authenticate -> refuses."""
    resolver = _FakeResolver(scope=_resolved())
    context = CallContext(customer_id=_CUSTOMER, engagement_id=_ENGAGEMENT)  # no identity_token
    async with enter_call_scope(_scope_obj(resolver, context)):
        with pytest.raises(EngagementScopeUnavailableError, match="no identity_token"):
            await resolve_engagement_scope()
    assert resolver.calls == []


async def test_propagates_resolver_error() -> None:
    """A hub rejection surfaced by the resolver propagates unchanged."""
    resolver = _FakeResolver(error=ResolveEngagementScopeError("engagement scope rejected: ENGAGEMENT_NOT_FOUND: nope"))
    context = CallContext(customer_id=_CUSTOMER, identity_token=_TOKEN, engagement_id=_ENGAGEMENT)
    async with enter_call_scope(_scope_obj(resolver, context)):
        with pytest.raises(ResolveEngagementScopeError, match="ENGAGEMENT_NOT_FOUND"):
            await resolve_engagement_scope()


async def test_refuses_echoed_customer_mismatch() -> None:
    """Defense in depth: a scope resolved against a DIFFERENT customer is refused.

    The hub derives the customer from the same token, so this never happens on a
    correct path; the assertion catches a resolver/hub bug before it authorizes.
    """
    resolver = _FakeResolver(scope=_resolved(customer=_OTHER_CUSTOMER))
    context = CallContext(customer_id=_CUSTOMER, identity_token=_TOKEN, engagement_id=_ENGAGEMENT)
    async with enter_call_scope(_scope_obj(resolver, context)):
        with pytest.raises(EngagementScopeUnavailableError, match="different customer"):
            await resolve_engagement_scope()


async def test_refuses_empty_scope() -> None:
    """An engagement with no active targets authorizes nothing -> refuse."""
    resolver = _FakeResolver(scope=_resolved(targets=()))
    context = CallContext(customer_id=_CUSTOMER, identity_token=_TOKEN, engagement_id=_ENGAGEMENT)
    async with enter_call_scope(_scope_obj(resolver, context)):
        with pytest.raises(EngagementScopeUnavailableError, match="no active authorized targets"):
            await resolve_engagement_scope()


async def test_tool_server_wires_injected_engagement_resolver_into_scope() -> None:
    """An injected engagement resolver flows onto every per-call scope (like the store)."""
    resolver = _FakeResolver(scope=None)
    server = ToolServer(
        nats_url="nats://localhost:4222",
        namespace_collection=None,
        engagement_resolver=resolver,  # type: ignore[arg-type]
    )
    request = CallRequest(
        tool_name="t",
        tool_version="1.0.0",
        arguments={},
        context=CallContext(customer_id=_CUSTOMER, engagement_id=_ENGAGEMENT),
    )
    scope = await server._build_call_scope(request)  # noqa: SLF001 -- wiring seam: server propagates its resolver to the per-call scope
    assert scope.engagement_resolver is resolver
