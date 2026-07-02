"""Tests for the pod-side hub engagement-scope resolver.

Covers :class:`HubEngagementScopeResolver`: it forwards the invoking agent's
``identity_token`` (never agent_id/session_id/customer_id) to the hub engagement
subject, returns the resolved scope carrying the echoed customer + active targets,
and fails closed on transport error, a hub error reply, or a malformed success
reply. Crucially it does NOT cache -- an engagement's target set is mutable, so
authorization is resolved fresh on every call.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from threetears.agent.tools.engagement_resolver import (
    EngagementScopeRequestModel,
    EngagementScopeResponseModel,
    HubEngagementScopeResolver,
    ResolveEngagementScopeError,
    ScopeTarget,
)
from threetears.nats import RequestError

_CUSTOMER = UUID("06a41d51-a6d5-7824-8000-29ab66754fc0")
_ENGAGEMENT = UUID("019f1924-1a31-72d3-81b4-855415bd34ba")
_TOKEN = "hub.identity.token.value"


# a recording double for threetears.nats.NatsClient; named ``_Recording*`` (not
# ``_Fake*``) following test_object_resolver.py's convention, so the fake-parity
# enforcement -- which targets Fake<Name> -- does not require a full NatsClient
# mirror for a shim that only serves .request().
class _RecordingNats:
    """Records scope requests + returns queued responses (or raises)."""

    def __init__(
        self,
        *,
        responses: object = None,
        error: Exception | None = None,
    ) -> None:
        # ``responses`` is either one response model reused every call, or a list
        # popped left-to-right so a test can script successive replies.
        self._responses = responses
        self._error = error
        self.requests: list[EngagementScopeRequestModel] = []

    async def request(self, *, subject: object, message: object, response_type: object, timeout: object) -> object:
        assert isinstance(message, EngagementScopeRequestModel)
        self.requests.append(message)
        if self._error is not None:
            raise self._error
        if isinstance(self._responses, list):
            return self._responses.pop(0)
        return self._responses


def _target(value: str = "10.23.70.16", ttype: str = "ip", label: str | None = "primary") -> ScopeTarget:
    """Build one authorized target descriptor."""
    return ScopeTarget(target_type=ttype, value=value, label=label)


def _ok(targets: list[ScopeTarget] | None = None, customer: UUID = _CUSTOMER) -> EngagementScopeResponseModel:
    """Build a success scope reply."""
    return EngagementScopeResponseModel(
        success=True,
        customer_id=customer,
        targets=targets if targets is not None else [_target()],
    )


def _err(code: str = "ENGAGEMENT_NOT_FOUND", message: str = "no engagement") -> EngagementScopeResponseModel:
    """Build an error scope reply."""
    return EngagementScopeResponseModel(success=False, error_code=code, error_message=message)


async def test_resolve_returns_scope_and_forwards_token() -> None:
    """A success reply yields a scope; the request carries the identity_token."""
    nc = _RecordingNats(
        responses=_ok(targets=[_target("10.0.0.1", "ip", "a"), _target("example.com", "hostname", None)])
    )
    resolver = HubEngagementScopeResolver(nc, request_timeout_seconds=5.0)  # type: ignore[arg-type]
    scope = await resolver.resolve(_ENGAGEMENT, customer_id=_CUSTOMER, identity_token=_TOKEN)
    assert scope.engagement_id == _ENGAGEMENT
    assert scope.customer_id == _CUSTOMER
    assert [t.value for t in scope.targets] == ["10.0.0.1", "example.com"]
    assert [t.target_type for t in scope.targets] == ["ip", "hostname"]
    assert scope.targets[0].label == "a"
    assert scope.targets[1].label is None
    # the request forwarded the identity_token as the caller proof -- no
    # self-asserted agent_id / session_id / customer_id.
    sent = nc.requests[0]
    assert sent.identity_token == _TOKEN
    assert sent.engagement_id == _ENGAGEMENT
    dumped = sent.model_dump()
    assert "agent_id" not in dumped
    assert "session_id" not in dumped
    assert "customer_id" not in dumped


async def test_resolve_is_not_cached() -> None:
    """Two resolves of the same engagement each hit the hub (no stale-auth cache).

    The key divergence from the object resolver: an engagement's target set is
    mutable, so a resolved scope is NEVER reused -- authorization is always fresh.
    """
    # only TWO scripted replies: if the resolver cached, the second call would not
    # consume the second reply (and later index errors would never surface). we
    # assert BOTH were consumed and the request count is 2.
    nc = _RecordingNats(responses=[_ok(targets=[_target("10.0.0.1")]), _ok(targets=[_target("10.0.0.2")])])
    resolver = HubEngagementScopeResolver(nc, request_timeout_seconds=5.0)  # type: ignore[arg-type]
    first = await resolver.resolve(_ENGAGEMENT, customer_id=_CUSTOMER, identity_token=_TOKEN)
    second = await resolver.resolve(_ENGAGEMENT, customer_id=_CUSTOMER, identity_token=_TOKEN)
    assert first.targets[0].value == "10.0.0.1"
    # the second resolve reflects the UPDATED scope -- a cache would have returned
    # the stale first answer.
    assert second.targets[0].value == "10.0.0.2"
    assert len(nc.requests) == 2


async def test_resolve_carries_empty_targets_through() -> None:
    """An empty target list is a valid success reply (the helper refuses it, not the resolver)."""
    nc = _RecordingNats(responses=_ok(targets=[]))
    resolver = HubEngagementScopeResolver(nc, request_timeout_seconds=5.0)  # type: ignore[arg-type]
    scope = await resolver.resolve(_ENGAGEMENT, customer_id=_CUSTOMER, identity_token=_TOKEN)
    assert scope.targets == ()


async def test_fail_closed_on_transport_error() -> None:
    """A NATS RequestError surfaces as ResolveEngagementScopeError (fail-closed)."""
    nc = _RecordingNats(error=RequestError("no responders"))
    resolver = HubEngagementScopeResolver(nc, request_timeout_seconds=5.0)  # type: ignore[arg-type]
    with pytest.raises(ResolveEngagementScopeError, match="request failed"):
        await resolver.resolve(_ENGAGEMENT, customer_id=_CUSTOMER, identity_token=_TOKEN)


async def test_fail_closed_on_hub_error_reply() -> None:
    """A hub error reply raises with the hub's reason."""
    nc = _RecordingNats(responses=_err(code="IDENTITY_UNVERIFIED", message="no session"))
    resolver = HubEngagementScopeResolver(nc, request_timeout_seconds=5.0)  # type: ignore[arg-type]
    with pytest.raises(ResolveEngagementScopeError, match="IDENTITY_UNVERIFIED"):
        await resolver.resolve(_ENGAGEMENT, customer_id=_CUSTOMER, identity_token=_TOKEN)


async def test_fail_closed_on_success_without_customer() -> None:
    """success=True but no echoed customer is malformed -> fail closed."""
    nc = _RecordingNats(responses=EngagementScopeResponseModel(success=True, customer_id=None, targets=[_target()]))
    resolver = HubEngagementScopeResolver(nc, request_timeout_seconds=5.0)  # type: ignore[arg-type]
    with pytest.raises(ResolveEngagementScopeError, match="no customer_id"):
        await resolver.resolve(_ENGAGEMENT, customer_id=_CUSTOMER, identity_token=_TOKEN)


async def test_fail_closed_on_success_without_targets() -> None:
    """success=True but a null target list is malformed -> fail closed.

    (An EMPTY list is valid data; a MISSING list is malformed -- they differ.)
    """
    nc = _RecordingNats(responses=EngagementScopeResponseModel(success=True, customer_id=_CUSTOMER, targets=None))
    resolver = HubEngagementScopeResolver(nc, request_timeout_seconds=5.0)  # type: ignore[arg-type]
    with pytest.raises(ResolveEngagementScopeError, match="no customer_id / targets"):
        await resolver.resolve(_ENGAGEMENT, customer_id=_CUSTOMER, identity_token=_TOKEN)
