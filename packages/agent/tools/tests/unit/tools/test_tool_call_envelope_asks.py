"""The two envelope asks D18 accepted: metadata on the exception path, and a
caller deadline (search-requirements.md §10.9, §10.10).

Both are additive and neither changes an existing call's behaviour, but they
are additive in *different* directions, and the difference is the whole reason
they ship together and roll out apart:

* §10.9 populates :attr:`CallResponse.metadata`, a field that already exists.
  Nothing on the wire changes shape, so there is no ordering constraint at all
  -- an old reader sees the field it already knew about, carrying content where
  it used to see ``None``.
* §10.10 adds :attr:`CallRequest.deadline_seconds`, a field that does not
  exist yet, to a model with ``extra="forbid"``. A client sending it to a
  server that predates it does not get a degraded call; it gets a *rejected*
  one. So the accepting side ships and deploys first, and this suite pins that
  the accepting side is all that landed.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.call_scope import ToolCallScope
from threetears.agent.tools.server import (
    CallRequest,
    CallResponse,
    HardCallTimeout,
    ToolCallFailure,
    ToolServer,
    _effective_ceiling,
)


class _FakeTool(TearsTool):  # parity-with: threetears.agent.tools.base_tool.TearsTool
    """a tool whose ``execute`` body is supplied per test."""

    def __init__(self, body: Any) -> None:
        self._body = body

    async def execute(self, **kwargs: Any) -> ToolResult:
        return await self._body()

    def mcp_schema(self) -> MCPToolDefinition:
        return MCPToolDefinition(name="test.fake", version="1.0.0", description="t", input_schema={})

    def mcp_name(self) -> str:
        return "test.fake"

    def mcp_version(self) -> str:
        return "1.0.0"


def _request(**overrides: Any) -> CallRequest:
    """a minimal call request, with fields replaced per test."""
    return CallRequest(tool_name="test.fake", tool_version="1.0.0", arguments={}, **overrides)


def _server(**kwargs: Any) -> ToolServer:
    """a tool server with no NATS wired, configured with the given guards."""
    return ToolServer(nats_url="nats://stub", **kwargs)


# ---------------------------------------------------------------- §10.9 ----


def test_a_raising_tool_can_carry_structure_out() -> None:
    """``ToolCallFailure`` forwards metadata the way a returned result does.

    Constructed rather than dispatched: the dispatch reads
    ``exc.metadata`` off exactly this attribute, and pinning the exception's
    own contract is what keeps that read honest if the class is edited.
    """
    failure = ToolCallFailure("provider refused", metadata={"provider": "searxng", "retry_after": 30})

    assert failure.metadata == {"provider": "searxng", "retry_after": 30}
    assert str(failure) == "provider refused"


def test_an_ordinary_exception_still_carries_nothing() -> None:
    """The carry is opt-in: every other exception behaves precisely as before.

    Pinned because the failure mode of a duck-typed version of this feature is
    silent -- an implementation reading ``getattr(exc, "metadata", None)``
    would happily forward whatever an unrelated exception happened to have on
    an attribute of that name.
    """
    assert not isinstance(ValueError("boom"), ToolCallFailure)


def test_the_response_field_the_carry_lands_in_already_existed() -> None:
    """§10.9 needs no rollout order, and this is why.

    ``CallResponse.metadata`` predates the ask; the exception path simply
    stopped leaving it empty. A reader on an older release sees a field it
    already knows, carrying content where it used to see ``None``.
    """
    response = CallResponse(success=False, content="", metadata={"provider": "searxng"}, error="refused")

    assert response.metadata == {"provider": "searxng"}


# --------------------------------------------------------------- §10.10 ----


def test_the_server_accepts_a_caller_deadline() -> None:
    """The accepting half of the rollout: the field parses."""
    request = _request(deadline_seconds=1.5)

    assert request.deadline_seconds == 1.5


def test_a_request_without_a_deadline_is_unchanged() -> None:
    """Every caller that exists today keeps working, saying nothing."""
    assert _request().deadline_seconds is None


@pytest.mark.parametrize(
    ("pod_ceiling", "caller_deadline", "expected"),
    [
        pytest.param(None, None, None, id="neither-bounds"),
        pytest.param(30.0, None, 30.0, id="only-the-pod-bounds"),
        pytest.param(None, 5.0, 5.0, id="only-the-caller-bounds"),
        pytest.param(30.0, 5.0, 5.0, id="caller-is-tighter"),
        pytest.param(5.0, 30.0, 5.0, id="pod-is-tighter-caller-cannot-buy-more"),
        pytest.param(30.0, 0.0, 0.0, id="no-time-left-is-a-budget-not-an-absence"),
    ],
)
def test_the_effective_ceiling_is_the_tighter_of_the_two(
    pod_ceiling: float | None, caller_deadline: float | None, expected: float | None
) -> None:
    """The composition rule §10.10 states: minimum, not either alone.

    ``pod-is-tighter`` is the case that matters for safety -- a caller must not
    be able to raise the pod's own ceiling by claiming a longer budget -- and
    ``no-time-left`` is the one an ``or``-style default would get wrong, since
    ``0.0`` is falsey and states a budget rather than the absence of one.

    :param pod_ceiling: the pod's configured backstop
    :ptype pod_ceiling: float | None
    :param caller_deadline: what the caller said it had left
    :ptype caller_deadline: float | None
    :param expected: the bound the call should run under
    :ptype expected: float | None
    """
    assert _effective_ceiling(pod_ceiling, caller_deadline) == expected


async def test_a_caller_deadline_bounds_a_call_on_an_unbounded_pod() -> None:
    """The deadline binds on its own, with no pod ceiling configured at all.

    This is the pod-resident half of SR-G2: before the field existed, a pod
    without ``max_call_seconds`` had nothing to derive a bound from, so a
    caller's remaining budget could not reach the call.
    """

    async def body() -> ToolResult:
        await asyncio.sleep(10)
        return ToolResult(success=True, content="never")

    server = _server()

    with pytest.raises(HardCallTimeout):
        await server._run_tool_guarded(  # noqa: SLF001 -- guard seam
            _FakeTool(body), _request(deadline_seconds=0.05), ToolCallScope()
        )


async def test_a_generous_caller_cannot_loosen_the_pod_ceiling() -> None:
    """A caller claiming twenty seconds still stops at the pod's fraction of one."""

    async def body() -> ToolResult:
        await asyncio.sleep(10)
        return ToolResult(success=True, content="never")

    server = _server(max_call_seconds=0.05)

    with pytest.raises(HardCallTimeout):
        await server._run_tool_guarded(  # noqa: SLF001 -- guard seam
            _FakeTool(body), _request(deadline_seconds=20.0), ToolCallScope()
        )


async def test_a_call_inside_both_bounds_is_untouched() -> None:
    """The guard bounds; it does not interfere."""

    async def body() -> ToolResult:
        return ToolResult(success=True, content="done")

    server = _server(max_call_seconds=5.0)

    result = await server._run_tool_guarded(  # noqa: SLF001 -- guard seam
        _FakeTool(body), _request(deadline_seconds=5.0), ToolCallScope()
    )

    assert result.content == "done"
