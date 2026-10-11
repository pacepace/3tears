"""The FastMCP transport's hooks for a multi-tenant host, through an in-process MCP client.

What a host gets without rebuilding :meth:`CatalogueTool.over`:

- **Every refusal kind reaches the client as data**: ``{"refusal": {"kind", "status"}}`` in the
  structured content of an error result, for each public engine error a handler raises.
- **The host is resolved per call** — sync or async, after the caller — when ``host_resolver`` is given
  in place of ``host``; exactly one of the two is accepted, refused before any tool is added.
- **A resolver that fails fails the call before anything runs**: no action, no hook. One that answers
  with the wrong type is refused naming it.
- **One ``around`` hook wraps each call**: it sees the tool, the action as given, the caller and the
  outcome; what it opens before its yield is current while the action runs; a defect is thrown in at the
  yield and still propagates. It cannot change the outcome, and a hook that raises — before its yield,
  after it, or by yielding twice — is logged while the call goes on as if there were no hook. A hook that
  is not an async generator function is refused at mount.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable
from contextvars import ContextVar
from typing import Any

import pytest
from fastmcp import Client, FastMCP
from pydantic import Field

from threetears.evals.actions import Action, ActionCatalogue, ActionOutcome, Caller, Refusal, ToolSpec
from threetears.evals.kernel import (
    AdmissionRefusedError,
    ConflictError,
    EvalServiceError,
    NotFoundError,
    ProviderRefusedError,
    StorageError,
    ValidationFailedError,
)
from threetears.evals.ops import OpsHost
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.transports import fastmcp as fastmcp_transport
from threetears.evals.transports.fastmcp import CallHook, ToolBinding, ToolCall, mount_fastmcp
from packages.evals.tests.ops_support import CALLER, ops_fixture

#: Set by a hook before its yield; read by the action, to show the hook's context is the call's.
_SPAN: ContextVar[str | None] = ContextVar("_SPAN", default=None)

_LOGGER = "threetears.evals.transports.fastmcp"


class _Note(EvalBaseModel):
    """The host action's parameters."""

    note: str = Field(
        description="What to do: 'ok', 'defect', 'slow' (outlive a short call timeout), or an error class to raise."
    )


class _Noted(EvalBaseModel):
    """The host action's result: what it saw."""

    note: str
    scope_id: str
    span: str | None


#: The public errors a handler raises, by name, as the note names them.
_ERRORS: dict[str, Callable[[], EvalServiceError]] = {
    "EvalServiceError": lambda: EvalServiceError("no", status_code=418, remedy="ask the teapot"),
    "NotFoundError": lambda: NotFoundError("widget", "w-1", remedy="list them first"),
    "ConflictError": lambda: ConflictError("busy", remedy="try again"),
    "ValidationFailedError": lambda: ValidationFailedError("bad", remedy="send a good one"),
    "StorageError": lambda: StorageError("down"),
    "ProviderRefusedError": lambda: ProviderRefusedError("rate limited"),
    "AdmissionRefusedError": lambda: AdmissionRefusedError(requested=2, admitted=4, limit=5, limit_name="EVAL_MAX"),
}


class _Host:
    """A catalogue of one host action that records which host each call ran in."""

    def __init__(self) -> None:
        self.ran_in: list[OpsHost] = []

    async def handle(self, host: OpsHost, caller: Caller, params: _Note) -> _Noted:
        self.ran_in.append(host)
        if params.note == "defect":
            raise RuntimeError("a bug in the action")
        if params.note == "slow":
            await asyncio.sleep(1)
        if params.note in _ERRORS:
            raise _ERRORS[params.note]()
        return _Noted(note=params.note, scope_id=caller.scope_id, span=_SPAN.get())

    def catalogue(self) -> ActionCatalogue:
        return ActionCatalogue(
            [
                Action(
                    name="note_take",
                    summary="Take a note.",
                    workflow="Host",
                    permission="write",
                    params=_Note,
                    result=_Noted,
                    handler=self.handle,
                    render=lambda result: f"noted {result.note}",
                    example={"note": "ok"},
                )
            ]
        )


_TOOLS = (ToolSpec(name="host", description="A host's tool.", permissions=frozenset({"write"})),)


def _served(actions: _Host, **mount: Any) -> FastMCP:
    server = FastMCP("tenants")
    mount.setdefault("caller", lambda: CALLER)
    if "host_resolver" not in mount:
        mount.setdefault("host", ops_fixture().host)
    mount_fastmcp(server, actions.catalogue(), tools=_TOOLS, **mount)
    return server


async def _call(server: FastMCP, arguments: dict[str, Any]) -> tuple[str, dict[str, Any] | None, bool]:
    async with Client(server) as client:
        result = await client.call_tool("host", arguments, raise_on_error=False)
    return result.content[0].text, result.structured_content, result.is_error


# =============================================================================
# Refusals as data
# =============================================================================


@pytest.mark.parametrize("error", sorted(_ERRORS))
async def test_every_refusal_kind_round_trips_through_the_client(error: str) -> None:
    raised = _ERRORS[error]()
    text, structured, is_error = await _call(_served(_Host()), {"action": "note_take", "note": error})
    expected = Refusal.of(raised)
    assert is_error and structured == {"refusal": {"kind": expected.kind, "status": raised.status_code}}
    remedy = "" if raised.remedy is None else f"\nremedy: {raised.remedy}"
    assert text == f"refused: note_take: {raised.message}{remedy}"


# =============================================================================
# The host, per call
# =============================================================================


async def test_the_host_is_resolved_for_every_call_and_may_be_async() -> None:
    tenants = [ops_fixture().host, ops_fixture().host]
    asked: list[str] = []
    actions = _Host()

    def caller() -> Caller:
        asked.append("caller")
        return CALLER

    async def host_resolver() -> OpsHost:
        asked.append("host")
        return tenants[len(actions.ran_in) % 2]

    server = _served(actions, caller=caller, host_resolver=host_resolver)
    for _ in range(3):
        _, structured, is_error = await _call(server, {"action": "note_take", "note": "ok"})
        assert not is_error and structured is not None
    assert actions.ran_in == [tenants[0], tenants[1], tenants[0]], "each call ran in the host resolved for it"
    assert asked == ["caller", "host"] * 3, "the caller first, then the host, once per call"


async def test_a_sync_host_resolver_is_accepted() -> None:
    tenant = ops_fixture().host
    actions = _Host()
    _, _, is_error = await _call(_served(actions, host_resolver=lambda: tenant), {"action": "note_take", "note": "ok"})
    assert not is_error and actions.ran_in == [tenant]


@pytest.mark.parametrize("given", ["both", "neither"])
def test_exactly_one_of_host_and_host_resolver_is_accepted(given: str) -> None:
    server = FastMCP("tenants")
    host = ops_fixture().host
    mount: dict[str, Any] = {"host": host, "host_resolver": lambda: host} if given == "both" else {}
    with pytest.raises(ValueError, match=f"{given} of host and host_resolver were given; .* exactly one"):
        mount_fastmcp(server, _Host().catalogue(), tools=_TOOLS, caller=lambda: CALLER, **mount)


async def test_a_refused_mount_adds_no_tool() -> None:
    server = FastMCP("tenants")
    with pytest.raises(ValueError):
        mount_fastmcp(server, _Host().catalogue(), caller=lambda: CALLER)
    async with Client(server) as client:
        assert await client.list_tools() == []


# =============================================================================
# Resolver failures
# =============================================================================


def _failing() -> Any:
    raise LookupError("no such tenant")


async def _failing_async() -> Any:
    raise LookupError("no such tenant")


@pytest.mark.parametrize(
    "mount",
    [
        {"caller": _failing},
        {"caller": _failing_async},
        {"host_resolver": _failing},
        {"host_resolver": _failing_async},
        {"caller": lambda: "agent:me"},
        {"host_resolver": lambda: "tenant-a"},
    ],
    ids=["caller-raises", "async-caller-raises", "host-raises", "async-host-raises", "caller-wrong", "host-wrong"],
)
async def test_a_resolver_that_fails_fails_the_call_before_anything_runs(mount: dict[str, Any]) -> None:
    actions = _Host()
    entered: list[ToolCall] = []

    async def around(call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
        entered.append(call)
        yield

    _, structured, is_error = await _call(
        _served(actions, around=around, **mount), {"action": "note_take", "note": "ok"}
    )
    assert is_error and structured is None, "a failed resolution is the transport's error, not a refusal"
    assert actions.ran_in == [] and entered == [], "no action ran and the hook was never entered"


@pytest.mark.parametrize(
    ("mount", "says"),
    [
        ({"caller": lambda: "agent:me"}, "the caller resolver answered 'agent:me', not a Caller"),
        ({"host_resolver": lambda: "tenant-a"}, "the host resolver answered 'tenant-a', not an OpsHost"),
    ],
)
async def test_a_resolver_answering_the_wrong_type_is_named(mount: dict[str, Any], says: str) -> None:
    mounted = _Host().catalogue().mount(_TOOLS[0])
    host = {} if "host_resolver" in mount else {"host": ops_fixture().host}
    binding = ToolBinding(mounted, **({"caller": lambda: CALLER} | host | mount))
    with pytest.raises(TypeError, match=says):
        await binding.call({"action": "note_take", "note": "ok"})


# =============================================================================
# The around hook
# =============================================================================


async def test_the_hook_sees_the_call_and_its_outcome_and_wraps_the_action() -> None:
    seen: list[tuple[ToolCall, ActionOutcome]] = []

    async def around(call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
        token = _SPAN.set(f"span:{call.action}")
        try:
            outcome = yield
            seen.append((call, outcome))
        finally:
            _SPAN.reset(token)

    server = _served(_Host(), around=around)
    _, result, _ = await _call(server, {"action": "note_take", "note": "ok"})
    _, refused, _ = await _call(server, {"action": "note_take", "note": "ConflictError"})
    _, unknown, _ = await _call(server, {"action": 7})

    assert result is not None and result["span"] == "span:note_take", "the hook's context is current in the action"
    (ok_call, ok), (no_call, no), (bad_call, bad) = seen
    assert ok_call == ToolCall(tool="host", action="note_take", caller=CALLER)
    assert (ok.structured, ok.refusal, ok.action) == (result, None, "note_take")
    assert no_call.action == "note_take" and no.refusal == Refusal(kind="conflict", status=409)
    assert bad_call.action is None, "an action argument that is no string is shown as none"
    assert bad.refusal == Refusal(kind="validation_failed", status=422)
    assert refused == {"refusal": {"kind": "conflict", "status": 409}}
    assert unknown == {"refusal": {"kind": "validation_failed", "status": 422}}


async def test_a_defect_is_thrown_into_the_hook_and_still_propagates() -> None:
    saw: list[BaseException] = []
    closed: list[bool] = []

    async def around(call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
        try:
            yield
        except RuntimeError as defect:
            saw.append(defect)
            yield  # tries to swallow it and carry on
        finally:
            closed.append(True)

    binding = ToolBinding(
        _Host().catalogue().mount(_TOOLS[0]), host=ops_fixture().host, caller=lambda: CALLER, around=around
    )
    with pytest.raises(RuntimeError, match="a bug in the action"):
        await binding.call({"action": "note_take", "note": "defect"})
    assert [str(defect) for defect in saw] == ["a bug in the action"], "the hook saw the defect at its yield"
    assert closed == [True], "and was closed when it tried to carry on past it"

    _, structured, is_error = await _call(_served(_Host(), around=around), {"action": "note_take", "note": "defect"})
    assert is_error and structured is None, "through the client, a defect is the transport's error, not a refusal"


async def test_the_hook_cannot_change_the_outcome() -> None:
    async def around(call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
        outcome = yield
        yield ActionOutcome("replaced", refusal=Refusal(kind="conflict", status=409))  # type: ignore[misc]
        del outcome

    _, structured, is_error = await _call(_served(_Host(), around=around), {"action": "note_take", "note": "ok"})
    assert not is_error and structured is not None and structured["note"] == "ok"


async def _raises_before(call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
    raise ConnectionError("exporter is down")
    yield  # pragma: no cover - makes this an async generator


async def _never_yields(call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
    if call.tool:
        return
    yield  # pragma: no cover - makes this an async generator


async def _raises_after(call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
    yield
    raise ConnectionError("exporter is down")


async def _yields_twice(call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
    yield
    yield


@pytest.mark.parametrize(
    ("around", "logged"),
    [
        (_raises_before, "failed before its yield on host/note_take; the call runs unobserved"),
        (_never_yields, "failed before its yield on host/note_take; the call runs unobserved"),
        (_raises_after, "failed after its yield on host/note_take; the outcome stands"),
        (_yields_twice, "yielded twice on host/note_take; it yields once, and the outcome stands"),
    ],
    ids=["raises-before", "never-yields", "raises-after", "yields-twice"],
)
@pytest.mark.parametrize("note", ["ok", "NotFoundError"])
async def test_a_hook_that_fails_is_logged_and_the_call_goes_on(
    around: CallHook, logged: str, note: str, caplog: pytest.LogCaptureFixture
) -> None:
    actions = _Host()
    plain_text, plain, plain_error = await _call(_served(_Host()), {"action": "note_take", "note": note})
    with caplog.at_level(logging.ERROR, logger=_LOGGER):
        text, structured, is_error = await _call(_served(actions, around=around), {"action": "note_take", "note": note})
    assert (text, structured, is_error) == (plain_text, plain, plain_error), "the outcome is the one with no hook"
    assert len(actions.ran_in) == 1, "the action ran exactly once"
    assert [record.getMessage() for record in caplog.records if record.name == _LOGGER] == [f"around hook {logged}"]


def test_a_hook_that_is_no_async_generator_is_refused_at_mount() -> None:
    def around(call: ToolCall) -> None:
        return None

    server = FastMCP("tenants")
    with pytest.raises(TypeError, match="not an async generator function; write the hook as"):
        mount_fastmcp(
            server,
            _Host().catalogue(),
            tools=_TOOLS,
            host=ops_fixture().host,
            caller=lambda: CALLER,
            around=around,  # type: ignore[arg-type]
        )


# =============================================================================
# The one call timeout over the hooks
# =============================================================================


async def test_the_call_timeout_holds_a_slow_host_resolver() -> None:
    """The one timeout covers the whole call, the host's resolution with it: a slow resolver is stopped too."""
    actions = _Host()

    async def slow_host() -> OpsHost:
        await asyncio.sleep(1)
        return ops_fixture().host

    text, structured, is_error = await _call(
        _served(actions, host_resolver=slow_host, call_timeout_s=0.05), {"action": "note_take", "note": "ok"}
    )
    assert is_error and structured is None and "outlived this server's 0.05s call timeout" in text
    assert actions.ran_in == [], "no action ran"


async def test_a_call_the_timeout_stops_closes_its_hook() -> None:
    """A call stopped by the timeout is closed in the hook at its yield, so the hook's ``finally`` runs."""
    seen: list[str] = []

    async def around(call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
        seen.append("entered")
        try:
            yield
        finally:
            seen.append("closed")

    text, _, is_error = await _call(
        _served(_Host(), around=around, call_timeout_s=0.05), {"action": "note_take", "note": "slow"}
    )
    assert is_error and "call timeout" in text
    assert seen == ["entered", "closed"]


async def test_a_slow_hook_after_the_action_never_turns_its_outcome_into_a_timeout() -> None:
    """A slow audit flush after the yield read as the call outliving its timeout, after the work was done and paid."""
    actions = _Host()
    flushed: list[str] = []

    async def around(call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
        outcome = yield
        await asyncio.sleep(0.3)
        flushed.append(outcome.text)

    text, structured, is_error = await _call(
        _served(actions, around=around, call_timeout_s=0.1), {"action": "note_take", "note": "ok"}
    )
    assert not is_error and structured is not None and structured["note"] == "ok", text
    assert len(actions.ran_in) == 1 and flushed == ["noted ok"]


async def test_a_hook_overrunning_after_the_action_is_cut_short_and_logged_and_the_outcome_stands(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(fastmcp_transport, "HOOK_TAIL_TIMEOUT_S", 0.05)

    async def around(call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
        yield
        await asyncio.sleep(5)

    with caplog.at_level(logging.ERROR, logger=_LOGGER):
        _, structured, is_error = await _call(_served(_Host(), around=around), {"action": "note_take", "note": "ok"})
    assert not is_error and structured is not None and structured["note"] == "ok"
    assert [record.getMessage() for record in caplog.records if record.name == _LOGGER] == [
        "around hook ran past its 0.05s after its yield on host/note_take and was cut short; the outcome stands"
    ]


async def test_a_hook_that_will_not_close_does_not_hold_a_stopped_call_open(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(fastmcp_transport, "HOOK_TAIL_TIMEOUT_S", 0.05)

    async def around(call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
        try:
            yield
        finally:
            await asyncio.sleep(5)

    started = asyncio.get_running_loop().time()
    with caplog.at_level(logging.ERROR, logger=_LOGGER):
        text, _, is_error = await _call(
            _served(_Host(), around=around, call_timeout_s=0.05), {"action": "note_take", "note": "slow"}
        )
    assert is_error and "call timeout" in text
    assert asyncio.get_running_loop().time() - started < 2, "the stopped call waited on its hook"
    assert "around hook ran past its 0.05s closing host/note_take and was cut short" in [
        record.getMessage() for record in caplog.records if record.name == _LOGGER
    ]
