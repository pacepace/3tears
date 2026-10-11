"""The FastMCP transport's hooks for a multi-tenant host, through an in-process MCP client.

What a host gets without rebuilding :meth:`CatalogueTool.over`:

- **Every refusal kind reaches the client as data**: ``{"refusal": {"kind", "status"}}`` in the
  structured content of an error result, for each public engine error a handler raises.
- **The host is resolved per call** — sync or async, after the caller — when ``host_resolver`` is given
  in place of ``host``; exactly one of the two is accepted, refused before any tool is added.
- **A resolver that fails fails the call before anything runs**: no action, no hook. One that answers
  with the wrong type is refused naming it. **One that refuses** — raises an engine error, a host's own
  subclass among them — answers the call as a refusal with that error's kind, status and remedy, read as an
  action's refusal is, before any action runs. A host resolver's refusal enters the ``around`` hook, which is
  handed it as its outcome; a caller resolver's enters none, since there is no call without a caller.
- **One ``around`` hook wraps each call**: it sees the tool, the action as given, the caller and the
  outcome; what it opens before its yield is current while the action runs; a defect is thrown in at the
  yield and still propagates. It cannot change the outcome, and a hook that raises — before its yield,
  after it, or by yielding twice — is logged while the call goes on as if there were no hook. A hook that
  is not an async generator function is refused at mount. What a hook sets in a context variable before
  its yield is the action's, and its tail undoes it in the call's own task.
- **The call timeout may be a function**, read once as each call starts; the value a call read is the one
  it is held to, the one its timeout reports and the one its ``ToolCall`` carries, and a value that is not positive
  fails that call naming it. A call that outlives it has ``CallTimedOut`` thrown in at its hook's yield; a call
  cancelled from outside closes the hook with nothing thrown, so the hook tells the two apart.
- **One ``annotate`` hook adds a note** as a text block after the outcome's, changing nothing else; one that
  raises, answers no string or overruns is logged, and the call returns as if there were no hook.
- **The host resolver hands per-call state to both hooks**: answering a ``HostResolution`` puts its ``context`` on
  the ``ToolCall`` beside the resolved ``host``; a bare host still works, with no context; a host resolver's refusal
  shows both as ``None``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import AsyncGenerator, Callable, Iterator
from contextlib import contextmanager
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
from threetears.evals.transports.fastmcp import (
    CallAnnotator,
    CallHook,
    DEFAULT_CALL_TIMEOUT_S,
    CallTimedOut,
    HostResolution,
    ToolBinding,
    ToolCall,
    mount_fastmcp,
)
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


async def _blocks(server: FastMCP, arguments: dict[str, Any]) -> tuple[list[str], dict[str, Any] | None, bool]:
    """Every text block a call returns, in order, with its structured content and error flag."""
    async with Client(server) as client:
        result = await client.call_tool("host", arguments, raise_on_error=False)
    return [block.text for block in result.content], result.structured_content, result.is_error


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

    host = ops_fixture().host
    server = _served(_Host(), around=around, host=host)
    _, result, _ = await _call(server, {"action": "note_take", "note": "ok"})
    _, refused, _ = await _call(server, {"action": "note_take", "note": "ConflictError"})
    _, unknown, _ = await _call(server, {"action": 7})

    assert result is not None and result["span"] == "span:note_take", "the hook's context is current in the action"
    (ok_call, ok), (no_call, no), (bad_call, bad) = seen
    assert ok_call == ToolCall(
        tool="host", action="note_take", caller=CALLER, host=host, timeout_s=DEFAULT_CALL_TIMEOUT_S
    )
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


async def test_a_call_the_timeout_stops_throws_call_timed_out_into_its_hook() -> None:
    """A stopped call is finished in the hook at its yield, as ``CallTimedOut``, so it tells a timeout from a cancel."""
    seen: list[Any] = []

    async def around(call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
        seen.append(("entered", call.timeout_s))
        try:
            yield
        except CallTimedOut as stopped:
            seen.append(("timed out", stopped.timeout_s))
            raise
        finally:
            seen.append("closed")

    text, _, is_error = await _call(
        _served(_Host(), around=around, call_timeout_s=0.05), {"action": "note_take", "note": "slow"}
    )
    assert is_error and "call timeout" in text
    assert seen == [("entered", 0.05), ("timed out", 0.05), "closed"]


async def test_a_hook_that_swallows_call_timed_out_leaves_the_call_timed_out() -> None:
    caught: list[float] = []

    async def around(call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
        try:
            yield
        except CallTimedOut as stopped:
            caught.append(stopped.timeout_s)

    binding = ToolBinding(
        _Host().catalogue().mount(_TOOLS[0]),
        host=ops_fixture().host,
        caller=lambda: CALLER,
        around=around,
        call_timeout_s=lambda: 0.05,
    )
    with pytest.raises(CallTimedOut) as stopped:
        await binding.call({"action": "note_take", "note": "slow"})
    assert stopped.value.timeout_s == 0.05 and caught == [0.05], "the hook saw it, and the call still timed out"


async def test_a_call_cancelled_from_outside_closes_its_hook_with_nothing_thrown() -> None:
    """The client giving up, or the process stopping, is no timeout, and the hook is not told it is one."""
    seen: list[Any] = []
    entered = asyncio.Event()

    async def around(call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
        entered.set()
        try:
            yield
        except BaseException as finished:
            seen.append(type(finished))
            raise

    binding = ToolBinding(
        _Host().catalogue().mount(_TOOLS[0]), host=ops_fixture().host, caller=lambda: CALLER, around=around
    )
    call = asyncio.create_task(binding.call({"action": "note_take", "note": "slow"}))
    await entered.wait()
    await asyncio.sleep(0)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    assert seen == [GeneratorExit], "closed at its yield, never handed a CallTimedOut"


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


# =============================================================================
# A resolver's refusal
# =============================================================================


class _Booting(EvalServiceError):
    """A host's own refusal: it is still booting, and says when to come back."""

    def __init__(self) -> None:
        super().__init__("this host is still booting", status_code=503, remedy="call again in a minute")


def _booting() -> Any:
    raise _Booting()


async def _booting_async() -> Any:
    raise _Booting()


@pytest.mark.parametrize(
    "mount",
    [{"host_resolver": _booting}, {"host_resolver": _booting_async}, {"caller": _booting}, {"caller": _booting_async}],
    ids=["host", "async-host", "caller", "async-caller"],
)
async def test_a_resolver_that_refuses_answers_a_refusal_before_any_action_runs(mount: dict[str, Any]) -> None:
    """A host booting answered a 503 with its remedy, read as an action's refusal is — not a defect with no remedy."""
    actions = _Host()
    entered: list[ToolCall] = []

    async def around(call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
        entered.append(call)
        yield

    text, structured, is_error = await _call(
        _served(actions, around=around, **mount), {"action": "note_take", "note": "ok"}
    )
    assert is_error and structured == {"refusal": {"kind": "refused", "status": 503}}
    assert text == "refused: note_take: this host is still booting\nremedy: call again in a minute"
    assert actions.ran_in == [], "no action ran"
    if "caller" in mount:
        assert entered == [], "with no caller there is no ToolCall, so the hook is never entered"
    else:
        assert [call.action for call in entered] == ["note_take"], "the caller is known, so the hook is entered"


async def test_a_host_resolver_s_refusal_is_the_around_hook_s_outcome() -> None:
    """Discodon audits "no host could serve this call" from the hook that writes every other call's line."""
    seen: list[tuple[str, ToolCall, ActionOutcome | None]] = []

    async def around(call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
        seen.append(("entered", call, None))
        outcome = yield
        seen.append(("finished", call, outcome))

    binding = ToolBinding(
        _Host().catalogue().mount(_TOOLS[0]), host_resolver=_booting, caller=lambda: CALLER, around=around
    )
    outcome = await binding.call({"action": "note_take", "note": "ok"})
    assert outcome.refusal == Refusal(kind="refused", status=503)
    assert [(step, call.action) for step, call, _ in seen] == [("entered", "note_take"), ("finished", "note_take")]
    assert seen[1][2] is outcome, "the hook is handed the refusal exactly as the call returns it"


async def test_a_host_resolver_s_refusal_survives_a_hook_that_fails(caplog: pytest.LogCaptureFixture) -> None:
    for around in (_raises_before, _raises_after):
        binding = ToolBinding(
            _Host().catalogue().mount(_TOOLS[0]), host_resolver=_booting, caller=lambda: CALLER, around=around
        )
        with caplog.at_level(logging.ERROR, logger=_LOGGER):
            outcome = await binding.call({"action": "note_take", "note": "ok"})
        assert outcome.refusal == Refusal(kind="refused", status=503), "the refusal stands whatever the hook does"


async def test_a_resolver_refusal_is_the_outcome_the_binding_returns() -> None:
    mounted = _Host().catalogue().mount(_TOOLS[0])
    binding = ToolBinding(mounted, host_resolver=_booting, caller=lambda: CALLER)

    named = await binding.call({"action": "note_take", "note": "ok"})
    assert (named.refusal, named.action, named.structured) == (Refusal(kind="refused", status=503), "note_take", None)

    unnamed = await binding.call({"action": "no_such", "note": "ok"})
    assert unnamed.action is None, "the action was never checked, so an outcome names only one the tool has"
    assert unnamed.text == "refused: host: this host is still booting\nremedy: call again in a minute"


async def test_a_resolver_refusal_of_a_public_class_keeps_that_class_s_kind() -> None:
    def suspended() -> OpsHost:
        raise ConflictError("this tenant is suspended", remedy="ask its owner to reinstate it")

    _, structured, is_error = await _call(_served(_Host(), host_resolver=suspended), {"action": "note_take"})
    assert is_error and structured == {"refusal": {"kind": "conflict", "status": 409}}


# =============================================================================
# The call timeout, read per call
# =============================================================================


async def test_a_call_timeout_function_is_read_once_as_each_call_starts() -> None:
    reads: list[float] = []
    settings = iter([30.0, 0.05])

    def call_timeout() -> float:
        reads.append(next(settings))
        return reads[-1]

    server = _served(_Host(), call_timeout_s=call_timeout)
    assert reads == [], "nothing is read at mount"
    _, structured, is_error = await _call(server, {"action": "note_take", "note": "slow"})
    assert not is_error and structured is not None, "the first call read 30s and outlived nothing"
    text, _, is_error = await _call(server, {"action": "note_take", "note": "slow"})
    assert is_error and "outlived this server's 0.05s call timeout" in text, "the second read the retuned value"
    assert reads == [30.0, 0.05], "once per call"


async def test_the_tool_call_carries_the_timeout_this_call_read() -> None:
    settings = iter([30.0, 12.5])
    seen: list[float] = []

    async def around(call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
        seen.append(call.timeout_s)
        yield

    def annotate(call: ToolCall, outcome: ActionOutcome) -> None:
        seen.append(call.timeout_s)

    server = _served(_Host(), around=around, annotate=annotate, call_timeout_s=lambda: next(settings))
    for _ in range(2):
        await _call(server, {"action": "note_take", "note": "ok"})
    _, _, _ = await _call(_served(_Host(), around=around), {"action": "note_take", "note": "ok"})
    assert seen == [30.0, 30.0, 12.5, 12.5, DEFAULT_CALL_TIMEOUT_S], "the value each call read, to both hooks"


async def test_a_call_timed_out_reports_the_timeout_that_call_read() -> None:
    mounted = _Host().catalogue().mount(_TOOLS[0])
    binding = ToolBinding(mounted, host=ops_fixture().host, caller=lambda: CALLER, call_timeout_s=lambda: 0.04)
    with pytest.raises(CallTimedOut) as stopped:
        await binding.call({"action": "note_take", "note": "slow"})
    assert stopped.value.timeout_s == 0.04


@pytest.mark.parametrize("read", [0, -1.0, float("nan"), "60", True, None])
async def test_a_call_timeout_function_answering_no_positive_number_fails_the_call_naming_it(read: Any) -> None:
    actions = _Host()
    binding = ToolBinding(
        actions.catalogue().mount(_TOOLS[0]),
        host=ops_fixture().host,
        caller=lambda: CALLER,
        call_timeout_s=lambda: read,
    )
    with pytest.raises(ValueError, match=f"the call_timeout_s function answered {read!r}, not a positive number"):
        await binding.call({"action": "note_take", "note": "ok"})
    assert actions.ran_in == [], "no call is run under a timeout that is no timeout"


# =============================================================================
# The annotate hook
# =============================================================================


async def test_a_note_is_a_block_after_the_outcome_s_and_changes_nothing_else() -> None:
    seen: list[tuple[ToolCall, ActionOutcome]] = []

    def annotate(call: ToolCall, outcome: ActionOutcome) -> str | None:
        seen.append((call, outcome))
        return "the pricing catalogue is stale, so every model is unpriced" if outcome.refusal else None

    host = ops_fixture().host
    server = _served(_Host(), annotate=annotate, host=host)
    for note in ("ok", "AdmissionRefusedError"):
        plain = await _blocks(_served(_Host()), {"action": "note_take", "note": note})
        blocks, structured, is_error = await _blocks(server, {"action": "note_take", "note": note})
        assert (blocks[:1], structured, is_error) == plain, "the outcome's own block, data and flag are untouched"
        assert blocks[1:] == ([] if note == "ok" else ["the pricing catalogue is stale, so every model is unpriced"])
    assert [(call.action, outcome.is_error) for call, outcome in seen] == [("note_take", False), ("note_take", True)]
    assert seen[0][0] == ToolCall(
        tool="host", action="note_take", caller=CALLER, host=host, timeout_s=DEFAULT_CALL_TIMEOUT_S
    )


async def test_an_async_annotator_and_an_empty_note() -> None:
    async def annotate(call: ToolCall, outcome: ActionOutcome) -> str | None:
        return "" if outcome.refusal else "noted by the host"

    blocks, _, _ = await _blocks(_served(_Host(), annotate=annotate), {"action": "note_take", "note": "ok"})
    assert blocks == ["noted ok", "noted by the host"]
    blocks, _, _ = await _blocks(_served(_Host(), annotate=annotate), {"action": "note_take", "note": "ConflictError"})
    assert len(blocks) == 1, "an empty note adds nothing"


async def test_a_note_follows_a_host_resolver_s_refusal_but_not_a_caller_resolver_s() -> None:
    def annotate(call: ToolCall, outcome: ActionOutcome) -> str:
        return f"host note on {call.action}"

    blocks, structured, _ = await _blocks(
        _served(_Host(), host_resolver=_booting, annotate=annotate), {"action": "note_take", "note": "ok"}
    )
    assert structured == {"refusal": {"kind": "refused", "status": 503}}
    assert blocks[1:] == ["host note on note_take"]
    blocks, _, _ = await _blocks(_served(_Host(), caller=_booting, annotate=annotate), {"action": "note_take"})
    assert len(blocks) == 1, "with no caller there is no ToolCall to annotate"


def _annotate_raises(call: ToolCall, outcome: ActionOutcome) -> str:
    raise ConnectionError("pricing service is down")


def _annotate_answers_no_string(call: ToolCall, outcome: ActionOutcome) -> str:
    return 7  # type: ignore[return-value]


@pytest.mark.parametrize(
    ("annotate", "logged"),
    [
        (_annotate_raises, "annotate hook failed on host/note_take; the outcome stands with no note"),
        (
            _annotate_answers_no_string,
            "annotate hook answered 7 on host/note_take, not a string or None; the outcome stands with no note",
        ),
    ],
    ids=["raises", "no-string"],
)
async def test_an_annotator_that_fails_is_logged_and_the_call_returns_as_if_there_were_none(
    annotate: CallAnnotator, logged: str, caplog: pytest.LogCaptureFixture
) -> None:
    actions = _Host()
    plain = await _blocks(_served(_Host()), {"action": "note_take", "note": "ok"})
    with caplog.at_level(logging.ERROR, logger=_LOGGER):
        answered = await _blocks(_served(actions, annotate=annotate), {"action": "note_take", "note": "ok"})
    assert answered == plain and len(actions.ran_in) == 1
    assert [record.getMessage() for record in caplog.records if record.name == _LOGGER] == [logged]


async def test_an_annotator_overrunning_is_cut_short_and_the_outcome_stands(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(fastmcp_transport, "HOOK_TAIL_TIMEOUT_S", 0.05)

    async def annotate(call: ToolCall, outcome: ActionOutcome) -> str:
        await asyncio.sleep(5)
        return "too late"

    with caplog.at_level(logging.ERROR, logger=_LOGGER):
        blocks, structured, is_error = await _blocks(
            _served(_Host(), annotate=annotate, call_timeout_s=0.05), {"action": "note_take", "note": "ok"}
        )
    assert blocks == ["noted ok"] and not is_error and structured is not None, "nor is it on the call's clock"
    assert [record.getMessage() for record in caplog.records if record.name == _LOGGER] == [
        "annotate hook ran past its 0.05s on host/note_take and was cut short; the outcome stands with no note"
    ]


# =============================================================================
# What a hook sets in context
# =============================================================================


@contextmanager
def _span(name: str) -> Iterator[None]:
    token = _SPAN.set(name)
    try:
        yield
    finally:
        _SPAN.reset(token)


async def test_what_a_hook_sets_before_its_yield_is_the_action_s_and_its_tail_undoes_it_in_the_call_s_task(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A ``ContextVar.reset`` from another task raises; the hook's tail resetting cleanly is the same-task proof."""

    async def by_token(call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
        token = _SPAN.set("by-token")
        yield
        _SPAN.reset(token)

    async def by_with(call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
        with _span("by-with"):
            yield

    mounted = _Host().catalogue().mount(_TOOLS[0])
    with caplog.at_level(logging.ERROR, logger=_LOGGER):
        for around, expected in ((by_token, "by-token"), (by_with, "by-with")):
            binding = ToolBinding(mounted, host=ops_fixture().host, caller=lambda: CALLER, around=around)
            outcome = await binding.call({"action": "note_take", "note": "ok"})
            assert outcome.structured is not None and outcome.structured["span"] == expected
            assert _SPAN.get() is None, "the tail undid it in the task that called"
    assert [record for record in caplog.records if record.name == _LOGGER] == [], "no reset failed"


# =============================================================================
# State the host resolver hands the hooks
# =============================================================================


class _Seen:
    """The ToolCall each hook was handed, in the order they were handed it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ToolCall]] = []

    async def around(self, call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
        self.calls.append(("around", call))
        yield

    def annotate(self, call: ToolCall, outcome: ActionOutcome) -> None:
        self.calls.append(("annotate", call))


async def test_a_host_resolution_s_context_reaches_both_hooks_beside_its_host() -> None:
    """Discodon passed this through two ContextVars its resolver reset on every call."""
    tenants = {"tenant-a": ops_fixture().host, "tenant-b": ops_fixture().host}
    asked = iter(["tenant-a", "tenant-b"])
    actions = _Host()
    seen = _Seen()

    async def host_resolver() -> HostResolution:
        tenant = next(asked)
        return HostResolution(tenants[tenant], context={"tenant": tenant})

    server = _served(actions, host_resolver=host_resolver, around=seen.around, annotate=seen.annotate)
    for _ in range(2):
        _, _, is_error = await _call(server, {"action": "note_take", "note": "ok"})
        assert not is_error
    assert actions.ran_in == [tenants["tenant-a"], tenants["tenant-b"]], "the action ran in the resolution's host"
    assert [(hook, call.host, call.context) for hook, call in seen.calls] == [
        ("around", tenants["tenant-a"], {"tenant": "tenant-a"}),
        ("annotate", tenants["tenant-a"], {"tenant": "tenant-a"}),
        ("around", tenants["tenant-b"], {"tenant": "tenant-b"}),
        ("annotate", tenants["tenant-b"], {"tenant": "tenant-b"}),
    ]
    assert seen.calls[0][1] is seen.calls[1][1], "both hooks of one call hold the one ToolCall"


@pytest.mark.parametrize("given", ["host", "bare-resolver", "sync-resolution"])
async def test_a_bare_host_still_serves_and_carries_no_context(given: str) -> None:
    tenant = ops_fixture().host
    mount: dict[str, Any] = {
        "host": {"host": tenant},
        "bare-resolver": {"host_resolver": lambda: tenant},
        "sync-resolution": {"host_resolver": lambda: HostResolution(tenant)},
    }[given]
    actions = _Host()
    seen = _Seen()
    _, _, is_error = await _call(
        _served(actions, around=seen.around, annotate=seen.annotate, **mount), {"action": "note_take", "note": "ok"}
    )
    assert not is_error and actions.ran_in == [tenant]
    assert [(hook, call.host, call.context) for hook, call in seen.calls] == [
        ("around", tenant, None),
        ("annotate", tenant, None),
    ]


async def test_a_host_resolver_s_refusal_shows_the_hooks_no_host_and_no_context() -> None:
    seen = _Seen()
    _, structured, _ = await _call(
        _served(_Host(), host_resolver=_booting, around=seen.around, annotate=seen.annotate),
        {"action": "note_take", "note": "ok"},
    )
    assert structured == {"refusal": {"kind": "refused", "status": 503}}
    assert [(hook, call.host, call.context) for hook, call in seen.calls] == [
        ("around", None, None),
        ("annotate", None, None),
    ]


async def test_a_host_resolution_holding_no_host_is_named() -> None:
    binding = ToolBinding(
        _Host().catalogue().mount(_TOOLS[0]),
        host_resolver=lambda: HostResolution("tenant-a", context=1),  # type: ignore[arg-type]
        caller=lambda: CALLER,
    )
    with pytest.raises(TypeError, match="answered a HostResolution whose host is 'tenant-a', not an OpsHost"):
        await binding.call({"action": "note_take", "note": "ok"})


def test_a_host_resolution_and_a_tool_call_are_frozen() -> None:
    resolution = HostResolution(ops_fixture().host, context="ctx")
    call = ToolCall(tool="host", action=None, caller=CALLER, host=None, timeout_s=1.0)
    assert call.context is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        resolution.context = "changed"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        call.context = "changed"  # type: ignore[misc]
    assert "HostResolution" in fastmcp_transport.__all__
