"""The FastMCP transport: mounts the action catalogue as tools on a FastMCP server.

Install with the ``fastmcp`` extra (``3tears-evals[fastmcp]``). The adapter is thin by construction:
each tool's name, description, input schema and behaviour hints are read off a
:class:`~threetears.evals.actions.MountedTool`, and a call is handed to
:meth:`~threetears.evals.actions.MountedTool.call` whole — selecting the action, refusing an undeclared
parameter, generating help and rendering the result all happen in the catalogue, so this transport and
any other answer a call identically.

Serving an eval host's runs is one line: handed the :class:`~threetears.evals.kernel.host.EvalHost`
:func:`~threetears.evals.quick.as_host` builds, and the scope its calls act in, it builds the server, the
:class:`~threetears.evals.ops.OpsHost` (:meth:`~threetears.evals.ops.OpsHost.reading`: every read, nothing
launched) and the read-only tool, and returns the server::

    from threetears.evals.quick import as_host
    from threetears.evals.transports.fastmcp import mount_fastmcp

    mount_fastmcp(as_host([my_eval], store=".evals/evals.sqlite"), scope="dev").run()

A host of the full shape mounts it with its own server, its catalogue (the engine's actions plus its own), its
:class:`~threetears.evals.ops.OpsHost` and a caller resolver — who is calling and the scope they act in,
asked once per call::

    from fastmcp import FastMCP
    from threetears.evals.actions import Caller, eval_catalogue
    from threetears.evals.transports.fastmcp import mount_fastmcp

    server = FastMCP("myapp")
    mount_fastmcp(server, eval_catalogue(my_actions), host=ops_host, caller=lambda: Caller(scope_id="dev", identity="me"))

These hooks are what a multi-tenant host needs, so no host rebuilds :meth:`CatalogueTool.over` to get them:

- **The host, per call.** ``host_resolver`` in place of ``host`` answers which
  :class:`~threetears.evals.ops.OpsHost` this call works in — a tenant's storage and job manager —
  read from the request context like the caller is. Exactly one of the two is given. **A resolver may refuse**:
  one that raises an :class:`~threetears.evals.kernel.EvalServiceError` — a host still booting, a tenant
  suspended — answers the call as a refusal with that error's kind, status and remedy, exactly as an action's
  refusal is answered; anything else it raises is a defect. The error it raised reaches both hooks whole, as
  :attr:`ToolCall.refused`, so a host whose one error class covers two cases — a host booting, one that failed to
  boot — tells them apart by the instance's own attributes. **A resolver may hand state on**: one that answers a
  :class:`HostResolution` in place of the bare host carries a ``context`` of the host's own — the tenant it
  resolved, a span, an audit record — to both hooks as :attr:`ToolCall.context`, beside the host as
  :attr:`ToolCall.host`, so no per-call state passes through context variables the resolver must reset.
- **One ``around`` hook** wraps every call for the host's tracing, metrics or audit log
  (:data:`CallHook`). It sees the call and its outcome and cannot change either — a host resolver's refusal among
  them, handed to the hook as its outcome exactly as an action's is, so "no host could serve this call" reaches the
  host's audit log where every other call's line is written.
- **One ``annotate`` hook** adds the host's own word to what a call returns (:data:`CallAnnotator`): a note
  appended as a text block of its own after the outcome's, never in place of it — the pricing catalogue a refused
  launch was priced against is stale, say.
- **A refusal as data.** A refused call is an MCP error result (``isError``) carrying the refusal's
  teaching text, and its :class:`~threetears.evals.actions.Refusal` as structured content —
  ``{"ending": "refused", "refusal": {"kind": ..., "status": ...}}``, ``"raised"`` where the engine or host failed
  the call (a 5xx status) — so a client branches on the kind, never on the text. A
  result carries its text and the result model's JSON as structured content.

**One call timeout, never one computed from the call.** No action's call waits on a model — work that calls one
runs as a job the caller polls — so a call is a few store round-trips, and every call is held to one timeout
(:data:`DEFAULT_CALL_TIMEOUT_S`, or the ``call_timeout_s`` a host mounts with), never one computed from what the
call might spend. A host may mount with a function in place of the number (:data:`CallTimeout`), read once as
each call starts, so an operator retunes it from live configuration without remounting. A call that outlives it is
answered as an error saying so, rather than holding the client. The timeout holds a call up to its action's
outcome; an ``around`` hook's code after its yield, and the ``annotate`` hook, run past it, under
:data:`HOOK_TAIL_TIMEOUT_S` of their own, so a finished action's outcome always reaches its caller.

**This module is a public root.** A host imports from here and from no module below it.
"""

from __future__ import annotations

import logging
import asyncio
import inspect
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, TypeVar, overload

# ``FastMCP`` is named only in ``mount_fastmcp``'s annotation, so it is imported for the type checker alone: importing
# it loads FastMCP's whole server stack (its context, sampling, HTTP app and event store), about as long again as the
# tool classes this module subclasses, and a host that builds its server imports it there anyway.
if TYPE_CHECKING:
    from fastmcp import FastMCP

try:
    from fastmcp.tools import Tool
    from fastmcp.tools.base import ToolResult
except ModuleNotFoundError as missing:
    if missing.name != "fastmcp":
        raise
    raise ModuleNotFoundError(
        'threetears.evals.transports.fastmcp needs the fastmcp extra: pip install "3tears-evals[fastmcp]"',
        name=missing.name,
    ) from missing
from pydantic import ConfigDict, Field

from threetears.evals.actions import (
    ActionCatalogue,
    ActionOutcome,
    Caller,
    MountedTool,
    ToolSpec,
    eval_catalogue,
    read_only_tools,
    standard_tools,
)
from threetears.evals.kernel import EvalServiceError
from threetears.evals.kernel.host import EvalHost
from threetears.evals.ops import OpsHost
from threetears.evals.schema.logs import contain, get_logger

log = get_logger(__name__)

#: How long one call may take, in seconds, unless a host mounts with another: every call, whatever its action. No call
#: waits on a model, so this bounds store round-trips and a host's own caller resolution, never a model's latency.
DEFAULT_CALL_TIMEOUT_S = 60.0

#: How long an ``around`` hook may take, in seconds, past the outcome it was handed — or closing, once a call is
#: stopped — and, separately, how long the ``annotate`` hook may take over that outcome. The call's own timeout ends
#: when the action does, so the outcome reaches the caller whatever either hook does after it; a hook still running
#: past this is cut short and logged, and the outcome stands.
HOOK_TAIL_TIMEOUT_S = 10.0

#: The timeout every call is held to, in seconds: a fixed number, or a function the binding reads once as each call
#: starts — a host reading it from live configuration, so an operator retunes it without remounting.
#:
#: A number is checked positive when the tool is mounted. A function is checked as each call reads it, since its
#: value is not known before: one that answers anything but a positive number fails that call before anything runs,
#: with a :class:`ValueError` naming the value, rather than holding the call to no timeout, or to one it would
#: outlive at once — a misread setting is a fault for the operator to see, not a call to wave through unbounded.
#: What it raises of its own propagates the same way. The value a call read is the one it is held to, whatever the
#: function answers while it runs, and the one :class:`CallTimedOut` reports.
CallTimeout = float | Callable[[], float]


#: What a caller does about a call the timeout stopped: the action may have finished, or begun a write, before it was
#: stopped, so a call retried blind can launch — and pay for — the same work twice. Every action it names is in the
#: engine's catalogue, which ``test_refusal_remedies.py`` reads it against, as it reads every remedy.
_TIMED_OUT_REMEDY = (
    "the action may have completed, or begun a write, before it was stopped, so read before calling it again: "
    "job_poll for a job it may have started (run_launch, sweep_launch, analysis_generate), or the list for what it "
    "was writing (runs_list, campaigns_list, case_sets_list)"
)


class CallTimedOut(Exception):
    """A call outlived the call timeout and was stopped before its action finished.

    Raised by :meth:`ToolBinding.call` only for the call's own deadline — a ``TimeoutError`` the action or a resolver
    raises of its own propagates as itself — and answered by :class:`CatalogueTool` as an error result saying so.

    :ivar timeout_s: The timeout it outlived, in seconds: the value this call read (:data:`CallTimeout`), which a
        host mounted with a function may have changed since.
    """

    def __init__(self, timeout_s: float) -> None:
        """Name the timeout the call outlived.

        :param timeout_s: The timeout, in seconds.
        :ptype timeout_s: float
        """
        super().__init__(f"the call outlived its {timeout_s:g}s call timeout and was stopped; {_TIMED_OUT_REMEDY}")
        self.timeout_s = timeout_s


#: Resolves who is calling, and their scope, for one call — sync or async. The host reads its own
#: request context (an access token, a session) to answer it.
CallerResolver = Callable[[], Caller | Awaitable[Caller]]


@dataclass(frozen=True)
class HostResolution:
    """What a host resolver may answer in place of the bare host: the host, and state of the host's own for the hooks.

    The resolver is where a multi-tenant host learns which tenant a call is for; what it learned there — the tenant,
    a span it opened, the audit record it started — is what its ``around`` and ``annotate`` hooks need too. Answered
    here, it reaches both as :attr:`ToolCall.context` for this call alone, rather than through a context variable the
    resolver must set and reset on every call.

    :ivar host: The host this call works in, exactly as a resolver answering an
        :class:`~threetears.evals.ops.OpsHost` gives it.
    :ivar context: Anything of the host's own, handed to the hooks unread and unchanged; ``None`` for nothing.
    """

    host: OpsHost
    context: Any = None


#: Resolves the host one call works in — sync or async — for a server that serves more than one: a
#: tenant's storage, job manager and settings, read from the same request context as the caller. It is
#: asked after the caller, once per call. It answers the :class:`~threetears.evals.ops.OpsHost`, or a
#: :class:`HostResolution` carrying it with a ``context`` for the hooks (:attr:`ToolCall.context`).
HostResolver = Callable[[], OpsHost | HostResolution | Awaitable[OpsHost | HostResolution]]


@dataclass(frozen=True, kw_only=True)
class ToolCall:
    """One call, as the ``around`` and ``annotate`` hooks see it once its caller and host are known.

    It is built once the host is resolved, or has refused, so it never changes while a hook holds it.

    :ivar tool: The tool called.
    :ivar action: The ``action`` argument as the call gave it — not yet checked against the tool, so it
        may name no action at all; ``None`` when the call gave none or a non-string. The outcome's
        ``action`` is the action the catalogue resolved.
    :ivar caller: Who is calling, and their scope, as the caller resolver answered.
    :ivar host: The host the call works in — the one mounted, or the one the host resolver answered; ``None`` when
        the host resolver refused the call, so none was resolved.
    :ivar timeout_s: The call timeout this call is held to, in seconds: the value it read as it started
        (:data:`CallTimeout`), and the one a :class:`CallTimedOut` thrown in at an ``around`` hook's yield reports.
    :ivar context: The ``context`` of the :class:`HostResolution` the host resolver answered; ``None`` when it
        answered a bare host, when the binding has one host and no resolver, or when the resolver refused.
    :ivar refused: The :class:`~threetears.evals.kernel.EvalServiceError` the host resolver raised, the very instance —
        a host's own subclass with its own attributes — when it refused the call, and ``host`` and ``context``
        are ``None``; ``None`` in every other case. The outcome carries only the refusal's kind and status, which
        two cases of one error class share; this is how a hook tells them apart.
    """

    tool: str
    action: str | None
    caller: Caller
    host: OpsHost | None
    timeout_s: float
    context: Any = None
    refused: EvalServiceError | None = None


#: Wraps every call for a host's tracing, metrics or audit log, seeing the call and its outcome and
#: changing neither.
#:
#: An async generator function handed the :class:`ToolCall`, which yields once — the call runs at the
#: yield — and receives the call's :class:`~threetears.evals.actions.ActionOutcome` as the yield's value::
#:
#:     async def around(call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
#:         with tracer.start_as_current_span(f"{call.tool}.{call.action}"):
#:             outcome = yield
#:             metrics.count(call.action, outcome.refusal.kind if outcome.refusal else "ok")
#:
#: Code before the yield runs in the call's own context, so a span it opens is current while the action
#: runs: a :class:`~contextvars.ContextVar` it sets, or a ``with`` it enters, before its yield is what the action
#: sees, and its code after the yield — a ``reset`` or the ``with``'s exit — undoes it in that same task, so
#: nothing it set outlives the call.
#:
#: **A host resolver's refusal enters the hook too.** The caller is known, so the :class:`ToolCall` exists: the hook
#: runs to its yield and is then handed the refusal as its outcome, exactly as an action's — no action runs between,
#: and an audit log written from the hook records "no host could serve this call" like any other refused call. Its
#: :class:`ToolCall` has ``host`` and ``context`` ``None``, since no host was resolved, and carries the error the
#: resolver raised, the very instance, as :attr:`ToolCall.refused` — so a hook logs a host still booting at one level
#: and one that failed to boot at another, though both answer the same kind and status. A call refused while its
#: *caller* was being resolved never enters the hook, since there is no :class:`ToolCall` without a caller.
#:
#: **State the host resolver found reaches the hook on the call.** A resolver answering a :class:`HostResolution`
#: hands its ``context`` to the hook as :attr:`ToolCall.context`, beside the host it resolved as
#: :attr:`ToolCall.host` — the tenant, an audit record — for this call alone.
#:
#: A call that raises — a defect, not a refusal — is thrown in at the yield, so a ``with`` or
#: ``finally`` closes; the hook cannot catch it on the call's behalf, since it propagates whatever the
#: hook does. **A call that outlives its timeout has** :class:`CallTimedOut` **thrown in at the yield** — the timeout
#: it read, as :attr:`ToolCall.timeout_s` — so the hook tells a timeout from a cancel: a call cancelled from outside
#: (the client gave up, the process is stopping) is closed at the yield instead, and its ``finally`` runs with nothing
#: thrown. Either way the call goes on as it would with no hook: :class:`CallTimedOut`, or the cancellation,
#: propagates. **It cannot change the outcome**: what it yields is discarded, and the outcome is frozen.
#:
#: **A hook that raises is logged, and the call goes on as if there were no hook.** The hook observes;
#: if its failure could fail the call, an exporter outage would refuse every eval call, and a failure
#: after the action ran would report an error for work that was done — a launch that spent, a write that
#: landed — inviting an agent to repeat it. So a hook that raises before its yield leaves the call to run
#: unobserved, and one that raises after it, or yields a second time, leaves the outcome as it was. A
#: host whose audit must hold before an action runs puts that check in the host or the action, which
#: can refuse; a hook is not a gate.
#:
#: **Its code after the yield is not on the call's clock.** The call timeout holds the call up to the moment its
#: action finishes; once it has, its outcome is the caller's, and a slow flush after the yield must not turn a
#: launch that spent into a timeout the client relaunches and pays for again. That code is held to
#: :data:`HOOK_TAIL_TIMEOUT_S` of its own, as is closing a hook whose call was stopped, and one that overruns is
#: cut short and logged.
CallHook = Callable[[ToolCall], AsyncGenerator[None, ActionOutcome]]

#: Adds the host's own word to what a call returns: handed the :class:`ToolCall` and its
#: :class:`~threetears.evals.actions.ActionOutcome` once the outcome exists, and answering a note or ``None`` —
#: sync or async, as a resolver may be::
#:
#:     def annotate(call: ToolCall, outcome: ActionOutcome) -> str | None:
#:         if outcome.refusal is not None and call.action == "run_launch" and pricing.stale:
#:             return "the pricing catalogue is stale, so every model is unpriced"
#:         return None
#:
#: **A note is added, never substituted.** A non-empty note is appended to the tool's result as a text block of its
#: own, after the outcome's; the outcome's text, its structured content and whether it is an error are exactly what
#: they would be without the hook, so a client branching on the refusal's kind is not misled by a word the host
#: added. ``None`` or an empty string adds nothing.
#:
#: It sees every outcome a call reaches once its caller is known — a result, a refusal the engine or the catalogue
#: made, and one the host resolver made — but not a call refused while its caller was being resolved, since there is
#: no :class:`ToolCall` without one, nor a call that raised or outlived its timeout, which has no outcome. Its
#: :class:`ToolCall` carries the resolved host and the resolver's ``context`` (:class:`HostResolution`) as ``around``'s
#: does — both ``None`` for a host resolver's refusal, which it sees as :attr:`ToolCall.refused`, the error the
#: resolver raised. **An
#: annotator that raises, answers something other than a string, or overruns** :data:`HOOK_TAIL_TIMEOUT_S` **is
#: logged, and the call returns as if there were no hook**: as with ``around``, the outcome is already the caller's,
#: and a note that failed must not turn a launch that spent into an error the agent repeats. Like ``around``'s code
#: after its yield it runs once the call's own timeout has stopped counting; a sync annotator cannot be cut short,
#: so one that may block is written async.
CallAnnotator = Callable[[ToolCall, ActionOutcome], str | None | Awaitable[str | None]]

_T = TypeVar("_T")


async def _resolved(value: _T | Awaitable[_T]) -> _T:
    """A resolver's answer, awaited when it is awaitable."""
    return await value if inspect.isawaitable(value) else value


class ToolBinding:
    """What a :class:`CatalogueTool` hands each call to: the mounted tool, the host, the caller resolver and the hooks.

    A plain class rather than a model or a dataclass, so the tool holds it by identity and FastMCP never
    walks the host's types to describe it.
    """

    def __init__(
        self,
        mounted: MountedTool,
        *,
        host: OpsHost | None = None,
        host_resolver: HostResolver | None = None,
        caller: CallerResolver,
        around: CallHook | None = None,
        annotate: CallAnnotator | None = None,
        call_timeout_s: CallTimeout = DEFAULT_CALL_TIMEOUT_S,
    ) -> None:
        """Bind a mounted tool to the host, caller resolver, hooks and call timeout its calls run under.

        :param mounted: The tool cut from the catalogue.
        :ptype mounted: MountedTool
        :param host: The one host its actions work in; or
        :ptype host: OpsHost | None
        :param host_resolver: Resolves the host per call. Exactly one of ``host`` and ``host_resolver``.
        :ptype host_resolver: HostResolver | None
        :param caller: Resolves who is calling, per call.
        :ptype caller: CallerResolver
        :param around: Wraps every call (:data:`CallHook`); ``None`` for none.
        :ptype around: CallHook | None
        :param annotate: Adds the host's note to what a call returns (:data:`CallAnnotator`); ``None`` for none. Held
            here and applied where the outcome becomes the tool's result (:meth:`CatalogueTool.run`):
            :meth:`call` returns the outcome bare.
        :ptype annotate: CallAnnotator | None
        :param call_timeout_s: The one timeout every call is held to, in seconds, or a function read once as each call
            starts (:data:`CallTimeout`).
        :ptype call_timeout_s: CallTimeout
        :raises ValueError: Both or neither of ``host`` and ``host_resolver`` were given, or ``call_timeout_s`` is a
            number that is not positive.
        :raises TypeError: ``around`` is not an async generator function.
        """
        if not callable(call_timeout_s) and not call_timeout_s > 0:
            raise ValueError(f"call_timeout_s must be positive; got {call_timeout_s!r}")
        if (host is None) == (host_resolver is None):
            given = "both" if host is not None else "neither"
            raise ValueError(
                f"tool {mounted.name!r}: {given} of host and host_resolver were given; pass host for a server "
                "with one host, or host_resolver to resolve it per call — exactly one"
            )
        if around is not None and not inspect.isasyncgenfunction(around):
            raise TypeError(
                f"tool {mounted.name!r}: around is {around!r}, not an async generator function; write the hook as "
                "`async def around(call): ... outcome = yield ...` (see CallHook)"
            )
        self.mounted = mounted
        self.host = host
        self.host_resolver = host_resolver
        self.caller = caller
        self.around = around
        self.annotate = annotate
        self.call_timeout_s = call_timeout_s

    async def call(self, arguments: dict[str, Any]) -> ActionOutcome:
        """Carry out one call: resolve the caller, then the host, and hand the call to the catalogue under the hook.

        The call timeout is read once, as the call starts (:data:`CallTimeout`), and holds everything up to the
        action's outcome: the resolvers, the hook's code before its yield, and the action. The hook's code after its
        yield runs once the outcome is in hand, under :data:`HOOK_TAIL_TIMEOUT_S` of its own, and cannot replace it
        (:data:`CallHook`).

        **A resolver's refusal is an outcome.** A caller or host resolver that raises an
        :class:`~threetears.evals.kernel.EvalServiceError` — a host's own subclass among them, a 503 for a host still
        booting — refuses the call as the engine refuses one from inside an action
        (:meth:`~threetears.evals.actions.MountedTool.refused_by_host`): its kind, its status, its reason and its
        remedy. A caller resolver's refusal is answered before the hook is entered, since there is no
        :class:`ToolCall` without a caller; a host resolver's enters the hook, which runs to its yield and is handed
        the refusal as its outcome, as it would an action's (:data:`CallHook`).

        :param arguments: The call's arguments, ``action`` among them.
        :ptype arguments: dict[str, Any]
        :return: The catalogue's outcome, exactly as it returned it, or the refusal a resolver raised.
        :rtype: ActionOutcome
        :raises CallTimedOut: The call outlived the call timeout before its action finished; the action was stopped,
            and a :class:`CallTimedOut` thrown into the hook at its yield. It reports the timeout this call read.
        :raises ValueError: ``call_timeout_s`` is a function and answered anything but a positive number; nothing ran.
        :raises TypeError: A resolver answered with something other than a
            :class:`~threetears.evals.actions.Caller`, or an :class:`~threetears.evals.ops.OpsHost` or a
            :class:`HostResolution` holding one.
        :raises Exception: Whatever else a resolver raises — anything but an
            :class:`~threetears.evals.kernel.EvalServiceError` — before the hook is entered and before any action
            runs; and whatever the catalogue lets propagate (a defect). A refusal is an outcome, not an exception.
        """
        outcome, _ = await _answered(self, arguments)
        return outcome


def _timeout_now(binding: ToolBinding) -> float:
    """The timeout a call starting now is held to: the number mounted, or what the mounted function answers.

    :raises ValueError: The function answered anything but a positive number; the value is named, and no call is run
        under it (:data:`CallTimeout`).
    """
    timeout = binding.call_timeout_s
    if not callable(timeout):
        return timeout
    read = timeout()
    if isinstance(read, bool) or not isinstance(read, int | float) or not read > 0:
        raise ValueError(
            f"tool {binding.mounted.name!r}: the call_timeout_s function answered {read!r}, not a positive number of "
            "seconds, so the call was not run; fix the setting it reads"
        )
    return float(read)


async def _answered(binding: ToolBinding, arguments: dict[str, Any]) -> tuple[ActionOutcome, ToolCall | None]:
    """Carry out one call (:meth:`ToolBinding.call`), and the call as the hooks see it once its caller is known."""
    timeout_s = _timeout_now(binding)
    deadline = asyncio.timeout(timeout_s)
    hook: AsyncGenerator[None, ActionOutcome] | None = None
    action = arguments.get("action")
    try:
        async with deadline:
            try:
                caller = await _resolved(binding.caller())
            except EvalServiceError as refused:
                return binding.mounted.refused_by_host(refused, action=action), None
            if not isinstance(caller, Caller):
                raise TypeError(f"tool {binding.mounted.name!r}: the caller resolver answered {caller!r}, not a Caller")
            host = binding.host
            context: Any = None
            host_refusal: ActionOutcome | None = None
            refused_by: EvalServiceError | None = None
            if binding.host_resolver is not None:
                try:
                    resolved = await _resolved(binding.host_resolver())
                except EvalServiceError as refused:
                    # The caller is known, so the hook is entered and handed this refusal as its outcome.
                    host_refusal = binding.mounted.refused_by_host(refused, action=action)
                    refused_by = refused
                else:
                    host, context = _resolution(binding, resolved)
            # Built once the host is known, so the call the hooks hold carries it and never changes under them.
            observed = ToolCall(
                tool=binding.mounted.name,
                action=action if isinstance(action, str) else None,
                caller=caller,
                host=host,
                timeout_s=timeout_s,
                context=context,
                refused=refused_by,
            )
            if binding.around is not None:
                hook = await _entered(binding.around, observed)
            if host_refusal is not None:
                outcome = host_refusal
            else:
                assert host is not None  # one of the two is set, which __init__ holds
                try:
                    outcome = await binding.mounted.call(arguments, host=host, caller=caller)
                except BaseException as defect:
                    if hook is not None:
                        # Stopped by this call's own deadline, the hook is told so; a cancel from outside closes it.
                        stopped = isinstance(defect, asyncio.CancelledError) and deadline.expired()
                        await _closed_within(hook, observed, CallTimedOut(timeout_s) if stopped else defect)
                    raise
    except TimeoutError:
        # A TimeoutError the caller resolver or the action raised of its own is not this call outliving its
        # timeout, and answering it as one would blame the store for the action's own fault: it propagates, as
        # anything else the action raises does.
        if not deadline.expired():
            raise
        raise CallTimedOut(timeout_s) from None
    if hook is not None:
        await _finished_within(hook, observed, outcome)
    return outcome, observed


def _resolution(binding: ToolBinding, resolved: object) -> tuple[OpsHost, Any]:
    """The host a host resolver answered, and the context it carried for the hooks (``None`` with a bare host).

    :raises TypeError: It answered neither an :class:`~threetears.evals.ops.OpsHost` nor a :class:`HostResolution`
        holding one.
    """
    if isinstance(resolved, HostResolution):
        if not isinstance(resolved.host, OpsHost):
            raise TypeError(
                f"tool {binding.mounted.name!r}: the host resolver answered a HostResolution whose host is "
                f"{resolved.host!r}, not an OpsHost"
            )
        return resolved.host, resolved.context
    if not isinstance(resolved, OpsHost):
        raise TypeError(
            f"tool {binding.mounted.name!r}: the host resolver answered {resolved!r}, not an OpsHost or a HostResolution"
        )
    return resolved, None


async def _note(annotate: CallAnnotator, call: ToolCall, outcome: ActionOutcome) -> str | None:
    """The annotator's note on this outcome, within :data:`HOOK_TAIL_TIMEOUT_S`; ``None`` for none, or for a failure.

    The outcome is already the caller's: an annotator that raises, answers something other than a string, or
    overruns is logged, and the call returns with no note (:data:`CallAnnotator`).
    """
    try:
        async with asyncio.timeout(HOOK_TAIL_TIMEOUT_S):
            note = await _resolved(annotate(call, outcome))
    except TimeoutError:
        # Log only: a tool call is a transport exchange with no record of its own, and its outcome is already the caller's.
        log.error(
            "annotate hook ran past its %gs on %s/%s and was cut short; the outcome stands with no note",
            HOOK_TAIL_TIMEOUT_S,
            call.tool,
            call.action,
        )
        return None
    except Exception as unannotated:
        # Log only: a tool call is a transport exchange with no record of its own, and its outcome is already the caller's.
        contain(
            unannotated,
            log=log,
            what=f"annotate hook failed on {call.tool}/{call.action}; the outcome stands with no note",
            level=logging.ERROR,
        )
        return None
    if note is None:
        return None
    if not isinstance(note, str):
        log.error(
            "annotate hook answered %r on %s/%s, not a string or None; the outcome stands with no note",
            note,
            call.tool,
            call.action,
        )
        return None
    return note if note.strip() else None


async def _entered(around: CallHook, call: ToolCall) -> AsyncGenerator[None, ActionOutcome] | None:
    """The hook, run to its yield; ``None`` when it failed there, which is logged and leaves the call unobserved."""
    hook = around(call)
    try:
        await anext(hook)
    except Exception as unentered:
        # Log only: a tool call is a transport exchange with no record of its own, and its outcome is already the caller's.
        contain(
            unentered,
            log=log,
            what=f"around hook failed before its yield on {call.tool}/{call.action}; the call runs unobserved",
            level=logging.ERROR,
        )
        return None
    return hook


async def _finished_within(hook: AsyncGenerator[None, ActionOutcome], call: ToolCall, outcome: ActionOutcome) -> None:
    """Hand the hook the outcome and let it finish, within :data:`HOOK_TAIL_TIMEOUT_S`; what it does is logged.

    The outcome is already the caller's: a hook that raises, yields again or overruns is logged, and nothing it
    does reaches the caller.
    """
    try:
        async with asyncio.timeout(HOOK_TAIL_TIMEOUT_S):
            try:
                await hook.asend(outcome)
            except StopAsyncIteration:
                # NOSILENT: the hook took its outcome and finished — the one way a hook is meant to end
                return
            except Exception as unfinished:
                # Log only: a tool call is a transport exchange with no record of its own, and its outcome is already the caller's.
                contain(
                    unfinished,
                    log=log,
                    what=f"around hook failed after its yield on {call.tool}/{call.action}; the outcome stands",
                    level=logging.ERROR,
                )
                return
            log.error(
                "around hook yielded twice on %s/%s; it yields once, and the outcome stands", call.tool, call.action
            )
            await _close(hook, call, None)
    except TimeoutError:
        # NOSILENT: the hook only observes the call; the outcome it was handed is already the caller's, unchanged.
        log.error(
            "around hook ran past its %gs after its yield on %s/%s and was cut short; the outcome stands",
            HOOK_TAIL_TIMEOUT_S,
            call.tool,
            call.action,
        )


async def _closed_within(hook: AsyncGenerator[None, ActionOutcome], call: ToolCall, defect: BaseException) -> None:
    """Finish a hook whose call raised or was stopped (:func:`_close`), within :data:`HOOK_TAIL_TIMEOUT_S`.

    Run where the call's own timeout may already have fired, so it carries a bound of its own: a hook that will not
    close must not hold a stopped call open.
    """
    try:
        async with asyncio.timeout(HOOK_TAIL_TIMEOUT_S):
            await _close(hook, call, defect)
    except TimeoutError:
        # NOSILENT: the hook only observes the call; the caller re-raises the call's own defect, unchanged.
        log.error(
            "around hook ran past its %gs closing %s/%s and was cut short", HOOK_TAIL_TIMEOUT_S, call.tool, call.action
        )


async def _close(hook: AsyncGenerator[None, ActionOutcome], call: ToolCall, defect: BaseException | None) -> None:
    """Finish a hook the call cannot hand an outcome: throw it the defect, or close it.

    A defect is thrown in at the yield, so the hook records it; a cancellation, or a hook that already
    had its outcome, is closed, so its ``finally`` runs. Whatever the hook does then is logged and
    otherwise ignored: the caller re-raises the defect itself.
    """
    try:
        if isinstance(defect, Exception):
            await hook.athrow(defect)
        else:
            await hook.aclose()
            return
    except StopAsyncIteration:
        # NOSILENT: the hook took the defect at its yield and finished — the one way a hook is meant to end
        # here; the caller re-raises the defect itself, so there is nothing to report on the hook's behalf.
        return
    except Exception as raised:
        # Log only: a tool call is a transport exchange with no record of its own, and its outcome is already the caller's.
        if raised is not defect:
            contain(raised, log=log, what=f"around hook failed closing {call.tool}/{call.action}", level=logging.ERROR)
        return
    # It caught the defect and yielded again: close it, so it does not linger half-run.
    await _close(hook, call, None)


class CatalogueTool(Tool):
    """One mounted tool, as FastMCP serves it: every call handed to the catalogue.

    :ivar binding: What each call is handed to. Excluded from the tool's serialized form: a client is shown
        what the catalogue declares — name, description, schema, hints — and nothing of the host.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    binding: ToolBinding = Field(exclude=True)

    @classmethod
    def over(
        cls,
        mounted: MountedTool,
        *,
        host: OpsHost | None = None,
        host_resolver: HostResolver | None = None,
        caller: CallerResolver,
        around: CallHook | None = None,
        annotate: CallAnnotator | None = None,
        call_timeout_s: CallTimeout = DEFAULT_CALL_TIMEOUT_S,
    ) -> CatalogueTool:
        """The FastMCP tool for one mounted tool.

        :param mounted: The tool cut from the catalogue.
        :ptype mounted: MountedTool
        :param host: The one host its actions work in; or
        :ptype host: OpsHost | None
        :param host_resolver: Resolves the host per call. Exactly one of ``host`` and ``host_resolver``.
        :ptype host_resolver: HostResolver | None
        :param caller: Resolves who is calling, per call.
        :ptype caller: CallerResolver
        :param around: Wraps every call (:data:`CallHook`); ``None`` for none.
        :ptype around: CallHook | None
        :param annotate: Adds the host's note to what a call returns (:data:`CallAnnotator`); ``None`` for none.
        :ptype annotate: CallAnnotator | None
        :param call_timeout_s: The one timeout every call is held to, in seconds, or a function read once as each call
            starts (:data:`CallTimeout`).
        :ptype call_timeout_s: CallTimeout
        :return: The tool, ready for :meth:`fastmcp.FastMCP.add_tool`.
        :rtype: CatalogueTool
        :raises ValueError: Both or neither of ``host`` and ``host_resolver`` were given, or ``call_timeout_s`` is a
            number that is not positive.
        :raises TypeError: ``around`` is not an async generator function.
        """
        binding = ToolBinding(
            mounted,
            host=host,
            host_resolver=host_resolver,
            caller=caller,
            around=around,
            annotate=annotate,
            call_timeout_s=call_timeout_s,
        )
        hints = mounted.hints
        # Validated rather than constructed, so the hints are the MCP annotation model FastMCP declares
        # without this adapter importing the SDK beneath it.
        return cls.model_validate(
            {
                "name": mounted.name,
                "description": mounted.description,
                "parameters": mounted.input_schema(),
                "annotations": {
                    "readOnlyHint": hints.read_only,
                    "destructiveHint": hints.destructive,
                    "idempotentHint": False,
                    "openWorldHint": hints.open_world,
                },
                "binding": binding,
            }
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Carry out one call through the catalogue.

        :param arguments: The call's arguments, ``action`` among them.
        :ptype arguments: dict[str, Any]
        :return: The result: text always; as structured content, the result model's JSON when the call
            succeeded, or ``{"ending": "refused" | "raised", "refusal": {"kind": ..., "status": ...}}`` with the
            error flag when it was refused or raised (:attr:`~threetears.evals.actions.ActionOutcome.ending`); and the
            error flag alone, with text saying so and to read what it may have done before calling it again, when it
            outlived the call timeout. A note the ``annotate`` hook answered follows the outcome's text as a text block
            of its own, and changes nothing else (:data:`CallAnnotator`).
        :rtype: ToolResult
        """
        binding = self.binding
        try:
            # The call up to its outcome — the caller and host resolvers, the hook before its yield, the action — is
            # held to the one timeout; the hook after its yield is not (ToolBinding.call), nor is the annotator.
            outcome, call = await _answered(binding, arguments)
        except CallTimedOut as stopped:
            return ToolResult(
                content=(
                    f"the call outlived this server's {stopped.timeout_s:g}s call timeout and was stopped. No "
                    "action's call waits on a model — work that does runs as a job — so this is a fault of the "
                    f"server's store or of the action, not something to wait out: report it. But {_TIMED_OUT_REMEDY}"
                ),
                is_error=True,
            )
        structured = (
            {"ending": outcome.ending, "refusal": asdict(outcome.refusal)}
            if outcome.refusal is not None
            else outcome.structured
        )
        note = (
            await _note(binding.annotate, call, outcome) if binding.annotate is not None and call is not None else None
        )
        # Each text is made a content block on its own, by FastMCP's own conversion of a string, so the note is a
        # block after the outcome's and never joined into it — without this adapter importing the SDK beneath it.
        content = [block for text in (outcome.text, note) if text is not None for block in ToolResult(text).content]
        return ToolResult(content=content, structured_content=structured, is_error=outcome.is_error)


#: What :func:`mount_fastmcp` names the server it builds around a host it is handed alone.
_SERVER_NAME = "evals"


@overload
def mount_fastmcp(
    server: EvalHost | OpsHost,
    /,
    *,
    scope: str,
    identity: str = "mcp",
    tools: Sequence[ToolSpec] | None = None,
    call_timeout_s: CallTimeout = DEFAULT_CALL_TIMEOUT_S,
) -> FastMCP: ...


@overload
def mount_fastmcp(
    server: FastMCP,
    catalogue: ActionCatalogue | None = None,
    *,
    host: EvalHost | OpsHost | None = None,
    host_resolver: HostResolver | None = None,
    caller: CallerResolver | None = None,
    scope: str | None = None,
    identity: str = "mcp",
    around: CallHook | None = None,
    annotate: CallAnnotator | None = None,
    tools: Sequence[ToolSpec] | None = None,
    call_timeout_s: CallTimeout = DEFAULT_CALL_TIMEOUT_S,
) -> tuple[str, ...]: ...


def mount_fastmcp(
    server: FastMCP | EvalHost | OpsHost,
    catalogue: ActionCatalogue | None = None,
    *,
    host: EvalHost | OpsHost | None = None,
    host_resolver: HostResolver | None = None,
    caller: CallerResolver | None = None,
    scope: str | None = None,
    identity: str = "mcp",
    around: CallHook | None = None,
    annotate: CallAnnotator | None = None,
    tools: Sequence[ToolSpec] | None = None,
    call_timeout_s: CallTimeout = DEFAULT_CALL_TIMEOUT_S,
) -> FastMCP | tuple[str, ...]:
    """Add the catalogue's tools to a FastMCP server; or, handed a host alone, build the server and return it.

    **In one line**, ``mount_fastmcp(as_host([...], store=...), scope="dev").run()`` serves an eval host's runs: a
    server named ``evals``, the engine's catalogue, :func:`~threetears.evals.actions.read_only_tools` (nothing
    launches over it: :meth:`~threetears.evals.ops.OpsHost.reading`), and every call in ``scope`` as ``identity``. A
    host of the full shape mounts on its own server with every hook below.

    :param server: The host's server; or the one host every action works in, an
        :class:`~threetears.evals.kernel.host.EvalHost` or an :class:`~threetears.evals.ops.OpsHost`, to serve
        on a server built here, which is returned.
    :ptype server: FastMCP | EvalHost | OpsHost
    :param catalogue: The actions to serve — :func:`~threetears.evals.actions.eval_catalogue`, with the host's own;
        ``None`` for the engine's alone.
    :ptype catalogue: ActionCatalogue | None
    :param host: The one host every action works in: an :class:`~threetears.evals.ops.OpsHost`, or an
        :class:`~threetears.evals.kernel.host.EvalHost`, served through :meth:`~threetears.evals.ops.OpsHost.reading`;
        or
    :ptype host: EvalHost | OpsHost | None
    :param host_resolver: Resolves the host per call, for a server serving several. Exactly one of ``host``
        and ``host_resolver``.
    :ptype host_resolver: HostResolver | None
    :param caller: Resolves who is calling and their scope, once per call; or
    :ptype caller: CallerResolver | None
    :param scope: The one scope every call acts in, as ``identity``. Exactly one of ``caller`` and ``scope``.
    :ptype scope: str | None
    :param identity: Who the caller is, with ``scope``: recorded where a write records an author.
    :ptype identity: str
    :param around: Wraps every call to every tool, for the host's tracing, metrics or audit
        (:data:`CallHook`); ``None`` for none.
    :ptype around: CallHook | None
    :param annotate: Adds the host's note to what a call to any tool returns — a text block after the outcome's,
        never in its place (:data:`CallAnnotator`); ``None`` for none.
    :ptype annotate: CallAnnotator | None
    :param tools: The tools to cut; ``None`` for :func:`~threetears.evals.actions.standard_tools` under the
        default prefix, or :func:`~threetears.evals.actions.read_only_tools` over an eval host, which
        launches nothing. Pass ``read_only_tools()`` for an agent that may only read, or either with the
        host's own prefix.
    :ptype tools: Sequence[ToolSpec] | None
    :param call_timeout_s: The one timeout every call is held to, in seconds, or a function read once as each call
        starts, for a host that retunes it from live configuration (:data:`CallTimeout`).
    :ptype call_timeout_s: CallTimeout
    :return: The names of the tools added; or, handed a host alone, the server built, ready to ``run()``.
    :rtype: FastMCP | tuple[str, ...]
    :raises ValueError: The catalogue refuses a tool (see :meth:`~threetears.evals.actions.ActionCatalogue.mount`),
        or both or neither of ``host`` and ``host_resolver``, or of ``caller`` and ``scope``, were given, or
        ``call_timeout_s`` is a number that is not positive — before any tool is added. A function's value is
        checked per call instead.
    :raises TypeError: ``around`` is not an async generator function, before any tool is added.
    """
    if isinstance(server, EvalHost | OpsHost):
        if host is not None or host_resolver is not None:
            raise ValueError("mount_fastmcp(host, ...) serves the host it is handed; pass no host= or host_resolver=")
        from fastmcp import FastMCP  # the server stack, loaded only when a server is built here

        built = FastMCP(_SERVER_NAME)
        mount_fastmcp(
            built,
            catalogue,
            host=server,
            caller=caller,
            scope=scope,
            identity=identity,
            around=around,
            annotate=annotate,
            tools=tools,
            call_timeout_s=call_timeout_s,
        )
        return built
    if (caller is None) == (scope is None):
        raise ValueError("mount_fastmcp takes a caller resolver (caller=) or the one scope every call acts in (scope=)")
    resolve: CallerResolver
    if caller is not None:
        resolve = caller
    else:
        acting = Caller(scope_id=str(scope), identity=identity)

        def in_scope() -> Caller:
            return acting

        resolve = in_scope
    reading = isinstance(host, EvalHost)
    if isinstance(host, EvalHost):
        host = OpsHost.reading(host)
    mounted = (catalogue if catalogue is not None else eval_catalogue()).mount_all(
        tools if tools is not None else read_only_tools() if reading else standard_tools()
    )
    served = [
        CatalogueTool.over(
            tool,
            host=host,
            host_resolver=host_resolver,
            caller=resolve,
            around=around,
            annotate=annotate,
            call_timeout_s=call_timeout_s,
        )
        for tool in mounted
    ]
    for tool in served:
        server.add_tool(tool)
    return tuple(tool.name for tool in served)


__all__ = [
    "DEFAULT_CALL_TIMEOUT_S",
    "HOOK_TAIL_TIMEOUT_S",
    "CallAnnotator",
    "CallHook",
    "CallTimedOut",
    "CallTimeout",
    "CallerResolver",
    "CatalogueTool",
    "HostResolution",
    "HostResolver",
    "ToolBinding",
    "ToolCall",
    "mount_fastmcp",
]
