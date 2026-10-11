"""The FastMCP transport: mounts the action catalogue as tools on a FastMCP server.

Install with the ``fastmcp`` extra (``3tears-evals[fastmcp]``). The adapter is thin by construction:
each tool's name, description, input schema and behaviour hints are read off a
:class:`~threetears.evals.actions.MountedTool`, and a call is handed to
:meth:`~threetears.evals.actions.MountedTool.call` whole — selecting the action, refusing an undeclared
parameter, generating help and rendering the result all happen in the catalogue, so this transport and
any other answer a call identically.

A host mounts it with its own server, its catalogue (the engine's actions plus its own), its
:class:`~threetears.evals.ops.OpsHost` and a caller resolver — who is calling and the scope they act in,
asked once per call::

    from fastmcp import FastMCP
    from threetears.evals.actions import Caller, eval_catalogue
    from threetears.evals.transports.fastmcp import mount_fastmcp

    server = FastMCP("myapp")
    mount_fastmcp(server, eval_catalogue(my_actions), host=ops_host, caller=lambda: Caller(scope_id="dev", identity="me"))

Three hooks are what a multi-tenant host needs, so no host rebuilds :meth:`CatalogueTool.over` to get them:

- **The host, per call.** ``host_resolver`` in place of ``host`` answers which
  :class:`~threetears.evals.ops.OpsHost` this call works in — a tenant's storage and job manager —
  read from the request context like the caller is. Exactly one of the two is given.
- **One ``around`` hook** wraps every call for the host's tracing, metrics or audit log
  (:data:`CallHook`). It sees the call and its outcome and cannot change either.
- **A refusal as data.** A refused call is an MCP error result (``isError``) carrying the refusal's
  teaching text, and its :class:`~threetears.evals.actions.Refusal` as structured content —
  ``{"refusal": {"kind": ..., "status": ...}}`` — so a client branches on the kind, never on the text. A
  result carries its text and the result model's JSON as structured content.

**One fixed call timeout.** No action's call waits on a model — work that calls one runs as a job the caller
polls — so a call is a few store round-trips, and every call is held to one fixed timeout
(:data:`DEFAULT_CALL_TIMEOUT_S`, or the ``call_timeout_s`` a host mounts with), never one computed from what the
call might spend. A call that outlives it is answered as an error saying so, rather than holding the client. The
timeout holds a call up to its action's outcome; an ``around`` hook's code after its yield runs past it, under
:data:`HOOK_TAIL_TIMEOUT_S` of its own, so a finished action's outcome always reaches its caller.

**This module is a public root.** A host imports from here and from no module below it.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, TypeVar

from fastmcp import FastMCP
from fastmcp.tools import Tool
from fastmcp.tools.base import ToolResult
from pydantic import ConfigDict, Field

from threetears.evals.actions import ActionCatalogue, ActionOutcome, Caller, MountedTool, ToolSpec, standard_tools
from threetears.evals.ops import OpsHost
from threetears.observe import get_logger

log = get_logger(__name__)

#: How long one call may take, in seconds, unless a host mounts with another: every call, whatever its action. No call
#: waits on a model, so this bounds store round-trips and a host's own caller resolution, never a model's latency.
DEFAULT_CALL_TIMEOUT_S = 60.0

#: How long an ``around`` hook may take, in seconds, past the outcome it was handed — or closing, once a call is
#: stopped. The call's own timeout ends when the action does, so the outcome reaches the caller whatever the hook
#: does after it; a hook still running past this is cut short and logged, and the outcome stands.
HOOK_TAIL_TIMEOUT_S = 10.0


class CallTimedOut(Exception):
    """A call outlived the call timeout and was stopped before its action finished.

    Raised by :meth:`ToolBinding.call` only for the call's own deadline — a ``TimeoutError`` the action or a resolver
    raises of its own propagates as itself — and answered by :class:`CatalogueTool` as an error result saying so.

    Attributes:
        timeout_s: The timeout it outlived, in seconds.
    """

    def __init__(self, timeout_s: float) -> None:
        """Name the timeout the call outlived.

        Args:
            timeout_s: The timeout, in seconds.
        """
        super().__init__(f"the call outlived its {timeout_s:g}s call timeout and was stopped")
        self.timeout_s = timeout_s


#: Resolves who is calling, and their scope, for one call — sync or async. The host reads its own
#: request context (an access token, a session) to answer it.
CallerResolver = Callable[[], Caller | Awaitable[Caller]]

#: Resolves the host one call works in — sync or async — for a server that serves more than one: a
#: tenant's storage, job manager and settings, read from the same request context as the caller. It is
#: asked after the caller, once per call.
HostResolver = Callable[[], OpsHost | Awaitable[OpsHost]]


@dataclass(frozen=True, kw_only=True)
class ToolCall:
    """One call, as an ``around`` hook sees it before the catalogue carries it out.

    Attributes:
        tool: The tool called.
        action: The ``action`` argument as the call gave it — not yet checked against the tool, so it
            may name no action at all; ``None`` when the call gave none or a non-string. The outcome's
            ``action`` is the action the catalogue resolved.
        caller: Who is calling, and their scope, as the caller resolver answered.
    """

    tool: str
    action: str | None
    caller: Caller


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
#: runs. A call that raises — a defect, not a refusal — is thrown in at the yield, so a ``with`` or
#: ``finally`` closes; the hook cannot catch it on the call's behalf, since it propagates whatever the
#: hook does. **It cannot change the outcome**: what it yields is discarded, and the outcome is frozen.
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

_T = TypeVar("_T")


async def _resolved(value: _T | Awaitable[_T]) -> _T:
    """A resolver's answer, awaited when it is awaitable."""
    return await value if inspect.isawaitable(value) else value


class ToolBinding:
    """What a :class:`CatalogueTool` hands each call to: the mounted tool, the host, the caller resolver and the hook.

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
        call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
    ) -> None:
        """Bind a mounted tool to the host, caller resolver, hook and call timeout its calls run under.

        Args:
            mounted: The tool cut from the catalogue.
            host: The one host its actions work in; or
            host_resolver: Resolves the host per call. Exactly one of ``host`` and ``host_resolver``.
            caller: Resolves who is calling, per call.
            around: Wraps every call (:data:`CallHook`); ``None`` for none.
            call_timeout_s: The one timeout every call is held to, in seconds.

        Raises:
            ValueError: Both or neither of ``host`` and ``host_resolver`` were given, or ``call_timeout_s`` is
                not positive.
            TypeError: ``around`` is not an async generator function.
        """
        if not call_timeout_s > 0:
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
        self.call_timeout_s = call_timeout_s

    async def call(self, arguments: dict[str, Any]) -> ActionOutcome:
        """Carry out one call: resolve the caller, then the host, and hand the call to the catalogue under the hook.

        The call timeout holds everything up to the action's outcome: the resolvers, the hook's code before its
        yield, and the action. The hook's code after its yield runs once the outcome is in hand, under
        :data:`HOOK_TAIL_TIMEOUT_S` of its own, and cannot replace it (:data:`CallHook`).

        Args:
            arguments: The call's arguments, ``action`` among them.

        Returns:
            The catalogue's outcome, exactly as it returned it.

        Raises:
            CallTimedOut: The call outlived the call timeout before its action finished; the action was stopped,
                and the hook closed.
            TypeError: A resolver answered with something other than a
                :class:`~threetears.evals.actions.Caller` or an :class:`~threetears.evals.ops.OpsHost`.
            Exception: Whatever a resolver raises, before the hook is entered and before any action runs —
                with no caller there is no one to run it as — and whatever the catalogue lets propagate (a
                defect). A refusal is an outcome, not an exception.
        """
        deadline = asyncio.timeout(self.call_timeout_s)
        hook: AsyncGenerator[None, ActionOutcome] | None = None
        observed: ToolCall | None = None
        try:
            async with deadline:
                caller = await _resolved(self.caller())
                if not isinstance(caller, Caller):
                    raise TypeError(
                        f"tool {self.mounted.name!r}: the caller resolver answered {caller!r}, not a Caller"
                    )
                host = self.host
                if self.host_resolver is not None:
                    host = await _resolved(self.host_resolver())
                    if not isinstance(host, OpsHost):
                        raise TypeError(
                            f"tool {self.mounted.name!r}: the host resolver answered {host!r}, not an OpsHost"
                        )
                assert host is not None  # one of the two is set, which __init__ holds
                if self.around is not None:
                    action = arguments.get("action")
                    observed = ToolCall(
                        tool=self.mounted.name, action=action if isinstance(action, str) else None, caller=caller
                    )
                    hook = await _entered(self.around, observed)
                try:
                    outcome = await self.mounted.call(arguments, host=host, caller=caller)
                except BaseException as defect:
                    if hook is not None and observed is not None:
                        await _closed_within(hook, observed, defect)
                    raise
        except TimeoutError:
            # A TimeoutError the caller resolver or the action raised of its own is not this call outliving its
            # timeout, and answering it as one would blame the store for the action's own fault: it propagates, as
            # anything else the action raises does.
            if not deadline.expired():
                raise
            raise CallTimedOut(self.call_timeout_s) from None
        if hook is not None and observed is not None:
            await _finished_within(hook, observed, outcome)
        return outcome


async def _entered(around: CallHook, call: ToolCall) -> AsyncGenerator[None, ActionOutcome] | None:
    """The hook, run to its yield; ``None`` when it failed there, which is logged and leaves the call unobserved."""
    hook = around(call)
    try:
        await anext(hook)
    except Exception:
        log.exception("around hook failed before its yield on %s/%s; the call runs unobserved", call.tool, call.action)
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
            except Exception:
                log.exception("around hook failed after its yield on %s/%s; the outcome stands", call.tool, call.action)
                return
            log.error(
                "around hook yielded twice on %s/%s; it yields once, and the outcome stands", call.tool, call.action
            )
            await _close(hook, call, None)
    except TimeoutError:
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
        if raised is not defect:
            log.exception("around hook failed closing %s/%s", call.tool, call.action)
        return
    # It caught the defect and yielded again: close it, so it does not linger half-run.
    await _close(hook, call, None)


class CatalogueTool(Tool):
    """One mounted tool, as FastMCP serves it: every call handed to the catalogue.

    Attributes:
        binding: What each call is handed to. Excluded from the tool's serialized form: a client is shown
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
        call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
    ) -> CatalogueTool:
        """The FastMCP tool for one mounted tool.

        Args:
            mounted: The tool cut from the catalogue.
            host: The one host its actions work in; or
            host_resolver: Resolves the host per call. Exactly one of ``host`` and ``host_resolver``.
            caller: Resolves who is calling, per call.
            around: Wraps every call (:data:`CallHook`); ``None`` for none.
            call_timeout_s: The one timeout every call is held to, in seconds.

        Returns:
            The tool, ready for :meth:`fastmcp.FastMCP.add_tool`.

        Raises:
            ValueError: Both or neither of ``host`` and ``host_resolver`` were given.
            TypeError: ``around`` is not an async generator function.
        """
        binding = ToolBinding(
            mounted,
            host=host,
            host_resolver=host_resolver,
            caller=caller,
            around=around,
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

        Args:
            arguments: The call's arguments, ``action`` among them.

        Returns:
            The result: text always; as structured content, the result model's JSON when the call
            succeeded, or ``{"refusal": {"kind": ..., "status": ...}}`` with the error flag when it was
            refused; and the error flag alone, with text saying so, when it outlived the call timeout.
        """
        binding = self.binding
        try:
            # The call up to its outcome — the caller and host resolvers, the hook before its yield, the action — is
            # held to the one timeout; the hook after its yield is not (ToolBinding.call).
            outcome = await binding.call(arguments)
        except CallTimedOut as stopped:
            return ToolResult(
                content=(
                    f"the call outlived this server's {stopped.timeout_s:g}s call timeout and was stopped. No "
                    "action's call waits on a model — work that does runs as a job — so this is a fault of the "
                    "server's store or of the action, not something to wait out: report it"
                ),
                is_error=True,
            )
        structured = {"refusal": asdict(outcome.refusal)} if outcome.refusal is not None else outcome.structured
        return ToolResult(content=outcome.text, structured_content=structured, is_error=outcome.is_error)


def mount_fastmcp(
    server: FastMCP,
    catalogue: ActionCatalogue,
    *,
    host: OpsHost | None = None,
    host_resolver: HostResolver | None = None,
    caller: CallerResolver,
    around: CallHook | None = None,
    tools: Sequence[ToolSpec] | None = None,
    call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
) -> tuple[str, ...]:
    """Add the catalogue's tools to a FastMCP server.

    Args:
        server: The host's server.
        catalogue: The actions to serve — :func:`~threetears.evals.actions.eval_catalogue`, with the host's own.
        host: The one host every action works in; or
        host_resolver: Resolves the host per call, for a server serving several. Exactly one of ``host``
            and ``host_resolver``.
        caller: Resolves who is calling and their scope, once per call.
        around: Wraps every call to every tool, for the host's tracing, metrics or audit
            (:data:`CallHook`); ``None`` for none.
        tools: The tools to cut; ``None`` for :func:`~threetears.evals.actions.standard_tools` under the
            default prefix. Pass :func:`~threetears.evals.actions.read_only_tools` for an agent that may
            only read, or either with the host's own prefix.
        call_timeout_s: The one timeout every call is held to, in seconds.

    Returns:
        The names of the tools added.

    Raises:
        ValueError: The catalogue refuses a tool (see :meth:`~threetears.evals.actions.ActionCatalogue.mount`),
            or both or neither of ``host`` and ``host_resolver`` were given, or ``call_timeout_s`` is not
            positive — before any tool is added.
        TypeError: ``around`` is not an async generator function, before any tool is added.
    """
    mounted = catalogue.mount_all(tools if tools is not None else standard_tools())
    served = [
        CatalogueTool.over(
            tool,
            host=host,
            host_resolver=host_resolver,
            caller=caller,
            around=around,
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
    "CallHook",
    "CallTimedOut",
    "CallerResolver",
    "CatalogueTool",
    "HostResolver",
    "ToolBinding",
    "ToolCall",
    "mount_fastmcp",
]
