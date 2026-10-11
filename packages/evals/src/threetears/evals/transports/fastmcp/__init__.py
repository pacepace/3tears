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

**This module is a public root.** A host imports from here and from no module below it.
"""

from __future__ import annotations

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
    ) -> None:
        """Bind a mounted tool to the host, caller resolver and hook its calls run under.

        Args:
            mounted: The tool cut from the catalogue.
            host: The one host its actions work in; or
            host_resolver: Resolves the host per call. Exactly one of ``host`` and ``host_resolver``.
            caller: Resolves who is calling, per call.
            around: Wraps every call (:data:`CallHook`); ``None`` for none.

        Raises:
            ValueError: Both or neither of ``host`` and ``host_resolver`` were given.
            TypeError: ``around`` is not an async generator function.
        """
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

    async def call(self, arguments: dict[str, Any]) -> ActionOutcome:
        """Carry out one call: resolve the caller, then the host, and hand the call to the catalogue under the hook.

        Args:
            arguments: The call's arguments, ``action`` among them.

        Returns:
            The catalogue's outcome, exactly as it returned it.

        Raises:
            TypeError: A resolver answered with something other than a
                :class:`~threetears.evals.actions.Caller` or an :class:`~threetears.evals.ops.OpsHost`.
            Exception: Whatever a resolver raises, before the hook is entered and before any action runs —
                with no caller there is no one to run it as — and whatever the catalogue lets propagate (a
                defect). A refusal is an outcome, not an exception.
        """
        caller = await _resolved(self.caller())
        if not isinstance(caller, Caller):
            raise TypeError(f"tool {self.mounted.name!r}: the caller resolver answered {caller!r}, not a Caller")
        host = self.host
        if self.host_resolver is not None:
            host = await _resolved(self.host_resolver())
            if not isinstance(host, OpsHost):
                raise TypeError(f"tool {self.mounted.name!r}: the host resolver answered {host!r}, not an OpsHost")
        assert host is not None  # one of the two is set, which __init__ holds
        if self.around is None:
            return await self.mounted.call(arguments, host=host, caller=caller)
        action = arguments.get("action")
        observed = ToolCall(tool=self.mounted.name, action=action if isinstance(action, str) else None, caller=caller)
        return await _observed(self.around, observed, lambda: self.mounted.call(arguments, host=host, caller=caller))


async def _observed(around: CallHook, call: ToolCall, work: Callable[[], Awaitable[ActionOutcome]]) -> ActionOutcome:
    """Run ``work`` inside the hook, which sees it and cannot change it; the hook's own failures are logged.

    Args:
        around: The hook.
        call: The call, as the hook sees it.
        work: Starts the catalogue's call; started once, after the hook's code before its yield.

    Returns:
        ``work``'s outcome.
    """
    hook = around(call)
    try:
        await anext(hook)
    except Exception:
        log.exception("around hook failed before its yield on %s/%s; the call runs unobserved", call.tool, call.action)
        return await work()
    try:
        outcome = await work()
    except BaseException as defect:
        await _close(hook, call, defect)
        raise
    try:
        await hook.asend(outcome)
    except StopAsyncIteration:
        return outcome
    except Exception:
        log.exception("around hook failed after its yield on %s/%s; the outcome stands", call.tool, call.action)
        return outcome
    log.error("around hook yielded twice on %s/%s; it yields once, and the outcome stands", call.tool, call.action)
    await _close(hook, call, None)
    return outcome


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
    ) -> CatalogueTool:
        """The FastMCP tool for one mounted tool.

        Args:
            mounted: The tool cut from the catalogue.
            host: The one host its actions work in; or
            host_resolver: Resolves the host per call. Exactly one of ``host`` and ``host_resolver``.
            caller: Resolves who is calling, per call.
            around: Wraps every call (:data:`CallHook`); ``None`` for none.

        Returns:
            The tool, ready for :meth:`fastmcp.FastMCP.add_tool`.

        Raises:
            ValueError: Both or neither of ``host`` and ``host_resolver`` were given.
            TypeError: ``around`` is not an async generator function.
        """
        binding = ToolBinding(mounted, host=host, host_resolver=host_resolver, caller=caller, around=around)
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
            refused.
        """
        outcome = await self.binding.call(arguments)
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

    Returns:
        The names of the tools added.

    Raises:
        ValueError: The catalogue refuses a tool (see :meth:`~threetears.evals.actions.ActionCatalogue.mount`),
            or both or neither of ``host`` and ``host_resolver`` were given — before any tool is added.
        TypeError: ``around`` is not an async generator function, before any tool is added.
    """
    mounted = catalogue.mount_all(tools if tools is not None else standard_tools())
    served = [
        CatalogueTool.over(tool, host=host, host_resolver=host_resolver, caller=caller, around=around)
        for tool in mounted
    ]
    for tool in served:
        server.add_tool(tool)
    return tuple(tool.name for tool in served)


__all__ = [
    "CallHook",
    "CallerResolver",
    "CatalogueTool",
    "HostResolver",
    "ToolBinding",
    "ToolCall",
    "mount_fastmcp",
]
