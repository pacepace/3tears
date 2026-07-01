"""per-call execution scope for remote tools served over NATS.

the design problem: workspace tools and any other conversation-aware
tool need a :class:`ToolContextManager` scoped to the caller's
conversation. that manager carries the conversation id, the user id,
and the pointer into the per-agent ``context_items`` table, so it must
be constructed from per-call metadata arriving in the NATS envelope of
each tool invocation.

the tool instance itself is long-lived inside the pod, so the context
cannot be baked into the tool at registration time. this module
provides a :class:`contextvars.ContextVar` the tool server sets before
each dispatch and clears after, plus a zero-arg provider callable tools
wire as their ``context_provider``.

per-call identity dimensions (conversation_id, user_id, customer_id,
correlation_id, agent_id, trace) ride as a single
:class:`~threetears.agent.tools.context_envelope.CallContext` value on
:attr:`ToolCallScope.context`. new dimensions land by adding a field
to :class:`CallContext` once; scope consumers read through ``scope.context``.

usage from the tool server::

    from threetears.agent.tools.call_scope import (
        ToolCallScope,
        enter_call_scope,
    )
    from threetears.agent.tools.context_envelope import CallContext

    async def handle_call(request):
        ctx = await factory(request.context.conversation_id, request.context.user_id)
        scope = ToolCallScope(
            context=request.context or CallContext(),
            context_manager=ctx,
        )
        async with enter_call_scope(scope):
            await tool.run(**request.arguments)

usage from a tool that needs the manager::

    from threetears.agent.tools.call_scope import tool_context_provider

    register_workspace_tools(..., context_provider=tool_context_provider)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator

from threetears.media.contracts import ObjectStore

from threetears.agent.tools.context_envelope import CallContext

__all__ = [
    "ToolCallScope",
    "current_scope",
    "enter_call_scope",
    "get_current_context",
    "tool_context_provider",
]

if TYPE_CHECKING:
    from threetears.agent.tools.context import ToolContextManager
    from threetears.agent.tools.object_resolver import ObjectResolver


@dataclass(frozen=True)
class ToolCallScope:
    """metadata bound to a single tool invocation on the server side.

    the scope is constructed by the tool server from the incoming
    :class:`CallRequest` envelope, pushed onto the current asyncio task
    for the duration of :meth:`TearsTool.run`, and popped on exit. tools
    observe the scope through :func:`get_current_context` or by reading
    ``scope.context.<field>`` directly for identity dimensions.

    :param context: unified identity + trace envelope resolved from the
        wire; defaults to an empty :class:`CallContext` when the server
        was called without identity fields (stateless tools)
    :ptype context: CallContext
    :param context_manager: conversation-scoped context manager;
        ``None`` when the server has no factory wired or when the
        envelope did not carry conversation/user identifiers
    :ptype context_manager: ToolContextManager | None
    :param object_store: the pod's streaming object store, installed by
        the tool server from its single pod-level instance so producing
        tools reach it through :func:`current_scope` -- the same way they
        reach :attr:`context_manager` -- without per-tool constructor
        plumbing. ``None`` when the pod was not wired with an object store
        (no S3 configured); a producing tool that needs it fails closed at
        first use rather than running with no place to put bytes
    :ptype object_store: ObjectStore | None
    :param object_resolver: the pod's object-id resolver, installed by the tool
        server from its single self-provisioned instance so consuming tools
        reach it through :func:`current_scope` -- the same way they reach
        :attr:`object_store` -- to turn an object id into its stored key
        tenant-safely. ``None`` when the server was not wired with one (no NATS
        client, as in unit tests); a consuming tool that needs it fails closed
        at first use rather than resolving nothing
    :ptype object_resolver: ObjectResolver | None
    """

    context: CallContext = field(default_factory=CallContext)
    context_manager: "ToolContextManager | None" = None
    object_store: ObjectStore | None = None
    object_resolver: "ObjectResolver | None" = None


_current_scope: ContextVar[ToolCallScope | None] = ContextVar(
    "threetears_tool_call_scope",
    default=None,
)


@asynccontextmanager
async def enter_call_scope(
    scope: ToolCallScope,
) -> AsyncIterator[ToolCallScope]:
    """push ``scope`` onto the current task, yield it, pop on exit.

    async-context-manager so callers can ``async with`` around the tool
    dispatch site; the :class:`contextvars.ContextVar` token is reset in
    a ``finally`` block so an exception inside the body never leaks a
    stale scope into the next call handled by the same task.

    :param scope: scope to install for the duration of the ``async with``
    :ptype scope: ToolCallScope
    :return: async iterator yielding ``scope`` once
    :rtype: AsyncIterator[ToolCallScope]
    """
    token = _current_scope.set(scope)
    try:
        yield scope
    finally:
        _current_scope.reset(token)


def current_scope() -> ToolCallScope | None:
    """return the scope for the currently-dispatching tool call, or ``None``.

    :return: current scope, or ``None`` when invoked outside a dispatch
    :rtype: ToolCallScope | None
    """
    return _current_scope.get()


def get_current_context() -> ToolContextManager:
    """return the active :class:`ToolContextManager` for this call.

    tools wire this function as their ``context_provider`` parameter at
    construction time. it resolves lazily on each invocation, so the
    same tool instance can serve many concurrent conversations without
    the context leaking across them.

    :return: context manager scoped to this call
    :rtype: ToolContextManager
    :raises RuntimeError: when no scope has been installed (tool invoked
        outside a :func:`enter_call_scope` block), or when the scope
        carries no context manager (server had no factory wired, or the
        envelope lacked conversation/user identifiers)
    """
    scope = _current_scope.get()
    if scope is None:
        raise RuntimeError(
            "tool_context_provider called outside a ToolServer call scope; "
            "enter_call_scope must wrap every tool.run() dispatch"
        )
    if scope.context_manager is None:
        raise RuntimeError(
            "tool_context_provider invoked but the current call scope "
            "carries no ToolContextManager; the server was not wired with "
            "a context factory, or the envelope did not include "
            "conversation_id / user_id"
        )
    return scope.context_manager


def tool_context_provider() -> ToolContextManager:
    """zero-arg alias of :func:`get_current_context` for tool factories.

    factories accept ``context_provider: Callable[[], ToolContextManager]``;
    passing this function gives every built tool a resolver that reads
    the server's :class:`contextvars.ContextVar` at invocation time.

    :return: context manager scoped to the current call
    :rtype: ToolContextManager
    :raises RuntimeError: propagated from :func:`get_current_context`
    """
    return get_current_context()
