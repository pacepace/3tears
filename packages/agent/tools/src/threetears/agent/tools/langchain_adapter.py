"""LangChain adapter for :class:`TearsTool` instances.

The one way to wrap a :class:`TearsTool` as a
``langchain_core.tools.StructuredTool``, for any consumer that runs
tools inside a LangGraph graph rather than across NATS via
``ToolServer``. The wrapped tool behaves as the NATS path does:

* the model is shown the tool's own ``mcp_schema().input_schema`` --
  the schema the ToolServer registers -- as a JSON Schema
  ``args_schema``, so there is one schema per tool and nothing to
  drift from it;
* LangChain does not validate a JSON Schema ``args_schema``, so the
  arguments the model sent reach :meth:`TearsTool.run` as sent and its
  input coercion runs, exactly as on the NATS path. A pydantic
  ``args_schema`` was validated first: a list sent as a JSON string
  was refused before coercion saw it, nested values arrived as model
  instances instead of dicts, and omitted fields arrived filled with
  defaults. A tool that brought no pydantic model had its schema
  inferred from the wrapper's ``**kwargs`` -- one ``kwargs`` field --
  and every argument was dropped;
* a failed :class:`ToolResult` answers as a failed tool call (a
  ``ToolMessage`` with ``status="error"`` whose content names the
  error), never as a success, and keeps its metadata as the artifact.

* the tool runs inside the same
  :class:`~threetears.agent.tools.call_scope.ToolCallScope` the
  ToolServer installs around a dispatch, built by the same function
  (:func:`~threetears.agent.tools.call_scope.build_call_scope`) from the
  graph config's ``call_context``, so a tool reading the call's
  conversation, user, timezone or tokens reads them on either path.

What this path does NOT do that the ToolServer does: it applies no
``requires_confirmation`` gate -- the wrapped tool carries the flag for
the graph's own gate to read. See :func:`to_langchain_tool`.

Lives in its own module rather than on :class:`TearsTool` itself
because :mod:`threetears.agent.tools.base_tool` is enforced
platform-agnostic (no langchain imports) by
``test_no_platform_imports_in_base_tool``. Adapters that pull in
specific platforms live here.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool, ToolException

from threetears.media.contracts import ObjectStore

from threetears.agent.tools.base_tool import TearsTool, ToolResult
from threetears.agent.tools.call_scope import ContextFactory, build_call_scope, enter_call_scope
from threetears.agent.tools.context_envelope import CallContext

if TYPE_CHECKING:
    from threetears.agent.tools.engagement_resolver import EngagementScopeResolver
    from threetears.agent.tools.object_resolver import ObjectResolver

__all__ = ["to_langchain_tool"]

#: the id of the tool call a wrapped TearsTool is answering, while it answers one; ``None`` when
#: it was invoked with bare arguments. set by :class:`_TearsStructuredTool`'s ``invoke`` /
#: ``ainvoke``.
_answering_tool_call: ContextVar[str | None] = ContextVar("threetears_answering_tool_call", default=None)


def _tool_call_id(tool_input: Any) -> str | None:
    """the id of the tool call ``tool_input`` is, or ``None`` when it is bare arguments.

    :param tool_input: what the tool was invoked with
    :ptype tool_input: Any
    :return: the call's id
    :rtype: str | None
    """
    is_call = isinstance(tool_input, dict) and tool_input.get("type") == "tool_call"
    call_id = tool_input.get("id") if is_call else None
    return call_id if isinstance(call_id, str) else None


class _TearsStructuredTool(StructuredTool):
    """a ``StructuredTool`` that tells the wrapped TearsTool which tool call it is answering.

    LangChain marks a tool message failed only when the tool raises a
    ``ToolException``, and on that path it drops the artifact. a failed
    TearsTool result carries its typed failure record in its metadata --
    a refused search, a fetch that read nothing -- and callers read it
    off the artifact rather than parsing prose. so a failure answering a
    tool call builds its own ``ToolMessage`` (``status="error"``,
    artifact kept), which LangChain passes through as it stands. that
    needs the call's id, which arrives with the call at ``invoke`` /
    ``ainvoke`` -- where LangGraph's ``ToolNode`` and
    :class:`~threetears.agent.tools.executor.ToolExecutor` hand it over --
    and reaches nothing below them.

    it also carries the wrapped tool's ``requires_confirmation``: a gate
    reads the flag off the tool it is handed -- the aibots SDK's
    confirmation middleware with ``getattr`` -- and a ``StructuredTool``
    declares no such field, so without this every wrapped tool read as
    ungated.
    """

    requires_confirmation: bool = False

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:  # noqa: A002
        """invoke the tool with the call's id visible to the wrapper.

        :param input: a tool call, or the tool's bare arguments
        :ptype input: Any
        :param config: the runnable config
        :ptype config: RunnableConfig | None
        :param kwargs: forwarded to ``StructuredTool.invoke``
        :ptype kwargs: Any
        :return: the tool's output
        :rtype: Any
        """
        token = _answering_tool_call.set(_tool_call_id(input))
        try:
            output = super().invoke(input, config, **kwargs)
        finally:
            _answering_tool_call.reset(token)
        return output

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:  # noqa: A002
        """invoke the tool asynchronously with the call's id visible to the wrapper.

        :param input: a tool call, or the tool's bare arguments
        :ptype input: Any
        :param config: the runnable config
        :ptype config: RunnableConfig | None
        :param kwargs: forwarded to ``StructuredTool.ainvoke``
        :ptype kwargs: Any
        :return: the tool's output
        :rtype: Any
        """
        token = _answering_tool_call.set(_tool_call_id(input))
        try:
            output = await super().ainvoke(input, config, **kwargs)
        finally:
            _answering_tool_call.reset(token)
        return output


def _failure_text(tool_name: str, outcome: ToolResult) -> str:
    """what a failed result says, for the model.

    the error names what went wrong; content the tool wrote beside
    it -- partial output, a remediation -- follows when it says
    something the error does not. a failure that says nothing is
    named as one rather than handed over as an empty message.

    :param tool_name: the tool's canonical name
    :ptype tool_name: str
    :param outcome: the failed result
    :ptype outcome: ToolResult
    :return: the failure's text
    :rtype: str
    """
    parts = [part for part in (outcome.error, outcome.content) if part]
    if len(parts) == 2 and parts[0] == parts[1]:
        parts = parts[:1]
    return "\n\n".join(parts) if parts else f"{tool_name} failed and gave no reason."


def _call_context(config: RunnableConfig) -> CallContext | None:
    """the call's identity, as the graph config carries it, or ``None`` when it carries none.

    Read from ``config["configurable"]["call_context"]``, where the aibots SDK's transport puts the
    :class:`CallContext` of every turn -- the key its own tool wrapper reads.

    :param config: the graph config LangChain hands the tool
    :ptype config: RunnableConfig
    :return: the call context, or ``None``
    :rtype: CallContext | None
    :raises TypeError: when the key holds anything but a :class:`CallContext` -- a host bug, said so
        rather than run as if the call had no identity
    """
    configurable = config.get("configurable") or {}
    call_context = configurable.get("call_context")
    if call_context is not None and not isinstance(call_context, CallContext):
        raise TypeError(
            "config['configurable']['call_context'] must be a threetears CallContext, "
            f"not {type(call_context).__name__}"
        )
    return call_context


def to_langchain_tool(
    tool: TearsTool,
    description: str | None = None,
    *,
    context_factory: ContextFactory | None = None,
    object_store: ObjectStore | None = None,
    object_resolver: ObjectResolver | None = None,
    engagement_resolver: EngagementScopeResolver | None = None,
) -> StructuredTool:
    """wrap a :class:`TearsTool` instance as a LangChain ``StructuredTool``.

    produces a ``StructuredTool`` whose ``coroutine`` AND ``func`` both
    delegate to the wrapped :class:`TearsTool`'s :meth:`run`, so
    ``StructuredTool.ainvoke()`` (async) and ``StructuredTool.invoke()``
    (sync) both work without the caller having to know which path the
    tool's logic uses internally.

    the returned tool's ``name`` is the tool's :meth:`mcp_name` -- the
    canonical dotted name the alias-resolver, RBAC and registry
    layers match on, whichever path registered the tool. a provider
    whose tool-name validator rejects the dot is handled at the
    chat-model boundary by
    :mod:`threetears.models.tool_name_translation`, not by renaming
    the tool here. its ``args_schema`` is ``mcp_schema().input_schema``
    (see the module docstring for why a JSON Schema and not a pydantic
    model).

    a successful result becomes the tool message's content, with the
    result's ``metadata`` as its artifact
    (``response_format="content_and_artifact"``). a failed result
    answering a tool call becomes a tool message with
    ``status="error"``, the failure's text (see :func:`_failure_text`)
    and the result's ``metadata`` still as its artifact -- a failure's
    typed record is what a caller reads to tell "refused" from "found
    nothing" (see :class:`_TearsStructuredTool`). invoked with bare
    arguments, a failure answers with its text, through a handled
    ``ToolException``.

    each call runs inside a
    :class:`~threetears.agent.tools.call_scope.ToolCallScope`, the one
    the ToolServer installs around a dispatch, built by the same
    :func:`~threetears.agent.tools.call_scope.build_call_scope`: its
    identity is the graph config's ``config["configurable"]["call_context"]``
    (where the aibots SDK puts each turn's :class:`CallContext`), and
    its pod-level resources are the ones passed here, as a ToolServer
    takes them at construction. the caller is the agent running the
    graph, never a tool pod. when the config carries no call context
    no scope is installed: a tool that needs none -- a calculator --
    runs as before, and one that needs the call's identity fails with
    an error naming that config key
    (:func:`~threetears.agent.tools.call_scope.no_call_scope_message`).

    it applies no ``requires_confirmation`` gate: the returned tool
    CARRIES the flag, for the graph's own gate to read (the aibots
    SDK's confirmation middleware does), but nothing here pauses a call.

    sync-path event-loop safety:

    * when the caller is on a thread WITH a running event loop,
      ``asyncio.run`` would raise ``RuntimeError("cannot be called
      from a running event loop")``. the sync wrapper detects that
      via :func:`asyncio.get_running_loop` and instead submits the
      coroutine to a one-shot :class:`ThreadPoolExecutor` whose
      worker thread runs a fresh loop via :func:`asyncio.run`. the
      worker thread waits the coroutine to completion and the caller
      thread blocks on the future; no loop nesting, no
      ``nest_asyncio`` hack.
    * when the caller is on a thread WITHOUT a running event loop
      (most pytest test runs, CLI entrypoints), the sync wrapper
      uses :func:`asyncio.run` directly -- no thread overhead.

    :param tool: the :class:`TearsTool` instance to wrap. its
        :meth:`run` is invoked on each LangChain dispatch (sync or
        async) with the arguments the model sent
    :ptype tool: TearsTool
    :param description: optional override for the tool description.
        when ``None``, ``tool.mcp_schema().description`` is used.
        relevant for factories that pass a configurable description
        string from per-tool config
    :ptype description: str | None
    :param context_factory: resolves a call's conversation to its
        :class:`~threetears.agent.tools.context.ToolContextManager`,
        as the ToolServer's does; without one a scope carries no
        context manager
    :ptype context_factory: ContextFactory | None
    :param object_store: the host's streaming object store, for tools
        that produce objects
    :ptype object_store: ObjectStore | None
    :param object_resolver: the host's object-id resolver, for tools
        that consume objects
    :ptype object_resolver: ObjectResolver | None
    :param engagement_resolver: the host's engagement-scope resolver,
        for engagement-bound tools
    :ptype engagement_resolver: EngagementScopeResolver | None
    :return: a LangChain ``StructuredTool`` ready to bind to an LLM,
        callable via both ``.invoke()`` and ``.ainvoke()``
    :rtype: StructuredTool
    """
    schema = tool.mcp_schema()
    tool_name = tool.mcp_name()

    async def _answer(
        kwargs: dict[str, Any], tool_call_id: str | None, call_context: CallContext | None
    ) -> tuple[str | ToolMessage, Any]:
        """run ``tool.run`` in the call's scope and project its ``ToolResult`` to (content, artifact).

        :param kwargs: the arguments the model sent
        :ptype kwargs: dict[str, Any]
        :param tool_call_id: the id of the tool call being answered, or
            ``None`` when the tool was invoked with bare arguments
        :ptype tool_call_id: str | None
        :param call_context: the call's identity from the graph config,
            or ``None`` when it carried none -- then no scope is installed
        :ptype call_context: CallContext | None
        :return: on success, the content and the metadata as the
            artifact; on a failure answering a tool call, the failed
            ``ToolMessage`` itself, which LangChain passes through
        :rtype: tuple[str | ToolMessage, Any]
        :raises ToolException: when the result is a failure and there is
            no tool call to answer; the tool handles it
            (``handle_tool_error=True``) into the failure's text
        """
        if call_context is None:
            outcome = await tool.run(**kwargs)
        else:
            scope = await build_call_scope(
                call_context,
                principal_is_tool_pod=False,
                context_factory=context_factory,
                object_store=object_store,
                object_resolver=object_resolver,
                engagement_resolver=engagement_resolver,
            )
            async with enter_call_scope(scope):
                outcome = await tool.run(**kwargs)
        metadata = outcome.metadata if isinstance(outcome.metadata, dict) else None
        answer: tuple[str | ToolMessage, Any] = (outcome.content, metadata)
        if not outcome.success:
            text = _failure_text(tool_name, outcome)
            if tool_call_id is None:
                raise ToolException(text)
            failed = ToolMessage(
                content=text, artifact=metadata, status="error", tool_call_id=tool_call_id, name=tool_name
            )
            answer = (failed, None)
        return answer

    async def _async_wrapper(runnable_config: RunnableConfig, **kwargs: Any) -> tuple[str | ToolMessage, Any]:
        """async entry to ``tool.run``.

        ``runnable_config`` is annotated as exactly ``RunnableConfig`` because that is how LangChain
        recognises the parameter to hand the graph config to; it never reaches the tool.

        :param runnable_config: the graph config, supplied by LangChain
        :ptype runnable_config: RunnableConfig
        :param kwargs: the arguments the model sent
        :ptype kwargs: Any
        :return: see :func:`_answer`
        :rtype: tuple[str | ToolMessage, Any]
        :raises ToolException: see :func:`_answer`
        :raises TypeError: see :func:`_call_context`
        """
        return await _answer(kwargs, _answering_tool_call.get(), _call_context(runnable_config))

    def _sync_wrapper(runnable_config: RunnableConfig, **kwargs: Any) -> tuple[str | ToolMessage, Any]:
        """sync entry to ``tool.run`` -- safe regardless of caller event loop.

        path A: no running loop on this thread -> ``asyncio.run`` directly.
        path B: running loop on this thread -> submit to a one-shot
        ``ThreadPoolExecutor`` whose worker runs a fresh
        ``asyncio.run`` and returns the result. blocks the caller
        thread on the future, never re-enters the caller's loop. the
        tool call's id is read here, on the caller's thread: a worker
        thread does not inherit the caller's context.

        :param runnable_config: the graph config, supplied by LangChain
        :ptype runnable_config: RunnableConfig
        :param kwargs: the arguments the model sent
        :ptype kwargs: Any
        :return: see :func:`_answer`
        :rtype: tuple[str | ToolMessage, Any]
        :raises ToolException: see :func:`_answer`
        :raises TypeError: see :func:`_call_context`
        """
        tool_call_id = _answering_tool_call.get()
        call_context = _call_context(runnable_config)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_answer(kwargs, tool_call_id, call_context))
        # running loop detected -- isolate the new run on a worker thread.
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _answer(kwargs, tool_call_id, call_context))
            return future.result()

    return _TearsStructuredTool.from_function(
        func=_sync_wrapper,
        coroutine=_async_wrapper,
        name=tool_name,
        description=description if description is not None else schema.description,
        args_schema=schema.input_schema,
        response_format="content_and_artifact",
        handle_tool_error=True,
        requires_confirmation=bool(tool.requires_confirmation),
    )
