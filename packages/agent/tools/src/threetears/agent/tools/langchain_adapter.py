"""LangChain adapter for :class:`TearsTool` instances.

Single source of truth for wrapping a :class:`TearsTool` as a
``langchain_core.tools.StructuredTool`` so the in-process LangChain
integration path (used by any consumer that runs tools
inside a LangGraph graph rather than across NATS via ``ToolServer``)
shares one execution code path with the NATS dispatch path.

Lives in its own module rather than on :class:`TearsTool` itself
because :mod:`threetears.agent.tools.base_tool` is enforced
platform-agnostic (no langchain imports) by
``test_no_platform_imports_in_base_tool``. Adapters that pull in
specific platforms live here.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.tools import StructuredTool

from threetears.agent.tools.base_tool import TearsTool

__all__ = ["to_langchain_tool"]


def to_langchain_tool(
    tool: TearsTool,
    description: str | None = None,
    args_schema: Any = None,
) -> StructuredTool:
    """wrap a :class:`TearsTool` instance as a LangChain ``StructuredTool``.

    produces a ``StructuredTool`` whose ``coroutine`` AND ``func`` both
    delegate to the wrapped :class:`TearsTool`'s :meth:`run`, so
    ``StructuredTool.ainvoke()`` (async) and ``StructuredTool.invoke()``
    (sync) both work without the caller having to know which path the
    tool's logic uses internally. previously each builtin's
    ``create_X_tool`` factory hand-rolled its own ``StructuredTool.from_function``
    call with a sync-only ``func`` body; this helper collapses that
    duplication while preserving sync-call compatibility.

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

    the returned tool's ``name`` / ``description`` / ``args_schema``
    are populated from the wrapped :class:`TearsTool`'s
    :meth:`mcp_name` / :meth:`mcp_schema` / the supplied
    ``args_schema`` so the alias-resolver / RBAC layers see the same
    canonical names regardless of registration path.

    :param tool: the :class:`TearsTool` instance to wrap. its
        :meth:`run` is invoked on each LangChain dispatch (sync or
        async); :meth:`run` already runs the platform's input
        coercion before calling ``execute``, so the LangChain path
        and the NATS path get identical normalization
    :ptype tool: TearsTool
    :param description: optional override for the tool description.
        when ``None``, ``tool.mcp_schema().description`` is used. mostly
        relevant for factories that historically passed a
        configurable description string from per-tool config
    :ptype description: str | None
    :param args_schema: optional pydantic ``BaseModel`` subclass
        describing the structured input. when ``None`` LangChain
        infers from the wrapper signature (``**kwargs``); supply the
        hand-written ``XInput`` class for the StructuredTool to
        surface a typed input schema to the LLM
    :ptype args_schema: Any
    :return: a LangChain ``StructuredTool`` ready to bind to an LLM,
        callable via both ``.invoke()`` and ``.ainvoke()``
    :rtype: StructuredTool
    """

    async def _async_wrapper(**kwargs: Any) -> str:
        """invoke ``tool.run`` and project ``ToolResult`` to a string.

        LangChain ``StructuredTool`` coroutines return strings; the
        ``ToolResult.content`` carries the human-readable payload
        (success or error). non-success outcomes still return error
        text rather than raising so the LLM can reason about
        failures instead of seeing a Python traceback.
        """
        outcome = await tool.run(**kwargs)
        return outcome.content if outcome.content is not None else (outcome.error or "")

    def _sync_wrapper(**kwargs: Any) -> str:
        """sync entry to ``tool.run`` -- safe regardless of caller event loop.

        path A: no running loop on this thread -> ``asyncio.run`` directly.
        path B: running loop on this thread -> submit to a one-shot
        ``ThreadPoolExecutor`` whose worker runs a fresh
        ``asyncio.run`` and returns the result. blocks the caller
        thread on the future, never re-enters the caller's loop.

        the path B branch is what makes this safe for callers like
        LangChain's ``StructuredTool.invoke()`` invoked from inside
        an async test or async LangGraph node.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_async_wrapper(**kwargs))
        # running loop detected -- isolate the new run on a worker thread.
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _async_wrapper(**kwargs))
            return future.result()

    schema = tool.mcp_schema()
    return StructuredTool.from_function(
        func=_sync_wrapper,
        coroutine=_async_wrapper,
        name=tool.mcp_name(),
        description=description if description is not None else schema.description,
        args_schema=args_schema,
    )
