"""LangGraph node factories for context enrichment and auto-save.

These nodes plug into a LangGraph StateGraph to provide:

- **Context enrichment**: Before the agent runs, search for related items
  and inject them as system messages. Configurable via an ``entity_searcher``
  callback and optional keyword gating.

- **Context save**: After the agent responds, scan tool call results and
  auto-save significant outputs to the conversation context store. Configurable
  via a ``saveable_tools`` set and optional chunker.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import SystemMessage, ToolMessage

from threetears.agent.tools.chunker import ChunkResult, chunk_content
from threetears.agent.tools.context import ToolContextManager
from threetears.core.logging import get_logger

logger = get_logger(__name__)


# -- Types for pluggable callbacks --

EntitySearcher = Callable[[str], Awaitable[list[Any]]]
"""Async callback: (user_message_text) -> list of relevant entities."""

EntityFormatter = Callable[[list[Any]], str]
"""Format a list of entities into a system prompt string."""


# ======================================================================
# Context Enrichment Node
# ======================================================================


def create_context_enrichment_node(
    entity_searcher: EntitySearcher,
    entity_formatter: EntityFormatter,
    *,
    context_manager: ToolContextManager | None = None,
    keywords: frozenset[str] | None = None,
    min_ledger_coverage: int = 3,
) -> Any:
    """Create a pre-agent context enrichment node.

    Searches for related entities and injects them as a SystemMessage.
    When ``keywords`` is provided, only enriches if the user message
    contains at least one keyword. When a ``context_manager`` is
    provided, skips enrichment if the ledger already has sufficient
    coverage.

    :param entity_searcher: async callback that searches for relevant entities
    :ptype entity_searcher: EntitySearcher
    :param entity_formatter: formats entities into a prompt string
    :ptype entity_formatter: EntityFormatter
    :param context_manager: optional context manager for ledger checking
    :ptype context_manager: ToolContextManager | None
    :param keywords: optional keyword gate — only enrich if message contains one
    :ptype keywords: frozenset[str] | None
    :param min_ledger_coverage: skip enrichment if ledger has this many items
    :ptype min_ledger_coverage: int
    :return: async node function for LangGraph
    :rtype: Any
    """

    async def enrichment_node(state: dict[str, Any]) -> dict[str, Any]:
        """Search for related entities and inject as context.

        :param state: graph state with messages
        :ptype state: dict
        :return: state update with optional system message
        :rtype: dict
        """
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}

        last_msg = messages[-1]
        content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        if keywords is not None:
            words = set(content.lower().split())
            if not (words & keywords):
                logger.debug("Enrichment skipped: no matching keywords")
                return {"messages": []}

        if context_manager is not None and len(context_manager.ledger) >= min_ledger_coverage:
            logger.debug("Enrichment skipped: ledger has sufficient coverage")
            return {"messages": []}

        try:
            entities = await entity_searcher(content)
        except Exception:
            logger.warning("Context enrichment search failed", exc_info=True)
            return {"messages": []}

        if not entities:
            return {"messages": []}

        logger.info("Enrichment injecting entities", extra={"count": len(entities)})
        context_text = entity_formatter(entities)
        return {"messages": [SystemMessage(content=context_text)]}

    return enrichment_node


# ======================================================================
# Context Save Node
# ======================================================================

_DEFAULT_SAVEABLE_TOOLS = frozenset({"web_search", "web_fetch"})
_MAX_SAVE_CONTENT = 4000


def create_context_save_node(
    context_manager: ToolContextManager,
    *,
    saveable_tools: frozenset[str] | None = None,
    saveable_suffixes: tuple[str, ...] = (),
    max_content: int = _MAX_SAVE_CONTENT,
) -> Any:
    """Create a post-response context save node.

    Scans ToolMessages for results from saveable tools and persists
    them to the conversation context store. Optionally chunks large
    results using the chunker.

    :param context_manager: conversation context manager
    :ptype context_manager: ToolContextManager
    :param saveable_tools: tool names whose output should be saved
    :ptype saveable_tools: frozenset[str] | None
    :param saveable_suffixes: tool name suffixes to match (e.g. ("_scan",))
    :ptype saveable_suffixes: tuple[str, ...]
    :param max_content: maximum content length before truncation
    :ptype max_content: int
    :return: async node function for LangGraph
    :rtype: Any
    """
    tools_to_save = saveable_tools or _DEFAULT_SAVEABLE_TOOLS

    def _is_saveable(tool_name: str | None) -> bool:
        """Check if a tool's results should be saved.

        :param tool_name: tool name
        :ptype tool_name: str | None
        :return: True if saveable
        :rtype: bool
        """
        if tool_name is None:
            return False
        if tool_name in tools_to_save:
            return True
        return any(tool_name.endswith(suffix) for suffix in saveable_suffixes)

    async def context_save_node(state: dict[str, Any]) -> dict[str, Any]:
        """Scan tool results and auto-save significant content.

        :param state: graph state with messages
        :ptype state: dict
        :return: state update (always empty messages)
        :rtype: dict
        """
        messages = state.get("messages", [])
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

        for msg in tool_messages:
            tool_name: str | None = getattr(msg, "name", None)
            if not _is_saveable(tool_name):
                continue
            assert tool_name is not None  # narrowed by _is_saveable

            raw = msg.content or ""
            content: str = raw if isinstance(raw, str) else str(raw)
            if len(content) > max_content:
                content = content[:max_content] + "\n[Content truncated]"

            short_desc = content[:200]
            long_desc = content[:1000]
            key = f"{tool_name}:{msg.tool_call_id}"

            try:
                ctx_id = await context_manager.save_tool_result(
                    tool_name=key,
                    result=content,
                    short_desc=short_desc,
                    long_desc=long_desc,
                    context_type="tool_result",
                )
                chunks = chunk_content(content, strategy_hint=tool_name)
                for chunk in chunks:
                    await _save_chunk(context_manager, ctx_id, chunk)
            except Exception:
                logger.warning("Failed to auto-save tool result", extra={"tool": tool_name})

        return {"messages": []}

    return context_save_node


async def _save_chunk(context_manager: ToolContextManager, ctx_id: str, chunk: ChunkResult) -> None:
    """Save a single memory chunk if the context manager supports it.

    :param context_manager: context manager
    :ptype context_manager: ToolContextManager
    :param ctx_id: parent context item ID
    :ptype ctx_id: str
    :param chunk: chunk to save
    :ptype chunk: ChunkResult
    """
    if not hasattr(context_manager, "save_chunk"):
        return
    try:
        await context_manager.save_chunk(
            context_id=ctx_id,
            chunk_index=chunk.chunk_index,
            short_desc=chunk.short_desc,
            long_desc=chunk.long_desc,
            content=chunk.content,
        )
    except Exception:
        logger.warning("Failed to save chunk", extra={"chunk_index": chunk.chunk_index})
