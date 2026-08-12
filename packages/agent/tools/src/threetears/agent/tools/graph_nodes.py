"""LangGraph node factories for context enrichment and auto-save.

These nodes plug into a LangGraph StateGraph to provide:

- **Context enrichment**: Before the agent runs, search for related items
  and inject them as system messages. Configurable via an ``entity_searcher``
  callback and optional keyword gating.

- **Context save**: After the agent responds, scan tool call results and
  auto-save significant outputs to the conversation context store. Configurable
  via a ``saveable_tools`` set and optional chunker.

Retention posture -- read this before wiring the save node
==========================================================

This module holds the seam where retrieved third-party content becomes *our
own* stored content, and `search-requirements.md` C8 requires the posture
stated **before** the first byte is retained rather than after. The node has
been shipped-but-inert since it was written (it matched bare tool names that
nothing binds), so that ordering is still available; wiring it into a graph is
what starts retention, and the wiring is deliberately not done here.

What is retained, when a host wires this node:

- **The rendered prose of a tool result**, truncated (``max_content``), plus
  chunks of it. This is the agent's working memory of what it read.
- **The typed structure a search result carried**, when it carried any --
  the query, candidate identities and titles, provider notices, and any typed
  failure, read off ``ToolResult.metadata`` under
  :data:`~threetears.search.contracts.SEARCH_RESULTS_METADATA_KEY` (D22). This
  is retained *because* it is what makes the prose re-checkable: a stored
  4000-character truncation with no provenance is a claim nobody can trace back
  (SR-A3), and structure is what turns it back into a citation.

The rules that govern it, none of which this module decides:

- **Queries are user content** (D11/SR-K2). The capability makes the query
  available for redaction; it does not redact on its own, because redaction
  policy is an opinion from above (P1). A host with a redaction policy applies
  it to what this node writes -- the node deliberately holds no policy of its
  own to be overridden or forgotten.
- **Retention and purge follow the consumer's policy** (D7/D12). The bytes land
  in the consumer's own context store, which is exactly what makes a retention
  stance dischargeable rather than aspirational.
- **Fetched page text is third-party content** and the deployment's agreement
  with the site governs keeping it (SR-K4/D12), the same way ``respect_robots``
  is deployment config rather than code.

**A rename is a data-retention change.** The node binds on what a result *is*
before it binds on what the tool is *called*, precisely so that renaming or
splitting a tool cannot silently change what gets remembered. It already fired
once in the silent-off direction -- see :data:`_DEFAULT_SAVEABLE_TOOLS`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import SystemMessage, ToolMessage

from threetears.agent.tools.chunker import ChunkResult, chunk_content
from threetears.agent.tools.context import ToolContextManager
from threetears.observe import get_logger
from threetears.search.contracts import SEARCH_RESULTS_METADATA_KEY, SearchResultsMetadata

__all__ = [
    "EntityFormatter",
    "EntitySearcher",
    "create_context_enrichment_node",
    "create_context_save_node",
]

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
    :param context_manager: optional context manager for surfaced-refs
        coverage checking
    :ptype context_manager: ToolContextManager | None
    :param keywords: optional keyword gate — only enrich if message contains one
    :ptype keywords: frozenset[str] | None
    :param min_ledger_coverage: skip enrichment if surfaced-refs
        projection has this many items
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

        if context_manager is not None and context_manager.memory_refs_count >= min_ledger_coverage:
            logger.debug("Enrichment skipped: surfaced-refs projection has sufficient coverage")
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

#: tools whose output is saved when a host names none of its own.
#:
#: These are the names the adapter actually **binds** -- ``mcp_name()``, not the
#: bare strings. They read as redundant with the structure check in
#: :func:`_search_structure_of` and are not: a tool can be saveable by name
#: without carrying search structure, and this set is what a host overrides.
#:
#: This constant is where C8 fired. It held ``{"web_search", "web_fetch"}``,
#: matched by exact equality against ``ToolMessage.name``, while
#: ``langchain_adapter`` binds every tool under its namespaced name -- so the
#: default set never matched anything, the node was inert in production, and
#: the tests could not see it because they all passed bare names explicitly.
#: A retention path that silently does nothing is the same class of defect as
#: one that silently does too much.
_DEFAULT_SAVEABLE_TOOLS = frozenset({"threetears.web_search", "threetears.web_fetch"})
_MAX_SAVE_CONTENT = 4000

#: how many candidate records ride the saved metadata. The prose is already
#: truncated; this bounds the structure beside it so one enormous result set
#: cannot turn a context row into a page dump.
_MAX_SAVED_CANDIDATES = 20


def _search_structure_of(message: ToolMessage) -> SearchResultsMetadata | None:
    """Read the typed search result off a ``ToolMessage``, if it carries one.

    Both ``web_search`` and ``web_fetch`` project under the same key, and this
    node wants either -- unlike a consumer that must tell them apart, what
    matters here is only that the result carries re-checkable provenance.

    A payload too new to read is skipped rather than raised: this node runs
    post-turn inside a graph, where raising would fail a turn whose real work
    already succeeded. The prose still gets saved; only the structure is lost.

    :param message: the tool message to inspect
    :ptype message: ToolMessage
    :return: the typed projection, or None when there is none to read
    :rtype: SearchResultsMetadata | None
    """
    artifact = getattr(message, "artifact", None)
    if not isinstance(artifact, dict):
        return None
    payload = artifact.get(SEARCH_RESULTS_METADATA_KEY)
    if not isinstance(payload, dict):
        return None
    try:
        return SearchResultsMetadata.from_metadata(payload)
    except ValueError:
        logger.warning("Context save could not read search structure; saving prose only", exc_info=True)
        return None


def _provenance_of(structure: SearchResultsMetadata) -> dict[str, Any]:
    """Project a search result into the facts worth storing beside the prose.

    Not the whole projection: candidate content slots can carry entire page
    bodies, and this is a context row, not a second copy of the corpus. What is
    kept is what makes the saved prose traceable (SR-A3) -- the query it
    answers, where each result can be re-read, and any reason the answer was
    partial.

    :param structure: the typed projection read off the message
    :ptype structure: SearchResultsMetadata
    :return: a JSON-safe provenance record for ``save_tool_result``'s metadata
    :rtype: dict[str, Any]
    """
    record: dict[str, Any] = {
        "schema_version": structure.schema_version,
        "query": structure.query,
        "candidate_count": len(structure.candidates),
        "candidates": [
            {"identity": candidate.identity, "title": candidate.title}
            for candidate in structure.candidates[:_MAX_SAVED_CANDIDATES]
        ],
    }
    if len(structure.candidates) > _MAX_SAVED_CANDIDATES:
        record["candidates_truncated"] = True
    if structure.notices:
        record["notices"] = list(structure.notices)
    if structure.failure is not None:
        # A failed search is worth remembering as a failure. Storing its prose
        # without this would leave a row that reads like a thin answer rather
        # than like no answer at all.
        record["failure_class"] = structure.failure.failure_class
    return record


def create_context_save_node(
    context_manager: ToolContextManager,
    *,
    saveable_tools: frozenset[str] | None = None,
    saveable_suffixes: tuple[str, ...] = (),
    max_content: int = _MAX_SAVE_CONTENT,
) -> Any:
    """Create a post-response context save node.

    Scans ToolMessages and persists the saveable ones to the conversation
    context store, chunking large results. A message is saveable if it carries
    typed search structure **or** its tool name matches; see this module's
    docstring for the retention posture that governs what that stores, and read
    it before wiring this node into a graph.

    :param context_manager: conversation context manager
    :ptype context_manager: ToolContextManager
    :param saveable_tools: tool names whose output should be saved; these are
        matched against the *bound* name a ``ToolMessage`` carries
        (``threetears.web_search``), not the bare one. ``None`` uses
        :data:`_DEFAULT_SAVEABLE_TOOLS`; an **empty** set means no tool is
        saveable by name -- it no longer falls back to the defaults, so
        "structure only" is expressible and an empty set cannot silently mean
        its opposite
    :ptype saveable_tools: frozenset[str] | None
    :param saveable_suffixes: tool name suffixes to match (e.g. ("_scan",))
    :ptype saveable_suffixes: tuple[str, ...]
    :param max_content: maximum content length before truncation
    :ptype max_content: int
    :return: async node function for LangGraph
    :rtype: Any
    """
    tools_to_save = saveable_tools if saveable_tools is not None else _DEFAULT_SAVEABLE_TOOLS

    def _is_saveable(tool_name: str | None) -> bool:
        """Check if a tool's results should be saved, by name.

        The name test alone; :func:`_search_structure_of` carries the
        type test, and the node takes either.

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
            # Structure first, name second: a result that carries re-checkable
            # provenance is worth keeping whatever the tool ended up being
            # called, which is what stops a rename from being a silent
            # retention change in either direction (C8).
            structure = _search_structure_of(msg)
            if structure is None and not _is_saveable(tool_name):
                continue
            if tool_name is None:
                tool_name = "unnamed_tool"

            raw = msg.content or ""
            content: str = raw if isinstance(raw, str) else str(raw)
            if len(content) > max_content:
                content = content[:max_content] + "\n[Content truncated]"

            short_desc = content[:200]
            long_desc = content[:1000]
            key = f"{tool_name}:{msg.tool_call_id}"

            metadata: dict[str, Any] | None = None
            fingerprint: str | None = None
            if structure is not None:
                # The same key the payload rides under at the tool border (D22),
                # deliberately: a literal here would be a second name for one
                # payload, free to drift from the reader -- the defect class this
                # whole change exists to close.
                metadata = {SEARCH_RESULTS_METADATA_KEY: _provenance_of(structure)}
                # The query IS the input, so it is the honest dedup key: asking
                # the same thing twice in one conversation refreshes the row
                # rather than stacking a second copy of the same page.
                fingerprint = structure.query or None
                if structure.candidates:
                    short_desc = f"{len(structure.candidates)} result(s) for {structure.query!r}"

            try:
                ctx_id = await context_manager.save_tool_result(
                    tool_name=key,
                    result=content,
                    short_desc=short_desc,
                    long_desc=long_desc,
                    context_type="tool_result",
                    metadata=metadata,
                    input_fingerprint=fingerprint,
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
