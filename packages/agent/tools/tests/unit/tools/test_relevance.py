"""unit tests for :mod:`threetears.agent.tools.relevance`.

Covers the contracts the sign-off artifact
(``metallm/.prawduct/artifacts/3tears-change-dynamic-tool-selection.md``)
calls out as hard requirements: top-K ordering, content-hash cache reuse +
invalidation, the embedder-failure and latency-ceiling fallbacks (both must
return the FULL unfiltered catalog, never a smaller one), and ``tool_search``
surfacing a tool absent from the initial top-K.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, StructuredTool

from threetears.agent.tools.relevance import (
    ToolRelevanceIndex,
    ToolSelectionResult,
    create_tool_search_tool,
)


def _make_tool(name: str, description: str) -> BaseTool:
    return StructuredTool.from_function(
        func=lambda: "ok",
        name=name,
        description=description,
    )


class _FakeEmbeddings(Embeddings):
    """Deterministic fake: cosine-similarity rank is driven by a caller-supplied
    ``vectors`` map (text -> vector), not real semantics. Raises / sleeps on
    demand to exercise the fallback paths.
    """

    def __init__(
        self,
        vectors: dict[str, list[float]],
        *,
        raise_on_documents: bool = False,
        raise_on_query: bool = False,
        sleep_s: float = 0.0,
    ) -> None:
        self.vectors = vectors
        self.raise_on_documents = raise_on_documents
        self.raise_on_query = raise_on_query
        self.sleep_s = sleep_s
        self.aembed_documents_calls: list[list[str]] = []
        self.aembed_query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors.get(t, [0.0, 0.0]) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.vectors.get(text, [0.0, 0.0])

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.aembed_documents_calls.append(list(texts))
        if self.sleep_s:
            await asyncio.sleep(self.sleep_s)
        if self.raise_on_documents:
            raise RuntimeError("embedder unavailable")
        return [self.vectors.get(t, [0.0, 0.0]) for t in texts]

    async def aembed_query(self, text: str) -> list[float]:
        self.aembed_query_calls.append(text)
        if self.sleep_s:
            await asyncio.sleep(self.sleep_s)
        if self.raise_on_query:
            raise RuntimeError("embedder unavailable")
        return self.vectors.get(text, [0.0, 0.0])


def _tool_text(tool: BaseTool) -> str:
    return f"{tool.name}: {tool.description or ''}"


def _catalog(n: int) -> list[BaseTool]:
    return [_make_tool(f"tool_{i}", f"description for tool {i}") for i in range(n)]


# ---------------------------------------------------------------------------
# select() -- top-K ordering + no-op case
# ---------------------------------------------------------------------------


async def test_select_returns_top_k_ordered_by_similarity() -> None:
    tools = _catalog(5)
    # tool_2 is the closest to the query vector, tool_4 second-closest.
    vectors = {
        _tool_text(tools[0]): [0.0, 1.0],
        _tool_text(tools[1]): [0.1, 0.9],
        _tool_text(tools[2]): [1.0, 0.0],
        _tool_text(tools[3]): [0.0, 1.0],
        _tool_text(tools[4]): [0.9, 0.1],
        "find the right one": [1.0, 0.0],
    }
    embedder = _FakeEmbeddings(vectors)
    index = ToolRelevanceIndex(embedder=embedder, top_k=2)

    result = await index.select(tools, "find the right one")

    assert not result.fallback_used
    assert result.fallback_reason is None
    assert [t.name for t in result.selected] == ["tool_2", "tool_4"]


async def test_select_noop_when_catalog_at_or_below_top_k() -> None:
    tools = _catalog(3)
    embedder = _FakeEmbeddings({})
    index = ToolRelevanceIndex(embedder=embedder, top_k=5)

    result = await index.select(tools, "anything")

    assert result.selected == tools
    assert not result.fallback_used
    # no-op means no embedding calls at all
    assert embedder.aembed_documents_calls == []
    assert embedder.aembed_query_calls == []


async def test_select_noop_when_catalog_size_exactly_equals_top_k() -> None:
    tools = _catalog(5)
    embedder = _FakeEmbeddings({})
    index = ToolRelevanceIndex(embedder=embedder, top_k=5)

    result = await index.select(tools, "anything")

    assert result.selected == tools
    assert not result.fallback_used
    assert embedder.aembed_documents_calls == []


async def test_select_on_empty_catalog_returns_empty() -> None:
    embedder = _FakeEmbeddings({})
    index = ToolRelevanceIndex(embedder=embedder, top_k=5)

    result = await index.select([], "anything")

    assert result.selected == []
    assert not result.fallback_used


# ---------------------------------------------------------------------------
# content-hash cache: reuse + invalidation
# ---------------------------------------------------------------------------


async def test_cache_hit_skips_reembedding_tool_descriptions() -> None:
    tools = _catalog(5)
    vectors = {_tool_text(t): [1.0, 0.0] for t in tools}
    vectors["query one"] = [1.0, 0.0]
    vectors["query two"] = [1.0, 0.0]
    embedder = _FakeEmbeddings(vectors)
    index = ToolRelevanceIndex(embedder=embedder, top_k=2)

    await index.select(tools, "query one")
    assert len(embedder.aembed_documents_calls) == 1

    await index.select(tools, "query two")
    # same tool set (same content hash) -> no second batch-embed call
    assert len(embedder.aembed_documents_calls) == 1
    # query is always re-embedded (turn-specific, never cached)
    assert embedder.aembed_query_calls == ["query one", "query two"]


async def test_cache_invalidates_on_description_edit() -> None:
    tools = _catalog(5)
    vectors = {_tool_text(t): [1.0, 0.0] for t in tools}
    vectors["query"] = [1.0, 0.0]
    embedder = _FakeEmbeddings(vectors)
    index = ToolRelevanceIndex(embedder=embedder, top_k=2)

    await index.select(tools, "query")
    assert len(embedder.aembed_documents_calls) == 1

    # Simulate an admin-editable service_tools row changing mid-conversation:
    # same tool NAME, different description -> different content hash.
    edited_tools = list(tools)
    edited_tools[0] = _make_tool(tools[0].name, "a brand new description")
    vectors[_tool_text(edited_tools[0])] = [1.0, 0.0]

    await index.select(edited_tools, "query")
    # content hash changed -> re-embedded, not served from cache
    assert len(embedder.aembed_documents_calls) == 2


async def test_cache_invalidates_on_tool_added_or_removed() -> None:
    tools = _catalog(5)
    vectors = {_tool_text(t): [1.0, 0.0] for t in tools}
    vectors["query"] = [1.0, 0.0]
    new_tool = _make_tool("tool_new", "a newly discovered MCP tool")
    vectors[_tool_text(new_tool)] = [1.0, 0.0]
    embedder = _FakeEmbeddings(vectors)
    index = ToolRelevanceIndex(embedder=embedder, top_k=2)

    await index.select(tools, "query")
    assert len(embedder.aembed_documents_calls) == 1

    await index.select(tools + [new_tool], "query")
    assert len(embedder.aembed_documents_calls) == 2


async def test_cache_is_bounded_lru() -> None:
    embedder = _FakeEmbeddings({}, )
    index = ToolRelevanceIndex(embedder=embedder, top_k=1, cache_size=2)

    # Three distinct tool sets (each > top_k so select() actually embeds),
    # bounded cache_size=2 -> the first set's entry gets evicted.
    for i in range(3):
        distinct_tools = _catalog(2)
        # make each set's content hash unique
        distinct_tools[0] = _make_tool(f"set{i}_tool_a", "a")
        distinct_tools[1] = _make_tool(f"set{i}_tool_b", "b")
        await index.select(distinct_tools, "q")

    assert len(index._cache) == 2  # noqa: SLF001 -- whitebox test of the LRU bound


# ---------------------------------------------------------------------------
# fallback contracts -- must NEVER return a smaller-than-full set
# ---------------------------------------------------------------------------


async def test_select_falls_back_to_full_catalog_on_embedder_error() -> None:
    tools = _catalog(10)
    embedder = _FakeEmbeddings({}, raise_on_documents=True)
    index = ToolRelevanceIndex(embedder=embedder, top_k=3)

    result = await index.select(tools, "anything")

    assert result.fallback_used
    assert result.fallback_reason == "embedder_error"
    assert result.selected == tools  # full, unfiltered, original order


async def test_select_falls_back_to_full_catalog_on_query_embed_error() -> None:
    tools = _catalog(10)
    vectors = {_tool_text(t): [1.0, 0.0] for t in tools}
    embedder = _FakeEmbeddings(vectors, raise_on_query=True)
    index = ToolRelevanceIndex(embedder=embedder, top_k=3)

    result = await index.select(tools, "anything")

    assert result.fallback_used
    assert result.fallback_reason == "embedder_error"
    assert result.selected == tools


async def test_select_falls_back_to_full_catalog_on_latency_ceiling() -> None:
    tools = _catalog(10)
    vectors = {_tool_text(t): [1.0, 0.0] for t in tools}
    embedder = _FakeEmbeddings(vectors, sleep_s=0.2)
    index = ToolRelevanceIndex(embedder=embedder, top_k=3, latency_ceiling_s=0.01)

    result = await index.select(tools, "anything")

    assert result.fallback_used
    assert result.fallback_reason == "latency_ceiling"
    assert result.selected == tools


async def test_fallback_result_is_never_smaller_than_full_catalog() -> None:
    """Regression shape for the review finding: a fallback must be
    indistinguishable in SIZE from "everything bound", not a silently
    smaller arbitrary set that happens to look like a real top-K result.
    """
    tools = _catalog(20)
    embedder = _FakeEmbeddings({}, raise_on_documents=True)
    index = ToolRelevanceIndex(embedder=embedder, top_k=5)

    result = await index.select(tools, "anything")

    assert len(result.selected) == len(tools)


# ---------------------------------------------------------------------------
# search() -- tool_search's underlying full-catalog query
# ---------------------------------------------------------------------------


async def test_search_ranks_full_catalog_by_relevance() -> None:
    tools = _catalog(5)
    vectors = {_tool_text(t): [0.0, 1.0] for t in tools}
    vectors[_tool_text(tools[3])] = [1.0, 0.0]
    vectors["query"] = [1.0, 0.0]
    embedder = _FakeEmbeddings(vectors)
    index = ToolRelevanceIndex(embedder=embedder, top_k=2)

    hits = await index.search(tools, "query", limit=1)

    assert [t.name for t in hits] == ["tool_3"]


async def test_search_returns_empty_on_embedder_failure_no_fallback_contract() -> None:
    tools = _catalog(5)
    embedder = _FakeEmbeddings({}, raise_on_documents=True)
    index = ToolRelevanceIndex(embedder=embedder, top_k=2)

    hits = await index.search(tools, "query")

    assert hits == []


async def test_search_returns_empty_on_latency_ceiling() -> None:
    """search() shares select()'s latency ceiling -- a slow (not raising)
    embedder must not hang a mid-turn tool_search call indefinitely.
    """
    tools = _catalog(5)
    vectors = {_tool_text(t): [1.0, 0.0] for t in tools}
    embedder = _FakeEmbeddings(vectors, sleep_s=0.2)
    index = ToolRelevanceIndex(embedder=embedder, top_k=2, latency_ceiling_s=0.01)

    hits = await index.search(tools, "query")

    assert hits == []


async def test_search_on_empty_catalog_returns_empty() -> None:
    embedder = _FakeEmbeddings({})
    index = ToolRelevanceIndex(embedder=embedder, top_k=2)

    hits = await index.search([], "query")

    assert hits == []


# ---------------------------------------------------------------------------
# create_tool_search_tool -- the escape hatch
# ---------------------------------------------------------------------------


async def test_tool_search_surfaces_tool_absent_from_initial_top_k() -> None:
    """The named regression shape: a tool that scores outside a synthetic
    top-K must still be discoverable via one tool_search call against the
    FULL catalog.
    """
    tools = _catalog(10)
    session_send = _make_tool("session_send", "send a message to a dev agent session")
    full_catalog = tools + [session_send]

    vectors = {_tool_text(t): [0.0, 1.0] for t in tools}
    vectors[_tool_text(session_send)] = [1.0, 0.0]
    vectors["initial turn query"] = [0.0, 1.0]  # matches the noise tools, not session_send
    vectors["send a message to the dev session"] = [1.0, 0.0]  # matches session_send
    embedder = _FakeEmbeddings(vectors)
    index = ToolRelevanceIndex(embedder=embedder, top_k=3)

    # Prove the miss: session_send is NOT in the initial top-K.
    initial = await index.select(full_catalog, "initial turn query")
    assert not initial.fallback_used
    assert "session_send" not in [t.name for t in initial.selected]

    hits: list[list[BaseTool]] = []
    search_tool = create_tool_search_tool(
        index=index,
        full_catalog_provider=lambda: full_catalog,
        on_hit=hits.append,
        limit=1,
    )

    result_text = await search_tool.ainvoke(
        {"query": "send a message to the dev session"}
    )

    assert "session_send" in result_text
    assert len(hits) == 1
    assert [t.name for t in hits[0]] == ["session_send"]


async def test_tool_search_reports_no_match_without_calling_on_hit() -> None:
    tools = _catalog(3)
    embedder = _FakeEmbeddings({}, raise_on_documents=True)  # forces search() -> []
    index = ToolRelevanceIndex(embedder=embedder, top_k=2)

    hits: list[list[BaseTool]] = []
    search_tool = create_tool_search_tool(
        index=index,
        full_catalog_provider=lambda: tools,
        on_hit=hits.append,
    )

    result_text = await search_tool.ainvoke({"query": "anything"})

    assert result_text == "No matching tools found."
    assert hits == []


async def test_tool_search_catalog_provider_is_reinvoked_per_call() -> None:
    """The provider closure must see a catalog mutated between calls -- the
    live-MCP-discovery / admin-edit volatility case, not a snapshot taken at
    tool-construction time.
    """
    tools = _catalog(3)
    catalog_state = {"tools": tools}
    new_tool = _make_tool("freshly_discovered", "a tool that appeared mid-conversation")

    vectors = {_tool_text(t): [0.0, 1.0] for t in tools}
    vectors[_tool_text(new_tool)] = [1.0, 0.0]
    vectors["find the fresh one"] = [1.0, 0.0]
    embedder = _FakeEmbeddings(vectors)
    index = ToolRelevanceIndex(embedder=embedder, top_k=2)

    hits: list[list[BaseTool]] = []
    search_tool = create_tool_search_tool(
        index=index,
        full_catalog_provider=lambda: catalog_state["tools"],
        on_hit=hits.append,
        limit=1,
    )

    # First call: the new tool doesn't exist in the catalog yet.
    first = await search_tool.ainvoke({"query": "find the fresh one"})
    assert "freshly_discovered" not in first

    # Mutate the "live" catalog the provider closes over.
    catalog_state["tools"] = tools + [new_tool]

    second = await search_tool.ainvoke({"query": "find the fresh one"})
    assert "freshly_discovered" in second


async def test_tool_search_hits_are_deduplicated_by_caller_not_this_module() -> None:
    """This module reports every hit on every call -- dedup across repeated
    calls in one turn is the caller's rebind-composition responsibility
    (metallm's tool_loop.py), documented here so the boundary is explicit.
    """
    tools = _catalog(5)
    target = _make_tool("target_tool", "the one relevant tool")
    catalog = tools + [target]
    vectors = {_tool_text(t): [0.0, 1.0] for t in tools}
    vectors[_tool_text(target)] = [1.0, 0.0]
    vectors["query"] = [1.0, 0.0]
    embedder = _FakeEmbeddings(vectors)
    index = ToolRelevanceIndex(embedder=embedder, top_k=2)

    hits: list[list[BaseTool]] = []
    search_tool = create_tool_search_tool(
        index=index, full_catalog_provider=lambda: catalog, on_hit=hits.append, limit=1
    )

    await search_tool.ainvoke({"query": "query"})
    await search_tool.ainvoke({"query": "query"})

    assert len(hits) == 2
    assert [t.name for t in hits[0]] == [t.name for t in hits[1]] == ["target_tool"]
