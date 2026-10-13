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
from collections.abc import Sequence
from typing import Annotated, Any

from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool
from pydantic import BaseModel, Field

from threetears.agent.tools.relevance import (
    ToolRelevanceIndex,
    ToolSearchResult,
    also_in_family,
    create_tool_search_tool,
    match_words,
    tool_arguments,
    tool_families,
    tool_hit_line,
)
from threetears.agent.tools.utils import tool_result_status


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
        documents_sleep_s: float | None = None,
    ) -> None:
        self.vectors = vectors
        self.documents_sleep_s = sleep_s if documents_sleep_s is None else documents_sleep_s
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
        if self.documents_sleep_s:
            await asyncio.sleep(self.documents_sleep_s)
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
    embedder = _FakeEmbeddings(
        {},
    )
    index = ToolRelevanceIndex(embedder=embedder, top_k=1, cache_size=2)

    # Three distinct tool sets (each > top_k so select() actually embeds),
    # bounded cache_size=2 -> the first set's entry gets evicted.
    sets = [[_make_tool(f"set{i}_tool_a", "a"), _make_tool(f"set{i}_tool_b", "b")] for i in range(3)]
    for tool_set in sets:
        await index.select(tool_set, "q")
    assert len(embedder.aembed_documents_calls) == 3

    # the two most recent sets are still cached: selecting them again embeds nothing.
    await index.select(sets[2], "q")
    await index.select(sets[1], "q")
    assert len(embedder.aembed_documents_calls) == 3

    # the oldest set was evicted to hold the bound, so it is embedded afresh.
    await index.select(sets[0], "q")
    assert len(embedder.aembed_documents_calls) == 4
    assert embedder.aembed_documents_calls[-1] == [_tool_text(t) for t in sets[0]]


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


async def test_a_catalog_embedding_cut_off_by_the_ceiling_still_lands_for_the_next_turn() -> None:
    """Found live: embedding a 51-tool catalog took longer than the 1 s ceiling, and the timeout
    cancelled the embedding along with the ranking -- the cache never filled, so every turn that
    started cold fell back to the full catalog and threw the work away again."""
    tools = _catalog(10)
    vectors = {_tool_text(t): [1.0, 0.0] for t in tools}
    embedder = _FakeEmbeddings(vectors, documents_sleep_s=0.2)
    index = ToolRelevanceIndex(embedder=embedder, top_k=3, latency_ceiling_s=0.05)

    first = await index.select(tools, "anything")
    assert first.fallback_reason == "latency_ceiling"

    await asyncio.sleep(0.3)
    second = await index.select(tools, "anything")

    assert not second.fallback_used
    assert len(second.selected) == 3
    assert len(embedder.aembed_documents_calls) == 1, "the catalog was embedded again"


async def test_turns_arriving_together_embed_one_catalog_once() -> None:
    tools = _catalog(10)
    vectors = {_tool_text(t): [1.0, 0.0] for t in tools}
    embedder = _FakeEmbeddings(vectors, documents_sleep_s=0.05)
    index = ToolRelevanceIndex(embedder=embedder, top_k=3, latency_ceiling_s=1.0)

    results = await asyncio.gather(*(index.select(tools, f"query {n}") for n in range(4)))

    assert all(not r.fallback_used for r in results)
    assert len(embedder.aembed_documents_calls) == 1


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
    """search() falls under select()'s latency ceiling unless given its own --
    a slow (not raising) embedder must not hang a mid-turn tool_search call
    indefinitely.
    """
    tools = _catalog(5)
    vectors = {_tool_text(t): [1.0, 0.0] for t in tools}
    embedder = _FakeEmbeddings(vectors, sleep_s=0.2)
    index = ToolRelevanceIndex(embedder=embedder, top_k=2, latency_ceiling_s=0.01)

    hits = await index.search(tools, "query")

    assert hits == []


async def test_search_waits_to_its_own_ceiling_while_select_keeps_the_tight_one() -> None:
    """A cold catalog that select gives up on is still one tool_search can
    finish: the two ceilings are separate because the two failures cost
    differently (select falls back to the full catalog; search reads as
    "no such tool").
    """
    tools = _catalog(5)
    vectors = {_tool_text(t): [1.0, 0.0] for t in tools}
    embedder = _FakeEmbeddings(vectors, sleep_s=0.05)
    index = ToolRelevanceIndex(embedder=embedder, top_k=2, latency_ceiling_s=0.01, search_latency_ceiling_s=1.0)

    selection = await index.select(tools, "query")
    hits = await index.search(tools, "query", limit=2)
    scored = await index.search_scored(tools, "query", limit=2)

    assert selection.fallback_reason == "latency_ceiling"
    assert len(hits) == 2
    assert len(scored) == 2


async def test_search_outcome_says_why_there_are_no_hits() -> None:
    tools = _catalog(3)
    vectors = {_tool_text(t): [1.0, 0.0] for t in tools}

    slow = ToolRelevanceIndex(embedder=_FakeEmbeddings(vectors, sleep_s=0.2), top_k=2, latency_ceiling_s=0.01)
    broken = ToolRelevanceIndex(embedder=_FakeEmbeddings(vectors, raise_on_documents=True), top_k=2)
    fine = ToolRelevanceIndex(embedder=_FakeEmbeddings(vectors), top_k=2)

    assert await slow.search_outcome(tools, "query") == ToolSearchResult(hits=[], fallback_reason="latency_ceiling")
    assert await broken.search_outcome(tools, "query") == ToolSearchResult(hits=[], fallback_reason="embedder_error")
    assert await fine.search_outcome([], "query") == ToolSearchResult(hits=[])
    finished = await fine.search_outcome(tools, "query", limit=2)
    assert finished.fallback_reason is None
    assert len(finished.hits) == 2


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

    result_text = await search_tool.ainvoke({"query": "send a message to the dev session"})

    assert "session_send" in result_text
    assert len(hits) == 1
    assert [t.name for t in hits[0]] == ["session_send"]


async def test_tool_search_hit_message_matches_next_round_description() -> None:
    """Live prod bug (metallm conv 019f6cf5-073a-7b50-bd44-721efb0c7b90): the
    tool's own DESCRIPTION correctly says hits become callable "starting your
    NEXT reply", but the hit return text used to say "now available to call"
    -- a direct contradiction the model reads immediately after invoking the
    tool, mid-round. That drove the model to attempt the tool right away, in
    the SAME round, and bounce off "No such tool available" (no caller can
    compose a hit into the bound set before the next round boundary -- see
    ``on_hit``'s docstring). The return text must never claim immediacy the
    caller cannot deliver.
    """
    tools = _catalog(3)
    session_send = _make_tool("session_send", "send a message to a dev agent session")
    full_catalog = tools + [session_send]
    vectors = {_tool_text(t): [0.0, 1.0] for t in tools}
    vectors[_tool_text(session_send)] = [1.0, 0.0]
    vectors["send a message"] = [1.0, 0.0]
    embedder = _FakeEmbeddings(vectors)
    index = ToolRelevanceIndex(embedder=embedder, top_k=3)

    search_tool = create_tool_search_tool(
        index=index,
        full_catalog_provider=lambda: full_catalog,
        on_hit=lambda _matches: None,
        limit=1,
    )

    result_text = await search_tool.ainvoke({"query": "send a message"})

    assert "now available to call" not in result_text
    assert "after this search returns" in result_text
    assert "session_send" in result_text


async def test_tool_search_reports_no_match_without_calling_on_hit() -> None:
    embedder = _FakeEmbeddings({})
    index = ToolRelevanceIndex(embedder=embedder, top_k=2)

    hits: list[list[BaseTool]] = []
    search_tool = create_tool_search_tool(
        index=index,
        full_catalog_provider=list,  # an empty catalog: nothing to match
        on_hit=hits.append,
    )

    result_text = await search_tool.ainvoke({"query": "anything"})

    assert result_text == "No matching tools found."
    assert hits == []


async def test_tool_search_says_it_did_not_finish_rather_than_nothing_matched() -> None:
    """A search cut off by the ceiling found nothing because it never ran.
    Told "No matching tools found." the model tells the person the tool does
    not exist (live: metallm's first turn after a deploy, cold catalog).
    """
    tools = _catalog(3)
    vectors = {_tool_text(t): [1.0, 0.0] for t in tools}
    index = ToolRelevanceIndex(embedder=_FakeEmbeddings(vectors, sleep_s=0.2), top_k=2, latency_ceiling_s=0.01)

    hits: list[list[BaseTool]] = []
    search_tool = create_tool_search_tool(index=index, full_catalog_provider=lambda: tools, on_hit=hits.append)

    result_text = await search_tool.ainvoke({"query": "anything"})

    assert "did not finish in time" in result_text
    assert "Run the same search once more" in result_text
    assert "No matching tools found" not in result_text
    assert hits == []


async def test_tool_search_says_the_index_failed_rather_than_nothing_matched() -> None:
    tools = _catalog(3)
    index = ToolRelevanceIndex(embedder=_FakeEmbeddings({}, raise_on_documents=True), top_k=2)

    hits: list[list[BaseTool]] = []
    search_tool = create_tool_search_tool(index=index, full_catalog_provider=lambda: tools, on_hit=hits.append)

    result_text = await search_tool.ainvoke({"query": "anything"})

    assert result_text.startswith("[TOOL ERROR] Tool search failed")
    assert "Nothing was found and nothing was ruled out" in result_text
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


async def test_search_scored_returns_the_similarity_beside_each_tool() -> None:
    """The score is the point: ranking alone cannot tell a message that wants
    a tool from one that merely sits nearest to it. Every query has a closest
    tool, so a caller acting on the top hit without a model in the loop needs
    to know how close it actually was.
    """
    tools = _catalog(5)
    vectors = {_tool_text(t): [0.0, 1.0] for t in tools}
    vectors[_tool_text(tools[3])] = [1.0, 0.0]
    vectors["query"] = [1.0, 0.0]
    index = ToolRelevanceIndex(embedder=_FakeEmbeddings(vectors), top_k=2)

    hits = await index.search_scored(tools, "query", limit=2)

    assert [t.name for t, _ in hits] == ["tool_3", "tool_0"]
    best, best_score = hits[0]
    _, runner_up_score = hits[1]
    assert best.name == "tool_3"
    assert best_score == 1.0
    assert runner_up_score < best_score


async def test_search_scored_separates_a_near_match_from_a_far_one() -> None:
    """The same catalog against two queries: one a tool answers, one it does
    not. Both have a top hit; only one of them should be acted on, and the
    score is the only thing that says so.
    """
    tools = _catalog(3)
    vectors = {_tool_text(t): [0.0, 1.0] for t in tools}
    vectors[_tool_text(tools[0])] = [1.0, 0.0]
    vectors["wants the tool"] = [1.0, 0.0]
    # Deliberately aligned with nothing in the catalog: a message that sits
    # between the tools rather than on one. Using a vector that matches the
    # other tools exactly would score 1.0 and prove nothing.
    vectors["wants nothing"] = [1.0, 1.0]
    index = ToolRelevanceIndex(embedder=_FakeEmbeddings(vectors), top_k=2)

    wanted = await index.search_scored(tools, "wants the tool", limit=1)
    unwanted = await index.search_scored(tools, "wants nothing", limit=1)

    # Which tool wins the far query is not the contract -- with nothing
    # aligned, the hits tie and stable order decides. The score is the
    # contract: it is what separates "this message wants a tool" from "this
    # message has a nearest tool", and every message has the latter.
    assert wanted[0][0].name == "tool_0"
    assert wanted[0][1] == 1.0
    assert unwanted[0][1] < wanted[0][1], "a far match must not score like a near one"


async def test_search_scored_returns_empty_on_embedder_failure() -> None:
    """Same soft-fail contract as search(): no hits rather than a guess."""
    index = ToolRelevanceIndex(embedder=_FakeEmbeddings({}, raise_on_documents=True), top_k=2)

    assert await index.search_scored(_catalog(5), "query") == []


async def test_search_scored_returns_empty_on_latency_ceiling() -> None:
    tools = _catalog(5)
    vectors = {_tool_text(t): [1.0, 0.0] for t in tools}
    embedder = _FakeEmbeddings(vectors, sleep_s=0.2)
    index = ToolRelevanceIndex(embedder=embedder, top_k=2, latency_ceiling_s=0.01)

    assert await index.search_scored(tools, "query") == []


async def test_search_scored_on_empty_catalog_returns_empty() -> None:
    index = ToolRelevanceIndex(embedder=_FakeEmbeddings({}), top_k=2)

    assert await index.search_scored([], "query") == []


# ---------------------------------------------------------------------------
# The word match: tool_search still finds a tool while the ranking is down
# ---------------------------------------------------------------------------


def _house_catalog() -> list[BaseTool]:
    return [
        _make_tool("web_search", "Search the web for current information."),
        _make_tool("ha_call_service", "Turn a device in the house on or off, or set it."),
        _make_tool("calculator", "Work out arithmetic."),
    ]


def test_match_words_ranks_by_how_many_words_match() -> None:
    names = [t.name for t in match_words(_house_catalog(), "turn on the house fan")]
    assert names == ["ha_call_service"]


def test_match_words_reads_a_name_s_separators_as_spaces() -> None:
    assert [t.name for t in match_words(_house_catalog(), "search")] == ["web_search"]


def test_match_words_with_only_common_words_matches_nothing() -> None:
    assert match_words(_house_catalog(), "what is the") == []


async def test_a_failed_ranking_still_finds_tools_by_their_words() -> None:
    """A consumer that binds only the pick and tool_search reaches nothing else while the embedder is down."""
    tools = _house_catalog()
    broken = ToolRelevanceIndex(embedder=_FakeEmbeddings({}, raise_on_documents=True), top_k=2)

    result = await broken.search_outcome(tools, "search the web")

    assert [t.name for t in result.hits] == ["web_search"]
    assert result.fallback_reason == "embedder_error"


async def test_tool_search_hands_over_word_matches_when_the_ranking_fails() -> None:
    tools = _house_catalog()
    index = ToolRelevanceIndex(embedder=_FakeEmbeddings({}, raise_on_documents=True), top_k=2)
    hits: list[list[BaseTool]] = []
    search_tool = create_tool_search_tool(index=index, full_catalog_provider=lambda: tools, on_hit=hits.append)

    result_text = await search_tool.ainvoke({"query": "search the web"})

    assert [[t.name for t in h] for h in hits] == [["web_search"]]
    assert "web_search" in result_text and "Tool search failed" not in result_text
    first, _, body = result_text.partition("\n")
    assert first == (
        "Found these tools by matching words only: the ranked search did not run, so a closer tool may exist. "
        "Call them after this search returns, not alongside it:"
    ), "word matches are said to be word matches, in the line the model reads first"
    assert body.startswith("- web_search(")


# ---------------------------------------------------------------------------
# What a hit says: the arguments, and the rest of its family
# ---------------------------------------------------------------------------


class _SendInput(BaseModel):
    to: str = Field(description="who")
    body: str = Field(description="what")
    urgent: bool = Field(default=False, description="whether")
    tags: list[str] | None = Field(default=None, description="labels")
    when: int | str = Field(default=0, description="a time or a count")
    kept_out: Annotated[dict[str, Any] | None, InjectedToolArg] = None


def _send_tool(name: str = "session_send") -> BaseTool:
    async def _send(**kwargs: Any) -> str:
        return "sent"

    return StructuredTool.from_function(
        coroutine=_send, name=name, description="send a message to a dev agent session", args_schema=_SendInput
    )


def test_a_tools_arguments_are_named_with_their_types_and_which_may_be_left_out() -> None:
    assert tool_arguments(_send_tool()) == "(to: string, body: string, urgent?: boolean, tags?: array, when?)"


def test_an_argument_the_host_injects_is_not_shown_to_the_model() -> None:
    assert "kept_out" not in tool_arguments(_send_tool())


def test_a_tool_that_takes_nothing_has_an_empty_bracket() -> None:
    assert tool_arguments(_make_tool("ping", "check it is up")) == "()"


def test_a_tool_described_by_a_json_schema_is_read_the_same_way() -> None:
    async def _call(**kwargs: Any) -> str:
        return "ok"

    remote = StructuredTool.from_function(
        coroutine=_call,
        name="lights",
        description="turn lights on or off",
        args_schema={
            "type": "object",
            "properties": {"room": {"type": "string"}, "on": {"type": "boolean"}, "level": {"type": "integer"}},
            "required": ["room", "on"],
        },
    )
    assert tool_arguments(remote) == "(room: string, on: boolean, level?: integer)"


def test_a_hit_line_is_the_name_the_arguments_and_what_it_does() -> None:
    assert tool_hit_line(_send_tool()) == (
        "- session_send(to: string, body: string, urgent?: boolean, tags?: array, when?): "
        "send a message to a dev agent session"
    )


def _family_catalog() -> list[BaseTool]:
    calendar = [_make_tool(f"calendar_{verb}", f"{verb} a calendar event") for verb in ("list", "create", "move")]
    mail = [_make_tool(f"email_{verb}", f"{verb} mail") for verb in ("search", "read")]
    return [*calendar, *mail, _make_tool("ping", "check it is up")]


def _family_of(tool: BaseTool) -> str | None:
    head = tool.name.split("_")[0]
    return {"calendar": "calendar", "email": "mail"}.get(head)


def _family_index(catalog: list[BaseTool], near: dict[str, float], query: str) -> ToolRelevanceIndex:
    """An index whose ranking for ``query`` is the order of ``near``: tool name -> closeness, 1.0 nearest."""
    vectors = {_tool_text(t): [near.get(t.name, 0.0), 1.0 - near.get(t.name, 0.0)] for t in catalog}
    vectors[query] = [1.0, 0.0]
    return ToolRelevanceIndex(embedder=_FakeEmbeddings(vectors), top_k=2)


def test_the_catalog_by_family_keeps_catalog_order_and_leaves_out_a_tool_with_no_family() -> None:
    families = tool_families(_family_catalog(), _family_of)
    assert {name: [t.name for t in tools] for name, tools in families.items()} == {
        "calendar": ["calendar_list", "calendar_create", "calendar_move"],
        "mail": ["email_search", "email_read"],
    }


def test_a_hits_family_is_named_with_its_other_tools_once_per_family() -> None:
    catalog = _family_catalog()
    by_name = {t.name: t for t in catalog}
    hits = [by_name["calendar_list"], by_name["ping"], by_name["calendar_move"], by_name["email_read"]]
    assert also_in_family(hits, catalog, _family_of) == [
        "Also in the calendar family: calendar_create.",
        "Also in the mail family: email_search.",
    ]


def test_a_family_with_nothing_else_in_it_adds_no_line() -> None:
    catalog = _family_catalog()
    mail = [t for t in catalog if t.name.startswith("email_")]
    assert also_in_family(mail, catalog, _family_of) == []


def test_a_big_family_names_its_first_tools_and_counts_the_rest() -> None:
    catalog = [_make_tool(f"house_{n:02d}", "a device") for n in range(12)]
    lines = also_in_family(catalog[:1], catalog, lambda _tool: "house")
    assert lines == [
        "Also in the house family: house_01, house_02, house_03, house_04, house_05, house_06, house_07, "
        "house_08, and 3 more."
    ]


async def test_tool_search_shows_each_hits_arguments_and_the_rest_of_its_family() -> None:
    catalog = [*_family_catalog(), _send_tool()]
    index = _family_index(catalog, {"calendar_move": 1.0, "session_send": 0.9}, "move my meeting")
    hits: list[list[BaseTool]] = []
    search_tool = create_tool_search_tool(
        index=index, full_catalog_provider=lambda: catalog, on_hit=hits.append, limit=2, group_of=_family_of
    )

    result_text = await search_tool.ainvoke({"query": "move my meeting"})

    assert result_text.splitlines() == [
        "Found these tools. Call them after this search returns, not alongside it:",
        "- calendar_move(): move a calendar event",
        "- session_send(to: string, body: string, urgent?: boolean, tags?: array, when?): "
        "send a message to a dev agent session",
        "Also in the calendar family: calendar_list, calendar_create.",
    ]
    assert [[t.name for t in h] for h in hits] == [["calendar_move", "session_send"]], "siblings are named, not bound"


async def test_tool_search_returns_as_many_as_the_model_asks_for_up_to_the_most_allowed() -> None:
    catalog = _family_catalog()
    near = {t.name: 1.0 - n / 10 for n, t in enumerate(catalog)}
    index = _family_index(catalog, near, "anything")
    hits: list[list[BaseTool]] = []
    search_tool = create_tool_search_tool(
        index=index, full_catalog_provider=lambda: catalog, on_hit=hits.append, limit=2, max_limit=4
    )

    await search_tool.ainvoke({"query": "anything"})
    await search_tool.ainvoke({"query": "anything", "limit": 3})
    await search_tool.ainvoke({"query": "anything", "limit": 50})
    await search_tool.ainvoke({"query": "anything", "limit": 0})

    assert [len(h) for h in hits] == [2, 3, 4, 1]
    assert [t.name for t in hits[1]] == ["calendar_list", "calendar_create", "calendar_move"]


async def test_tool_search_in_one_family_ranks_only_that_family() -> None:
    catalog = _family_catalog()
    index = _family_index(catalog, {"ping": 1.0, "email_read": 0.6, "calendar_list": 0.5}, "read it")
    hits: list[list[BaseTool]] = []
    search_tool = create_tool_search_tool(
        index=index, full_catalog_provider=lambda: catalog, on_hit=hits.append, limit=1, group_of=_family_of
    )

    everywhere = await search_tool.ainvoke({"query": "read it"})
    in_mail = await search_tool.ainvoke({"query": "read it", "family": "MAIL"})

    assert "- ping()" in everywhere
    assert in_mail.splitlines() == [
        "Found these tools. Call them after this search returns, not alongside it:",
        "- email_read(): read mail",
        "Also in the mail family: email_search.",
    ]
    assert [[t.name for t in h] for h in hits] == [["ping"], ["email_read"]]


async def test_tool_search_with_a_family_and_no_query_lists_the_family() -> None:
    catalog = _family_catalog()
    index = ToolRelevanceIndex(embedder=_FakeEmbeddings({}), top_k=2)
    hits: list[list[BaseTool]] = []
    search_tool = create_tool_search_tool(
        index=index, full_catalog_provider=lambda: catalog, on_hit=hits.append, limit=2, group_of=_family_of
    )

    part = await search_tool.ainvoke({"family": "calendar"})
    whole = await search_tool.ainvoke({"family": "calendar", "limit": 5})

    assert part.splitlines() == [
        "The calendar family has 3 tools, the first 2. Call them after this search returns, not alongside it:",
        "The rest: calendar_move.",
        "- calendar_list(): list a calendar event",
        "- calendar_create(): create a calendar event",
    ]
    assert whole.splitlines()[0] == (
        "The calendar family has 3 tools. Call them after this search returns, not alongside it:"
    )
    assert "The rest" not in whole and whole.count("\n- ") == 3
    assert [[t.name for t in h] for h in hits] == [
        ["calendar_list", "calendar_create"],
        ["calendar_list", "calendar_create", "calendar_move"],
    ]


async def test_tool_search_for_a_family_that_is_not_there_names_the_families_that_are() -> None:
    catalog = _family_catalog()
    index = ToolRelevanceIndex(embedder=_FakeEmbeddings({}), top_k=2)
    hits: list[list[BaseTool]] = []
    search_tool = create_tool_search_tool(
        index=index, full_catalog_provider=lambda: catalog, on_hit=hits.append, group_of=_family_of
    )

    result_text = await search_tool.ainvoke({"query": "anything", "family": "weather"})

    assert (
        result_text == "[NO CHANGE] tool_search: there is no tool family named 'weather'. The families: calendar, mail."
    )
    assert tool_result_status(result_text) == "no_change" and hits == []


async def test_tool_search_with_neither_a_query_nor_a_family_is_an_error() -> None:
    index = ToolRelevanceIndex(embedder=_FakeEmbeddings({}), top_k=2)
    search_tool = create_tool_search_tool(
        index=index, full_catalog_provider=_family_catalog, on_hit=lambda _m: None, group_of=_family_of
    )
    assert (await search_tool.ainvoke({})).startswith("[TOOL ERROR] tool_search: say what you need")


def test_tool_search_offers_a_family_only_when_the_catalog_has_families() -> None:
    index = ToolRelevanceIndex(embedder=_FakeEmbeddings({}), top_k=2)
    plain = create_tool_search_tool(index=index, full_catalog_provider=list, on_hit=lambda _m: None)
    grouped = create_tool_search_tool(
        index=index, full_catalog_provider=list, on_hit=lambda _m: None, group_of=_family_of
    )

    assert set(plain.args) == {"query", "limit"} and set(grouped.args) == {"query", "limit", "family"}
    assert "family" not in plain.description
    assert grouped.description == (
        plain.description + " Pass family to search one family of tools, or family alone to list it."
    )
    assert plain.tool_call_schema.model_json_schema()["required"] == ["query"]


# ---------------------------------------------------------------------------
# Whether anything fits is the caller's to judge, by the scores
# ---------------------------------------------------------------------------


async def test_a_search_outcome_carries_the_whole_scored_ranking_past_the_limit() -> None:
    catalog = _family_catalog()
    index = _family_index(catalog, {"calendar_move": 1.0, "calendar_list": 0.8, "email_read": 0.3}, "move it")

    outcome = await index.search_outcome(catalog, "move it", limit=2)

    assert [t.name for t in outcome.hits] == ["calendar_move", "calendar_list"]
    assert [t.name for t, _ in outcome.ranked[:3]] == ["calendar_move", "calendar_list", "email_read"]
    assert len(outcome.ranked) == len(catalog), "the ranking is whole, so a margin can reach past the hits"
    scores = [score for _, score in outcome.ranked]
    assert scores == sorted(scores, reverse=True) and scores[0] > scores[2] > scores[-1]


async def test_a_ranking_that_did_not_run_has_no_scores() -> None:
    broken = ToolRelevanceIndex(embedder=_FakeEmbeddings({}, raise_on_documents=True), top_k=2)
    outcome = await broken.search_outcome(_house_catalog(), "search the web")
    assert [t.name for t in outcome.hits] == ["web_search"] and outcome.ranked == []


async def test_tool_search_offers_nothing_when_the_caller_says_nothing_fits() -> None:
    catalog = _family_catalog()
    index = _family_index(catalog, {"calendar_move": 0.52, "calendar_list": 0.5, "email_read": 0.49}, "morning, you")
    hits: list[list[BaseTool]] = []
    asked: list[tuple[str, list[tuple[str, float]]]] = []

    def _flat(query: str, ranked: Sequence[tuple[BaseTool, float]]) -> bool:
        asked.append((query, [(tool.name, score) for tool, score in ranked]))
        return ranked[0][1] - ranked[2][1] < 0.1  # a margin, the caller's own; never one fixed score

    search_tool = create_tool_search_tool(
        index=index, full_catalog_provider=lambda: catalog, on_hit=hits.append, limit=2, nothing_fits=_flat
    )

    result_text = await search_tool.ainvoke({"query": "morning, you"})

    assert result_text == "No tool fits that. The nearest, none of them handed to you: calendar_move, calendar_list."
    assert hits == [], "nothing is bound when nothing fits"
    assert asked[0][0] == "morning, you" and len(asked[0][1]) == len(catalog), "the judge sees the whole ranking"


async def test_tool_search_offers_its_hits_when_the_caller_says_one_fits() -> None:
    catalog = _family_catalog()
    index = _family_index(catalog, {"calendar_move": 1.0}, "move my meeting")
    hits: list[list[BaseTool]] = []
    search_tool = create_tool_search_tool(
        index=index,
        full_catalog_provider=lambda: catalog,
        on_hit=hits.append,
        limit=1,
        nothing_fits=lambda _query, ranked: ranked[0][1] - ranked[4][1] < 0.1,
    )

    result_text = await search_tool.ainvoke({"query": "move my meeting"})

    assert result_text.startswith("Found these tools.") and [[t.name for t in h] for h in hits] == [["calendar_move"]]


async def test_word_matches_are_not_judged_for_they_have_no_score() -> None:
    tools = _house_catalog()
    index = ToolRelevanceIndex(embedder=_FakeEmbeddings({}, raise_on_documents=True), top_k=2)
    hits: list[list[BaseTool]] = []

    def _never_asked(query: str, ranked: Sequence[tuple[BaseTool, float]]) -> bool:
        raise AssertionError("a word match has no score to judge")

    search_tool = create_tool_search_tool(
        index=index, full_catalog_provider=lambda: tools, on_hit=hits.append, nothing_fits=_never_asked
    )

    assert "by matching words only" in await search_tool.ainvoke({"query": "search the web"})
    assert [[t.name for t in h] for h in hits] == [["web_search"]]


def _through_the_door(found: Sequence[BaseTool], catalog: Sequence[BaseTool]) -> tuple[list[str], list[BaseTool]]:
    """A caller's ``offer``: mail tools are reached through ``ping``, which is handed over once in their place."""
    door = next(t for t in catalog if t.name == "ping")
    lines: list[str] = []
    handed: list[BaseTool] = []
    for tool in found:
        if tool.name.startswith("email_"):
            lines.append(f"- {tool.name}: ask ping for it")
            offered = door
        else:
            lines.append(f"- {tool.name}: yours")
            offered = tool
        if offered not in handed:
            handed.append(offered)
    return lines, handed


async def test_tool_search_shows_and_hands_over_what_the_callers_offer_says() -> None:
    catalog = _family_catalog()
    index = _family_index(catalog, {"email_read": 1.0, "email_search": 0.9, "calendar_move": 0.8}, "read my mail")
    hits: list[list[BaseTool]] = []
    search_tool = create_tool_search_tool(
        index=index,
        full_catalog_provider=lambda: catalog,
        on_hit=hits.append,
        limit=3,
        group_of=_family_of,
        offer=_through_the_door,
    )

    result_text = await search_tool.ainvoke({"query": "read my mail"})

    assert result_text.splitlines() == [
        "Found these tools. Call them after this search returns, not alongside it:",
        "- email_read: ask ping for it",
        "- email_search: ask ping for it",
        "- calendar_move: yours",
        "Also in the calendar family: calendar_list, calendar_create.",
    ]
    assert [[t.name for t in h] for h in hits] == [["ping", "calendar_move"]]


async def test_a_listed_family_is_shown_and_handed_over_by_the_callers_offer_too() -> None:
    catalog = _family_catalog()
    index = ToolRelevanceIndex(embedder=_FakeEmbeddings({}), top_k=2)
    hits: list[list[BaseTool]] = []
    search_tool = create_tool_search_tool(
        index=index,
        full_catalog_provider=lambda: catalog,
        on_hit=hits.append,
        group_of=_family_of,
        offer=_through_the_door,
    )

    listed = await search_tool.ainvoke({"family": "mail"})

    assert listed.splitlines() == [
        "The mail family has 2 tools. Call them after this search returns, not alongside it:",
        "- email_search: ask ping for it",
        "- email_read: ask ping for it",
    ]
    assert [[t.name for t in h] for h in hits] == [["ping"]]


async def test_an_offer_that_hands_nothing_over_binds_nothing() -> None:
    catalog = _family_catalog()
    index = _family_index(catalog, {"email_read": 1.0}, "read my mail")
    hits: list[list[BaseTool]] = []
    search_tool = create_tool_search_tool(
        index=index,
        full_catalog_provider=lambda: catalog,
        on_hit=hits.append,
        limit=1,
        offer=lambda found, _catalog: ([f"- {t.name}: not yours here" for t in found], []),
    )

    result_text = await search_tool.ainvoke({"query": "read my mail"})

    assert result_text.splitlines()[1:] == ["- email_read: not yours here"]
    assert hits == []
