"""Tool-relevance selection: given N tool definitions + a query, return a
relevant subset, plus a way to search the rest.

A large bound tool surface degrades a model's tool-selection accuracy and, in
the extreme, its willingness to dispatch a structured call at all instead of
narrating one in prose (see the platform-neutral write-up this module
implements: metallm's ``.prawduct/artifacts/3tears-change-dynamic-tool-selection.md``
sign-off). :class:`ToolRelevanceIndex` embeds each tool's name + description
once per distinct tool set (cached by content hash, not "per turn" — a live-
discovered or admin-editable tool set can change mid-conversation), embeds the
current turn's query, and ranks by cosine similarity.

Two contracts are load-bearing, not incidental:

- **Fallback never shrinks the set.** An embedder failure or a selection that
  exceeds the latency ceiling returns the FULL, unfiltered input list, never a
  smaller arbitrary one. A relevance-index outage must degrade to "today's
  behaviour" (everything bound), not to a silently reduced surface the caller
  can't distinguish from a real top-K result.
- **``tool_search`` is the escape hatch, not the primary mechanism.** Even a
  perfect top-K is a bet; :func:`create_tool_search_tool` lets the model query
  the FULL catalog mid-turn for anything the top-K missed. Hits are reported
  back to the caller via ``on_hit`` so the caller can add them to its next
  round's bound set — this module has no notion of "rounds" or "turns", that
  composition is the caller's (see metallm's ``tool_loop.py``).

The embedder is constructor-injected as a plain ``langchain_core.embeddings.
Embeddings`` instance — this module does NOT import from ``agent/memory``
(wrong direction of coupling for an ``agent/tools`` capability); callers pass
whatever embedding client they already use elsewhere.

This module has no notion of an "always-bound core set" (e.g. metallm's
``recall_context`` / ``invoke_tool_llm`` workflow primitives, which must stay
callable regardless of top-K). That composition is entirely the caller's
responsibility: a caller adds its always-bound tools to :meth:`ToolRelevanceIndex
.select`'s result AFTER selection runs, so ``select`` never has to be asked to
avoid excluding them.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from threetears.observe import get_logger

__all__ = [
    "ToolRelevanceIndex",
    "ToolSearchResult",
    "ToolSelectionResult",
    "create_tool_search_tool",
]

_log = get_logger(__name__)

#: default top-K — the middle of the ~10-20 "safe zone" independent 2025-2026
#: studies converged on before tool-selection accuracy degrades. Tunable per
#: instance; needs empirical tuning against a real catalog + provider before
#: being treated as final (see the sign-off artifact's open questions).
DEFAULT_TOP_K = 15

#: default hard latency ceiling. A selection call slower than this is
#: functionally an unavailable one for this purpose — same fallback path as
#: an embedder error. Needs a real measurement against a real catalog +
#: provider before being treated as final.
DEFAULT_LATENCY_CEILING_S = 0.4

#: default bound on the number of distinct tool-set content-hashes cached at
#: once. Bounded (not unbounded growth) because a long-running process can
#: see many distinct hashes over its lifetime (per-user tool sets, admin
#: edits, MCP discovery churn); LRU eviction keeps memory flat.
DEFAULT_CACHE_SIZE = 64


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors; 0.0 for a zero vector."""
    dot: float = sum(x * y for x, y in zip(a, b))
    norm_a: float = sum(x * x for x in a) ** 0.5
    norm_b: float = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _tool_set_content_hash(tools: list[BaseTool]) -> str:
    """Deterministic hash of a tool set's (name, description) pairs.

    Order-independent (sorted before hashing) so the same set of tools in a
    different assembly order still hits the cache. Any addition, removal, or
    description edit changes the hash and invalidates automatically — no
    separate invalidation event is needed. This is the mechanism that makes
    caching safe for the live-discovered MCP slice and admin-editable
    ``service_tools`` rows, both of which can change between turns of the
    SAME conversation.
    """
    pairs = sorted((t.name, t.description or "") for t in tools)
    payload = json.dumps(pairs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class ToolSelectionResult:
    """Outcome of :meth:`ToolRelevanceIndex.select`.

    :ivar selected: the bound-tool subset — either the real top-K, or (on
        fallback) the full, unfiltered input list in its original order.
    :ivar fallback_used: ``True`` when ``selected`` is the full catalog
        because relevance selection could not run, NOT because top-K happened
        to equal the input size. Callers need this distinction for the
        narration-guard / tool_search-hit-rate metrics that are the actual
        test of the tool-count hypothesis (a silent "fallback that looks like
        a real top-K" would confound that measurement the same way the prod
        retrospective already is).
    :ivar fallback_reason: ``"embedder_error"``, ``"latency_ceiling"``, or
        ``None`` when no fallback occurred.
    """

    selected: list[BaseTool]
    fallback_used: bool = False
    fallback_reason: str | None = None


@dataclass(frozen=True)
class ToolSearchResult:
    """What one search of the catalog came back with.

    :meth:`ToolRelevanceIndex.search` drops the reason and returns the hits
    alone. ``tool_search`` needs the reason: a search that never finished is
    not a search that found nothing, and telling the model "no matching
    tools" when the index was still warming up sends it back to the person
    saying the tool does not exist.

    :ivar hits: up to ``limit`` tools, most relevant first; when the ranking
        did not run to completion, the tools whose name or description holds
        the query's words (:func:`match_words`), which may be none
    :ivar fallback_reason: ``"embedder_error"``, ``"latency_ceiling"``, or
        ``None`` when the search ran to completion
    """

    hits: list[BaseTool]
    fallback_reason: str | None = None


class ToolRelevanceIndex:
    """Embeds tool name+description and a query; returns a relevant top-K subset.

    One instance is safe to reuse across many turns/conversations — the
    content-hash cache is what makes that reuse both correct (distinct tool
    sets never collide) and valuable (an unchanged tool set across turns of
    the same conversation re-embeds nothing but the query).
    """

    def __init__(
        self,
        embedder: Embeddings,
        top_k: int = DEFAULT_TOP_K,
        latency_ceiling_s: float = DEFAULT_LATENCY_CEILING_S,
        cache_size: int = DEFAULT_CACHE_SIZE,
        search_latency_ceiling_s: float | None = None,
    ) -> None:
        """
        :param latency_ceiling_s: how long :meth:`select` waits before it
            falls back to the full catalog. Tight, because a turn is waiting
            on it and the fallback costs nothing but precision.
        :ptype latency_ceiling_s: float
        :param search_latency_ceiling_s: how long :meth:`search`,
            :meth:`search_scored` and :meth:`search_outcome` wait. Defaults
            to ``latency_ceiling_s``. A ``tool_search`` call is the model
            asking for a tool it needs, mid-turn, and has no fallback: a
            search cut off at the select ceiling comes back empty and reads
            as "there is no such tool". A cold catalog embeds in more than a
            select ceiling sized for the warm case, so a caller gives the
            search its own, longer, wait.
        :ptype search_latency_ceiling_s: float | None
        """
        self._embedder = embedder
        self._top_k = top_k
        self._latency_ceiling_s = latency_ceiling_s
        self._search_latency_ceiling_s = (
            latency_ceiling_s if search_latency_ceiling_s is None else search_latency_ceiling_s
        )
        self._cache_size = cache_size
        # content_hash -> {tool_name: embedding_vector}. OrderedDict as a
        # simple bounded LRU (move-to-end on hit, popitem(last=False) on
        # overflow) -- no external cache dependency needed at this size.
        self._cache: OrderedDict[str, dict[str, list[float]]] = OrderedDict()
        # content_hash -> the embedding of that tool set, while it runs. Its own task, so the
        # latency ceiling cancels a caller's wait and not the work: a catalog that takes longer
        # than the ceiling to embed still lands in the cache for the next turn, and turns that
        # arrive together share one embedding call.
        self._inflight: dict[str, asyncio.Task[dict[str, list[float]] | None]] = {}

    async def _safe_aembed_query(self, text: str) -> list[float] | None:
        """Soft-fail single-text embed. ``None`` on any failure or empty input."""
        if not text:
            return None
        try:
            result = await self._embedder.aembed_query(text)
        except Exception as exc:
            _log.warning(
                "tool-relevance query embedding failed (soft-fail)",
                extra={"extra_data": {"error": str(exc)}},
            )
            return None
        if not result:
            return None
        return result

    async def _safe_aembed_documents(self, texts: list[str]) -> list[list[float]] | None:
        """Soft-fail batch embed. ``None`` on any failure or a mismatched result length."""
        if not texts:
            return []
        try:
            result = await self._embedder.aembed_documents(texts)
        except Exception as exc:
            _log.warning(
                "tool-relevance batch embedding failed (soft-fail)",
                extra={"extra_data": {"error": str(exc), "text_count": len(texts)}},
            )
            return None
        if not result or len(result) != len(texts):
            return None
        return result

    async def _embed_tool_set(self, tools: list[BaseTool]) -> dict[str, list[float]] | None:
        """Return ``{tool_name: vector}`` for ``tools``, via cache when possible.

        ``None`` propagates a soft-fail from the batch embed call up to the
        caller, which must fall back to the full catalog -- this method never
        returns a partial/smaller mapping.
        """
        content_hash = _tool_set_content_hash(tools)
        cached = self._cache.get(content_hash)
        if cached is not None:
            self._cache.move_to_end(content_hash)
            return cached

        task = self._inflight.get(content_hash)
        if task is None:
            task = asyncio.get_running_loop().create_task(self._embed_and_cache(content_hash, list(tools)))
            self._inflight[content_hash] = task
            task.add_done_callback(functools.partial(self._forget_inflight, content_hash))
        return await asyncio.shield(task)

    def _forget_inflight(self, content_hash: str, _done: asyncio.Future[Any]) -> None:
        """Drop a finished embedding's task handle; its result, if any, is in the cache."""
        self._inflight.pop(content_hash, None)

    async def _embed_and_cache(self, content_hash: str, tools: list[BaseTool]) -> dict[str, list[float]] | None:
        """Embed one tool set and cache it; ``None`` on a soft-failed embed (nothing cached)."""
        texts = [f"{t.name}: {t.description or ''}" for t in tools]
        vectors = await self._safe_aembed_documents(texts)
        if vectors is None:
            return None

        embeddings = {t.name: vec for t, vec in zip(tools, vectors)}
        self._cache[content_hash] = embeddings
        self._cache.move_to_end(content_hash)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return embeddings

    async def _rank(self, tools: list[BaseTool], query: str) -> tuple[list[tuple[BaseTool, float]], str | None]:
        """Embed + rank; returns ``(ranked_pairs, fallback_reason)``.

        Each pair is a tool and its cosine similarity to the query, most
        relevant first. The score is carried rather than discarded because a
        caller deciding *whether* to act on the top hit needs it: an ordering
        alone makes the nearest tool look equally relevant to "what time is
        it" and to "morning, you". :meth:`select` and :meth:`search` keep
        their tool-only contracts and drop it.

        ``fallback_reason`` is ``None`` on success. On any embedding failure
        the ranked list is empty and the caller substitutes the full catalog
        -- this helper never invents a partial ranking.
        """
        tool_embeddings = await self._embed_tool_set(tools)
        if tool_embeddings is None:
            return [], "embedder_error"
        query_vec = await self._safe_aembed_query(query)
        if query_vec is None:
            return [], "embedder_error"

        scored: list[tuple[BaseTool, float]] = [
            (t, _cosine_similarity(query_vec, tool_embeddings[t.name])) for t in tools
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored, None

    async def select(self, tools: list[BaseTool], query: str) -> ToolSelectionResult:
        """Return the top-K most relevant tools for ``query``.

        No-ops (returns everything, no embedding calls) when ``len(tools) <=
        top_k`` -- nothing to filter. Otherwise embeds and ranks under the
        latency ceiling; a `TimeoutError` or any embedder failure returns the
        FULL unfiltered ``tools`` list, in its original order, with
        ``fallback_used=True``.

        :param tools: the full assembled tool set for this turn
        :ptype tools: list[BaseTool]
        :param query: the current turn's query text (typically the user
            message, optionally with a short recent-context window)
        :ptype query: str
        :return: the selection result, including whether fallback fired
        :rtype: ToolSelectionResult
        """
        if len(tools) <= self._top_k:
            return ToolSelectionResult(selected=list(tools))

        try:
            ranked, fallback_reason = await asyncio.wait_for(self._rank(tools, query), timeout=self._latency_ceiling_s)
        except TimeoutError:
            _log.warning(
                "tool-relevance selection exceeded latency ceiling; falling back to full catalog",
                extra={
                    "extra_data": {
                        "tool_count": len(tools),
                        "ceiling_s": self._latency_ceiling_s,
                    }
                },
            )
            return ToolSelectionResult(selected=list(tools), fallback_used=True, fallback_reason="latency_ceiling")

        if fallback_reason is not None:
            return ToolSelectionResult(selected=list(tools), fallback_used=True, fallback_reason=fallback_reason)
        return ToolSelectionResult(selected=[tool for tool, _ in ranked][: self._top_k])

    async def search_scored(self, tools: list[BaseTool], query: str, limit: int = 5) -> list[tuple[BaseTool, float]]:
        """Rank ``tools`` by relevance to ``query``, keeping each score.

        :meth:`search` answers "which tools are closest". This answers "and
        how close", which is the question a caller must ask before acting on
        a hit without a model in the loop. Ranking alone cannot separate a
        message that wants a tool from one that merely sits nearest to it:
        every message has a closest tool, including "morning, you".

        Same soft-fail contract as :meth:`search` -- an embedder failure, a
        run past the latency ceiling, or an empty catalog returns no hits
        rather than a guess, because "nothing was relevant enough" is a
        normal answer here and a wrong fetch is not.

        :param tools: the catalog to rank (the caller decides scope)
        :ptype tools: list[BaseTool]
        :param query: natural-language text to rank against
        :ptype query: str
        :param limit: maximum number of hits to return
        :ptype limit: int
        :return: up to ``limit`` ``(tool, score)`` pairs, most relevant
            first; empty on failure, timeout, or an empty catalog
        :rtype: list[tuple[BaseTool, float]]
        """
        if not tools:
            return []
        ranked, fallback_reason = await self._rank_for_search(tools, query, what="scored search")
        if fallback_reason is not None:
            return []
        return ranked[:limit]

    async def _rank_for_search(
        self, tools: list[BaseTool], query: str, *, what: str
    ) -> tuple[list[tuple[BaseTool, float]], str | None]:
        """:meth:`_rank` under the search ceiling; a timeout is a fallback reason, not an exception."""
        try:
            return await asyncio.wait_for(self._rank(tools, query), timeout=self._search_latency_ceiling_s)
        except TimeoutError:
            _log.warning(
                f"tool-relevance {what} exceeded latency ceiling; returning no hits",
                extra={
                    "extra_data": {
                        "tool_count": len(tools),
                        "ceiling_s": self._search_latency_ceiling_s,
                    }
                },
            )
            return [], "latency_ceiling"

    async def search(self, tools: list[BaseTool], query: str, limit: int = 5) -> list[BaseTool]:
        """Rank ``tools`` (typically the FULL catalog) by relevance to ``query``.

        No smaller-than-input fallback contract here (unlike :meth:`select`):
        a failed OR slow search soft-fails to an empty list of hits.
        ``tool_search`` failing to find anything is a normal, recoverable
        outcome for its caller -- it does not gate a turn's entire tool
        surface the way :meth:`select` does. Bounded by the search ceiling
        (``search_latency_ceiling_s``, the select ceiling unless the caller
        set one) -- an unresponsive-but-not-raising embedder must not hang a
        mid-turn ``tool_search`` call indefinitely. A caller that needs to
        tell "nothing matched" from "did not finish" uses
        :meth:`search_outcome`.

        :param tools: the catalog to search (the caller decides scope -- the
            sign-off contract for ``tool_search`` is that this is the FULL
            catalog, not the already-filtered top-K)
        :ptype tools: list[BaseTool]
        :param query: natural-language description of the desired tool
        :ptype query: str
        :param limit: maximum number of hits to return
        :ptype limit: int
        :return: up to ``limit`` tools, most relevant first; empty on
            failure, timeout, or an empty catalog
        :rtype: list[BaseTool]
        """
        return (await self.search_outcome(tools, query, limit=limit)).hits

    async def search_outcome(self, tools: list[BaseTool], query: str, limit: int = 5) -> ToolSearchResult:
        """:meth:`search`, keeping why the hits are empty when they are.

        :param tools: the catalog to search (the caller decides scope)
        :ptype tools: list[BaseTool]
        :param query: natural-language description of the desired tool
        :ptype query: str
        :param limit: maximum number of hits to return
        :ptype limit: int
        :return: the hits and, when the search did not run to completion,
            the reason
        :rtype: ToolSearchResult
        """
        if not tools:
            return ToolSearchResult(hits=[])
        ranked, fallback_reason = await self._rank_for_search(tools, query, what="search")
        if fallback_reason is not None:
            # The ranking failed, not the catalog: match on words so the search
            # still finds a tool. A consumer that binds only the pick and
            # tool_search has no other way to reach anything while the embedder
            # is down.
            return ToolSearchResult(hits=match_words(tools, query)[:limit], fallback_reason=fallback_reason)
        return ToolSearchResult(hits=[tool for tool, _ in ranked][:limit])


#: Words too common to say anything about which tool is wanted.
_STOP_WORDS = frozenset(
    "the and for with that this from into what when where which about have has are was were can could "
    "would should will you your our their them then than there here some any all not but how who why".split()
)


def match_words(tools: list[BaseTool], query: str) -> list[BaseTool]:
    """The tools whose name or description holds the query's words, most words first.

    For when the embedding ranking cannot run. Words of three letters or more,
    common words dropped; a tool's name is read with its separators as spaces,
    so ``web_search`` matches "search". Ties keep catalog order.

    :param tools: the catalog
    :ptype tools: list[BaseTool]
    :param query: what the caller asked for
    :ptype query: str
    :return: the matching tools, best first; empty when nothing matches
    :rtype: list[BaseTool]
    """
    words = {w for w in _words(query) if len(w) >= 3 and w not in _STOP_WORDS}
    if not words:
        return []
    scored = []
    for position, tool in enumerate(tools):
        text = set(_words(f"{tool.name} {tool.description or ''}"))
        score = sum(1 for w in words if w in text)
        if score:
            scored.append((-score, position, tool))
    return [tool for _, _, tool in sorted(scored, key=lambda item: (item[0], item[1]))]


def _words(text: str) -> list[str]:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()


class _ToolSearchInput(BaseModel):
    """Input schema for the ``tool_search`` meta-tool."""

    query: str = Field(
        description="What you need the tool to do, in plain words.",
    )


def create_tool_search_tool(
    index: ToolRelevanceIndex,
    full_catalog_provider: Callable[[], list[BaseTool]],
    on_hit: Callable[[list[BaseTool]], None],
    limit: int = 5,
) -> BaseTool:
    """Build the ``tool_search`` meta-tool.

    Mirrors Anthropic's Tool Search Tool behaviour: the model queries the
    FULL catalog mid-turn for a capability it needs but wasn't handed. Hits
    are reported to ``on_hit`` (the caller's mid-turn rebind hook -- e.g.
    metallm's tool-loop composes them into the next round's bound set,
    the same shape it already uses for ``skill_invoke``); this factory has no
    notion of "next round", that composition is entirely the caller's.

    ``full_catalog_provider`` is a closure re-invoked on every call (not
    snapshotted at factory-construction time) so a live-discovered or
    admin-edited catalog is always searched as of THIS call, not as of
    whenever the tool was bound.

    :param index: the shared relevance index (embeds + ranks)
    :ptype index: ToolRelevanceIndex
    :param full_catalog_provider: zero-arg callable returning the current
        full tool catalog; called fresh on every ``tool_search`` invocation
    :ptype full_catalog_provider: Callable[[], list[BaseTool]]
    :param on_hit: called with the matched tools when the search finds at
        least one; the caller uses this to widen its bound tool set
    :ptype on_hit: Callable[[list[BaseTool]], None]
    :param limit: maximum number of hits to return per call
    :ptype limit: int
    :return: the ``tool_search`` ``BaseTool``
    :rtype: BaseTool
    """

    async def _search(query: str) -> str:
        catalog = full_catalog_provider()
        outcome = await index.search_outcome(catalog, query, limit=limit)
        matches = outcome.hits
        if outcome.fallback_reason is not None and not matches:
            # The search did not run, so nothing was found and nothing was
            # ruled out. Live (metallm, the first turn after a deploy): a
            # cold catalog ran past the ceiling, the model read "No matching
            # tools found." and told the person the tool did not exist.
            _log.info(
                "tool_search: did not finish",
                extra={
                    "extra_data": {
                        "query": query[:200],
                        "catalog_size": len(catalog),
                        "reason": outcome.fallback_reason,
                    }
                },
            )
            if outcome.fallback_reason == "latency_ceiling":
                return (
                    "Tool search did not finish in time. Nothing was found and nothing was ruled out. "
                    "Run the same search once more."
                )
            return "Tool search failed. Nothing was found and nothing was ruled out."
        if not matches:
            _log.info(
                "tool_search: no matches",
                extra={"extra_data": {"query": query[:200], "catalog_size": len(catalog)}},
            )
            return "No matching tools found."

        on_hit(matches)
        _log.info(
            "tool_search: hits",
            extra={
                "extra_data": {
                    "query": query[:200],
                    "hit_count": len(matches),
                    "hit_names": [t.name for t in matches],
                }
            },
        )
        lines = [f"- {t.name}: {t.description or ''}" for t in matches]
        # Live prod evidence (metallm conv 019f6cf5-073a-7b50-bd44-721efb0c7b90):
        # this tool's own DESCRIPTION correctly says hits become callable
        # "starting your NEXT reply", but this return string used to say "now
        # available to call" -- a direct contradiction the model reads
        # immediately after invoking the tool, mid-round. That's what drove
        # the model to attempt the newly-found tool right away, in the SAME
        # round, and bounce off "No such tool available": the caller (e.g.
        # metallm's tool-loop) can only compose a hit into the bound set at
        # the next round boundary (see ``on_hit`` docstring above), never
        # sooner -- no caller can make a tool available mid-round. Wording
        # here must match the description's honest framing, not promise
        # immediacy the caller cannot deliver.
        return "Found these tools. Call them after this search returns, not alongside it:\n" + "\n".join(lines)

    return StructuredTool.from_function(
        coroutine=_search,
        name="tool_search",
        description=(
            "Find a tool you do not have yet. Describe what you need in plain words, e.g. "
            "'send a message to a dev agent session'. Call what it finds after this search "
            "returns, not alongside it."
        ),
        args_schema=_ToolSearchInput,
    )
