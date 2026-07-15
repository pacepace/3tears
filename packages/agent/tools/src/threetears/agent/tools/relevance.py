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
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from threetears.observe import get_logger

__all__ = [
    "ToolRelevanceIndex",
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
    ) -> None:
        self._embedder = embedder
        self._top_k = top_k
        self._latency_ceiling_s = latency_ceiling_s
        self._cache_size = cache_size
        # content_hash -> {tool_name: embedding_vector}. OrderedDict as a
        # simple bounded LRU (move-to-end on hit, popitem(last=False) on
        # overflow) -- no external cache dependency needed at this size.
        self._cache: OrderedDict[str, dict[str, list[float]]] = OrderedDict()

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

    async def _rank(
        self, tools: list[BaseTool], query: str
    ) -> tuple[list[BaseTool], str | None]:
        """Embed + rank; returns ``(ranked_tools, fallback_reason)``.

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
        return [t for t, _ in scored], None

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
            ranked, fallback_reason = await asyncio.wait_for(
                self._rank(tools, query), timeout=self._latency_ceiling_s
            )
        except TimeoutError:
            _log.warning(
                "tool-relevance selection exceeded latency ceiling; "
                "falling back to full catalog",
                extra={
                    "extra_data": {
                        "tool_count": len(tools),
                        "ceiling_s": self._latency_ceiling_s,
                    }
                },
            )
            return ToolSelectionResult(
                selected=list(tools), fallback_used=True, fallback_reason="latency_ceiling"
            )

        if fallback_reason is not None:
            return ToolSelectionResult(
                selected=list(tools), fallback_used=True, fallback_reason=fallback_reason
            )
        return ToolSelectionResult(selected=ranked[: self._top_k])

    async def search(self, tools: list[BaseTool], query: str, limit: int = 5) -> list[BaseTool]:
        """Rank ``tools`` (typically the FULL catalog) by relevance to ``query``.

        No smaller-than-input fallback contract here (unlike :meth:`select`):
        a failed OR slow search soft-fails to an empty list of hits.
        ``tool_search`` failing to find anything is a normal, recoverable
        outcome for its caller -- it does not gate a turn's entire tool
        surface the way :meth:`select` does. Still bounded by the SAME
        latency ceiling as :meth:`select` (both share one embedder and one
        failure surface) -- an unresponsive-but-not-raising embedder must
        not hang a mid-turn ``tool_search`` call indefinitely.

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
        if not tools:
            return []
        try:
            ranked, fallback_reason = await asyncio.wait_for(
                self._rank(tools, query), timeout=self._latency_ceiling_s
            )
        except TimeoutError:
            _log.warning(
                "tool-relevance search exceeded latency ceiling; returning no hits",
                extra={
                    "extra_data": {
                        "tool_count": len(tools),
                        "ceiling_s": self._latency_ceiling_s,
                    }
                },
            )
            return []
        if fallback_reason is not None:
            return []
        return ranked[:limit]


class _ToolSearchInput(BaseModel):
    """Input schema for the ``tool_search`` meta-tool."""

    query: str = Field(
        description=(
            "Natural-language description of the tool you need, e.g. "
            "'a tool to send a message to a dev agent session'"
        ),
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
        matches = await index.search(catalog, query, limit=limit)
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
        return "Found the following tools, now available to call:\n" + "\n".join(lines)

    return StructuredTool.from_function(
        coroutine=_search,
        name="tool_search",
        description=(
            "Search the full tool catalog for a capability not in your currently "
            "bound tools. Matching tools become callable starting your NEXT reply. "
            "Use a natural-language query describing what you need, e.g. 'a tool "
            "to send a message to a dev agent session'."
        ),
        args_schema=_ToolSearchInput,
    )
