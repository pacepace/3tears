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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from threetears.observe.text_block import under
from threetears.observe import get_logger
from threetears.tool_schema.self_contained import declared_type

from threetears.agent.tools.utils import tool_no_change

__all__ = [
    "DEFAULT_SEARCH_MAX_LIMIT",
    "ToolRelevanceIndex",
    "ToolSearchFamilyInput",
    "ToolSearchInput",
    "ToolSearchResult",
    "ToolSelectionResult",
    "also_in_family",
    "create_tool_search_tool",
    "tool_arguments",
    "tool_families",
    "tool_hit_line",
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
    :ivar ranked: the whole ranking, most relevant first, each tool beside its
        similarity to the query; ``hits`` is its first ``limit`` tools. Empty
        when the ranking did not run, and the hits are then word matches with
        no score. A caller that must decide whether anything fits reads the
        scores here and judges by a margin -- the gap between the first hit and
        a later one, or against a baseline of its own -- never by one fixed
        number: two embedders score in different ranges
    """

    hits: list[BaseTool]
    fallback_reason: str | None = None
    ranked: list[tuple[BaseTool, float]] = field(default_factory=list)


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
        return ToolSearchResult(hits=[tool for tool, _ in ranked][:limit], ranked=ranked)


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


#: The sentence every list of found tools ends its first line with: found is not bound until the next round.
_CALL_AFTER = "Call them after this search returns, not alongside it:"

#: The most tools one ``tool_search`` call returns unless its builder says otherwise.
DEFAULT_SEARCH_MAX_LIMIT = 20

#: How many of a family's other tools a hit list names before it counts the rest.
_SIBLINGS_NAMED = 8


def _input_schema(tool: BaseTool) -> Mapping[str, Any]:
    """The JSON schema of what a model passes ``tool``, without the arguments a host injects."""
    schema = tool.tool_call_schema
    if isinstance(schema, Mapping):
        return schema
    if issubclass(schema, BaseModel):
        return schema.model_json_schema()
    return schema.schema()  # a pydantic v1 model, which LangChain still accepts as a tool's schema


def tool_arguments(tool: BaseTool) -> str:
    """A tool's arguments as one bracket: ``(title: string, list_name?: string)``.

    A ``?`` marks an argument that may be left out. The type is the one the
    schema declares; an argument that may be several types is named without
    one. A tool that takes nothing is ``()``.

    :param tool: the tool
    :ptype tool: BaseTool
    :return: the bracket
    :rtype: str
    """
    schema = _input_schema(tool)
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    parts = []
    for name, prop in properties.items():
        kind = declared_type(prop, schema)
        parts.append(f"{name}{'' if name in required else '?'}" + (f": {kind}" if kind else ""))
    return f"({', '.join(parts)})"


def tool_hit_line(tool: BaseTool) -> str:
    """One found tool as the line a model reads: its name, its arguments, what it does.

    :param tool: the tool
    :ptype tool: BaseTool
    :return: ``- name(arguments): description``
    :rtype: str
    """
    head = f"- {tool.name}{tool_arguments(tool)}: "
    # a description of many lines sits under its tool, or its later lines read as the next tool's
    return "\n".join(under(tool.description or "", "  ", first=head))


def tool_families(catalog: Sequence[BaseTool], group_of: Callable[[BaseTool], str | None]) -> dict[str, list[BaseTool]]:
    """The catalog by family, each family's tools in catalog order; a tool with no family is left out.

    :param catalog: the tools
    :ptype catalog: Sequence[BaseTool]
    :param group_of: a tool's family, or ``None`` when it has none
    :ptype group_of: Callable[[BaseTool], str | None]
    :return: family name to its tools, families in the order they first appear
    :rtype: dict[str, list[BaseTool]]
    """
    families: dict[str, list[BaseTool]] = {}
    for tool in catalog:
        family = group_of(tool)
        if family:
            families.setdefault(family, []).append(tool)
    return families


def also_in_family(
    hits: Sequence[BaseTool], catalog: Sequence[BaseTool], group_of: Callable[[BaseTool], str | None]
) -> list[str]:
    """For each family a hit belongs to, a line naming that family's other tools.

    A search answers with a few tools, and the one wanted is often the hit's
    neighbour: found ``calendar_list``, wanted ``calendar_move``. Names only,
    so the line stays short; the tools named are not handed over.

    :param hits: the tools the search answers with
    :ptype hits: Sequence[BaseTool]
    :param catalog: the tools a family's other members are taken from
    :ptype catalog: Sequence[BaseTool]
    :param group_of: a tool's family, or ``None`` when it has none
    :ptype group_of: Callable[[BaseTool], str | None]
    :return: one ``Also in the <family> family: ...`` line per family that has other tools
    :rtype: list[str]
    """
    families = tool_families(catalog, group_of)
    shown = {hit.name for hit in hits}
    lines: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        family = group_of(hit)
        if not family or family in seen:
            continue
        seen.add(family)
        others = [tool.name for tool in families.get(family, []) if tool.name not in shown]
        if not others:
            continue
        named = ", ".join(others[:_SIBLINGS_NAMED])
        more = f", and {len(others) - _SIBLINGS_NAMED} more" if len(others) > _SIBLINGS_NAMED else ""
        lines.append(f"Also in the {family} family: {named}{more}.")
    return lines


class ToolSearchInput(BaseModel):
    """What ``tool_search`` takes over a catalog without families.

    Public so a consumer that adds its own argument to the tool extends this
    class and checks ``tool.args_schema`` against it.
    """

    query: str = Field(
        description="What you need the tool to do, in plain words.",
    )
    limit: int | None = Field(
        default=None,
        description="The most tools to return. Leave it out for the usual few.",
    )


class ToolSearchFamilyInput(ToolSearchInput):
    """What ``tool_search`` takes over a catalog that has families (``group_of`` given).

    Public for the same reason as :class:`ToolSearchInput`.
    """

    query: str = Field(
        default="",
        description="What you need the tool to do, in plain words. Leave it out to list a whole family.",
    )
    family: str | None = Field(
        default=None,
        description="Search only this family of tools, by the family's name.",
    )


def create_tool_search_tool(
    index: ToolRelevanceIndex,
    full_catalog_provider: Callable[[], list[BaseTool]],
    on_hit: Callable[[list[BaseTool]], None],
    limit: int = 5,
    *,
    max_limit: int = DEFAULT_SEARCH_MAX_LIMIT,
    group_of: Callable[[BaseTool], str | None] | None = None,
    nothing_fits: Callable[[str, Sequence[tuple[BaseTool, float]]], bool] | None = None,
    offer: Callable[[Sequence[BaseTool], Sequence[BaseTool]], tuple[list[str], list[BaseTool]]] | None = None,
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

    Each tool found is shown with its arguments (:func:`tool_hit_line`), so
    the call that follows the search needs no second look at the tool.

    With ``group_of`` the catalog has families. The model may then pass
    ``family`` to search one family, or ``family`` with no query to list it;
    a family that does not exist is answered with the families that do; and
    a hit list ends by naming each hit's family's other tools
    (:func:`also_in_family`).

    ``nothing_fits`` is how the caller says a ranking found nothing worth
    offering. It is handed the query and the whole scored ranking
    (:attr:`ToolSearchResult.ranked`) and answers ``True`` for "offer none of
    these": the search then says no tool fits, names the nearest, and hands
    nothing to ``on_hit``. This factory holds no threshold of its own. A fixed
    similarity floor is wrong across embedders, whose scores sit in different
    ranges; a caller judges by a margin it has measured. It is not asked when
    the ranking did not run: word matches carry no score to judge.

    ``offer`` is how the caller shows what was found and decides what is
    handed over. It is given the tools found and the whole catalog, and
    answers with one line per tool found and the tools ``on_hit`` receives. A
    caller whose surface keeps some tools behind another (a helper that runs
    them) names the way in on the tool's line and hands that over instead.
    Left out, each tool found is shown as :func:`tool_hit_line` and handed
    over as it is. Every other sentence of the result stays this factory's.

    :param index: the shared relevance index (embeds + ranks)
    :ptype index: ToolRelevanceIndex
    :param full_catalog_provider: zero-arg callable returning the current
        full tool catalog; called fresh on every ``tool_search`` invocation
    :ptype full_catalog_provider: Callable[[], list[BaseTool]]
    :param on_hit: called with the matched tools when the search finds at
        least one; the caller uses this to widen its bound tool set
    :ptype on_hit: Callable[[list[BaseTool]], None]
    :param limit: how many hits a call returns when the model names no number
    :ptype limit: int
    :param max_limit: the most hits one call returns, whatever the model asks for
    :ptype max_limit: int
    :param group_of: a tool's family, or ``None`` when it has none
    :ptype group_of: Callable[[BaseTool], str | None] | None
    :param nothing_fits: given the query and the scored ranking, whether to offer none of it
    :ptype nothing_fits: Callable[[str, Sequence[tuple[BaseTool, float]]], bool] | None
    :param offer: given the tools found and the whole catalog, the line for each and the tools to hand over
    :ptype offer: Callable[[Sequence[BaseTool], Sequence[BaseTool]], tuple[list[str], list[BaseTool]]] | None
    :return: the ``tool_search`` ``BaseTool``
    :rtype: BaseTool
    """
    default_limit = max(1, min(limit, max_limit))

    def _offered(matches: list[BaseTool], catalog: list[BaseTool]) -> list[str]:
        """Hand the tools found to ``on_hit`` and answer with the line for each."""
        if offer is None:
            on_hit(matches)
            return [tool_hit_line(t) for t in matches]
        lines, handed = offer(matches, catalog)
        if handed:
            on_hit(handed)
        return lines

    def _found(matches: list[BaseTool], catalog: list[BaseTool], header: str) -> str:
        lines = _offered(matches, catalog)
        if group_of is not None:
            lines += also_in_family(matches, catalog, group_of)
        return header + "\n" + "\n".join(lines)

    async def _search(query: str = "", limit: int | None = None, family: str | None = None) -> str:
        whole_catalog = full_catalog_provider()
        catalog = whole_catalog
        wanted = default_limit if limit is None else max(1, min(limit, max_limit))
        query = query.strip()
        family = (family or "").strip() or None
        if family is not None and group_of is not None:
            families = tool_families(whole_catalog, group_of)
            by_folded_name = {name.casefold(): name for name in families}
            family_name = by_folded_name.get(family.casefold())
            if family_name is None:
                known = ", ".join(sorted(families)) or "none"
                return tool_no_change(
                    "tool_search", f"there is no tool family named {family!r}. The families: {known}."
                )
            catalog = families[family_name]
            if not query:
                shown = catalog[:wanted]
                count = (
                    f"{len(catalog)} tools, the first {len(shown)}"
                    if len(shown) < len(catalog)
                    else f"{len(catalog)} tools"
                )
                _log.info(
                    "tool_search: family listed",
                    extra={"extra_data": {"family": family_name, "family_size": len(catalog), "shown": len(shown)}},
                )
                shown_lines = _offered(shown, whole_catalog)
                rest = (
                    [f"The rest: {', '.join(t.name for t in catalog[wanted:])}."] if len(shown) < len(catalog) else []
                )
                return "\n".join([f"The {family_name} family has {count}. {_CALL_AFTER}", *rest, *shown_lines])
        if not query:
            return "[TOOL ERROR] tool_search: say what you need the tool to do."

        outcome = await index.search_outcome(catalog, query, limit=wanted)
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
                # a failure, so a caller records it as one: plain text read as a search that ran
                return (
                    "[TOOL ERROR] Tool search did not finish in time. Nothing was found and nothing was ruled out. "
                    "Run the same search once more."
                )
            return "[TOOL ERROR] Tool search failed. Nothing was found and nothing was ruled out."
        if not matches:
            _log.info(
                "tool_search: no matches",
                extra={"extra_data": {"query": query[:200], "catalog_size": len(catalog)}},
            )
            return "No matching tools found."

        if outcome.fallback_reason is not None:
            # Word matches, and said so: a tool whose name shares a word with the query is
            # not the tool ranked nearest to it, and read as that it is called for the wrong job.
            _log.info(
                "tool_search: word matches",
                extra={
                    "extra_data": {
                        "query": query[:200],
                        "reason": outcome.fallback_reason,
                        "hit_names": [t.name for t in matches],
                    }
                },
            )
            return _found(
                matches,
                whole_catalog,
                "Found these tools by matching words only: the ranked search did not run, so a closer tool may "
                f"exist. {_CALL_AFTER}",
            )

        if nothing_fits is not None and nothing_fits(query, outcome.ranked):
            _log.info(
                "tool_search: nothing fits",
                extra={
                    "extra_data": {
                        "query": query[:200],
                        "nearest": [(t.name, round(score, 4)) for t, score in outcome.ranked[:wanted]],
                    }
                },
            )
            nearest = ", ".join(t.name for t in matches)
            return f"No tool fits that. The nearest, none of them handed to you: {nearest}."

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
        return _found(matches, whole_catalog, f"Found these tools. {_CALL_AFTER}")

    description = (
        "Find a tool you do not have yet. Describe what you need in plain words, e.g. "
        "'send a message to a dev agent session'. Call what it finds after this search "
        "returns, not alongside it."
    )
    if group_of is not None:
        description += " Pass family to search one family of tools, or family alone to list it."
    return StructuredTool.from_function(
        coroutine=_search,
        name="tool_search",
        description=description,
        args_schema=ToolSearchFamilyInput if group_of is not None else ToolSearchInput,
    )
