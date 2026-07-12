"""Dream consolidation -- merges clusters of near-duplicate memories into gists.

A ``DreamService`` sibling to :class:`~threetears.agent.memory.extraction.MemoryExtractor`,
reusing the SAME ``Embeddings`` + ``ChatModelFactory`` deps (no new external
dependency). It reads the salience/scope substrate (v024) and writes the
``memory_consolidations`` edge table (v026).

The cycle, per pinned scope grain (agent / customer / user):

1. Load active memories with embeddings, excluding ``evergreen`` and any
   already superseded by a LIVE gist
   (:meth:`MemoriesCollection.find_active_for_consolidation`).
2. Greedy union-find clustering at cosine >= ``cluster_threshold``.
3. For each cluster of at least ``min_cluster_size``: the reflector
   (``ChatModelFactory``, ``purpose="consolidation"``) synthesizes a gist.
   The gist is embedded and committed as a new memory (salience seeded,
   scope inherited from the pinned grain, ``conversation_id`` inherited from
   the newest source since the column is NOT NULL and a gist spans several
   conversations). ``consolidates`` edges (gist <- each source) are recorded
   with a ``rationale``, then each source's ``superseded_by`` is set to the
   gist.

Auto-apply, non-destructive, no consent gate: sources are kept (superseded,
still directly recallable), never deleted. The edge + ``rationale`` are the
audit trail. A :class:`~threetears.agent.memory.events.MemoryConsolidatedEvent`
is emitted per merge for the presence timeline.

``run_consolidation`` is resilient at the cluster grain: a failing cluster
(reflector error, unparseable response, embedding failure, or a DB write
error) is logged and skipped so the remaining clusters still consolidate. A
failure after a gist commits but before its sources are superseded leaves a
committed-but-unreferenced gist -- non-destructive (the sources are untouched
and still recallable) and self-correcting (a later pass re-clusters and
re-consolidates the sources), so no rollback is attempted.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from uuid_utils import uuid7

from threetears.agent.memory.collections import (
    MemoriesCollection,
    MemoryConsolidationsCollection,
)
from threetears.agent.memory.embedding_utils import _safe_aembed_query
from threetears.agent.memory.entities import MemoryConsolidationEntity, MemoryEntity
from threetears.agent.memory.events import MemoryConsolidatedEvent
from threetears.agent.memory.prompts import ExtractionPrompts
from threetears.agent.memory.types import MemoryConfig, MemoryType
from threetears.langgraph.events import dispatch_event
from threetears.observe import get_logger, traced

__all__ = [
    "ConsolidationResult",
    "DreamService",
    "ReflectorChatModelFactory",
]

log = get_logger(__name__)

# declaration order pins the deterministic tie-break when a cluster's source
# types are evenly split -- the earliest-declared type wins.
_TYPE_ORDER: dict[str, int] = {t.value: i for i, t in enumerate(MemoryType)}


@runtime_checkable
class ReflectorChatModelFactory(Protocol):
    """Protocol for creating the chat model that synthesizes a gist.

    Structurally identical to
    :class:`~threetears.agent.memory.extraction.ChatModelFactory`; declared
    here so ``dream`` does not import the extractor's surface. Production
    wiring passes one factory to both.
    """

    async def create_chat_model(self, purpose: str = "extraction") -> Any:
        """Create a chat model (returns LangChain BaseChatModel or compatible).

        purpose: ``"consolidation"`` for the Dream reflector.
        """
        ...


class ConsolidationResult:
    """Immutable summary of one :meth:`DreamService.run_consolidation` pass.

    Carries the counts a scheduler / observability surface reports and the
    tests assert against. No behaviour — a plain value object.

    :ivar memories_considered: candidates loaded for this scope grain
    :ivar clusters_formed: clusters at or above ``min_cluster_size``
    :ivar gists_created: gists successfully committed (a cluster whose
        reflector failed is counted in ``clusters_formed`` but not here)
    :ivar sources_superseded: source memories pointed at a new gist
    :ivar gist_ids: the committed gist memory ids
    """

    __slots__ = (
        "memories_considered",
        "clusters_formed",
        "gists_created",
        "sources_superseded",
        "gist_ids",
    )

    def __init__(
        self,
        *,
        memories_considered: int,
        clusters_formed: int,
        gists_created: int,
        sources_superseded: int,
        gist_ids: list[UUID],
    ) -> None:
        """initialize the result value object.

        :param memories_considered: candidates loaded
        :ptype memories_considered: int
        :param clusters_formed: clusters at or above min size
        :ptype clusters_formed: int
        :param gists_created: gists committed
        :ptype gists_created: int
        :param sources_superseded: sources marked superseded
        :ptype sources_superseded: int
        :param gist_ids: committed gist ids
        :ptype gist_ids: list[UUID]
        """
        self.memories_considered = memories_considered
        self.clusters_formed = clusters_formed
        self.gists_created = gists_created
        self.sources_superseded = sources_superseded
        self.gist_ids = gist_ids


def _cosine(a: list[float], b: list[float], norm_a: float, norm_b: float) -> float:
    """cosine similarity of two vectors given their precomputed norms.

    :param a: first vector
    :ptype a: list[float]
    :param b: second vector
    :ptype b: list[float]
    :param norm_a: L2 norm of ``a``
    :ptype norm_a: float
    :param norm_b: L2 norm of ``b``
    :ptype norm_b: float
    :return: cosine similarity, or 0.0 when either norm is zero
    :rtype: float
    """
    denom = norm_a * norm_b
    if denom == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=False)) / denom


def _cluster_by_cosine(
    embeddings: list[list[float]],
    threshold: float,
) -> list[list[int]]:
    """greedy union-find clustering of embeddings at a cosine threshold.

    Pure function (no I/O) so the clustering logic is unit-tested
    exhaustively. Two memories join the same cluster when their cosine
    similarity is at least ``threshold``; membership is transitive (union-
    find connected components), so a chain of pairwise-similar memories
    forms one cluster. O(n^2) pairwise — bounded by a single scope grain's
    memory count (hundreds), well within a background pass.

    :param embeddings: candidate embedding vectors, index-aligned to the
        caller's candidate list
    :ptype embeddings: list[list[float]]
    :param threshold: minimum cosine similarity to union two memories
    :ptype threshold: float
    :return: clusters as lists of candidate indices; every index appears in
        exactly one cluster (singletons included)
    :rtype: list[list[int]]
    """
    n = len(embeddings)
    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        # path compression keeps repeated finds near-constant.
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    norms = [math.sqrt(sum(v * v for v in e)) for e in embeddings]
    for i in range(n):
        for j in range(i + 1, n):
            if _cosine(embeddings[i], embeddings[j], norms[i], norms[j]) >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _modal_type(types: list[str]) -> str:
    """pick the most common memory type in a cluster, deterministically.

    Ties break by :class:`MemoryType` declaration order (the earliest-
    declared type wins) so the gist's type is reproducible across runs.

    :param types: source ``type_memory`` values in the cluster (non-empty)
    :ptype types: list[str]
    :return: the modal type
    :rtype: str
    """
    counts: dict[str, int] = {}
    for t in types:
        counts[t] = counts.get(t, 0) + 1
    return max(counts, key=lambda t: (counts[t], -_TYPE_ORDER.get(t, len(_TYPE_ORDER))))


def _strip_code_block(content: str) -> str:
    """strip a markdown fence + surrounding prose from an LLM JSON response.

    Mirrors the extractor's tolerance: models sometimes wrap JSON in a
    ```` ``` ```` fence or prepend reasoning prose. Extracts the outermost
    ``{...}`` object so ``json.loads`` sees only the payload.

    :param content: raw model content
    :ptype content: str
    :return: the inner JSON text
    :rtype: str
    """
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        if len(lines) > 2:
            content = "\n".join(lines[1:-1]).strip()
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end > start:
        content = content[start : end + 1]
    return content


class DreamService:
    """Generates consolidation gists from clusters of near-duplicate memories."""

    def __init__(
        self,
        config: MemoryConfig,
        embedding_provider: Embeddings,
        chat_model_factory: ReflectorChatModelFactory,
        memories_collection: MemoriesCollection,
        consolidations_collection: MemoryConsolidationsCollection,
        prompts: ExtractionPrompts | None = None,
    ) -> None:
        """initialize the Dream consolidation service.

        No RBAC authorizer: consolidation is a platform-invoked background
        maintenance pass over a single agent partition (the same class of
        operation as :meth:`MemoriesCollection.decay_salience` /
        ``bump_salience``, which are also unauthorized), not a request-path
        operation with a caller identity to gate.

        :param config: memory configuration; supplies the consolidation
            knobs (cluster threshold, min cluster size, gist salience seed)
        :ptype config: MemoryConfig
        :param embedding_provider: embedding provider used to embed the
            synthesized gist (same protocol the extractor uses)
        :ptype embedding_provider: Embeddings
        :param chat_model_factory: factory producing the reflector chat
            model (``purpose="consolidation"``)
        :ptype chat_model_factory: ReflectorChatModelFactory
        :param memories_collection: three-tier memories collection; the
            candidate load, gist commit, and supersession write go through it
        :ptype memories_collection: MemoriesCollection
        :param consolidations_collection: edge-table collection; the
            provenance edges + cycle-guard go through it
        :ptype consolidations_collection: MemoryConsolidationsCollection
        :param prompts: prompt bundle (defaults to :class:`ExtractionPrompts`);
            the ``consolidation`` field is the reflector template
        :ptype prompts: ExtractionPrompts | None
        """
        self._config = config
        self._embedding_provider = embedding_provider
        self._chat_model_factory = chat_model_factory
        self._memories = memories_collection
        self._consolidations = consolidations_collection
        self._prompts = prompts or ExtractionPrompts()

    @traced(record_args=False)
    async def run_consolidation(
        self,
        agent_id: UUID,
        *,
        customer_id: UUID | None = None,
        user_id: UUID | None = None,
    ) -> ConsolidationResult:
        """cluster + merge the active memories in one scope grain into gists.

        Clusters WITHIN the pinned scope grain (see
        :meth:`MemoriesCollection.find_active_for_consolidation`); the gist
        inherits that grain's identity — ``customer_id`` / ``user_id`` are
        passed straight through, so an agent-scoped run mints agent-scoped
        gists (both null), a user-scoped run mints user-scoped gists.
        metallm always invokes per-user, so its gists are user-scoped and
        never leak across users.

        Fire-and-forget safe at the cluster grain: a cluster whose reflector
        or embedding fails is logged and skipped; the pass still returns a
        result summarizing what committed.

        :param agent_id: partition column (required)
        :ptype agent_id: UUID
        :param customer_id: customer grain; ``None`` = agent-/customer-scoped
        :ptype customer_id: UUID | None
        :param user_id: user grain; ``None`` = agent-/customer-scoped
        :ptype user_id: UUID | None
        :return: a summary of the pass
        :rtype: ConsolidationResult
        """
        candidates = await self._memories.find_active_for_consolidation(
            agent_id,
            customer_id=customer_id,
            user_id=user_id,
        )
        # drop any row whose embedding failed to decode (defensive: the load
        # already filters embedding IS NOT NULL, so this is belt-and-braces).
        candidates = [c for c in candidates if c.get("embedding")]
        considered = len(candidates)
        min_size = self._config.consolidation_min_cluster_size
        if considered < min_size:
            return ConsolidationResult(
                memories_considered=considered,
                clusters_formed=0,
                gists_created=0,
                sources_superseded=0,
                gist_ids=[],
            )

        clusters = _cluster_by_cosine(
            [c["embedding"] for c in candidates],
            self._config.consolidation_cluster_threshold,
        )
        mergeable = [idxs for idxs in clusters if len(idxs) >= min_size]

        gists_created = 0
        sources_superseded = 0
        gist_ids: list[UUID] = []
        for idxs in mergeable:
            members = [candidates[i] for i in idxs]
            try:
                gist_id = await self._consolidate_cluster(
                    members,
                    agent_id=agent_id,
                    customer_id=customer_id,
                    user_id=user_id,
                )
            except Exception as exc:
                # a per-cluster failure (DB write, cycle-guard, dispatch)
                # is logged and skipped so the remaining clusters still
                # consolidate. A failure AFTER the gist commits but before
                # supersession leaves a committed-but-unreferenced gist;
                # that is non-destructive (sources are untouched and still
                # recallable) and self-correcting (a later pass re-clusters
                # + re-consolidates the sources), so no rollback is needed.
                log.warning(
                    "Dream consolidation skipped a cluster after an error: %s",
                    exc,
                    extra={"extra_data": {"agent_id": str(agent_id), "source_count": len(members)}},
                )
                continue
            if gist_id is not None:
                gists_created += 1
                sources_superseded += len(members)
                gist_ids.append(gist_id)

        return ConsolidationResult(
            memories_considered=considered,
            clusters_formed=len(mergeable),
            gists_created=gists_created,
            sources_superseded=sources_superseded,
            gist_ids=gist_ids,
        )

    async def _consolidate_cluster(
        self,
        members: list[dict[str, Any]],
        *,
        agent_id: UUID,
        customer_id: UUID | None,
        user_id: UUID | None,
    ) -> UUID | None:
        """synthesize + commit one gist for a cluster; return its id or None.

        Skips (returns ``None``, logs) on reflector failure, unparseable
        output, or a failed gist embedding so a single bad cluster does not
        abort the pass.

        :param members: the cluster's source memory dicts (>= min size)
        :ptype members: list[dict[str, Any]]
        :param agent_id: partition column
        :ptype agent_id: UUID
        :param customer_id: inherited customer grain (may be ``None``)
        :ptype customer_id: UUID | None
        :param user_id: inherited user grain (may be ``None``)
        :ptype user_id: UUID | None
        :return: the committed gist id, or ``None`` on skip
        :rtype: UUID | None
        """
        source_ids: list[UUID] = [m["memory_id"] for m in members]
        # inherit the conversation anchor from the newest source: memories
        # require a NOT NULL conversation_id (v019, unrelaxed) but a gist
        # spans several source conversations, so it borrows the most-recent
        # contributing one as a grouping key. The true N-source provenance
        # lives in the memory_consolidations edge table, and conversation_id
        # carries no FK, so a later delete of that conversation leaves the
        # gist as a harmless dangling ref rather than cascading it away.
        newest = max(members, key=lambda m: m["date_created"])
        conversation_id = newest["conversation_id"]
        type_memory = _modal_type([m["type_memory"] for m in members])

        gist = await self._reflect(members, user_id=user_id, conversation_id=conversation_id)
        if gist is None:
            return None
        gist_text, rationale = gist

        embedding = await _safe_aembed_query(self._embedding_provider, gist_text)
        if embedding is None:
            log.warning(
                "Dream consolidation skipped cluster: gist embedding failed",
                extra={"extra_data": {"agent_id": str(agent_id), "source_count": len(source_ids)}},
            )
            return None

        # UUID(str(...)) normalises uuid_utils.UUID -> stdlib uuid.UUID at the
        # mint border (the package's canonical uuid7 pattern), so the gist id
        # matches the stdlib-UUID typed collection surface and compares like
        # with like against asyncpg-read ids.
        gist_id = UUID(str(uuid7()))
        # defensive DAG guard (the gist is freshly minted so nothing can
        # reach it yet, but the guard also proves no source is itself a
        # descendant of a prior gist in the chain).
        await self._consolidations.assert_no_cycle(
            agent_id,
            consolidated_memory_id=gist_id,
            source_memory_ids=source_ids,
        )

        now = datetime.now(UTC)
        gist_entity: MemoryEntity = self._memories.create(
            {
                "memory_id": gist_id,
                "agent_id": agent_id,
                "customer_id": customer_id,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "type_memory": type_memory,
                "content": gist_text,
                "embedding": embedding,
                "salience": self._config.consolidation_gist_salience_seed,
                "date_created": now,
                "date_updated": now,
            }
        )
        await self._memories.save_entity(gist_entity)

        # record the provenance edges (gist committed first so the composite
        # FK targets exist), then supersede the sources.
        for source_id in source_ids:
            edge: MemoryConsolidationEntity = self._consolidations.create(
                {
                    "agent_id": agent_id,
                    "consolidated_memory_id": gist_id,
                    "source_memory_id": source_id,
                    "rationale": rationale,
                    "date_created": now,
                    "date_updated": now,
                }
            )
            await self._consolidations.save_entity(edge)

        await self._memories.mark_superseded(
            agent_id,
            source_memory_ids=source_ids,
            gist_id=gist_id,
        )

        await self._emit_consolidated_event(
            agent_id=agent_id,
            customer_id=customer_id,
            user_id=user_id,
            gist_id=gist_id,
            gist_text=gist_text,
            source_ids=source_ids,
        )
        return gist_id

    async def _reflect(
        self,
        members: list[dict[str, Any]],
        *,
        user_id: UUID | None,
        conversation_id: UUID,
    ) -> tuple[str, str | None] | None:
        """call the reflector to synthesize a gist + rationale for a cluster.

        :param members: the cluster's source memory dicts
        :ptype members: list[dict[str, Any]]
        :param user_id: owning user for gateway-model identity (may be None)
        :ptype user_id: UUID | None
        :param conversation_id: conversation for gateway-model attribution
        :ptype conversation_id: UUID
        :return: ``(gist_text, rationale)`` or ``None`` on any failure /
            empty gist (fail-safe: skip the cluster)
        :rtype: tuple[str, str | None] | None
        """
        sources_section = "\n".join(f"- [{m['type_memory']}] {m['content']}" for m in members)
        prompt = self._prompts.consolidation.format(sources_section=sources_section)
        try:
            model = await self._chat_model_factory.create_chat_model(
                purpose="consolidation",
            )
            response = await model.ainvoke(
                [
                    SystemMessage(
                        content="You consolidate related memories into a single dense memory. Return only valid JSON.",
                    ),
                    HumanMessage(content=prompt),
                ],
                **_identity_kwargs(user_id, conversation_id),
            )
            raw = response.content if hasattr(response, "content") else str(response)
            content = (raw if isinstance(raw, str) else str(raw)).strip()
            result = json.loads(_strip_code_block(content))
            gist_text = str(result.get("gist", "")).strip()
            if not gist_text:
                log.warning("Dream reflector returned an empty gist; skipping cluster")
                return None
            rationale_raw = result.get("rationale")
            rationale = str(rationale_raw).strip() if rationale_raw else None
            return gist_text, rationale
        except json.JSONDecodeError, KeyError:
            log.warning("Failed to parse Dream reflector response; skipping cluster")
            return None
        except Exception as exc:
            log.warning("Dream reflector LLM call failed; skipping cluster: %s", exc)
            return None

    async def _emit_consolidated_event(
        self,
        *,
        agent_id: UUID,
        customer_id: UUID | None,
        user_id: UUID | None,
        gist_id: UUID,
        gist_text: str,
        source_ids: list[UUID],
    ) -> None:
        """dispatch a :class:`MemoryConsolidatedEvent` for one merge.

        Best-effort: the gist + edges are already committed, so a missing
        stream event (no langgraph run manager in scope — cli / background
        job) is logged at debug and swallowed, never raised. ``str(None)``
        is avoided for a null scope grain — the event fields are typed
        ``str | None`` and receive ``None`` directly.

        :param agent_id: partition agent id
        :ptype agent_id: UUID
        :param customer_id: owning customer or ``None``
        :ptype customer_id: UUID | None
        :param user_id: owning user or ``None``
        :ptype user_id: UUID | None
        :param gist_id: the committed gist id
        :ptype gist_id: UUID
        :param gist_text: the gist content (for the preview)
        :ptype gist_text: str
        :param source_ids: the superseded source ids
        :ptype source_ids: list[UUID]
        :return: nothing
        :rtype: None
        """
        event = MemoryConsolidatedEvent(
            # convert at border: FrameworkEvent wire payload (str-typed fields)
            agent_id=str(agent_id),
            consolidated_memory_id=str(gist_id),  # convert at border: event wire payload
            source_memory_ids=[str(s) for s in source_ids],
            source_count=len(source_ids),
            user_id=str(user_id) if user_id is not None else None,  # convert at border: event wire payload
            customer_id=str(customer_id) if customer_id is not None else None,  # convert at border: event wire payload
            content_preview=gist_text[:120],
        )
        try:
            await dispatch_event(event)
        except RuntimeError as exc:
            # no run manager in scope (background job / cli / test harness):
            # the merge is committed; the stream event is a best-effort
            # surface. Narrowed to RuntimeError so a real dispatch bug (e.g.
            # a pydantic schema regression) still surfaces.
            log.debug(
                "MemoryConsolidatedEvent dropped (no run manager in scope; gist_id=%s, reason=%s)",
                event.consolidated_memory_id,
                exc,
            )


def _identity_kwargs(
    user_id: UUID | None,
    conversation_id: UUID | None,
) -> dict[str, Any]:
    """build the identity kwargs a gateway-routed chat model needs on invoke.

    Mirrors the extractor's helper: a ``GatewayChatModel`` resolves the
    invoking ``user_id`` + optional ``conversation_id`` from the invoke
    kwargs. ``None`` values are omitted so a stubbed / non-gateway model is
    unaffected (and an agent-scoped run with no user passes no ``user_id``).

    :param user_id: invoking user identifier
    :ptype user_id: UUID | None
    :param conversation_id: conversation identifier for usage attribution
    :ptype conversation_id: UUID | None
    :return: kwargs to splat into ``model.ainvoke``
    :rtype: dict[str, Any]
    """
    kwargs: dict[str, Any] = {}
    if user_id is not None:
        kwargs["user_id"] = user_id
    if conversation_id is not None:
        kwargs["conversation_id"] = conversation_id
    return kwargs
