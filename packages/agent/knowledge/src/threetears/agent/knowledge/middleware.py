"""3tears governed-knowledge injection middleware for ``langchain.agents.create_agent``.

The framework-aligned successor to the old ``knowledge_retrieval_node`` (a
hand-rolled graph node that ran before the agent node). Re-expressed onto
:meth:`langchain.agents.middleware.AgentMiddleware.awrap_model_call`: before each
model call, when a knowledge integration is injected on
``config["configurable"]["knowledge_integration"]`` and the verified per-call
identity is known (read off ``config["configurable"]["call_context"]``),
domain-routed CONCEPTS (a glossary) and playbook ENTRIES (procedures) are
retrieved, ranked, trimmed to a shared token budget, rendered, and FOLDED into
``request.system_message`` -- definitions before procedures (task-03 Note 5).

Why the fold and not an extra ``SystemMessage`` in the message list: ``create_agent``
assembles the model prompt as ``[request.system_message, *request.messages]`` with
NO consolidation, and LangChain's Anthropic binding then RAISES on multiple
non-consecutive system messages (``ValueError("Received multiple non-consecutive
system messages")``). Appending the governed block as a second ``SystemMessage``
(the pattern a ``before_model`` hook would use) triggers exactly that crash against
the real gateway. Folding the block into the single ``request.system_message`` --
the proven :class:`~threetears.langgraph.middleware_context.ContextMergeMiddleware`
pattern -- keeps one system message and sidesteps the failure.

The governed block + the shadow-disclosure ledgers are persisted onto the
``metadata`` state channel via an
:class:`~langchain.agents.middleware.types.ExtendedModelResponse` carrying a
:class:`~langgraph.types.Command` state update, so a downstream node can read the
governed-rule TEXT after the model node folds the injected prompt away. The
``metadata`` channel carries a :func:`threetears.langgraph.merge_metadata` reducer
(declared on :class:`KnowledgeInjectionState`) so this injector's keys compose
with the sibling memory / schema injectors rather than clobbering them; an update
to a channel the state schema does not declare would be silently dropped.

Identity comes from the VERIFIED ``call_context`` (the realign handler makes
``call_context`` the canonical verified identity), NOT from graph state -- the old
node read ``state["user_id"]`` / ``state["customer_id"]``, which the framework no
longer threads through state.

Every operation soft-fails exactly like the original node: a missing / unavailable
knowledge backend, an embedding outage, or a render fault logs a warning naming the
error class and proceeds on the un-merged request rather than failing the turn.

Async is the real path -- retrieval + embedding are ``async def``. The synchronous
:meth:`~KnowledgeInjectionMiddleware.wrap_model_call` mirror cannot drive them, so
it degrades to a pass-through (no injection this turn) and warns when an
integration was in fact configured so the misconfiguration is visible.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, NotRequired
from uuid import UUID

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import AgentState, ExtendedModelResponse
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.config import get_config
from langgraph.types import Command
from threetears.agent.memory.embedding_utils import embedding_attribution_scope
from threetears.knowledge import ConceptEffective, EntryEffective, Scope
from threetears.langgraph import merge_metadata
from threetears.observe import get_logger

from threetears.agent.knowledge.integration import (
    KnowledgeIntegration,
    retrieve_concepts,
    retrieve_entries,
)

__all__ = ["KnowledgeInjectionMiddleware", "KnowledgeInjectionState"]

log = get_logger(__name__)

#: Platform default for the situational-tail token budget (KNW-17 / note 4). The
#: budget governs ONLY the situational (non-invariant) tail, SHARED across both
#: concepts and entries (task-03 Note 5). Invariants inject in full regardless
#: (KNW-17 / KNW-25). Configured per-middleware via the constructor.
_DEFAULT_KNOWLEDGE_RETRIEVAL_TOKEN_BUDGET: int = 2000

#: Conventional characters-per-token heuristic for the deterministic token
#: estimate. The budget is a soft cost/latency guard, not a hard correctness
#: boundary, so a cheap deterministic estimator (no tokenizer dependency) is the
#: right tool -- and keeps the trim reproducible in tests.
_CHARS_PER_TOKEN: int = 4

_GLOSSARY_INVARIANT_HEADER = (
    "## Data concepts that ALWAYS apply\n"
    "These business terms have fixed governed definitions and bindings. "
    "Resolve every one of them to its definition before writing SQL.\n\n"
)

_GLOSSARY_SITUATIONAL_HEADER = (
    "## Data concepts relevant to this question\n"
    "Business terms retrieved as relevant to this question. Resolve every "
    "one that appears in your task to its governed definition + binding "
    "before writing SQL, and use that binding rather than a guess.\n\n"
)

_INVARIANT_HEADER = (
    "## Data knowledge that ALWAYS applies\n"
    "These are hard rules for the data in scope. Apply every one of "
    "them whenever you touch this data.\n\n"
)

_SITUATIONAL_HEADER = (
    "## Data knowledge relevant to this question\n"
    "Procedures retrieved as relevant to this question. Before you write "
    "or run a query, check every one against what you are about to do and "
    "apply each that touches the tables or columns in your query. If a "
    "procedure describes a trap, a required filter, or a correct source "
    "for a table you are querying, you MUST follow it -- these are "
    "governed rules, not optional background.\n\n"
)

#: Top-of-block lead-in prepended to the WHOLE injected knowledge block
#: (invariants + situational, concepts + entries). It states the binding
#: pre-flight contract ONCE for the entire block and -- per the "be explicit, no
#: room to fabricate" principle -- forbids inventing past the governed knowledge:
#: when nothing below covers part of the task, the agent must fall back to the
#: introspected schema and state its assumption rather than guess.
_BLOCK_PREAMBLE = (
    "# Governed data knowledge\n"
    "The rules below are binding. Apply every rule that touches your query "
    "before you run it. They are not background reading. "
    "If no rule covers part of your task, use the schema and state your "
    "assumption. Never invent a rule, a column, or a value.\n\n"
)

#: Stable scope ordering for rendering: platform first, then customer, then user
#: -- so user shadows visibly "speak last" (note 5).
_SCOPE_RENDER_RANK: dict[Scope, int] = {
    Scope.PLATFORM: 0,
    Scope.CUSTOMER: 1,
    Scope.USER: 2,
}


class KnowledgeInjectionState(AgentState[Any]):
    """Agent state extended with the ``metadata`` governed-knowledge channel.

    Extends the base :class:`~langchain.agents.middleware.types.AgentState` with a
    ``metadata`` dict channel so :class:`KnowledgeInjectionMiddleware` can stash the
    rendered governed-knowledge block + the shadow-disclosure ledgers for a
    downstream consumer. The channel carries the
    :func:`threetears.langgraph.merge_metadata` reducer so this injector's keys
    compose with the sibling memory / schema injectors (each writes disjoint
    top-level keys) rather than the last writer clobbering the others; LangGraph
    silently drops an update to a channel the state schema does not declare, so the
    channel must be declared here for the ledgers to persist.

    :ivar metadata: per-run metadata bag; the middleware writes the governed block
        under ``governed_knowledge_block`` and the shadow ledgers under
        ``knowledge_shadow_disclosures`` / ``knowledge_concept_shadow_disclosures``
        / ``knowledge_injected_concepts`` / ``knowledge_injected_entries``.
    """

    metadata: NotRequired[Annotated[dict[str, Any], merge_metadata]]


def _configurable() -> dict[str, Any]:
    """Return the active run's ``configurable`` dict, or empty when unavailable.

    Reads the :class:`~langchain_core.runnables.RunnableConfig` for the active run
    via :func:`langgraph.config.get_config` -- the canonical way a
    ``wrap_model_call`` hook reaches ``configurable`` (the ``Runtime`` on a
    ``ModelRequest`` does not carry ``config`` for the model-call seam). Outside a
    runnable context ``get_config`` raises ``RuntimeError``; with no config there is
    no injected integration, so this soft-fails to an empty dict and the seam
    becomes a pass-through rather than crashing.

    :return: the run's ``configurable`` mapping, or an empty dict.
    :rtype: dict[str, Any]
    """
    try:
        config = get_config()
    except RuntimeError:
        return {}
    return config.get("configurable") or {}


def _fold_governed_into_system(
    system_message: SystemMessage | None,
    block: str,
) -> SystemMessage:
    """Return a single ``SystemMessage`` carrying the base prompt plus the block.

    Preserves the base system prompt and appends the governed-knowledge block after
    a blank line. When the base message has structured (list) content -- e.g. a
    prior middleware already annotated it -- the block is appended as an extra text
    part so existing parts (and any ``cache_control`` on them) are left intact. When
    there is no base system message, the block becomes the whole system message.
    Mirrors :func:`threetears.langgraph.middleware_context._fold_context_into_system`.

    :param system_message: the request's current system message, or ``None``.
    :ptype system_message: SystemMessage | None
    :param block: the non-empty governed-knowledge block to fold in.
    :ptype block: str
    :return: the merged system message.
    :rtype: SystemMessage
    """
    if system_message is None:
        return SystemMessage(content=block)
    base_content = system_message.content
    if isinstance(base_content, list):
        merged_parts = [*base_content, {"type": "text", "text": block}]
        return system_message.model_copy(update={"content": merged_parts})
    base_text = base_content if isinstance(base_content, str) else str(base_content)
    return system_message.model_copy(update={"content": f"{base_text}\n\n{block}"})


class KnowledgeInjectionMiddleware(AgentMiddleware[KnowledgeInjectionState, Any, Any]):
    """Fold retrieved governed knowledge into the system prompt before the model.

    The ``create_agent`` successor to the ``knowledge_retrieval_node``. On every
    model call, when a knowledge integration is injected on
    ``config["configurable"]["knowledge_integration"]`` and the verified per-call
    identity is present on ``call_context``, domain-routed concepts + entries are
    retrieved, invariant/situational-split, shared-budget ranked + trimmed, rendered
    glossary-before-procedures, and FOLDED into ``request.system_message`` (yielding
    exactly one system message). The rendered block + shadow-disclosure ledgers are
    persisted onto ``metadata`` via a returned ``Command`` state update. Opt-in: no
    integration (or no verified identity) on ``config["configurable"]`` -> the seam
    is a pass-through. Every operation soft-fails to the un-merged request.
    """

    name = "KnowledgeInjectionMiddleware"

    state_schema: type[KnowledgeInjectionState] = KnowledgeInjectionState

    def __init__(self, *, token_budget: int = _DEFAULT_KNOWLEDGE_RETRIEVAL_TOKEN_BUDGET) -> None:
        """Configure the situational-tail token budget.

        :param token_budget: token budget for the shared situational (non-invariant)
            tail across concepts + entries; invariants inject in full regardless
            (KNW-17 / KNW-25 / task-03 Note 5).
        :ptype token_budget: int
        :return: nothing.
        :rtype: None
        """
        super().__init__()
        self.token_budget = token_budget

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Any,
    ) -> ModelResponse[Any] | ExtendedModelResponse[Any]:
        """Retrieve + render governed knowledge, fold it into the system prompt.

        Reads the knowledge integration + the verified ``call_context`` identity off
        ``config["configurable"]``. When both are present, retrieves the
        domain-routed concepts + entries, renders the governed block, folds it into
        ``request.system_message``, runs the call, and returns an
        :class:`ExtendedModelResponse` whose ``Command`` persists the block + shadow
        ledgers onto ``metadata``. Proceeds on the un-merged request (a plain
        response) when no integration/identity is configured, when nothing is
        retrieved, or when retrieval / render soft-fails.

        :param request: the model request (carries ``system_message`` + ``messages``
            + ``state``).
        :ptype request: ModelRequest
        :param handler: the next handler in the middleware chain.
        :ptype handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
        :return: the model response, wrapped with a metadata ``Command`` when a
            governed block was injected.
        :rtype: ModelResponse[Any] | ExtendedModelResponse[Any]
        """
        configurable = _configurable()
        integration = configurable.get("knowledge_integration")
        # identity comes from the VERIFIED call_context, never from graph state.
        # a missing integration or call_context means no injection this turn.
        call_context = configurable.get("call_context") if integration is not None else None
        block = ""
        metadata: dict[str, Any] = {}
        if integration is not None and call_context is not None:
            try:
                block, metadata = await _build_governed_injection(
                    integration,
                    user_id=call_context.user_id,
                    customer_id=call_context.customer_id,
                    datasource_id=configurable.get("knowledge_datasource_id"),
                    datasource_table_id=configurable.get("knowledge_datasource_table_id"),
                    scope_context=configurable.get("knowledge_scope_context"),
                    messages=request.messages,
                    budget=self.token_budget,
                )
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- knowledge injection is best-effort context enrichment; a fault proceeds on the un-merged request rather than failing the turn
                log.warning(
                    "knowledge injection middleware failed (soft-fail): %s",
                    type(exc).__name__,
                )
                block, metadata = "", {}
        if not block:
            passthrough: ModelResponse[Any] = await handler(request)
            return passthrough
        merged = _fold_governed_into_system(request.system_message, block)
        response: ModelResponse[Any] = await handler(request.override(system_message=merged))
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={"metadata": metadata}),
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Any,
    ) -> ModelResponse[Any]:
        """Sync mirror: cannot drive the async retrieval, so pass the call through.

        Retrieval + embedding are ``async def``; a synchronous ``wrap_model_call``
        path cannot await them. Rather than fail the run, injection is skipped this
        turn (the request passes through un-merged) and a warning is logged when an
        integration was in fact configured so the misconfiguration is visible.

        :param request: the model request (passed through unchanged).
        :ptype request: ModelRequest
        :param handler: the next handler in the middleware chain.
        :ptype handler: Callable[[ModelRequest], ModelResponse]
        :return: the model response from the un-merged request.
        :rtype: ModelResponse[Any]
        """
        if _configurable().get("knowledge_integration") is not None:
            log.warning(
                "knowledge injection skipped: an integration is configured but the "
                "agent ran on the synchronous model path; retrieval is async-only, so "
                "no governed knowledge is injected this turn",
            )
        result: ModelResponse[Any] = handler(request)
        return result


async def _build_governed_injection(
    integration: KnowledgeIntegration,
    *,
    user_id: UUID,
    customer_id: UUID,
    datasource_id: UUID | None,
    datasource_table_id: UUID | None,
    scope_context: str | None,
    messages: Sequence[BaseMessage],
    budget: int,
) -> tuple[str, dict[str, Any]]:
    """Retrieve, rank, trim, and render the governed-knowledge block + ledgers.

    The two retrieves are INDEPENDENT soft-fail calls: a concept backend fault
    returns ``([], [])`` and never breaks the entry injection or the turn. Splits
    invariants from situational across BOTH types, injects all invariants in full +
    the SHARED-budget-trimmed situational tail (concepts + entries ranked + trimmed
    together, similarity-ranked when the turn query embeds), renders the glossary
    (concepts) before the procedures (entries), and builds the shadow-disclosure
    ledgers. Returns ``("", {})`` when nothing resolves.

    :param integration: knowledge integration instance
    :ptype integration: KnowledgeIntegration
    :param user_id: caller (conversation end-user) UUID from the verified call
        context
    :ptype user_id: UUID
    :param customer_id: hub-authenticated customer from the verified call context,
        threaded per-call into every Class-B knowledge read
    :ptype customer_id: UUID
    :param datasource_id: optional datasource-anchor domain filter
    :ptype datasource_id: UUID | None
    :param datasource_table_id: optional binding-domain filter (concepts only)
    :ptype datasource_table_id: UUID | None
    :param scope_context: optional eval scope-context clamp (KNW-35)
    :ptype scope_context: str | None
    :param messages: the running message list (source of the turn query)
    :ptype messages: Sequence[BaseMessage]
    :param budget: shared situational-tail token budget
    :ptype budget: int
    :return: tuple ``(block, metadata)``; ``("", {})`` when nothing is injected
    :rtype: tuple[str, dict[str, Any]]
    """
    concept_effective, _concept_layered = await retrieve_concepts(
        integration,
        user_id,
        datasource_id=datasource_id,
        datasource_table_id=datasource_table_id,
        scope_context=scope_context,
        customer_scope=customer_id,
    )
    entry_effective, _entry_layered = await retrieve_entries(
        integration,
        user_id,
        datasource_id=datasource_id,
        scope_context=scope_context,
        customer_scope=customer_id,
    )

    if not (concept_effective or entry_effective):
        return "", {}

    concept_inv, concept_sit = _split_invariant_concepts(concept_effective)
    entry_inv, entry_sit = _split_invariant_entries(entry_effective)
    # D9: invariants inject in full, exempt from the trim. when the invariant set
    # ALONE exceeds the situational budget they still all inject (correctness beats
    # a soft budget) -- warn so the cost/latency overrun is observable, never silent.
    invariant_cost = _invariant_token_cost(concept_inv, entry_inv)
    if invariant_cost > budget:
        log.warning(
            "knowledge invariant set alone exceeds situational budget "
            "(invariant_tokens=%d budget=%d): injecting all invariants anyway "
            "(D9 correctness over soft budget)",
            invariant_cost,
            budget,
        )
    # KNW-92 / KNW-94: embed the turn query + fetch the situational candidates'
    # vectors so the shared trim can rank the SITUATIONAL pool by query<->item
    # cosine similarity. both steps soft-fail to empty (the stable-order branch).
    query_embedding = await _embed_turn_query(integration, messages=messages, user_id=user_id)
    situational_embeddings = await _fetch_situational_embeddings(
        integration,
        situational_concepts=concept_sit,
        situational_entries=entry_sit,
        query_embedding=query_embedding,
        customer_scope=customer_id,
    )
    # ONE shared budget across BOTH situational tails: rank + trim the unified
    # pool, then split kept items back by type for rendering (task-03 Note 5).
    kept_concept_sit, kept_entry_sit = _rank_and_trim_shared(
        situational_concepts=concept_sit,
        situational_entries=entry_sit,
        budget=budget,
        query_embedding=query_embedding,
        embeddings=situational_embeddings,
    )
    block = _render_block(
        invariant_concepts=concept_inv,
        situational_concepts=kept_concept_sit,
        invariant_entries=entry_inv,
        situational_entries=kept_entry_sit,
    )
    if not block:
        return "", {}

    injected_entries = entry_inv + kept_entry_sit
    injected_concepts = concept_inv + kept_concept_sit
    metadata: dict[str, Any] = {
        # stash the rendered block so a downstream node can read the governed-rule
        # TEXT after the model node folds the injected prompt away and the
        # adversarial review (which runs AFTER the agent) message-scans nothing;
        # metadata is the only channel that survives.
        "governed_knowledge_block": block,
        "knowledge_shadow_disclosures": _entry_shadow_disclosures(injected_entries),
        "knowledge_concept_shadow_disclosures": _concept_shadow_disclosures(injected_concepts),
        # the provenance footer renders injected-knowledge counts per scope; record
        # one scope record per injected item so the builder can fold them.
        "knowledge_injected_concepts": [{"scope": c.concept.scope.value} for c in injected_concepts],
        "knowledge_injected_entries": [{"scope": e.entry.scope.value} for e in injected_entries],
    }
    return block, metadata


def _split_invariant_entries(
    effective: list[EntryEffective],
) -> tuple[list[EntryEffective], list[EntryEffective]]:
    """Partition effective entry views into invariants + situational (KNW-17).

    Invariants are the entries whose nearest-scope winner carries
    ``always_inject == True``; everything else is situational. Both lists preserve
    the merge's stable input order so downstream rendering is deterministic.

    :param effective: merged effective entry views from the shared merge
    :ptype effective: list[EntryEffective]
    :return: tuple ``(invariants, situational)``
    :rtype: tuple[list[EntryEffective], list[EntryEffective]]
    """
    invariants = [e for e in effective if e.always_inject]
    situational = [e for e in effective if not e.always_inject]
    return invariants, situational


def _split_invariant_concepts(
    effective: list[ConceptEffective],
) -> tuple[list[ConceptEffective], list[ConceptEffective]]:
    """Partition effective concept views into invariants + situational (KNW-25).

    Invariants are the concepts whose nearest-scope winner carries
    ``always_inject == True``; everything else is situational. Both lists preserve
    the merge's stable input order so downstream rendering is deterministic.

    :param effective: merged effective concept views from the shared merge
    :ptype effective: list[ConceptEffective]
    :return: tuple ``(invariants, situational)``
    :rtype: tuple[list[ConceptEffective], list[ConceptEffective]]
    """
    invariants = [c for c in effective if c.always_inject]
    situational = [c for c in effective if not c.always_inject]
    return invariants, situational


def _invariant_token_cost(
    invariant_concepts: list[ConceptEffective],
    invariant_entries: list[EntryEffective],
) -> int:
    """Sum rendered-token cost of the invariant set for the D9 overflow warning.

    Invariants inject in full regardless of budget (D9); this measures their
    combined rendered cost so the middleware can warn when that cost alone exceeds
    the situational budget -- an observable cost/latency overrun, never a silent
    trim. Costs each invariant by the EXACT rendered text that will be injected,
    matching the situational trim's accounting.

    :param invariant_concepts: always-inject effective concept views
    :ptype invariant_concepts: list[ConceptEffective]
    :param invariant_entries: always-inject effective entry views
    :ptype invariant_entries: list[EntryEffective]
    :return: combined rendered-token cost of all invariants
    :rtype: int
    """
    concept_cost = sum(_estimate_tokens(_render_concept(concept)) for concept in invariant_concepts)
    entry_cost = sum(_estimate_tokens(_render_entry(entry)) for entry in invariant_entries)
    return concept_cost + entry_cost


@dataclass(frozen=True)
class _RankedSituational:
    """One situational item (concept OR entry) in the shared trim pool.

    The situational tail of BOTH types shares ONE budget (task-03 Note 5), so the
    trim ranks a UNIFIED pool of concepts + entries and keeps a deterministic
    prefix. Each item carries its rendered text (so the trim costs the EXACT bytes
    that will be injected), its scope + id for the stable order, and a
    discriminator so kept items can be split back by type for the
    glossary-before-procedures rendering.

    :ivar scope: the item's effective scope (stable-order primary key)
    :ivar id_bytes: the item's id bytes (stable-order tie-break)
    :ivar is_concept: ``True`` for a concept (glossary), ``False`` for an entry
    :ivar concept: the concept effective view, or ``None`` for an entry
    :ivar entry: the entry effective view, or ``None`` for a concept
    :ivar rendered: the item's rendered text (the trim cost basis)
    """

    scope: Scope
    id_bytes: bytes
    is_concept: bool
    concept: ConceptEffective | None
    entry: EntryEffective | None
    rendered: str


def _rank_and_trim_shared(
    *,
    situational_concepts: list[ConceptEffective],
    situational_entries: list[EntryEffective],
    budget: int,
    query_embedding: list[float] | None = None,
    embeddings: dict[UUID, list[float]] | None = None,
) -> tuple[list[ConceptEffective], list[EntryEffective]]:
    """Rank the SHARED situational tail (concepts + entries) and trim to budget.

    The situational tail of BOTH types shares ONE token budget (task-03 Note 5):
    situational concepts and situational entries are pooled, ranked together, and
    trimmed greedily to the SINGLE budget. Each item is costed by the EXACT bytes
    that will be injected (its rendered text). Kept items are split back by type so
    the renderer can place concepts in the glossary and entries in the procedures
    section while honoring one shared budget across both.

    Rank key (KNW-92 / KNW-94):

    - when ``query_embedding`` is a non-empty vector AND an item carries a stored
      embedding, the item is ranked by query<->item COSINE similarity DESC. Cosine
      (not raw dot) because embeddings are not guaranteed normalized.
    - an item whose embedding is NULL (id absent from ``embeddings``) FALLS BACK to
      the stable scope+id order and is appended BEHIND every embedded item.
    - when ``query_embedding`` is empty / ``None``, EVERY item takes the
      stable-order branch, making the result byte-identical to the pre-similarity
      stable order (an embedding outage degrades ranking quality, never the turn).

    The concept-leads-entry tiebreak is preserved at EQUAL similarity (and in the
    stable-order fallback) so the glossary still leads definitions ahead of the
    procedures that depend on them. Invariants are split out BEFORE this call (D9)
    and never enter the pool.

    :param situational_concepts: non-invariant effective concept views
    :ptype situational_concepts: list[ConceptEffective]
    :param situational_entries: non-invariant effective entry views
    :ptype situational_entries: list[EntryEffective]
    :param budget: shared situational-tail token budget
    :ptype budget: int
    :param query_embedding: the turn-query vector; empty / ``None`` takes the
        stable-order branch for every item (soft-fail contract)
    :ptype query_embedding: list[float] | None
    :param embeddings: map of situational item id to its stored embedding; an absent
        id is a NULL-embedding item (stable-order fallback)
    :ptype embeddings: dict[UUID, list[float]] | None
    :return: tuple ``(kept_concepts, kept_entries)`` -- each list in the RANKED
        order the trim survived; the renderer injects situational items in THIS
        order and must not re-sort these lists
    :rtype: tuple[list[ConceptEffective], list[EntryEffective]]
    """
    embeddings = embeddings or {}
    similarity_active = bool(query_embedding)
    pool: list[_RankedSituational] = []
    for concept in situational_concepts:
        pool.append(
            _RankedSituational(
                scope=concept.concept.scope,
                id_bytes=concept.concept.id.bytes,
                is_concept=True,
                concept=concept,
                entry=None,
                rendered=_render_concept(concept),
            ),
        )
    for entry in situational_entries:
        pool.append(
            _RankedSituational(
                scope=entry.entry.scope,
                id_bytes=entry.entry.id.bytes,
                is_concept=False,
                concept=None,
                entry=entry,
                rendered=_render_entry(entry),
            ),
        )

    def _item_id(item: _RankedSituational) -> UUID:
        """Resolve the effective-view id of one pooled situational item.

        :param item: pooled situational item (concept or entry)
        :ptype item: _RankedSituational
        :return: the item's id (concept id or entry id)
        :rtype: UUID
        """
        if item.is_concept and item.concept is not None:
            result: UUID = item.concept.concept.id
        else:
            assert item.entry is not None  # pool invariant: a non-concept item carries an entry
            result = item.entry.entry.id
        return result

    def _rank_key(item: _RankedSituational) -> tuple[int, float]:
        """Build the similarity-group sort key (the PRIMARY sort component).

        Sort is ASCENDING, so the key groups embedded items (rank 0) ahead of
        NULL-embedding items (rank 1) and, WITHIN the embedded group, orders by
        cosine DESC (the key negates similarity so higher cosine sorts first). It
        carries NOTHING else: ALL deterministic tiebreaking is delegated to
        :func:`_stable_key`, the SECONDARY sort component, so an all-NULL active
        turn is byte-identical to the pre-similarity stable order.

        :param item: pooled situational item
        :ptype item: _RankedSituational
        :return: ``(embedded_rank, -similarity)`` -- tiebreak delegated to
            :func:`_stable_key`
        :rtype: tuple[int, float]
        """
        vector = embeddings.get(_item_id(item)) if similarity_active else None
        if vector is not None:
            sim = _cosine_similarity(query_embedding or [], vector)
            embedded_rank = 0
            neg_sim = -sim
        else:
            embedded_rank = 1
            neg_sim = 0.0
        return (embedded_rank, neg_sim)

    def _stable_key(item: _RankedSituational) -> tuple[int, int, bytes]:
        """The deterministic stable scope+id key (the pre-similarity order).

        Used directly when similarity is inactive (byte-identical to the former
        behaviour) and as the scope-aware tiebreak component when similarity is
        active.

        :param item: pooled situational item
        :ptype item: _RankedSituational
        :return: ``(scope_rank, concept_first, id_bytes)``
        :rtype: tuple[int, int, bytes]
        """
        return (
            _SCOPE_RENDER_RANK[item.scope],
            0 if item.is_concept else 1,
            item.id_bytes,
        )

    if similarity_active:
        ranked = sorted(
            pool,
            key=lambda item: (_rank_key(item), _stable_key(item)),
        )
    else:
        ranked = sorted(pool, key=_stable_key)

    kept_concepts: list[ConceptEffective] = []
    kept_entries: list[EntryEffective] = []
    spent = 0
    for item in ranked:
        cost = _estimate_tokens(item.rendered)
        if spent + cost > budget:
            break
        if item.is_concept and item.concept is not None:
            kept_concepts.append(item.concept)
        elif item.entry is not None:
            kept_entries.append(item.entry)
        spent += cost
    return kept_concepts, kept_entries


def _stable_order_entries(
    views: list[EntryEffective],
) -> list[EntryEffective]:
    """Order effective entry views by scope (platform, customer, user) then id.

    The SAME ordering the rendering uses, so a deterministic prefix holds.
    Platform-scope entries rank first so the most-governed knowledge is rendered
    ahead of user-scope overrides.

    :param views: effective entry views to order
    :ptype views: list[EntryEffective]
    :return: views in stable scope+id order
    :rtype: list[EntryEffective]
    """
    return sorted(
        views,
        key=lambda v: (_SCOPE_RENDER_RANK[v.entry.scope], v.entry.id.bytes),
    )


def _stable_order_concepts(
    views: list[ConceptEffective],
) -> list[ConceptEffective]:
    """Order effective concept views by scope (platform, customer, user), id.

    The SAME ordering the rendering uses. Platform-scope concepts rank first so the
    most-governed definitions render ahead of user-scope overrides.

    :param views: effective concept views to order
    :ptype views: list[ConceptEffective]
    :return: views in stable scope+id order
    :rtype: list[ConceptEffective]
    """
    return sorted(
        views,
        key=lambda v: (_SCOPE_RENDER_RANK[v.concept.scope], v.concept.id.bytes),
    )


def _render_block(
    *,
    invariant_concepts: list[ConceptEffective],
    situational_concepts: list[ConceptEffective],
    invariant_entries: list[EntryEffective],
    situational_entries: list[EntryEffective],
) -> str:
    """Render the injected block: GLOSSARY (concepts) before PROCEDURES (entries).

    Definitions before procedures (task-03 Note 5): the concept glossary sections
    render FIRST (invariant glossary, then situational glossary), then the entry
    procedure sections (invariant procedures, then situational procedures).

    ORDER within each subsection differs by tier (KNW-92 / D9): INVARIANT
    subsections render in the deterministic stable scope ordering; SITUATIONAL
    subsections render in the ORDER GIVEN -- the similarity-ranked order
    :func:`_rank_and_trim_shared` already produced. The renderer MUST NOT re-sort
    the situational lists. Returns an empty string when every list is empty.

    :param invariant_concepts: invariant effective concept views
    :ptype invariant_concepts: list[ConceptEffective]
    :param situational_concepts: kept situational effective concept views, ALREADY
        in similarity-ranked order
    :ptype situational_concepts: list[ConceptEffective]
    :param invariant_entries: invariant effective entry views
    :ptype invariant_entries: list[EntryEffective]
    :param situational_entries: kept situational effective entry views, ALREADY in
        similarity-ranked order
    :ptype situational_entries: list[EntryEffective]
    :return: rendered context block, or empty string
    :rtype: str
    """
    sections: list[str] = []
    # glossary (concepts) FIRST -- definitions before procedures.
    if invariant_concepts:
        rendered = "\n\n".join(_render_concept(v) for v in _stable_order_concepts(invariant_concepts))
        sections.append(_GLOSSARY_INVARIANT_HEADER + rendered)
    if situational_concepts:
        # ranked order preserved -- do NOT _stable_order_* the situational kept list
        # (KNW-92): _rank_and_trim_shared already ordered it by similarity DESC.
        rendered = "\n\n".join(_render_concept(v) for v in situational_concepts)
        sections.append(_GLOSSARY_SITUATIONAL_HEADER + rendered)
    # procedures (entries) AFTER the glossary.
    if invariant_entries:
        rendered = "\n\n".join(_render_entry(v) for v in _stable_order_entries(invariant_entries))
        sections.append(_INVARIANT_HEADER + rendered)
    if situational_entries:
        rendered = "\n\n".join(_render_entry(v) for v in situational_entries)
        sections.append(_SITUATIONAL_HEADER + rendered)
    if not sections:
        result = ""
    else:
        # prepend the block-level binding contract ONCE ahead of every section.
        result = _BLOCK_PREAMBLE + "\n\n".join(sections)
    return result


def _render_concept(view: ConceptEffective) -> str:
    """Render one effective concept as a glossary entry with provenance.

    The definition + binding are emitted VERBATIM (whole-concept, never spliced --
    D4). The concept is labeled with its scope provenance; when it shadows an
    ancestor a disclosure line names the shadowed scope (D5); when it is
    ``ambiguous`` a disclosure line discloses the ambiguity rather than silently
    picking one definition (Note 2). The binding fields
    (``datasource_table_ref`` / ``sql_fragment``) and caveats ride through as
    curated context -- ``sql_fragment`` is NEVER executed (datasource tool boundary
    owns SQL safety).

    :param view: effective concept view to render
    :ptype view: ConceptEffective
    :return: rendered concept text
    :rtype: str
    """
    concept = view.concept
    lines = [f"### {concept.name} ({concept.scope.value}-scope definition)"]
    if concept.aliases:
        lines.append(f"Also known as: {', '.join(concept.aliases)}")
    lines.append(concept.definition)
    # emit the resolved schema.table NAME, NEVER the raw datasource_table_id UUID:
    # the agent addresses tables by schema.table and has no tool to resolve a table
    # UUID, so a raw-UUID binding is un-actionable governance.
    if concept.datasource_table_ref is not None:
        lines.append(f"Bound table: {concept.datasource_table_ref}")
    elif concept.datasource_table_id is not None:
        lines.append("Bound table: (unresolved -- bound table no longer exists)")
    if concept.sql_fragment:
        lines.append(f"Filter fragment: {concept.sql_fragment}")
    if concept.caveats:
        lines.append(f"Caveats: {concept.caveats}")
    if view.shadows_scope is not None:
        lines.append(f"(overrides the {view.shadows_scope.value}-scope definition)")
    if view.ambiguous:
        lines.append(
            "(AMBIGUOUS: another in-scope definition uses this same term -- "
            "disclose the competing definitions; do not silently pick one)",
        )
    return "\n".join(lines)


def _render_entry(view: EntryEffective) -> str:
    """Render one effective entry as a titled bullet with shadow disclosure.

    The body is emitted VERBATIM (whole-entry, never spliced -- D4). When the entry
    shadows an ancestor, a disclosure line names the shadowed scope (D5) so the
    model -- and the provenance footer -- see the override.

    :param view: effective view to render
    :ptype view: EntryEffective
    :return: rendered entry text
    :rtype: str
    """
    entry = view.entry
    lines = [f"### {entry.title}", entry.body]
    if view.shadows_scope is not None:
        lines.append(f"(overrides the {view.shadows_scope.value}-scope guidance)")
    return "\n".join(lines)


def _entry_shadow_disclosures(
    views: list[EntryEffective],
) -> list[dict[str, str]]:
    """Build the entry shadow-disclosure ledger the provenance footer consumes (D5).

    One entry per injected view that shadows an ancestor: the entry id, its TITLE
    (so the footer renders the fixed disclosure wording without a second lookup),
    and the shadowed scope. Views that shadow nothing are omitted. The list is
    JSON-serializable (string values) so it rides the state metadata dict cleanly.

    :param views: injected effective entry views (invariants + kept situational)
    :ptype views: list[EntryEffective]
    :return: list of ``{"entry_id", "title", "shadows_scope"}`` dicts
    :rtype: list[dict[str, str]]
    """
    disclosures: list[dict[str, str]] = []
    for view in views:
        if view.shadows_scope is not None:
            disclosures.append(
                {
                    "entry_id": str(view.entry.id),
                    "title": view.entry.title,
                    "shadows_scope": view.shadows_scope.value,
                }
            )
    return disclosures


def _concept_shadow_disclosures(
    views: list[ConceptEffective],
) -> list[dict[str, str]]:
    """Build the concept shadow / ambiguity ledger the provenance footer consumes (D5).

    One entry per injected concept view that shadows an ancestor OR is ambiguous:
    the concept id, its NAME (so the footer renders the fixed disclosure wording
    without a second lookup), the shadowed scope (empty string when the disclosure
    is ambiguity-only), and the ambiguity flag. Views that neither shadow nor
    diverge are omitted. The list is JSON-serializable so it rides the state
    metadata dict cleanly alongside the entry disclosures.

    :param views: injected effective concept views (invariants + kept situational)
    :ptype views: list[ConceptEffective]
    :return: list of ``{"concept_id", "name", "shadows_scope", "ambiguous"}`` dicts
    :rtype: list[dict[str, str]]
    """
    disclosures: list[dict[str, str]] = []
    for view in views:
        if view.shadows_scope is not None or view.ambiguous:
            disclosures.append(
                {
                    "concept_id": str(view.concept.id),
                    "name": view.concept.name,
                    "shadows_scope": (view.shadows_scope.value if view.shadows_scope is not None else ""),
                    "ambiguous": "true" if view.ambiguous else "false",
                }
            )
    return disclosures


def _estimate_tokens(text: str) -> int:
    """Estimate token count via the conventional chars-per-token heuristic.

    The budget is a soft cost/latency guard, so a cheap deterministic estimator (no
    tokenizer dependency) is the right tool and keeps the trim reproducible in
    tests.

    :param text: text to estimate
    :ptype text: str
    :return: estimated token count (>= 1 for non-empty text)
    :rtype: int
    """
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _turn_query_text(messages: Sequence[BaseMessage]) -> str:
    """Extract the user's turn-query text for the situational-similarity match.

    The query side embeds the SAME shape the back-fill matches against: the user's
    turn MESSAGE (the situational question). The most-recent
    :class:`~langchain_core.messages.HumanMessage` in the running message list is
    the turn query; its string content is returned (a non-string content payload --
    a multimodal message -- yields an empty string so the ranker soft-fails to
    stable-order rather than embedding garbage).

    :param messages: the running message list
    :ptype messages: Sequence[BaseMessage]
    :return: the latest human turn-query text, or empty string when none
    :rtype: str
    """
    result = ""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            result = content if isinstance(content, str) else ""
            break
    return result


async def _embed_turn_query(
    integration: KnowledgeIntegration,
    *,
    messages: Sequence[BaseMessage],
    user_id: UUID,
) -> list[float]:
    """Embed the turn query via the integration's model with the memory soft-fail.

    Reuses the SAME embedding model the memory system uses (the integration's
    ``embedding_model``): no new transport, no new model (D7). Returns an EMPTY list
    -- the stable-order branch -- when the model is unwired, the query text is empty,
    or the embedding call fails / soft-fails to ``[]`` (KNW-94). An embedding outage
    degrades ranking quality, never the turn; the error class is named in a warning,
    never swallowed silently.

    :param integration: knowledge integration carrying the embedding model
    :ptype integration: KnowledgeIntegration
    :param messages: the running message list (source of the turn query text)
    :ptype messages: Sequence[BaseMessage]
    :param user_id: the verified turn user, bound as the usage-attribution principal
        for this embedding
    :ptype user_id: UUID
    :return: the query embedding vector, or ``[]`` on any soft-fail
    :rtype: list[float]
    """
    result: list[float] = []
    model = integration.embedding_model
    query_text = _turn_query_text(messages)
    if model is not None and query_text:
        try:
            # bind the turn's user as the USAGE-attribution principal for this
            # embedding; the authorization principal stays the agent inside the
            # embedder. the scope wraps the embed await directly so the contextvar
            # is read within the same coroutine.
            with embedding_attribution_scope(user_id):
                vector = await model.aembed_query(query_text)
            result = vector if vector else []
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- query embedding is best-effort ranking input; a fault soft-fails to stable order rather than failing the turn
            log.warning(
                "knowledge situational query embedding failed (soft-fail to stable order): %s: %s",
                type(exc).__name__,
                exc,
            )
            result = []
    return result


async def _fetch_situational_embeddings(
    integration: KnowledgeIntegration,
    *,
    situational_concepts: list[ConceptEffective],
    situational_entries: list[EntryEffective],
    query_embedding: list[float],
    customer_scope: UUID,
) -> dict[UUID, list[float]]:
    """Fetch the situational candidates' stored vectors over the proxy (KNW-95).

    A SEPARATE, bounded read for ONLY the situational candidates about to be ranked
    -- the hot ``list_visible_to_user`` projection stays vector-free; this pulls
    vectors for the small to-be-ranked set via the collections' ``fetch_embeddings``
    (the ``embedding::text`` cast path). Skipped entirely (empty map) when
    ``query_embedding`` is empty. Each side soft-fails INDEPENDENTLY to an empty
    contribution so a concept-vector fault never blocks the entry ranking or the
    turn; a missing vector simply falls the item back to stable-order (KNW-92).

    :param integration: knowledge integration carrying the collections
    :ptype integration: KnowledgeIntegration
    :param situational_concepts: situational concept views to fetch for
    :ptype situational_concepts: list[ConceptEffective]
    :param situational_entries: situational entry views to fetch for
    :ptype situational_entries: list[EntryEffective]
    :param query_embedding: the turn-query vector; empty short-circuits to an empty
        map (stable order)
    :ptype query_embedding: list[float]
    :param customer_scope: conversation customer the broker clamps the Class-B
        vector reads to (broker-isolation-task-01)
    :ptype customer_scope: UUID
    :return: map of situational item id to its stored embedding vector
    :rtype: dict[UUID, list[float]]
    """
    result: dict[UUID, list[float]] = {}
    if not query_embedding:
        return result
    entry_collection = integration.entry_collection
    if entry_collection is not None and situational_entries:
        entry_ids = [e.entry.id for e in situational_entries]
        try:
            result.update(await entry_collection.fetch_embeddings(entry_ids, customer_scope=customer_scope))
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- vector fetch is best-effort ranking input; a fault falls the items back to stable order rather than failing the turn
            log.warning(
                "knowledge situational entry-vector fetch failed (soft-fail to stable order): %s: %s",
                type(exc).__name__,
                exc,
            )
    concept_collection = integration.concept_collection
    if concept_collection is not None and situational_concepts:
        concept_ids = [c.concept.id for c in situational_concepts]
        try:
            result.update(await concept_collection.fetch_embeddings(concept_ids, customer_scope=customer_scope))
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- vector fetch is best-effort ranking input; a fault falls the items back to stable order rather than failing the turn
            log.warning(
                "knowledge situational concept-vector fetch failed (soft-fail to stable order): %s: %s",
                type(exc).__name__,
                exc,
            )
    return result


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (KNW-92, cosine not raw dot).

    Embeddings are not guaranteed normalized, so the dot product is divided by the
    product of the L2 norms. Returns ``0.0`` when either vector is empty or a zero
    vector (no usable direction) or when the dimensions disagree (a defensive guard
    -- a dimension mismatch is a back-fill / model-config bug, and ``0.0`` degrades
    that item to the bottom of the embedded group rather than crashing the rank).
    The range is ``[-1.0, 1.0]``; higher is more similar.

    :param a: first vector (typically the query embedding)
    :ptype a: list[float]
    :param b: second vector (typically a stored item embedding)
    :ptype b: list[float]
    :return: cosine similarity in ``[-1.0, 1.0]``, or ``0.0`` on a degenerate input
    :rtype: float
    """
    if not a or not b or len(a) != len(b):
        result = 0.0
    else:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            result = 0.0
        else:
            result = dot / (norm_a * norm_b)
    return result
