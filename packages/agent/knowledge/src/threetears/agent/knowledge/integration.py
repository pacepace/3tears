"""Knowledge integration for agent runtime -- playbook entries + concepts.

Wires the governed-knowledge read layer into the agent runtime, mirroring
:mod:`threetears.agent.memory.integration`. The integration holds the agent-side
:class:`~threetears.agent.knowledge.collections.PlaybookEntryCollection` +
:class:`~threetears.agent.knowledge.collections.ConceptCollection` (both
constructed over the ``system.platform.rbac`` NATS proxy pool by the host
bootstrap) plus the embedding model the situational-tail similarity ranker uses,
and exposes :func:`retrieve_entries` / :func:`retrieve_concepts` -- the per-call
helpers the :class:`~threetears.agent.knowledge.middleware.KnowledgeInjectionMiddleware`
delegates to.

The merge that resolves the three-scope shadow chains is the SHARED authority
:func:`threetears.knowledge.merge_entry_views` /
:func:`threetears.knowledge.merge_concept_views`: the hub eval fingerprint builder
and this module import the SAME functions so a fingerprint and a live turn agree
byte-for-byte on the effective view. This module NEVER reimplements the merge.

Soft-fail contract (mirrors memory): a missing / unavailable knowledge backend
yields an EMPTY result + a warning log naming the error class. The turn is never
crashed and the exception is never swallowed silently.

Per-conversation isolation: this module holds NO module-level mutable state. The
integration is one per pod; every read is a fresh per-call proxy query keyed on
the caller's identity + domain filters. Caching the effective view at
construction time would break the per-conversation isolation the multiplexing
design mandates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from threetears.knowledge import (
    ConceptEffective,
    ConceptLayered,
    EntryEffective,
    EntryLayered,
    KnowledgeDraftCommand,
    KnowledgeDraftEvent,
    KnowledgeDraftMessage,
    Scope,
    merge_concept_views,
    merge_entry_views,
)
from threetears.nats import Subjects
from threetears.observe import get_logger

__all__ = [
    "DraftView",
    "KnowledgeIntegration",
    "retrieve_concepts",
    "retrieve_entries",
    "scope_context_admitted_scopes",
]

log = get_logger(__name__)


#: The three scope-context values (mirrors the hub's ``eval_runs.scope_context``
#: + ``eval_fingerprint``; the vocabulary is shared, the agent cannot import the
#: hub module).
SCOPE_CONTEXT_BASELINE = "baseline"
SCOPE_CONTEXT_CUSTOMER_EFFECTIVE = "customer_effective"
SCOPE_CONTEXT_USER_EFFECTIVE = "user_effective"


@dataclass(frozen=True)
class DraftView:
    """One correction-harvest draft surfaced to its author (KNW-54).

    The lightweight projection the ``knowledge_drafts`` management tool renders:
    which entity the draft is (``target``), its id, the author-facing text
    (``title`` / ``body``), and the provenance the author can use to recall the
    turn it came from. Drafts are NEVER part of retrieval -- this view exists only
    for the management surface.

    :ivar draft_id: the draft row id
    :ivar target: ``'entry'`` or ``'concept'``
    :ivar title: entry title / concept name
    :ivar body: entry body / concept definition
    :ivar related_domain: the anchoring datasource id (the routing anchor;
        rendered for the author's context)
    :ivar conversation_id: conversation the correction occurred in
    :ivar turn_count: turn number within that conversation
    """

    draft_id: UUID
    target: str
    title: str
    body: str
    related_domain: str
    conversation_id: UUID | None
    turn_count: int | None


def scope_context_admitted_scopes(scope_context: str) -> frozenset[Scope]:
    """Resolve which ladder scopes an eval scope context admits (KNW-35).

    - ``baseline`` -> platform only
    - ``customer_effective`` -> platform + customer
    - ``user_effective`` -> platform + customer + user

    This is the agent mirror of the hub's ``scope_context_admitted_scopes`` so a
    live eval turn clamps its effective view to exactly the scopes the run's
    fingerprint recorded. The SCOPE-MEMBERSHIP filter is trivial and duplicated
    across the repo boundary (the agent cannot import the hub); the MERGE that
    resolves shadow chains stays the single shared authority
    (:func:`threetears.knowledge.merge_entry_views`).

    :param scope_context: one of the three scope-context values
    :ptype scope_context: str
    :return: the admitted ladder scopes
    :rtype: frozenset[Scope]
    :raises ValueError: when ``scope_context`` is not a known value
    """
    if scope_context == SCOPE_CONTEXT_BASELINE:
        result = frozenset({Scope.PLATFORM})
    elif scope_context == SCOPE_CONTEXT_CUSTOMER_EFFECTIVE:
        result = frozenset({Scope.PLATFORM, Scope.CUSTOMER})
    elif scope_context == SCOPE_CONTEXT_USER_EFFECTIVE:
        result = frozenset({Scope.PLATFORM, Scope.CUSTOMER, Scope.USER})
    else:
        raise ValueError(f"unknown scope_context {scope_context!r}")
    return result


class KnowledgeIntegration:
    """Wires the playbook-entry + concept knowledge layer for agent use.

    Holds the agent-side
    :class:`~threetears.agent.knowledge.collections.PlaybookEntryCollection` and
    :class:`~threetears.agent.knowledge.collections.ConceptCollection` (both
    constructed over the ``system.platform.rbac`` proxy pool by the bootstrap) and
    the embedding model the situational-tail ranker uses. Downstream the
    :class:`~threetears.agent.knowledge.middleware.KnowledgeInjectionMiddleware`
    consumes the collections through :func:`retrieve_entries` /
    :func:`retrieve_concepts`; the middleware owns the shared invariant split, the
    shared budget trim, and the glossary-before-procedures rendering.

    The integration is one instance per pod and carries NO per-conversation state
    -- every retrieval is a fresh per-call proxy query (per-conversation isolation
    lives at the call boundary, not on this object).

    :ivar entry_collection: agent-side ``PlaybookEntryCollection`` over the proxy;
        ``None`` disables entry retrieval (soft-fail to empty)
    :ivar embedding_model: optional embedding model the situational ranker
        queries; ``None`` disables similarity ranking and falls back to a stable
        deterministic order
    :ivar concept_collection: agent-side ``ConceptCollection`` over the proxy;
        ``None`` disables concept injection (soft-fail to empty, entries
        unaffected)
    :ivar nats_client: connected NATS client used to PUBLISH draft events /
        commands on ``{ns}.knowledge.draft`` (the agent cannot write ``platform.*``
        directly -- the hub-side emitter materializes the row,
        knowledge-task-06); ``None`` disables the write path (the harvester
        soft-fails to a no-op)
    """

    def __init__(
        self,
        entry_collection: Any = None,
        embedding_model: Any = None,
        concept_collection: Any = None,
        nats_client: Any = None,
    ) -> None:
        """Initialize knowledge integration with optional components.

        :param entry_collection: agent-side ``PlaybookEntryCollection`` over the
            proxy; ``None`` disables entry retrieval (soft-fail to empty)
        :ptype entry_collection: Any
        :param embedding_model: optional embedding model for situational-tail
            similarity ranking; ``None`` falls back to a stable deterministic order
        :ptype embedding_model: Any
        :param concept_collection: agent-side ``ConceptCollection`` over the proxy;
            ``None`` disables concept retrieval (soft-fail to empty; entries
            unaffected)
        :ptype concept_collection: Any
        :param nats_client: connected NATS client for publishing draft events /
            commands; ``None`` disables the write path
        :ptype nats_client: Any
        :return: nothing
        :rtype: None
        """
        self.entry_collection = entry_collection
        self.embedding_model = embedding_model
        self.concept_collection = concept_collection
        self.nats_client = nats_client

    async def publish_draft(self, event: KnowledgeDraftEvent) -> None:
        """Publish a NEW correction-harvest draft to the hub (KNW-51).

        The agent cannot write ``platform.*`` (the broker read-only carve-out
        admits only SELECT), so a detected correction travels as a
        :class:`threetears.knowledge.KnowledgeDraftEvent` on
        ``{ns}.knowledge.draft``; the hub-side ``KnowledgeDraftEmitter``
        materializes the ``status='draft'`` row. The event is wrapped in a
        :class:`threetears.knowledge.KnowledgeDraftMessage` envelope so the single
        subject carries both events + commands behind a typed discriminator.

        Soft-fails when ``nats_client`` is unwired (no-op + debug log) so a partial
        integration never crashes the harvest.

        :param event: the new-draft event to publish
        :ptype event: KnowledgeDraftEvent
        :return: nothing
        :rtype: None
        """
        if self.nats_client is None:
            log.debug("knowledge draft publish skipped (no nats client)")
            return
        await self.nats_client.publish(
            subject=Subjects.knowledge_draft(),
            message=KnowledgeDraftMessage.for_event(event),
        )

    async def publish_command(self, command: KnowledgeDraftCommand) -> None:
        """Publish a draft lifecycle command to the hub (KNW-54).

        The ``knowledge_drafts`` tool's confirm / edit / discard ops travel the
        same ``{ns}.knowledge.draft`` subject as a
        :class:`threetears.knowledge.KnowledgeDraftCommand` inside a
        :class:`threetears.knowledge.KnowledgeDraftMessage` envelope; the hub
        emitter enforces own-draft-only and applies the op. Soft-fails when
        ``nats_client`` is unwired.

        :param command: the lifecycle command to publish
        :ptype command: KnowledgeDraftCommand
        :return: nothing
        :rtype: None
        """
        if self.nats_client is None:
            log.debug("knowledge draft command skipped (no nats client)")
            return
        await self.nats_client.publish(
            subject=Subjects.knowledge_draft(),
            message=KnowledgeDraftMessage.for_command(command),
        )

    async def list_own_drafts(self, user_id: UUID, *, customer_scope: UUID) -> list[DraftView]:
        """List the caller's own drafts across entries + concepts (KNW-54).

        Unions the entry-collection and concept-collection own-drafts reads (each
        a SELECT over the rbac-read proxy admitted by the broker carve-out).
        Returns the caller's OWN ``status='draft'`` rows only -- drafts are private
        to their author and never surface in retrieval. Soft-fails each side
        independently so a concept backend fault never hides the entry drafts.

        :param user_id: authoring user UUID; only this user's drafts are returned
        :ptype user_id: UUID
        :param customer_scope: conversation customer the broker clamps the Class-B
            draft reads to (broker-isolation-task-01)
        :ptype customer_scope: UUID
        :return: the caller's own drafts (entries + concepts)
        :rtype: list[DraftView]
        """
        drafts: list[DraftView] = []
        if self.entry_collection is not None:
            try:
                drafts.extend(
                    await self.entry_collection.list_own_drafts(user_id, customer_scope=customer_scope),
                )
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- draft listing is best-effort; an entry-backend fault must not hide the concept drafts or crash the tool
                log.warning(
                    "entry drafts list failed (soft-fail): %s: %s",
                    type(exc).__name__,
                    exc,
                )
        if self.concept_collection is not None:
            try:
                drafts.extend(
                    await self.concept_collection.list_own_drafts(user_id, customer_scope=customer_scope),
                )
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- draft listing is best-effort; a concept-backend fault must not hide the entry drafts or crash the tool
                log.warning(
                    "concept drafts list failed (soft-fail): %s: %s",
                    type(exc).__name__,
                    exc,
                )
        return drafts


async def retrieve_entries(
    integration: KnowledgeIntegration,
    user_id: UUID,
    *,
    datasource_id: UUID | None = None,
    scope_context: str | None = None,
    customer_scope: UUID,
) -> tuple[list[EntryEffective], list[EntryLayered]]:
    """Retrieve the merged effective + layered entry views for a caller.

    Fetches the caller-visible, domain-filtered entry snapshots over the proxy
    (visibility SQL-side) and runs the SHARED merge
    :func:`threetears.knowledge.merge_entry_views` to resolve the three-scope
    shadow chains (whole-entry, nearest scope wins). Returns both the effective
    views (for injection) and the layered views (for the provenance footer) so the
    caller does not double-walk.

    KNW-35: when ``scope_context`` is set (an eval-initiated turn), the
    caller-visible snapshots are CLAMPED to the context's admitted scopes BEFORE
    the merge -- the SAME filter the hub fingerprint applies -- so a ``baseline``
    run measures platform knowledge only even when the caller's visibility admits
    customer / user rows. ``None`` (the normal turn) honors the full visible set.

    Soft-fails on error -- returns ``([], [])`` and logs a warning naming the error
    class. NEVER crashes the turn; NEVER a bare ``except Exception: pass``.

    :param integration: knowledge integration instance
    :ptype integration: KnowledgeIntegration
    :param user_id: caller (conversation end-user) UUID for caller-visibility
        filtering
    :ptype user_id: UUID
    :param datasource_id: optional datasource-anchor restriction
    :ptype datasource_id: UUID | None
    :param scope_context: optional eval scope-context clamp (KNW-35); ``None`` for
        a normal turn
    :ptype scope_context: str | None
    :param customer_scope: the conversation's hub-authenticated customer the broker
        clamps the Class-B reads to (broker-isolation-task-01), threaded per-call
        from the verified call identity
    :ptype customer_scope: UUID
    :return: tuple ``(effective_views, layered_views)`` aligned by index;
        ``([], [])`` when retrieval is disabled or soft-fails
    :rtype: tuple[list[EntryEffective], list[EntryLayered]]
    """
    if integration.entry_collection is None:
        return [], []

    try:
        snapshots = await integration.entry_collection.list_visible_to_user(
            user_id,
            datasource_id=datasource_id,
            customer_scope=customer_scope,
        )
        if scope_context is not None:
            admitted = scope_context_admitted_scopes(scope_context)
            snapshots = [s for s in snapshots if s.scope in admitted]
        effective, layered = merge_entry_views(snapshots)
    except Exception as exc:  # prawduct:allow prawduct/broad-except -- retrieval is best-effort context enrichment; a fault yields no knowledge rather than failing the turn
        log.warning(
            "knowledge retrieval failed (soft-fail): %s: %s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        effective, layered = [], []

    return effective, layered


async def retrieve_concepts(
    integration: KnowledgeIntegration,
    user_id: UUID,
    *,
    datasource_id: UUID | None = None,
    datasource_table_id: UUID | None = None,
    scope_context: str | None = None,
    customer_scope: UUID,
) -> tuple[list[ConceptEffective], list[ConceptLayered]]:
    """Retrieve the merged effective + layered concept views for a caller.

    Fetches the caller-visible, binding-domain-filtered concept snapshots over the
    proxy (visibility SQL-side) and runs the SHARED merge
    :func:`threetears.knowledge.merge_concept_views` to resolve the three-scope
    shadow chains (whole-concept, nearest scope wins) and flag cross-scope
    same-name AMBIGUITY (Note 2). Returns both the effective views (for the
    glossary injection) and the layered views (for the provenance footer) so the
    caller does not double-walk.

    KNW-35: when ``scope_context`` is set (an eval-initiated turn), the
    caller-visible snapshots are CLAMPED to the context's admitted scopes BEFORE
    the merge -- the SAME filter the hub fingerprint applies -- so a ``baseline``
    run measures platform concepts only. ``None`` honors the full visible set.

    Soft-fails on error -- returns ``([], [])`` and logs a warning naming the error
    class. NEVER crashes the turn; NEVER a bare ``except Exception: pass``. A
    concept-fetch failure here is contained so the caller's entry injection (a
    SEPARATE retrieve call) is never broken by a concept backend fault.

    :param integration: knowledge integration instance
    :ptype integration: KnowledgeIntegration
    :param user_id: caller (conversation end-user) UUID for caller-visibility
        filtering
    :ptype user_id: UUID
    :param datasource_id: optional datasource-anchor restriction
    :ptype datasource_id: UUID | None
    :param datasource_table_id: optional binding-domain restriction
    :ptype datasource_table_id: UUID | None
    :param scope_context: optional eval scope-context clamp (KNW-35); ``None`` for
        a normal turn
    :ptype scope_context: str | None
    :param customer_scope: the conversation's hub-authenticated customer the broker
        clamps the Class-B reads to (broker-isolation-task-01), threaded per-call
        from the verified call identity
    :ptype customer_scope: UUID
    :return: tuple ``(effective_views, layered_views)`` aligned by index;
        ``([], [])`` when retrieval is disabled or soft-fails
    :rtype: tuple[list[ConceptEffective], list[ConceptLayered]]
    """
    if integration.concept_collection is None:
        return [], []

    try:
        snapshots = await integration.concept_collection.list_visible_to_user(
            user_id,
            datasource_id=datasource_id,
            datasource_table_id=datasource_table_id,
            customer_scope=customer_scope,
        )
        if scope_context is not None:
            admitted = scope_context_admitted_scopes(scope_context)
            snapshots = [s for s in snapshots if s.scope in admitted]
        effective, layered = merge_concept_views(snapshots)
    except Exception as exc:  # prawduct:allow prawduct/broad-except -- concept retrieval is best-effort; a fault yields no concepts and never breaks the separate entry injection or the turn
        log.warning(
            "concept retrieval failed (soft-fail): %s: %s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        effective, layered = [], []

    return effective, layered
