"""Unit tests for the knowledge integration + module retrieval helpers.

Covers the pure, non-trivial logic that does not need live infrastructure: the
eval scope-context clamp (:func:`scope_context_admitted_scopes` and its effect on
:func:`retrieve_entries` / :func:`retrieve_concepts`), the disabled-collection and
soft-fail contracts, the draft publish no-op when the NATS client is unwired, and
the own-drafts union with independent per-side soft-fail.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid7

import pytest

from threetears.knowledge import ConceptSnapshot, EntrySnapshot, Scope
from threetears.knowledge import KnowledgeDraftCommand, KnowledgeDraftEvent

from threetears.agent.knowledge.integration import (
    DraftView,
    KnowledgeIntegration,
    retrieve_concepts,
    retrieve_entries,
    scope_context_admitted_scopes,
)


class _EntryColl:
    """entry collection stand-in returning a fixed snapshot list."""

    def __init__(self, snapshots: list[EntrySnapshot], *, raises: bool = False) -> None:
        self._snapshots = snapshots
        self._raises = raises

    async def list_visible_to_user(
        self, user_id: Any, *, datasource_id: Any = None, customer_scope: Any
    ) -> list[EntrySnapshot]:
        if self._raises:
            raise RuntimeError("entry boom")
        return list(self._snapshots)


class _ConceptColl:
    """concept collection stand-in returning a fixed snapshot list."""

    def __init__(self, snapshots: list[ConceptSnapshot], *, raises: bool = False) -> None:
        self._snapshots = snapshots
        self._raises = raises

    async def list_visible_to_user(
        self,
        user_id: Any,
        *,
        datasource_id: Any = None,
        datasource_table_id: Any = None,
        customer_scope: Any,
    ) -> list[ConceptSnapshot]:
        if self._raises:
            raise RuntimeError("concept boom")
        return list(self._snapshots)


class _DraftColl:
    """collection stand-in exposing ``list_own_drafts``."""

    def __init__(self, drafts: list[DraftView], *, raises: bool = False) -> None:
        self._drafts = drafts
        self._raises = raises

    async def list_own_drafts(self, user_id: Any, *, customer_scope: Any) -> list[DraftView]:
        if self._raises:
            raise RuntimeError("draft boom")
        return list(self._drafts)


def _entry(scope: Scope) -> EntrySnapshot:
    return EntrySnapshot(id=uuid7(), scope=scope, title="t", body="b")


def _concept(scope: Scope) -> ConceptSnapshot:
    return ConceptSnapshot(id=uuid7(), scope=scope, name="n", definition="d")


def _draft(target: str) -> DraftView:
    return DraftView(
        draft_id=uuid7(),
        target=target,
        title="title",
        body="body",
        related_domain="",
        conversation_id=None,
        turn_count=None,
    )


class TestScopeContextAdmittedScopes:
    def test_baseline_admits_platform_only(self) -> None:
        assert scope_context_admitted_scopes("baseline") == frozenset({Scope.PLATFORM})

    def test_customer_effective_admits_platform_customer(self) -> None:
        assert scope_context_admitted_scopes("customer_effective") == frozenset({Scope.PLATFORM, Scope.CUSTOMER})

    def test_user_effective_admits_all(self) -> None:
        assert scope_context_admitted_scopes("user_effective") == frozenset(
            {Scope.PLATFORM, Scope.CUSTOMER, Scope.USER}
        )

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown scope_context"):
            scope_context_admitted_scopes("nonsense")


class TestRetrieveEntries:
    async def test_disabled_collection_returns_empty(self) -> None:
        integration = KnowledgeIntegration(entry_collection=None)
        eff, lay = await retrieve_entries(integration, uuid7(), customer_scope=uuid7())
        assert eff == []
        assert lay == []

    async def test_soft_fail_returns_empty(self) -> None:
        integration = KnowledgeIntegration(entry_collection=_EntryColl([], raises=True))
        eff, lay = await retrieve_entries(integration, uuid7(), customer_scope=uuid7())
        assert eff == []
        assert lay == []

    async def test_scope_clamp_baseline_drops_non_platform(self) -> None:
        integration = KnowledgeIntegration(
            entry_collection=_EntryColl([_entry(Scope.PLATFORM), _entry(Scope.CUSTOMER)]),
        )
        eff, _lay = await retrieve_entries(integration, uuid7(), scope_context="baseline", customer_scope=uuid7())
        assert len(eff) == 1
        assert eff[0].entry.scope == Scope.PLATFORM

    async def test_no_scope_context_keeps_all(self) -> None:
        integration = KnowledgeIntegration(
            entry_collection=_EntryColl([_entry(Scope.PLATFORM), _entry(Scope.CUSTOMER)]),
        )
        eff, _lay = await retrieve_entries(integration, uuid7(), customer_scope=uuid7())
        assert len(eff) == 2


class TestRetrieveConcepts:
    async def test_disabled_collection_returns_empty(self) -> None:
        integration = KnowledgeIntegration(concept_collection=None)
        eff, lay = await retrieve_concepts(integration, uuid7(), customer_scope=uuid7())
        assert eff == []
        assert lay == []

    async def test_soft_fail_returns_empty(self) -> None:
        integration = KnowledgeIntegration(concept_collection=_ConceptColl([], raises=True))
        eff, lay = await retrieve_concepts(integration, uuid7(), customer_scope=uuid7())
        assert eff == []
        assert lay == []

    async def test_scope_clamp_customer_effective(self) -> None:
        integration = KnowledgeIntegration(
            concept_collection=_ConceptColl([_concept(Scope.PLATFORM), _concept(Scope.USER)]),
        )
        eff, _lay = await retrieve_concepts(
            integration,
            uuid7(),
            scope_context="customer_effective",
            customer_scope=uuid7(),
        )
        # customer_effective admits platform + customer; the USER concept is dropped.
        assert len(eff) == 1
        assert eff[0].concept.scope == Scope.PLATFORM


class TestPublishSoftFail:
    async def test_publish_draft_noop_without_client(self) -> None:
        integration = KnowledgeIntegration(nats_client=None)
        # returns before touching the (unused) event -> no raise.
        await integration.publish_draft(cast("KnowledgeDraftEvent", object()))

    async def test_publish_command_noop_without_client(self) -> None:
        integration = KnowledgeIntegration(nats_client=None)
        await integration.publish_command(cast("KnowledgeDraftCommand", object()))


class TestListOwnDrafts:
    async def test_unions_entry_and_concept_drafts(self) -> None:
        integration = KnowledgeIntegration(
            entry_collection=_DraftColl([_draft("entry")]),
            concept_collection=_DraftColl([_draft("concept")]),
        )
        drafts = await integration.list_own_drafts(uuid7(), customer_scope=uuid7())
        targets = {d.target for d in drafts}
        assert targets == {"entry", "concept"}

    async def test_concept_fault_does_not_hide_entry_drafts(self) -> None:
        integration = KnowledgeIntegration(
            entry_collection=_DraftColl([_draft("entry")]),
            concept_collection=_DraftColl([], raises=True),
        )
        drafts = await integration.list_own_drafts(uuid7(), customer_scope=uuid7())
        assert [d.target for d in drafts] == ["entry"]
