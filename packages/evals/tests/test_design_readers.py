"""The declaration's design fields have readers: short cells, verdict order, and the campaign archive.

A field an operator declares and nothing reads is a promise the engine silently breaks. This file
pins the readers of three of them, and the curation write the fourth was missing:

- ``intended_repetitions`` → ``AnalysisContextBundle.short_cells``, one entry per cell whose
  least-repeated case ran fewer times than declared;
- ``merit_priority`` and ``Question.merit_axes`` → ``AnalysisContextBundle.verdict_order``, the bars
  ranked by the declared priority and scoped per question, read off each bar's own ``merit_axis``;
- ``EvalCampaign.archived`` → :func:`~threetears.evals.run.set_campaign_archived`.

Mutations that turn this file red: comparing ``>`` for ``>=`` in ``_short_cells``; taking the
priority order from the bars rather than the declaration in ``_verdict_order``; dropping
``merit_axis`` from the bar adjudication; removing the ``NotFoundError`` raise or the idempotence
check in ``set_campaign_archived``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle
from threetears.evals.contracts import NotFoundError
from threetears.evals.contracts.declaration import Question
from threetears.evals.run import set_campaign_archived
from packages.evals.tests.factories import make_campaign, memory_storage
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_QUESTION_ID, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile


def _bundle(**design_updates: Any) -> AnalysisContextBundle:
    """The toy campaign's bundle, with its declaration amended."""
    campaign, storage = toyhost_campaign()
    design = campaign.declared_design
    assert design is not None
    campaign = campaign.model_copy(update={"declared_design": design.model_copy(update=design_updates)})
    return assemble_context_bundle(campaign, storage=storage, profile=toyhost_profile())


class TestShortCells:
    """The toy campaign's two cells each run every document three times."""

    def test_a_cell_short_of_the_declared_repetitions_is_named_with_its_sentence(self) -> None:
        bundle = _bundle(intended_repetitions=4)

        assert [cell.repeats_per_case_min for cell in bundle.cells] == [3, 3]
        assert [(short.intended, short.observed) for short in bundle.short_cells] == [(4, 3), (4, 3)]
        assert [(s.variant_key, s.apparatus_class_id) for s in bundle.short_cells] == [
            (cell.variant_key, cell.apparatus_class_id) for cell in bundle.cells
        ]
        assert bundle.short_cells[0].sentence.startswith("This cell ran its least-repeated case 3 times against the 4")

    def test_a_cell_meeting_the_declared_repetitions_exactly_is_not_short(self) -> None:
        assert _bundle(intended_repetitions=3).short_cells == []

    def test_an_unstated_intention_names_no_cell(self) -> None:
        """Undetectable, not zero: the declaration's null is where a reader learns which."""
        bundle = _bundle(intended_repetitions=None)

        assert bundle.short_cells == []
        assert bundle.declared_design is not None and bundle.declared_design.intended_repetitions is None


class TestVerdictOrder:
    """The toy campaign adjudicates `cost_usd` (no axis) and `field_accuracy` (quality)."""

    def test_each_bar_carries_the_axis_its_descriptor_declares(self) -> None:
        bars = {bar.measure_id: bar for bar in _bundle().bar_adjudications}

        assert bars["field_accuracy"].merit_axis == "quality"
        assert bars["p95_extract_ms"].merit_axis == "latency", "a bar with no verdict still names its axis"
        assert bars["cost_usd"].merit_axis is None, "cost_usd includes the judge, so it serves no merit axis"

    def test_with_no_stated_priority_every_bar_is_unranked(self) -> None:
        order = _bundle(merit_priority=[]).verdict_order

        assert order.merit_priority == []
        assert order.tiers == []
        assert order.unranked_bar_measure_ids == ["cost_usd", "field_accuracy"]

    def test_the_declared_priority_ranks_the_axes_and_names_one_with_no_bar(self) -> None:
        order = _bundle(merit_priority=["latency", "quality"]).verdict_order

        assert [(tier.axis, tier.bar_measure_ids) for tier in order.tiers] == [
            ("latency", []),
            ("quality", ["field_accuracy"]),
        ], "latency is ranked first and its one bar has no verdict, so the tier is empty rather than skipped"
        assert order.unranked_bar_measure_ids == ["cost_usd"]

    def test_the_priority_orders_the_tiers_not_the_bars(self) -> None:
        """Reversing the declaration reverses the tiers — the order is the declaration's, not the bars'."""
        forward = _bundle(merit_priority=["quality", "latency"]).verdict_order
        backward = _bundle(merit_priority=["latency", "quality"]).verdict_order

        assert [tier.axis for tier in forward.tiers] == ["quality", "latency"]
        assert [tier.axis for tier in backward.tiers] == ["latency", "quality"]

    def test_a_question_lists_the_bars_on_its_axes_and_the_axes_no_bar_answers(self) -> None:
        question = Question(
            id=TOYHOST_QUESTION_ID, text="is it accurate, and what does it cost?", merit_axes=["cost", "quality"]
        )
        order = _bundle(merit_priority=["quality"], questions=[question]).verdict_order

        (scope,) = order.questions
        assert scope.question_id == TOYHOST_QUESTION_ID
        assert scope.merit_axes == ["quality", "cost"], "the ranked axis first, then the question's own order"
        assert scope.bar_measure_ids == ["field_accuracy"]
        assert scope.unbarred_axes == ["cost"]

    def test_an_unscoped_or_retired_question_is_absent(self) -> None:
        unscoped = Question(id="q-unscoped", text="anything else?")
        retired = Question(id="q-retired", text="old?", merit_axes=["quality"], retired_at="2026-01-01T00:00:00+00:00")
        order = _bundle(questions=[unscoped, retired]).verdict_order

        assert order.questions == []


class TestSetCampaignArchived:
    """The archive write the campaign's `archived` field never had."""

    def test_archiving_hides_the_campaign_from_an_active_listing_and_restoring_brings_it_back(self) -> None:
        storage, _store = memory_storage()
        campaign = make_campaign()
        storage.save_campaign(campaign)

        archived = set_campaign_archived(storage, campaign.id, campaign.scope_id, archived=True)

        assert archived.archived is True
        assert storage.list_campaigns(campaign.scope_id, archived=False) == []
        assert [c.id for c in storage.list_campaigns(campaign.scope_id, archived=True)] == [campaign.id]
        restored = set_campaign_archived(storage, campaign.id, campaign.scope_id, archived=False)
        assert restored.archived is False
        assert [c.id for c in storage.list_campaigns(campaign.scope_id, archived=False)] == [campaign.id]

    def test_archiving_keeps_everything_else(self) -> None:
        storage, _store = memory_storage()
        campaign = make_campaign(run_ids=["r-1", "r-2"], description="kept")
        storage.save_campaign(campaign)

        archived = set_campaign_archived(storage, campaign.id, campaign.scope_id, archived=True)

        assert archived.model_dump(exclude={"archived"}) == campaign.model_dump(exclude={"archived"})

    def test_setting_the_state_it_already_has_writes_nothing(self) -> None:
        campaign = make_campaign()
        storage = MagicMock()
        storage.load_campaign.return_value = campaign

        assert set_campaign_archived(storage, campaign.id, campaign.scope_id, archived=False) is campaign
        storage.save_campaign.assert_not_called()

    def test_an_unknown_campaign_is_refused(self) -> None:
        storage, _store = memory_storage()

        with pytest.raises(NotFoundError):
            set_campaign_archived(storage, "no-such-campaign", "uni-1", archived=True)
