"""The declaration's design fields have readers: short cells, verdict order, and the campaign archive.

A field an operator declares and nothing reads is a promise the engine silently breaks. This file
pins the readers of three of them, and the curation write the fourth was missing:

- ``intended_repetitions`` → ``AnalysisContextBundle.short_cells``, one entry per cell whose
  least-repeated case ran fewer times than declared;
- ``merit_priority`` and ``Question.merit_axes`` → ``AnalysisContextBundle.verdict_order``, the bars
  ranked by the declared priority and scoped per question, read off each bar's own ``merit_axis``;
- ``EvalCampaign.archived`` → :func:`~threetears.evals.run.set_campaign_archived`.

Mutations that turn this file red: comparing ``>`` for ``>=`` in ``_short_cells``; reading a cell's
``repeats_per_case_max`` where it reads the minimum; taking the
priority order from the bars rather than the declaration in ``_verdict_order``; dropping
``merit_axis`` from the bar adjudication; removing the ``NotFoundError`` raise or the idempotence
check in ``set_campaign_archived``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from threetears.evals.analysis import (
    AnalysisContextBundle,
    DisclosureBlock,
    assemble_context_bundle,
    build_code_only_report,
)
from threetears.evals.contracts import NotFoundError
from threetears.evals.contracts.declaration import Question
from threetears.evals.run import set_campaign_archived
from packages.evals.tests.factories import make_campaign, memory_storage
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_QUESTION_ID, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
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

    def test_the_least_repeated_case_decides_not_the_most(self) -> None:
        """One case of one cell ran twice and the rest three times: that cell is short of three, the other is not.

        Every case in the toy campaign otherwise runs exactly three times, so min and max agree in every cell
        and nothing above could tell which the reader takes.
        """
        campaign, storage = toyhost_campaign()
        runs = storage.load_eval_runs(campaign.run_ids, campaign.scope_id)
        results = {run.id: storage.query_eval_results_by_run(run.id, campaign.scope_id) for run in runs}
        thinned, other = runs
        first_case = results[thinned.id][0].test_case_id
        dropped = next(r for r in results[thinned.id] if r.test_case_id == first_case)
        results[thinned.id] = [r for r in results[thinned.id] if r.id != dropped.id]
        design = campaign.declared_design
        assert design is not None
        declared = campaign.model_copy(update={"declared_design": design.model_copy(update={"intended_repetitions": 3})})

        bundle = assemble_context_bundle(declared, storage=ToyhostStorage(runs, results), profile=toyhost_profile())

        uneven = next(cell for cell in bundle.cells if cell.repeats_per_case_min != cell.repeats_per_case_max)
        assert (uneven.repeats_per_case_min, uneven.repeats_per_case_max) == (2, 3), "the fixture is uneven"
        (short,) = bundle.short_cells
        assert (short.variant_key, short.apparatus_class_id) == (uneven.variant_key, uneven.apparatus_class_id)
        assert (short.intended, short.observed) == (3, 2)
        assert other.id != thinned.id

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


class TestTheDeclaredPriorityReachesTheReport:
    """A code-only report has no writer, so the tie-break order the campaign declared is stated by code."""

    def test_a_declared_priority_is_disclosed_with_each_tiers_bars(self) -> None:
        report = build_code_only_report(
            _bundle(merit_priority=["quality", "latency"]),
            measures=toyhost_profile().measures,
            assembled_at="2026-10-05T00:00:00+00:00",
        )
        texts = [block.text for block in report.blocks if isinstance(block, DisclosureBlock)]

        (sentence,) = [text for text in texts if text.startswith("The campaign ranks its merit axes")]
        assert "quality (field_accuracy); latency (no adjudicated bar)" in sentence
        assert "which the order does not place: cost_usd." in sentence

    def test_no_stated_priority_states_none(self) -> None:
        report = build_code_only_report(
            _bundle(merit_priority=[]), measures=toyhost_profile().measures, assembled_at="2026-10-05T00:00:00+00:00"
        )
        texts = [block.text for block in report.blocks if isinstance(block, DisclosureBlock)]
        assert not [text for text in texts if text.startswith("The campaign ranks its merit axes")]


class TestAMeritRankingNamesEachAxisOnce:
    """A ranking with an axis twice put one bar in two strongest-first tiers; it is refused where it is authored."""

    def test_a_priority_naming_an_axis_twice_is_refused(self) -> None:
        campaign, _storage = toyhost_campaign()
        design = campaign.declared_design
        assert design is not None
        type(design).model_validate(design.model_dump() | {"merit_priority": ["quality", "latency"]})

        with pytest.raises(ValueError, match="merit_priority names quality more than once"):
            type(design).model_validate(design.model_dump() | {"merit_priority": ["quality", "quality"]})

    def test_a_question_naming_an_axis_twice_is_refused(self) -> None:
        Question(id="q", text="accurate and cheap?", merit_axes=["quality", "cost"])

        with pytest.raises(ValueError, match="merit_axes names quality more than once"):
            Question(id="q", text="accurate?", merit_axes=["quality", "quality"])


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
