"""The code-only report charts only readings a chart can say something about.

Three kinds of chart carried nothing and are not emitted:

- **A classifier's per-label statistics.** Precision and recall were a chart per label and statistic, and
  F1, which has no interval in any cell by construction, only ever a disclosure that every cell was left
  out. All three are the per-label table (``test_compact_code_only_report.py``).
- **``match`` beside ``accuracy``**, which is derived from it observation for observation.
- **Cost no result observed.** A result with no usage row carrying dollars stores ``cost_usd`` 0.0 as the
  sum of nothing. The bundle reads it as no observation (:func:`spend_observed`), so a cell where nothing
  reported spend has no cost reading to chart or test, and the bundle says so in one sentence. A MEASURED
  $0 — a row carrying 0 dollars — is a reading like any other.

Mutations that turn this file red: dropping the per-label or ``match`` filter in ``_chartable``; reading
``cost_usd`` in ``_lineage_leaves`` whatever ``spend_observed`` says; listing a cell with a measured $0 in
``_cost_unmeasured``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from threetears.evals.analysis import ChartBlock, DisclosureBlock, TableBlock, inspect_campaign_bundle
from threetears.evals.contracts.models import RoleUsage
from threetears.evals.contracts.usage_capture import spend_observed
from threetears.evals.quick import Answer, Comparison, compare

SCOPE = "code-only-chart-tests"

CASES = [
    {"text": "a cat", "label": "animal"},
    {"text": "a dog", "label": "animal"},
    {"text": "a fir", "label": "plant"},
    {"text": "an oak", "label": "plant"},
]


def _expected(case: Mapping[str, Any]) -> str:
    return str(case["label"])


async def right(case: Mapping[str, Any]) -> str:
    return str(case["label"])


async def guess(case: Mapping[str, Any]) -> str:
    return "plant"


async def right_priced(case: Mapping[str, Any]) -> Answer:
    return Answer(case["label"], model="m-1", input_tokens=10, output_tokens=1, cost_usd=2e-5)


async def guess_priced(case: Mapping[str, Any]) -> Answer:
    return Answer("plant", model="m-2", input_tokens=10, output_tokens=1, cost_usd=1e-5)


async def right_free(case: Mapping[str, Any]) -> Answer:
    return Answer(case["label"], model="local", cost_usd=0.0)


async def _compare(candidate: Any, control: Any) -> Comparison:
    return await compare(
        CASES, {"control": control, "candidate": candidate}, expected=_expected, control="control", scope_id=SCOPE, k=2
    )


def _chart_titles(comparison: Comparison) -> list[str]:
    return [
        block.intent.title
        for block in comparison.report.blocks
        if isinstance(block, ChartBlock) and block.intent is not None
    ]


def _disclosures(comparison: Comparison) -> list[str]:
    return [block.text for block in comparison.report.blocks if isinstance(block, DisclosureBlock)]


def _cost_disclosures(comparison: Comparison) -> list[str]:
    return [text for text in _disclosures(comparison) if text.startswith("Cost was not measured")]


class TestReadingsWithNoChart:
    async def test_no_per_label_chart_and_no_notice_for_one(self) -> None:
        comparison = await _compare(right, guess)
        titles = _chart_titles(comparison)
        assert not [title for title in titles if title.startswith("classifier:")]
        assert not [text for text in _disclosures(comparison) if "classifier:" in text]
        # Precision, recall and F1 are the per-label table's, every label in it.
        (table,) = [
            block for block in comparison.report.blocks if isinstance(block, TableBlock) and block.name == "labels"
        ]
        assert {row["label"] for row in table.rows} == {"animal", "plant"}

    async def test_match_is_not_charted_beside_the_accuracy_derived_from_it(self) -> None:
        titles = _chart_titles(await _compare(right, guess))
        assert "accuracy" in titles and "match" not in titles


class TestUnobservedCost:
    async def test_no_arm_reporting_spend_charts_and_tests_no_cost_and_says_so_once(self) -> None:
        comparison = await _compare(right, guess)
        assert not [title for title in _chart_titles(comparison) if title.startswith("cost_usd")]
        assert {row["reading"] for row in comparison.contrasts()} == {"accuracy"}
        (said,) = _cost_disclosures(comparison)
        assert "no result reported its spend" in said and "returning an Answer" in said

        bundle = inspect_campaign_bundle(comparison.host, comparison.campaign_id, SCOPE).bundle
        assert len(bundle.cost_unmeasured_cells) == len(bundle.cell_measures) == 2
        assert bundle.cost_unmeasured == said
        assert not any(m.name == "cost_usd" for cell in bundle.cell_measures for m in cell.measures.measures)

    async def test_reported_spend_is_charted_and_tested_as_before(self) -> None:
        comparison = await _compare(right_priced, guess_priced)
        assert [title for title in _chart_titles(comparison) if title.startswith("cost_usd")]
        assert "cost_usd" in {row["reading"] for row in comparison.contrasts()}
        assert _cost_disclosures(comparison) == []

    async def test_a_measured_zero_is_a_reading_not_an_unmeasured_cost(self) -> None:
        comparison = await _compare(right_free, guess_priced)
        assert _cost_disclosures(comparison) == []
        bundle = inspect_campaign_bundle(comparison.host, comparison.campaign_id, SCOPE).bundle
        assert bundle.cost_unmeasured_cells == [] and bundle.cost_unmeasured is None
        costs = sorted(
            next(m.mean for m in cell.measures.measures if m.name == "cost_usd") for cell in bundle.cell_measures
        )
        assert costs == [0.0, 1e-5]
        assert "cost_usd" in {row["reading"] for row in comparison.contrasts()}

    async def test_one_arm_unmeasured_is_named_and_left_out_of_the_cost_chart(self) -> None:
        comparison = await _compare(right, guess_priced)
        (said,) = _cost_disclosures(comparison)
        assert said.startswith("Cost was not measured in 1 of 2 cells") and said.endswith(
            "Unmeasured: model=candidate."
        )
        bundle = inspect_campaign_bundle(comparison.host, comparison.campaign_id, SCOPE).bundle
        (unmeasured,) = bundle.cost_unmeasured_cells
        (cell,) = [c for c in bundle.cell_measures if c.variant_key == unmeasured.variant_key]
        assert not any(m.name == "cost_usd" for m in cell.measures.measures)


class TestSpendObserved:
    def test_a_row_carrying_dollars_in_the_cost_roles_is_an_observation(self) -> None:
        priced = RoleUsage(role="candidate", model="m", cost_usd=0.0)
        assert spend_observed([priced], ("candidate",))

    def test_no_row_or_no_dollars_or_a_role_outside_the_total_is_not(self) -> None:
        assert not spend_observed([], ("candidate", "judge"))
        assert not spend_observed([RoleUsage(role="candidate", model="m", prompt_tokens=5)], ("candidate",))
        assert not spend_observed([RoleUsage(role="external", model=None, cost_usd=1.0)], ("candidate",))
