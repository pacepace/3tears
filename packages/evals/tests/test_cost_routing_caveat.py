"""A cost is what its runs' provider routing charged, not the model's alone (owner, 2026-10-25), and the page says so
once, at report level, wherever it presents a cost figure.

- **The page** carries one first-screen caveat line (``caveat-cost``) among the report-wide caveats, on every shape
  that prints a cost, in HTML and in Markdown; no fold, figure or chart restates it, and an unpriced campaign has none.
  An aspect's own page that prints a cost says it once too, under its verdict among its report-wide caveats.
- **The priced aspects** are each aspect that prices its arms (``priced_aspects``), what the caveat holds on.
"""

from __future__ import annotations

import pytest

from threetears.evals.analysis import campaign_report_html, campaign_report_markdown, report_html, report_markdown
from threetears.evals.analysis.report.campaign.provenance import COST_ROUTING_CAVEAT, priced_aspects
from threetears.evals.analysis.report.model import CampaignReport, DisclosureBlock, Report
from packages.evals.tests.report_shapes_support import SHAPES, ArmTrials, AspectTrials, Shape, shape_of

#: The id the caveat line carries on the page.
_ANCHOR = "caveat-cost"


def _unpriced() -> Shape:
    """A change of one aspect whose arms carry no cost at all, so its page prints no cost figure."""
    return shape_of(
        "unpriced",
        [
            AspectTrials(
                "billing",
                [ArmTrials("production", passing=12, cost=None), ArmTrials("v2", passing=14, cost=None)],
                margin=None,
            )
        ],
        control=True,
    )


def _caveats(page: CampaignReport | Report) -> list[DisclosureBlock]:
    return [block for block in page.blocks if isinstance(block, DisclosureBlock) and block.anchor == _ANCHOR]


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_a_page_that_prints_a_cost_says_once_on_its_first_screen_that_cost_is_the_routing_s(name: str) -> None:
    shape = SHAPES[name]()
    assert priced_aspects(list(shape.bundles.values())), "every shape prices its arms"
    page = shape.page()
    assert [(block.section, block.source, block.text) for block in _caveats(page)] == [
        ("warnings", "cost", COST_ROUTING_CAVEAT)
    ]
    # Among the report-wide caveats, after what warns: nothing of the first screen follows it but the call.
    warnings = [block for block in page.blocks if block.section == "warnings"]
    assert warnings[-1].anchor == _ANCHOR
    for fold in page.aspects:
        assert COST_ROUTING_CAVEAT not in fold.report.model_dump_json()
    html = campaign_report_html(page)
    assert html.count(COST_ROUTING_CAVEAT) == 1 and html.count(f'id="{_ANCHOR}"') == 1
    assert campaign_report_markdown(page).count(COST_ROUTING_CAVEAT) == 1


def test_a_page_that_prints_no_cost_does_not_say_it() -> None:
    shape = _unpriced()
    assert priced_aspects(list(shape.bundles.values())) == []
    page = shape.page()
    assert _caveats(page) == []
    assert COST_ROUTING_CAVEAT not in campaign_report_html(page)
    assert COST_ROUTING_CAVEAT not in campaign_report_markdown(page)


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_an_aspect_s_own_page_that_prints_a_cost_says_it_once_among_its_caveats(name: str) -> None:
    shape = SHAPES[name]()
    for own in shape.reports.values():
        assert [(block.section, block.source, block.text) for block in _caveats(own)] == [
            ("header", "cost", COST_ROUTING_CAVEAT)
        ]
        assert report_html(own).count(COST_ROUTING_CAVEAT) == 1
        assert report_markdown(own).count(COST_ROUTING_CAVEAT) == 1


def test_an_aspect_s_own_page_that_prints_no_cost_does_not_say_it() -> None:
    for own in _unpriced().reports.values():
        assert _caveats(own) == []
        assert COST_ROUTING_CAVEAT not in report_html(own) and COST_ROUTING_CAVEAT not in report_markdown(own)


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_the_priced_aspects_are_every_aspect_of_a_shape_that_prices_its_arms(name: str) -> None:
    shape = SHAPES[name]()
    labels = [aspect.label for aspect in shape.campaign.aspects]
    assert priced_aspects([shape.bundles[label] for label in labels]) == labels
