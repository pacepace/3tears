"""A cost is what its runs' provider routing charged, not the model's alone (owner, 2026-10-25), and the page says so
once, at report level, wherever it presents a cost figure: one step down, never on its first screen, where to a
business reader the cost is what it costs (owner reading, 2026-11-06, H13; presentation build Q6).

- **The page** carries one Design & provenance row (``cost as routed`` in the test-setup table) on every shape that
  prints a cost, in HTML and in Markdown, naming the aspects it holds on; no first-screen line, fold, figure or chart
  says it, and an unpriced campaign has none. An aspect's own page that prints a cost says it once too, in its methods
  (``setup-cost``), and its fold on the campaign's page leaves that line out.
- **The priced aspects** are each aspect that prices its arms (``priced_aspects``), what the row holds on.
"""

from __future__ import annotations

import pytest

from threetears.evals.analysis import campaign_report_html, campaign_report_markdown, report_html, report_markdown
from threetears.evals.analysis.report.campaign.provenance import COST_ROUTING_CAVEAT, priced_aspects
from threetears.evals.analysis.report.model import FIRST_SCREEN, CampaignReport, DisclosureBlock, Report, TableBlock
from packages.evals.tests.report_shapes_support import SHAPES, ArmTrials, AspectTrials, Shape, shape_of

#: The id an aspect's own methods line stating it carries.
_ANCHOR = "setup-cost"


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


def _routed_rows(page: CampaignReport) -> list[tuple[str, str]]:
    """The Design & provenance rows saying it: each row's detail and the aspects it holds on."""
    return [
        (row["detail"].text, row["aspects"].text)
        for block in page.blocks
        if isinstance(block, TableBlock) and block.name == "test_setup"
        for row in block.rows
        if row["setup"].text == "cost as routed"
    ]


def _first_screen_words(page: CampaignReport | Report) -> list[str]:
    return [block.text for block in page.blocks if isinstance(block, DisclosureBlock) and block.section in FIRST_SCREEN]


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_a_page_that_prints_a_cost_says_once_in_design_and_provenance_that_cost_is_the_routing_s(name: str) -> None:
    shape = SHAPES[name]()
    assert priced_aspects(list(shape.bundles.values())), "every shape prices its arms"
    page = shape.page()
    (block,) = [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "test_setup"]
    assert block.section == "provenance"
    held = [fold.aspect for fold in page.aspects]
    assert _routed_rows(page) == [(COST_ROUTING_CAVEAT, held[0] if len(held) == 1 else "every aspect")]
    # Never on the first screen, and no fold restates it.
    assert COST_ROUTING_CAVEAT not in _first_screen_words(page)
    for fold in page.aspects:
        assert COST_ROUTING_CAVEAT not in fold.report.model_dump_json()
    html = campaign_report_html(page)
    assert html.count(COST_ROUTING_CAVEAT) == 1 and 'id="caveat-cost"' not in html
    assert campaign_report_markdown(page).count(COST_ROUTING_CAVEAT) == 1


def test_a_page_that_prints_no_cost_does_not_say_it() -> None:
    shape = _unpriced()
    assert priced_aspects(list(shape.bundles.values())) == []
    page = shape.page()
    assert _routed_rows(page) == []
    assert COST_ROUTING_CAVEAT not in campaign_report_html(page)
    assert COST_ROUTING_CAVEAT not in campaign_report_markdown(page)


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_an_aspect_s_own_page_that_prints_a_cost_says_it_once_in_its_methods(name: str) -> None:
    shape = SHAPES[name]()
    for own in shape.reports.values():
        assert [(block.section, block.source, block.text) for block in _caveats(own)] == [
            ("methods", "cost", COST_ROUTING_CAVEAT)
        ]
        assert COST_ROUTING_CAVEAT not in _first_screen_words(own)
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
