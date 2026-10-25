"""A cost is what its runs' provider routing charged, not the model's alone (owner, 2026-10-25), and the page and the
writer's view each say so once, at report level, wherever they present a cost figure.

- **The page** carries one first-screen caveat line (``caveat-cost``) among the report-wide caveats, on every shape
  that prints a cost, in HTML and in Markdown; no fold, figure or chart restates it, and an unpriced campaign has none.
- **The writer's view** hands it once, in the campaign's own ``disclosures`` (``cost_routing``), as the page spells
  it, naming the priced aspects; no aspect's part carries it, and an unpriced campaign's view has none.
"""

from __future__ import annotations

import json

import pytest

from threetears.evals.analysis import campaign_report_html, campaign_report_markdown
from threetears.evals.analysis.campaign_bundle import build_campaign_bundle, writer_view
from threetears.evals.analysis.report.campaign.provenance import COST_ROUTING_CAVEAT, priced_aspects
from threetears.evals.analysis.report.model import CampaignReport, DisclosureBlock
from packages.evals.tests.report_shapes_support import SHAPES, ArmTrials, AspectTrials, Shape, shape_of

#: The id the caveat line carries on the page.
_ANCHOR = "caveat-cost"

#: The key the writer's view hands it under.
_KEY = "cost_routing"


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


def _caveats(page: CampaignReport) -> list[DisclosureBlock]:
    return [block for block in page.blocks if isinstance(block, DisclosureBlock) and block.anchor == _ANCHOR]


def _view(shape: Shape) -> dict[str, object]:
    bundles = [shape.bundles[aspect.label] for aspect in shape.campaign.aspects]
    return writer_view(build_campaign_bundle(shape.campaign, bundles, arms=shape.arms, readings=shape.readings))


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
def test_the_writer_s_view_hands_it_once_in_the_campaign_s_disclosures_and_in_no_aspect_s_part(name: str) -> None:
    shape = SHAPES[name]()
    view = _view(shape)
    disclosures = view["disclosures"]
    assert isinstance(disclosures, dict)
    labels = [aspect.label for aspect in shape.campaign.aspects]
    assert disclosures[_KEY] == [{"aspects": labels, "said": COST_ROUTING_CAVEAT}]
    assert json.dumps(view).count(COST_ROUTING_CAVEAT) == 1
    parts = view["aspects"]
    assert isinstance(parts, list)
    for part in parts:
        assert _KEY not in part["disclosures"] and COST_ROUTING_CAVEAT not in json.dumps(part)


def test_a_view_handing_no_cost_does_not_say_it() -> None:
    view = _view(_unpriced())
    disclosures = view.get("disclosures", {})
    assert isinstance(disclosures, dict) and _KEY not in disclosures
    assert COST_ROUTING_CAVEAT not in json.dumps(view)
