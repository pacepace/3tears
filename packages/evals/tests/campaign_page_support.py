"""The page tests' fixture campaigns, and how a test reads a campaign's page.

Two real runs read as one campaign of two aspects: on billing the candidate regresses (it passes six of twenty cases
to production's eighteen) and breaks the 10-point floor declared there; on tone it improves (nineteen to six). Steady
holds within a 20-point floor and margin, and late improves on a question asked only after its runs. The other kinds
are read off alpha and beta (a choice) and alpha alone (a measurement). Each aspect is measured once per process.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


from threetears.evals.analysis import (
    AnalysisContextBundle,
    CampaignReadings,
    CampaignReport,
    build_code_only_report,
    build_campaign_report,
    campaign_readings,
    campaign_report_html,
    campaign_report_markdown,
)
from threetears.evals.analysis.report.model import (
    DisclosureBlock,
    TableBlock,
    TextBlock,
)
from threetears.evals.analysis.report.words import campaign_arm_tags
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.declaration import CampaignCall, RateDrop

from packages.evals.tests.report_shapes_support import COST_BAR
from packages.evals.tests.campaign_support import (
    ASSEMBLED_AT,
    MeasuredAspect,
    MeasuredKind,
    arms_of,
    joined,
    measured_aspect,
    measured_kind,
)


_MEASURED: dict[str, MeasuredAspect] = {}


#: When the late aspect was asked, its margin set and its control named: long after its readings were visible.
_ASKED_LATE = "2099-01-01T00:00:00+00:00"


async def aspect_fixture(label: str) -> MeasuredAspect:
    """Each fixture aspect, measured once for the module: billing regresses past its floor, tone improves, steady
    holds within a 20-point floor and margin (both arms pass every case, which on twenty cases bounds the drop at 19
    points), and late improves on a question asked only after its runs."""
    if label not in _MEASURED:
        if label == "billing":
            _MEASURED[label] = await measured_aspect(
                "billing", production=18, candidate=6, floors=(RateDrop(max_points=10),)
            )
        elif label == "tone":
            _MEASURED[label] = await measured_aspect("tone", production=6, candidate=19)
        elif label == "steady":
            _MEASURED[label] = await measured_aspect(
                "steady", production=20, candidate=20, margin=20, floors=(RateDrop(max_points=20),)
            )
        else:
            _MEASURED[label] = await measured_aspect(label, production=6, candidate=19, asked_at=_ASKED_LATE)
    return _MEASURED[label]


async def campaign_fixture(
    *labels: str, margins: dict[str, float | None] | None = None, **changes: Any
) -> tuple[EvalCampaign, list[MeasuredAspect]]:
    """One campaign holding the fixture aspects named, billing's campaign carrying every one and every floor, with
    ``changes`` to its fields."""
    measured = [await aspect_fixture(label) for label in labels]
    aspects = [
        aspect.model_copy(update={"margin": (margins or {}).get(aspect.label, aspect.margin)})
        for one in measured
        for aspect in one.campaign.aspects
    ]
    floors = [floor for one in measured for floor in one.campaign.floors]
    first = measured[0].campaign
    campaign = first.model_copy(update={"aspects": aspects, "floors": floors, "name": "support queue", **changes})
    return campaign, measured


async def page_of(
    *labels: str,
    call: CampaignCall | None = None,
    margins: dict[str, float | None] | None = None,
    bundle: Callable[[AnalysisContextBundle], AnalysisContextBundle] = lambda bundle: bundle,
    readings: Callable[[CampaignReadings], CampaignReadings] = lambda readings: readings,
    **changes: Any,
) -> CampaignReport:
    """The page over the fixture aspects named, each aspect's page built as a host builds it, with its arms named by
    the campaign's labels; ``bundle`` and ``readings`` change what is read, ``changes`` the campaign."""
    campaign, measured = await campaign_fixture(*labels, margins=margins, **changes)
    bundles = {changed.aspect: changed for changed in (bundle(one.bundle) for one in measured)}
    read = readings(campaign_readings(campaign, bundles, arms=arms_of(measured[0])))
    tags = campaign_arm_tags(bundles.values(), read.control)
    reports = {
        label: build_code_only_report(
            evidence, assembled_at=ASSEMBLED_AT, campaign_name="support", arm_labels=read.arm_labels(), arm_tags=tags
        )
        for label, evidence in bundles.items()
    }
    return build_campaign_report(campaign, read, reports, call=call)


def opening(page: CampaignReport) -> list[tuple[str, str]]:
    """What the page says up to its first table, less its warnings and its question: each text block by its role, and
    each other line by its source and text."""
    said: list[tuple[str, str]] = []
    for block in page.blocks:
        if isinstance(block, TableBlock):
            break
        if isinstance(block, TextBlock) and block.role != "override":
            said.append(("text", block.role))
        elif isinstance(block, DisclosureBlock) and block.source not in {"bar", "floor", "guardrails", "basis"}:
            said.append((block.source, block.text))
    return said


def first_screen_markdown(page: CampaignReport) -> str:
    """The Markdown above the first section heading: the first screen."""
    return campaign_report_markdown(page).split("\n## ", 1)[0]


def header_html(page: CampaignReport) -> str:
    html = campaign_report_html(page)
    return html[html.index("<header>") : html.index("</header>")]


_KINDS: dict[str, MeasuredKind] = {}


async def kind_fixture(label: str) -> MeasuredKind:
    """Each fixture aspect of the other two kinds, measured once: on tone alpha passes nineteen cases to beta's six,
    on billing ten to nine, on refunds ten to ten; solo is alpha alone."""
    if label not in _KINDS:
        shapes: dict[str, tuple[dict[str, int], bool]] = {
            "tone": ({"alpha": 19, "beta": 6}, False),
            "billing": ({"alpha": 10, "beta": 9}, False),
            "refunds": ({"alpha": 10, "beta": 10}, False),
            "solo": ({"alpha": 15}, True),
        }
        arms, control = shapes[label]
        _KINDS[label] = await measured_kind(label, arms, control=control, bars=[COST_BAR], costs={"beta": 0.0005})
    return _KINDS[label]


async def kind_page_of(
    *labels: str,
    bundle: Callable[[AnalysisContextBundle], AnalysisContextBundle] = lambda bundle: bundle,
) -> CampaignReport:
    measured = [await kind_fixture(label) for label in labels]
    campaign = joined(measured)
    bundles = {one.bundle.aspect: bundle(one.bundle) for one in measured}
    read = campaign_readings(campaign, bundles, arms=measured[0].arms())
    tags = campaign_arm_tags(bundles.values(), read.control)
    reports = {
        label: build_code_only_report(
            evidence, assembled_at=ASSEMBLED_AT, campaign_name="support", arm_labels=read.arm_labels(), arm_tags=tags
        )
        for label, evidence in bundles.items()
    }
    return build_campaign_report(campaign, read, reports)


__all__ = [
    "aspect_fixture",
    "campaign_fixture",
    "first_screen_markdown",
    "header_html",
    "kind_fixture",
    "kind_page_of",
    "opening",
    "page_of",
]
