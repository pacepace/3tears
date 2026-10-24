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
from threetears.evals.kernel.authored import (
    AspectSummary,
    AuthoredCampaignAnalysis,
    Finding,
    NextStep,
    Recommendation,
)
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.declaration import CampaignCall, RateDrop

from packages.evals.tests.report_shapes_support import COST_BAR
from packages.evals.tests.campaign_support import (
    ASSEMBLED_AT,
    CONTROL_LABEL,
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
    memo: AuthoredCampaignAnalysis | None = None,
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
    reports = {
        label: build_code_only_report(
            evidence, assembled_at=ASSEMBLED_AT, campaign_name="support", arm_labels=read.arm_labels()
        )
        for label, evidence in bundles.items()
    }
    return build_campaign_report(
        campaign, read, reports, memo, call=call, recommended_variant_key=recorded_key(read, memo)
    )


def recorded_key(read: CampaignReadings, memo: AuthoredCampaignAnalysis | None) -> str | None:
    """The key an analysis records for the arm its memo recommends: the arm the readings label so."""
    recommended = memo.recommended() if memo is not None else None
    if recommended is None:
        return None
    return {label: key for key, label in read.arm_labels().items()}.get(recommended.arm)


def two_aspect_memo() -> AuthoredCampaignAnalysis:
    finding = Finding(
        title="v2 resolves tone and breaks billing.",
        body="It passes far more tone requests and far fewer billing ones.",
        confidence="high",
        axes=[],
        evidence=[],
        trials=[],
        caveats=[],
        invalidates=[],
        durable="",
    )
    return AuthoredCampaignAnalysis(
        findings=[finding],
        recommendation=[
            Recommendation(
                arm=CONTROL_LABEL,
                reasons=["The billing floor is broken, and billing is where refunds are lost."],
                confidence="high",
                rests_on=[0],
                floors_overridden=[],
                behind_on=[],
                bars_missed=[],
            )
        ],
        aspect_summaries=[
            AspectSummary(
                aspect="billing",
                what_changed="v2 fails most billing requests production passes.",
                how_sure="Shown worse past the margin.",
                means="Adopting v2 would lose refunds.",
            ),
            AspectSummary(
                aspect="tone",
                what_changed="v2 answers in the house tone far more often.",
                how_sure="Shown better.",
                means="Worth keeping for a later candidate.",
            ),
        ],
        summary=SUMMARY,
        next=[
            NextStep(
                title="Fix the billing prompt and run again.", why="It is the only loss.", leverage="high", lever=""
            )
        ],
    )


#: The test memo's read across its aspects.
SUMMARY = "v2 wins on tone and loses billing, where refunds are lost, so production stays."


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
    memo: AuthoredCampaignAnalysis | None = None,
    bundle: Callable[[AnalysisContextBundle], AnalysisContextBundle] = lambda bundle: bundle,
) -> CampaignReport:
    measured = [await kind_fixture(label) for label in labels]
    campaign = joined(measured)
    bundles = {one.bundle.aspect: bundle(one.bundle) for one in measured}
    read = campaign_readings(campaign, bundles, arms=measured[0].arms())
    reports = {
        label: build_code_only_report(
            evidence, assembled_at=ASSEMBLED_AT, campaign_name="support", arm_labels=read.arm_labels()
        )
        for label, evidence in bundles.items()
    }
    return build_campaign_report(campaign, read, reports, memo, recommended_variant_key=recorded_key(read, memo))


def choice_memo(arm: str = "beta") -> AuthoredCampaignAnalysis:
    memo = two_aspect_memo()
    recommended = memo.recommended()
    assert recommended is not None
    recommendation = recommended.model_copy(update={"arm": arm, "reasons": ["beta costs a quarter as much."]})
    return memo.model_copy(update={"recommendation": [recommendation], "aspect_summaries": []})


def recommending_nothing(summary: str) -> AuthoredCampaignAnalysis:
    """The test memo recommending no arm, with ``summary`` read across its aspects."""
    return two_aspect_memo().model_copy(update={"recommendation": [], "aspect_summaries": [], "summary": summary})


__all__ = [
    "SUMMARY",
    "aspect_fixture",
    "campaign_fixture",
    "choice_memo",
    "first_screen_markdown",
    "header_html",
    "kind_fixture",
    "kind_page_of",
    "opening",
    "page_of",
    "recommending_nothing",
    "recorded_key",
    "two_aspect_memo",
]
