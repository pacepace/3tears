"""Each arm's pricing is read off its production usage rows, reaches the options table, and moves no fingerprint.

The bundle's census reads, for each arm, how its production cost was priced (``ArmScoreboard.pricing``): the price
sources its priced rows name, the providers its rows name, and the calls counted and priced. The report input copies
it (``ScoreboardRow.pricing``), and the options table prints it beside the cost. Owner ruling (2026-10-28): the price
source is no part of the campaign fingerprint or any evidence digest ("I recognize the implications, but it's ok").
Contracts held here:

- **the census** reads pricing off the roles a production deployment pays for, never the judge's; a row that did not
  count its calls leaves the priced share uncounted; an arm assembled without pricing carries no key;
- **the report input** carries an arm's pricing, and an arm without it carries no key;
- **the options table** of a real assembled campaign shows each arm's price source, serving provider and priced share;
- **no fingerprint moves**: two bundles, and two campaign bundles, that differ only in pricing share their fingerprint
  and their evidence digest, a planted pair and a real assembled campaign alike, while a change to an evidence field
  still moves them; the report input's content address (``report_input_digest``, a snapshot's id) still tells them
  apart.
"""

from __future__ import annotations

from collections.abc import Sequence

from packages.evals.tests.fixtures.report_plans.dataset import kairo_bundle
from packages.evals.tests.run_facts_support import CASES, aspect_bundle, campaign_over, result_of, run_of
from threetears.evals.analysis.analyst.snapshot import analyst_input
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, ArmPricing, evidence_digest
from threetears.evals.analysis.campaign_bundle import CampaignBundle, campaign_evidence_digest
from threetears.evals.analysis.planned_analysis import report_input_digest
from threetears.evals.analysis.report.block_spec import block_look
from threetears.evals.analysis.report.plan_page.tables import arms_table
from threetears.evals.analysis.report.plan_page.words import ALL_PRICED_TEMPLATE, SERVED_BY_TEMPLATE
from threetears.evals.analysis.report_input import Pricing
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.analysis.report_plan_facts import fact_arms
from threetears.evals.schema.models import EvalResult, EvalRun, RoleUsage

#: The arms of the planted campaign: the control and the candidate.
BASE, FAST = "base", "fast"

#: The pricing planted on every arm of a stored bundle.
PLANTED = ArmPricing(sources=["openrouter:reported"], providers=["DeepInfra"], calls=10, priced_calls=9)


def _row_of(role: str, *, cost: float | None, calls: int | None, source: str = "", provider: str = "") -> RoleUsage:
    return RoleUsage(
        role=role, model=FAST, cost_usd=cost, price_source=source or None, call_count=calls, provider=provider or None
    )


def _trials(usage: Sequence[Sequence[RoleUsage]]) -> list[tuple[EvalRun, list[EvalResult]]]:
    """The control's trials as the support writes them; the candidate's on ``usage``, one row list per case."""
    base, fast = run_of(BASE), run_of(FAST)
    return [
        (base, [result_of(base, index) for index in range(CASES)]),
        (fast, [result_of(fast, index, usage=rows) for index, rows in enumerate(usage)]),
    ]


#: A candidate whose candidate calls are priced and whose external call is not, beside a judge no production pays.
_PARTLY = [
    [
        _row_of("candidate", cost=0.002, calls=3, source="openrouter:reported"),
        _row_of("external", cost=None, calls=1, provider="searx"),
        _row_of("judge", cost=0.5, calls=1, source="anthropic:published"),
    ]
    for _ in range(CASES)
]


def _priced(bundle: CampaignBundle, pricing: ArmPricing | None) -> CampaignBundle:
    """The campaign bundle with every arm's pricing on every aspect set to ``pricing``."""
    aspects = [
        aspect.model_copy(
            update={"scoreboard": [row.model_copy(update={"pricing": pricing}) for row in aspect.scoreboard]}
        )
        for aspect in bundle.aspects
    ]
    return bundle.model_copy(update={"aspects": aspects})


def _unpriced(bundle: AnalysisContextBundle) -> AnalysisContextBundle:
    return bundle.model_copy(
        update={"scoreboard": [row.model_copy(update={"pricing": None}) for row in bundle.scoreboard]}
    )


# --- the census --------------------------------------------------------------------------------------------------


def test_the_census_reads_an_arms_pricing_off_its_production_usage_rows() -> None:
    """Sources from the priced rows only, providers from every row, the calls and priced calls the rows count; the
    judge's rows, which no production deployment pays for, are no part of it."""
    trials = _trials(_PARTLY)
    _, bundle, _ = aspect_bundle(trials)
    (pricing,) = [row.pricing for row in bundle.scoreboard if row.pricing is not None and row.pricing.providers]
    assert pricing == ArmPricing(
        sources=["openrouter:reported"], providers=["searx"], calls=4 * CASES, priced_calls=3 * CASES
    )
    built = report_input(campaign_over(trials))
    (row,) = [row for row in built.scoreboard.values() if row.pricing is not None and row.pricing.providers]
    assert row.pricing == Pricing(
        sources=["openrouter:reported"], providers=["searx"], calls=4 * CASES, priced_calls=3 * CASES
    )
    table = arms_table(built, block_look(built, fact_arms(built)), ())
    assert any(
        cell == f"{3 * CASES} of {4 * CASES} calls priced" for cell in (r["cost_per_1k"].value for r in table.rows)
    )


def test_a_row_that_did_not_count_its_calls_leaves_the_priced_share_uncounted() -> None:
    usage = [[_row_of("candidate", cost=0.002, calls=None, source="openrouter:reported")] for _ in range(CASES)]
    _, bundle, _ = aspect_bundle(_trials(usage))
    priced = [row.pricing for row in bundle.scoreboard if row.pricing is not None and row.pricing.calls is None]
    assert priced and priced[0].priced_calls is None and priced[0].sources == ["openrouter:reported"]


def test_an_arms_pricing_reaches_the_report_input_and_an_arm_without_it_carries_no_key() -> None:
    bundle = kairo_bundle()
    plain = analyst_input(bundle)
    assert all(row.pricing is None for row in plain.scoreboard.values())
    assert '"pricing"' not in plain.model_dump_json()
    priced = analyst_input(_priced(bundle, PLANTED))
    for row in priced.scoreboard.values():
        assert row.pricing == Pricing(
            sources=["openrouter:reported"], providers=["DeepInfra"], calls=10, priced_calls=9
        )


# --- the options table, end to end ---------------------------------------------------------------------------------


def test_the_options_table_of_an_assembled_campaign_shows_each_arms_source_provider_and_priced_share() -> None:
    usage = [
        [_row_of("candidate", cost=0.002, calls=2, source="openrouter:reported", provider="DeepInfra")]
        for _ in range(CASES)
    ]
    built = report_input(campaign_over(_trials(usage)))
    table = arms_table(built, block_look(built, fact_arms(built)), ())
    assert "priced_by" in {column.key for column in table.columns}
    by_option = {row["option"].value: row for row in table.rows}
    (fast,) = [row for option, row in by_option.items() if option is not None and FAST in str(option)]
    assert fast["priced_by"].value == "openrouter:reported"
    assert fast["priced_by"].flag == SERVED_BY_TEMPLATE.format(providers="DeepInfra")
    assert ALL_PRICED_TEMPLATE.format(calls=2 * CASES) in (fast["cost_per_1k"].flag or "")


# --- no fingerprint moves (owner ruling, 2026-10-28) ---------------------------------------------------------------


def test_two_bundles_differing_only_in_pricing_share_their_fingerprints_and_evidence_digests() -> None:
    plain = kairo_bundle()
    priced = _priced(plain, PLANTED)
    other = _priced(plain, ArmPricing(sources=["anthropic:published"], providers=[], calls=4, priced_calls=4))
    for campaign in (priced, other):
        assert campaign.fingerprint() == plain.fingerprint()
        assert campaign_evidence_digest(campaign.to_dict()) == campaign_evidence_digest(plain.to_dict())
        for held, bare in zip(campaign.aspects, plain.aspects, strict=True):
            assert held.fingerprint() == bare.fingerprint()
            assert evidence_digest(held.to_dict()) == evidence_digest(bare.to_dict())
    assert report_input_digest(analyst_input(priced)) != report_input_digest(analyst_input(plain)), (
        "a snapshot's content address still tells two pricings apart"
    )


def test_a_change_to_an_evidence_field_still_moves_the_fingerprints() -> None:
    plain = _priced(kairo_bundle(), PLANTED)
    aspect = plain.aspects[0]
    first, *rest = aspect.scoreboard
    moved_aspect = aspect.model_copy(
        update={"scoreboard": [first.model_copy(update={"passed": first.passed + 1}), *rest]}
    )
    moved = plain.model_copy(update={"aspects": [moved_aspect, *plain.aspects[1:]]})
    assert moved_aspect.fingerprint() != aspect.fingerprint()
    assert evidence_digest(moved_aspect.to_dict()) != evidence_digest(aspect.to_dict())
    assert moved.fingerprint() != plain.fingerprint()
    assert campaign_evidence_digest(moved.to_dict()) != campaign_evidence_digest(plain.to_dict())


def test_a_real_assembled_campaign_fingerprints_as_it_would_without_its_census_pricing() -> None:
    trials = _trials(_PARTLY)
    _, bundle, _ = aspect_bundle(trials)
    assert any(row.pricing is not None for row in bundle.scoreboard), "the census read pricing"
    bare = _unpriced(bundle)
    assert bundle.fingerprint() == bare.fingerprint()
    assert evidence_digest(bundle.to_dict()) == evidence_digest(bare.to_dict())

    campaign = campaign_over(trials)
    stripped = campaign.model_copy(update={"aspects": [_unpriced(aspect) for aspect in campaign.aspects]})
    assert campaign.fingerprint() == stripped.fingerprint()
    assert campaign_evidence_digest(campaign.to_dict()) == campaign_evidence_digest(stripped.to_dict())
    assert '"pricing"' not in stripped.to_json() and '"pricing"' in campaign.to_json()
