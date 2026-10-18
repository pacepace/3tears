"""A judge campaign's report never labels the subject's own spend measuring spend (#830).

In a judge campaign the judge under test is the subject and the only role billed, so ``cost_usd`` is exactly the
arm's production cost. Calling it "measuring spend" beside ``production_replicating_cost`` put two labels on one figure.
Any other kind still reads ``cost_usd`` as the measuring spend it is: the judge's and simulator's beside the candidate's.
"""

from __future__ import annotations

import pytest

from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from threetears.evals.analysis.bundle.assemble import assemble_context_bundle
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.kernel.judge_cases import JUDGE_KIND
from threetears.evals.kernel.metrics import MEASURING_SPEND_MEASURES, measure_title

_SPEND = "cost_usd"


def _bundle(subject_kind: str) -> AnalysisContextBundle:
    profile = toyhost_profile()
    campaign, storage = toyhost_campaign(profile=profile)
    return assemble_context_bundle(
        campaign.model_copy(update={"subject_kind": subject_kind}), storage=storage, profile=profile
    )


def test_a_judge_campaigns_catalog_labels_no_spend_measuring_spend() -> None:
    catalog = _bundle(JUDGE_KIND).measure_catalog
    assert catalog["production_replicating_cost"].reader_name == "Production cost"
    assert _SPEND in catalog
    assert [
        name for name, measure in catalog.items() if "measuring spend" in (measure.reader_name or "").casefold()
    ] == []


def test_a_judge_campaigns_surface_headings_carry_no_measuring_spend_label() -> None:
    surface = bundle_decision_surface(_bundle(JUDGE_KIND))
    assert surface.measure_heading(_SPEND) == "Spend, all roles"


def test_another_kinds_spend_is_still_measuring_spend() -> None:
    bundle = _bundle("agent")
    assert bundle.measure_catalog[_SPEND].reader_name == "Measuring spend, all roles"
    assert bundle_decision_surface(bundle).measure_heading(_SPEND) == "Measuring spend, all roles"


@pytest.mark.parametrize("name", sorted(MEASURING_SPEND_MEASURES))
def test_a_measure_titled_without_a_name_is_still_labelled_measuring_spend(name: str) -> None:
    assert measure_title(name).endswith(", measuring spend")
