"""A judge campaign's report never labels the subject's own spend measuring spend (#830).

In a campaign of judge-kind results the judge under test is the subject and the only role billed, so ``cost_usd`` is
exactly the arm's production cost. Calling it "measuring spend" beside ``production_replicating_cost`` put two labels on
one figure. The campaign is judge-kind by its results' ``candidate_kind`` (the field the cost measure keys on), not by
the free ``subject_kind`` string. Any other campaign, and a mixed one, still reads ``cost_usd`` as the measuring spend
it is: the judge's and simulator's beside the candidate's.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from packages.evals.tests.g2_support import CASES, _queue
from packages.evals.tests.off_loop import off_loop
from threetears.evals import Eval, Result
from threetears.evals.analysis import (
    AnalysisContextBundle,
    assemble_context_bundle,
    build_code_only_report,
    get_campaign,
    report_html,
    report_markdown,
)
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.kernel.judge_cases import JUDGE_KIND
from threetears.evals.kernel.metrics import MEASURING_SPEND_MEASURES, measure_title
from threetears.evals.schema import RoleUsage

_SPEND = "cost_usd"


async def _run() -> Result:
    return await Eval(CASES, {"a": _queue, "b": _queue}, ["label"], k=1, name="Judge spend").run(
        store=":memory:", scope="quick"
    )


def _bundle(comparison: Result, kinds: Mapping[str, str], subject_kind: str | None = None) -> AnalysisContextBundle:
    """The campaign's bundle once each arm's stored results are of the kind ``kinds`` names, billed to its one role."""
    storage = comparison.host.storage
    for name, kind in kinds.items():
        role = "judge" if kind == JUDGE_KIND else "candidate"
        for result in storage.query_eval_results_by_run(comparison.arms[name].run_id, comparison.scope_id):
            billed = RoleUsage(role=role, model=None, served_model="m", cost_usd=0.01, call_count=1)
            storage.replace_eval_result(
                result.model_copy(
                    update={
                        "candidate_kind": kind,
                        "cost_usd": 0.01,
                        "termination": "completed",
                        "cost_roles": [role],
                        "usage": [billed],
                    }
                ),
                if_match=None,
            )
    campaign = get_campaign(storage, comparison.campaign_id, comparison.scope_id)
    if subject_kind is not None:
        campaign = campaign.model_copy(update={"subject_kind": subject_kind})
    return assemble_context_bundle(campaign, storage=storage, profile=comparison.host.profile)


async def test_a_judge_campaign_labels_no_spend_measuring_spend_and_its_report_says_none() -> None:
    comparison = await _run()
    bundle = off_loop(_bundle, comparison, {"a": JUDGE_KIND, "b": JUDGE_KIND})

    catalog = bundle.measure_catalog
    assert catalog["production_replicating_cost"].reader_name == "Production cost"
    assert catalog[_SPEND].reader_name == "Spend, all roles"
    assert [n for n, m in catalog.items() if "measuring spend" in (m.reader_name or "").casefold()] == []
    assert bundle_decision_surface(bundle).measure_heading(_SPEND) == "Spend, all roles"
    report = build_code_only_report(bundle, assembled_at="2026-10-14T01:31:58+00:00", campaign_name="Judge spend")
    for page in (report_markdown(report), report_html(report)):
        assert "measuring spend" not in page.casefold()


async def test_the_free_subject_kind_string_decides_nothing() -> None:
    comparison = await _run()
    named = off_loop(_bundle, comparison, {"a": "agent", "b": "agent"}, JUDGE_KIND)
    assert named.measure_catalog[_SPEND].reader_name == "Measuring spend, all roles"


async def test_another_kinds_spend_is_still_measuring_spend() -> None:
    comparison = await _run()
    bundle = off_loop(_bundle, comparison, {"a": "agent", "b": "agent"})
    assert bundle_decision_surface(bundle).measure_heading(_SPEND) == "Measuring spend, all roles"


async def test_a_campaign_mixing_judge_kind_results_with_others_keeps_the_label() -> None:
    comparison = await _run()
    bundle = off_loop(_bundle, comparison, {"a": JUDGE_KIND, "b": "agent"})
    assert bundle.measure_catalog[_SPEND].reader_name == "Measuring spend, all roles"
    assert bundle_decision_surface(bundle).measure_heading(_SPEND) == "Measuring spend, all roles"


@pytest.mark.parametrize("name", sorted(MEASURING_SPEND_MEASURES))
def test_measure_title_labels_measuring_spend_unless_told_it_is_the_subjects(name: str) -> None:
    assert measure_title(name).endswith(", measuring spend")
    assert measure_title(name, "Spend", subject_spend=True) == "Spend"
