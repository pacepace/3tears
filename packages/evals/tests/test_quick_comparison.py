"""A mapping of runners is a comparison: each runner is an arm, the first key is the control, and the report contrasts.

Two classifiers over one case list: ``careful`` reads every case right and ``hasty`` misreads the
cases that carry a second label's words. Compared with ``hasty`` as the control (the mapping's first key), the campaign
declares one axis at the two arm names, its control resolves to the variant key ``hasty``'s run
carries, and the code-only report tests ``careful`` against it and finds it improved — under the
campaign's name, or by its id when it has none. Two prompts compared on a model's recorded answers are rung 3
(``examples/3-invoices/``), held by ``test_invoices_rung.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals.analysis import (
    DisclosureBlock,
    TableBlock,
    build_code_only_report,
    report_markdown,
    variant_key_of_run,
)
from threetears.evals import Eval, Result
from threetears.evals.quick import as_host
from threetears.evals.quick.one_call import ARM_LEVER, SHARED_ARM_MODEL
from packages.evals.tests.off_loop import off_loop

SCOPE = "compare-test"

#: Eight cases; the last four carry a second label's word, which only ``careful`` reads past.
CASES = [
    {"text": "refund please", "label": "billing"},
    {"text": "charged twice", "label": "billing"},
    {"text": "app crashes", "label": "bug"},
    {"text": "button broken", "label": "bug"},
    {"text": "charged, but the page crashes", "label": "billing"},
    {"text": "refund page crashes", "label": "billing"},
    {"text": "crash after refund", "label": "billing"},
    {"text": "charged then crashed", "label": "billing"},
]


async def careful(case: Mapping[str, Any]) -> str:
    """Billing whenever money is mentioned, else a bug."""
    text = case["text"]
    return "billing" if "refund" in text or "charged" in text else "bug"


async def hasty(case: Mapping[str, Any]) -> str:
    """A bug whenever anything crashed, else billing."""
    text = case["text"]
    return "bug" if "crash" in text or "broken" in text else "billing"


async def _compare(
    runners: Mapping[str, Any] | None = None,
    *,
    host: Any = None,
    scope: str = SCOPE,
    name: str | None = None,
) -> Result:
    """Compare the runners (``hasty``, the control, then ``careful``), each case graded on its ``label`` field."""
    arms = {"hasty": hasty, "careful": careful} if runners is None else runners
    ev = Eval(CASES, arms, ["label"], k=2, name=name)
    if host is not None:
        return await ev.run(host=host, scope=scope)
    return await ev.run(store=":memory:", scope=scope)


def _comparisons(comparison: Result, name: str = "comparisons") -> TableBlock:
    (table,) = [block for block in comparison.report.blocks if isinstance(block, TableBlock) and block.name == name]
    return table


async def test_each_candidate_runs_as_one_arm_labelled_by_its_name() -> None:
    comparison = await _compare()
    assert list(comparison.arms) == ["hasty", "careful"]
    # The names are arms, not models: each is stated on its run as the arm lever, every arm at one model.
    assert {summary.candidate_model for summary in comparison.arms.values()} == {SHARED_ARM_MODEL}
    assert [summary.arm for summary in comparison.arms.values()] == ["hasty", "careful"]
    assert (
        comparison.arms["careful"]
        .render()
        .startswith(f"run {comparison.arms['careful'].run_id} completed: arm careful (model {SHARED_ARM_MODEL})")
    )
    by_arm = {arm: {m.name: m.mean for m in summary.measures} for arm, summary in comparison.arms.items()}
    assert by_arm["careful"]["match"] == pytest.approx(1.0)
    assert by_arm["hasty"]["match"] == pytest.approx(0.5)


async def test_the_control_resolves_to_the_variant_the_control_arm_ran() -> None:
    """The declared control is the key the control run's observations carry, as the engine reads it."""
    comparison = await _compare()
    storage = comparison.host.storage
    campaign = off_loop(storage.load_campaign, comparison.campaign_id, SCOPE)
    assert campaign is not None and campaign.declared_design is not None
    design = campaign.declared_design
    run_id = comparison.arms["hasty"].run_id
    assert design.control == variant_key_of_run(off_loop(storage.query_eval_results_by_run, run_id, SCOPE))
    assert design.control != variant_key_of_run(
        off_loop(storage.query_eval_results_by_run, comparison.arms["careful"].run_id, SCOPE)
    )
    (axis,) = design.axes
    assert axis.axis_id == ARM_LEVER
    assert [value.display for value in axis.values] == ["hasty", "careful"]
    assert design.intended_repetitions == 2
    assert sorted(campaign.run_ids) == sorted(summary.run_id for summary in comparison.arms.values())


async def test_every_other_arm_is_tested_against_the_control_and_the_better_one_separates() -> None:
    comparison = await _compare()
    rows = comparison.contrasts()
    by_reading = {row["reading"]: row for row in rows}
    accuracy = by_reading["Accuracy"]
    assert (accuracy["contrast"], accuracy["control"]) == ("careful", "hasty")
    assert accuracy["verdict"] == "careful better than hasty"
    assert accuracy["delta"] == pytest.approx(50.0), "a rate's change, in points"
    assert {row["control"] for row in rows} == {"hasty"}
    assert [row["reading"] for row in _comparisons(comparison).values()] == [row["reading"] for row in rows]
    markdown = report_markdown(comparison.report)
    assert comparison.render(full=True).endswith(markdown)
    assert "No control resolved" not in markdown
    assert "hasty (control) vs careful" in markdown


async def test_the_report_is_titled_by_the_question_its_campaign_asks() -> None:
    comparison = await _compare()
    assert comparison.name == "hasty vs careful"
    assert comparison.report.source.campaign_name == "hasty vs careful"
    lines = report_markdown(comparison.report).splitlines()
    # A name that only lists the arms adds nothing to the question it asks.
    assert lines[0] == "# Does careful do better than hasty?"
    assert lines[1].startswith("hasty (control) vs careful · ")

    named = await _compare(name="triage rules", scope="compare-named")
    assert report_markdown(named.report).startswith("# triage rules: does careful do better than hasty?\n")


async def test_a_report_without_the_campaign_s_name_is_titled_by_its_id() -> None:
    comparison = await _compare()
    report = comparison.report
    nameless = build_code_only_report(report.bundle, assembled_at=report.source.generated_at)
    assert report_markdown(nameless).startswith(f"# Campaign {comparison.campaign_id}: does careful do better")


async def test_the_arms_share_a_host_the_caller_hands_over() -> None:
    host = as_host([Eval(CASES, {"hasty": hasty, "careful": careful}, ["label"])], store=":memory:")
    comparison = await _compare(host=host)
    assert comparison.host is host
    assert off_loop(host.storage.load_campaign, comparison.campaign_id, SCOPE) is not None


async def test_every_arm_is_started_in_one_launch_so_the_report_discloses_no_separate_starts() -> None:
    """The arms are one launch group, started together, so their runs interleave and no timing disclosure fires.

    Run one after another, each arm was its own launch over its own window, and the code-only report said
    so twice: the runs "were not started as one launch", and they "were measured over non-overlapping spans".
    """
    comparison = await _compare()
    runs = [
        off_loop(comparison.host.storage.load_eval_run, summary.run_id, SCOPE) for summary in comparison.arms.values()
    ]
    groups = {run.launch_group_id for run in runs if run is not None}
    assert len(groups) == 1 and None not in groups
    disclosures = [block.text for block in comparison.report.blocks if isinstance(block, DisclosureBlock)]
    assert disclosures
    assert not [text for text in disclosures if "not started as one launch" in text]
    assert not [text for text in disclosures if "non-overlapping spans" in text]


async def test_a_docstring_every_arm_shares_is_the_intent_of_the_one_template() -> None:
    async def routed_careful(case: Mapping[str, Any]) -> str:
        """Route each ticket to its queue."""
        return await careful(case)

    async def routed_hasty(case: Mapping[str, Any]) -> str:
        """Route each ticket to its queue."""
        return await hasty(case)

    comparison = await _compare({"hasty": routed_hasty, "careful": routed_careful})
    (template_id,) = {summary.template_id for summary in comparison.arms.values()}
    assert (
        off_loop(comparison.host.storage.load_template, template_id or "", SCOPE).intent
        == "Route each ticket to its queue."
    )


async def test_arms_whose_docstrings_differ_share_the_generic_intent() -> None:
    comparison = await _compare()
    (template_id,) = {summary.template_id for summary in comparison.arms.values()}
    intent = off_loop(comparison.host.storage.load_template, template_id or "", SCOPE).intent
    assert intent == "Answer each case so that every scorer grades the answer well."


async def test_arms_whose_candidates_share_a_docstring_share_its_first_line_as_the_intent() -> None:
    comparison = await _compare({"a": careful, "b": careful})
    (template_id,) = {summary.template_id for summary in comparison.arms.values()}
    intent = off_loop(comparison.host.storage.load_template, template_id or "", SCOPE).intent
    assert intent == "Billing whenever money is mentioned, else a bug."


@pytest.mark.parametrize(
    ("runners", "refusal"),
    [
        ({}, "names no arm"),
        ({"hasty": hasty, " ": careful}, "is blank"),
        ({"hasty": hasty, "prompt=v2": careful}, "not both"),
    ],
)
async def test_unusable_arms_are_refused_before_anything_runs(runners: dict[str, Any], refusal: str) -> None:
    host = as_host([Eval(CASES, {"hasty": hasty, "careful": careful}, ["label"])], store=":memory:")
    with pytest.raises(ValueError, match=refusal):
        await _compare(runners, host=host)
    assert off_loop(host.storage.list_campaigns, SCOPE) == []


async def test_a_mapping_of_one_is_one_arm_named_not_a_comparison() -> None:
    result = await _compare({"only": careful})
    assert list(result.arms) == ["only"] and result.control == "only"
    assert result.summary.candidate_model == "only"
