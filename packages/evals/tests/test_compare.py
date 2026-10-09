"""``compare`` runs each candidate as an arm, designates the control through the engine, and reports the contrasts.

Two classifiers over one case list: ``careful`` reads every case right and ``hasty`` misreads the
cases that carry a second label's words. Compared with ``hasty`` as the control, the campaign
declares one axis at the two arm names, its control resolves to the variant key ``hasty``'s run
carries, and the code-only report tests ``careful`` against it and finds it improved — under the
campaign's name, with its id in the byline.

``examples/compare_two_prompts.py`` is run here with no API key, so it takes its offline path.
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from threetears.evals.analysis import DisclosureBlock, TableBlock, report_markdown, variant_key_of_run
from threetears.evals.contracts.host import CANDIDATE_MODEL_LEVER
from threetears.evals.quick import Comparison, callable_host, compare
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations

#: The example under test.
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "compare_two_prompts.py"

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


def _expected(case: Mapping[str, Any]) -> str:
    return str(case["label"])


async def _compare(**overrides: Any) -> Comparison:
    arguments: dict[str, Any] = {
        "candidates": {"hasty": hasty, "careful": careful},
        "expected": _expected,
        "control": "hasty",
        "scope_id": SCOPE,
        "k": 2,
    }
    arguments.update(overrides)
    candidates = arguments.pop("candidates")
    return await compare(CASES, candidates, **arguments)


def _comparisons(comparison: Comparison) -> TableBlock:
    (table,) = [
        block for block in comparison.report.blocks if isinstance(block, TableBlock) and block.name == "comparisons"
    ]
    return table


async def test_each_candidate_runs_as_one_arm_labelled_by_its_name() -> None:
    comparison = await _compare()
    assert list(comparison.arms) == ["hasty", "careful"]
    assert comparison.arms["careful"].candidate_model == "careful"
    assert comparison.arms["hasty"].candidate_model == "hasty"
    by_arm = {arm: {m.name: m.mean for m in summary.measures} for arm, summary in comparison.arms.items()}
    assert by_arm["careful"]["match"] == pytest.approx(1.0)
    assert by_arm["hasty"]["match"] == pytest.approx(0.5)


async def test_the_control_resolves_to_the_variant_the_control_arm_ran() -> None:
    """The declared control is the key the control run's observations carry, as the engine reads it."""
    comparison = await _compare()
    storage = comparison.host.storage
    campaign = storage.load_campaign(comparison.campaign_id, SCOPE)
    assert campaign is not None and campaign.declared_design is not None
    design = campaign.declared_design
    run_id = comparison.arms["hasty"].run_id
    assert design.control == variant_key_of_run(storage.query_eval_results_by_run(run_id, SCOPE))
    assert design.control != variant_key_of_run(
        storage.query_eval_results_by_run(comparison.arms["careful"].run_id, SCOPE)
    )
    (axis,) = design.axes
    assert axis.axis_id == CANDIDATE_MODEL_LEVER
    assert [value.display for value in axis.values] == ["hasty", "careful"]
    assert design.intended_repetitions == 2
    assert sorted(campaign.run_ids) == sorted(summary.run_id for summary in comparison.arms.values())


async def test_every_other_arm_is_tested_against_the_control_and_the_better_one_separates() -> None:
    comparison = await _compare()
    rows = _comparisons(comparison).rows
    by_reading = {row["reading"]: row for row in rows}
    accuracy = by_reading["accuracy"]
    assert (accuracy["contrast"], accuracy["control"]) == ("model=careful", "model=hasty")
    assert accuracy["verdict"] == "improved on the control"
    assert accuracy["delta"] == pytest.approx(0.5)
    assert {row["control"] for row in rows} == {"model=hasty"}
    markdown = comparison.render()
    assert markdown == report_markdown(comparison.report)
    assert "No control resolved" not in markdown
    assert "model=hasty (control)" in markdown


async def test_the_report_is_titled_by_the_campaign_s_name_and_names_its_id_below() -> None:
    comparison = await _compare()
    assert comparison.name == "hasty vs careful"
    assert comparison.report.source.campaign_name == "hasty vs careful"
    lines = comparison.render().splitlines()
    assert lines[0] == "# Campaign hasty vs careful: its evidence, with no analysis"
    assert f"Code-only report of campaign {comparison.campaign_id}" in lines[2]

    named = await _compare(name="triage rules", scope_id="compare-named")
    assert named.render().startswith("# Campaign triage rules: its evidence, with no analysis\n")


async def test_a_report_without_the_campaign_s_name_is_titled_by_its_id() -> None:
    comparison = await _compare()
    report = comparison.report
    nameless = report.model_copy(update={"source": report.source.model_copy(update={"campaign_name": None})})
    assert report_markdown(nameless).startswith(
        f"# Campaign {comparison.campaign_id}: its evidence, with no analysis\n"
    )


async def test_the_arms_share_a_host_the_caller_hands_over() -> None:
    host = callable_host()
    comparison = await _compare(host=host)
    assert comparison.host is host
    assert host.storage.load_campaign(comparison.campaign_id, SCOPE) is not None


async def test_every_arm_is_started_in_one_launch_so_the_report_discloses_no_separate_starts() -> None:
    """The arms are one launch group, started together, so their runs interleave and no timing disclosure fires.

    Run one after another, each arm was its own launch over its own window, and the code-only report said
    so twice: the runs "were not started as one launch", and they "were measured over non-overlapping spans".
    """
    comparison = await _compare()
    runs = [comparison.host.storage.load_eval_run(summary.run_id, SCOPE) for summary in comparison.arms.values()]
    groups = {run.launch_group_id for run in runs if run is not None}
    assert len(groups) == 1 and None not in groups
    disclosures = [block.text for block in comparison.report.blocks if isinstance(block, DisclosureBlock)]
    assert disclosures
    assert not [text for text in disclosures if "not started as one launch" in text]
    assert not [text for text in disclosures if "non-overlapping spans" in text]


async def test_an_explicit_intent_is_the_one_template_every_arm_shares() -> None:
    comparison = await _compare(intent="Route each ticket to its queue.")
    (template_id,) = {summary.template_id for summary in comparison.arms.values()}
    assert comparison.host.storage.load_template(template_id or "", SCOPE).intent == "Route each ticket to its queue."


async def test_with_no_intent_arms_whose_docstrings_differ_share_the_generic_one() -> None:
    comparison = await _compare()
    (template_id,) = {summary.template_id for summary in comparison.arms.values()}
    intent = comparison.host.storage.load_template(template_id or "", SCOPE).intent
    assert intent == "Answer each case so that every scorer grades the answer well."


async def test_arms_whose_candidates_share_a_docstring_share_its_first_line_as_the_intent() -> None:
    comparison = await _compare(candidates={"a": careful, "b": careful}, control="a")
    (template_id,) = {summary.template_id for summary in comparison.arms.values()}
    intent = comparison.host.storage.load_template(template_id or "", SCOPE).intent
    assert intent == "Billing whenever money is mentioned, else a bug."


@pytest.mark.parametrize(
    ("candidates", "control", "refusal"),
    [
        ({"only": careful}, "only", "at least two candidates"),
        ({"hasty": hasty, "careful": careful}, "neither", "names no arm"),
        ({"hasty": hasty, " ": careful}, "hasty", "is blank"),
    ],
)
async def test_unusable_arms_are_refused_before_anything_runs(
    candidates: dict[str, Any], control: str, refusal: str
) -> None:
    host = callable_host()
    with pytest.raises(ValueError, match=refusal):
        await _compare(candidates=candidates, control=control, host=host)
    assert host.storage.list_campaigns(SCOPE) == []


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("compare_two_prompts_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_the_example_runs_offline_and_prints_the_verdict(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no API key the example takes its keyword stand-ins, says so, and ends on the contrast's verdict."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    comparison = await _load(EXAMPLE).main()
    assert isinstance(comparison, Comparison)
    assert comparison.control == "baseline"
    assert all(summary.status == "completed" for summary in comparison.arms.values())
    out = capsys.readouterr().out
    assert out.startswith("ANTHROPIC_API_KEY is not set: running OFFLINE")
    assert "(offline)" in comparison.name
    accuracy = {row["reading"]: row for row in _comparisons(comparison).rows}["accuracy"]
    assert (accuracy["contrast"], accuracy["verdict"]) == ("model=candidate", "improved on the control")
    assert re.search(r"\ncandidate vs baseline on accuracy: \+0\.42 \(p=[\d.e-]+\): improved on the control\n", out)
    assert comparison.render() not in out


def test_the_example_reaches_the_engine_only_through_public_roots() -> None:
    assert public_root_violations(SOURCE_ROOT, [(EXAMPLE.name, EXAMPLE)], consumer_root=REPO_ROOT) == []
