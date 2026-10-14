"""``examples/compare_two_efforts.py`` answers "is less effort good enough?" only as the evidence allows.

The package's headline question, so the example is held to the decision rule: "good enough" only on
``equivalent`` against the margin it declares; ``not separated`` means the cases could not tell the two settings
apart and never licenses the switch. With no API key the example replays its recording, so no test calls the API;
the arms differ in one request setting, so each replays its own calls from the one file. Nothing here assumes what
Claude answered: the decision printed is the one the typed verdict maps to.
"""

from __future__ import annotations

from types import ModuleType
from typing import get_args

import pytest

from threetears.evals.analysis import ContrastOutcome
from threetears.evals.quick import Comparison
from threetears.evals.testing import RecordedPrompt
from packages.evals.tests.example_loader import EXAMPLES, check_covers, load_example, replaying
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations

NAME = "compare_two_efforts.py"


def _load() -> ModuleType:
    return load_example(NAME)


async def test_with_no_api_key_it_replays_both_efforts_and_decides_by_the_typed_verdict(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load()
    asked = [RecordedPrompt(module.PROMPT, case["email"]) for case in module.CASES for _ in range(2)]
    for settings in module.EFFORTS.values():
        check_covers(module, asked, settings=settings, model=module.MODEL)
    replay = replaying(module, monkeypatch)

    comparison = await module.main()

    assert replay.served() == 48, "12 emails x 2 repeats x 2 efforts, as the docstring says"
    today, cheaper = module.EFFORTS
    assert isinstance(comparison, Comparison) and comparison.control == today
    assert list(comparison.arms) == [today, cheaper]
    for summary in comparison.arms.values():
        assert summary.status == "completed" and (summary.n_cases, summary.k_runs, summary.n_scored) == (12, 2, 24)
        assert summary.role_spend("candidate").calls == 24
    assert len({summary.candidate_model for summary in comparison.arms.values()}) == 1, "the arms share one model"
    # The margin is declared on accuracy, on both runs, so equivalence could be tested; no duplicate scorer is needed.
    assert "correct" not in comparison.host.profile.measures.names
    (verdict,) = comparison.verdicts("accuracy")
    assert (verdict.margin, verdict.margin_source) == (module.MARGIN, "run")
    (accuracy,) = comparison.contrasts("accuracy")
    assert accuracy["arm"] == cheaper and accuracy["outcome"] == verdict.outcome

    out = capsys.readouterr().out
    assert out.startswith("ANTHROPIC_API_KEY is not set: replaying recordings/compare_two_efforts.jsonl, Claude (")
    assert f"at {today} and at {cheaper} effort" in out.splitlines()[0]
    assert f"\nOn accuracy, {verdict.outcome.replace('_', ' ')}: {module.WHAT_TO_DO[verdict.outcome]}\n" in out
    for arm, summary in comparison.arms.items():
        spend = summary.role_spend("candidate")
        cost = "unpriced" if spend.cost_usd is None else f"${spend.cost_usd:.5f}"
        assert f"{arm} effort: 24 calls, candidate spend {cost}\n" in out


def test_each_verdict_has_its_decision_and_only_improved_or_equivalent_switch() -> None:
    module = _load()
    assert set(module.WHAT_TO_DO) == set(get_args(ContrastOutcome)), "one decision per typed contrast outcome"
    assert "good enough" in module.WHAT_TO_DO["equivalent"]
    assert "does not show low effort is good enough" in module.WHAT_TO_DO["not_separated"]
    assert "switch," not in module.WHAT_TO_DO["not_separated"]


def test_the_two_arms_differ_in_exactly_one_request_setting() -> None:
    module = _load()
    today, cheaper = (settings.model_dump() for settings in module.EFFORTS.values())
    assert {name for name in today if today[name] != cheaper[name]} == {"reasoning_effort"}


def test_the_example_reaches_the_engine_only_through_public_roots() -> None:
    assert public_root_violations(SOURCE_ROOT, [(NAME, EXAMPLES / NAME)], consumer_root=REPO_ROOT) == []
