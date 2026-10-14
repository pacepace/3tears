"""``examples/guardrails.py`` declares a guardrail, reads held and breached per arm, and never ships a breach.

The example is the quick path's guardrail rung: a prompt that answers more questions may repeat the card's digits,
and an arm that breached the guardrail is never the one to ship, whatever it gained. With no API key the example
replays its recording, so no test calls the API, and nothing here assumes what Claude answered: each arm's printed
decision is the one its typed outcomes map to, and every leak printed is a reply that failed the guardrail.
"""

from __future__ import annotations

from types import ModuleType

import pytest

from threetears.evals import Result
from threetears.evals.testing import RecordedPrompt
from packages.evals.tests.example_loader import EXAMPLES, check_covers, load_example, replaying
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations

NAME = "guardrails.py"


def _load() -> ModuleType:
    return load_example(NAME)


async def test_with_no_api_key_it_replays_and_an_arm_that_breached_is_never_the_one_to_ship(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load()
    asked = [
        RecordedPrompt(prompt, f"{case['record']}\n\nCustomer: {case['question']}")
        for prompt in module.PROMPTS.values()
        for case in module.CASES
    ]
    check_covers(module, asked, settings=module.SETTINGS, model=module.MODEL)
    replay = replaying(module, monkeypatch)

    comparison = await module.main()

    assert replay.served() == 150, "50 questions x 3 prompts, as the docstring says"
    assert isinstance(comparison, Result) and list(comparison.arms) == ["current", "friendlier", "careful"]
    for summary in comparison.arms.values():
        assert summary.status == "completed" and (summary.n_cases, summary.k_runs, summary.n_scored) == (50, 1, 50)
    assert {row["arm"] for row in comparison.contrasts("gives_status")} == {"friendlier", "careful"}
    assert comparison.contrasts("keeps_card_private") == [], "the guardrail is in no contrast"
    outcomes = {row["arm"]: row["outcome"] for row in comparison.guardrails()}
    assert set(outcomes) == {"friendlier", "careful"}
    assert set(outcomes.values()) <= {"held", "breached", "undecided"}

    out = capsys.readouterr().out
    assert out.startswith("ANTHROPIC_API_KEY is not set: replaying recordings/guardrails.jsonl, Claude (")
    for row in comparison.contrasts("gives_status"):
        standing = comparison.guardrail_standing(row["arm"])
        guardrail = "breached" if standing.breached else "undecided" if standing.undecided else "held"
        assert guardrail == outcomes[row["arm"]]
        decision = module.what_to_do(row["outcome"], guardrail)
        assert f"  {row['arm']}: {row['verdict']}; guardrail {guardrail}: {decision}\n" in out
        if guardrail == "breached":
            assert decision == "do not ship it, whatever it gained."
    for arm in comparison.arms:  # the example ends by reading a leak, when an arm leaked
        leaks = [result for result in comparison.results(arm) if result.scores.get("keeps_card_private") == 0]
        if leaks:
            assert f"  {arm}, order {leaks[0].input['order']}: {leaks[0].answer!r}\n" in out


def test_only_a_held_guardrail_with_a_gain_is_a_candidate_to_ship() -> None:
    module = _load()
    for outcome in ("improved", "not_separated", "regressed"):  # typed outcomes, never the printed words
        assert module.what_to_do(outcome, "breached") == "do not ship it, whatever it gained."
        assert "not known to be safe" in module.what_to_do(outcome, "undecided")
    assert module.what_to_do("improved", "held") == "a candidate to ship."
    assert "keep the current prompt" in module.what_to_do("not_separated", "held")
    assert "keep the current prompt" in module.what_to_do("improved on the control", "held"), "words are no outcome"


def test_the_example_reaches_the_engine_only_through_public_roots() -> None:
    assert public_root_violations(SOURCE_ROOT, [(NAME, EXAMPLES / NAME)], consumer_root=REPO_ROOT) == []
