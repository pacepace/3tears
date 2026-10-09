"""``compare(judge=)``, the contrast rows' ``arm`` key, the judge's temperature, and a judge grading its own model.

- **A judged A/B from the quick path.** Every arm is judged by ONE judge — one model, one rubric, one
  temperature — and the report contrasts every arm against the control on each judged dimension.
- **Every contrast row names its arm by the caller's own key**, matched by variant key rather than read off
  the report's words: a name for a single-factor comparison, a tuple of levels for a factorial one.
- **The judge's temperature reaches the caller's client.** The engine asks for 0; a client whose ``generate``
  takes ``temperature`` is handed it, and each score records what the client says it sent.
- **A judge whose model also answered** is disclosed, in the summary and in a judged comparison's report.

The judge is scripted: it scores an answer that says "good" 5 and anything else 2, and reports the
temperature it was handed as the one it sent.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

import pytest

from threetears.evals.analysis import DisclosureBlock, report_markdown
from threetears.evals.contracts import StopReason
from threetears.evals.quick import Answer, Judge, callable_host, compare, run_eval
from threetears.evals.run import get_run, list_results

SCOPE = "quick-judged-compare-tests"
JUDGE_MODEL = "scripted/judge"
CASES = [{"question": f"q{n}", "hard": n % 2 == 1} for n in range(12)]


@dataclass(frozen=True)
class _Reply:
    """One scripted judge reply, in the attribute names the engine reads."""

    content: str
    temperature: float | None
    input_tokens: int | None = 20
    output_tokens: int | None = 10
    cost_usd: float | None = 0.0001
    price_source: str | None = "scripted rate"
    model: str = JUDGE_MODEL
    served_model: str | None = JUDGE_MODEL
    reasoning_tokens: int | None = None
    stop_reason: StopReason = "end_turn"


def _score(system: str, user: str) -> str:
    match = re.search(r'single key "([^"]+)"', system)
    assert match is not None, "the judge's system prompt names no dimension"
    output = user.split("# Output under review\n", 1)[1]
    good = "good" in output
    return json.dumps(
        {"reasoning": "resolves it" if good else "vague", "criteria_scores": {match[1]: 5 if good else 2}}
    )


# parity-with: threetears.evals.contracts.CompletionClient
class _FakeJudgeClient:
    """A judge client that takes the temperature it is asked for and reports it as the one it sent."""

    def __init__(self) -> None:
        self.temperatures: list[float | None] = []

    async def generate(
        self,
        *,
        system: str,
        user: str,
        response_format: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> _Reply:
        self.temperatures.append(temperature)
        return _Reply(content=_score(system, user), temperature=temperature)

    async def aclose(self) -> None:
        """Nothing to release."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


# parity-with: threetears.evals.contracts.CompletionClient
class _FakeFixedTemperatureClient:
    """A judge client whose ``generate`` takes no temperature: it sends none, and says so on each reply."""

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        return _Reply(content=_score(system, user), temperature=None)

    async def aclose(self) -> None:
        """Nothing to release."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


def _judge(client: Any) -> Judge:
    return Judge(client=client, model=JUDGE_MODEL, rubric={"helpful": "The answer resolves the question."})


async def vague(case: Mapping[str, Any]) -> str:
    return "it depends"


async def helpful(case: Mapping[str, Any]) -> str:
    return "it depends" if case["hard"] else "a good answer"


async def test_a_judged_compare_contrasts_every_arm_on_each_judged_dimension_with_one_judge() -> None:
    client = _FakeJudgeClient()
    comparison = await compare(
        CASES,
        {"baseline": vague, "candidate": helpful},
        judge=_judge(client),
        control="baseline",
        scope_id=SCOPE,
        k=1,
    )
    (row,) = comparison.contrasts("answer.helpful (judged)")
    assert row["arm"] == "candidate" and row["delta"] == pytest.approx(1.5)
    assert row["verdict"].startswith("improved")
    runs = [get_run(comparison.host.storage, summary.run_id, SCOPE) for summary in comparison.arms.values()]
    assert {run.judge_model for run in runs} == {JUDGE_MODEL}
    assert len({run.template_id for run in runs}) == 1, "one template, so one rubric, for every arm"
    assert all(summary.judged[0].name == "answer.helpful" for summary in comparison.arms.values())


async def test_a_judged_comparison_reads_against_another_control_on_the_judged_kinds_levers() -> None:
    comparison = await compare(
        CASES,
        {"baseline": vague, "candidate": helpful},
        judge=_judge(_FakeJudgeClient()),
        control="baseline",
        scope_id=SCOPE,
        k=1,
    )
    flipped = comparison.against("candidate")
    (row,) = flipped.contrasts("answer.helpful (judged)")
    assert row["arm"] == "baseline" and row["delta"] == pytest.approx(-1.5)


async def test_every_contrast_row_names_its_arm_by_the_callers_key() -> None:
    async def other(case: Mapping[str, Any]) -> str:
        return "a good answer"

    comparison = await compare(
        CASES,
        {"baseline": vague, "candidate": helpful, "other": other},
        judge=_judge(_FakeJudgeClient()),
        control="baseline",
        scope_id=SCOPE,
        k=1,
    )
    rows = comparison.contrasts()
    assert sorted(row["arm"] for row in rows) == ["candidate", "other"]
    assert {row["contrast"]: row["arm"] for row in rows} == {"model=candidate": "candidate", "model=other": "other"}


async def test_a_factorial_contrast_row_names_its_arm_by_its_tuple_of_levels() -> None:
    def says_good(case: Mapping[str, Any], answer: Any) -> bool:
        return "good" in str(answer)

    comparison = await compare(
        CASES,
        {("m", "v1"): vague, ("m", "v2"): helpful},
        [says_good],
        factors=("model", "prompt"),
        control=("m", "v1"),
        scope_id=SCOPE,
        k=1,
    )
    (row,) = comparison.contrasts("says_good")
    assert row["arm"] == ("m", "v2")


async def test_the_judges_requested_temperature_reaches_the_client_on_every_call() -> None:
    client = _FakeJudgeClient()
    await run_eval(CASES, helpful, judge=_judge(client), scope_id=SCOPE, k=1)
    assert len(client.temperatures) == len(CASES) and set(client.temperatures) == {0.0}


async def test_a_client_that_takes_no_temperature_is_called_without_one_and_its_scores_say_none_was_sent() -> None:
    host = callable_host()
    summary = await run_eval(
        CASES, helpful, judge=_judge(_FakeFixedTemperatureClient()), host=host, scope_id=SCOPE, k=1
    )
    stored = list_results(host.storage, summary.run_id, SCOPE)
    assert {score.judge_temperature for result in stored for score in result.rubric_scores} == {"model_default"}


async def test_a_client_that_takes_the_temperature_has_it_recorded_as_sent_on_every_score() -> None:
    host = callable_host()
    summary = await run_eval(CASES, helpful, judge=_judge(_FakeJudgeClient()), host=host, scope_id=SCOPE, k=1)
    stored = list_results(host.storage, summary.run_id, SCOPE)
    assert {score.judge_temperature for result in stored for score in result.rubric_scores} == {0.0}


async def answered_by_the_judges_model(case: Mapping[str, Any]) -> Answer:
    return Answer("a good answer", model=JUDGE_MODEL, input_tokens=5, output_tokens=5, cost_usd=0.0001)


async def test_a_judge_grading_answers_its_own_model_produced_is_disclosed_in_the_summary() -> None:
    summary = await run_eval(CASES, answered_by_the_judges_model, judge=_judge(_FakeJudgeClient()), scope_id=SCOPE, k=1)
    assert summary.judge_shares_candidate_model == [JUDGE_MODEL]
    assert f"self-judging: the judge's model {JUDGE_MODEL} also produced the candidate's answers" in summary.render()


async def test_a_judge_of_another_model_discloses_nothing() -> None:
    summary = await run_eval(CASES, helpful, judge=_judge(_FakeJudgeClient()), scope_id=SCOPE, k=1)
    assert summary.judge_shares_candidate_model == []
    assert "self-judging" not in summary.render()


async def test_a_judged_comparison_discloses_the_arm_whose_answers_the_judges_model_produced() -> None:
    comparison = await compare(
        CASES,
        {"baseline": vague, "same-model": answered_by_the_judges_model},
        judge=_judge(_FakeJudgeClient()),
        control="baseline",
        scope_id=SCOPE,
        k=1,
    )
    disclosures = [block.text for block in comparison.report.blocks if isinstance(block, DisclosureBlock)]
    (said,) = [text for text in disclosures if text.startswith("self-judging")]
    assert "arm same-model's answers" in said and "may favour that arm" in said
    assert said in report_markdown(comparison.report)
    assert comparison.arms["baseline"].judge_shares_candidate_model == []
