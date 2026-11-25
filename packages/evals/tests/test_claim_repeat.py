"""Repeating a stored claim grade: the A/A check and a second judge both reach the web step, and neither searches.

A first grade is made by the real claims steps over a scripted judge and a scripted search: one claim, nominated by
the judge's doubt and found true on a page. Its repeat, on the same judge or on a second one, keeps the claims as
listed, verifies them again, and asks the web verdict again over the windows the first grade stored, with no search
port anywhere in reach. A claim the repeat doubts that the first grade never checked has no window to read, so it
reads unchecked, never searched. A grade whose web results a recording served is not repeated: every check over it
reads not available (``replayed``). The text is invented: a reading room's almanac.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from threetears.evals.kernel.claims.grading import grade_criterion
from threetears.evals.kernel.claims.repeat import repeat_agreement, repeat_grade, replayed_state
from threetears.evals.schema import Segment, WebPage, WebSearchResult
from threetears.evals.schema.models import (
    ClaimCriterion,
    ClaimEvidence,
    ClaimGrade,
    PageBudget,
    RepairBudget,
    WebBudget,
)
from threetears.evals.schema.reading_states import StateOf

_FIRST = "The Lantern Almanac was first printed in 1921."
_SECOND = "Its pages were bound in green cloth."

_EVIDENCE = ClaimEvidence(
    segmentation="sentences-v1", segments=(Segment(text=_FIRST, unit=0), Segment(text=_SECOND, unit=1)), documents=()
)

_PAGE = WebPage(
    url="https://example.org/almanac",
    title="The Lantern Almanac",
    text="The Lantern Almanac was first printed in 1921 by Harrow Press. It ran to forty pages.",
)


def _listed(segment: int, text: str) -> dict[str, Any]:
    return {
        "segment": segment,
        "claim": text,
        "excerpt": text.rstrip("."),
        "specific": True,
        "hedged": False,
        "attributed": False,
        "scope": "world",
        "restates": None,
    }


def _truth(claim: int, doubted: bool) -> str:
    """The truth step's answer for one claim, asked in a batch of its own (each claim is its own unit)."""
    return json.dumps({"claims": [{"claim": claim, "citations": [], "suspect": doubted}]})


def _verdict(outcome: str) -> str:
    answer: dict[str, Any] = {"outcome": outcome, "ref": None, "quote": None, "subject": None, "value": None}
    if outcome != "inconclusive":
        answer.update(
            ref="W1",
            quote="The Lantern Almanac was first printed in 1921 by Harrow Press.",
            subject="The Lantern Almanac",
            value="1921",
        )
    return json.dumps(answer)


@dataclass(frozen=True)
class _Reply:
    content: str
    stop_reason: str = "end_turn"
    model: str = "scripted/judge-a"
    served_model: str | None = "scripted/judge-a"
    input_tokens: int | None = 300
    output_tokens: int | None = 60
    reasoning_tokens: int | None = None
    cost_usd: float | None = 0.001
    price_source: str | None = "scripted"
    temperature: float | None = None
    sent_settings: Any = None


# parity-with: threetears.evals.schema.completion.BoundCompletionClient
@dataclass
class _FakeJudge:
    """Answers each call by the reply schema it is sent, from its answers (a list repeats its last), and records each
    schema asked and each prompt sent."""

    answers: dict[str, list[str]]
    model_name: str = "scripted/judge-a"
    asked: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        del system, user, response_format
        return None

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        name = (response_format or {}).get("json_schema", {}).get("name", "")
        self.asked.append(name)
        self.prompts.append(user)
        answers = self.answers[name]
        answer = answers[min(self.asked.count(name), len(answers)) - 1]
        return _Reply(content=answer, model=self.model_name, served_model=self.model_name)

    async def aclose(self) -> None:
        """Nothing to release."""


@dataclass
class _Search:
    """A search that answers every query with the almanac's page, and records each query; flagged as a recording's
    answer where ``replayed``."""

    replayed: bool = False
    queries: list[str] = field(default_factory=list)

    async def __call__(self, query: str) -> WebSearchResult:
        self.queries.append(query)
        return WebSearchResult(pages=(_PAGE,), usd=0.0, replayed=self.replayed)


_CRITERION = ClaimCriterion(
    name="almanac.claims",
    pages=PageBudget(per_document_chars=2000, total_chars=4000, window_chars=1000),
    repairs=RepairBudget(rounds=2, max_items_per_cell=4),
    web=WebBudget(
        max_queries_per_claim=2, max_checked_per_cell=2, results_per_query=2, page_chars=400, window_chars=200
    ),
    judge="builtin",
)


def _judge(
    model: str = "scripted/judge-a", *, verdict: str = "true", doubted: tuple[bool, ...] = (True, False)
) -> _FakeJudge:
    return _FakeJudge(
        answers={
            "claim_listing": [json.dumps({"claims": [_listed(0, _FIRST), _listed(1, _SECOND)]})],
            "claim_truth": [_truth(at, doubt) for at, doubt in enumerate(doubted)],
            "claim_web_queries": [json.dumps({"queries": ["Lantern Almanac first printed"]})],
            "claim_web_verdict": [_verdict(verdict)],
            "claim_web_repair": [_verdict(verdict)],
        },
        model_name=model,
    )


async def _first(*, replayed: bool = False) -> tuple[ClaimGrade, _Search]:
    search = _Search(replayed=replayed)
    graded = await grade_criterion(
        _CRITERION, _EVIDENCE, key=None, client=_judge(), graded={}, search=search, derivation=None
    )
    assert graded.grade.claims[0].verdict == "true_ungrounded" and graded.grade.web[0].windows
    return graded.grade, search


async def test_the_same_judges_repeat_asks_the_web_verdict_again_over_the_stored_windows_and_searches_nothing() -> None:
    first, search = await _first()
    judge = _judge()

    repeat = await repeat_grade(_CRITERION, _EVIDENCE, first, key=None, client=judge)

    assert search.queries == ["Lantern Almanac first printed"], "only the first grade searched"
    assert "claim_listing" not in judge.asked and "claim_web_queries" not in judge.asked
    assert judge.asked.count("claim_web_verdict") == 1
    assert first.web[0].windows[0].text in judge.prompts[-1], "the verdict is asked over the stored window"
    assert repeat.reread == (0,) and repeat.unsearched == ()
    assert repeat.grade is not None and repeat.grade.web[0].windows == first.web[0].windows
    agreement = repeat_agreement(first, repeat)
    assert agreement.truth == {(0, 0): 2}
    assert agreement.web == {("true", "true"): 1}


async def test_a_second_judge_reaches_the_web_step_and_its_disagreement_is_counted() -> None:
    first, search = await _first()
    second = _judge("scripted/judge-b", verdict="false")

    repeat = await repeat_grade(_CRITERION, _EVIDENCE, first, key=None, client=second)

    assert search.queries == ["Lantern Almanac first printed"]
    assert second.asked.count("claim_web_verdict") == 1 and "claim_web_queries" not in second.asked
    assert repeat.grade is not None
    assert repeat.grade.claims[0].verdict == "contradicted" and repeat.grade.claims[0].graded_against == "web"
    assert {call.served_model for call in repeat.grade.calls} == {"scripted/judge-b"}
    agreement = repeat_agreement(first, repeat)
    assert agreement.truth == {(0, 1): 1, (0, 0): 1}
    assert agreement.web == {("true", "false"): 1}


async def test_a_claim_the_repeat_doubts_that_was_never_checked_reads_unchecked_and_is_never_searched() -> None:
    first, search = await _first()
    judge = _judge(doubted=(True, True))

    repeat = await repeat_grade(_CRITERION, _EVIDENCE, first, key=None, client=judge)

    assert repeat.unsearched == (1,)
    assert repeat.grade is not None
    assert repeat.grade.claims[1].web == "skipped_cap" and repeat.grade.claims[1].verdict == "unverifiable"
    assert judge.asked.count("claim_web_verdict") == 1 and search.queries == ["Lantern Almanac first printed"]


async def test_a_claim_the_repeat_no_longer_doubts_is_decided_by_its_truth_step_with_no_web_call() -> None:
    first, _ = await _first()
    judge = _judge(doubted=(False, False))

    repeat = await repeat_grade(_CRITERION, _EVIDENCE, first, key=None, client=judge)

    assert repeat.reread == () and "claim_web_verdict" not in judge.asked
    assert repeat.grade is not None and repeat.grade.claims[0].verdict == "unverifiable"
    assert repeat_agreement(first, repeat).web == {}


async def test_a_replayed_grade_is_not_repeated_and_every_check_over_it_reads_not_available_replayed() -> None:
    first, _ = await _first(replayed=True)
    judge = _judge()

    repeat = await repeat_grade(_CRITERION, _EVIDENCE, first, key=None, client=judge)

    replayed = StateOf(state="not_available", reason="replayed")
    assert first.web_replayed and replayed_state(first) == replayed
    assert repeat.grade is None and repeat.state == replayed and judge.asked == []
    assert repeat_agreement(first, repeat).state == replayed


async def test_a_repeat_under_another_definition_is_refused_before_any_call() -> None:
    first, _ = await _first()
    judge = _judge()
    edited = _CRITERION.model_copy(update={"guidance": "Read years as printings."})

    with pytest.raises(ValueError, match="another definition measures another instrument"):
        await repeat_grade(edited, _EVIDENCE, first, key=None, client=judge)
    assert judge.asked == []
