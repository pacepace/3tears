"""Reading a quick run's answers: ``summary.results()`` and ``misses()``, case names, and a sync candidate.

A newcomer's first question after a run is which cases it got wrong. The summary carries every result —
the case, the answer, the grades and why it failed — so it can be read after ``run_eval`` returns, though
the default store was the call's own. Cases are called by their own ``id`` when they carry one, else by
their position, in the results and in every error line. A plain ``def`` candidate is called in a worker
thread; one handed tools must be async, and is refused before anything runs.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals.quick import UNUSABLE_ANSWER, callable_host, compare, run_eval, summarize_run

SCOPE = "quick-results-tests"

#: Three tickets; ``hasty`` files the last under the wrong queue.
CASES = [
    {"id": "refund", "text": "refund please", "queue": "billing"},
    {"text": "app crashes", "queue": "bug"},
    {"text": "charged, then it crashed", "queue": "billing"},
]


async def hasty(case: Mapping[str, Any]) -> str:
    return "bug" if "crash" in case["text"] else "billing"


def _queue(case: Mapping[str, Any]) -> str:
    return str(case["queue"])


async def test_every_result_reads_back_after_the_call_with_its_case_answer_and_grades() -> None:
    summary = await run_eval(CASES, hasty, expected=_queue, scope_id=SCOPE, k=2)
    results = summary.results()
    assert [(result.case, result.repeat) for result in results] == [
        ("refund", 1),
        ("refund", 2),
        ("1", 1),
        ("1", 2),
        ("2", 1),
        ("2", 2),
    ]
    first = results[0]
    assert first.input == CASES[0] and first.expected == "billing" and first.answer == "billing"
    assert first.outcome == "scored" and first.scores["match"] is True and not first.missed


async def test_misses_are_the_wrong_answers_each_saying_why() -> None:
    summary = await run_eval(CASES, hasty, expected=_queue, scope_id=SCOPE, k=1)
    (miss,) = summary.misses()
    assert miss.case == "2" and miss.answer == "bug" and miss.expected == "billing"
    assert miss.missed_because == ["answered 'bug', expected 'billing'"]
    assert miss.render().splitlines()[0] == "case 2 (repeat 1, scored): answered 'bug', expected 'billing'"


async def test_a_failed_candidate_is_a_miss_and_its_error_line_names_the_case_by_its_id() -> None:
    async def refuses_refunds(case: Mapping[str, Any]) -> str:
        if "refund" in case["text"]:
            raise RuntimeError("no refunds")
        return "bug"

    summary = await run_eval(CASES, refuses_refunds, expected=_queue, scope_id=SCOPE, k=1)
    assert summary.errors == ["case refund: the candidate raised RuntimeError: no refunds"]
    failed = next(result for result in summary.misses() if result.case == "refund")
    assert failed.outcome == "failed" and failed.answer is None
    assert failed.missed_because == ["failed: the candidate raised RuntimeError: no refunds"]
    assert failed.scores["confusion_cell"] and failed.scores["match"] is False
    assert UNUSABLE_ANSWER in str(failed.scores["confusion_cell"])


async def test_a_scorer_that_gives_zero_marks_a_miss_and_an_excluded_result_is_never_one() -> None:
    def short(case: Mapping[str, Any], answer: Any) -> bool:
        return len(str(answer)) <= 3

    def picky(case: Mapping[str, Any], answer: Any) -> float:
        if case["text"] == "app crashes":
            raise ValueError("cannot grade crashes")
        return 1.0

    summary = await run_eval(CASES, hasty, [short, picky], scope_id=SCOPE, k=1)
    by_case = {result.case: result for result in summary.results()}
    assert by_case["refund"].missed_because == ["short gave 0"]
    assert by_case["1"].outcome == "excluded" and not by_case["1"].missed
    assert by_case["1"].errors == ["the scorer picky raised ValueError: cannot grade crashes"]
    assert [result.case for result in summary.misses()] == ["refund"]


async def test_cases_without_ids_are_called_by_their_position() -> None:
    summary = await run_eval([{"n": 1}, {"n": 2}], hasty_n, [even], scope_id=SCOPE, k=1)
    assert [result.case for result in summary.results()] == ["0", "1"]
    assert [result.case for result in summary.misses()] == ["0"]


async def hasty_n(case: Mapping[str, Any]) -> int:
    return int(case["n"])


def even(case: Mapping[str, Any], answer: Any) -> bool:
    return answer % 2 == 0


@pytest.mark.parametrize(
    ("cases", "said"),
    [
        ([{"id": "a"}, {"id": "a"}], "more than one case is called 'a'"),
        ([{"id": 1}, {}], "more than one case is called '1'"),
        ([{"id": ""}], "case 0's id is ''"),
        ([{"id": True}], "case 0's id is True"),
        ([{"id": {"x": 1}}], "case 0's id is {'x': 1}"),
    ],
)
async def test_case_ids_that_cannot_name_one_case_each_are_refused_before_anything_runs(
    cases: list[dict[str, Any]], said: str
) -> None:
    host = callable_host([even])
    with pytest.raises(ValueError, match=said.replace("(", r"\(").replace("{", r"\{")):
        await run_eval(cases, hasty_n, [even], host=host, scope_id=SCOPE, k=1)
    assert host.storage.query_templates(SCOPE) == []


async def test_an_expected_label_refusal_names_the_case_by_its_id() -> None:
    with pytest.raises(ValueError, match="expected= gave case refund"):
        await run_eval(CASES, hasty, expected=lambda case: "" if "id" in case else "bug", scope_id=SCOPE, k=1)


async def test_a_summary_read_back_without_its_cases_says_so_rather_than_reading_as_no_misses() -> None:
    host = callable_host()
    summary = await run_eval(CASES, hasty, expected=_queue, host=host, scope_id=SCOPE, k=1)
    read_back = summarize_run(host, summary.run_id, SCOPE)
    assert read_back.case_results is None
    with pytest.raises(ValueError, match="carries no per-case results"):
        read_back.misses()


async def test_a_comparison_reads_each_arms_results_and_misses_by_its_key() -> None:
    async def careful(case: Mapping[str, Any]) -> str:
        return "billing" if "refund" in case["text"] or "charged" in case["text"] else "bug"

    comparison = await compare(
        CASES, {"hasty": hasty, "careful": careful}, expected=_queue, control="hasty", scope_id=SCOPE, k=1
    )
    assert [result.case for result in comparison.misses("hasty")] == ["2"]
    assert comparison.misses("careful") == []
    assert len(comparison.results("careful")) == 3
    with pytest.raises(ValueError, match="arm 'nobody' names no arm"):
        comparison.results("nobody")


async def test_a_sync_candidate_runs_in_a_worker_thread() -> None:
    threads: set[int] = set()

    def blocking(case: Mapping[str, Any]) -> str:
        threads.add(threading.get_ident())
        return "bug" if "crash" in case["text"] else "billing"

    summary = await run_eval(CASES, blocking, expected=_queue, scope_id=SCOPE, k=1)
    assert summary.n_scored == 3 and summary.n_candidate_failed == 0
    assert threading.get_ident() not in threads


async def test_a_sync_callable_returning_a_coroutine_has_it_awaited() -> None:
    summary = await run_eval(CASES, lambda case: hasty(case), expected=_queue, model="wrapped", scope_id=SCOPE, k=1)
    assert summary.n_scored == 3 and [result.case for result in summary.misses()] == ["2"]


async def test_a_sync_candidate_handed_tools_is_refused_once_before_anything_runs() -> None:
    def lookup(*, ticket: str) -> str:
        return ticket

    def sync_agent(case: Mapping[str, Any], tools: Any) -> str:
        return "billing"

    host = callable_host()
    with pytest.raises(ValueError, match="sync_agent is not an async function, and a candidate handed tools must be"):
        await run_eval(CASES, sync_agent, expected=_queue, tools={"lookup": lookup}, host=host, scope_id=SCOPE)
    assert host.storage.query_templates(SCOPE) == []
