"""Reading an eval's answers: ``summary.results()`` and ``misses()``, case names, and a sync runner.

A newcomer's first question after a run is which cases it got wrong. The summary carries every result —
the case, the answer, the grades and why it failed — so it can be read after ``Eval.run`` returns, whatever
store the run used. Cases are called by their own ``id`` when they carry one, else by their position, in the
results and in every error line; the ``id`` itself is withheld from the runner and the graders. A plain ``def``
runner is called in a worker thread; one handed a world's tools must be async, and is refused before anything runs.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals import Eval, World
from threetears.evals.quick.graders import UNUSABLE_ANSWER
from threetears.evals.quick.one_call import callable_host
from threetears.evals.analysis import summarize_run
from packages.evals.tests.off_loop import off_loop

SCOPE = "quick-results-tests"

#: Three tickets; ``hasty`` files the last under the wrong queue.
CASES = [
    {"id": "refund", "text": "refund please", "queue": "billing"},
    {"text": "app crashes", "queue": "bug"},
    {"text": "charged, then it crashed", "queue": "billing"},
]


async def hasty(case: Mapping[str, Any]) -> str:
    return "bug" if "crash" in case["text"] else "billing"


#: The field each case carries its expected queue in, graded as the classifier.
QUEUE = "queue"


async def _summary(cases: Any, runner: Any, graders: list[Any], *, k: int = 1, **run: Any) -> Any:
    """The one arm's summary of an eval, in memory unless a host is named."""
    if "host" not in run:
        run["store"] = ":memory:"
    return (await Eval(cases, runner, graders, k=k).run(scope=SCOPE, **run)).summary


async def test_every_result_reads_back_after_the_call_with_its_case_answer_and_grades() -> None:
    summary = await _summary(CASES, hasty, [QUEUE], k=2)
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
    # The case as the runner was handed it: its id names it, and is withheld.
    assert first.input == {"text": "refund please", "queue": "billing"}
    assert first.expected == "billing" and first.answer == "billing"
    assert first.outcome == "scored" and first.scores["match"] is True and not first.missed


async def test_misses_are_the_wrong_answers_each_saying_why() -> None:
    summary = await _summary(CASES, hasty, [QUEUE])
    (miss,) = summary.misses()
    assert miss.case == "2" and miss.answer == "bug" and miss.expected == "billing"
    assert miss.missed_because == ["answered 'bug', expected 'billing'"]
    assert miss.render().splitlines()[0] == "case 2 (repeat 1, scored): answered 'bug', expected 'billing'"


async def test_a_failed_runner_is_a_miss_and_its_error_line_names_the_case_by_its_id() -> None:
    async def refuses_refunds(case: Mapping[str, Any]) -> str:
        if "refund" in case["text"]:
            raise RuntimeError("no refunds")
        return "bug"

    summary = await _summary(CASES, refuses_refunds, [QUEUE])
    assert summary.errors == ["case refund: the candidate raised RuntimeError: no refunds"]
    failed = next(result for result in summary.misses() if result.case == "refund")
    assert failed.outcome == "failed" and failed.answer is None
    assert failed.missed_because == ["failed: the candidate raised RuntimeError: no refunds"]
    assert failed.scores["confusion_cell"] and failed.scores["match"] is False
    assert UNUSABLE_ANSWER in str(failed.scores["confusion_cell"])


async def test_a_grader_that_gives_zero_marks_a_miss_and_an_excluded_result_is_never_one() -> None:
    def short(case: Mapping[str, Any], answer: Any) -> bool:
        return len(str(answer)) <= 3

    def picky(case: Mapping[str, Any], answer: Any) -> float:
        if case["text"] == "app crashes":
            raise ValueError("cannot grade crashes")
        return 1.0

    summary = await _summary(CASES, hasty, [short, picky])
    by_case = {result.case: result for result in summary.results()}
    assert by_case["refund"].missed_because == ["short gave 0"]
    assert by_case["1"].outcome == "excluded" and not by_case["1"].missed
    assert by_case["1"].errors == ["the scorer picky raised ValueError: cannot grade crashes"]
    assert [result.case for result in summary.misses()] == ["refund"]


async def test_cases_without_ids_are_called_by_their_position() -> None:
    summary = await _summary([{"n": 1}, {"n": 2}], hasty_n, [even])
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
        await Eval(cases, hasty_n, [even], k=1).run(host=host, scope=SCOPE)
    assert off_loop(host.storage.query_templates, SCOPE) == []


async def test_an_expected_label_refusal_names_the_case_by_its_id() -> None:
    blank = [{**CASES[0], QUEUE: ""}, *CASES[1:]]
    with pytest.raises(ValueError, match="case refund: ValueError: the case's 'queue' is ''"):
        await _summary(blank, hasty, [QUEUE])


async def test_a_summary_read_back_without_its_cases_says_so_rather_than_reading_as_no_misses() -> None:
    host = callable_host()
    summary = await _summary(CASES, hasty, [QUEUE], host=host)
    read_back = off_loop(summarize_run, host, summary.run_id, SCOPE)
    assert read_back.case_results is None
    with pytest.raises(ValueError, match="carries no per-case results"):
        read_back.misses()


async def test_a_comparison_reads_each_arms_results_and_misses_by_its_key() -> None:
    async def careful(case: Mapping[str, Any]) -> str:
        return "billing" if "refund" in case["text"] or "charged" in case["text"] else "bug"

    comparison = await Eval(CASES, {"hasty": hasty, "careful": careful}, [QUEUE], k=1).run(
        store=":memory:", scope=SCOPE
    )
    assert [result.case for result in comparison.misses("hasty")] == ["2"]
    assert comparison.misses("careful") == []
    assert len(comparison.results("careful")) == 3
    with pytest.raises(ValueError, match="'nobody' names no arm"):
        comparison.results("nobody")


def misses_two_and_ten(case: Mapping[str, Any], answer: Any) -> bool:
    return answer not in (2, 10)


async def test_a_comparison_over_unnamed_cases_lists_its_misses_case_by_case_on_both_surfaces() -> None:
    """The console and the page's Trials table share one order, case 2 before case 10, then arm, for cases called by
    their position."""

    async def twin(case: Mapping[str, Any]) -> int:
        return int(case["n"])

    comparison = await Eval([{"n": n} for n in range(12)], {"a": hasty_n, "b": twin}, [misses_two_and_ten], k=1).run(
        store=":memory:", scope=SCOPE
    )
    arm_of = {variant: arm for arm, variant in comparison.arm_variants.items()}
    page = [
        (trial.case_id.rsplit("-", 1)[1], arm_of[trial.arm])
        for trial in comparison.report.bundle.trials
        if trial.state == "failed"
    ]
    console = re.findall(r"^  case (\d+) \((\w+), repeat 1\): ", comparison.render(), flags=re.M)
    assert page == console == [("2", "a"), ("2", "b"), ("10", "a"), ("10", "b")]


async def test_a_sync_runner_runs_in_a_worker_thread() -> None:
    threads: set[int] = set()

    def blocking(case: Mapping[str, Any]) -> str:
        threads.add(threading.get_ident())
        return "bug" if "crash" in case["text"] else "billing"

    summary = await _summary(CASES, blocking, [QUEUE])
    assert summary.n_scored == 3 and summary.n_candidate_failed == 0
    assert threading.get_ident() not in threads


async def test_a_sync_callable_returning_a_coroutine_has_it_awaited() -> None:
    summary = await _summary(CASES, {"wrapped": lambda case: hasty(case)}, [QUEUE])
    assert summary.n_scored == 3 and [result.case for result in summary.misses()] == ["2"]


async def test_a_sync_runner_handed_a_worlds_tools_is_refused_once_before_anything_runs() -> None:
    def lookup(*, ticket: str) -> str:
        """Look a ticket up."""
        return ticket

    def sync_agent(case: Mapping[str, Any], tools: Any) -> str:
        return "billing"

    host = callable_host()
    refusal = "sync_agent is not an async function, and a candidate handed a world's tools must be one"
    with pytest.raises(ValueError, match=refusal):
        await Eval(CASES, World(tools=[lookup])(sync_agent), [QUEUE]).run(host=host, scope=SCOPE)
    assert off_loop(host.storage.query_templates, SCOPE) == []
