"""Reading an eval's answers: ``summary.results()`` and ``misses()``, case names, and a sync runner.

A newcomer's first question after a run is which cases it got wrong. The summary carries every result —
the case, the answer, the grades and why it failed — so it can be read after ``Eval.run`` returns, whatever
store the run used. Cases are called by their own ``id`` when they carry one, else by their position, in the
results and in every error line; the ``id`` itself is withheld from the runner and the graders. A plain ``def``
runner is called in a worker thread; one handed a world's tools must be async, and is refused before anything runs.

The quiet print's lines, read through ``render()`` and the files ``write`` leaves:

- **A miss shows the graded field**, never the whole answer object it was read from: a runner's answer can run to
  hundreds of characters, and the label is what was wrong.
- **The judge's next step reads the sheet**: how many rows ``to_label.csv`` holds per dimension, and how many more
  judged answers the calibrated tier's floor needs when it holds fewer.
- **Total spend splits by role** when more than one role spent, each role's dollars read off its usage rows.
- **A comparison leads on its classifier, else its first grader**, never on a cost alone, with a guardrail not held
  said after the lead.
"""

from __future__ import annotations

import csv
import json
import re
import threading
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import pytest

from threetears.evals import Answer, Eval, Judge, World, grader
from threetears.evals.kernel import CALIBRATION_MIN_RESULTS
from threetears.evals.schema import StopReason
from threetears.evals.quick.graders import UNUSABLE_ANSWER
from threetears.evals.quick.one_call import callable_host
from threetears.evals.analysis import summarize_run
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.clock_support import frozen_clock


@pytest.fixture(autouse=True)
def _frozen_clock() -> Iterator[None]:
    """Nothing here asserts on the time, so every stamp the code under test makes reads a frozen clock, not the wall's."""
    with frozen_clock():
        yield


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


# --- the quiet print's lines --------------------------------------------------------------------------------------


async def routed(case: Mapping[str, Any]) -> dict[str, Any]:
    """Files every crash under bug, beside a long reason and a usage id: the shape of an app's own answer object."""
    queue = "bug" if "crash" in case["text"] else "billing"
    return {"queue": queue, "reason": "the ticket reads " + "much like a crash report " * 6, "usage_id": "u-1"}


def _miss_lines(printed: str) -> list[str]:
    return [line for line in printed.splitlines() if line.startswith("  case ")]


async def test_a_classifier_miss_prints_the_graded_field_never_the_whole_answer() -> None:
    result = await Eval(CASES, routed, [QUEUE], k=1).run(store=":memory:", scope=SCOPE)
    (line,) = _miss_lines(result.render())
    assert line == "  case 2 (repeat 1): answered queue 'bug', expected 'billing'"
    # The whole answer is still on the result, for a reader who asks for it.
    (miss,) = result.misses()
    assert isinstance(miss.answer, dict) and "usage_id" in miss.answer


async def test_a_guardrail_field_listed_first_leaves_the_classifier_s_miss_line_on_the_classifier() -> None:
    async def routed_safely(case: Mapping[str, Any]) -> dict[str, Any]:
        return {**await routed(case), "safe": True}

    cases = [{**case, "safe": True} for case in CASES]
    result = await Eval(cases, routed_safely, [grader("safe", guard=0.05), QUEUE], k=1).run(
        store=":memory:", scope=SCOPE
    )
    (line,) = _miss_lines(result.render())
    assert line == "  case 2 (repeat 1): answered queue 'bug', expected 'billing'"
    assert any(text.startswith("  queue: mean") for text in result.render().splitlines())
    matrix, _ = result.labels()
    assert {(cell.expected, cell.predicted) for cell in matrix} >= {("billing", "bug")}
    # The guardrail field, named, is counted as itself: every answer said safe, as every case expected.
    guarded, _ = result.labels("safe")
    assert {(cell.expected, cell.predicted) for cell in guarded} == {("true", "true")}


@dataclass(frozen=True)
class Routing:
    """An app's own answer that JSON cannot hold, so the run stores its repr."""

    queue: str
    note: str


async def test_an_answer_json_cannot_hold_prints_cut_never_whole() -> None:
    async def as_dataclass(case: Mapping[str, Any]) -> Routing:
        return Routing(queue="billing", note="routed after reading " + "the whole ticket " * 20)

    result = await Eval(CASES, as_dataclass, [QUEUE], k=1).run(store=":memory:", scope=SCOPE)
    (miss, *_) = result.misses()
    assert isinstance(miss.answer, str) and len(miss.answer) > 300, "stored as its repr"
    line = _miss_lines(result.render())[0]
    said = line.split("): ", 1)[1]
    shown = said.removeprefix("answered no queue it can be graded on: ").rsplit(", expected ", 1)[0]
    assert said.startswith("answered no queue it can be graded on: Routing(queue='billing'")
    assert len(shown) == 120 and shown.endswith("…")


async def test_an_answer_without_the_graded_field_prints_a_short_compact_answer_without_its_id() -> None:
    async def unlabelled(case: Mapping[str, Any]) -> dict[str, Any]:
        return {"id": "row-1", "note": "x" * 300}

    result = await Eval(CASES, unlabelled, [QUEUE], k=1).run(store=":memory:", scope=SCOPE)
    lines = _miss_lines(result.render())
    assert len(lines) == 3
    said = lines[0].split("): ", 1)[1]
    assert said.startswith('answered no queue it can be graded on: {"note":"xxx')
    shown = said.split("graded on: ", 1)[1].rsplit(", expected ", 1)[0]
    assert len(shown) == 120 and shown.endswith("…"), "the answer is cut at 120 characters, the cut marked"
    assert "row-1" not in said, "the answer's id is left out"


async def test_a_miss_s_reasons_print_on_one_line_and_the_first_line_keeps_and_n_more() -> None:
    async def raises_over_lines(case: Mapping[str, Any]) -> str:
        raise ValueError("first line\nsecond line")

    cases = [{"text": f"ticket {n}", "queue": "billing"} for n in range(7)]
    result = await Eval(cases, raises_over_lines, [QUEUE], k=1).run(store=":memory:", scope=SCOPE)
    printed = result.render().splitlines()
    assert printed[0].endswith("7 misses: 0, 1, 2, 3, 4, and 2 more")
    lines = _miss_lines(result.render())
    assert lines[0] == "  case 0 (repeat 1): failed: the candidate raised ValueError: first line second line"
    assert not [line for line in printed if line.strip() == "second line"], "no reason runs onto a line of its own"


@dataclass(frozen=True)
class _Verdict:
    """One scripted judge reply, in the attribute names the engine reads, priced."""

    content: str
    model: str = "scripted/judge"
    served_model: str | None = "scripted/judge"
    input_tokens: int | None = 10
    output_tokens: int | None = 5
    cost_usd: float | None = 0.0002
    price_source: str | None = "scripted rate"
    reasoning_tokens: int | None = None
    stop_reason: StopReason = "end_turn"
    temperature: float | None = None


# parity-with: threetears.evals.schema.CompletionClient
class FakeJudge:
    """A judge that passes every answer naming billing, and fails the rest, at a priced call each."""

    model_name = "scripted/judge"

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Verdict:
        dimension = system.split('single key "', 1)[1].split('"', 1)[0]
        passed = "billing" in user.split("# Output under review\n", 1)[1]
        return _Verdict(
            json.dumps({"reasoning": "read it", "criteria_scores": {dimension: "pass" if passed else "fail"}})
        )

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


#: Two pass/fail dimensions, so the sheet is counted per dimension.
RUBRIC = {"billing": "The answer says the ticket is about billing.", "plain": "The answer is one word."}


async def priced_hasty(case: Mapping[str, Any]) -> Answer:
    """``hasty``'s answer, reported as one call at a tenth of a cent."""
    return Answer(await hasty(case), model="m-1", input_tokens=40, output_tokens=2, cost_usd=0.001)


def _sheet(written: Mapping[str, Path]) -> Counter[str]:
    with written["to_label"].open(encoding="utf-8", newline="") as read:
        return Counter(row["dimension"] for row in csv.DictReader(read))


async def test_the_judge_s_next_step_says_what_the_sheet_holds_and_how_many_more_answers_the_floor_needs(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "labels.csv"
    result = await Eval(CASES, hasty, [QUEUE, Judge(FakeJudge(), RUBRIC, labels=labels)], k=1).run(
        store=":memory:", scope=SCOPE
    )
    sheet = _sheet(result.write(tmp_path / "report"))
    assert sheet == {"answer.billing": 3, "answer.plain": 3}, "one row per judged answer per dimension"
    needed = CALIBRATION_MIN_RESULTS - 3
    assert result.render().splitlines()[1] == (
        f"judge: 0 labels: judge uncalibrated ({labels} is not there yet); next step: rate every output in the "
        "to_label.csv the report writes, which holds 3 per dimension, without looking at the judge's verdict; the "
        f"calibrated tier's floor is {CALIBRATION_MIN_RESULTS} per dimension, so it needs {needed} more judged "
        "answers per dimension"
    )


async def test_a_sheet_at_the_floor_asks_for_the_floor_s_ratings_of_each_dimension(tmp_path: Path) -> None:
    cases = [{"text": f"refund {n}", "queue": "billing"} for n in range(CALIBRATION_MIN_RESULTS)]
    result = await Eval(cases, hasty, [QUEUE, Judge(FakeJudge(), RUBRIC)], k=1).run(store=":memory:", scope=SCOPE)
    sheet = _sheet(result.write(tmp_path / "report"))
    assert set(sheet.values()) == {CALIBRATION_MIN_RESULTS}
    assert result.calibration_lines() == [
        "judge: 0 labels: judge uncalibrated (Judge(labels=...) names a person's labels); next step: rate about "
        f"{CALIBRATION_MIN_RESULTS} outputs of each dimension in the to_label.csv the report writes, which holds "
        f"{CALIBRATION_MIN_RESULTS} per dimension, without looking at the judge's verdict"
    ]


async def test_total_spend_splits_by_role_when_more_than_one_role_spent() -> None:
    judged = await Eval(CASES, priced_hasty, [QUEUE, Judge(FakeJudge(), RUBRIC)], k=1).run(
        store=":memory:", scope=SCOPE
    )
    summary = judged.summary
    candidate, judge = summary.role_spend("candidate"), summary.role_spend("judge")
    assert candidate is not None and judge is not None
    (line,) = [line.strip() for line in judged.render().splitlines() if "total spend" in line]
    # Three candidate calls at $0.001; six judge calls (three answers, two dimensions) at $0.0002, read off the rows.
    assert line == "total spend: $0.00420, the results' cost (candidate $0.00300, judge $0.00120)"
    assert candidate.cost_usd == pytest.approx(0.003) and judge.cost_usd == pytest.approx(0.0012)

    alone = await Eval(CASES, priced_hasty, [QUEUE], k=1).run(store=":memory:", scope=SCOPE)
    (line,) = [line.strip() for line in alone.render().splitlines() if "total spend" in line]
    assert line == "total spend: $0.00300, the results' cost", "one role's total is not split"


# parity-with: threetears.evals.schema.CompletionClient
class FakeUnpricedJudge(FakeJudge):
    """:class:`FakeJudge`, whose provider prices none of its calls."""

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Verdict:
        verdict = await super().generate(system=system, user=user, response_format=response_format)
        return _Verdict(verdict.content, cost_usd=None, price_source=None)


async def test_an_unpriced_role_leaves_the_total_unsplit_and_never_sums_it_as_zero() -> None:
    result = await Eval(CASES, priced_hasty, [QUEUE, Judge(FakeUnpricedJudge(), RUBRIC)], k=1).run(
        store=":memory:", scope=SCOPE
    )
    (line,) = [line.strip() for line in result.render().splitlines() if "total spend" in line]
    assert line == "total spend: unknown in full: $0.00300 over the priced calls, plus 6 call(s) unpriced"


#: Enough cases for a contrast to be tested, each priced by its length so the spend varies case to case.
PAIRED = [{"text": f"refund number {'x' * n}", "queue": "billing", "safe": n % 4 != 0} for n in range(12)]


async def cheap(case: Mapping[str, Any]) -> Answer:
    """Files every ticket right, at a cost that grows with its text."""
    return Answer({"queue": case["queue"], "leaks": False}, model="m-1", cost_usd=len(case["text"]) * 1e-6)


async def dear(case: Mapping[str, Any]) -> Answer:
    """``cheap``'s answers, at ten times the price."""
    answered = await cheap(case)
    assert answered.cost_usd is not None
    return Answer(answered.value, model="m-2", cost_usd=answered.cost_usd * 10)


@grader(guard=0.05)
def no_leak(case: Mapping[str, Any], out: Any) -> bool:
    """Never leaks; a guardrail every arm holds here, on too few cases to show it."""
    return not out["leaks"]


async def test_a_comparison_leads_on_its_classifier_never_on_cost_and_names_a_guardrail_not_held_after() -> None:
    result = await Eval(PAIRED, {"dear": dear, "cheap": cheap}, [QUEUE, no_leak], k=2).run(
        store=":memory:", scope=SCOPE
    )
    cost = [row for row in result.contrasts() if row["measure_id"] == "production_replicating_cost"]
    assert [row["outcome"] for row in cost] == ["improved"], "the cost separated, and is no lead on its own"
    (accuracy,) = result.contrasts("accuracy")
    (guarded,) = result.guardrails("cheap")
    assert guarded["outcome"] == "undecided"
    assert result.render().splitlines()[0] == (
        f"{result.name}: Accuracy: {accuracy['verdict']}; also {cost[0]['reading']}: {cost[0]['verdict']}; "
        f"guardrail {guarded['guardrail']} undecided for cheap"
    )


async def test_with_no_classifier_a_comparison_leads_on_its_first_grader() -> None:
    def resolved(case: Mapping[str, Any], out: Any) -> bool:
        return bool(out["queue"])

    def short(case: Mapping[str, Any], out: Any) -> bool:
        return len(str(out)) < 1000

    result = await Eval(PAIRED, {"dear": dear, "cheap": cheap}, [resolved, short], k=2).run(
        store=":memory:", scope=SCOPE
    )
    (first,) = result.contrasts("resolved")
    (cost,) = result.contrasts("production_replicating_cost")
    assert result.render().splitlines()[0] == (
        f"{result.name}: {first['reading']}: {first['verdict']}; also {cost['reading']}: {cost['verdict']}"
    )


def _worded(rows: list[dict[str, Any]]) -> str:
    return "; ".join(f"{row['reading']}: {row['verdict']}" for row in rows)


async def test_a_comparison_graded_by_a_guardrail_alone_never_leads_on_cost() -> None:
    result = await Eval(PAIRED, {"dear": dear, "cheap": cheap}, [no_leak], k=2).run(store=":memory:", scope=SCOPE)
    (cost,) = result.contrasts("production_replicating_cost")
    assert cost["outcome"] == "improved"
    (guarded,) = result.guardrails("cheap")
    assert result.render().splitlines()[0] == (
        f"{result.name}: nothing separated cheap from dear but cost; {_worded([cost])}; "
        f"guardrail {guarded['guardrail']} {guarded['outcome']} for cheap"
    )


async def test_a_judge_only_comparison_leads_on_its_first_judged_dimension() -> None:
    result = await Eval(PAIRED, {"dear": dear, "cheap": cheap}, [Judge(FakeJudge(), RUBRIC)], k=2).run(
        store=":memory:", scope=SCOPE
    )
    (first,) = result.contrasts("answer.billing")
    (cost,) = result.contrasts("production_replicating_cost")
    assert first["outcome"] != "improved" and cost["outcome"] == "improved"
    assert result.render().splitlines()[0] == f"{result.name}: {_worded([first])}; also {_worded([cost])}"


async def test_a_judge_listed_first_leads_before_a_function_listed_after_it() -> None:
    def short(case: Mapping[str, Any], out: Any) -> bool:
        return len(str(out)) < 1000

    result = await Eval(PAIRED, {"dear": dear, "cheap": cheap}, [Judge(FakeJudge(), RUBRIC), short], k=2).run(
        store=":memory:", scope=SCOPE
    )
    (first,) = result.contrasts("answer.billing")
    assert result.render().splitlines()[0].startswith(f"{result.name}: {_worded([first])}; ")


async def cheapest(case: Mapping[str, Any]) -> Answer:
    """``cheap``'s answers, at a tenth of its price."""
    answered = await cheap(case)
    assert answered.cost_usd is not None
    return Answer(answered.value, model="m-3", cost_usd=answered.cost_usd / 10)


async def test_three_arms_lead_on_the_classifier_for_every_arm_tested() -> None:
    result = await Eval(PAIRED, {"dear": dear, "cheap": cheap, "cheapest": cheapest}, [QUEUE], k=2).run(
        store=":memory:", scope=SCOPE
    )
    accuracy = result.contrasts("accuracy")
    cost = [row for row in result.contrasts("production_replicating_cost") if row["outcome"] == "improved"]
    assert sorted(row["arm"] for row in accuracy) == ["cheap", "cheapest"]
    assert cost, "the cost separated, and follows the lead"
    assert result.render().splitlines()[0] == f"{result.name}: {_worded(accuracy)}; also {_worded(cost)}"


async def test_a_lead_with_nothing_else_separated_says_so() -> None:
    async def twin(case: Mapping[str, Any]) -> Answer:
        return await cheap(case)

    result = await Eval(PAIRED, {"cheap": cheap, "twin": twin}, [QUEUE], k=2).run(store=":memory:", scope=SCOPE)
    (accuracy,) = result.contrasts("accuracy")
    assert not [row for row in result.contrasts() if row["outcome"] in ("improved", "regressed")]
    assert result.render().splitlines()[0] == f"{result.name}: {_worded([accuracy])}; nothing else separated"
