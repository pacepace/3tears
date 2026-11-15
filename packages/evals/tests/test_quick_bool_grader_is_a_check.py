"""A function grader that returns a bool is the author saying pass or fail, so it is a check wherever it is read.

Its False fails the case it is on: in the first line's count of cases passed, in the misses below it, in what a
comparison's gate reads, and in ONE printed form, a pass rate with its count, whether its return is annotated
``-> bool`` or not. A grader whose bool the package cannot tell from its code before it runs is stopped at its first
answer, naming the fix, never landed as a score no case can fail.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pytest

from threetears.evals import Eval
from threetears.evals.kernel.scoring import failed_checks, measure_catalog_of
from threetears.evals.quick.one_call import callable_host, scorer_measure
from threetears.evals.run import list_results, list_runs
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.off_loop import off_loop

CASES = [
    {"id": "a", "text": "x", "want": "p"},
    {"id": "b", "text": "y", "want": "q"},
    {"id": "c", "text": "z", "want": "r"},
]


async def classify(case: Mapping[str, Any]) -> str:
    """Answer p, whatever the case."""
    return "p"


async def perfect(case: Mapping[str, Any]) -> str:
    """Answer what the case wants."""
    return str(case["want"])


# Written as a newcomer writes one: no annotation, a comparison returned.
def right(case, out):
    return out == case["want"]


def right_annotated(case: Mapping[str, Any], out: Any) -> bool:
    return bool(out == case["want"])


def closeness(case, out):
    return 0.5 if out == case["want"] else 0.25


def starts(case, out):
    return out.startswith(case["want"])


async def _lines(runner: Callable[..., Any], graders: list[Any]) -> tuple[list[str], int]:
    result = await Eval(CASES, runner, graders, k=1).run(store=":memory:")
    return result.render().splitlines(), result.exit_code()


async def test_a_bool_grader_with_no_annotation_fails_the_cases_it_returned_false_on() -> None:
    lines, exit_code = await _lines(classify, [right])

    assert lines[0] == "classify: classify passed 1 of 3 cases. Cases that failed a trial: b, c."
    assert "  Right score: 33%, 1 of 3 trials" in lines
    assert "  case b (repeat 1): Right score is 0" in lines
    # One arm has no gate: it exits 0 as every completed run does, a check failing or not.
    assert exit_code == 0


async def test_a_bool_grader_that_passed_every_case_says_so() -> None:
    lines, exit_code = await _lines(perfect, [right])

    assert lines[0] == "perfect: perfect passed 3 of 3 cases."
    assert "  Right score: 100%, 3 of 3 trials" in lines
    assert exit_code == 0


async def test_a_numeric_grader_alone_is_a_score_no_case_fails_and_prints_its_mean() -> None:
    lines, exit_code = await _lines(classify, [closeness])

    assert lines[0] == "classify: classify passed 3 of 3 cases."
    assert "  Closeness score: mean 0.3333 (3 trials, min 0.25, max 0.5)" in lines
    assert exit_code == 0


async def test_beside_a_numeric_grader_the_bool_grader_alone_decides_the_count() -> None:
    lines, _ = await _lines(classify, [right, closeness])

    assert lines[0] == "classify: classify passed 1 of 3 cases. Cases that failed a trial: b, c."
    assert "  Right score: 33%, 1 of 3 trials" in lines
    assert "  Closeness score: mean 0.3333 (3 trials, min 0.25, max 0.5)" in lines


def test_an_eval_with_no_grader_is_refused_before_anything_runs() -> None:
    with pytest.raises(ValueError, match="an eval needs at least one grader"):
        Eval(CASES, classify, [])


async def test_a_bool_grader_prints_one_form_annotated_or_not() -> None:
    bare, _ = await _lines(classify, [right])
    annotated, _ = await _lines(classify, [right_annotated])

    assert bare[0] == annotated[0]
    assert [line.replace("Right annotated score", "Right score") for line in annotated[1:4]] == bare[1:4]


async def test_a_bool_from_a_grader_whose_code_does_not_say_pass_or_fail_is_stopped_naming_the_fix() -> None:
    result = await Eval(CASES, classify, [starts], k=1).run(store=":memory:")

    first = result.render().splitlines()[0]
    assert "passed" not in first
    assert result.summary.n_scored == 0 and result.summary.n_excluded == 3
    assert (
        "annotate it def starts(case, out) -> bool, or return float(...) to grade a number" in result.summary.errors[0]
    )
    assert result.exit_code() == 1


async def test_a_comparison_gates_on_a_bool_grader_with_no_annotation() -> None:
    cases = [{"id": f"c{index}", "want": "p"} for index in range(30)]

    async def good(case: Mapping[str, Any]) -> str:
        return "p"

    async def wrong(case: Mapping[str, Any]) -> str:
        return "q"

    result = await Eval(cases, {"good": good, "wrong": wrong}, [right], k=1, name="gated").run(store=":memory:")

    assert result.exit_code() == 4
    assert (why := result.why_failed()) is not None and why.startswith("Right score: wrong worse than good.")


async def test_a_result_stored_as_a_number_reads_back_as_a_check_through_the_bool_grader() -> None:
    store = InMemoryDocumentStore()

    def stored_as_a_number(case: Mapping[str, Any], out: Any) -> float:
        return 1.0 if out == case["want"] else 0.0

    # The grader as it was written before: the same measure, landed as 1.0 and 0.0 under a number's descriptor.
    stored_as_a_number.__name__ = "right"
    await Eval(CASES, classify, [stored_as_a_number], k=1).run(store=store, scope="kept")

    # Read back as any reader does: through a host built from the grader as written now, its measures described as
    # that host declares them. Its 0.0s fail their trials; the run's recorded definition of the measure is the old
    # one, so it never pools with a run graded under the new one.
    host = callable_host([right], store=store)
    (run,) = off_loop(list_runs, host, "kept")
    stored = off_loop(list_results, host.storage, run.id, "kept")
    catalog = measure_catalog_of(stored, host.profile.measures)
    assert sorted(len(failed_checks(result, catalog)) for result in stored) == [0, 1, 1]


def nested_number(case, out):
    def size(value):
        return len(value)

    return size(out) > 0


def no_return(case, out):
    pass


def a_count(case, out):
    return 1 if out == case["want"] else 0


def sometimes_a_number(case, out):
    if out is None:
        return 0.0
    return out == case["want"]


def either_way(case, out):
    return (out == case["want"] and not out.isupper()) if out else isinstance(out, str)


def a_float(case: Mapping[str, Any], out: Any) -> float:
    return 1.0


@pytest.mark.parametrize(
    ("grader", "pass_fail"),
    [
        (right, True),
        (right_annotated, True),
        (nested_number, True),
        (either_way, True),
        (closeness, False),
        (starts, False),
        (no_return, False),
        (a_count, False),
        (sometimes_a_number, False),
        (a_float, False),
    ],
)
def test_a_grader_is_a_pass_fail_when_its_annotation_or_every_return_says_bool(
    grader: Callable[..., Any], pass_fail: bool
) -> None:
    assert scorer_measure(grader).pass_fail is pass_fail
