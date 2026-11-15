"""A function grader is a check unless its author declares it grades a number, and what it returns is held to that.

Its kind is read from its declaration alone, never its source: annotated ``-> bool``, a check; annotated as a number
(``-> float``, ``-> int``) or given a range, a number; neither, a check, since the grader a newcomer writes first
returns a bool. A check's False fails the case it is on: in the first line's count of cases passed, in the misses below
it, in what a comparison's gate reads, and in ONE printed form, a pass rate with its count. A value the declaration
contradicts (0.25 from a check, False from a number) fails the run at the first such answer, naming the declaration to
write: never a dropped cell that shrinks the count. Where no grader is a check, the first line states no pass count.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from threetears.evals import Eval, grader
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


def closeness(case: Mapping[str, Any], out: Any) -> float:
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


async def test_a_numeric_grader_alone_states_no_pass_count_and_prints_its_mean() -> None:
    lines, exit_code = await _lines(classify, [closeness])

    # Re-pinned deliberately (Critic B-3): it read "classify passed 3 of 3 cases.", a count no grader judged.
    assert lines[0] == "classify: classify ran 3 of 3 cases; no grader here says pass or fail."
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


async def test_a_bool_from_a_call_with_no_annotation_is_a_check() -> None:
    lines, exit_code = await _lines(classify, [starts])

    assert lines[0] == "classify: classify passed 1 of 3 cases. Cases that failed a trial: b, c."
    assert exit_code == 0


def unannotated_number(case, out):
    return 0.5 if out == case["want"] else 0.25


def both_kinds(case, out):
    return 0.5 if out == case["want"] else False


def number_returning_a_bool(case: Mapping[str, Any], out: Any) -> float:
    return out == case["want"]


def ranged_returning_a_bool(case, out):
    return out == case["want"]


@pytest.mark.parametrize(
    ("graders", "fix"),
    [
        ([unannotated_number], "def unannotated_number(case, out) -> float or grader(unannotated_number, range="),
        ([both_kinds], "def both_kinds(case, out) -> float or grader(both_kinds, range="),
        ([right, both_kinds], "def both_kinds(case, out) -> float or grader(both_kinds, range="),
        ([number_returning_a_bool], "annotate it def number_returning_a_bool(case, out) -> bool to grade pass or fail"),
        (
            [grader(ranged_returning_a_bool, range=(0, 5))],
            "it is given a range, so its False would read as a low score, never as a failed case",
        ),
    ],
    ids=["a number from a check", "both kinds", "beside a check", "a bool from -> float", "a bool from a range"],
)
async def test_a_value_its_declaration_contradicts_fails_the_run_naming_the_fix(graders: list[Any], fix: str) -> None:
    result = await Eval(CASES, classify, graders, k=1).run(store=":memory:")

    first = result.render().splitlines()[0]
    # The run does not complete with a pass count over the cells left: it fails, naming the declaration to write.
    assert first.startswith("classify: the run failed, so there is no verdict: the grader ")
    assert fix in first
    assert "passed" not in first
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
    assert run.measure_definitions is not None
    assert run.measure_definitions["right"] != host.profile.measures.definition_digests()["right"]


def no_return(case, out):
    pass


def a_count(case, out):
    return 1 if out == case["want"] else 0


def a_float(case: Mapping[str, Any], out: Any) -> float:
    return 1.0


def an_int(case: Mapping[str, Any], out: Any) -> int:
    return 1


def either(case: Mapping[str, Any], out: Any) -> bool | float:
    return 1.0


def anything(case: Mapping[str, Any], out: Any) -> Any:
    return True


@functools.wraps(right_annotated)
def decorated(case: Any, out: Any) -> Any:
    return right_annotated(case, out)


@functools.cache
def cached(case: str, out: str) -> float:
    return 1.0


def _made_without_source(text: str) -> Callable[..., Any]:
    """A grader defined by exec, with no __future__ import: no source to read, and its annotation evaluated lazily, so
    a name nothing defines is a forward reference."""
    namespace: dict[str, Any] = {}
    # dont_inherit: this module's postponed annotations would otherwise turn every annotation into a string.
    exec(compile(text, "<made>", "exec", dont_inherit=True), namespace)
    made: Callable[..., Any] = namespace["made"]
    return made


@pytest.mark.parametrize(
    ("graders", "pass_fail"),
    [
        (right, True),
        (right_annotated, True),
        (starts, True),
        (no_return, True),
        (a_count, True),
        (anything, True),
        (decorated, True),
        (closeness, False),
        (a_float, False),
        (an_int, False),
        (either, False),
        (cached, False),
        (_made_without_source("def made(case, out) -> bool:\n    return True\n"), True),
        (_made_without_source("def made(case, out):\n    return 0.5\n"), True),
        (_made_without_source("def made(case, out) -> 'bool':\n    return True\n"), True),
        (_made_without_source("def made(case, out) -> 'float':\n    return 1.0\n"), False),
        (_made_without_source("def made(case, out) -> Undefined:\n    return 1.0\n"), True),
        (_made_without_source("def made(case, out) -> float | Undefined:\n    return 1.0\n"), False),
        (_made_without_source("def made(case, out) -> bool | None:\n    return True\n"), True),
    ],
    ids=[
        "no annotation, a comparison",
        "-> bool",
        "no annotation, a call",
        "no annotation, no return",
        "no annotation, 1 or 0",
        "-> Any",
        "wraps-decorated, the wrapped -> bool",
        "-> float",
        "-> float again",
        "-> int",
        "-> bool | float",
        "functools.cache over -> float",
        "no source, -> bool",
        "no source, no annotation",
        "a string 'bool'",
        "a string 'float'",
        "an unresolvable name",
        "a union with an unresolvable name",
        "bool or None",
    ],
)
def test_a_grader_is_a_check_unless_it_declares_a_number(graders: Callable[..., Any], pass_fail: bool) -> None:
    assert scorer_measure(graders).pass_fail is pass_fail


def test_a_range_declares_a_number_and_a_bool_annotation_outranks_it() -> None:
    assert scorer_measure(right, value_range=(1.0, 5.0)).pass_fail is False
    assert scorer_measure(right_annotated, value_range=(1.0, 5.0)).pass_fail is True
