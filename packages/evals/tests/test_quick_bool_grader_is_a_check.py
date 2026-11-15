"""A function grader is a check unless its author declares it grades a number, and what it returns is held to that.

Its kind is read from its declaration alone, never its source: annotated ``-> bool``, a check; annotated as a number
(``-> float``, ``-> int``) or given a range, a number; neither, a check, since the grader a newcomer writes first
returns a bool. A check's False fails the case it is on: in the first line's count of cases passed, in the misses below
it, in what a comparison's gate reads, and in ONE printed form, a pass rate with its count. A value the declaration
contradicts (0.25, None or a numpy-like bool from a check, False from a number) fails the run at the first such answer,
naming the case and the fix: never a dropped cell that shrinks the count. An annotation read as both kinds
(``-> bool | float``) or as neither (an alias, ``np.float64``, ``Any``) is refused when the eval is built, saying what
was read. Where no grader is a check, neither one arm's first line nor two arms' under the case floor states a pass
count or compares one.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from threetears.evals import Eval, grader
from threetears.evals.kernel.scoring import failed_checks, measure_catalog_of
from threetears.evals.quick import measure
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
        (decorated, True),
        (closeness, False),
        (a_float, False),
        (an_int, False),
        (cached, False),
        (_made_without_source("def made(case, out) -> bool:\n    return True\n"), True),
        (_made_without_source("def made(case, out):\n    return 0.5\n"), True),
        (_made_without_source("def made(case, out) -> 'bool':\n    return True\n"), True),
        (_made_without_source("def made(case, out) -> 'float':\n    return 1.0\n"), False),
        (_made_without_source("def made(case, out) -> builtins.int:\n    return 1\n"), False),
    ],
    ids=[
        "no annotation, a comparison",
        "-> bool",
        "no annotation, a call",
        "no annotation, no return",
        "no annotation, 1 or 0",
        "wraps-decorated, the wrapped -> bool",
        "-> float",
        "-> float again",
        "-> int",
        "functools.cache over -> float",
        "no source, -> bool",
        "no source, no annotation",
        "a string 'bool'",
        "a string 'float'",
        "builtins.int",
    ],
)
def test_a_grader_is_a_check_unless_it_declares_a_number(graders: Callable[..., Any], pass_fail: bool) -> None:
    assert scorer_measure(graders).pass_fail is pass_fail


def test_a_range_declares_a_number_and_a_bool_annotation_outranks_it() -> None:
    assert scorer_measure(right, value_range=(1.0, 5.0)).pass_fail is False
    assert scorer_measure(right_annotated, value_range=(1.0, 5.0)).pass_fail is True


# Re-pinned deliberately (coordinator ruling, second fix round): -> Any, -> bool | float, an unresolvable name and a union
# with one were read as a kind (check, number, check, number); an annotation the engine cannot read as exactly one kind
# is now refused when the eval is built, so they moved from the kind table above to the refusals below. Third round, the
# same: -> bool | None, Optional[float] and typing.Union[bool, None] were read through their None, yet a None returned
# fails the run; a declaration naming None is refused, and the three moved here too.
@pytest.mark.parametrize(
    ("annotation", "read"),
    [
        ("bool | float", "names both a pass or fail (bool) and a number"),
        ("float | bool", "names both a pass or fail (bool) and a number"),
        ("int | bool", "names both a pass or fail (bool) and a number"),
        ("Union[bool, Union[int, float]]", "names both a pass or fail (bool) and a number"),
        ("Score", "and Score is read as neither a pass or fail nor a number"),
        ("np.float64", "and np.float64 is read as neither a pass or fail nor a number"),
        ("Decimal", "and Decimal is read as neither a pass or fail nor a number"),
        ("Any", "and Any is read as neither a pass or fail nor a number"),
        ("Literal[0, 1]", "and Literal[0, 1] is read as neither a pass or fail nor a number"),
        ("t.Optional[float]", "and t.Optional[float] is read as neither a pass or fail nor a number"),
        ("Undefined", "and Undefined is read as neither a pass or fail nor a number"),
        ("float | Undefined", "and Undefined is read as neither a pass or fail nor a number"),
        ("None", "and None is no grade"),
        ("bool | None", "and None is no grade"),
        ("Optional[bool]", "and None is no grade"),
        ("Optional[float]", "and None is no grade"),
        ("typing.Union[bool, None]", "and None is no grade"),
        ("builtins.float | None", "and None is no grade"),
    ],
)
def test_an_annotation_not_read_as_exactly_one_kind_is_refused_when_the_eval_is_built(
    annotation: str, read: str
) -> None:
    # Never evaluated, so a name nothing defines here (np, Decimal, Score) is read as written.
    made = _made_without_source(f"def made(case, out) -> {annotation}:\n    return True\n")
    ran: list[str] = []

    async def counted(case: Mapping[str, Any]) -> str:
        ran.append(str(case["id"]))
        return "p"

    with pytest.raises(ValueError) as refused:
        Eval(CASES, counted, [made])

    message = str(refused.value)
    assert message.startswith(f"the grader made is annotated -> {annotation}, ")
    assert read in message
    # It names what was written, and never tells its author they wrote none.
    assert "no annotation" not in message
    assert "def made(case, out) -> bool" in message and "def made(case, out) -> float" in message
    assert ran == []


def test_a_bool_or_number_union_is_refused_by_a_host_built_over_it_too() -> None:
    def either_kind(case: Mapping[str, Any], out: Any) -> bool | float:
        return bool(out == case["want"])

    with pytest.raises(ValueError, match=r"annotate it def either_kind\(case, out\) -> bool to grade pass or fail"):
        callable_host([either_kind])


TEN = [{"id": f"c{index}", "want": "q" if index < 3 else "p"} for index in range(10)]


def falls_off(case, out):
    if out == case["want"]:
        return True


class _NotAPythonBool:
    """A truthy value that is no Python bool, as a numpy bool is."""

    def __init__(self, held: bool) -> None:
        self.held = held

    def __bool__(self) -> bool:
        return self.held

    def __repr__(self) -> str:
        return f"NotAPythonBool({self.held})"


def python_or_not(case, out):
    said = out == case["want"]
    return said if said else _NotAPythonBool(said)


def an_annotated_none(case: Mapping[str, Any], out: Any) -> bool:
    return None  # type: ignore[return-value]


@pytest.mark.parametrize(
    ("graders", "said"),
    [
        ([falls_off], "the grader falls_off returned None on case c"),
        ([right, falls_off], "the grader falls_off returned None on case c"),
        ([python_or_not], "._NotAPythonBool, on case c"),
        ([an_annotated_none], "the grader an_annotated_none returned None on case c"),
    ],
    ids=["falls off its end", "beside a check", "python and other bools mixed", "None from -> bool"],
)
async def test_a_check_returning_anything_but_a_python_bool_fails_the_run_naming_the_case(
    graders: list[Any], said: str
) -> None:
    result = await Eval(TEN, classify, graders, k=1).run(store=":memory:")

    first = result.render().splitlines()[0]
    # 10 cases with 3 such answers: never "passed 7 of 7", a count over the cells left.
    assert first.startswith("classify: the run failed, so there is no verdict: the grader ")
    assert said in first
    assert "passed" not in first
    assert result.exit_code() == 1


async def test_the_fix_a_check_s_failure_names_is_the_one_for_what_it_returned() -> None:
    off, _ = await _lines(classify, [falls_off])
    other = (await Eval(TEN, classify, [python_or_not], k=1).run(store=":memory:")).render().splitlines()

    assert "(it is not annotated and given no range), so it returns True or False: return one on every path" in off[0]
    assert "return bool(...) of it to grade pass or fail" in other[0]


async def test_two_arms_under_the_case_floor_with_no_check_compare_no_pass_shares() -> None:
    async def also(case: Mapping[str, Any]) -> str:
        return "p"

    result = await Eval(CASES, {"a": classify, "b": also}, [closeness], k=1, name="few").run(store=":memory:")

    first = result.render().splitlines()[0]
    assert first == (
        "few: Exploratory: b and a ran 3 cases each; no grader here says pass or fail; too few cases to call a "
        "difference."
    )


async def test_two_arms_under_the_case_floor_with_a_check_still_compare_their_pass_shares() -> None:
    result = await Eval(CASES, {"a": classify, "b": perfect}, [right], k=1, name="few").run(store=":memory:")

    first = result.render().splitlines()[0]
    assert first == "few: Exploratory: b passed more often than a; too few cases to call a difference."


@pytest.mark.parametrize(
    ("bad", "ranged", "said"),
    [
        (None, False, r"returned None on case c[012], and it is annotated to return a number"),
        (float("nan"), False, r"returned nan on case c[012], and it is annotated to return a number"),
        (float("inf"), False, r"returned inf on case c[012], and it is annotated to return a number"),
        (False, False, r"returned False on case c[012], a pass or fail, but it is annotated to return a number"),
        ("0.5", False, r"returned '0\.5', a builtins\.str, on case c[012]"),
        (_NotAPythonBool(False), False, r"\._NotAPythonBool, on case c[012]"),
        (7.0, True, r"returned 7\.0 on case c[012], outside the range 0 to 1 declared for it"),
    ],
    ids=["None", "nan", "infinity", "a bool", "a string", "another type", "off its declared range"],
)
async def test_a_number_grader_s_unusable_value_fails_the_run_naming_the_case(
    bad: Any, ranged: bool, said: str
) -> None:
    def rating(case: Mapping[str, Any], out: Any) -> float:
        return bad if case["want"] == "q" else 0.5  # type: ignore[no-any-return]

    graded = grader(rating, range=(0, 1)) if ranged else rating
    # 10 cases, 3 such answers: the first the run meets fails it.
    result = await Eval(TEN, classify, [graded], k=1).run(store=":memory:")

    lines = result.render().splitlines()
    assert lines[0].startswith("classify: the run failed, so there is no verdict: the grader rating ")
    assert re.search(said, lines[0])
    # Never a completed run whose mean is over the 7 cells that happened to be usable.
    assert result.summary.status == "failed"
    assert result.exit_code() == 1


async def _async_right(case: Mapping[str, Any], out: Any) -> bool:
    return bool(out == case["want"])


@functools.wraps(_async_right)
async def _wrapped_async(case: Mapping[str, Any], out: Any) -> bool:
    return await _async_right(case, out)


@pytest.mark.parametrize("graded", [_async_right, _wrapped_async], ids=["async def", "wraps-decorated async def"])
def test_an_async_grader_is_refused_when_the_eval_is_built(graded: Callable[..., Any]) -> None:
    # A grader is called as a plain function of (case, out), never awaited: an async one would grade a coroutine.
    with pytest.raises(
        ValueError, match=r"is an async def, and a grader is called as a plain function of \(case, out\)"
    ):
        Eval(CASES, classify, [graded])


def raises_on_three(case, out):
    if case["want"] == "q":
        raise TypeError("string indices must be integers")
    return out == case["want"]


@pytest.mark.expect_contained(TypeError, why="the grader raises on three cases on purpose: their cases are not graded")
async def test_a_grader_that_raises_leaves_its_cases_counted_as_not_graded_on_the_first_line() -> None:
    result = await Eval(TEN, classify, [raises_on_three], k=1).run(store=":memory:")

    first = result.render().splitlines()[0]
    # Never "passed 7 of 7 cases": every case that ran is the count, and the three nothing graded are named, with why.
    assert first == (
        "classify: classify passed 7 of 10 cases; 3 were not graded. Not graded: c0, c1, c2 (`test setup: the scorer "
        "raises_on_three raised TypeError: string indices must be integers`)."
    )


@measure(higher_is_better=True, merit_axis="quality", value_range=(0.0, 1.0), population="scored")
def answered(case: Mapping[str, Any], out: Any) -> Any:
    """Whether the answer is the case's, or no number on the three it cannot read."""
    return None if case["want"] == "q" else float(out == case["want"])


async def test_a_measure_s_non_number_leaves_its_case_counted_as_not_graded() -> None:
    result = await Eval(TEN, classify, [right_annotated, answered], k=1).run(store=":memory:")

    first = result.render().splitlines()[0]
    assert first.startswith("classify: classify passed 7 of 10 cases; 3 were not graded. Not graded: c0, c1, c2 (")
    assert "the scorer answered returned None, not a finite number or a bool" in first


@pytest.mark.expect_contained(TypeError, why="the grader raises on three cases on purpose: their cases are not graded")
async def test_a_comparison_says_beside_its_figures_how_many_of_each_arm_were_not_graded() -> None:
    async def also(case: Mapping[str, Any]) -> str:
        return "p"

    result = await Eval(TEN, {"a": classify, "b": also}, [raises_on_three], k=1, name="cmp").run(store=":memory:")

    first = result.render().splitlines()[0]
    assert "Of 7 trials each" in first
    assert " a's figures are over its graded trials: 3 of 10 were not graded. Not graded on a: c0, c1, c2 (" in first
    assert " b's figures are over its graded trials: 3 of 10 were not graded. Not graded on b: c0, c1, c2 (" in first
