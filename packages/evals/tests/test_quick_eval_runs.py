"""``Eval``: what each cell records, every refusal it makes, and the store it leaves behind.

The rung-one program (``test_rung_one_program.py``) is the happy path read from outside. These are the
edges: a runner that raises fails its cell and a grader that misbehaves excludes it; every input an
eval refuses is refused before anything is stored; a caller's own host is used as given; the case set
is content-addressed; a cancelled run settles the run it started; and a classifier (the first field
grader) lands ``match`` and ``confusion_cell``, an unusable answer as a label of its own.

The tests that count cells pass ``k=1``, so each count is one per case rather than one per case per
repeat; the default repeat count is pinned in ``test_launch_k_default.py``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import math
from collections.abc import Mapping
from typing import Any

import pydantic
import pytest

from threetears.evals import Eval
from threetears.evals.analysis.stats import wilson_interval
from threetears.evals.kernel import ValidationFailedError, classifier_label_measure, confusion_cell, confusion_of
from threetears.evals.kernel.host import EvalHost, KindContract
from threetears.evals.quick.graders import UNUSABLE_ANSWER
from threetears.evals.quick.one_call import CALLABLE_KIND, CALLABLE_KIND_CONTRACT, callable_host
from threetears.evals.run import get_result_trace, list_results, list_runs, list_templates
from packages.evals.tests.bundle_support import one_batch_bundle
from packages.evals.tests.fixtures.courierhost import ON_TIME_RATE, courier_host
from packages.evals.tests.off_loop import off_loop

SCOPE = "eval-run-tests"
CASES = [{"n": 1}, {"n": 2}, {"n": 3}]

#: Where an eval with no host of its own keeps its runs in these tests: memory, for the test alone.
MEMORY = ":memory:"


async def double(case: Mapping[str, Any]) -> int:
    return int(case["n"]) * 2


def even(case: Mapping[str, Any], answer: Any) -> bool:
    return answer % 2 == 0


def size(case: Mapping[str, Any], answer: Any) -> float:
    return float(answer)


async def test_a_raising_runner_fails_its_cell_and_says_why() -> None:
    async def flaky(case: Mapping[str, Any]) -> int:
        if case["n"] == 2:
            raise RuntimeError("cannot do two")
        return int(case["n"])

    summary = (await Eval(CASES, flaky, [size], k=1).run(store=MEMORY, scope=SCOPE)).summary
    assert summary.status == "completed"
    assert (summary.n_scored, summary.n_candidate_failed, summary.n_excluded) == (2, 1, 0)
    assert summary.measures[0].n == 2 and summary.measures[0].mean == pytest.approx(2.0)
    assert len(summary.errors) == 1 and "the candidate raised RuntimeError: cannot do two" in summary.errors[0]


@pytest.mark.parametrize(
    ("returned", "said"),
    [("yes", "'yes'"), (None, "None"), (math.nan, "nan"), (math.inf, "inf")],
    ids=["a string", "None", "nan", "inf"],
)
async def test_a_grader_returning_no_finite_number_excludes_the_cell(returned: Any, said: str) -> None:
    def odd_grade(case: Mapping[str, Any], answer: Any) -> Any:
        return returned

    summary = (await Eval(CASES[:1], double, [odd_grade], k=1).run(store=MEMORY, scope=SCOPE)).summary
    assert (summary.n_scored, summary.n_candidate_failed, summary.n_excluded) == (0, 0, 1)
    assert f"the scorer odd_grade returned {said}" in summary.errors[0]


async def test_a_raising_grader_excludes_the_cell_and_names_the_grader() -> None:
    def broken(case: Mapping[str, Any], answer: Any) -> float:
        raise KeyError("expected")

    summary = (await Eval(CASES[:1], double, [even, broken], k=1).run(store=MEMORY, scope=SCOPE)).summary
    assert summary.n_excluded == 1
    assert "the scorer broken raised KeyError" in summary.errors[0]


async def test_a_bool_scores_as_one_or_zero() -> None:
    summary = (await Eval(CASES, double, [even], k=1).run(store=MEMORY, scope=SCOPE)).summary
    (measure,) = summary.measures
    assert (measure.mean, measure.minimum, measure.maximum, measure.n) == (1.0, 1.0, 1.0, 3)


async def test_the_answer_is_stored_verbatim_when_json_holds_it_and_as_its_repr_when_not() -> None:
    def graded(case: Mapping[str, Any], answer: Any) -> int:
        return 1

    async def as_set(case: Mapping[str, Any]) -> Any:
        return {case["n"]} if case["n"] == 1 else case["n"]

    host = callable_host([graded])
    summary = (await Eval(CASES[:2], as_set, [graded], k=1).run(host=host, scope=SCOPE)).summary
    stored = []
    for result in off_loop(list_results, host.storage, summary.run_id, SCOPE):
        trace = off_loop(get_result_trace, host.storage, result)
        assert trace is not None
        stored.append(trace.trace)
    assert sorted(stored, key=str) == [[{"repr": "{1}"}], [{"value": 2}]]


async def test_a_pydantic_answer_is_stored_as_its_json() -> None:
    class Doubled(pydantic.BaseModel):
        value: int

    async def as_model(case: Mapping[str, Any]) -> Doubled:
        return Doubled(value=int(case["n"]) * 2)

    def positive(case: Mapping[str, Any], answer: Doubled) -> bool:
        return answer.value > 0

    host = callable_host([positive])
    summary = (await Eval(CASES[:1], as_model, [positive], k=1).run(host=host, scope=SCOPE)).summary
    (result,) = off_loop(list_results, host.storage, summary.run_id, SCOPE)
    trace = off_loop(get_result_trace, host.storage, result)
    assert trace is not None and trace.trace == [{"value": {"value": 2}}]


async def test_a_callers_host_is_used_as_given_world_and_all() -> None:
    """The courier declares a world and a measure; the eval runs in it and leaves the run in its store."""
    host = courier_host()

    def on_time_rate(case: Mapping[str, Any], answer: Any) -> float:
        return 1.0

    result = await Eval(CASES, {"doubler": double}, [on_time_rate], k=1).run(host=host, scope=SCOPE)
    summary = result.summary
    assert summary.status == "completed"
    assert {m.name: m.n for m in summary.measures}[ON_TIME_RATE] == 3
    (run,) = off_loop(list_runs, host, SCOPE)
    assert run.candidate_model == "doubler"
    # The callable kind attaches no carrier, so the courier's one dimension is out of play.
    assert run.world_placements == {"road_closures": "out_of_play"}


async def test_two_evals_over_the_same_cases_share_a_template_and_different_cases_do_not() -> None:
    host = callable_host([even])
    first = await Eval(CASES, double, [even]).run(host=host, scope=SCOPE)
    second = await Eval(CASES, {"again": double}, [even]).run(host=host, scope=SCOPE)
    other = await Eval(CASES[:2], double, [even]).run(host=host, scope=SCOPE)
    assert first.template_id == second.template_id != other.template_id
    assert {template.id for template in off_loop(list_templates, host.storage, SCOPE)} == {
        first.template_id,
        other.template_id,
    }
    assert len(off_loop(list_runs, host, SCOPE)) == 3


async def test_a_cancelled_run_settles_its_run_as_cancelled() -> None:
    host = callable_host([even])
    started = asyncio.Event()

    async def stuck(case: Mapping[str, Any]) -> int:
        started.set()
        await asyncio.Event().wait()
        return 0

    call = asyncio.create_task(Eval(CASES[:1], stuck, [even]).run(host=host, scope=SCOPE))
    await asyncio.wait_for(started.wait(), timeout=5.0)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    (run,) = off_loop(list_runs, host, SCOPE)
    assert run.status == "cancelled"


# --- refusals: each made before anything is stored -----------------------------------------------


@pytest.mark.parametrize(
    ("cases", "said"),
    [
        ([], "needs at least one case"),
        ({"n": 1}, "a list of JSON objects or a file of them"),
        ("n", "reads its cases from a"),
        ([{"n": 1}, ["n", 1]], r"case\(s\) 1 are not JSON objects"),
        ([{1: "n"}], "case 0 is not a mapping with string keys"),
        ([{"n": {1, 2}}], "every case must be JSON"),
    ],
    ids=["empty", "one mapping", "a string", "a list among them", "a non-string key", "a set value"],
)
async def test_cases_that_are_not_a_list_of_json_objects_are_refused(cases: Any, said: str) -> None:
    host = callable_host([even])
    with pytest.raises(ValueError, match=said):
        await Eval(cases, double, [even]).run(host=host, scope=SCOPE)
    assert off_loop(list_templates, host.storage, SCOPE) == []


@pytest.mark.parametrize(
    ("graders", "said"),
    [
        ([], "at least one grader"),
        ([lambda case, answer: 1.0], "has none a measure can carry"),
        ([functools.partial(size)], "has none a measure can carry"),
        ([even, even], "even is graded more than once"),
    ],
    ids=["none", "a lambda", "a partial", "a repeated name"],
)
async def test_graders_that_cannot_name_a_measure_each_are_refused(graders: Any, said: str) -> None:
    with pytest.raises(ValueError, match=said):
        await Eval(CASES, double, graders).run(store=MEMORY, scope=SCOPE)


async def test_a_grader_the_callers_host_declares_no_measure_for_is_refused() -> None:
    host = courier_host()
    with pytest.raises(ValueError, match="declares no measure named even"):
        await Eval(CASES, double, [even]).run(host=host, scope=SCOPE)
    assert off_loop(list_templates, host.storage, SCOPE) == []


class _ATemperature(pydantic.BaseModel):
    """A knob a launch could turn, which an eval's run neither turns nor states."""

    temperature: float = pydantic.Field(default=0.0, description="How freely the candidate samples its answer.")


def _host_whose_callable_contract_is(contract: KindContract | None) -> EvalHost:
    """``callable_host([even])`` with its callable-kind contract replaced, or dropped for ``None``."""
    host = callable_host([even])
    kinds = () if contract is None else (contract,)
    return dataclasses.replace(host, profile=dataclasses.replace(host.profile, kinds=kinds))


@pytest.mark.parametrize(
    ("contract", "said"),
    [
        (None, "has no contract for the 'callable' kind"),
        (KindContract(CALLABLE_KIND), "a contract that declares no seats"),
        (KindContract(CALLABLE_KIND, seats=frozenset({"judge"})), "seats judge for the 'callable' kind"),
        (KindContract(CALLABLE_KIND, seats=frozenset({"simulator_model"})), "seats simulator_model"),
        (KindContract(CALLABLE_KIND, seats=frozenset({"max_cost_usd"})), "seats max_cost_usd"),
        (
            dataclasses.replace(CALLABLE_KIND_CONTRACT, overlays=_ATemperature, prefix="callable"),
            "declares overlays or a spec for the 'callable' kind",
        ),
        (
            dataclasses.replace(CALLABLE_KIND_CONTRACT, spec=_ATemperature),
            "declares overlays or a spec for the 'callable' kind",
        ),
    ],
    ids=[
        "no contract",
        "no seats declared",
        "the judge role",
        "a simulator pin",
        "the spend ceiling",
        "overlays",
        "a spec",
    ],
)
async def test_a_callers_host_must_declare_what_an_eval_runs_rig_holds(
    contract: KindContract | None, said: str
) -> None:
    """Without it every blank judge and simulator reads as unrecoverable, and two eval runs never compare."""
    host = _host_whose_callable_contract_is(contract)
    with pytest.raises(ValueError, match=said):
        await Eval(CASES, double, [even]).run(host=host, scope=SCOPE)
    assert off_loop(list_templates, host.storage, SCOPE) == []


async def test_a_callers_host_declaring_the_callable_contract_runs_and_its_runs_omit_the_simulator() -> None:
    """The accepting side, and why it matters: the simulator axis of an eval's run is omitted, not undecided."""
    host = _host_whose_callable_contract_is(CALLABLE_KIND_CONTRACT)
    summary = (await Eval(CASES, double, [even], k=1).run(host=host, scope=SCOPE)).summary
    assert summary.status == "completed"
    (run,) = off_loop(list_runs, host, SCOPE)
    assert run.max_cost_usd is None, "the one-call launch runs uncapped, which is why the ceiling is unseated"
    assert host.profile.omits_apparatus("simulator_model", [(run, None)])
    assert host.profile.omits_apparatus("judge_model", [(run, None)])


async def test_a_runner_with_no_name_needs_an_arm_name() -> None:
    unnamed = functools.partial(double)
    with pytest.raises(ValueError, match="no __name__ to label its arm by"):
        await Eval(CASES, unnamed, [even], name="unnamed").run(store=MEMORY, scope=SCOPE)
    summary = (await Eval(CASES, {"partial-doubler": unnamed}, [even]).run(store=MEMORY, scope=SCOPE)).summary
    assert summary.candidate_model == "partial-doubler"


def test_an_eval_with_no_name_to_be_known_by_is_refused() -> None:
    with pytest.raises(ValueError, match="an eval is named by name=, its cases file or its runner"):
        Eval(CASES, functools.partial(double), [even])


@pytest.mark.parametrize("k", [0, 21])
async def test_a_repeat_count_outside_the_runs_bounds_is_refused_by_the_launch(k: int) -> None:
    with pytest.raises(ValidationFailedError, match="invalid k_runs"):
        await Eval(CASES, double, [even], k=k).run(store=MEMORY, scope=SCOPE)


# --- a classifier: the first field grader lands match and confusion_cell ---------------------------

LABELLED = [
    {"text": "go left", "expected": "move"},
    {"text": "go right", "expected": "move"},
    {"text": "hit it", "expected": "attack"},
    {"text": "hit back", "expected": "attack"},
    {"text": "wait", "expected": "wait"},
]

#: A deliberately imperfect classifier's answers: one move read as attack, and "wait" never given.
ANSWERS = {"go left": "move", "go right": "attack", "hit it": "attack", "hit back": "attack", "wait": "move"}

#: The field each case carries its expected label in, graded as the classifier.
LABEL = "expected"


async def classify(case: Mapping[str, Any]) -> Any:
    return ANSWERS[case["text"]]


async def test_an_imperfect_classifier_reports_its_confusion_matrix_and_each_labels_statistics() -> None:
    host = callable_host()
    summary = (await Eval(LABELLED, classify, [LABEL], k=1).run(host=host, scope=SCOPE)).summary

    assert (summary.status, summary.n_scored) == ("completed", 5)
    by_name = {measure.name: measure for measure in summary.measures}
    assert (by_name["match"].mean, by_name["match"].n) == (pytest.approx(0.6), 5)
    assert [(cell.expected, cell.predicted, cell.count) for cell in summary.confusion] == [
        ("attack", "attack", 2),
        ("move", "attack", 1),
        ("move", "move", 1),
        ("wait", "move", 1),
    ]
    labels = {statistics.label: statistics for statistics in summary.labels}
    attack, move, wait = labels["attack"], labels["move"], labels["wait"]
    assert (attack.expected, attack.predicted, attack.correct) == (2, 3, 2)
    assert (attack.precision, attack.recall, attack.f1) == (pytest.approx(2 / 3), 1.0, pytest.approx(0.8))
    assert attack.precision_interval == wilson_interval(2, 3)
    assert attack.recall_interval == wilson_interval(2, 2)
    assert (move.precision, move.recall, move.f1) == (0.5, 0.5, 0.5)
    # Never given: no precision and so no F1, rather than either stated as 0 over no evidence.
    assert (wait.predicted, wait.precision, wait.precision_interval, wait.recall, wait.f1) == (0, None, None, 0.0, None)
    rendered = summary.render()
    assert "move → attack: 1" in rendered
    assert "wait: precision none (n=0), recall 0 (0/1" in rendered

    # Each cell landed the two core measures a classifier kind lands, and its case carries its label.
    stored_cases = {
        case.id: case for case in off_loop(host.storage.query_test_cases, SCOPE, template_id=summary.template_id)
    }
    for result in off_loop(list_results, host.storage, summary.run_id, SCOPE):
        case = stored_cases[result.test_case_id]
        expected, given = case.host_payload["expected"], ANSWERS[case.host_payload["case"]["text"]]
        assert result.host_measures == {"match": given == expected, "confusion_cell": confusion_cell(expected, given)}


async def test_the_analysis_reads_an_evals_classifier_as_it_reads_a_classifier_kind() -> None:
    """The hop past the summary: the stored results give the bundle accuracy and the per-label statistics."""
    host = callable_host()
    summary = (await Eval(LABELLED, classify, [LABEL], k=1).run(host=host, scope=SCOPE)).summary
    bundle = one_batch_bundle(off_loop(list_results, host.storage, summary.run_id, SCOPE), profile=host.profile)
    (cell,) = bundle.cell_measures
    measures = {measure.name: measure for measure in cell.measures.measures}

    assert measures["accuracy"].mean == pytest.approx(0.6)
    labels = {statistics.label: statistics for statistics in summary.labels}
    for label, statistics in labels.items():
        for name, value in (("precision", statistics.precision), ("recall", statistics.recall)):
            reading = measures.get(classifier_label_measure(name, label))
            assert (None if reading is None else reading.rate) == value, (name, label)
        f1 = measures.get(classifier_label_measure("f1", label))
        assert (None if f1 is None else f1.mean) == statistics.f1, label


@pytest.mark.parametrize(
    "answer",
    [None, "", "   ", ["move"], {"move": True}],
    ids=["None", "empty", "blank", "a list", "a mapping without the field"],
)
async def test_an_answer_that_is_no_label_is_counted_as_unusable_and_never_as_a_label(answer: Any) -> None:
    async def unusable(case: Mapping[str, Any]) -> Any:
        return answer if case["text"] == "go left" else ANSWERS[case["text"]]

    summary = (await Eval(LABELLED, unusable, [LABEL], k=1).run(store=MEMORY, scope=SCOPE)).summary

    assert (summary.n_scored, summary.n_excluded, summary.errors) == (5, 0, [])
    assert ("move", UNUSABLE_ANSWER, 1) in [(cell.expected, cell.predicted, cell.count) for cell in summary.confusion]
    labels = {statistics.label: statistics for statistics in summary.labels}
    assert (labels[UNUSABLE_ANSWER].predicted, labels[UNUSABLE_ANSWER].expected) == (1, 0)
    assert labels["move"].correct == 0, "the move case read as move only by being folded into it"
    assert {measure.name: measure.mean for measure in summary.measures}["match"] == pytest.approx(0.4)


@pytest.mark.parametrize(("answer", "label"), [(3, "3"), (True, "true")], ids=["a number", "a bool"])
async def test_a_number_or_a_bool_answer_is_the_label_its_json_text_is(answer: Any, label: str) -> None:
    """A label is the value's text: ``3`` is the label ``"3"``, a miss against ``move``, never folded into it."""

    async def scalar(case: Mapping[str, Any]) -> Any:
        return answer if case["text"] == "go left" else ANSWERS[case["text"]]

    summary = (await Eval(LABELLED, scalar, [LABEL], k=1).run(store=MEMORY, scope=SCOPE)).summary

    assert ("move", label, 1) in [(cell.expected, cell.predicted, cell.count) for cell in summary.confusion]
    assert {statistics.label: statistics.correct for statistics in summary.labels}["move"] == 0
    assert {measure.name: measure.mean for measure in summary.measures}["match"] == pytest.approx(0.4)


async def test_an_answer_with_whitespace_at_its_ends_is_a_label_of_its_own() -> None:
    """``"move "`` is not ``"move"``: it misses, and the matrix says so rather than counting a hit."""

    async def padded(case: Mapping[str, Any]) -> str:
        return "move " if case["text"] == "go left" else ANSWERS[case["text"]]

    host = callable_host()
    summary = (await Eval(LABELLED, padded, [LABEL], k=1).run(host=host, scope=SCOPE)).summary

    assert ("move", "move ", 1) in [(cell.expected, cell.predicted, cell.count) for cell in summary.confusion]
    assert {statistics.label: statistics.correct for statistics in summary.labels}["move"] == 0
    assert {measure.name: measure.mean for measure in summary.measures}["match"] == pytest.approx(0.4), (
        "match agrees with the matrix: the padded answer is a miss in both"
    )
    assert "'move '" in summary.render(), "a label's edge whitespace is shown, not left invisible"
    stored = [
        result.host_measures["confusion_cell"] for result in off_loop(list_results, host.storage, summary.run_id, SCOPE)
    ]
    assert ("move", "move ") in [confusion_of(cell) for cell in stored if isinstance(cell, str)]


async def test_a_classifier_with_graders_reports_both() -> None:
    def short(case: Mapping[str, Any], answer: Any) -> bool:
        return len(str(answer)) <= 4

    summary = (await Eval(LABELLED, classify, [LABEL, short], k=1).run(store=MEMORY, scope=SCOPE)).summary
    assert [measure.name for measure in summary.measures] == ["short", "match", "confusion_cell"]
    assert summary.confusion


async def test_a_grader_only_run_reports_no_confusion_matrix() -> None:
    summary = (await Eval(CASES, double, [even], k=1).run(store=MEMORY, scope=SCOPE)).summary
    assert (summary.confusion, summary.labels) == ([], [])
    assert [measure.name for measure in summary.measures] == ["even"]


async def test_a_classifiers_labels_are_part_of_its_case_set() -> None:
    """Two classifiers expecting the same labels are one scenario; other labels, or a grader-only eval, are not."""
    host = callable_host([even])

    async def zero(case: Mapping[str, Any]) -> int:
        return 0

    relabelled_cases = [{**case, LABEL: "move"} for case in LABELLED]
    first = await Eval(LABELLED, classify, [LABEL], k=1).run(host=host, scope=SCOPE)
    again = await Eval(LABELLED, {"again": classify}, [LABEL], k=1).run(host=host, scope=SCOPE)
    relabelled = await Eval(relabelled_cases, classify, [LABEL], k=1).run(host=host, scope=SCOPE)
    scored = await Eval(LABELLED, zero, [even], k=1).run(host=host, scope=SCOPE)
    assert first.template_id == again.template_id
    assert len({first.template_id, relabelled.template_id, scored.template_id}) == 3


# --- a classifier's refusals: each made before anything is stored ---------------------------------


async def test_an_eval_with_no_grader_is_refused() -> None:
    host = callable_host()
    with pytest.raises(ValueError, match="an eval needs at least one grader"):
        await Eval(LABELLED, classify, []).run(host=host, scope=SCOPE)
    assert off_loop(list_templates, host.storage, SCOPE) == []


@pytest.mark.parametrize("name", ["match", "confusion_cell", "accuracy"])
@pytest.mark.parametrize("classifies", [True, False], ids=["a classifier", "graders alone"])
async def test_a_grader_taking_a_classifier_measures_name_is_refused(name: str, classifies: bool) -> None:
    def grade(case: Mapping[str, Any], answer: Any) -> bool:
        return True

    grade.__name__ = name
    host = callable_host()
    with pytest.raises(ValueError, match=f"a scorer named {name} takes a measure the classifier track owns"):
        await Eval(LABELLED, classify, [LABEL, grade] if classifies else [grade]).run(host=host, scope=SCOPE)
    assert off_loop(list_templates, host.storage, SCOPE) == []
    with pytest.raises(ValueError, match="takes a measure the classifier track owns"):
        callable_host([grade])


@pytest.mark.parametrize("name", ["score", "f1", "cost_usd", "mean_score"])
@pytest.mark.parametrize("given_host", [False, True], ids=["the eval builds the host", "a host is given"])
async def test_a_grader_taking_a_core_measures_name_is_refused(name: str, given_host: bool) -> None:
    """A core name would be read under the core's meaning, direction and range, and pool with the engine's own."""

    def grade(case: Mapping[str, Any], answer: Any) -> bool:
        return True

    grade.__name__ = name
    host = callable_host()
    ran: list[Mapping[str, Any]] = []

    async def candidate(case: Mapping[str, Any]) -> str:
        ran.append(case)
        return "answer"

    with pytest.raises(ValueError, match=f"a scorer named {name} takes the name of an engine core measure"):
        if given_host:
            await Eval(LABELLED, candidate, [grade]).run(host=host, scope=SCOPE)
        else:
            await Eval(LABELLED, candidate, [grade]).run(store=MEMORY, scope=SCOPE)
    assert ran == [], "the runner ran before the grader's name was refused"
    assert off_loop(list_templates, host.storage, SCOPE) == []
    with pytest.raises(ValueError, match=f"rename the scorer's def \\(for example, {name}_grade\\)"):
        callable_host([grade])


#: Stands for a case that carries no label at all.
_ABSENT = object()


@pytest.mark.parametrize(
    ("label", "said"),
    [
        (_ABSENT, "case 0: ValueError: the field grader 'expected' reads the case's 'expected', which it lacks"),
        (None, "case 0: ValueError: the case's 'expected' is None; a field grader's label is a non-blank string"),
        ("  ", "case 0: ValueError: the case's 'expected' is '  '; a field grader's label is a non-blank string"),
        (["move"], r"case 0: ValueError: the case's 'expected' is \['move'\]; a field grader's label"),
        (UNUSABLE_ANSWER, "the label an unusable answer is counted under"),
    ],
    ids=["absent", "None", "blank", "a list", "the unusable label"],
)
async def test_an_expected_label_no_confusion_matrix_could_hold_is_refused(label: Any, said: str) -> None:
    first = {"text": "go left"} if label is _ABSENT else {"text": "go left", LABEL: label}
    host = callable_host()
    with pytest.raises(ValueError, match=said):
        await Eval([first, *LABELLED[1:]], classify, [LABEL]).run(host=host, scope=SCOPE)
    assert off_loop(list_templates, host.storage, SCOPE) == []


@pytest.mark.parametrize(
    ("contract", "said"),
    [
        (None, "has no contract for the 'callable' kind"),
        (KindContract(CALLABLE_KIND, seats=frozenset({"judge"})), "seats judge for the 'callable' kind"),
    ],
    ids=["no contract", "a judge seat"],
)
async def test_a_callers_host_is_held_to_the_callable_contract_for_a_classifier_too(
    contract: KindContract | None, said: str
) -> None:
    host = _host_whose_callable_contract_is(contract)
    with pytest.raises(ValueError, match=said):
        await Eval(LABELLED, classify, [LABEL]).run(host=host, scope=SCOPE)
    assert off_loop(list_templates, host.storage, SCOPE) == []


async def test_a_classifiers_grader_the_callers_host_declares_no_measure_for_is_refused() -> None:
    """``match`` and ``confusion_cell`` are core, so the host declares neither; a grader's measure it must."""
    host = courier_host()
    with pytest.raises(ValueError, match="declares no measure named even"):
        await Eval(LABELLED, classify, [LABEL, even]).run(host=host, scope=SCOPE)
    assert off_loop(list_templates, host.storage, SCOPE) == []
    summary = (await Eval(LABELLED, classify, [LABEL], k=1).run(host=host, scope=SCOPE)).summary
    assert summary.status == "completed" and summary.confusion
