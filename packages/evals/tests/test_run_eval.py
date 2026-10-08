"""``run_eval``: what each cell records, every refusal it makes, and the store it leaves behind.

The rung-zero example (``test_rung_zero.py``) is the happy path read from outside. These are the
edges: a candidate that raises fails its cell and a scorer that misbehaves excludes it; every input
``run_eval`` refuses is refused before anything is stored; a caller's own host is used as given; the
case set is content-addressed; a cancelled call settles the run it started; and a classifier
(``expected=``) lands ``match`` and ``confusion_cell``, an unusable answer as a label of its own.

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

from threetears.evals.analysis.stats import wilson_interval
from threetears.evals.contracts import ValidationFailedError, classifier_label_measure, confusion_cell, confusion_of
from threetears.evals.contracts.host import EvalHost, KindContract
from threetears.evals.quick import (
    CALLABLE_KIND,
    CALLABLE_KIND_CONTRACT,
    UNUSABLE_ANSWER,
    callable_host,
    run_eval,
)
from threetears.evals.run import get_result_trace, list_results, list_runs, list_templates
from packages.evals.tests.bundle_support import one_batch_bundle
from packages.evals.tests.fixtures.courierhost import ON_TIME_RATE, courier_host

SCOPE = "run-eval-tests"
CASES = [{"n": 1}, {"n": 2}, {"n": 3}]


async def double(case: Mapping[str, Any]) -> int:
    return int(case["n"]) * 2


def even(case: Mapping[str, Any], answer: Any) -> bool:
    return answer % 2 == 0


def size(case: Mapping[str, Any], answer: Any) -> float:
    return float(answer)


async def test_a_raising_candidate_fails_its_cell_and_says_why() -> None:
    async def flaky(case: Mapping[str, Any]) -> int:
        if case["n"] == 2:
            raise RuntimeError("cannot do two")
        return int(case["n"])

    summary = await run_eval(CASES, flaky, [size], scope_id=SCOPE, k=1)
    assert summary.status == "completed"
    assert (summary.n_scored, summary.n_candidate_failed, summary.n_excluded) == (2, 1, 0)
    assert summary.measures[0].n == 2 and summary.measures[0].mean == pytest.approx(2.0)
    assert len(summary.errors) == 1 and "the candidate raised RuntimeError: cannot do two" in summary.errors[0]


@pytest.mark.parametrize(
    ("returned", "said"),
    [("yes", "'yes'"), (None, "None"), (math.nan, "nan"), (math.inf, "inf")],
    ids=["a string", "None", "nan", "inf"],
)
async def test_a_scorer_returning_no_finite_number_excludes_the_cell(returned: Any, said: str) -> None:
    def odd_grade(case: Mapping[str, Any], answer: Any) -> Any:
        return returned

    summary = await run_eval(CASES[:1], double, [odd_grade], scope_id=SCOPE, k=1)
    assert (summary.n_scored, summary.n_candidate_failed, summary.n_excluded) == (0, 0, 1)
    assert f"the scorer odd_grade returned {said}" in summary.errors[0]


async def test_a_raising_scorer_excludes_the_cell_and_names_the_scorer() -> None:
    def broken(case: Mapping[str, Any], answer: Any) -> float:
        raise KeyError("expected")

    summary = await run_eval(CASES[:1], double, [even, broken], scope_id=SCOPE, k=1)
    assert summary.n_excluded == 1
    assert "the scorer broken raised KeyError" in summary.errors[0]


async def test_a_bool_scores_as_one_or_zero() -> None:
    summary = await run_eval(CASES, double, [even], scope_id=SCOPE, k=1)
    (measure,) = summary.measures
    assert (measure.mean, measure.minimum, measure.maximum, measure.n) == (1.0, 1.0, 1.0, 3)


async def test_the_answer_is_stored_verbatim_when_json_holds_it_and_as_its_repr_when_not() -> None:
    def graded(case: Mapping[str, Any], answer: Any) -> int:
        return 1

    async def as_set(case: Mapping[str, Any]) -> Any:
        return {case["n"]} if case["n"] == 1 else case["n"]

    host = callable_host([graded])
    summary = await run_eval(CASES[:2], as_set, [graded], scope_id=SCOPE, host=host, k=1)
    stored = []
    for result in list_results(host.storage, summary.run_id, SCOPE):
        trace = get_result_trace(host.storage, result)
        assert trace is not None
        stored.append(trace.trace)
    assert sorted(stored, key=str) == [[{"repr": "{1}"}], [{"value": 2}]]


async def test_a_callers_host_is_used_as_given_world_and_all() -> None:
    """The courier declares a world and a measure; run_eval runs in it and leaves the run in its store."""
    host = courier_host()

    def on_time_rate(case: Mapping[str, Any], answer: Any) -> float:
        return 1.0

    summary = await run_eval(CASES, double, [on_time_rate], scope_id=SCOPE, host=host, model="doubler", k=1)
    assert summary.status == "completed"
    assert {m.name: m.n for m in summary.measures}[ON_TIME_RATE] == 3
    (run,) = list_runs(host, SCOPE)
    assert run.candidate_model == "doubler"
    # The callable kind attaches no carrier, so the courier's one dimension is out of play.
    assert run.world_placements == {"road_closures": "out_of_play"}


async def test_two_calls_over_the_same_cases_share_a_template_and_different_cases_do_not() -> None:
    host = callable_host([even])
    first = await run_eval(CASES, double, [even], scope_id=SCOPE, host=host)
    second = await run_eval(CASES, double, [even], scope_id=SCOPE, host=host, model="again")
    other = await run_eval(CASES[:2], double, [even], scope_id=SCOPE, host=host)
    assert first.template_id == second.template_id != other.template_id
    assert {template.id for template in list_templates(host.storage, SCOPE)} == {first.template_id, other.template_id}
    assert len(list_runs(host, SCOPE)) == 3


async def test_a_cancelled_call_settles_its_run_as_cancelled() -> None:
    host = callable_host([even])
    started = asyncio.Event()

    async def stuck(case: Mapping[str, Any]) -> int:
        started.set()
        await asyncio.Event().wait()
        return 0

    call = asyncio.create_task(run_eval(CASES[:1], stuck, [even], scope_id=SCOPE, host=host))
    await asyncio.wait_for(started.wait(), timeout=5.0)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    (run,) = list_runs(host, SCOPE)
    assert run.status == "cancelled"


# --- refusals: each made before anything is stored -----------------------------------------------


@pytest.mark.parametrize(
    ("cases", "said"),
    [
        ([], "non-empty list of cases"),
        ({"n": 1}, "non-empty list of cases"),
        ("n", "non-empty list of cases"),
        ([{"n": 1}, ["n", 1]], "case 1 is not a mapping"),
        ([{1: "n"}], "case 0 is not a mapping with string keys"),
        ([{"n": {1, 2}}], "every case must be JSON"),
    ],
    ids=["empty", "one mapping", "a string", "a list among them", "a non-string key", "a set value"],
)
async def test_cases_that_are_not_a_list_of_json_objects_are_refused(cases: Any, said: str) -> None:
    host = callable_host([even])
    with pytest.raises(ValueError, match=said):
        await run_eval(cases, double, [even], scope_id=SCOPE, host=host)
    assert list_templates(host.storage, SCOPE) == []


@pytest.mark.parametrize(
    ("scorers", "said"),
    [
        ([], "at least one scorer"),
        ([lambda case, answer: 1.0], "has none a measure can carry"),
        ([functools.partial(size)], "has none a measure can carry"),
        ([even, even], "scorers named even more than once"),
    ],
    ids=["none", "a lambda", "a partial", "a repeated name"],
)
async def test_scorers_that_cannot_name_a_measure_each_are_refused(scorers: Any, said: str) -> None:
    with pytest.raises(ValueError, match=said):
        await run_eval(CASES, double, scorers, scope_id=SCOPE)


async def test_a_scorer_the_callers_host_declares_no_measure_for_is_refused() -> None:
    host = courier_host()
    with pytest.raises(ValueError, match="declares no measure named even"):
        await run_eval(CASES, double, [even], scope_id=SCOPE, host=host)
    assert list_templates(host.storage, SCOPE) == []


class _ATemperature(pydantic.BaseModel):
    """A knob a launch could turn, which a run_eval run neither turns nor states."""

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
async def test_a_callers_host_must_declare_what_a_run_eval_runs_rig_holds(
    contract: KindContract | None, said: str
) -> None:
    """Without it every blank judge and simulator reads as unrecoverable, and two run_eval runs never compare."""
    host = _host_whose_callable_contract_is(contract)
    with pytest.raises(ValueError, match=said):
        await run_eval(CASES, double, [even], scope_id=SCOPE, host=host)
    assert list_templates(host.storage, SCOPE) == []


async def test_a_callers_host_declaring_the_callable_contract_runs_and_its_runs_omit_the_simulator() -> None:
    """The accepting side, and why it matters: the simulator axis of a run_eval run is omitted, not undecided."""
    host = _host_whose_callable_contract_is(CALLABLE_KIND_CONTRACT)
    summary = await run_eval(CASES, double, [even], scope_id=SCOPE, host=host, k=1)
    assert summary.status == "completed"
    (run,) = list_runs(host, SCOPE)
    assert run.max_cost_usd is None, "the one-call launch runs uncapped, which is why the ceiling is unseated"
    assert host.profile.omits_apparatus("simulator_model", [(run, None)])
    assert host.profile.omits_apparatus("judge_model", [(run, None)])


async def test_a_candidate_with_no_name_needs_a_model_label() -> None:
    unnamed = functools.partial(double)
    with pytest.raises(ValueError, match="no __name__ to label its arm by"):
        await run_eval(CASES, unnamed, [even], scope_id=SCOPE)
    summary = await run_eval(CASES, unnamed, [even], scope_id=SCOPE, model="partial-doubler")
    assert summary.candidate_model == "partial-doubler"


@pytest.mark.parametrize("k", [0, 21])
async def test_a_repeat_count_outside_the_runs_bounds_is_refused_by_the_launch(k: int) -> None:
    with pytest.raises(ValidationFailedError, match="invalid k_runs"):
        await run_eval(CASES, double, [even], scope_id=SCOPE, k=k)


# --- a classifier: expected= lands match and confusion_cell ---------------------------------------

LABELLED = [
    {"text": "go left", "expected": "move"},
    {"text": "go right", "expected": "move"},
    {"text": "hit it", "expected": "attack"},
    {"text": "hit back", "expected": "attack"},
    {"text": "wait", "expected": "wait"},
]

#: A deliberately imperfect classifier's answers: one move read as attack, and "wait" never given.
ANSWERS = {"go left": "move", "go right": "attack", "hit it": "attack", "hit back": "attack", "wait": "move"}


async def classify(case: Mapping[str, Any]) -> Any:
    return ANSWERS[case["text"]]


def expected_label(case: Mapping[str, Any]) -> str:
    return str(case["expected"])


async def test_an_imperfect_classifier_reports_its_confusion_matrix_and_each_labels_statistics() -> None:
    host = callable_host()
    summary = await run_eval(LABELLED, classify, scope_id=SCOPE, expected=expected_label, host=host, k=1)

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
    stored_cases = {case.id: case for case in host.storage.query_test_cases(SCOPE, template_id=summary.template_id)}
    for result in list_results(host.storage, summary.run_id, SCOPE):
        case = stored_cases[result.test_case_id]
        expected, given = case.host_payload["expected"], ANSWERS[case.host_payload["case"]["text"]]
        assert result.host_measures == {"match": given == expected, "confusion_cell": confusion_cell(expected, given)}


async def test_the_analysis_reads_a_rung_zero_classifier_as_it_reads_a_classifier_kind() -> None:
    """The hop past the summary: the stored results give the bundle accuracy and the per-label statistics."""
    host = callable_host()
    summary = await run_eval(LABELLED, classify, scope_id=SCOPE, expected=expected_label, host=host, k=1)
    bundle = one_batch_bundle(list_results(host.storage, summary.run_id, SCOPE), profile=host.profile)
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
    [None, "", "   ", 3, ["move"], True],
    ids=["None", "empty", "blank", "a number", "a list", "a bool"],
)
async def test_an_answer_that_is_no_label_is_counted_as_unusable_and_never_as_a_label(answer: Any) -> None:
    async def unusable(case: Mapping[str, Any]) -> Any:
        return answer if case["text"] == "go left" else ANSWERS[case["text"]]

    summary = await run_eval(LABELLED, unusable, scope_id=SCOPE, expected=expected_label, k=1)

    assert (summary.n_scored, summary.n_excluded, summary.errors) == (5, 0, [])
    assert ("move", UNUSABLE_ANSWER, 1) in [(cell.expected, cell.predicted, cell.count) for cell in summary.confusion]
    labels = {statistics.label: statistics for statistics in summary.labels}
    assert (labels[UNUSABLE_ANSWER].predicted, labels[UNUSABLE_ANSWER].expected) == (1, 0)
    assert labels["move"].correct == 0, "the move case read as move only by being folded into it"
    assert {measure.name: measure.mean for measure in summary.measures}["match"] == pytest.approx(0.4)


async def test_an_answer_with_whitespace_at_its_ends_is_a_label_of_its_own() -> None:
    """``"move "`` is not ``"move"``: it misses, and the matrix says so rather than counting a hit."""

    async def padded(case: Mapping[str, Any]) -> str:
        return "move " if case["text"] == "go left" else ANSWERS[case["text"]]

    host = callable_host()
    summary = await run_eval(LABELLED, padded, scope_id=SCOPE, expected=expected_label, host=host, k=1)

    assert ("move", "move ", 1) in [(cell.expected, cell.predicted, cell.count) for cell in summary.confusion]
    assert {statistics.label: statistics.correct for statistics in summary.labels}["move"] == 0
    assert {measure.name: measure.mean for measure in summary.measures}["match"] == pytest.approx(0.4), (
        "match agrees with the matrix: the padded answer is a miss in both"
    )
    assert "'move '" in summary.render(), "a label's edge whitespace is shown, not left invisible"
    stored = [result.host_measures["confusion_cell"] for result in list_results(host.storage, summary.run_id, SCOPE)]
    assert ("move", "move ") in [confusion_of(cell) for cell in stored if isinstance(cell, str)]


async def test_a_classifier_with_scorers_reports_both() -> None:
    def short(case: Mapping[str, Any], answer: Any) -> bool:
        return len(str(answer)) <= 4

    summary = await run_eval(LABELLED, classify, [short], scope_id=SCOPE, expected=expected_label, k=1)
    assert [measure.name for measure in summary.measures] == ["short", "match", "confusion_cell"]
    assert summary.confusion


async def test_a_scorer_only_run_reports_no_confusion_matrix() -> None:
    summary = await run_eval(CASES, double, [even], scope_id=SCOPE, k=1)
    assert (summary.confusion, summary.labels) == ([], [])
    assert [measure.name for measure in summary.measures] == ["even"]


async def test_a_classifiers_labels_are_part_of_its_case_set() -> None:
    """Two classifier calls expecting the same labels are one scenario; other labels, or a scorer call, are not."""
    host = callable_host([even])

    async def zero(case: Mapping[str, Any]) -> int:
        return 0

    first = await run_eval(LABELLED, classify, scope_id=SCOPE, expected=expected_label, host=host, k=1)
    again = await run_eval(LABELLED, classify, scope_id=SCOPE, expected=expected_label, host=host, k=1, model="again")
    relabelled = await run_eval(LABELLED, classify, scope_id=SCOPE, expected=lambda case: "move", host=host, k=1)
    scored = await run_eval(LABELLED, zero, [even], scope_id=SCOPE, host=host, k=1)
    assert first.template_id == again.template_id
    assert len({first.template_id, relabelled.template_id, scored.template_id}) == 3


# --- a classifier's refusals: each made before anything is stored ---------------------------------


async def test_a_run_with_no_scorer_expected_label_or_judge_is_refused() -> None:
    host = callable_host()
    with pytest.raises(
        ValueError, match=r"at least one scorer, a classifier's expected labels \(expected=\) or a judge"
    ):
        await run_eval(LABELLED, classify, scope_id=SCOPE, host=host)
    assert list_templates(host.storage, SCOPE) == []


@pytest.mark.parametrize("name", ["match", "confusion_cell", "accuracy"])
@pytest.mark.parametrize("classifies", [True, False], ids=["a classifier", "scorers alone"])
async def test_a_scorer_taking_a_classifier_measures_name_is_refused(name: str, classifies: bool) -> None:
    def grade(case: Mapping[str, Any], answer: Any) -> bool:
        return True

    grade.__name__ = name
    host = callable_host()
    with pytest.raises(ValueError, match=f"a scorer named {name} takes a measure the classifier track owns"):
        await run_eval(
            LABELLED, classify, [grade], scope_id=SCOPE, expected=expected_label if classifies else None, host=host
        )
    assert list_templates(host.storage, SCOPE) == []
    with pytest.raises(ValueError, match="takes a measure the classifier track owns"):
        callable_host([grade])


def _raises(case: Mapping[str, Any]) -> str:
    raise KeyError("label")


@pytest.mark.parametrize(
    ("expected", "said"),
    [
        (_raises, "expected= raised on case 0: KeyError"),
        (lambda case: None, "gave case 0 None; an expected label is a non-blank string"),
        (lambda case: "  ", "gave case 0 '  '; an expected label is a non-blank string"),
        (lambda case: 1, "gave case 0 1; an expected label is a non-blank string"),
        (lambda case: UNUSABLE_ANSWER, "the label an unusable answer is counted under"),
    ],
    ids=["raises", "None", "blank", "a number", "the unusable label"],
)
async def test_an_expected_label_no_confusion_matrix_could_hold_is_refused(expected: Any, said: str) -> None:
    host = callable_host()
    with pytest.raises(ValueError, match=said):
        await run_eval(LABELLED, classify, scope_id=SCOPE, expected=expected, host=host)
    assert list_templates(host.storage, SCOPE) == []


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
        await run_eval(LABELLED, classify, scope_id=SCOPE, expected=expected_label, host=host)
    assert list_templates(host.storage, SCOPE) == []


async def test_a_classifiers_scorer_the_callers_host_declares_no_measure_for_is_refused() -> None:
    """``match`` and ``confusion_cell`` are core, so the host declares neither; a scorer's measure it must."""
    host = courier_host()
    with pytest.raises(ValueError, match="declares no measure named even"):
        await run_eval(LABELLED, classify, [even], scope_id=SCOPE, expected=expected_label, host=host)
    assert list_templates(host.storage, SCOPE) == []
    summary = await run_eval(LABELLED, classify, scope_id=SCOPE, expected=expected_label, host=host, k=1)
    assert summary.status == "completed" and summary.confusion
