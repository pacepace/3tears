"""The check record: what decides one, which check speaks for a judge, and that the tier no longer reaches a reader.

Held here, through ``analysis/checks.py`` and the quick print:

- which check speaks for a judge on a criterion is the design's table, row for row (people once decided; else a
  decided check against a model, unless that model is shown below the bar against people; else none);
- a model's own check against people returns each of the four criterion states and no other, from blind reads with 0,
  15, 40 and 200 pair items read by both, and is ``met`` only with its lean shown inside ``±δ`` by the bounded test,
  ``δ`` the package's bar recorded on the check, never a margin a caller passes;
- an undecided check carries the bound the equivalence test reached (the larger lean it cannot rule out, not a t
  interval's half-width) and about how many pair items like these would decide it, and the print states that bound;
- pair items from a run with an unrecorded coordinate are left out of the lean, and counted; where every one was, the
  check says so by the shared reason and its coordinate;
- one model id answering under two served models is two reviewers, never pooled; answers whose served model was not
  recorded are left out of every figure, and counted;
- a model that answered changes no stored tier: a judge's tier reads as it did with no model check;
- a check is one of three kinds, each refusing another's fields, and names its kind; it carries a reason only where its
  result does;
- the decision that sets an improvement aside reads the judge's checks, not its tier;
- no line the quick print says of the judge carries a tier value, "calibrated", "standing" or "anchor".
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import TypeAdapter, ValidationError

from threetears.evals.analysis.agreement import (
    judge_agreement,
    judge_evidence_tiers,
    judge_instrument,
    judge_key,
    judge_self_agreement,
)
from threetears.evals.analysis.blind_read import close_blind_read, draw_blind_read
from threetears.evals.analysis.checks import (
    REVIEWER_LEAN_BAR,
    Check,
    JudgeCheck,
    ReviewerCheck,
    RigCheck,
    blind_read_checks,
    judge_checks,
    reviewer_check,
    speaking_check,
    tier_checks,
)
from threetears.evals.analysis.caveats import caveat_step, raise_caveats, remedy
from threetears.evals.analysis.analyst.facts import caveats_list
from threetears.evals.analysis.report.call import every_caveat, shown_better
from threetears.evals.analysis.report_input import CaveatEntry, ReportInput
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.analysis.stats import paired_equivalence
from threetears.evals.kernel.evidence_tiers import CriterionState, tier_of
from threetears.evals import Eval
from threetears.evals.kernel.host import EvalHost
from threetears.evals.quick import as_host
from threetears.evals.quick.blind import judge_lines
from threetears.evals.schema import EvalResult, JudgeEvidence, RubricDim, RubricScore, label_key_of
from threetears.evals.schema.blind_read import BlindAnswer, BlindArm, BlindRead, BlindVerdict
from threetears.evals.schema.hashing import input_digest
from threetears.evals.schema.reading_states import NotAvailableReason
from packages.evals.tests.factories import make_eval_result

GROUNDED = RubricDim(name="answer.grounded", description="Every fact is in the policy.", scale="pass_fail")
JUDGE = "acme/judge-1"
REVIEWER = "other/reviewer-9"
CONTROL_RUN, CANDIDATE_RUN = "run-control", "run-candidate"
JUDGE_INSTRUMENT = "the-judge"


CHECK: TypeAdapter[Check] = TypeAdapter(Check)


def _check(against: str, result: str, reason: str | None = None, **extra: Any) -> Check:
    if against == "reviewer":
        shape = {"kind": "reviewer", "instrument": REVIEWER}
    else:
        shape = {"kind": "judge", "instrument": JUDGE_INSTRUMENT, "against": against}
        shape |= {"against_id": REVIEWER} if against == "model" else {}
    return CHECK.validate_python({**shape, "criterion": GROUNDED.name, "result": result, "reason": reason, **extra})


_REASON = {"not_met": "instrument_below_bar", "insufficient": "too_few"}


# --- which check speaks: the design's table -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("people", "model", "reviewer", "speaks"),
    [
        # Checked against people, with that result: the model check is silenced.
        ("met", "met", None, "people"),
        ("met", "not_met", None, "people"),
        ("not_met", "met", None, "people"),
        ("not_met", None, None, "people"),
        # People undecided or too few: a decided check against a model speaks.
        ("undecided", "met", None, "model"),
        ("undecided", "not_met", None, "model"),
        ("insufficient", "met", None, "model"),
        ("insufficient", "not_met", None, "model"),
        (None, "met", None, "model"),
        # Nothing decided: not checked.
        ("undecided", "undecided", None, None),
        ("insufficient", "insufficient", None, None),
        ("insufficient", None, None, None),
        (None, None, None, None),
        # The model shown below the bar against people: the judge reads as not checked against it.
        ("undecided", "met", ("not_met", "instrument_below_bar"), None),
        # The model shown to lean still speaks: its lean sets an improvement aside where the decision is read.
        ("undecided", "met", ("not_met", "instrument_leans"), "model"),
        ("insufficient", "met", ("undecided", None), "model"),
    ],
)
def test_which_check_speaks_for_a_judge_is_the_design_s_table(
    people: str | None, model: str | None, reviewer: tuple[str, str | None] | None, speaks: str | None
) -> None:
    checks = []
    if people is not None:
        checks.append(_check("people", people, _REASON.get(people)))
    if model is not None:
        checks.append(_check("model", model, _REASON.get(model)))
    if reviewer is not None:
        checks.append(_check("reviewer", *reviewer))
    spoken = speaking_check(checks, JUDGE_INSTRUMENT, GROUNDED.name)
    assert (spoken.against if spoken is not None else None) == speaks
    # Another judge's checks never speak for this one.
    assert speaking_check(checks, "another-judge", GROUNDED.name) is None


# --- a model's own check against people -----------------------------------------------------------------------------


def _agreeing_pairs(outputs: int, *, disagree: int = 0) -> list[tuple[int, int, str, str]]:
    """``outputs`` outputs both read, half passed, the model and the person disagreeing on ``disagree`` of them."""
    pairs = []
    for n in range(outputs):
        person = n % 2
        mine = 1 - person if n < disagree else person
        pairs.append((mine, person, "owner", f"output-{n}"))
    return pairs


def _leans(items: int, *, toward: int = 0, away: int = 0) -> list[float]:
    """Pair-item leans: ``toward`` of them +1 (the model alone passed the candidate), ``away`` -1, the rest 0."""
    return [1.0] * toward + [-1.0] * away + [0.0] * (items - toward - away)


@pytest.mark.parametrize(
    ("items", "leans", "expected"),
    [
        (0, _leans(0), ("insufficient", "not_recorded")),
        (15, _leans(15, toward=1), ("undecided", None)),
        (40, _leans(40, toward=12), ("not_met", "instrument_leans")),
        (200, _leans(200, toward=3, away=3), ("met", None)),
    ],
)
def test_a_model_s_own_check_reads_each_of_the_four_states_by_its_pair_items(
    items: int, leans: list[float], expected: tuple[str, str | None]
) -> None:
    check = reviewer_check(
        REVIEWER, GROUNDED.name, "pass_fail", kappa_pairs=_agreeing_pairs(2 * max(items, 20)), leans=leans
    )
    assert (check.result, check.reason) == expected
    assert (check.kind, check.instrument, check.against, check.pair_items) == ("reviewer", REVIEWER, "people", items)
    assert check.lean_bar == pytest.approx(REVIEWER_LEAN_BAR * 1.0), "the bar it was read against is on the check"
    if items:
        assert check.lean is not None
    if items >= 2:
        assert check.lean_interval is not None
    # Only the undecided check carries the bound it reached and the items it needs.
    assert (check.lean_bound is not None) == (check.result == "undecided")
    assert (check.pair_items_needed is not None or check.pair_items_needed_above is not None) == (
        check.result == "undecided"
    )


def test_met_is_an_equivalence_reading_held_to_the_package_s_bar() -> None:
    quiet = _leans(200, toward=3, away=3)
    met = reviewer_check(REVIEWER, GROUNDED.name, "pass_fail", kappa_pairs=_agreeing_pairs(400), leans=quiet)
    assert met.result == "met" and met.lean_bar == pytest.approx(0.10)
    assert paired_equivalence(quiet, 0.10, value_range=(-1.0, 1.0))[0] is True
    # The bar is the package's, a tenth of the scale's width: on 1-5 it is 0.4 points, and no caller passes another.
    on_five = reviewer_check(REVIEWER, GROUNDED.name, "ordinal", kappa_pairs=[], leans=[0.0, 1.0])
    assert on_five.lean_bar == pytest.approx(0.4)
    with pytest.raises(TypeError):
        reviewer_check(  # type: ignore[call-arg]
            REVIEWER, GROUNDED.name, "pass_fail", kappa_pairs=_agreeing_pairs(400), leans=quiet, margin=5.0
        )
    # Quiet, and fewer items than any valid test needs at this bar (0.95 ** n stays above 5% below 59): undecided.
    few = reviewer_check(REVIEWER, GROUNDED.name, "pass_fail", kappa_pairs=_agreeing_pairs(400), leans=_leans(50))
    assert few.result == "undecided"
    # Quiet and inside, but a kappa too few outputs decide: undecided.
    loose = reviewer_check(REVIEWER, GROUNDED.name, "pass_fail", kappa_pairs=_agreeing_pairs(15), leans=quiet)
    assert loose.result == "undecided"
    assert loose.pair_items_needed == 200, "the lean already shows: its agreement with people is what is undecided"
    # A kappa shown below the bar outranks everything: the judge is not checked against this model.
    below = reviewer_check(
        REVIEWER, GROUNDED.name, "pass_fail", kappa_pairs=_agreeing_pairs(200, disagree=100), leans=quiet
    )
    assert (below.result, below.reason) == ("not_met", "instrument_below_bar")


def test_an_undecided_check_carries_the_bound_the_test_reached_and_the_items_it_needs() -> None:
    # A lean of 0.03 with a t interval of about [-0.03, 0.09]: the lean not ruled out is the interval's far end or
    # more, never its half-width.
    leans = _leans(100, toward=6, away=3)
    check = reviewer_check(REVIEWER, GROUNDED.name, "pass_fail", kappa_pairs=_agreeing_pairs(400), leans=leans)
    assert check.result == "undecided" and check.lean_interval is not None and check.lean_bound is not None
    low, high = check.lean_interval
    assert check.lean_bound >= max(abs(low), abs(high)) > (high - low) / 2
    # The bound is where the bounded test turns: shown inside it, not shown a little inside it.
    assert paired_equivalence(leans, check.lean_bound, value_range=(-1.0, 1.0))[0] is True
    assert paired_equivalence(leans, check.lean_bound - 0.002, value_range=(-1.0, 1.0))[0] is False
    # Items like these, repeated to the count it names, are shown inside the bar, and those in hand are not. (The
    # bounded test bets in an order fixed by the count, so near the count it turns more than once: a plan's size.)
    needed = check.pair_items_needed
    assert needed is not None and needed > 100 and check.pair_items_needed_above is None

    def shown(count: int) -> bool | None:
        return paired_equivalence([leans[i % 100] for i in range(count)], 0.10, value_range=(-1.0, 1.0))[0]

    assert shown(needed) is True and shown(100) is False
    # A lean at the bar or past it is shown inside it at no count of items like these: the check carries no count.
    at_bar = reviewer_check(
        REVIEWER, GROUNDED.name, "pass_fail", kappa_pairs=_agreeing_pairs(400), leans=_leans(30, toward=3)
    )
    assert at_bar.result == "undecided"
    assert (at_bar.pair_items_needed, at_bar.pair_items_needed_above) == (None, None)
    with pytest.raises(ValidationError, match="at no count"):
        at_bar.model_validate({**at_bar.model_dump(), "pair_items_needed_above": 1000})
    # Each projection belongs to an undecided check only.
    with pytest.raises(ValidationError, match="only an undecided check"):
        _check("reviewer", "met", lean_bound=0.2)
    with pytest.raises(ValidationError, match="says how many it needs"):
        _check("reviewer", "undecided", pair_items=12)


def test_no_fifth_result_and_each_result_carries_only_its_own_reasons() -> None:
    assert set(get_args(CriterionState)) == {"met", "not_met", "undecided", "insufficient"}
    seen = {
        reviewer_check(
            REVIEWER,
            GROUNDED.name,
            "pass_fail",
            kappa_pairs=_agreeing_pairs(2 * max(items, 20), disagree=disagree),
            leans=_leans(items, toward=toward),
        ).result
        for items in (0, 5, 15, 40, 120, 200)
        for toward in (0, 2, 12)
        if toward <= items
        for disagree in (0, 4)
    }
    assert seen == set(get_args(CriterionState))
    with pytest.raises(ValidationError):
        _check("people", "leans")
    with pytest.raises(ValidationError, match="carries no reason"):
        _check("people", "met", "instrument_below_bar")
    with pytest.raises(ValidationError, match="sets a reading aside"):
        _check("people", "not_met", "too_few")
    with pytest.raises(ValidationError, match="says why"):
        _check("people", "insufficient")
    with pytest.raises(ValidationError, match="names its coordinate"):
        _check("model", "insufficient", "identity_unrecorded")


#: One field each kind holds and the others do not.
_OWN_FIELD: dict[str, tuple[str, Any]] = {
    "judge": ("same_family_as_judge", True),
    "reviewer": ("pair_items", 3),
    "rig": ("covers_arms", ["v2"]),
}

#: The least each kind is built from.
_LEAST: dict[str, dict[str, Any]] = {
    "judge": {"kind": "judge", "instrument": JUDGE_INSTRUMENT, "against": "people", "result": "met"},
    "reviewer": {"kind": "reviewer", "instrument": REVIEWER, "result": "insufficient", "reason": "not_recorded"},
    "rig": {"kind": "rig", "instrument": "launch-1", "result": "met"},
}


@pytest.mark.parametrize("kind", sorted(_LEAST))
def test_each_kind_of_check_holds_its_own_fields_and_refuses_another_s(kind: str) -> None:
    shape = {"judge": JudgeCheck, "reviewer": ReviewerCheck, "rig": RigCheck}[kind]
    held = CHECK.validate_python(_LEAST[kind])
    assert type(held) is shape and CHECK.validate_python(held.model_dump(mode="json")) == held
    name, value = _OWN_FIELD[kind]
    assert getattr(CHECK.validate_python({**_LEAST[kind], name: value}), name) == value
    for other, (foreign, foreign_value) in _OWN_FIELD.items():
        if other != kind:
            with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
                CHECK.validate_python({**_LEAST[kind], foreign: foreign_value})
    # A check names its kind: none is filled in, from what it was checked against or otherwise.
    with pytest.raises(ValidationError, match="kind"):
        CHECK.validate_python({key: field for key, field in _LEAST[kind].items() if key != "kind"})
    # Each kind is checked only against what it is checked against.
    against = {"judge": "identical_arms", "reviewer": "model", "rig": "people"}[kind]
    with pytest.raises(ValidationError):
        CHECK.validate_python({**_LEAST[kind], "against": against})


# --- one blind read, end to end -------------------------------------------------------------------------------------


def _campaign(cases: int, *, served: str | None = JUDGE) -> tuple[list[EvalResult], dict[str, JudgeEvidence]]:
    results: list[EvalResult] = []
    evidence: dict[str, JudgeEvidence] = {}
    for case in range(cases):
        for run, text in ((CONTROL_RUN, f"old reply to {case}"), (CANDIDATE_RUN, f"new reply to {case}")):
            read = JudgeEvidence(case_material=f"Question {case}.", artifact=text)
            key = label_key_of(read, GROUNDED)
            result = make_eval_result(
                id=f"res-{run}-{case}",
                eval_run_id=run,
                test_case_id=f"case-{case:03d}",
                rubric_scores=[
                    RubricScore(
                        dim=GROUNDED.name,
                        score=int(case % 3 != 0 or run == CANDIDATE_RUN),
                        scale="pass_fail",
                        axis="capability",
                        served_model=served,
                        judge_temperature=0.0,
                        output_fingerprint=key.output_fingerprint,
                        criterion_fingerprint=key.criterion_fingerprint,
                    )
                ],
            )
            results.append(result)
            evidence[result.id] = read
    return results, evidence


def _read(results: list[EvalResult], evidence: dict[str, JudgeEvidence]) -> BlindRead:
    return draw_blind_read(
        results,
        evidence,
        [GROUNDED],
        [
            BlindArm(key="v1", role="control", run_ids=[CONTROL_RUN]),
            BlindArm(key="v2", role="candidate", run_ids=[CANDIDATE_RUN]),
        ],
        scope_id="scope",
        created_by="test",
        created_at="t0",
    )


def _judged(read: BlindRead, results: list[EvalResult], position: int) -> dict[str, int]:
    """What the judge gave each arm's output of one item."""
    by_id = {result.id: result for result in results}
    item = read.items[position]
    return {
        output.arm: score.score
        for output in item.outputs
        if (score := by_id[output.result_id].judge_score(GROUNDED.name)) is not None
    }


def _answer(read: BlindRead, position: int, scores: dict[str, int], *, model: bool = False) -> BlindAnswer:
    item = read.items[position]
    return BlindAnswer(
        item=item.item,
        rater=REVIEWER if model else "owner",
        rater_kind="reference" if model else "person",
        reviewer_served_model=f"{REVIEWER}-0613" if model else None,
        answered_at="t1",
        verdicts=[BlindVerdict(slot=o.slot, dim=GROUNDED.name, score=scores[o.arm]) for o in item.outputs],
    )


def _both_read(cases: int, people: int, *, lean: bool = False) -> tuple[BlindRead, list[EvalResult]]:
    """A read the model answered whole and a person answered from the top, each as the judge scored it. A leaning
    model fails the control's answer on every third item where everyone else passed it: a lean toward the candidate
    that people do not share."""
    results, evidence = _campaign(cases)
    read = _read(results, evidence)
    answers = []
    for position in range(cases):
        judged = _judged(read, results, position)
        mine = dict(judged)
        if lean and judged["v1"] == 1 and position % 3 == 0:
            mine["v1"] = 0
        answers.append(_answer(read, position, mine, model=True))
        if position < people:
            answers.append(_answer(read, position, judged))
    return read.model_copy(update={"answers": answers}), results


def test_a_read_yields_the_judge_against_people_and_the_model_and_the_model_against_people() -> None:
    read, results = _both_read(30, 30)
    closed = close_blind_read(read, results)
    checks = blind_read_checks(
        closed,
        results,
        family_of=_host().family_of,
        candidate_models={"v1": "acme/small", "v2": "other/large"},
    )
    by = {(check.kind, check.against): check for check in checks}
    assert set(by) == {("judge", "people"), ("judge", "model"), ("reviewer", "people")}
    people, model, own = by[("judge", "people")], by[("judge", "model")], by[("reviewer", "people")]
    key = judge_key(results[0], GROUNDED.name)
    assert key is not None and people.instrument == model.instrument == judge_instrument(key)
    assert (people.result, people.sample, people.reads, people.sample_cases) == ("met", "random", 60, 30)
    assert isinstance(model, JudgeCheck) and isinstance(own, ReviewerCheck)
    assert (model.result, model.against_id, model.same_family_as, model.same_family_as_judge) == (
        "met",
        REVIEWER,
        ["v2"],
        False,
    )
    assert (model.against_served_model, own.served_model) == (f"{REVIEWER}-0613", f"{REVIEWER}-0613")
    # The model and the person agree on every output, and neither leans: undecided at 30 pair items, fewer than the
    # bounded test can show inside a tenth of the width at.
    assert (own.result, own.pair_items, own.left_out, own.lean, own.reads) == ("undecided", 30, 0, 0.0, 60)
    # People decided: their check speaks for the judge.
    assert speaking_check(checks, people.instrument, GROUNDED.name) == people


def test_pair_items_from_a_run_with_an_unrecorded_coordinate_are_left_out_of_the_lean_and_counted() -> None:
    read, results = _both_read(30, 30, lean=True)
    closed = close_blind_read(read, results)
    whole = next(c for c in blind_read_checks(closed, results) if c.kind == "reviewer")
    assert (whole.pair_items, whole.left_out, whole.result, whole.reason) == (30, 0, "not_met", "instrument_leans")
    left_out = blind_read_checks(closed, results, unrecorded_runs={CANDIDATE_RUN: "template"})
    unrecorded = next(c for c in left_out if isinstance(c, ReviewerCheck))
    assert isinstance(whole, ReviewerCheck)
    assert (unrecorded.pair_items, unrecorded.left_out, unrecorded.lean) == (0, 30, None)
    # Every item was left out, and the check says why by the shared reason, naming the coordinate.
    assert (unrecorded.result, unrecorded.reason, unrecorded.coordinate) == (
        "insufficient",
        "identity_unrecorded",
        "template",
    )
    printed = "\n".join(judge_lines(closed.read, labels_name="labels.csv", labels_there=True, checks=left_out))
    assert "every case read on both arms was left out" in printed and "no case read on both arms" not in printed
    # Kappa is about an instrument reading texts: it still reads every output.
    assert unrecorded.kappa == whole.kappa and unrecorded.sample_results == whole.sample_results


def test_a_judge_s_own_model_checks_it_against_itself_and_an_unrecorded_one_checks_nothing() -> None:
    results, evidence = _campaign(24)
    read = _read(results, evidence)
    own = [
        _answer(read, n, _judged(read, results, n), model=True).model_copy(update={"reviewer_served_model": JUDGE})
        for n in range(24)
    ]
    closed = close_blind_read(read.model_copy(update={"answers": own}), results)
    (itself,) = [c for c in blind_read_checks(closed, results) if isinstance(c, JudgeCheck)]
    assert (itself.against, itself.against_id, itself.against_served_model) == ("itself", REVIEWER, JUDGE)
    # Held to the bar a judge's agreement with its repeats is: 120 results, which 48 do not reach.
    assert (itself.result, itself.reason) == ("insufficient", "too_few")
    unnamed, unnamed_evidence = _campaign(24, served=None)
    blank_read = _read(unnamed, unnamed_evidence)
    answers = [_answer(blank_read, n, _judged(blank_read, unnamed, n), model=True) for n in range(24)]
    blank = close_blind_read(blank_read.model_copy(update={"answers": answers}), unnamed)
    (nothing,) = [c for c in blind_read_checks(blank, unnamed) if isinstance(c, JudgeCheck)]
    assert (nothing.against, nothing.result, nothing.reason, nothing.coordinate) == (
        "model",
        "insufficient",
        "identity_unrecorded",
        "served_model",
    )
    assert nothing.kappa is None, "an unrecorded model matches nothing, so no figure is read"


def test_one_model_id_served_by_two_models_is_two_reviewers_and_an_unrecorded_one_is_left_out_and_counted() -> None:
    read, results = _both_read(30, 30)
    # The same model id, its answers served by one snapshot on the first 18 items, another on the next 8, and nobody
    # recorded which on the last 4.
    served = {**dict.fromkeys(range(18), f"{REVIEWER}-0613"), **dict.fromkeys(range(18, 26), f"{REVIEWER}-0901")}
    items = {item.item: position for position, item in enumerate(read.items)}
    answers = [
        answer.model_copy(update={"reviewer_served_model": served.get(items[answer.item])})
        if answer.rater_kind == "reference"
        else answer
        for answer in read.answers
    ]
    closed = close_blind_read(read.model_copy(update={"answers": answers}), results)
    checks = blind_read_checks(closed, results)
    against = {
        check.against_served_model: check
        for check in checks
        if isinstance(check, JudgeCheck) and check.against == "model"
    }
    own = {check.served_model: check for check in checks if isinstance(check, ReviewerCheck)}
    assert set(against) == set(own) == {f"{REVIEWER}-0613", f"{REVIEWER}-0901", None}
    # Each configuration reads only its own answers: 18 and 8 items, two outputs each.
    assert (against[f"{REVIEWER}-0613"].reads, against[f"{REVIEWER}-0901"].reads) == (36, 16)
    assert (own[f"{REVIEWER}-0613"].pair_items, own[f"{REVIEWER}-0901"].pair_items) == (18, 8)
    # The unrecorded answers are in no figure, and are counted.
    for check in (against[None], own[None]):
        assert (check.result, check.reason, check.coordinate, check.kappa, check.reads) == (
            "insufficient",
            "identity_unrecorded",
            "served_model",
            None,
            8,
        )
    # A model shown below the bar under one served model does not silence the judge's check against the other.
    below = own[f"{REVIEWER}-0901"].model_copy(update={"result": "not_met", "reason": "instrument_below_bar"})
    spoken = speaking_check(
        [against[f"{REVIEWER}-0613"].model_copy(update={"result": "met"}), below],
        against[f"{REVIEWER}-0613"].instrument,
        GROUNDED.name,
    )
    assert spoken is not None and spoken.against_served_model == f"{REVIEWER}-0613"
    printed = "\n".join(judge_lines(closed.read, labels_name="labels.csv", labels_there=True, checks=checks))
    assert f"a model ({REVIEWER}, served as {REVIEWER}-0901)" in printed
    assert f"not checked against a model ({REVIEWER}, on the answers that name no served model)" in printed


# --- the decision reads checks --------------------------------------------------------------------------------------

#: kairo's report input (r5): every judge stated undetermined and insufficient, so an improvement it reads is set aside.
_R5 = Path(__file__).resolve().parent / "fixtures" / "report_inputs" / "r5" / "input.json"


def _r5_with(checks: list[dict[str, Any]], *, tier: str) -> tuple[ReportInput, str]:
    """kairo's input with its candidate's outcome contrast improved, the outcome's judge stating ``tier`` and holding
    ``checks``."""
    data: dict[str, Any] = json.loads(_R5.read_text(encoding="utf-8"))
    contrast = next(c for c in data["contrasts"] if c["arm"] == "qwen3.8-flash" and c["measure"] == "__outcome__")
    contrast.update(aspect="character adherence (v2)", verdict="improved", delta=0.2, interval=[0.1, 0.3])
    data["judged"]["__outcome__"].update(tier=tier, calibration="met" if tier == "calibrated" else "insufficient")
    data["caveats"]["judge_tiers"]["__outcome__"] = tier
    data["judged"]["__outcome__"]["checks"] = checks
    return ReportInput.model_validate(data), "qwen3.8-flash"


def _judge(against: str, result: str, reason: str | None = None) -> dict[str, Any]:
    return {"kind": "judge", "instrument": JUDGE_INSTRUMENT, "against": against, "result": result, "reason": reason}


def test_an_improvement_is_set_aside_by_the_judge_s_checks_and_not_by_its_tier() -> None:
    # Its tier says undetermined, its checks say people agree with it: the improvement counts.
    checked, arm = _r5_with([_judge("people", "met"), _judge("itself", "insufficient", "too_few")], tier="undetermined")
    assert shown_better(checked, arm)
    # Its tier says calibrated, its checks say nothing has checked it: the improvement is set aside.
    unchecked, arm = _r5_with(
        [_judge("people", "insufficient", "too_few"), _judge("itself", "not_met", "instrument_below_bar")],
        tier="calibrated",
    )
    assert not shown_better(unchecked, arm)
    # Checked and found short is checked: its improvement counts, as a judge measured below its bar always did.
    short, arm = _r5_with(
        [_judge("people", "not_met", "instrument_below_bar"), _judge("itself", "insufficient", "too_few")],
        tier="undetermined",
    )
    assert shown_better(short, arm)
    # An input holding no check is read by the tier it states, as before checks were recorded.
    stated, arm = _r5_with([], tier="undetermined")
    assert not shown_better(stated, arm)


def _r5_checked(checks: dict[str, list[dict[str, Any]]]) -> ReportInput:
    """kairo's input, every judged dimension's judge stating ``undetermined``, the named dimensions holding ``checks``."""
    data: dict[str, Any] = json.loads(_R5.read_text(encoding="utf-8"))
    for dimension, held in checks.items():
        data["judged"][dimension]["checks"] = held
    return ReportInput.model_validate(data)


def test_a_judge_with_no_passing_check_against_people_is_said_once_by_its_check() -> None:
    """One fact, one caveat, in the one list: a judge's check against people too thin to decide says the judge has no
    passing check against people, with which check and what it reached; the tier's caveat is silenced by it there
    (``subsumed_by``), and still stated in the part the analyst and the writer read, where no check caveat is."""
    thin = [_judge("people", "insufficient", "too_few"), _judge("itself", "not_met", "instrument_below_bar")]
    dimensions = ["__outcome__", "__transcript__", "persona.booth_stays_offstage", "persona.holds_character"]
    every = _r5_checked({dimension: thin for dimension in dimensions})
    raised = {entry.id: entry for entry in raise_caveats(every)}
    people = [entry for entry in raised.values() if entry.kind == "check" and entry.against == "people"]
    assert len(people) == len(dimensions)
    assert all(entry.check == "insufficient" for entry in people)
    assert raised["judge_uncalibrated"].subsumed_by == min(entry.id for entry in people)
    assert "judge_partly_uncalibrated" not in raised
    assert [caveat.id for caveat in every_caveat(every)].count("judge_uncalibrated") == 1
    # The fact still decides: an improvement on such a judge is set aside, by its checks.
    set_aside, arm = _r5_with(thin, tier="undetermined")
    assert not shown_better(set_aside, arm)

    # The same dimension: a check on one dimension says that dimension's fact only. The other three hold no check,
    # so the tier's caveat still says something no check caveat does, and is not silenced.
    one = _r5_checked({"__outcome__": thin})
    raised = {entry.id: entry for entry in raise_caveats(one)}
    assert raised["judge_uncalibrated"].subsumed_by is None
    assert any(entry.kind == "check" and entry.against == "people" for entry in raised.values())


def test_a_judge_with_no_check_of_any_kind_is_said_uncalibrated_by_its_tier_as_before() -> None:
    stated = _r5_checked({})
    listed = {entry.id: entry for entry in raise_caveats(stated)}
    raised = set(listed)
    assert "judge_uncalibrated" in raised and listed["judge_uncalibrated"].subsumed_by is None
    assert "judge_partly_uncalibrated" not in raised
    assert not any(caveat_id.startswith("check:") for caveat_id in raised)
    # A judge checked and found short is checked: neither its tier's caveat nor a thin check's says it is unchecked.
    short = [_judge("people", "not_met", "instrument_below_bar"), _judge("itself", "insufficient", "too_few")]
    dimensions = ["__outcome__", "__transcript__", "persona.booth_stays_offstage", "persona.holds_character"]
    raised = {entry.id for entry in raise_caveats(_r5_checked({dimension: short for dimension in dimensions}))}
    assert not {"judge_uncalibrated", "judge_partly_uncalibrated"} & raised


@pytest.mark.parametrize("reason", get_args(NotAvailableReason))
def test_a_thin_check_against_people_is_closed_by_a_person_rating_answers_whatever_made_it_thin(reason: str) -> None:
    """The step that closes it is a person's ratings: never a launch of more cases, which adds no person's label, and
    never a rerun, which records none."""
    entry = CaveatEntry.model_validate(
        {
            "id": f"check:{GROUNDED.name}:{GROUNDED.name}:people",
            "kind": "check",
            "class": "C",
            "reach": "report",
            "against": "people",
            "criterion": GROUNDED.name,
            "check": "insufficient",
            "reason": reason,
        }
    )
    assert remedy(entry) == "check_against_people"
    step = caveat_step(entry)
    assert step.action == "result_rate_person" and not step.spends and step.planned_cases is None


def test_the_scaled_campaign_s_thin_people_check_offers_the_step_of_rating_its_judge_s_answers() -> None:
    """The scaled campaign's judge has no person's rating. The first screen says so once, by the check's caveat, with
    the step of a person rating its answers; the analyst's caveat list and the writer's page caveats, which read the
    stated part, say it once each, as the tier's caveat."""
    from packages.evals.tests.test_hot_path_oracle import subject
    from threetears.evals.analysis.service import assemble_campaign_bundle

    read = subject("scaled")
    built = report_input(assemble_campaign_bundle(read.host, read.campaign_id, read.scope_id))
    assert built.first_screen is not None and built.caveat_list is not None
    thin = [entry for entry in built.caveat_list if entry.against == "people" and entry.check == "insufficient"]
    assert thin, "the scaled campaign's judge has no person's rating"
    unchecked = {entry.id for entry in thin} | {"judge_uncalibrated", "judge_partly_uncalibrated"}
    # The first screen: exactly once, by the check's caveat, with the person's step.
    shown = [reason for reason in built.first_screen.reasons if reason in unchecked]
    assert len(shown) == 1 and shown[0] in {entry.id for entry in thin}
    steps = {step.closes: step for step in built.first_screen.steps}
    assert (steps[shown[0]].action, steps[shown[0]].remedy) == ("result_rate_person", "check_against_people")
    (tier,) = [entry for entry in built.caveat_list if entry.kind == "judge_uncalibrated"]
    assert tier.subsumed_by in {entry.id for entry in thin}
    # The writer's page caveats and the analyst's list: exactly once each.
    assert [caveat.id for caveat in every_caveat(built) if caveat.id in unchecked] == ["judge_uncalibrated"]
    assert [row.id for row in caveats_list(built).caveats if row.id in unchecked] == ["judge_uncalibrated"]


def test_a_judge_whose_checks_hold_none_against_people_is_said_uncalibrated_once_with_the_person_s_step() -> None:
    """A missing check against people is no check: with checks only against itself, none met, the judge is set aside
    as unchecked, and the uncalibrated caveat (no check caveat names it) says so once, closed by a person's ratings."""
    itself_only = [_judge("itself", "insufficient", "too_few")]
    dimensions = ["__outcome__", "__transcript__", "persona.booth_stays_offstage", "persona.holds_character"]
    held = _r5_checked({dimension: itself_only for dimension in dimensions})
    raised = {entry.id: entry for entry in raise_caveats(held)}
    assert "judge_uncalibrated" in raised and "judge_partly_uncalibrated" not in raised
    assert not any(entry.kind == "check" for entry in raised.values())
    assert caveat_step(raised["judge_uncalibrated"]).action == "result_rate_person"
    set_aside, arm = _r5_with(itself_only, tier="calibrated")
    assert not shown_better(set_aside, arm)
    # One dimension so, beside three whose thin check says it: no check caveat says the first's, so the tier's
    # caveat is not silenced.
    thin = [_judge("people", "insufficient", "too_few")]
    mixed = _r5_checked({"__outcome__": itself_only, **{dimension: thin for dimension in dimensions[1:]}})
    listed = {entry.id: entry for entry in raise_caveats(mixed)}
    assert listed["judge_uncalibrated"].subsumed_by is None
    # Met against itself, it is checked: neither caveat.
    met = _r5_checked({dimension: [_judge("itself", "met")] for dimension in dimensions})
    assert not {"judge_uncalibrated", "judge_partly_uncalibrated"} & {entry.id for entry in raise_caveats(met)}


def test_an_input_with_no_check_is_read_by_the_checks_its_stated_tier_implies() -> None:
    """The stated tier, read once as checks: undetermined and insufficient is no check; separation is checked; an input
    that states nothing about a judge is not stated uncalibrated (that it should read not recorded is the owner's)."""
    from threetears.evals.analysis.answer import judge_described, stated_uncalibrated

    for tier, calibration, unchecked in (
        ("undetermined", "insufficient", True),
        ("undetermined", None, True),
        ("undetermined", "undecided", False),
        ("separation", "insufficient", False),
        ("incidental", "insufficient", False),
        ("calibrated", None, False),
    ):
        data: dict[str, Any] = json.loads(_R5.read_text(encoding="utf-8"))
        data["judged"]["__outcome__"].update(tier=tier, calibration=calibration)
        data["caveats"]["judge_tiers"]["__outcome__"] = tier
        assert stated_uncalibrated(ReportInput.model_validate(data), "__outcome__") is unchecked, (tier, calibration)
    data = json.loads(_R5.read_text(encoding="utf-8"))
    data["judged"]["__outcome__"].update(tier=None, calibration=None, judge_model=None)
    del data["caveats"]["judge_tiers"]["__outcome__"]
    silent = ReportInput.model_validate(data)
    assert not stated_uncalibrated(silent, "__outcome__") and not judge_described(silent, "__outcome__")


def test_a_stored_check_leaves_its_unset_fields_out_and_reads_the_same_either_way() -> None:
    check = _check("people", "insufficient", "too_few")
    stored = CHECK.dump_python(check, mode="json")
    assert None not in stored.values() and [] not in stored.values()
    assert set(stored) == {"kind", "instrument", "against", "criterion", "result", "reason"}
    with_nulls = {**{name: None for name in type(check).model_fields}, **stored, "same_family_as": []}
    assert CHECK.validate_python(with_nulls) == CHECK.validate_python(stored) == check
    assert input_digest(stored) == input_digest(dict(reversed(list(stored.items()))))


def _printed(read: BlindRead, results: list[EvalResult], own: ReviewerCheck) -> str:
    """The judge's lines with the model's own check replaced by ``own``, as one run of words."""
    closed = close_blind_read(read, results)
    checks = [check for check in blind_read_checks(closed, results) if not isinstance(check, ReviewerCheck)]
    lines = judge_lines(closed.read, labels_name="labels.csv", labels_there=True, checks=[*checks, own])
    return " ".join(" ".join(lines).split())


def _own(leans: list[float]) -> ReviewerCheck:
    check = reviewer_check(
        REVIEWER,
        GROUNDED.name,
        "pass_fail",
        kappa_pairs=_agreeing_pairs(400),
        leans=leans,
        served_model=f"{REVIEWER}-0613",
    )
    assert isinstance(check, ReviewerCheck) and check.result == "undecided"
    return check


def test_the_undecided_line_labels_its_count_a_projection_and_says_where_the_read_cannot_supply_it() -> None:
    own = _own(_leans(30))
    assert own.pair_items_needed is not None and own.pair_items_needed > 30
    more = own.pair_items_needed - 30
    projection = (
        f"if further cases lean as these did, about {more} more read on both arms would show it inside the bar, and "
        "a fresh sample may not"
    )
    # The read holds the cases: the projection, and nothing about a larger run.
    roomy = _printed(*_both_read(30 + more + 5, 30), own)
    assert projection in roomy and "a run with more cases" not in roomy
    # The read holds fewer than it would take.
    short = _printed(*_both_read(30 + more - 4, 30), own)
    assert projection in short
    assert f"this read holds {more - 4} more case(s) to read on both arms, so the rest needs a run with more cases" in (
        short
    )
    # The read holds none.
    whole = _printed(*_both_read(30, 30), own)
    assert "this read holds no more case to read on both arms, so they need a run with more cases" in whole


def test_the_undecided_line_says_a_lean_at_or_past_the_bar_is_not_decided_and_why_no_count_is_given() -> None:
    own = _own(_leans(9, toward=3))
    assert own.lean is not None and own.lean_bar is not None and abs(own.lean) >= own.lean_bar
    assert own.lean_interval is not None
    printed = _printed(*_both_read(30, 30), own)
    low, high = own.lean_interval
    assert (
        f"lean 0.33 over 9 case(s) read on both arms: not decided: the lean seen, 0.33 [{low:.2f}, {high:.2f}], is at "
        "or past the bar of 0.10, so no count is given: more cases decide it either way, shown inside the bar if they "
        "lean less, shown to lean if they lean as these did"
    ) in printed
    assert "fails" not in printed and "more than" not in printed and "only if" not in printed


def test_the_undecided_line_states_the_bound_the_test_reached_not_the_half_width() -> None:
    own = _own(_leans(30))
    assert own.lean_bound is not None and own.lean_interval is not None
    printed = _printed(*_both_read(30, 30), own)
    assert f"a lean toward either arm up to {own.lean_bound:.2f} is not ruled out (the bar of 0.10)" in printed
    half = (own.lean_interval[1] - own.lean_interval[0]) / 2
    assert f"{own.lean_bound:.2f}" != f"{half:.2f}"


def test_the_undecided_line_says_what_remains_is_the_agreement_with_people() -> None:
    own = reviewer_check(
        REVIEWER,
        GROUNDED.name,
        "pass_fail",
        kappa_pairs=_agreeing_pairs(40, disagree=4),
        leans=_leans(200),
        served_model=f"{REVIEWER}-0613",
    )
    assert isinstance(own, ReviewerCheck) and own.result == "undecided" and own.pair_items_needed == 200
    printed = _printed(*_both_read(30, 30), own)
    assert "is not ruled out (the bar of 0.10); its agreement with people is what is not yet decided" in printed
    assert "if further cases lean" not in printed


def test_the_undecided_line_says_more_than_the_search_read_where_no_count_up_to_it_shows_the_lean_inside() -> None:
    own = _own(_leans(100, toward=30, away=21))
    assert (own.pair_items_needed, own.pair_items_needed_above) == (None, 1000)
    printed = _printed(*_both_read(30, 30), own)
    assert "if further cases lean as these did, more than 900 more read on both arms would be needed" in printed


# --- the tier stays what it was ---------------------------------------------------------------------------------------


def test_a_model_check_moves_no_stored_tier_and_the_tier_is_its_people_and_itself_checks() -> None:
    read, results = _both_read(30, 30)
    closed = close_blind_read(read, results)
    judged = {key for result in results if (key := judge_key(result, GROUNDED.name)) is not None}
    people = [rating for rating in closed.ratings if rating.rater_kind == "person"]
    (tier,) = judge_evidence_tiers(judge_agreement(people, results), judge_self_agreement(results), judged)
    # The stored tier, byte for byte, as the tier code wrote it before checks existed: the pin was computed from this
    # very fixture by the tree at f9ab6f74d (origin/evals/main before this change), and a model check moves none of it.
    stored = json.dumps(tier.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    assert tier.tier == "calibrated"
    assert hashlib.sha256(stored.encode()).hexdigest()[:16] == "e24ea1f9fc4f8e6e", stored
    # The model answered every case and agrees with the judge: the tier does not move.
    assert any(check.against == "model" and check.result == "met" for check in blind_read_checks(closed, results))
    (again,) = judge_evidence_tiers(judge_agreement(people, results), judge_self_agreement(results), judged)
    assert again == tier
    # The tier is the two checks' results as tier_of reads them, and nothing else.
    people_check, itself_check = tier_checks(tier)
    assert (people_check.against, itself_check.against) == ("people", "itself")
    assert tier_of(tier.calibration, tier.separation) == tier.tier
    assert (people_check.result, itself_check.result) == (tier.calibration.state, tier.separation.state)
    assert judge_checks([tier], judged) == [people_check, itself_check]
    # A judge with no tier was measured by nothing.
    unmeasured = judge_checks([], judged)
    assert [(c.result, c.reason, c.sample) for c in unmeasured] == [("insufficient", "not_recorded", "none")] * 2


# --- the family of a model ------------------------------------------------------------------------------------------


def _graded(case: Any, answer: Any) -> float:
    return 1.0


async def _answer_case(case: Any) -> float:
    return 1.0


def _host(family: Callable[[str], str | None] | None = None) -> EvalHost:
    """A host in memory, saying ``family`` of a model, or nothing."""
    return replace(as_host([Eval([{"q": "a"}], _answer_case, [_graded])], store=":memory:"), model_family=family)


def test_a_model_s_family_is_the_host_s_word_else_its_provider_else_not_recorded() -> None:
    bare = _host()
    assert bare.family_of("acme/judge-1") == "acme" and bare.family_of("judge-1") is None
    told = _host(lambda model: "lineage-a" if "judge" in model else None)
    assert told.family_of("acme/judge-1") == "lineage-a" and told.family_of("acme/other") is None


# --- the print ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("people", [0, 3, 30])
def test_no_line_the_print_says_of_the_judge_names_a_tier_or_an_anchor(people: int) -> None:
    read, results = _both_read(30, people, lean=people == 3)
    closed = close_blind_read(read, results)
    lines = judge_lines(
        closed.read, labels_name="labels.csv", labels_there=True, checks=blind_read_checks(closed, results)
    )
    assert lines
    printed = "\n".join(lines)
    for word in ("calibrated", "tier", "standing", "anchor", "undetermined", "incidental", "separation"):
        assert word not in printed, (word, printed)
    if people == 30:
        assert any("checked against people: agrees with people" in line for line in lines)
    assert any(line.strip().startswith(f"{REVIEWER} against people on {GROUNDED.name}") for line in lines) == bool(
        people
    )
