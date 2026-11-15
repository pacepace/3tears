"""The check record: what decides one, which check speaks for a judge, and that the tier no longer reaches a reader.

Held here, through ``analysis/checks.py`` and the quick print:

- which check speaks for a judge on a criterion is the design's table, row for row (people once decided; else a
  decided check against a model, unless that model is shown below the bar against people; else none);
- a model's own check against people returns each of the four criterion states and no other, from blind reads with 0,
  15, 40 and 200 pair items read by both, and is ``met`` only with its lean's interval inside ``±δ``;
- pair items from a run with an unrecorded coordinate are left out of the lean, and counted;
- a model that answered changes no stored tier: a judge's tier reads as it did with no model check;
- a check carries a reason only where its result does, and the rig alone is checked against identical arms;
- no line the quick print says of the judge carries a tier value, "calibrated", "standing" or "anchor".
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from threetears.evals.analysis.agreement import (
    judge_agreement,
    judge_evidence_tiers,
    judge_instrument,
    judge_key,
    judge_self_agreement,
)
from threetears.evals.analysis.blind_read import close_blind_read, draw_blind_read
from threetears.evals.analysis.checks import (
    REVIEWER_MET_MIN_PAIR_ITEMS,
    Check,
    blind_read_checks,
    judge_checks,
    reviewer_check,
    speaking_check,
    tier_checks,
)
from threetears.evals.kernel.evidence_tiers import CriterionState, tier_of
from threetears.evals import Eval
from threetears.evals.kernel.host import EvalHost
from threetears.evals.quick import as_host
from threetears.evals.quick.blind import judge_lines
from threetears.evals.schema import EvalResult, JudgeEvidence, RubricDim, RubricScore, label_key_of
from threetears.evals.schema.blind_read import BlindAnswer, BlindArm, BlindRead, BlindVerdict
from packages.evals.tests.factories import make_eval_result

GROUNDED = RubricDim(name="answer.grounded", description="Every fact is in the policy.", scale="pass_fail")
JUDGE = "acme/judge-1"
REVIEWER = "other/reviewer-9"
CONTROL_RUN, CANDIDATE_RUN = "run-control", "run-candidate"
JUDGE_INSTRUMENT = "the-judge"


def _check(against: str, result: str, reason: str | None = None, **extra: Any) -> Check:
    return Check.model_validate(
        {
            "instrument": JUDGE_INSTRUMENT if against != "reviewer" else REVIEWER,
            "kind": "reviewer" if against == "reviewer" else "judge",
            "criterion": GROUNDED.name,
            "against": "people" if against == "reviewer" else against,
            "against_id": REVIEWER if against == "model" else None,
            "result": result,
            "reason": reason,
            **extra,
        }
    )


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
    check = reviewer_check(REVIEWER, GROUNDED.name, "pass_fail", kappa_pairs=_agreeing_pairs(2 * max(items, 20)), leans=leans)
    assert (check.result, check.reason) == expected
    assert (check.kind, check.instrument, check.against, check.pair_items) == ("reviewer", REVIEWER, "people", items)
    if items:
        assert check.lean is not None
    if items >= 2:
        assert check.lean_interval is not None and check.half_width is not None


def test_met_is_an_equivalence_reading_and_needs_the_interval_inside_the_bound() -> None:
    quiet = _leans(200, toward=3, away=3)
    met = reviewer_check(REVIEWER, GROUNDED.name, "pass_fail", kappa_pairs=_agreeing_pairs(400), leans=quiet)
    assert met.result == "met" and met.lean_interval is not None
    assert -0.10 < met.lean_interval[0] and met.lean_interval[1] < 0.10
    # The same items held to a margin the interval does not sit inside: undecided, never met.
    tight = reviewer_check(
        REVIEWER, GROUNDED.name, "pass_fail", kappa_pairs=_agreeing_pairs(400), leans=quiet, margin=0.02
    )
    assert (tight.result, tight.reason) == ("undecided", None)
    # Quiet and inside the bound at the floor of pair items, met; one item under it, undecided.
    at_floor, under_floor = (
        reviewer_check(
            REVIEWER,
            GROUNDED.name,
            "pass_fail",
            kappa_pairs=_agreeing_pairs(400),
            leans=_leans(items, toward=1, away=1),
        )
        for items in (REVIEWER_MET_MIN_PAIR_ITEMS, REVIEWER_MET_MIN_PAIR_ITEMS - 1)
    )
    assert (at_floor.result, under_floor.result) == ("met", "undecided")
    assert under_floor.lean_interval is not None and under_floor.lean_interval[1] < 0.10
    # Quiet and inside, but a kappa too few outputs decide: undecided.
    loose = reviewer_check(REVIEWER, GROUNDED.name, "pass_fail", kappa_pairs=_agreeing_pairs(15), leans=quiet)
    assert loose.result == "undecided"
    # A kappa shown below the bar outranks everything: the judge is not checked against this model.
    below = reviewer_check(
        REVIEWER, GROUNDED.name, "pass_fail", kappa_pairs=_agreeing_pairs(200, disagree=100), leans=quiet
    )
    assert (below.result, below.reason) == ("not_met", "instrument_below_bar")


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
    with pytest.raises(ValidationError, match="only the rig"):
        Check(kind="rig", instrument="launch-1", against="people", result="met")
    with pytest.raises(ValidationError, match="checked against people"):
        Check(kind="reviewer", instrument=REVIEWER, against="model", result="met")
    assert Check(instrument="launch-1", against="identical_arms", result="met").kind == "rig"


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
    assert (model.result, model.against_id, model.same_family_as, model.same_family_as_judge) == (
        "met",
        REVIEWER,
        ["v2"],
        False,
    )
    # The model and the person agree on every output, and neither leans: undecided at 30 pair items, under the floor.
    assert (own.result, own.pair_items, own.left_out, own.lean, own.reads) == ("undecided", 30, 0, 0.0, 60)
    # People decided: their check speaks for the judge.
    assert speaking_check(checks, people.instrument, GROUNDED.name) == people


def test_pair_items_from_a_run_with_an_unrecorded_coordinate_are_left_out_of_the_lean_and_counted() -> None:
    read, results = _both_read(30, 30, lean=True)
    closed = close_blind_read(read, results)
    whole = next(c for c in blind_read_checks(closed, results) if c.kind == "reviewer")
    assert (whole.pair_items, whole.left_out, whole.result, whole.reason) == (30, 0, "not_met", "instrument_leans")
    unrecorded = next(
        c for c in blind_read_checks(closed, results, unrecorded_runs={CANDIDATE_RUN}) if c.kind == "reviewer"
    )
    assert (unrecorded.pair_items, unrecorded.left_out, unrecorded.lean) == (0, 30, None)
    assert (unrecorded.result, unrecorded.reason) == ("insufficient", "not_recorded")
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
    (itself,) = [c for c in blind_read_checks(closed, results) if c.kind == "judge"]
    assert (itself.against, itself.against_id) == ("itself", REVIEWER)
    # Held to the bar a judge's agreement with its repeats is: 120 results, which 48 do not reach.
    assert (itself.result, itself.reason) == ("insufficient", "too_few")
    unnamed, unnamed_evidence = _campaign(24, served=None)
    blank_read = _read(unnamed, unnamed_evidence)
    answers = [_answer(blank_read, n, _judged(blank_read, unnamed, n), model=True) for n in range(24)]
    blank = close_blind_read(blank_read.model_copy(update={"answers": answers}), unnamed)
    (nothing,) = [c for c in blind_read_checks(blank, unnamed) if c.kind == "judge"]
    assert (nothing.against, nothing.result, nothing.reason, nothing.coordinate) == (
        "model",
        "insufficient",
        "identity_unrecorded",
        "served_model",
    )
    assert nothing.kappa is None, "an unrecorded model matches nothing, so no figure is read"


# --- the tier stays what it was ---------------------------------------------------------------------------------------


def test_a_model_check_moves_no_stored_tier_and_the_tier_is_its_people_and_itself_checks() -> None:
    read, results = _both_read(30, 0)
    closed = close_blind_read(read, results)
    judged = {key for result in results if (key := judge_key(result, GROUNDED.name)) is not None}
    people = [rating for rating in closed.ratings if rating.rater_kind == "person"]
    (tier,) = judge_evidence_tiers(judge_agreement(people, results), judge_self_agreement(results), judged)
    # A judge with no model check: the stored tier, byte for byte, as the tier code wrote it before checks existed.
    stored = json.dumps(tier.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(stored.encode()).hexdigest()[:16] == "e9c1a2f1d0b3c7a4", stored
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
