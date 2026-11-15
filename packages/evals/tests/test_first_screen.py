"""The first screen is computed by one function, from typed caveats, by rules each held here.

``first_screen`` (``threetears.evals.analysis.first_screen``) chooses what a report opens with beside its answer. Every
test below builds a caveat list by hand and reads the typed value back: none reads a word of a page.

- R1 to R11: one test per rule of the function (eligible, gate, the four ways one caveat silences another, rank, touch,
  id, cap, precision, call, count, stable). R8 is the constraint document's F8: it fails if the function stops
  holding a class E caveat off the first screen where the answer does not turn on its reading.
- The ranking is data (``RANKS``), the statistician's eight ranks exactly: evidence not whole (every withheld
  reading), identical arms separated, identity unrecorded, a late criterion, a judge not checked, a judge checked
  against a model only, repeated looks, few cases. Each adjacent pair is tested, and each tie rule inside a rank.
- The information architecture's three worked campaigns (section 3.3) give the slot contents it states.
- The function reads no eval kind, measure name or prose: its module imports only the closed sets and the input's
  shapes, and every string it compares is a member of a closed set.
"""

from __future__ import annotations

import ast
import json
import random
from itertools import pairwise
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import first_screen as first_screen_module
from threetears.evals.schema import reading_states
from threetears.evals.analysis.caveats import CLASS_OF_KIND, caveat_list, from_legacy, remedy, screen
from threetears.evals.analysis.checks import JudgeCheck, ReviewerCheck, RigCheck
from threetears.evals.analysis.first_screen import RANKS, REASONS_SHOWN, first_screen, subsumed
from threetears.evals.analysis.report_input import CallRead, CaveatEntry, FirstScreen, Gate, GateRun, ReportInput

FIXTURES = Path(__file__).resolve().parent / "fixtures"

READING = "support/helpfulness"
OTHER = "support/accuracy"


def caveat(id: str, kind: str | None = None, **facts: Any) -> CaveatEntry:
    """A report-wide caveat on an asked reading, its class its kind's unless given."""
    kind = kind or id.split(":")[0]
    facts.setdefault("cls", CLASS_OF_KIND.get(kind))
    facts.setdefault("reach", "report")
    facts.setdefault("asked", True)
    return CaveatEntry(id=id, kind=kind, **facts)


def set_aside(reading: str, reason: str, **facts: Any) -> CaveatEntry:
    return caveat(f"set_aside:{reading}", state="set_aside", reason=reason, readings=[reading], **facts)


def withheld(reading: str, reason: str, coordinate: str | None = None, **facts: Any) -> CaveatEntry:
    return caveat(
        f"withheld:{reading}", state="withheld", reason=reason, coordinate=coordinate, readings=[reading], **facts
    )


def judge(id: str, against: str | None = None, **facts: Any) -> CaveatEntry:
    return caveat(id, "judge_uncalibrated", against=against, **facts)


def shown(caveats: list[CaveatEntry], **inputs: Any) -> FirstScreen:
    inputs.setdefault("gate", None)
    inputs.setdefault("called_arms", ())
    return first_screen(caveats, **inputs)


def shuffles(caveats: list[CaveatEntry], count: int = 50) -> list[list[CaveatEntry]]:
    """``count`` seeded orderings of one list."""
    orders = []
    for seed in range(count):
        order = list(caveats)
        random.Random(seed).shuffle(order)
        orders.append(order)
    return orders


# --- the rules ----------------------------------------------------------------------------------------------------


def test_r1_only_an_open_report_wide_caveat_nothing_silences_is_eligible() -> None:
    open_one = caveat("late:margin")
    excluded = [
        caveat("confound:tools", reach="scoped"),
        caveat("latency", reach="method"),
        caveat("late:floor", immaterial=True),
        caveat("late:bar", subsumed_by="late:margin"),
    ]

    value = shown([open_one, *excluded])

    assert value.reasons == ["late:margin"]
    assert value.remainder == 0 and value.open_report_caveats == 1
    assert shown([caveat("late:floor", immaterial=False), caveat("late:bar", immaterial=None)]).open_report_caveats == 2


def test_r2_a_gate_silences_the_reasons_and_counts_what_waits_below_it() -> None:
    gate = Gate(rule="unfinished", runs=[GateRun(run="run-1", status="failed", ended_because="the provider refused")])
    three = [caveat("late:margin"), judge("judge_uncalibrated"), caveat("looks")]

    value = shown(three, gate=gate)

    assert value.reasons == [] and value.remainder == 0
    assert value.gate is not None and value.gate.waiting == 3
    assert value.gate.rule == "unfinished" and value.gate.runs == gate.runs


def test_r3a_a_withheld_reading_silences_a_caveat_scoped_only_to_it() -> None:
    split = withheld(READING, "split", "served_model")
    only_here = judge("judge_uncalibrated:helpfulness", readings=[READING])
    two_readings = judge("judge_uncalibrated:both", readings=[READING, OTHER])

    marked = {entry.id: entry.subsumed_by for entry in subsumed([only_here, two_readings, split])}

    assert marked == {only_here.id: split.id, two_readings.id: None, split.id: None}
    assert shown([only_here, two_readings, split]).reasons == [split.id, two_readings.id]


def test_r3a_a_reading_withheld_on_one_arm_silences_only_what_is_scoped_to_that_arm() -> None:
    """An arm read per answering model withholds that arm's reading: a caveat on the same reading of that arm is
    silenced, and one on the reading of every arm (repeated looks at the aspect's test) stands."""
    split = withheld(READING, "split", "served_model", arms=["candidate-b"])
    same_arm = judge("judge_uncalibrated:b", readings=[READING], arms=["candidate-b"])
    every_arm = caveat("looks", readings=[READING])

    marked = {entry.id: entry.subsumed_by for entry in subsumed([split, same_arm, every_arm])}

    assert marked == {split.id: None, same_arm.id: split.id, every_arm.id: None}


def test_r3b_a_reading_set_aside_for_a_cause_is_one_reason_and_the_cause_is_not_listed_again() -> None:
    aside = set_aside(READING, "instrument_unchecked")
    cause = judge("judge_unchecked:helpfulness", reason="instrument_unchecked", readings=[READING])
    another_cause = judge("judge_leans:helpfulness", reason="instrument_leans", readings=[READING])
    elsewhere = judge("judge_unchecked:accuracy", reason="instrument_unchecked", readings=[OTHER])

    value = shown([cause, aside, another_cause, elsewhere])

    assert {entry.id: entry.subsumed_by for entry in subsumed([cause, aside])} == {cause.id: aside.id, aside.id: None}
    assert cause.id not in value.reasons
    assert set(value.reasons) == {aside.id, another_cause.id, elsewhere.id} and value.open_report_caveats == 3


@pytest.mark.parametrize(
    "reason", ["instrument_unchecked", "instrument_below_bar", "instrument_leans", "people_disagree"]
)
def test_r3c_identical_arms_that_separated_silence_every_instrument_caveat_on_the_reading(reason: str) -> None:
    separated = set_aside(READING, "identical_arms_separated")
    instrument = judge(f"judge:{reason}", reason=reason, readings=[READING])
    elsewhere = judge("judge:elsewhere", reason=reason, readings=[OTHER])

    value = shown([instrument, elsewhere, separated])

    assert value.reasons == [separated.id, elsewhere.id]
    assert subsumed([instrument, separated])[0].subsumed_by == separated.id


@pytest.mark.parametrize("people_first", [True, False])
def test_r3d_a_deciding_check_against_people_silences_the_check_against_a_model_on_the_same_criterion(
    people_first: bool,
) -> None:
    people = judge("check:kappa:people", against="people", criterion="kappa", check="not_met")
    model = judge("check:kappa:model", against="model", criterion="kappa", check="met")
    other = judge("check:lean:model", against="model", criterion="lean", check="met")
    order = [people, model, other] if people_first else [other, model, people]

    value = shown(order)

    assert value.reasons == [people.id, other.id]
    assert {entry.id: entry.subsumed_by for entry in subsumed(order)}[model.id] == people.id


@pytest.mark.parametrize("result", ["undecided", "insufficient", None])
def test_r3d_a_check_against_people_that_does_not_decide_silences_nothing(result: str | None) -> None:
    """A people check that is undecided or insufficient does not speak for the judge, which then reads as checked
    against a model."""
    people = judge("check:kappa:people", against="people", criterion="kappa", check=result)
    model = judge("check:kappa:model", against="model", criterion="kappa", check="met")

    assert {entry.id: entry.subsumed_by for entry in subsumed([people, model])}[model.id] is None
    assert model.id in shown([people, model]).reasons


def test_r3d_a_met_check_against_people_is_said_one_step_down_and_still_silences_the_model() -> None:
    people = judge("check:kappa:people", against="people", criterion="kappa", check="met", reach="method")
    model = judge("check:kappa:model", against="model", criterion="kappa", check="met")

    value = shown([people, model])

    assert value.reasons == [] and value.open_report_caveats == 0


def one_of_each_class() -> list[CaveatEntry]:
    return [
        withheld(READING, "unfinished"),
        caveat("late:margin"),
        judge("judge_uncalibrated"),
        caveat("looks"),
        caveat("unsettled:support", deciding=True),
        caveat("analyst:hunch"),
    ]


def ranked(caveats: list[CaveatEntry], **inputs: Any) -> list[str]:
    """Every eligible caveat's id in rank order, read through the cap three at a time."""
    order: list[str] = []
    rest = list(caveats)
    while rest:
        head = shown(rest, **inputs).reasons
        if not head:
            break
        order += head
        rest = [entry for entry in rest if entry.id not in head]
    return order


def test_r4_reasons_come_back_in_class_order_and_a_caveat_with_no_class_last() -> None:
    expected = [
        f"withheld:{READING}",
        "late:margin",
        "judge_uncalibrated",
        "looks",
        "unsettled:support",
        "analyst:hunch",
    ]
    assert [entry.cls for entry in one_of_each_class()] == ["A", "B", "C", "D", "E", None]

    for order in shuffles(one_of_each_class()):
        assert ranked(order) == expected


def test_r5_within_a_rank_a_caveat_on_an_asked_reading_of_a_called_arm_comes_first() -> None:
    on_called = caveat("confound:zeta", arms=["candidate"])
    on_other = caveat("confound:alpha", arms=["runner-up"])
    not_asked = caveat("confound:beta", arms=["candidate"], asked=False)
    every_arm = caveat("confound:omega")

    assert ranked([on_other, not_asked, on_called, every_arm], called_arms=["candidate"]) == [
        "confound:omega",
        "confound:zeta",
        "confound:alpha",
        "confound:beta",
    ]
    assert shown([on_other, on_called], called_arms=["candidate"]).called_arms == ["candidate"]


def test_r6_caveats_equal_on_rank_and_touch_come_in_id_order() -> None:
    assert shown([caveat("late:margin"), caveat("late:bar:latency")]).reasons == ["late:bar:latency", "late:margin"]


@pytest.mark.parametrize("count", [0, 3, 4, 9])
def test_r7_three_reasons_are_shown_and_the_rest_are_counted_never_dropped(count: int) -> None:
    caveats = [caveat(f"late:criterion-{index}") for index in range(count)]

    value = shown(caveats)

    assert REASONS_SHOWN == 3
    assert value.reasons == [entry.id for entry in caveats[:3]]
    assert value.remainder == max(count - 3, 0)
    assert len(value.reasons) + value.remainder == value.open_report_caveats == count
    with pytest.raises(ValidationError):
        FirstScreen(reasons=["a", "b", "c", "d"])


def precision(reading: str, *, deciding: bool, reach: str = "report") -> CaveatEntry:
    """A class E caveat: no interval can be computed on ``reading``."""
    return caveat(
        f"not_available:{reading}",
        "unsettled",
        state="not_available",
        reason="too_few",
        readings=[reading],
        deciding=deciding,
        reach=reach,
    )


def test_r8_missing_precision_is_a_reason_only_on_the_measure_the_answer_turns_on() -> None:
    """A cost interval that cannot be computed is a first-screen reason where the answer turns on cost, and is stated
    in its table cell anywhere else, even if whoever raised it gave it the whole report's reach."""
    off_the_answer = precision("support/cost", deciding=False)
    on_the_answer = precision("support/pass_rate", deciding=True)
    assert off_the_answer.cls == on_the_answer.cls == "E" and off_the_answer.reach == "report"

    alone = shown([off_the_answer])
    assert alone.reasons == [] and alone.remainder == 0 and alone.open_report_caveats == 0
    both = shown([off_the_answer, on_the_answer])
    assert both.reasons == [on_the_answer.id] and both.open_report_caveats == 1
    assert shown([off_the_answer], gate=Gate(rule="too_small")).open_report_caveats == 0


def test_r8_holds_for_class_e_alone() -> None:
    """A caveat of any other class is a reason whether or not the answer turns on a reading it names."""
    late = caveat("late:margin", deciding=False)
    looks = caveat("looks", deciding=False)
    assert shown([late, looks]).reasons == [late.id, looks.id]
    assert shown([precision("support/cost", deciding=True, reach="scoped")]).reasons == [], "scoped is still scoped"


@pytest.mark.parametrize("made_by", ["person", "model", None])
def test_r9_the_call_is_copied_through_and_is_never_a_reason(made_by: str | None) -> None:
    call = CallRead(arm="candidate", made_by=made_by, model="reviewer-1" if made_by == "model" else None)
    nine = [caveat(f"late:criterion-{index}") for index in range(9)]

    value = shown(nine, call=call)

    assert value.call == call and value.call is not None and value.call.made_by == made_by
    assert len(value.reasons) == 3 and value.remainder == 6
    assert shown(nine).call is None
    with pytest.raises(ValidationError):
        CallRead(arm="candidate", made_by="committee")


def test_r10_the_count_of_open_report_wide_caveats_is_the_same_under_a_gate() -> None:
    caveats = [*one_of_each_class(), caveat("late:floor", immaterial=True), caveat("latency", reach="method")]

    assert shown(caveats).open_report_caveats == 6
    assert shown(caveats, gate=Gate(rule="too_small")).open_report_caveats == 6
    assert shown([]).open_report_caveats == 0


def test_r11_the_value_is_the_same_whatever_order_the_caveats_come_in() -> None:
    split = withheld(READING, "split", "served_model")
    caveats = [
        *one_of_each_class()[1:],
        split,
        judge("judge:here", readings=[READING]),
        judge("check:kappa:people", against="people", criterion="kappa"),
        judge("check:kappa:model", against="model", criterion="kappa"),
        caveat("late:floor", immaterial=True),
    ]
    first = shown(caveats, called_arms=["b", "a"])

    for order in shuffles(caveats):
        assert shown(order, called_arms=["a", "b"]) == first
        assert {e.id: e.subsumed_by for e in subsumed(order)} == {e.id: e.subsumed_by for e in subsumed(caveats)}


# --- the ranking --------------------------------------------------------------------------------------------------


#: The eight ranks, each as one caveat that takes it, on a reading of its own.
EIGHT_RANKS: dict[str, Any] = {
    "evidence_not_whole": lambda: withheld("a/unfinished", "unfinished"),
    "identical_arms_separated": lambda: set_aside("a/aa", "identical_arms_separated"),
    "identity_unrecorded": lambda: caveat(
        "not_available:a/corrected",
        "not_available",
        cls="B",
        state="not_available",
        reason="identity_unrecorded",
        coordinate="served_model",
        readings=["a/corrected"],
    ),
    "late_criterion": lambda: caveat("late:looks"),
    "judge_unchecked": lambda: judge("judge:no_check"),
    "judge_checked_against_a_model_only": lambda: judge("judge:model", against="model"),
    "repeated_looks": lambda: caveat("looks"),
    "few_cases": lambda: precision("a/few", deciding=True),
}


def test_the_ranking_holds_exactly_the_statistician_s_eight_ranks_in_order() -> None:
    names = list(dict.fromkeys(row.name for row in RANKS))
    assert names == list(EIGHT_RANKS)
    assert [next(row.rank for row in RANKS if row.name == name) for name in names] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert all(row.rank == next(r.rank for r in RANKS if r.name == row.name) for row in RANKS), "one name, one rank"
    assert [row.rank for row in RANKS] == sorted(row.rank for row in RANKS)


@pytest.mark.parametrize(("higher", "lower"), list(pairwise(EIGHT_RANKS)))
def test_each_rank_comes_before_the_next(higher: str, lower: str) -> None:
    first, second = EIGHT_RANKS[higher](), EIGHT_RANKS[lower]()
    # The lower rank is given every tie-break in its favour: it is on a called arm, and its id sorts first.
    second = second.model_copy(update={"id": f"a:{second.id}", "arms": ["called"]})
    first = first.model_copy(update={"id": f"z:{first.id}", "arms": ["other"]})

    assert next(row.name for row in RANKS if row.places(first)) == higher
    assert next(row.name for row in RANKS if row.places(second)) == lower
    assert shown([second, first], called_arms=["called"]).reasons == [first.id, second.id]


def test_all_eight_ranks_together_come_back_in_order() -> None:
    caveats = [make() for make in EIGHT_RANKS.values()]
    for order in shuffles(caveats, count=20):
        assert ranked(order) == [entry.id for entry in caveats]


@pytest.mark.parametrize(
    ("reason", "coordinate"),
    [
        ("unfinished", None),
        ("split", "served_model"),
        ("identity_unrecorded", "measure_definition"),
        ("measured_under_concurrency", None),
        ("trials_lost", None),
    ],
)
def test_every_withheld_reading_is_rank_one_whatever_withheld_it(reason: str, coordinate: str | None) -> None:
    entry = withheld("a/r", reason, coordinate)
    separated = set_aside("a/aa", "identical_arms_separated")

    assert next(row for row in RANKS if row.places(entry)).rank == 1
    assert shown([separated, entry]).reasons == [entry.id, separated.id]


def test_a_figure_the_rig_lost_the_trials_of_is_rank_one_too() -> None:
    """Evidence that is not whole is rank 1 whether it withheld a reading or left a figure with nothing to compute."""
    lost = caveat(
        "not_available:a/on_time",
        "unsettled",
        state="not_available",
        reason="trials_lost",
        readings=["a/on_time"],
        deciding=True,
    )
    separated = set_aside("a/aa", "identical_arms_separated")

    assert next(row for row in RANKS if row.places(lost)).name == "evidence_not_whole"
    assert shown([separated, lost]).reasons == [lost.id, separated.id]
    assert remedy(lost) == "repair_rig"


def test_within_rank_one_a_caveat_on_a_called_arm_comes_first() -> None:
    """Rank 1 is one rank: what withheld a reading does not order it, the arm it touches does."""
    unfinished_elsewhere = withheld("a/unfinished", "unfinished", arms=["other"])
    split_on_called = withheld("a/split", "split", "served_model", arms=["called"])
    unrecorded_on_called = withheld("a/identity", "identity_unrecorded", "rig", arms=["called"])

    assert ranked([unfinished_elsewhere, split_on_called, unrecorded_on_called], called_arms=["called"]) == [
        unrecorded_on_called.id,
        split_on_called.id,
        unfinished_elsewhere.id,
    ]


def test_within_the_judge_unchecked_rank_the_reasons_come_in_the_order_one_is_stated() -> None:
    """Rank 5's own order: people disagree, the judge leans, below its bar, unchecked, then a counted reading whose
    judge has no check. It outranks the arm a caveat touches."""
    in_order = [
        set_aside("a/people", "people_disagree", arms=["other"]),
        set_aside("a/leans", "instrument_leans", arms=["other"]),
        set_aside("a/below", "instrument_below_bar", arms=["other"]),
        set_aside("a/unchecked", "instrument_unchecked", arms=["other"]),
        judge("judge:no_check", arms=["called"]),
    ]
    assert {next(row for row in RANKS if row.places(entry)).rank for entry in in_order} == {5}

    for order in shuffles(in_order, count=10):
        assert ranked(order, called_arms=["called"]) == [entry.id for entry in in_order]


def test_within_the_model_only_rank_a_model_of_a_candidate_s_family_comes_first() -> None:
    same_family = judge("judge:z_family", against="model", same_family=True, arms=["other"])
    another_family = judge("judge:a_other", against="model", arms=["called"])
    looks = caveat("looks")
    unchecked = judge("judge:none")

    assert {next(row for row in RANKS if row.places(entry)).rank for entry in (same_family, another_family)} == {6}
    assert ranked([looks, another_family, same_family, unchecked], called_arms=["called"]) == [
        unchecked.id,
        same_family.id,
        another_family.id,
        looks.id,
    ], "looks stay below both judge ranks"


def test_every_reason_of_a_non_counted_state_is_placed_by_a_rank() -> None:
    kinds = {"set_aside": "set_aside", "withheld": "withheld", "not_available": "unsettled"}
    for state, kind in kinds.items():
        for reason in sorted(reading_states.REASONS_OF[state]):
            entry = caveat(
                f"{state}:r",
                kind,
                state=state,
                reason=reason,
                coordinate="served_model" if reason in ("split", "identity_unrecorded") else None,
                readings=["a/r"],
            )
            assert any(row.places(entry) for row in RANKS), (state, reason)


def test_a_class_a_caveat_carries_the_state_that_changed_the_answer() -> None:
    with pytest.raises(ValidationError, match="set aside or withheld"):
        caveat("set_aside:r", cls="A")
    with pytest.raises(ValidationError, match="set aside or withheld"):
        caveat("x", "late", cls="A", state="not_available", reason="too_few")


# --- the three worked campaigns -----------------------------------------------------------------------------------


def test_a_clean_campaign_shows_no_gate_no_reason_and_the_person_who_called_it() -> None:
    call = CallRead(arm="candidate", made_by="person", runs_joined_since=0, identity_changed=False)

    value = shown([], called_arms=["candidate"], call=call)

    assert value == FirstScreen(called_arms=["candidate"], call=call)
    assert value.gate is None and value.reasons == [] and value.remainder == 0 and value.open_report_caveats == 0


def test_a_stale_host_measure_called_by_a_model_opens_on_the_withheld_readings_and_says_who_decided() -> None:
    """A host measure whose definition changed and was not recorded; one arm decided; a model made the call. Seven
    caveats stand on the page today, two of them proved immaterial."""
    host_measure = withheld("replies/on_time_rate", "identity_unrecorded", "measure_definition", arms=["candidate"])
    caveats = [
        caveat("late:margin", immaterial=True),
        caveat("late:floor:replies/on_time_rate", immaterial=True),
        judge("judge_uncalibrated"),
        caveat("confound:tool_bound", arms=["candidate"]),
        caveat("short_run:run-7", arms=["runner-up"]),
        caveat("unsettled:replies", arms=["candidate"], deciding=True),
        host_measure,
    ]
    call = CallRead(arm="candidate", made_by="model", model="analyst-1", identity_changed=True)

    value = shown(caveats, called_arms=["candidate"], call=call)

    assert value.gate is None, "every run finished"
    assert value.reasons[0] == host_measure.id and host_measure.cls == "A"
    assert value.reasons == [host_measure.id, "confound:tool_bound", "short_run:run-7"]
    assert value.remainder == 2 and value.open_report_caveats == 5, "the two proved immaterial are not counted"
    assert remedy(host_measure) == "remeasure_or_attest"
    assert value.call is not None and value.call.made_by == "model" and value.call.identity_changed is True


def messy() -> list[CaveatEntry]:
    """An arm split by answering model, a judge checked against a model only, and three looks, none planned."""
    return [
        caveat("looks"),
        judge("judge_uncalibrated", against="model", criterion="kappa", reason="instrument_unchecked"),
        withheld(READING, "split", "served_model", arms=["candidate-b"]),
    ]


def test_an_unfinished_run_gates_the_messy_campaign_and_counts_three_waiting() -> None:
    gate = Gate(rule="unfinished", runs=[GateRun(run="run-3", status="cancelled", ended_because="stopped by hand")])

    value = shown(messy(), gate=gate, called_arms=["candidate-a"])

    assert value.gate is not None and value.gate.waiting == 3 and value.gate.runs[0].run == "run-3"
    assert value.reasons == [] and value.remainder == 0, "an answer that does not exist has no reasons"
    assert remedy(withheld("a/r", "unfinished")) == "resume_or_rerun"


def test_the_messy_campaign_resumed_shows_exactly_three_reasons_in_class_order() -> None:
    value = shown(messy(), called_arms=["candidate-a"])
    by_id = {entry.id: entry for entry in messy()}

    assert value.gate is None
    assert value.reasons == [f"withheld:{READING}", "judge_uncalibrated", "looks"]
    assert [by_id[reason].cls for reason in value.reasons] == ["A", "C", "D"]
    assert value.remainder == 0 and value.open_report_caveats == 3
    assert [remedy(by_id[reason]) for reason in value.reasons] == [
        "rerun_on_one_cell",
        "check_against_people",
        "declare_looks",
    ]


# --- a caveat is typed --------------------------------------------------------------------------------------------


def test_a_caveat_s_state_reason_and_coordinate_belong_together() -> None:
    with pytest.raises(ValidationError):
        caveat("withheld:r", state="withheld", reason="too_few")
    with pytest.raises(ValidationError):
        caveat("withheld:r", state="withheld", reason="split")
    entry = judge("check:lean:model", against="model", reason="instrument_leans")
    assert entry.state is None and entry.reason == "instrument_leans", "a check that is not met carries its reason"
    assert entry.model_dump(mode="json", by_alias=True)["class"] == "C"
    assert CaveatEntry.model_validate(entry.model_dump(mode="json", by_alias=True)) == entry


def test_a_stored_input_s_caveats_become_entries_with_the_same_ids_and_their_kind_s_class() -> None:
    """A stored input carries no list; the legacy raisers give it one on read, each entry its kind's class, no state
    (absent reads counted), and a class E entry deciding exactly where it reaches the whole report."""
    stored = ReportInput.model_validate(
        json.loads((FIXTURES / "report_inputs" / "r5" / "input.json").read_text(encoding="utf-8"))
    )

    entries = from_legacy(stored)

    assert stored.caveat_list is None and entries
    assert [entry.cls for entry in entries] == [CLASS_OF_KIND.get(entry.kind) for entry in entries]
    assert {"late", "short_run"} <= {entry.kind for entry in entries}
    assert all(entry.state is None or entry.kind == "latency" for entry in entries), "absent reads counted"
    assert all(entry.deciding == (entry.cls == "E" and entry.reach == "report") for entry in entries)
    assert screen(stored, caveat_list(stored)).reasons == shown(caveat_list(stored), called_arms=["September"]).reasons


# --- the function reads no kind, no name and no prose ---------------------------------------------------------------


def _closed_words() -> set[str]:
    def strings(alias: Any) -> set[str]:
        return {
            word for member in get_args(alias) for word in ({member} if isinstance(member, str) else strings(member))
        }

    words = set().union(*(strings(getattr(reading_states, name)) for name in reading_states.__all__))
    # What a check of each kind is checked against, and its result.
    kinds = (JudgeCheck, ReviewerCheck, RigCheck)
    return words.union(
        *(strings(kind.model_fields[name].annotation) for kind in kinds for name in ("against", "result"))
    )


def test_the_function_s_module_imports_only_the_closed_sets_and_the_input_s_shapes() -> None:
    tree = ast.parse(Path(first_screen_module.__file__).read_text(encoding="utf-8"))
    engine = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None and node.module.startswith("threetears")
    }
    assert engine == {"threetears.evals.schema.reading_states", "threetears.evals.analysis.report_input"}
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Import)]


def test_every_string_the_function_s_module_compares_is_a_closed_set_s_word() -> None:
    tree = ast.parse(Path(first_screen_module.__file__).read_text(encoding="utf-8"))
    compared = {
        operand.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        for operand in (node.left, *node.comparators)
        if isinstance(operand, ast.Constant) and isinstance(operand.value, str)
    }
    ranked_on = {
        value for row in RANKS for value in (row.cls, row.state, row.reason, row.against) if isinstance(value, str)
    }
    closed = _closed_words()

    assert {"withheld", "set_aside", "report"} <= compared, "the walk found the comparisons"
    assert compared <= closed, sorted(compared - closed)
    assert ranked_on <= closed, sorted(ranked_on - closed)
