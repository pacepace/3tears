"""The first screen is computed by one function, from typed caveats, by rules each held here.

``first_screen`` (``threetears.evals.analysis.first_screen``) chooses what a report opens with beside its answer. Every
test below builds a caveat list by hand and reads the typed value back: none reads a word of a page.

- R1 to R11: one test per rule of the function (eligible, gate, the four ways one caveat silences another, rank, touch,
  id, cap, precision, call, count, stable).
- The ranking is data (``RANKS``), in the statistician's order: evidence not whole, identical arms separated, identity
  unrecorded, a late criterion, a judge not checked, a judge checked against a model only (a model of a candidate's
  family first), repeated looks, a judge checked against a model that is itself checked, few cases.
- The information architecture's three worked campaigns (section 3.3) give the slot contents it states.
- The function reads no eval kind, measure name or prose: its module imports only the closed sets and the input's
  shapes, and every string it compares is a member of a closed set.
"""

from __future__ import annotations

import ast
import random
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import first_screen as first_screen_module
from threetears.evals.analysis import reading_states
from threetears.evals.analysis.caveats import CLASS_OF_KIND, from_legacy, remedy
from threetears.evals.analysis.checks import Check
from threetears.evals.analysis.first_screen import RANKS, REASONS_SHOWN, first_screen, subsumed
from threetears.evals.analysis.report.call import Caveat
from threetears.evals.analysis.report_input import CallRead, CaveatEntry, FirstScreen, Gate, GateRun

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
def test_r3d_a_check_against_people_silences_the_check_against_a_model_on_the_same_criterion(
    people_first: bool,
) -> None:
    people = judge("check:kappa:people", against="people", criterion="kappa")
    model = judge("check:kappa:model", against="model", criterion="kappa")
    other = judge("check:lean:model", against="model", criterion="lean")
    order = [people, model, other] if people_first else [other, model, people]

    value = shown(order)

    assert value.reasons == [people.id, other.id]
    assert {entry.id: entry.subsumed_by for entry in subsumed(order)}[model.id] == people.id


def one_of_each_class() -> list[CaveatEntry]:
    return [
        withheld(READING, "unfinished"),
        caveat("late:margin"),
        judge("judge_uncalibrated"),
        caveat("looks"),
        caveat("unsettled:support"),
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


def test_r8_missing_precision_off_the_deciding_measure_is_scoped_so_it_is_never_a_reason() -> None:
    """The raiser's rule, asserted on what a raiser hands in: class E is report-wide only where the answer turns on
    the measure."""
    cost = caveat(
        "not_available:support/cost",
        "unsettled",
        state="not_available",
        reason="too_few",
        readings=["support/cost"],
        deciding=False,
        reach="scoped",
    )
    deciding = cost.model_copy(update={"id": "not_available:support/pass_rate", "deciding": True, "reach": "report"})

    assert cost.cls == "E"
    assert shown([cost]).reasons == [] and shown([cost]).open_report_caveats == 0
    assert shown([cost, deciding]).reasons == [deciding.id]


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


def the_statistician_s_order() -> list[CaveatEntry]:
    """One caveat per row of the order, first to last, each on a reading of its own."""
    return [
        withheld("a/unfinished", "unfinished"),
        withheld("a/split", "split", "served_model"),
        set_aside("a/aa", "identical_arms_separated"),
        withheld("a/identity", "identity_unrecorded", "measure_definition"),
        caveat("late:looks"),
        set_aside("a/people", "people_disagree"),
        set_aside("a/leans", "instrument_leans"),
        set_aside("a/below", "instrument_below_bar"),
        set_aside("a/unchecked", "instrument_unchecked"),
        judge("judge:no_check"),
        judge("judge:model_same_family", against="model", same_family=True, against_checked=True),
        judge("judge:model_unchecked", against="model"),
        caveat("looks"),
        judge("judge:model_checked", against="model", against_checked=True),
        caveat("unsettled:few_cases"),
    ]


def test_the_ranking_is_the_statistician_s_order() -> None:
    expected = [entry.id for entry in the_statistician_s_order()]

    for order in shuffles(the_statistician_s_order(), count=20):
        assert ranked(order) == expected


def test_the_ranking_names_its_rows_in_the_eight_rank_order() -> None:
    names = list(dict.fromkeys(row.name for row in RANKS if row.name != "answer_changed"))
    assert names == [
        "evidence_not_whole",
        "identical_arms_separated",
        "identity_unrecorded",
        "late_criterion",
        "judge_unchecked",
        "judge_checked_against_a_model_only",
        "repeated_looks",
        "judge_checked_against_a_checked_model",
        "few_cases",
    ]


def test_looks_stay_below_both_judge_rows_and_above_a_model_that_is_itself_checked() -> None:
    looks = caveat("looks")
    unchecked = judge("judge:none")
    model_only = judge("judge:model", against="model")
    checked_model = judge("judge:model_checked", against="model", against_checked=True)

    assert ranked([looks, checked_model, model_only, unchecked]) == [
        unchecked.id,
        model_only.id,
        looks.id,
        checked_model.id,
    ]


def test_every_reason_of_a_non_counted_state_is_placed_by_a_row_of_its_own() -> None:
    """No set-aside or withheld caveat falls through to the row for an untyped change to the answer."""
    for state in ("set_aside", "withheld"):
        for reason in sorted(reading_states.REASONS_OF[state]):
            entry = caveat(
                f"{state}:r",
                state=state,
                reason=reason,
                coordinate="served_model" if reason == "split" else None,
                readings=["a/r"],
            )
            placed = next(row for row in RANKS if row.places(entry))
            assert placed.name != "answer_changed", reason


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
        caveat("unsettled:replies", arms=["candidate"]),
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


def test_caveats_of_the_earlier_shape_become_entries_with_the_same_ids_and_their_kind_s_class() -> None:
    legacy = [
        Caveat("late:margin", "late", "report"),
        Caveat("judge_uncalibrated", "judge_uncalibrated", "report"),
        Caveat("confound:tools", "confound", "scoped"),
        Caveat("latency", "latency", "method"),
        Caveat("analyst:flag", "analyst", "report"),
    ]

    entries = from_legacy(legacy)

    assert [(entry.id, entry.kind, entry.reach) for entry in entries] == [(c.id, c.kind, c.reach) for c in legacy]
    assert [entry.cls for entry in entries] == ["B", "C", "B", None, None]
    assert all(entry.state is None and entry.reason is None for entry in entries), "absent reads counted"
    assert shown(entries).reasons == ["late:margin", "judge_uncalibrated", "analyst:flag"]


# --- the function reads no kind, no name and no prose ---------------------------------------------------------------


def _closed_words() -> set[str]:
    def strings(alias: Any) -> set[str]:
        return {
            word for member in get_args(alias) for word in ({member} if isinstance(member, str) else strings(member))
        }

    words = set().union(*(strings(getattr(reading_states, name)) for name in reading_states.__all__))
    return words | strings(Check.model_fields["against"].annotation)


def test_the_function_s_module_imports_only_the_closed_sets_and_the_input_s_shapes() -> None:
    tree = ast.parse(Path(first_screen_module.__file__).read_text(encoding="utf-8"))
    engine = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None and node.module.startswith("threetears")
    }
    assert engine == {"threetears.evals.analysis.reading_states", "threetears.evals.analysis.report_input"}
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
