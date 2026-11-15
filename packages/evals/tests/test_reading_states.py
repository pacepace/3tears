"""The shared reading states: one closed set, one set of reasons, and every earlier word mapped onto them.

Contracts held here (``threetears.evals.schema.reading_states``, ``state_words`` and ``caveats``):

- every state and reason pair the information architecture lists validates, with the one pair the judges design
  adds (a figure not available because an identity is unrecorded), and every cross pair is refused;
- ``counted`` carries no reason; a split and an unrecorded identity each name their coordinate, and no other reason
  may; an arm's identity holds a state for every coordinate;
- every state the designs in flight need is expressible as a state, a reason and a coordinate, on the shape that
  will hold it;
- no other ``Literal`` in the package holds two or more of the four state words, so a second vocabulary for "is this
  counted" cannot appear unseen;
- every closed vocabulary of the package that could be a word about a reading, found by structure
  (``reading_state_vocabularies.py``), is mapped word for word in the one table or listed as no such word, with why;
  every row names a word that still exists, and each row is a valid state and reason, or says what the word is
  instead;
- the one stored word that collides with a reading state is mapped as what it is, until it is renamed;
- every reason has a remedy, every remedy is used, and a caveat kind's class is one of the five or none.
"""

from __future__ import annotations

import ast
from itertools import product
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import ValidationError

import threetears.evals
from packages.evals.tests.reading_state_vocabularies import NOT_ABOUT_A_READING, aliases, discovered
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, FamilyComparison, RunSummary
from threetears.evals.analysis.campaign_readings import AnswerSplit, CampaignReadings
from threetears.evals.analysis.caveats import CLASS_OF_KIND, REMEDY_OF_REASON
from threetears.evals.analysis.report_input import Remedy
from threetears.evals.analysis.checks import Check, Corrected, Rescore
from threetears.evals.schema.reading_states import (
    REASONS_OF,
    ArmIdentity,
    CaveatClass,
    Coordinate,
    IdentityState,
    NotAvailableReason,
    ReadingState,
    Reason,
    SetAsideReason,
    StateOf,
    WithheldReason,
    check_state,
)
from threetears.evals.analysis.report_input import Contrast, Stat
from threetears.evals.analysis.run_ending import StoppedShort, stopped_short
from threetears.evals.analysis.state_words import JUDGING_NOT_ASKED, STATE_WORDS, StateWord, state_word
from threetears.evals.kernel.evidence_tiers import CriterionState
from threetears.evals.kernel.result_condition import JudgingState, ResultCondition

SOURCE = Path(threetears.evals.__file__).resolve().parent

#: The information architecture's table (section 2.3), written out: the test is that the code holds exactly this.
ARCHITECTURE: dict[str, set[str]] = {
    "counted": set(),
    "set_aside": {
        "instrument_unchecked",
        "instrument_below_bar",
        "instrument_leans",
        "people_disagree",
        "identical_arms_separated",
    },
    # ``trials_lost`` (ruled 2026-11-15): the rig lost the trials. No other reason is true of it: the run finished,
    # the cases were not too few, and the figure was recorded as lost.
    "withheld": {"unfinished", "split", "identity_unrecorded", "measured_under_concurrency", "trials_lost"},
    "not_available": {
        "too_few",
        "no_variation",
        "not_recorded",
        "replayed",
        "not_drawn_at_random",
        "skipped_past_limit",
        # The judges design (its sections 0.3 and 9.3) and the statistician's ruling S1: a check's result carried onto
        # a contrast cannot be computed across runs with an unrecorded coordinate.
        "identity_unrecorded",
        "trials_lost",
    },
}

#: The reason about a coordinate that two states share.
SHARED_REASON = "identity_unrecorded"

#: The two reasons two states share, and the only two.
SHARED_REASONS = ["identity_unrecorded", "trials_lost"]


def _coordinate(reason: str) -> str | None:
    return "served_model" if reason in ("split", "identity_unrecorded") else None


# --- the closed sets ----------------------------------------------------------------------------------------------


def test_the_states_and_their_reasons_are_the_architecture_s() -> None:
    assert set(get_args(ReadingState)) == set(ARCHITECTURE)
    assert {state: set(reasons) for state, reasons in REASONS_OF.items()} == ARCHITECTURE
    assert set(get_args(SetAsideReason)) == ARCHITECTURE["set_aside"]
    assert set(get_args(WithheldReason)) == ARCHITECTURE["withheld"]
    assert set(get_args(NotAvailableReason)) == ARCHITECTURE["not_available"]
    assert set(get_args(IdentityState)) == {"one", "split", "unrecorded"}


def test_two_reasons_belong_to_two_states_and_no_other_does() -> None:
    every = [reason for reasons in ARCHITECTURE.values() for reason in reasons]
    assert sorted(reason for reason in set(every) if every.count(reason) > 1) == SHARED_REASONS
    assert len(set(every)) == 16 and len(every) == 18
    assert all(
        reason in ARCHITECTURE["withheld"] and reason in ARCHITECTURE["not_available"] for reason in SHARED_REASONS
    )
    assert sorted(set(every)) == [
        "identical_arms_separated",
        "identity_unrecorded",
        "instrument_below_bar",
        "instrument_leans",
        "instrument_unchecked",
        "measured_under_concurrency",
        "no_variation",
        "not_drawn_at_random",
        "not_recorded",
        "people_disagree",
        "replayed",
        "skipped_past_limit",
        "split",
        "too_few",
        "trials_lost",
        "unfinished",
    ]


@pytest.mark.parametrize(("state", "reason"), [(s, r) for s, reasons in ARCHITECTURE.items() for r in sorted(reasons)])
def test_every_pair_of_the_architecture_validates(state: str, reason: str) -> None:
    check_state(state, reason, _coordinate(reason))
    held = StateOf.model_validate({"state": state, "reason": reason, "coordinate": _coordinate(reason)})
    assert StateOf.model_validate(held.model_dump(mode="json")) == held


@pytest.mark.parametrize(
    ("state", "reason"),
    [(s, r) for s, r in product(ARCHITECTURE, sorted(set().union(*ARCHITECTURE.values()))) if r not in ARCHITECTURE[s]],
)
def test_every_cross_pair_is_refused(state: str, reason: str) -> None:
    with pytest.raises(ValueError, match="carries"):
        check_state(state, reason, _coordinate(reason))
    with pytest.raises(ValidationError):
        StateOf.model_validate({"state": state, "reason": reason, "coordinate": _coordinate(reason)})


def test_counted_carries_no_reason_and_every_other_state_carries_one() -> None:
    check_state("counted", None)
    for state in ("set_aside", "withheld", "not_available"):
        with pytest.raises(ValueError, match="carries one of its own reasons"):
            check_state(state, None)
    with pytest.raises(ValueError, match="is not a reading state"):
        check_state("excluded", None)


def test_a_split_and_an_unrecorded_identity_name_their_coordinate_and_no_other_reason_may() -> None:
    for state, reason in (("withheld", "split"), ("withheld", SHARED_REASON), ("not_available", SHARED_REASON)):
        with pytest.raises(ValueError, match="names the coordinate"):
            check_state(state, reason)
        for coordinate in get_args(Coordinate):
            check_state(state, reason, coordinate)
    with pytest.raises(ValueError, match="names no coordinate"):
        check_state("withheld", "unfinished", "served_model")
    with pytest.raises(ValueError, match="is not a coordinate"):
        check_state("withheld", "split", "weather")


def test_a_state_with_no_coordinate_dumps_none() -> None:
    assert StateOf(state="not_available", reason="too_few").model_dump(mode="json") == {
        "state": "not_available",
        "reason": "too_few",
    }


def test_an_arm_s_identity_holds_a_state_for_every_coordinate() -> None:
    assert set(ArmIdentity.model_fields) == set(get_args(Coordinate))
    whole = ArmIdentity(
        served_model="one", measure_definition={"on_time_rate": "unrecorded"}, template="one", rig="split"
    )
    assert ArmIdentity.model_validate(whole.model_dump(mode="json")) == whole
    with pytest.raises(ValidationError):
        ArmIdentity(rig="pooled")


# --- every state the designs in flight need -------------------------------------------------------------------------

#: What each design in flight must be able to say about a reading, a contrast or a figure, as the state, reason and
#: coordinate it will store. A later chunk that needs a pair this table lacks adds it here first.
NEEDED_STATES: dict[str, tuple[str, str | None, str | None]] = {
    "an arm answered by two models has no arm-level reading": ("withheld", "split", "served_model"),
    "an arm whose host measure was read under two definitions": ("withheld", "split", "measure_definition"),
    "an arm whose runs ran two template revisions": ("withheld", "split", "template"),
    "an arm measured under two rigs": ("withheld", "split", "rig"),
    "a run that recorded no measure definition": ("withheld", "identity_unrecorded", "measure_definition"),
    "a run whose responses named no model": ("withheld", "identity_unrecorded", "served_model"),
    "a contrast across two runs with an unrecorded coordinate": (
        "withheld",
        "identity_unrecorded",
        "measure_definition",
    ),
    "a run whose definitions a host attested reads as recorded": ("counted", None, None),
    "a reading over a run that did not finish": ("withheld", "unfinished", None),
    "latency read while cells ran at once": ("withheld", "measured_under_concurrency", None),
    "a finished run whose rig lost too many trials to read": ("withheld", "trials_lost", None),
    "a measure every one of whose trials the rig lost": ("not_available", "trials_lost", None),
    "a cost interval over two to four cases": ("not_available", "too_few", None),
    "a cost interval where every case cost the same": ("not_available", "no_variation", None),
    "an improvement whose judge has no check": ("set_aside", "instrument_unchecked", None),
    "an improvement whose judge fell short of its bar": ("set_aside", "instrument_below_bar", None),
    "an improvement whose judge leans toward the arm": ("set_aside", "instrument_leans", None),
    "an improvement people's reading contradicts": ("set_aside", "people_disagree", None),
    "an improvement on a reading where identical arms separated": ("set_aside", "identical_arms_separated", None),
    "a corrected reading over a run with an unrecorded coordinate": (
        "not_available",
        "identity_unrecorded",
        "served_model",
    ),
    "a corrected reading with too few labelled cases": ("not_available", "too_few", None),
    "a corrected reading where nobody read the contrast": ("not_available", "not_recorded", None),
    "an A/A check whose two sides replayed a recording": ("not_available", "replayed", None),
    "a preference over a sample that was not drawn at random": ("not_available", "not_drawn_at_random", None),
    "a blind read skipped past its limit": ("not_available", "skipped_past_limit", None),
    "a re-score with no repeated result": ("not_available", "not_recorded", None),
    "a judged improvement its judge's check supports": ("counted", None, None),
}

#: What a check must be able to say where its result is not met: the result, the reason and the coordinate.
NEEDED_CHECKS: dict[str, tuple[str, str, str, str | None]] = {
    "identical arms separated": ("identical_arms", "not_met", "identical_arms_separated", None),
    "the candidate was launched in another group": ("identical_arms", "insufficient", "not_recorded", None),
    "a run of the A/A has an unrecorded coordinate": ("identical_arms", "insufficient", "identity_unrecorded", "rig"),
    "a judge below its bar against people": ("people", "not_met", "instrument_below_bar", None),
    "a judge that leans, by people's double read": ("people", "not_met", "instrument_leans", None),
    "a reviewer that leans": ("model", "not_met", "instrument_leans", None),
    "a judge against people on too few results": ("people", "insufficient", "too_few", None),
    "a judge against a sample not drawn at random": ("people", "insufficient", "not_drawn_at_random", None),
    "people's reading contradicts the contrast": ("people", "not_met", "people_disagree", None),
    "a judge against its own repeats, too few": ("itself", "insufficient", "too_few", None),
}


@pytest.mark.parametrize("need", NEEDED_STATES)
def test_every_state_a_design_in_flight_needs_is_expressible(need: str) -> None:
    state, reason, coordinate = NEEDED_STATES[need]
    check_state(state, reason, coordinate)
    held = StateOf.model_validate({"state": state, "reason": reason, "coordinate": coordinate})
    # Each shape that will hold such a state takes it.
    contrast = Contrast(arm="b", measure="m", verdict="improved", state=held)
    assert contrast.state == held and Stat(state=held).state == held
    assert Corrected(against="people", state=held).state == held and Rescore(state=held).state == held
    assert FamilyComparison.model_fields["state"].annotation == StateOf | None


@pytest.mark.parametrize("need", NEEDED_CHECKS)
def test_every_check_result_a_design_in_flight_needs_is_expressible(need: str) -> None:
    against, result, reason, coordinate = NEEDED_CHECKS[need]
    check = Check.model_validate(
        {"instrument": "helpfulness", "against": against, "result": result, "reason": reason, "coordinate": coordinate}
    )
    assert Check.model_validate(check.model_dump(mode="json")) == check
    assert result in get_args(CriterionState) and reason in set().union(*ARCHITECTURE.values())


def test_a_check_that_could_not_be_made_says_so_as_a_state() -> None:
    void = Check(
        instrument="rig",
        against="identical_arms",
        result="insufficient",
        state=StateOf(state="not_available", reason="replayed"),
    )
    assert void.state is not None and void.state.reason == "replayed"


def _literals(tree: ast.AST) -> list[tuple[int, set[str]]]:
    """Each ``Literal[...]`` subscript of a module, with its line and the strings it holds."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "Literal":
            members = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            found.append(
                (node.lineno, {m.value for m in members if isinstance(m, ast.Constant) and isinstance(m.value, str)})
            )
    return found


def test_no_other_literal_in_the_package_holds_two_of_the_state_words() -> None:
    """A second closed vocabulary for "can this be used" would hold at least two of the four words."""
    states = set(get_args(ReadingState))
    home = SOURCE / "schema" / "reading_states.py"
    walked = 0
    offenders = []
    for path in sorted(SOURCE.rglob("*.py")):
        walked += 1
        for line, members in _literals(ast.parse(path.read_text(encoding="utf-8"))):
            if len(members & states) >= 2 and path != home:
                offenders.append(f"{path.relative_to(SOURCE)}:{line} holds {sorted(members & states)}")
    assert walked > 300, "the walk did not find the package"
    assert _literals(ast.parse(home.read_text(encoding="utf-8"))), "the walk finds no Literal where the states live"
    assert offenders == []


# --- every earlier word is mapped ---------------------------------------------------------------------------------

#: The words the table maps that are no closed set in code: a class, a field or a function that is itself the word.
#: Read off the code, so a rename or a retirement fails here until its row follows.
NAMED_WORDS: dict[str, set[str]] = {
    "AnswerSplit": {"answer_split"} if "answer_splits" in CampaignReadings.model_fields and AnswerSplit else set(),
    "StoppedShort": {stopped_short.__name__} if StoppedShort else set(),
    "RunSummary": {"ended_because"} & set(RunSummary.model_fields),
    "AnalysisContextBundle": {"latency_contended"} & set(AnalysisContextBundle.model_fields),
}


def _mapped() -> dict[str, set[str]]:
    mapped: dict[str, set[str]] = {}
    for row in STATE_WORDS:
        mapped.setdefault(row.vocabulary, set()).add(row.word)
    return mapped


def _short(name: str) -> str:
    """A discovered vocabulary's name as the table spells it: an alias by its name, a field as it is."""
    return name.rsplit(":", 1)[-1]


def test_every_vocabulary_that_could_be_about_a_reading_is_mapped_or_listed_as_not_one() -> None:
    """The structural half: nothing found by name or by reach is in neither place, and nothing is in both."""
    found = discovered()
    mapped = _mapped()
    listed = {name for names in NOT_ABOUT_A_READING.values() for name in names}

    neither = sorted(name for name in found if _short(name) not in mapped and name not in listed)
    both = sorted(name for name in found if _short(name) in mapped and name in listed)
    assert len(found) > 120, "the discovery found too little to be the package"
    assert neither == [], "map each in state_words.py, or list it in NOT_ABOUT_A_READING with why"
    assert both == []
    assert sorted(listed - set(found)) == [], "listed, and no longer in the package"
    assert sum(len(names) for names in NOT_ABOUT_A_READING.values()) == len(listed), "listed under two reasons"


def test_every_mapped_vocabulary_is_mapped_word_for_word() -> None:
    """A word added to a mapped vocabulary, or a row naming a word that is gone, fails here."""
    every_alias = aliases()
    found = discovered()
    in_code: dict[str, set[str]] = dict(NAMED_WORDS)
    for name, words in {**every_alias, **found}.items():
        if _short(name) in _mapped() and _short(name) not in NAMED_WORDS:
            assert in_code.setdefault(_short(name), set(words)) == set(words), f"two vocabularies are named {name}"

    assert _mapped() == in_code
    assert len(STATE_WORDS) == sum(len(words) for words in in_code.values()), "a word is mapped twice"


#: How many mapped vocabularies the package still spells in live code (issue #1197). The table is a debt register: a
#: vocabulary leaves it the day its last reader reads the shared state instead (a stored spelling stays, read through
#: the table), and this count falls with it. It never rises: a new way of saying "not counted" is a member of the
#: shared set, never a vocabulary mapped onto it. 37 when the ratchet started (S5a-2), and 37 after it: the call
#: sites that chunk moved onto the shared state read no mapped vocabulary to decide whether a reading counts. 36 since
#: B1: ``PredictionPoweredReading.unavailable_reason``, a sentence, became ``state``, a shared state and reason.
MAPPED_VOCABULARIES_IN_USE = 36


def test_the_mapped_vocabularies_still_in_use_only_fall() -> None:
    in_code = {_short(name) for name in {**aliases(), **discovered()}} | {
        name for name, words in NAMED_WORDS.items() if words
    }
    live = set(_mapped()) & in_code

    assert len(live) <= MAPPED_VOCABULARIES_IN_USE, "a mapped vocabulary was added: read the shared state instead"
    assert len(live) == MAPPED_VOCABULARIES_IN_USE, "a mapped vocabulary was retired: lower the count to match"
    assert set(_mapped()) == live, "a vocabulary gone from the code keeps rows only for spellings a store holds"


def test_the_vocabularies_the_review_named_are_each_mapped() -> None:
    assert {
        "VerdictReason",
        "MeasureState",
        "MechanismUncheckedReason",
        "EvidenceTier",
        "DefinitionState",
        "ContrastOutcome",
        "BarOutcome",
        "CaseOutcome",
        "DriftVerdict",
        "FrontierCostDecision",
        "LeverVerdict",
        "UnpairedReason",
        "UnrepeatedReason",
        "RaterKind",
    } <= set(_mapped())


def test_a_tier_is_no_reading_state() -> None:
    tiers = [row for row in STATE_WORDS if row.vocabulary in ("JudgedEvidenceTier", "EvidenceTier")]
    assert len(tiers) == 10 and all(row.state is None and row.note for row in tiers)


def test_the_words_for_trials_the_rig_lost_are_one_reason_that_takes_no_coordinate() -> None:
    lost = {
        f"{row.vocabulary}.{row.word}": (row.state, row.coordinate)
        for row in STATE_WORDS
        if row.reason == "trials_lost"
    }
    assert lost == {
        "VerdictRule.setup": ("withheld", None),
        "MeasureState.no_successful_results": ("not_available", None),
        "MeasureState.all_excluded": ("not_available", None),
    }
    assert REMEDY_OF_REASON["trials_lost"] == "repair_rig"
    for state in ("withheld", "not_available"):
        with pytest.raises(ValueError, match="names no coordinate"):
            check_state(state, "trials_lost", "rig")
    assert "ruling_needed" not in StateWord._fields, "every mapped word has a truthful reason, or says why it has none"


def test_the_judging_state_no_longer_spells_a_reading_state_and_its_earlier_spelling_reads_as_its_new_one() -> None:
    """``JudgingState.set_aside`` was judging skipped under a cost cap, and no reading set aside. It is ``not_asked``
    now: its row is the new word's, a condition handed back under the old one reads as the new, and the one closed set
    outside the shared one still spelling a reading state is the remainder's ``withheld``."""
    assert "set_aside" not in get_args(JudgingState) and "not_asked" in get_args(JudgingState)
    row = state_word("JudgingState", "not_asked")
    assert row.state is None and row.reason is None and row.note == JUDGING_NOT_ASKED
    assert (
        ResultCondition.model_validate({"termination": "completed", "judging": "set_aside", "scoring": "ok"}).judging
        == "not_asked"
    )
    colliding = sorted(
        f"{name}:{word}"
        for name, words in aliases().items()
        if not name.startswith("schema/reading_states.py:")
        for word in words & set(get_args(ReadingState))
    )
    assert colliding == [
        "analysis/remainder.py:RemainderState:withheld",
    ], "a closed set outside the shared one spells a reading state: map it, and say what it means there"


@pytest.mark.parametrize("row", STATE_WORDS, ids=lambda row: f"{row.vocabulary}.{row.word}")
def test_each_row_is_a_valid_state_or_says_what_the_word_is_instead(row: Any) -> None:
    assert state_word(row.vocabulary, row.word) is row
    if row.state is None:
        assert row.reason is None and row.coordinate is None and row.identity is None
        assert row.note, "a word that is no state says what it is"
        return
    if row.state != "counted" and row.reason is None:
        assert row.note, "a state whose reason the word does not fix says where the reason comes from"
        return
    check_state(row.state, row.reason, row.coordinate)
    if row.identity is not None and row.state != "counted":
        assert row.coordinate is not None, "an identity state is a state of one coordinate"
        assert {"split": "split", "unrecorded": "identity_unrecorded"}[row.identity] == row.reason


def test_the_cell_reasons_of_one_coordinate_split_and_unrecorded_alike() -> None:
    """The two words a coordinate has today (another value, none recorded) are one split and one unrecorded."""
    by_coordinate: dict[str, set[str]] = {}
    for row in STATE_WORDS:
        if row.vocabulary in ("UnpooledReason", "RefusedMerge.reason"):
            assert row.state == "withheld" and row.coordinate is not None
            by_coordinate.setdefault(row.coordinate, set()).add(str(row.reason))
    assert by_coordinate == dict.fromkeys(get_args(Coordinate), {"split", "identity_unrecorded"})
    assert set(get_args(Coordinate)) == {"served_model", "measure_definition", "template", "rig"}


def test_a_word_nobody_mapped_is_refused() -> None:
    with pytest.raises(KeyError):
        state_word("ServedModelState", "mixed")


# --- the caveat tables --------------------------------------------------------------------------------------------


def test_every_reason_has_a_remedy_and_every_remedy_closes_something() -> None:
    assert set(REMEDY_OF_REASON) == set(
        get_args(SetAsideReason) + get_args(WithheldReason) + get_args(NotAvailableReason)
    )
    # The kinds' remedies close the rest; test_caveat_list holds that every remedy closes some caveat.
    assert set(REMEDY_OF_REASON.values()) < set(get_args(Remedy))
    assert "no_action" not in REMEDY_OF_REASON.values()
    assert Reason == SetAsideReason | WithheldReason | NotAvailableReason


def test_a_kind_s_class_is_one_of_the_five_or_none() -> None:
    assert set(CLASS_OF_KIND.values()) == {*get_args(CaveatClass), None}
    assert CLASS_OF_KIND["set_aside"] == CLASS_OF_KIND["withheld"] == "A"
    assert CLASS_OF_KIND["late"] == CLASS_OF_KIND["confound"] == "B"
    assert CLASS_OF_KIND["judge_uncalibrated"] == "C"
    assert CLASS_OF_KIND["looks"] == "D"
    assert CLASS_OF_KIND["latency"] is None and CLASS_OF_KIND["unreported"] is None
