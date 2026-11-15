"""The shared reading states: one closed set, one set of reasons, and every earlier word mapped onto them.

Contracts held here (``threetears.evals.analysis.reading_states``, ``state_words`` and ``caveats``):

- every state and reason pair the information architecture lists validates, and every cross pair is refused;
- ``counted`` carries no reason; a split names its coordinate; only a split or an unrecorded identity may name one;
- no other ``Literal`` in the package holds two or more of the four state words, so a second vocabulary for "is this
  counted" cannot appear unseen;
- every member of every earlier vocabulary has a row in the one mapping table, every row names a word that still
  exists, and each row is a valid state and reason, or says what the word is instead;
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
from threetears.evals.analysis.bundle import schema as bundle_schema
from threetears.evals.analysis.campaign_readings import AnswerSplit, CampaignReadings
from threetears.evals.analysis.caveats import CLASS_OF_KIND, REMEDY_OF_REASON, Remedy
from threetears.evals.analysis.reading_states import (
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
from threetears.evals.analysis.remainder import RemainderState
from threetears.evals.analysis.report.verdict import VerdictRule
from threetears.evals.analysis.reporting import ServedModelState
from threetears.evals.analysis.run_ending import StoppedShort, stopped_short
from threetears.evals.analysis.state_words import STATE_WORDS, state_word
from threetears.evals.kernel.analysis_measures import BarDecision
from threetears.evals.kernel.evidence_tiers import JudgedEvidenceTier
from threetears.evals.kernel.result_condition import ExclusionCause, JudgingState
from threetears.evals.kernel.scoring import UnpooledReason
from threetears.evals.schema.models import EvalRunStatus

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
    "withheld": {"unfinished", "split", "identity_unrecorded", "measured_under_concurrency"},
    "not_available": {
        "too_few",
        "no_variation",
        "not_recorded",
        "replayed",
        "not_drawn_at_random",
        "skipped_past_limit",
    },
}


def _coordinate(reason: str) -> str | None:
    return "served_model" if reason == "split" else None


# --- the closed sets ----------------------------------------------------------------------------------------------


def test_the_states_and_their_reasons_are_the_architecture_s() -> None:
    assert set(get_args(ReadingState)) == set(ARCHITECTURE)
    assert {state: set(reasons) for state, reasons in REASONS_OF.items()} == ARCHITECTURE
    assert set(get_args(SetAsideReason)) == ARCHITECTURE["set_aside"]
    assert set(get_args(WithheldReason)) == ARCHITECTURE["withheld"]
    assert set(get_args(NotAvailableReason)) == ARCHITECTURE["not_available"]
    assert set(get_args(IdentityState)) == {"one", "split", "unrecorded"}


def test_no_reason_belongs_to_two_states() -> None:
    every = [reason for reasons in ARCHITECTURE.values() for reason in reasons]
    assert len(every) == len(set(every)) == 15


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


def test_a_split_names_its_coordinate_and_only_an_identity_reason_may() -> None:
    with pytest.raises(ValueError, match="names the coordinate"):
        check_state("withheld", "split")
    for coordinate in get_args(Coordinate):
        check_state("withheld", "split", coordinate)
        check_state("withheld", "identity_unrecorded", coordinate)
    check_state("withheld", "identity_unrecorded")
    with pytest.raises(ValueError, match="names no coordinate"):
        check_state("withheld", "unfinished", "served_model")
    with pytest.raises(ValueError, match="is not a coordinate"):
        check_state("withheld", "split", "weather")


def test_a_state_with_no_coordinate_dumps_none() -> None:
    assert StateOf(state="not_available", reason="too_few").model_dump(mode="json") == {
        "state": "not_available",
        "reason": "too_few",
    }
    assert ArmIdentity().model_dump(mode="json") == {"served_model": None, "measure_definition": {}}


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
    home = SOURCE / "analysis" / "reading_states.py"
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

#: Each earlier vocabulary the table maps, by the name the table gives it, with its words as the code holds them
#: today. A closed set gives its members; a word that is a class or a field gives itself, read off the code so a
#: rename or a retirement fails here until its row follows.
VOCABULARIES: dict[str, set[str]] = {
    "ServedModelState": set(get_args(ServedModelState)),
    "UnpooledReason": set(get_args(UnpooledReason)),
    "EvalRunStatus": set(get_args(EvalRunStatus)),
    "VerdictRule": set(get_args(VerdictRule)),
    "ExclusionCause": set(get_args(ExclusionCause)),
    "RemainderState": set(get_args(RemainderState)),
    "ComparisonVerdict": set(get_args(bundle_schema.ComparisonVerdict)),
    "JudgedEvidenceTier": set(get_args(JudgedEvidenceTier)),
    "BarDecision": set(get_args(BarDecision)),
    "JudgingState": set(get_args(JudgingState)),
    "AnswerSplit": {"answer_split"} if "answer_splits" in CampaignReadings.model_fields and AnswerSplit else set(),
    "StoppedShort": {stopped_short.__name__} if StoppedShort else set(),
    "RunSummary": {"ended_because"} & set(bundle_schema.RunSummary.model_fields),
    "AnalysisContextBundle": {"latency_contended"} & set(bundle_schema.AnalysisContextBundle.model_fields),
}


def test_every_word_of_every_earlier_vocabulary_is_mapped_and_no_row_names_a_word_that_is_gone() -> None:
    mapped: dict[str, set[str]] = {}
    for row in STATE_WORDS:
        mapped.setdefault(row.vocabulary, set()).add(row.word)
    assert mapped == VOCABULARIES
    assert len(STATE_WORDS) == sum(len(words) for words in VOCABULARIES.values()), "a word is mapped twice"


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
        if row.vocabulary == "UnpooledReason":
            assert row.state == "withheld" and row.coordinate is not None
            by_coordinate.setdefault(row.coordinate, set()).add(str(row.reason))
    assert by_coordinate == dict.fromkeys(get_args(Coordinate), {"split", "identity_unrecorded"})


def test_a_word_nobody_mapped_is_refused() -> None:
    with pytest.raises(KeyError):
        state_word("ServedModelState", "mixed")


# --- the caveat tables --------------------------------------------------------------------------------------------


def test_every_reason_has_a_remedy_and_every_remedy_closes_something() -> None:
    assert set(REMEDY_OF_REASON) == set(
        get_args(SetAsideReason) + get_args(WithheldReason) + get_args(NotAvailableReason)
    )
    assert set(REMEDY_OF_REASON.values()) | {"declare_looks"} == set(get_args(Remedy))
    assert Reason == SetAsideReason | WithheldReason | NotAvailableReason


def test_a_kind_s_class_is_one_of_the_five_or_none() -> None:
    assert set(CLASS_OF_KIND.values()) == {*get_args(CaveatClass), None}
    assert CLASS_OF_KIND["set_aside"] == CLASS_OF_KIND["withheld"] == "A"
    assert CLASS_OF_KIND["late"] == CLASS_OF_KIND["confound"] == "B"
    assert CLASS_OF_KIND["judge_uncalibrated"] == "C"
    assert CLASS_OF_KIND["looks"] == "D"
    assert CLASS_OF_KIND["latency"] is None and CLASS_OF_KIND["unreported"] is None
