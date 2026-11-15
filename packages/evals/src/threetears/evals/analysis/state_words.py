"""Every earlier word for "this reading is not being counted", and the shared state and reason it means.

The package grew a vocabulary per feature for one question. :mod:`~threetears.evals.analysis.reading_states` holds the
one closed set; this table says, for each earlier word still stored or still spelled in code, which state, reason and
coordinate it is. A stored value keeps its spelling and is read through this table; code that still decides by an
earlier word moves onto the shared one, and its rows leave with it.

A row is one of three things:

- **A reading state.** ``state`` is set, with the ``reason`` the word fixes, or none where the word alone does not fix
  one (the place that holds the word chooses it, and ``note`` says from what).
- **An identity state.** ``identity`` is set: the word says whether an arm is one thing on a coordinate.
- **Neither.** ``state`` and ``identity`` are None and ``note`` says what the word is instead: a lost trial's cause,
  a result of a counted reading, or a word that collides with a state and is renamed when its code is next touched.

``tests/test_reading_states.py`` holds that every member of every vocabulary named here has a row, so a word added
to one of them is mapped the day it lands.
"""

from __future__ import annotations

from typing import Final, NamedTuple

from threetears.evals.analysis.reading_states import Coordinate, IdentityState, ReadingState, Reason

__all__ = ["STATE_WORDS", "StateWord", "state_word"]


class StateWord(NamedTuple):
    """One earlier word and what it means.

    :ivar vocabulary: The closed set or field the word belongs to, by its name in code.
    :ivar word: The word, as stored or spelled.
    :ivar state: The reading state it is; None where it is no reading state.
    :ivar reason: The reason it fixes; None where it fixes none.
    :ivar coordinate: The coordinate it names.
    :ivar identity: The identity state it is, where the word says whether an arm is one thing.
    :ivar note: What the word is where it is no state, or where its reason comes from.
    """

    vocabulary: str
    word: str
    state: ReadingState | None = None
    reason: Reason | None = None
    coordinate: Coordinate | None = None
    identity: IdentityState | None = None
    note: str = ""


_LOST: Final = "a lost trial's cause, below the reading: no reading state"
_VERDICT: Final = "a verdict of a counted reading"
_CLAUSE: Final = "a clause's result on a counted reading"
_TIER: Final = "a summary of a judge's checks, never an input to a decision: the checks decide the state"

STATE_WORDS: Final[tuple[StateWord, ...]] = (
    # How many models answered an arm's candidate calls.
    StateWord("ServedModelState", "one", "counted", identity="one"),
    StateWord("ServedModelState", "pooled", "withheld", "split", "served_model", identity="split"),
    StateWord("ServedModelState", "unrecorded", "withheld", "identity_unrecorded", "served_model", "unrecorded"),
    # An arm answered by more than one model on an aspect, read per model.
    StateWord("AnswerSplit", "answer_split", "withheld", "split", "served_model", identity="split"),
    # Why a cell of one run pools with nothing, or is not another cell.
    StateWord("UnpooledReason", "other_served_model", "withheld", "split", "served_model", identity="split"),
    StateWord(
        "UnpooledReason", "no_recorded_served_model", "withheld", "identity_unrecorded", "served_model", "unrecorded"
    ),
    StateWord("UnpooledReason", "other_measure_definition", "withheld", "split", "measure_definition", "split"),
    StateWord(
        "UnpooledReason",
        "no_recorded_measure_definitions",
        "withheld",
        "identity_unrecorded",
        "measure_definition",
        "unrecorded",
    ),
    StateWord("UnpooledReason", "other_template", "withheld", "split", "template", identity="split"),
    StateWord("UnpooledReason", "no_recorded_template", "withheld", "identity_unrecorded", "template", "unrecorded"),
    StateWord("UnpooledReason", "changed_context", "withheld", "split", "rig", identity="split"),
    StateWord("UnpooledReason", "no_recorded_context", "withheld", "identity_unrecorded", "rig", "unrecorded"),
    # A run's stored status: the evidence is whole under one of them.
    StateWord("EvalRunStatus", "completed", "counted"),
    StateWord("EvalRunStatus", "failed", "withheld", "unfinished"),
    StateWord("EvalRunStatus", "exhausted", "withheld", "unfinished"),
    StateWord("EvalRunStatus", "cancelled", "withheld", "unfinished"),
    StateWord("EvalRunStatus", "budget_stopped", "withheld", "unfinished"),
    StateWord("EvalRunStatus", "running", "withheld", "unfinished"),
    StateWord("EvalRunStatus", "pending", "withheld", "unfinished"),
    # A reading over runs that did not finish, and the run record's own words for why.
    StateWord("StoppedShort", "stopped_short", "withheld", "unfinished"),
    StateWord("RunSummary", "ended_because", note="a run record's free text beside an unfinished gate: no reason"),
    # Which rule decided a page's verdict line.
    StateWord("VerdictRule", "setup", "not_available", "not_recorded", note="the first-screen gate's setup rule"),
    StateWord("VerdictRule", "unfinished", "withheld", "unfinished", note="the first-screen gate's unfinished rule"),
    StateWord("VerdictRule", "too_small", "not_available", "too_few", note="the first-screen gate's too_small rule"),
    StateWord("VerdictRule", "one_arm", "counted"),
    StateWord("VerdictRule", "declared", "counted"),
    StateWord("VerdictRule", "exploratory", "counted"),
    # Why a trial was lost.
    StateWord("ExclusionCause", "infra", note=_LOST),
    StateWord("ExclusionCause", "model_unserved", note=_LOST),
    StateWord("ExclusionCause", "cost_cap", note=_LOST),
    StateWord("ExclusionCause", "truncated", note=_LOST),
    StateWord("ExclusionCause", "judge_cannot_tell", note=_LOST),
    # Whether a remainder is a number.
    StateWord("RemainderState", "stated", "counted"),
    StateWord("RemainderState", "withheld", "withheld", note="the reason is its sentence's cause, typed with it"),
    # A contrast's verdict.
    StateWord("ComparisonVerdict", "improved", "counted", note=_VERDICT),
    StateWord("ComparisonVerdict", "regressed", "counted", note=_VERDICT),
    StateWord("ComparisonVerdict", "equivalent", "counted", note=_VERDICT),
    StateWord("ComparisonVerdict", "not_separated", "counted", note=_VERDICT),
    StateWord(
        "ComparisonVerdict", "untested", "not_available", note="too_few, no_variation or not_recorded, by its cause"
    ),
    # What a judged reading's scores can bear.
    StateWord("JudgedEvidenceTier", "calibrated", note=_TIER),
    StateWord("JudgedEvidenceTier", "separation", note=_TIER),
    StateWord("JudgedEvidenceTier", "incidental", note=_TIER),
    StateWord("JudgedEvidenceTier", "undetermined", "set_aside", "instrument_unchecked"),
    # A bar's verdict on one cell.
    StateWord("BarDecision", "cleared", "counted", note=_CLAUSE),
    StateWord("BarDecision", "missed", "counted", note=_CLAUSE),
    StateWord("BarDecision", "undecided", "counted", note=_CLAUSE),
    StateWord("BarDecision", "no_interval", "not_available", "too_few"),
    StateWord("BarDecision", "no_data", "not_available", "not_recorded"),
    # How a result's judging went: trial-level, and one member collides with a reading state.
    StateWord("JudgingState", "scored", note="a judged trial"),
    StateWord("JudgingState", "partial", note="a judged trial, some dimensions unscored"),
    StateWord("JudgingState", "failed", note=_LOST),
    StateWord("JudgingState", "not_attempted", note="a trial no judge was asked about"),
    StateWord("JudgingState", "set_aside", note="judging skipped under a cost cap: not_asked when next touched"),
    # Latency read while cells ran at once.
    StateWord("AnalysisContextBundle", "latency_contended", "withheld", "measured_under_concurrency"),
)

_BY_WORD: Final = {(row.vocabulary, row.word): row for row in STATE_WORDS}


def state_word(vocabulary: str, word: str) -> StateWord:
    """What an earlier word means.

    :param vocabulary: The closed set or field the word belongs to, by its name in code.
    :ptype vocabulary: str
    :param word: The word.
    :ptype word: str
    :return: Its row.
    :rtype: StateWord
    :raises KeyError: If the table holds no such word: a word nobody mapped is refused, never read as counted.
    """
    return _BY_WORD[vocabulary, word]
