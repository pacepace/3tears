"""Every earlier word for "this reading is not being counted", and the shared state and reason it means.

The package grew a vocabulary per feature for one question. :mod:`~threetears.evals.schema.reading_states` holds the
one closed set; this table says, for each earlier word still stored or still spelled in code, which state, reason and
coordinate it is. A stored value keeps its spelling and is read through this table; code that still decides by an
earlier word moves onto the shared one, and its rows leave with it.

A row is one of three things:

- **A reading state.** ``state`` is set, with the ``reason`` the word fixes, or none where the word alone does not fix
  one (the place that holds the word chooses it, and ``note`` says from what).
- **An identity state.** ``identity`` is set: the word says whether an arm is one thing on a coordinate.
- **Neither.** ``state`` and ``identity`` are None and ``note`` says what the word is instead: a lost trial's cause,
  a result of a check, who made a judgment, or a word that collides with a state and is renamed when its code is next
  touched.

``tests/test_reading_states.py`` finds the package's closed vocabularies by structure (every closed set of words a
stored bundle, report input, run or result can hold, and every one named as a state, a reason, a verdict or an
outcome) and holds that each is mapped here, word for word, or listed there as no word about a reading, with why. So
a vocabulary added later is mapped or listed the day it lands.
"""

from __future__ import annotations

from typing import Final, NamedTuple

from threetears.evals.schema.reading_states import Coordinate, IdentityState, ReadingState, Reason

#: Nothing is declared yet: the table's readers are the call sites that move onto the shared words.
__all__: list[str] = []


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
_TRIAL: Final = "one trial's outcome, below the reading: no reading state"
_VERDICT: Final = "a verdict of a counted reading"
_WHY: Final = "why a counted reading's verdict came out as it did"
_CLAUSE: Final = "a clause's result on a counted reading"
_TIER: Final = "a summary of a judge's checks, never an input to a decision: the checks decide the state"
_BY_CAUSE: Final = "too_few, no_variation or not_recorded, by its cause"
_ITEM: Final = "why one item left a check's sample, below the check as a lost trial is below a reading"
_SEED: Final = "a stored definition read against the seed corpus: an operator's drift read, no reading's identity"
_MECHANISM: Final = "why a lever's mechanism went unchecked: a check that is insufficient, never a reading state"


#: What ``JudgingState.not_asked`` is: judging the cost cap kept the judge from, spelled ``set_aside`` before the
#: shared reading state took that word, and read as ``not_asked`` when handed back under it.
JUDGING_NOT_ASKED: Final = (
    "judging skipped under a cost cap, a trial's fact; its earlier spelling set_aside reads as it, and is no reading "
    "set aside"
)

_BAR: Final = ("cleared", "missed", "undecided")
_MECHANISM_STATES: Final = ("moved", "inert", "unchecked")


def _verdicts(vocabulary: str, *counted: str, untested: str = "untested") -> tuple[StateWord, ...]:
    """A verdict vocabulary: each verdict of a counted reading, and the one word for no test."""
    return (
        *(StateWord(vocabulary, word, "counted", note=_VERDICT) for word in counted),
        StateWord(vocabulary, untested, "not_available", note=_BY_CAUSE),
    )


def _all(vocabulary: str, note: str, *words: str) -> tuple[StateWord, ...]:
    """Words of one vocabulary that are no reading state, for one stated cause."""
    return tuple(StateWord(vocabulary, word, note=note) for word in words)


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
    # Why two cells of one arm were not merged: the rig's own coordinate.
    StateWord("RefusedMerge.reason", "apparatus_differs", "withheld", "split", "rig", identity="split"),
    StateWord("RefusedMerge.reason", "provenance_differs", "withheld", "split", "rig", identity="split"),
    StateWord("RefusedMerge.reason", "apparatus_unknown", "withheld", "identity_unrecorded", "rig", "unrecorded"),
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
    # Which rule decided a page's verdict line. The first three are the first-screen gate's rules.
    StateWord("VerdictRule", "setup", "withheld", "trials_lost"),
    StateWord("VerdictRule", "unfinished", "withheld", "unfinished"),
    StateWord("VerdictRule", "too_small", "not_available", "too_few"),
    StateWord("VerdictRule", "one_arm", "counted"),
    StateWord("VerdictRule", "declared", "counted"),
    StateWord("VerdictRule", "exploratory", "counted"),
    # A trial, and why one was lost.
    *_all("ExclusionCause", _LOST, "infra", "model_unserved", "cost_cap", "truncated", "judge_cannot_tell"),
    *_all("TrialState", _TRIAL, "passed", "failed", "lost"),
    *_all("CaseOutcome", _TRIAL, "scored", "not_judged", "failed", "excluded"),
    # Whether a remainder is a number.
    StateWord("RemainderState", "stated", "counted"),
    StateWord("RemainderState", "withheld", "withheld", note="the reason is its sentence's cause, typed with it"),
    # A contrast's verdict, in each of its spellings.
    *_verdicts("ComparisonVerdict", "improved", "regressed", "equivalent", "not_separated"),
    *_verdicts("MovementDirection", "improved", "regressed", "equivalent", "not_separated"),
    *_verdicts("ContrastOutcome", "improved", "regressed", "equivalent", "not_separated"),
    *_verdicts("ChangeLabel", "improved", "regressed", "equivalent", "below_threshold", "not_separated"),
    *_verdicts("DriftVerdict", "separated", "not_separated"),
    *_verdicts("FrontierCostDecision", "shown_cheapest", "not_separated", "only_cleared"),
    *_verdicts("FrontierDominance", "dominated", "not_separated"),
    *_verdicts("LeverVerdict", "reverses", "differs", "within", untested="no_interval"),
    # Why a verdict came out as it did.
    *(
        StateWord("VerdictReason", word, "counted", note=_WHY)
        for word in (
            "separated",
            "inside_margin",
            "no_margin",
            "margin_untested",
            "not_inside_margin",
            "interval_clears",
            "interval_misses",
            "interval_straddles",
            "within_margin",
            "beyond_margin",
        )
    ),
    StateWord("VerdictReason", "untestable", "not_available", note=_BY_CAUSE),
    StateWord("VerdictReason", "no_interval", "not_available", note=_BY_CAUSE),
    StateWord("VerdictReason", "too_few_cases", "not_available", "too_few"),
    StateWord("VerdictReason", "too_few_observations", "not_available", "too_few"),
    StateWord("VerdictReason", "no_observations", "not_available", "not_recorded"),
    # What a judged reading's scores can bear: a tier is a summary of checks and is no state, any of them.
    *_all("JudgedEvidenceTier", _TIER, "calibrated", "separation", "incidental", "undetermined"),
    *_all("EvidenceTier", _TIER, "mechanical", "calibrated", "separation", "undetermined", "incidental", "none"),
    # A bar's verdict on one cell, in both its spellings, and whether a bar could be read at all.
    *(StateWord(name, word, "counted", note=_CLAUSE) for name in ("BarDecision", "BarOutcome") for word in _BAR),
    *(StateWord(name, "no_interval", "not_available", "too_few") for name in ("BarDecision", "BarOutcome")),
    *(StateWord(name, "no_data", "not_available", "not_recorded") for name in ("BarDecision", "BarOutcome")),
    StateWord("_BarState", "adjudicated", "counted", note=_CLAUSE),
    StateWord("_BarState", "names_no_stored_measure", "not_available", "not_recorded"),
    StateWord("_BarState", "not_numeric", note="a bar declared on something no bar can be read on"),
    # Whether a run's summary can state a measure.
    StateWord("MeasureState", "measured", "counted"),
    StateWord("MeasureState", "not_a_number", note="what kind of measure it is: described, with no figure to count"),
    StateWord("MeasureState", "in_confusion_matrix", note="what kind of measure it is: read in its own table"),
    StateWord("MeasureState", "not_carried", "not_available", "not_recorded"),
    StateWord("MeasureState", "no_successful_results", "not_available", "trials_lost"),
    StateWord("MeasureState", "all_excluded", "not_available", "trials_lost"),
    # A lever's mechanism check.
    *_all("MechanismCheck.state", "a finding about a lever, or a check of it that is insufficient", *_MECHANISM_STATES),
    *_all(
        "MechanismUncheckedReason",
        _MECHANISM,
        "not_declared",
        "not_swept",
        "levels_unobserved",
        "too_few_observations",
        "uniform_move_needs_range",
    ),
    # Why a rating or a repeat left a check's sample.
    *_all("UnpairedReason", _ITEM, "result_unresolved", "dimension_unscored", "scale_changed", "rated_by_an_agent"),
    *_all(
        "UnrepeatedReason",
        _ITEM,
        "repeat_failed",
        "judge_changed",
        "config_changed",
        "temperature_changed",
        "settings_changed",
    ),
    *_all("UnpairedSecondReason", _ITEM, "second_failed"),
    # Who made a rating: the shared word for a judgment's maker is person or model.
    StateWord("RaterKind", "person", note="who made a judgment: a person"),
    StateWord("RaterKind", "agent", note="who made a judgment: a model"),
    # A stored definition against the seed corpus.
    *_all("DefinitionState", _SEED, "current", "stale", "customized", "unknown", "absent"),
    # How a result's judging went: one trial's fact each, no reading state.
    StateWord("JudgingState", "scored", note="a judged trial"),
    StateWord("JudgingState", "partial", note="a judged trial, some dimensions unscored"),
    StateWord("JudgingState", "failed", note=_LOST),
    StateWord("JudgingState", "not_attempted", note="a trial no judge was asked about"),
    StateWord("JudgingState", "not_asked", note=JUDGING_NOT_ASKED),
    # Latency read while cells ran at once.
    StateWord("AnalysisContextBundle", "latency_contended", "withheld", "measured_under_concurrency"),
    # The one free-text reason a stored reading holds for an absent figure.
    StateWord("PredictionPoweredReading", "unavailable_reason", "not_available", note="a sentence today: " + _BY_CAUSE),
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
