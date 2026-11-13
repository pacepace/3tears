"""The report's first screen in words: the arms line and the verdict line, both written by code, never by the writer.

**The verdict is the first rule that fires**, in this order, and it leads with the result, never a method caveat:

1. **The setup gate.** An arm lost more than a third of its trials to the test setup, or the arms' loss rates differ
   by more than a third of their trials: the run measured its test setup, not the models, so there is no verdict,
   and the line names the arm, how many it lost, the cause most of them share in words, and the remedy.
2. **One arm.** ``agent passed 31 of 36 trials on 12 cases.`` — its pass rate as the count it is computed over
   (:func:`passed_clause`), the campaign page's lead; then the cases it missed: ``Cases that failed a trial: r08,
   r09, r10.``
3. **Too small.** An arm has fewer than :data:`MIN_CASES` usable cases: which arm passed more trials, and that no
   difference is called; the case floor itself is a caveat line under the scoreboard (:func:`caveat_lines`).
4. **A declared question.** What its contrasts came to on the measure it leads on, in their own verdict words
   (``Accuracy: the evidence cannot tell v2 and v1 apart.``), then what else the evidence tells apart.
5. **Exploratory.** ``Exploratory:`` and the same, or with no lead measure ``the evidence cannot tell v2 and v1 apart
   on any reading``, never "no difference" and never "not separated", which a first-time reader took for "the same"
   (:func:`~threetears.evals.analysis.style.cannot_tell_apart`).

A guardrail an arm is not shown safe on is a second sentence, what it means for the decision first: ``v2 may do worse
than v1 on the No invented total score guardrail.``

**One result, one sentence.** The aspect page's verdict and the quick print's first line print
:attr:`VerdictLine.text`; the exit line opens with :attr:`~VerdictLine.lead`. A campaign page leading on a measure a
contrast tested says the same decision and figures (:func:`led_sentences`, over the figures
:func:`~threetears.evals.analysis.report.build.contrast_evidence` reads); one leading on an aspect's pass-rate row (its
margin's reading) says the row's own, in the same form (``scorecard.row_sentence`` and ``scorecard.row_figures``).
After the decision, each arm
is the subject of its figures and the interval says what it allows (:func:`interval_words`), never its level, which
the table's header states. Each measure and guardrail is named as the page names it, and a cost is never the lead
(:func:`_is_cost`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, get_args

from threetears.evals.analysis.bundle.schema import (
    AnalysisContextBundle,
    ArmExclusions,
    ArmScoreboard,
    FailCause,
)
from threetears.evals.analysis.quantities import is_money
from threetears.evals.analysis.stats import SMALL_N_BAND_FLOOR
from threetears.evals.analysis.style import cannot_tell_apart, measure_name
from threetears.evals.analysis.report.words import PASS_RATE, comparison_words, judged_heading
from threetears.evals.kernel.metrics import ACCURACY_MEASURE, METRIC_DESCRIPTORS
from threetears.evals.kernel.result_condition import CandidateFailureCause, ExclusionCause

if TYPE_CHECKING:
    from threetears.evals.analysis.report.blocks import Verdict

#: The fewest independent cases with a usable trial an arm needs before a rate on it is estimated, charted or
#: compared: below it a t interval is too wide and unstable to read (:data:`~threetears.evals.analysis.stats.SMALL_N_BAND_FLOOR`),
#: so a rate is printed as a count ("1 of 2"), never as a percentage, a difference or an interval.
#:
#: The floor counts independent cases, never trials or repeats. Below it no contrast is tested, the verdict says too
#: few, and no rate (a pass rate, pass^k) is estimated for an arm; a measure's mean is still shown with its interval.
#: A rate floor is read neither broken nor held below it, since both need an interval; a cost floor can still be
#: broken on a cost observed no lower. So a host's seed gate covers every template whose results are compared
#: across arms, whatever its declared reading.
MIN_CASES: Final = SMALL_N_BAND_FLOOR

#: The share of an arm's trials whose loss to the test setup, alone or as the gap between two arms, means the run
#: measured its setup rather than the models. A stand-in until a test across arms decides it.
SETUP_GATE_SHARE: Final = 1 / 3

#: Which rule decided the verdict line.
VerdictRule = Literal["setup", "one_arm", "too_small", "declared", "exploratory"]


@dataclass(frozen=True)
class CauseWords:
    """One cause in words: what happened, what a reader does about it, and the two words a table uses."""

    words: str
    remedy: str
    short: str = ""


#: Why a trial was lost to the test setup, in words, by :data:`~threetears.evals.kernel.result_condition.ExclusionCause`.
#: ``truncated``'s words and remedy name the cap the judge was sent, filled in by :func:`loss_words`.
LOSS_CAUSE_WORDS: Final[dict[str, CauseWords]] = {
    "infra": CauseWords(
        words=(
            "the test setup faulted (a simulator, a tool, the deadline or the judge's call failed), or the provider"
            " declined a call"
        ),
        remedy=(
            "read the lost trials' errors below: fix the part of the setup that faulted, then rerun every arm; a call"
            " the provider declined needs no fix, only a rerun once the provider can serve it (later, or with fewer"
            " calls at once)"
        ),
        short="setup fault",
    ),
    "model_unserved": CauseWords(
        words="the provider does not serve the candidate's model (retired, or not a model id it knows)",
        remedy=(
            "read the lost trials' errors below, which name the model and quote the provider: choose a model it"
            " serves, or wait until it serves this one again, then resume the arm's run"
        ),
        short="model not served",
    ),
    "judge_cannot_tell": CauseWords(
        words="the judge said the transcript did not let it score a dimension the trial needs",
        remedy="give the judge the evidence it asked for, or narrow the dimension, then rerun every arm",
        short="judge unsure",
    ),
    "truncated": CauseWords(
        words="the judge's reply was truncated at its output cap ({cap})",
        remedy="raise the judge's max_tokens ({sent}), then rerun {every}",
        short="judge truncated",
    ),
}

#: Why a usable trial failed, in words, by :data:`~threetears.evals.analysis.bundle.schema.FailCause`.
FAIL_CAUSE_WORDS: Final[dict[str, CauseWords]] = {
    "model_failed": CauseWords(
        words="a model the candidate runs failed or refused",
        remedy="read the failed trials' errors below: a different model or provider is the lever",
        short="model failed",
    ),
    "turn_budget": CauseWords(
        words="the host's turn budget ended a turn before it finished",
        remedy="a faster configuration (fewer rounds, less reasoning per round, a quicker model) is the lever",
        short="turn budget",
    ),
    "output_cap": CauseWords(
        words="the candidate's own reply was cut off at its output cap",
        remedy="a higher cap or a lower reasoning setting for the candidate is the lever",
        short="output cap",
    ),
    "check": CauseWords(
        words="a check failed",
        remedy="read the failed checks below",
        short="check",
    ),
    "judged": CauseWords(
        words="failed by the judge, on a judged dimension",
        remedy="read the judge's reasons below",
        short="judge",
    ),
}


def _covers(words: dict[str, CauseWords], values: Sequence[str], what: str) -> None:
    """Refuse, at import, a word table that does not cover exactly the causes it words.

    :raises RuntimeError: The table and the vocabulary disagree.
    """
    if set(words) != set(values):
        raise RuntimeError(f"the report's words for {what} {sorted(words)} do not match the causes {sorted(values)}")


_covers(LOSS_CAUSE_WORDS, get_args(ExclusionCause), "a lost trial")
_covers(FAIL_CAUSE_WORDS, get_args(FailCause), "a failed trial")
if not set(get_args(CandidateFailureCause)) <= set(get_args(FailCause)):
    raise RuntimeError("every candidate failure cause is a way a trial fails, and FailCause leaves one out")


def arm_exclusions(bundle: AnalysisContextBundle) -> dict[str, list[ArmExclusions]]:
    """The bundle's ``exclusions`` by arm: every cell of each arm, one per rig it ran under, in the bundle's order."""
    by_arm: dict[str, list[ArmExclusions]] = {}
    for entry in bundle.exclusions:
        by_arm.setdefault(entry.variant_key, []).append(entry)
    return by_arm


def cap_words(exclusions: Iterable[ArmExclusions] = ()) -> str:
    """The output cap a judge was sent on an arm's truncated trials, as ``N tokens`` — or that it went unreported.

    :param exclusions: The arm's cells' exclusions, whose ``truncated_sent_caps`` and ``truncated_cap_unreported`` say
        what the cut-off judge calls were sent, as their completions reported it.
    :ptype exclusions: Iterable[ArmExclusions]
    :return: The cap in words; never filled from what the run asked for.
    :rtype: str
    """
    cells = list(exclusions)
    known = [str(cap) for cap in sorted({cap for cell in cells for cap in cell.truncated_sent_caps})]
    unreported = any(cell.truncated_cap_unreported for cell in cells)
    if not known:
        return "its max_tokens not reported by the judge's client"
    said = f"{' or '.join(known)} tokens"
    return said + (", and unreported on some" if unreported else "")


#: The remedy for ``infra`` losses on an arm that measured nothing and whose run failed for it
#: (:func:`measured_nothing`), in place of the rerun the infra remedy offers: a provider that declined every call is a
#: setup to change, not a moment to wait out. The verdict says nothing was measured; this says what to do.
NOTHING_MEASURED_REMEDY: Final = (
    "clear what the lost trials' errors below name (a broken part of the setup, a model no allowed provider serves, or"
    " a rate limit or overload that lasted the run), then rerun {every}"
)

#: The remedy for ``infra`` losses on an arm whose run stopped because the provider refused the calling account
#: (status ``exhausted``). Worded for the arm's run, so it is true of every lost trial it is attached to: the bundle
#: does not say which trial the refusal ended, and the others' errors name their own causes.
ACCOUNT_REFUSED_REMEDY: Final = (
    "this arm's run stopped when the provider refused the account paying for it (out of credit, or its key"
    " refused): top up the account or fix its key, clear anything else the lost trials' errors below name, then"
    " rerun {every}"
)


def arm_run_statuses(bundle: AnalysisContextBundle) -> dict[str, frozenset[str]]:
    """Each arm's member runs' stored statuses, by variant key: ``completed`` for a run ``incomplete_runs`` omits.

    :param bundle: The campaign's evidence.
    :ptype bundle: AnalysisContextBundle
    :return: The statuses, for every arm the realized design places.
    :rtype: dict[str, frozenset[str]]
    """
    design = bundle.design
    arms = [*([design.control_arm] if design.control_arm is not None else []), *design.contrasts]
    return {
        placed.variant_key: frozenset(bundle.incomplete_runs.get(run_id, "completed") for run_id in placed.run_ids)
        for placed in arms
    }


def measured_nothing(exclusions: Iterable[ArmExclusions], run_statuses: Iterable[str]) -> bool:
    """Whether an arm measured nothing and its run failed for it: every trial lost to ``infra``, every run ``failed``.

    Read off the run's stored status as well as the losses, so a run a cap, a cancel or the account stopped, and a run
    stored before a run that measured nothing failed, keep their own words.

    :param exclusions: The arm's cells' exclusions (:func:`arm_exclusions`).
    :ptype exclusions: Iterable[ArmExclusions]
    :param run_statuses: The arm's runs' statuses (:func:`arm_run_statuses`).
    :ptype run_statuses: Iterable[str]
    :return: True when the arm ran trials, every one was excluded as ``infra``, and every run of it ended ``failed``.
    :rtype: bool
    """
    cells = list(exclusions)
    ran = sum(cell.n_trials for cell in cells)
    statuses = set(run_statuses)
    return statuses == {"failed"} and ran > 0 and sum(cell.by_cause.get("infra", 0) for cell in cells) == ran


def loss_words(
    cause: str, *, exclusions: Iterable[ArmExclusions] = (), arms: int = 2, run_statuses: Iterable[str] = ()
) -> CauseWords:
    """Why trials were lost to ``cause``, in words, with its remedy.

    :param cause: The loss cause.
    :ptype cause: str
    :param exclusions: The arm's cells' losses (:func:`arm_exclusions`), which carry the cap a ``truncated`` judge was
        sent.
    :ptype exclusions: Iterable[ArmExclusions]
    :param arms: How many arms the campaign ran, which the remedy names.
    :ptype arms: int
    :param run_statuses: The arm's runs' stored statuses (:func:`arm_run_statuses`); none known, the cause's own words.
    :ptype run_statuses: Iterable[str]
    :return: The words. A cause this build has no words for is named as itself and says so, rather than taking another
        cause's remedy. ``infra`` on an arm whose run the account stopped takes :data:`ACCOUNT_REFUSED_REMEDY`, and on
        an arm that measured nothing (:func:`measured_nothing`) :data:`NOTHING_MEASURED_REMEDY`, never the rerun.
    :rtype: CauseWords
    """
    cells = list(exclusions)
    worded = LOSS_CAUSE_WORDS.get(cause)
    if worded is None:
        return CauseWords(
            words=f"{cause} (a cause this build has no words for)", remedy="read the lost trials' errors below"
        )
    cap = cap_words(cells)
    sent = f"sent {cap.split(' tokens', 1)[0]}" if " tokens" in cap and cap[0].isdigit() else cap
    every = "both arms" if arms == 2 else "every arm"
    statuses = frozenset(run_statuses)
    remedy = worded.remedy
    # An ``exhausted`` run stopped for the account or for its model; a run its model stopped lost trials to
    # ``model_unserved``, so the account's remedy is only for an arm whose run lost none.
    if (
        cause == "infra"
        and statuses == {"exhausted"}
        and not any(cell.by_cause.get("model_unserved") for cell in cells)
    ):
        remedy = ACCOUNT_REFUSED_REMEDY
    elif cause == "infra" and measured_nothing(cells, statuses):
        remedy = NOTHING_MEASURED_REMEDY
    return CauseWords(
        words=worded.words.format(cap=cap), remedy=remedy.format(sent=sent, every=every), short=worded.short
    )


def cause_short(cause: str) -> str:
    """A loss or failure cause in the two or three words a table uses; a cause this build has no words for, as itself."""
    worded = LOSS_CAUSE_WORDS.get(cause) or FAIL_CAUSE_WORDS.get(cause)
    return worded.short if worded is not None else cause


def fail_words(cause: str) -> CauseWords:
    """Why trials failed for ``cause``, in words, with its remedy; a cause this build has no words for says so."""
    worded = FAIL_CAUSE_WORDS.get(cause)
    if worded is None:
        return CauseWords(words=f"{cause} (a cause this build has no words for)", remedy="read the trials below")
    return worded


@dataclass(frozen=True)
class VerdictLine:
    """The verdict as the first screen prints it: which rule fired, and the sentences it wrote.

    :ivar rule: The rule that fired.
    :ivar text: Every sentence: the result, then what else the evidence tells apart and each guardrail an arm is not
        shown safe on. The aspect page and the quick print print it.
    :ivar lead: The result's sentence alone, which the campaign page leads with (its guardrails are its warnings) and
        the exit line opens with; ``text`` where the line has one sentence.
    :ivar decisions: Each decision ``text`` says, word for word and in its order, which a page sets in bold while the
        figures it rests on and the guardrails stay regular; none where nothing is decided (one arm's count, a failed
        setup, too few cases).
    :ivar undecided: Whether a decision ``text`` says leaves arms undecided, so its words say the evidence cannot tell
        them apart: a led contrast not separated, or no reading telling any arm apart from the control. Read off the
        verdicts, never off the words, so a page links that phrase's definition where the line says it.
    """

    rule: VerdictRule
    text: str
    lead: str = ""
    decisions: tuple[str, ...] = ()
    undecided: bool = False

    def __post_init__(self) -> None:
        if not self.lead:
            object.__setattr__(self, "lead", self.text)


def lost_share(board: ArmScoreboard) -> float:
    """The share of an arm's trials lost to the test setup; 0 for an arm that ran none."""
    return sum(board.lost.values()) / board.ran if board.ran else 0.0


def passed_clause(arm: str, passed: int, trials: int, cases: int) -> str:
    """One arm's pass rate as a sentence states it: the count it is computed over, ``reply passed 40 of 40 trials on
    20 cases``, or ``reply passed 18 of 20 cases`` where each case ran once. The percent and its interval are the
    scorecard's, beside the same count: a sentence carries at most three figures
    (:data:`~threetears.evals.analysis.style.MAX_NUMERALS`).

    :param arm: The arm, as the page names it.
    :ptype arm: str
    :param passed: Its usable trials that passed.
    :ptype passed: int
    :param trials: Its usable trials.
    :ptype trials: int
    :param cases: The independent cases they ran on.
    :ptype cases: int
    :return: The clause, with no closing stop.
    :rtype: str
    """
    if trials == cases:
        return f"{arm} passed {passed} of {cases} cases"
    return f"{arm} passed {passed} of {trials} trials on {cases} cases"


def interval_words(
    arm: str,
    low: float,
    high: float,
    *,
    amount: Callable[[float], str],
    unit: str = "",
    higher_is_better: bool = True,
) -> str:
    """What an interval on an arm's change from the control allows, in a reader's words, the arm the subject: ``The
    interval allows v2 to be as much as 19.7 points worse or 17.8 better.`` where it reaches both sides of zero,
    ``The interval puts qwen3.8-flash between 3.4 and 10.8 points worse.`` where it lies on one side. Never a level:
    the table's header states it. An interval reaching both sides never reads as "the same".

    :param arm: The arm, as the page names it.
    :ptype arm: str
    :param low: The interval's low end, on the change's ruler (points for a rate).
    :ptype low: float
    :param high: Its high end.
    :ptype high: float
    :param amount: A change's size as the sentence states it, unsigned.
    :ptype amount: Callable[[float], str]
    :param unit: What follows the sizes once, `` points`` for a rate; empty where ``amount`` carries the unit.
    :ptype unit: str
    :param higher_is_better: The reading's direction: a rise is better where True.
    :ptype higher_is_better: bool
    :return: The sentence, with its closing stop.
    :rtype: str
    """
    worse_end, better_end = (low, high) if higher_is_better else (high, low)
    sign = 1 if higher_is_better else -1
    if sign * worse_end >= 0:
        near, far = sorted((abs(low), abs(high)))
        return f"The interval puts {arm} between {amount(near)} and {amount(far)}{unit} better."
    if sign * better_end <= 0:
        near, far = sorted((abs(low), abs(high)))
        return f"The interval puts {arm} between {amount(near)} and {amount(far)}{unit} worse."
    return f"The interval allows {arm} to be as much as {amount(worse_end)}{unit} worse or {amount(better_end)} better."


def _is_cost(measure: str, bundle: AnalysisContextBundle) -> bool:
    """Whether a measure is a cost: money, or on the cost merit axis, as the aspect's catalog describes it. A cost is
    never what a result leads on.

    :param measure: The measure's key.
    :ptype measure: str
    :param bundle: The aspect's evidence, whose catalog describes it.
    :ptype bundle: AnalysisContextBundle
    :return: True for a cost.
    :rtype: bool
    """
    described = bundle.measure_catalog.get(measure) or METRIC_DESCRIPTORS.get(measure)
    return described is not None and (described.merit_axis == "cost" or is_money(described.unit or ""))


def _lead_measure(bundle: AnalysisContextBundle, tested: Iterable[str], order: Sequence[str]) -> str | None:
    """The measure a result leads on: the first ``order`` lists (an aspect design's ``measure_order``, which an eval
    files in the order it lists its graders) that a contrast tested and that is no cost; with no order listed, the
    classifier's accuracy where no question is asked of the aspect; None where neither names one.

    :param bundle: The aspect's evidence.
    :ptype bundle: AnalysisContextBundle
    :param tested: The measures a contrast tested, by key (never the pass rate, ``trial``).
    :ptype tested: Iterable[str]
    :param order: The graders' measures, in the order listed; empty where none is listed.
    :ptype order: Sequence[str]
    :return: The measure's key, or None.
    :rtype: str | None
    """
    held = set(tested)
    if not order:
        return None if bundle.question_declared else ACCURACY_MEASURE
    return next((name for name in order if name in held and not _is_cost(name, bundle)), None)


def led_sentences(
    bundle: AnalysisContextBundle,
    verdicts: Sequence[Verdict],
    order: Sequence[str],
    *,
    asked: bool,
    on: str = "",
    evidence: Callable[[Verdict], str | None] | None = None,
    also: Sequence[str] = (),
) -> list[str]:
    """The result's sentence for each arm on the measure it leads on (:func:`_lead_measure`), in its verdict's words:
    ``Accuracy: the evidence cannot tell v2 and v1 apart.`` A contrast the question does not ask, where one is asked,
    is marked a lead. Empty where no contrast was tested on a lead measure.

    :param bundle: The aspect's evidence.
    :ptype bundle: AnalysisContextBundle
    :param verdicts: The aspect's typed verdicts, worded as its page words them.
    :ptype verdicts: Sequence[Verdict]
    :param order: The graders' measures, in the order listed.
    :ptype order: Sequence[str]
    :param asked: Whether a question is asked, so a contrast outside it is marked a lead.
    :ptype asked: bool
    :param on: Where it was read, said after the measure's name (`` on billing``) on a page of several aspects.
    :ptype on: str
    :param evidence: The figures each verdict rests on, as sentences (both arms' readings, the change and its
        interval, and for an undecided one the cases that would settle it), said after its decision; None says the
        decision alone.
    :ptype evidence: Callable[[Verdict], str | None] | None
    :param also: What else the evidence tells apart, said after the last arm's decision, before its figures.
    :ptype also: Sequence[str]
    :return: One decision per arm, each with its closing stop and its figures after it.
    :rtype: list[str]
    """
    led = _led(bundle, verdicts, order)
    said = []
    for index, verdict in enumerate(led):
        decision = (
            f"{verdict.heading}{on}: {verdict.words}{'' if verdict.aspect is not None or not asked else ', a lead'}"
        )
        if also and index == len(led) - 1:
            decision += f"; also {'; '.join(also)}"
        figures = evidence(verdict) if evidence is not None else None
        said.append(f"{decision}." + (f" {figures}" if figures else ""))
    return said


def _led(bundle: AnalysisContextBundle, verdicts: Sequence[Verdict], order: Sequence[str]) -> list[Verdict]:
    """The contrast verdicts on the measure a result leads on (:func:`_lead_measure`)."""
    contrasts = [verdict for verdict in verdicts if verdict.kind == "contrast" and verdict.reading != "trial"]
    lead = _lead_measure(bundle, (verdict.name for verdict in contrasts), order)
    return [verdict for verdict in contrasts if verdict.name == lead]


def verdict_line(
    bundle: AnalysisContextBundle,
    arm: Callable[[str], str],
    *,
    case_name: Callable[[str], str] | None = None,
    verdicts: Sequence[Verdict] = (),
    order: Sequence[str] = (),
    evidence: Callable[[Verdict], str | None] | None = None,
) -> VerdictLine:
    """The verdict line, by the first rule that fires (the module docstring's order).

    :param bundle: The campaign's evidence.
    :ptype bundle: AnalysisContextBundle
    :param arm: Names an arm by its variant key, as every table of the report does.
    :ptype arm: Callable[[str], str]
    :param case_name: Names a case by its id, as the cases table does; its id where None. One arm's line names the
        cases it missed by it.
    :ptype case_name: Callable[[str], str] | None
    :param verdicts: The aspect's typed verdicts as its page words them, whose words a comparison's line prints, so
        it reads as every other surface does; none reads the bundle's contrasts in their verdict words alone.
    :ptype verdicts: Sequence[Verdict]
    :param order: The graders' measures, in the order the eval lists them (``AspectDesign.measure_order``), which a
        comparison's line leads on the first of (:func:`_lead_measure`).
    :ptype order: Sequence[str]
    :param evidence: The figures each lead verdict rests on, said after its decision (:func:`led_sentences`); None
        says the decision alone.
    :ptype evidence: Callable[[Verdict], str | None] | None
    :return: The rule and its sentences.
    :rtype: VerdictLine
    """
    boards = bundle.scoreboard
    if not boards:
        return VerdictLine(rule="exploratory", text="No trial ran, so there is nothing to decide.")
    if (setup := _setup_gate(bundle, arm)) is not None:
        return VerdictLine(rule="setup", text=setup)
    if len(boards) == 1:
        (board,) = boards
        result = f"{_one_arm(board, arm)}."
        return VerdictLine(
            rule="one_arm", text=result + _missed_cases(bundle, case_name or (lambda case: case)), lead=result
        )
    declared_question = bundle.question_declared
    if any(board.usable_cases < MIN_CASES for board in boards):
        lead = "" if declared_question else "Exploratory: "
        return VerdictLine(rule="too_small", text=lead + _too_small(bundle, arm))
    rule: VerdictRule = "declared" if declared_question else "exploratory"
    prefix = "" if declared_question else "Exploratory: "
    if verdicts:
        if led := _led(bundle, verdicts, order):
            others = _also(bundle, verdicts, {verdict.heading for verdict in led})
            result = " ".join(led_sentences(bundle, verdicts, order, asked=declared_question, evidence=evidence))
            text = " ".join(
                led_sentences(bundle, verdicts, order, asked=declared_question, evidence=evidence, also=others)
            )
            # each decision alone, its closing stop off, as the line says it before its figures (or its "; also")
            decisions = tuple(
                said.removesuffix(".") for said in led_sentences(bundle, verdicts, order, asked=declared_question)
            )
            return VerdictLine(
                rule=rule,
                text=prefix + text + _guardrail_sentences(verdicts),
                lead=result,
                decisions=decisions,
                undecided=any(verdict.outcome == "not_separated" for verdict in led),
            )
    said = (
        _told_apart(bundle, verdicts, asked=declared_question)
        if verdicts
        else _separated(bundle, arm, asked=declared_question)
    )
    costs_only = (
        bool(verdicts)
        and bool(said)
        and all(
            _is_cost(verdict.name, bundle)
            for verdict in verdicts
            if verdict.kind == "contrast" and f"{verdict.heading}: {verdict.words}" in said
        )
    )
    # Whether the line says no reading tells any arm apart from the control (:func:`_undecided`).
    undecided = True
    if said and costs_only:
        # A cost is never the lead on its own: what the evidence leaves undecided first, then the cost.
        result = f"{_undecided(bundle, arm)} on every reading but cost; " + "; ".join(said)
    elif said:
        result, undecided = "; ".join(said), False
    elif declared_question and not any(family.aspect is not None for family in bundle.multiple_comparisons.families):
        result, undecided = "The declared question is untested", False
    else:
        result = _undecided(bundle, arm) + (
            " on any reading the question asks" if declared_question else " on any reading"
        )
    if declared_question:
        result = result[:1].upper() + result[1:]
    guardrails = (
        _guardrail_sentences(verdicts) if verdicts else "".join(f" {line}" for line in guardrail_lines(bundle, arm))
    )
    return VerdictLine(
        rule=rule, text=f"{prefix}{result}.{guardrails}", lead=f"{result}.", decisions=(result,), undecided=undecided
    )


def caveat_lines(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> list[str]:
    """What qualifies the verdict, said under the scoreboard in normal weight: never bold, never first.

    :param bundle: The campaign's evidence.
    :ptype bundle: AnalysisContextBundle
    :param arm: Names an arm by its variant key.
    :ptype arm: Callable[[str], str]
    :return: The lines: today, that an arm is under the case floor. A "too few" line says what is withheld and why, in
        the reader's terms, never a bare count: under the floor no interval is estimated, so a rate is shown as a
        count and no gap between the arms is sized.
    :rtype: list[str]
    """
    small = [board for board in bundle.scoreboard if board.usable_cases < MIN_CASES]
    if not small:
        return []
    named = ", ".join(f"{arm(board.variant_key)} {board.usable_cases}" for board in small)
    return [
        f"Usable cases: {named}. Under {MIN_CASES} independent cases a rate is shown as a count, and whether the "
        "arms' gap holds on more cases is not estimated."
    ]


def _control_key(bundle: AnalysisContextBundle) -> str | None:
    """The control's variant key, when it ran."""
    control = bundle.design.control_arm.variant_key if bundle.design.control_arm is not None else None
    return control if any(board.variant_key == control for board in bundle.scoreboard) else None


def _undecided(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> str:
    """That the evidence tells no arm apart from the control, the arms by name, the control last:
    ``the evidence cannot tell v2 and v1 apart`` (:func:`~threetears.evals.analysis.style.cannot_tell_apart`). Never
    "no difference", and never "nothing separated": an undecided contrast claims none."""
    key = _control_key(bundle) or bundle.scoreboard[0].variant_key
    others = [arm(board.variant_key) for board in bundle.scoreboard if board.variant_key != key]
    return cannot_tell_apart(others, arm(key))


def _one_arm(board: ArmScoreboard, arm: Callable[[str], str]) -> str:
    """One arm's pass rate as the count it is computed over (:func:`passed_clause`)."""
    if not board.usable:
        return f"{arm(board.variant_key)} ran {board.ran} trials, all lost to the test setup"
    return passed_clause(arm(board.variant_key), board.passed, board.usable, board.usable_cases)


#: How many missed cases one arm's line names before it counts the rest.
_MISSES_NAMED = 5


def _missed_cases(bundle: AnalysisContextBundle, case_name: Callable[[str], str]) -> str:
    """`` Cases that failed a trial: crash, both.`` — the cases one arm failed on at least one usable trial, in the
    trials' order, the first :data:`_MISSES_NAMED` by name and a count of the rest, a sentence of its own so the
    count line before it keeps its three figures; empty where it failed on none, or where the bundle holds only some
    of its failed trials (``trials_omitted``)."""
    if bundle.trials_omitted:
        return ""
    missed = list(dict.fromkeys(trial.case_id for trial in bundle.trials if trial.state == "failed"))
    if not missed:
        return ""
    shown = [case_name(case) or case for case in missed[:_MISSES_NAMED]]
    more = f", and {len(missed) - _MISSES_NAMED} more" if len(missed) > _MISSES_NAMED else ""
    return f" Cases that failed a trial: {', '.join(shown)}{more}."


def _also(bundle: AnalysisContextBundle, verdicts: Sequence[Verdict], led: set[str]) -> list[str]:
    """What else the evidence tells apart, beside the measure a result leads on: each other contrast shown better or
    worse, in its verdict's words, a cost after every other reading."""
    found = [
        verdict
        for verdict in verdicts
        if verdict.kind == "contrast" and verdict.outcome in _DIRECTIONAL and verdict.heading not in led
    ]
    found.sort(key=lambda verdict: _is_cost(verdict.name, bundle))
    return list(dict.fromkeys(f"{verdict.heading}: {verdict.words}" for verdict in found))


def _told_apart(bundle: AnalysisContextBundle, verdicts: Sequence[Verdict], *, asked: bool) -> list[str]:
    """Each contrast shown better or worse, in its verdict's words (``Accuracy: v2 better than v1``), a cost after every
    other reading; ``asked`` reads only the question's family."""
    found = [
        verdict
        for verdict in verdicts
        if verdict.kind == "contrast" and verdict.outcome in _DIRECTIONAL and (verdict.aspect is not None or not asked)
    ]
    found.sort(key=lambda verdict: _is_cost(verdict.name, bundle))
    return list(dict.fromkeys(f"{verdict.heading}: {verdict.words}" for verdict in found))


#: How a result's line says what a guardrail an arm is not shown safe on means for the decision, by its decision, the
#: breached first: on the verdict line, the aspect page's contrasts and the campaign's warnings alike.
GUARDRAIL_SAID: dict[str, str] = {
    "breached": "{arm} breaches {named}",
    "undecided": "{arm} may do worse than {control} on {named}",
}


def _guardrail_sentences(verdicts: Sequence[Verdict]) -> str:
    """`` v2 may do worse than v1 on the No invented total score guardrail.`` — one sentence per arm and decision,
    each guardrail named as the page heads it; empty when every guardrail held."""
    said = []
    for decision, template in GUARDRAIL_SAID.items():
        by_arm: dict[tuple[str, str], list[str]] = {}
        for verdict in verdicts:
            if verdict.kind == "guardrail" and verdict.outcome == decision:
                names = by_arm.setdefault((verdict.arm, verdict.control or ""), [])
                if verdict.heading not in names:
                    names.append(verdict.heading)
        for (name, control), headings in by_arm.items():
            said.append(" " + template.format(arm=name, control=control, named=_guardrails_named(headings)) + ".")
    return "".join(said)


def _too_small(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> str:
    """Which arm passed more of its usable trials than the control, in words, and that no difference is called."""
    control = _control_key(bundle) or bundle.scoreboard[0].variant_key
    boards = {board.variant_key: board for board in bundle.scoreboard}
    reference = boards[control]

    def share(board: ArmScoreboard) -> float:
        return board.passed / board.usable if board.usable else 0.0

    said = []
    for board in bundle.scoreboard:
        if board.variant_key == control:
            continue
        compared = (
            "more often than"
            if share(board) > share(reference)
            else "less often than"
            if share(board) < share(reference)
            else "as often as"
        )
        said.append(f"{arm(board.variant_key)} passed {compared} {arm(control)}")
    return "; ".join(said) + "; too few cases to call a difference."


def guardrail_name(bundle: AnalysisContextBundle, name: str, reading: str) -> str:
    """A guardrail as every surface names it, as its verdict heads it: a judged dimension by
    :func:`~threetears.evals.analysis.report.words.judged_heading`, a goal check by its declared name else its
    expression, any other measure by its reader name.

    :param bundle: The aspect's evidence, whose catalog and check names name it.
    :ptype bundle: AnalysisContextBundle
    :param name: The guardrail's measure key or judged dimension.
    :ptype name: str
    :param reading: ``judged`` for a judged dimension; anything else reads a measure.
    :ptype reading: str
    :return: The name.
    :rtype: str
    """
    if reading == "judged":
        return judged_heading(name)
    described = bundle.measure_catalog.get(name)
    return measure_name(
        name, described.reader_name if described is not None else None, declared=bundle.check_name(name)
    )


def guardrail_lines(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> list[str]:
    """What each guardrail an arm is not shown safe on means for the decision, one sentence per arm and decision,
    read off the bundle's guardrail checks: ``v2 may do worse than v1 on the No invented total score guardrail.``

    :param bundle: The aspect's evidence.
    :ptype bundle: AnalysisContextBundle
    :param arm: Names an arm by its variant key.
    :ptype arm: Callable[[str], str]
    :return: The sentences, the breached first; none where every guardrail held.
    :rtype: list[str]
    """
    said = []
    for decision, template in GUARDRAIL_SAID.items():
        by_arm: dict[tuple[str, str], list[str]] = {}
        for check in bundle.guardrails.checks:
            if check.decision == decision:
                heading = guardrail_name(bundle, check.name, check.reading)
                names = by_arm.setdefault((arm(check.contrast.variant_key), arm(check.control.variant_key)), [])
                if heading not in names:
                    names.append(heading)
        for (name, control), headings in by_arm.items():
            said.append(template.format(arm=name, control=control, named=_guardrails_named(headings)) + ".")
    return said


def _guardrails_named(headings: Sequence[str]) -> str:
    """Guardrails named in a sentence: ``the No leak score guardrail``, ``the A and B guardrails``."""
    listed = headings[0] if len(headings) == 1 else f"{', '.join(headings[:-1])} and {headings[-1]}"
    return f"the {listed} guardrail{'' if len(headings) == 1 else 's'}"


def _setup_gate(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> str | None:
    """The setup gate's sentences, or None when no arm lost too many trials and the arms lost alike."""
    boards = [board for board in bundle.scoreboard if board.ran]
    if not boards:
        return None
    shares = [lost_share(board) for board in boards]
    over = [board for board in boards if lost_share(board) > SETUP_GATE_SHARE]
    gap = max(shares) - min(shares) > SETUP_GATE_SHARE
    if not over and not gap:
        return None
    named = over or [max(boards, key=lost_share)]
    named = sorted(named, key=lambda board: (-lost_share(board), arm(board.variant_key)))
    exclusions = arm_exclusions(bundle)
    statuses = arm_run_statuses(bundle)

    def nothing(board: ArmScoreboard) -> bool:
        return measured_nothing(exclusions.get(board.variant_key, ()), statuses.get(board.variant_key, ()))

    # Said once: in the lead when no arm measured anything, else on each arm that measured nothing.
    every_arm = all(nothing(board) for board in boards)
    parts = []
    remedies: list[str] = []
    for board in named:
        lost = sum(board.lost.values())
        cause = max(sorted(board.lost), key=lambda name: board.lost[name])
        said = loss_words(
            cause,
            exclusions=exclusions.get(board.variant_key, ()),
            arms=len(boards),
            run_statuses=statuses.get(board.variant_key, ()),
        )
        most = "all" if board.lost[cause] == lost else "most"
        alone = " Its run failed before it measured a trial." if nothing(board) and not every_arm else ""
        parts.append(f"{arm(board.variant_key)} lost {lost} of {board.ran} trials, {most} because {said.words}.{alone}")
        if said.remedy not in remedies:
            remedies.append(said.remedy)
    apart = (
        []
        if over
        else ["The arms lost different shares of their trials, so their losses differ too much to compare them."]
    )
    lead = "The test setup failed: it measured the setup instead of the models"
    if every_arm:
        lead = "The test setup failed or the provider declined every call, so every trial was lost"
    return " ".join([f"{lead}.", *parts, *apart, f"Fix: {'; '.join(remedies)}."])


#: The contrast verdicts the verdict line names as one arm better than the other: the directional ones. Narrower than
#: :data:`~threetears.evals.analysis.bundle.schema.SEPARATED_VERDICTS` by design: an ``equivalent`` verdict settles a
#: comparison but names no better arm, so this line does not say it.
_DIRECTIONAL = ("improved", "regressed")


def _separated(bundle: AnalysisContextBundle, arm: Callable[[str], str], *, asked: bool) -> list[str]:
    """Each contrast that separated from the control, in words, both arms by label:
    ``v2 better than v1 on <reading>``, a lower-is-better reading's better end in brackets after it. ``asked`` reads
    only the question's family; otherwise every family."""
    said = []
    for family in bundle.multiple_comparisons.families:
        if asked and family.aspect is None:
            continue
        for comparison in family.comparisons:
            if comparison.verdict in _DIRECTIONAL:
                heading = bundle.measure_catalog.get(comparison.name)
                reading = measure_name(
                    comparison.name,
                    heading.reader_name if heading is not None else None,
                    declared=bundle.check_name(comparison.name),
                )
                if comparison.reading == "judged":
                    reading = judged_heading(comparison.name)
                elif comparison.reading == "trial":
                    reading = PASS_RATE
                words = comparison_words(
                    comparison.verdict,
                    arm(comparison.contrast.variant_key),
                    arm(comparison.control.variant_key),
                    higher_is_better=comparison.higher_is_better,
                )
                line = f"{reading}: {words}"
                if line not in said:
                    said.append(line)
    return said


__all__ = [
    "FAIL_CAUSE_WORDS",
    "GUARDRAIL_SAID",
    "ACCOUNT_REFUSED_REMEDY",
    "LOSS_CAUSE_WORDS",
    "NOTHING_MEASURED_REMEDY",
    "MIN_CASES",
    "SETUP_GATE_SHARE",
    "CauseWords",
    "caveat_lines",
    "VerdictLine",
    "VerdictRule",
    "arm_exclusions",
    "arm_run_statuses",
    "cap_words",
    "cause_short",
    "fail_words",
    "guardrail_lines",
    "guardrail_name",
    "interval_words",
    "led_sentences",
    "loss_words",
    "measured_nothing",
    "lost_share",
    "passed_clause",
    "verdict_line",
]
