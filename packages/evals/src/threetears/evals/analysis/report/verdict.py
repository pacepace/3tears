"""The report's first screen in words: the arms line and the verdict line, both written by code, never by the writer.

**The verdict is the first rule that fires**, in this order, and it leads with the result, never a method caveat.
Before any of them, a reading with a run that did not finish and no failed setup opens on what stopped the run, with
no verdict (:func:`unfinished_line`, the ``unfinished`` rule): whether there is a verdict to open with is one decision,
:func:`~threetears.evals.analysis.run_ending.stopped_short`, which every surface asks.

1. **The setup gate.** An arm lost more than a third of its trials to the test setup, or the arms' loss rates differ
   by more than a third of their trials: the run measured its test setup, not the models, so there is no verdict,
   and the line names the arm, how many it lost, the cause most of them share in words, and the remedy.
2. **One arm.** ``agent passed 31 of 36 trials on 12 cases.`` — its pass rate as the count it is computed over
   (:func:`passed_clause`), the campaign page's lead; then the cases it missed: ``Cases that failed a trial: r08,
   r09, r10.`` An arm split on a coordinate of its identity has no one pass rate, and the line says why. An arm whose
   candidate failed every graded trial without answering says that and the first error, with no pass count and no
   word about its graders (``gloss failed on all 4 cases, so no answer was graded: ...``); arms that all did so each
   say it, and nothing is compared.
3. **Too small.** An arm has fewer than :data:`MIN_CASES` usable cases: which arm passed more trials, and that no
   difference is called; the case floor itself is a caveat line under the scoreboard (:func:`caveat_lines`).
4. **A declared question.** What its contrasts came to on the measure it leads on, in their own verdict words
   (``Accuracy: the evidence cannot tell v2 and v1 apart.``), then what else the evidence tells apart.
5. **Exploratory.** ``Exploratory:`` and the same, or with no lead measure ``the evidence cannot tell v2 and v1 apart
   on any reading``, never "no difference" and never "not separated", which a first-time reader took for "the same"
   (:func:`~threetears.evals.analysis.style.cannot_tell_apart`).

A guardrail an arm is not shown safe on is a second sentence, what it means for the decision first: ``v2 may do worse
than v1 on the No invented total score guardrail.``

**An arm not read against the control is never compared with it** (:func:`~threetears.evals.analysis.cells.unread_arms`,
the one question every arm-level reader asks): an arm split on the answering model or on a host measure's definition,
or two arms read under different definitions, or under none recorded. No rule says it passed more or less often, or
that the evidence cannot tell it apart; a sentence of its own says it is not compared, and why (:func:`_not_compared`).

**One result, one sentence.** The aspect page's verdict and the quick print's first line print
:attr:`VerdictLine.text`; the exit line opens with :attr:`~VerdictLine.lead`. A campaign page leading on a measure a
contrast tested says the same decision and figures (:func:`led_sentences`, over the figures
:func:`~threetears.evals.analysis.report.aspect.contrasts.contrast_evidence` reads); one leading on an aspect's pass-rate row (its
margin's reading) says the row's own, in the same form (``scorecard.row_sentence`` and ``scorecard.row_figures``).
After the decision, each arm
is the subject of its figures and the interval says what it allows (:func:`interval_words`), never its level, which
the table's header states. Each measure and guardrail is named as the page names it, and a cost is never the lead
(:func:`_is_cost`).

**No pass count where nothing graded can fail a trial.** An arm whose aspect declares no check and no judged dimension
that counts (:func:`_holds_a_check`, each reading by its declared kind, never the eval's kind nor the values a reading
took) states how many it ran and
that no grader says pass or fail: ``classify ran 3 of 3 cases; no grader here says pass or fail.`` Its readings carry
the means. Two arms under the case floor say the same, with no pass comparison: ``b and a ran 3 cases each; no grader
here says pass or fail; too few cases to call a difference.``
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
    JudgedMeasure,
)
from threetears.evals.analysis.bundle.trials import PassCount, pass_count
from threetears.evals.analysis.quantities import is_money
from threetears.evals.analysis.cells import split_arms, unread_arms
from threetears.evals.analysis.completeness import SETUP_GATE_SHARE, lost_share, setup_failed, unfinished_opening
from threetears.evals.analysis.floors import definitions_moved
from threetears.evals.analysis.stats import SMALL_N_BAND_FLOOR
from threetears.evals.analysis.style import cannot_tell_apart, measure_name
from threetears.evals.analysis.report.words import PASS_RATE, comparison_words, judged_heading
from threetears.evals.kernel.metrics import (
    ACCURACY_MEASURE,
    MATCH_MEASURE,
    METRIC_DESCRIPTORS,
    RUBRIC_FAMILY,
    reads_as_rate,
)
from threetears.evals.kernel.result_condition import CandidateFailureCause, ExclusionCause
from threetears.evals.schema.reading_states import Coordinate

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

#: Which rule decided the verdict line.
VerdictRule = Literal["setup", "unfinished", "one_arm", "too_small", "declared", "exploratory"]


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
    "cost_cap": CauseWords(
        words=(
            "the run's cost cap had too little left to judge it: it ran whole, and its other readings count it, but"
            " nothing scored it, so the judged readings leave it out; a trial a check had already failed stays"
            " failed, so the trials left can pass less often than all of them would have"
        ),
        remedy=(
            "re-judge each lost trial (result_regrade with apply), which reads the answer it stored and pays for no candidate"
            " call again; or give the next launch a higher cost cap (max_cost_usd), with room for the judging"
        ),
        short="cost cap",
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
    :return: The cap in words; never filled from what the run asked for.
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
    :return: The statuses, for every arm the realized design places.
    """
    design = bundle.design
    arms = [*([design.control_arm] if design.control_arm is not None else []), *design.contrasts]
    return {
        placed.variant_key: frozenset(bundle.incomplete_runs.get(run_id, "completed") for run_id in placed.run_ids)
        for placed in arms
    }


def unfinished_line(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> VerdictLine | None:
    """The line a page opens with in place of a verdict where a run of ``bundle`` did not finish, or None when every
    run completed (:func:`~threetears.evals.analysis.completeness.unfinished_opening`). Nothing in it is a decision, so
    nothing is set in bold.

    :param bundle: The campaign's evidence.
    :param arm: Names an arm by its variant key, as every table of the report does.
    :return: The line, its rule ``unfinished``, or None.
    """
    opening = unfinished_opening(bundle, arm)
    if opening is None:
        return None
    stopped, text = opening
    return VerdictLine(rule="unfinished", text=text, lead=stopped)


def measured_nothing(exclusions: Iterable[ArmExclusions], run_statuses: Iterable[str]) -> bool:
    """Whether an arm measured nothing and its run failed for it: every trial lost to ``infra``, every run ``failed``.

    Read off the run's stored status as well as the losses, so a run a cap, a cancel or the account stopped, and a run
    stored before a run that measured nothing failed, keep their own words.

    :param exclusions: The arm's cells' exclusions (:func:`arm_exclusions`).
    :param run_statuses: The arm's runs' statuses (:func:`arm_run_statuses`).
    :return: True when the arm ran trials, every one was excluded as ``infra``, and every run of it ended ``failed``.
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
    :param exclusions: The arm's cells' losses (:func:`arm_exclusions`), which carry the cap a ``truncated`` judge was
        sent.
    :param arms: How many arms the campaign ran, which the remedy names.
    :param run_statuses: The arm's runs' stored statuses (:func:`arm_run_statuses`); none known, the cause's own words.
    :return: The words. A cause this build has no words for is named as itself and says so, rather than taking another
        cause's remedy. ``infra`` on an arm whose run the account stopped takes :data:`ACCOUNT_REFUSED_REMEDY`, and on
        an arm that measured nothing (:func:`measured_nothing`) :data:`NOTHING_MEASURED_REMEDY`, never the rerun.
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
        verdicts, never off the words, so a page links that phrase's definition where the line says it. A led contrast
        the bet read is not one: it says where the arm stands and how strong the evidence is (``graded``).
    :ivar graded: Whether a decision ``text`` says leaves a contrast the bet read undecided, so its words say where the
        arm stands and grade the evidence, and a page links the evidence's definition.
    :ivar advice: What to do, said above ``text`` (the proof) on every first screen: the advice line over the led
        contrasts (:func:`~threetears.evals.analysis.report.aspect.contrasts.contrast_advice`); empty where there is
        none to say.
    """

    rule: VerdictRule
    text: str
    lead: str = ""
    decisions: tuple[str, ...] = ()
    undecided: bool = False
    graded: bool = False
    advice: str = ""

    @property
    def advised(self) -> str:
        """The first screen's opening: the advice, then the proof, as the page's verdict block and the print say them;
        ``text`` alone where there is no advice."""
        return f"{self.advice} {self.text}" if self.advice else self.text

    def __post_init__(self) -> None:
        if not self.lead:
            object.__setattr__(self, "lead", self.text)


def passed_clause(arm: str, count: PassCount) -> str:
    """One arm's pass count as every surface states it, the one owner of that clause: ``reply passed 18 of 20
    cases``, or ``reply passed 40 of 40 trials on 20 cases`` where cases ran more than once; where trials ran that
    nothing graded (a grader that raised, a declined call), the count is every trial that ran and says how many
    were not graded beside it: ``classify passed 7 of 10 cases; 3 were not graded``. The percent and its interval are
    the scorecard's, over the graded trials: a sentence carries at most three figures
    (:data:`~threetears.evals.analysis.style.MAX_NUMERALS`). Where the candidate failed every graded trial without
    answering (the runner raised on each), there is no pass to count and nothing a grader read: the clause says it
    failed on all of them and gives the first error, ``gloss failed on all 4 cases, so no answer was graded: the
    runner raised KeyError: 'word'``. The console's first line, a page's lead, an aspect fold's heading and a
    listing's line all say it through this.

    :param arm: The arm, as the page names it.
    :param count: Its count, read off the census
        (:func:`~threetears.evals.analysis.bundle.trials.pass_count`).
    :return: The clause, with no closing stop.
    """
    lost = count.ungraded
    if count.none_answered:
        # Every graded trial failed with no answer for a grader to read (the runner raised on each): no pass count
        # says that, and no statement about the graders either. How many it failed on, and the error, once.
        on = f"{count.graded_cases} case{'' if count.graded_cases == 1 else 's'}"
        if count.graded == count.graded_cases:
            every = f"all {on}" if count.graded_cases > 1 else f"its {on}"
        else:
            every = f"all {count.graded} trials on {on}"
        more = f" ({lost} more {'was' if lost == 1 else 'were'} not graded)" if lost else ""
        # The error in its own words, on the clause's one line (an error may run over several), the function under
        # test called the runner, as the package's docs teach it (`llms.txt`), where its words name it the candidate.
        said = _runner_words(" ".join(count.candidate_error.split())).rstrip(".") if count.candidate_error else ""
        error = f": {said}" if said else ""
        return f"{arm} failed on {every}{more}, so no answer was graded{error}"
    trials, cases = (count.ran, count.cases) if lost else (count.graded, count.graded_cases)
    said = (
        f"{arm} passed {count.passed} of {cases} cases"
        if trials == cases
        else f"{arm} passed {count.passed} of {trials} trials on {cases} cases"
    )
    if not lost:
        return said
    were = "was" if lost == 1 else "were"
    if count.ran == count.cases:
        return f"{said}; {lost} {were} not graded"
    return f"{said}; {lost} trial{'' if lost == 1 else 's'} {were} not graded"


#: How a recorded error names the function under test, and the word the package's docs teach for it.
_CANDIDATE_NAMED: Final = ("the candidate ", "the runner ")


def _runner_words(error: str) -> str:
    """A recorded error as a line this module writes says it: ``the candidate raised ...`` as ``the runner raised
    ...``, the word ``llms.txt`` teaches; any other error as recorded."""
    stored, taught = _CANDIDATE_NAMED
    return taught + error.removeprefix(stored) if error.startswith(stored) else error


#: Whose failure a recorded error was, by the role it was recorded under, as an ungraded trial's reason says it.
_FAILED_ROLE_WORDS: Final[dict[str, str]] = {
    "candidate": "the runner failed",
    "test setup": "the test setup failed",
    "judge": "the judge's call failed",
    "runner": "the engine failed",
}


def ungraded_sentence(count: PassCount, case_name: Callable[[str], str], *, arm: str | None = None) -> str:
    """`` Not graded: c0, c1, c2 (`the grader right raised TypeError: ...`).`` — after :func:`passed_clause`, the
    cases of the trials nothing graded, the first :data:`_MISSES_NAMED` by name and a count of the rest, and why, in
    the words the run recorded and whose failure it was: a grader's own error as it reads (``the grader right
    raised ...``), a judge's as ``the judge's call failed`` (its error names dimensions by stored ids no page prints),
    any other with its role (``the test setup failed: ...``), and where no trial's error is held, the census's cause in
    plain words. A sentence of its own so the count before it keeps its figures (``Not graded on
    b:`` where ``arm`` names the arm among several); empty where none was lost.

    :param count: The arm's count.
    :param case_name: Names a case by its id.
    :param arm: The arm's name, where the sentence stands among several arms'.
    :return: The sentence, with a leading space and its closing stop, or empty.
    """
    if not count.ungraded:
        return ""
    on = f" on {arm}" if arm is not None else ""
    if count.ungraded_role == "judge":
        # A judge's error names its dimensions by their stored ids, which no page prints: whose failure it was is said,
        # and the error itself stays on the trial's own record.
        why = _FAILED_ROLE_WORDS["judge"]
    elif count.ungraded_error is not None:
        grader_s = count.ungraded_role == "test setup" and count.ungraded_error.startswith("the grader ")
        whose = "" if grader_s else f"{_FAILED_ROLE_WORDS.get(count.ungraded_role or '', 'it failed')}: "
        why = f"{whose}`{count.ungraded_error}`"
    elif count.ungraded_causes:
        why = f"because {loss_words(count.ungraded_causes[0]).words}"
    else:
        return ""
    if not count.ungraded_cases:
        return f" Not graded{on}: {why}."
    cases = count.ungraded_cases
    shown = [case_name(case) or case for case in cases[:_MISSES_NAMED]]
    more = f", and {len(cases) - _MISSES_NAMED} more" if len(cases) > _MISSES_NAMED else ""
    return f" Not graded{on}: {', '.join(shown)}{more} ({why})."


def not_graded_words(count: PassCount) -> str:
    """``3 not graded``, or ``6 trials not graded`` where cases ran more than once: a cell's note beside a count over
    the graded trials, from the same count as :func:`passed_clause`; empty where every trial was graded.

    :param count: The arm's count.
    :return: The words, or empty.
    """
    if not count.ungraded:
        return ""
    if count.ran == count.cases:
        return f"{count.ungraded} not graded"
    return f"{count.ungraded} trial{'' if count.ungraded == 1 else 's'} not graded"


def ungraded_sentences(
    bundle: AnalysisContextBundle, arm: Callable[[str], str], case_name: Callable[[str], str]
) -> str:
    """For each arm of a comparison with trials nothing graded, that its figures are over its graded trials and how
    many were not (``a's figures are over its graded trials: 3 of 10 were not graded.``), then which and why
    (:func:`ungraded_sentence`); empty where every trial was graded. Every comparison's verdict line and a page's lead
    on a compared aspect say it through this.

    :param bundle: The aspect's evidence.
    :param arm: Names an arm by its variant key.
    :param case_name: Names a case by its id.
    :return: The sentences, each with a leading space and its closing stop, or empty.
    """
    said = ""
    for board in bundle.scoreboard:
        count = pass_count(bundle, board)
        if count.ungraded and count.graded:
            lost = count.ungraded
            said += (
                f" {arm(board.variant_key)}'s figures are over its graded trials: {lost} of {count.ran} "
                f"{'was' if lost == 1 else 'were'} not graded."
            )
            said += ungraded_sentence(count, case_name, arm=arm(board.variant_key))
    return said


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
    :param low: The interval's low end, on the change's ruler (points for a rate).
    :param high: Its high end.
    :param amount: A change's size as the sentence states it, unsigned.
    :param unit: What follows the sizes once, `` points`` for a rate; empty where ``amount`` carries the unit.
    :param higher_is_better: The reading's direction: a rise is better where True.
    :return: The sentence, with its closing stop.
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
    :param bundle: The aspect's evidence, whose catalog describes it.
    :return: True for a cost.
    """
    described = bundle.measure_catalog.get(measure) or METRIC_DESCRIPTORS.get(measure)
    return described is not None and (described.merit_axis == "cost" or is_money(described.unit or ""))


def _lead_measure(bundle: AnalysisContextBundle, tested: Iterable[str], order: Sequence[str]) -> str | None:
    """The measure a result leads on: the first ``order`` lists (an aspect design's ``measure_order``, which an eval
    files in the order it lists its graders) that a contrast tested and that is no cost; with no order listed, the
    classifier's accuracy where no question is asked of the aspect; None where neither names one.

    :param bundle: The aspect's evidence.
    :param tested: The measures a contrast tested, by key (never the pass rate, ``trial``).
    :param order: The graders' measures, in the order listed; empty where none is listed.
    :return: The measure's key, or None.
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
    :param verdicts: The aspect's typed verdicts, worded as its page words them.
    :param order: The graders' measures, in the order listed.
    :param asked: Whether a question is asked, so a contrast outside it is marked a lead.
    :param on: Where it was read, said after the measure's name (`` on billing``) on a page of several aspects.
    :param evidence: The figures each verdict rests on, as sentences (both arms' readings, the change and its
        interval, and for an undecided one the cases that would settle it), said after its decision; None says the
        decision alone.
    :param also: What else the evidence tells apart, said after the last arm's decision, before its figures.
    :return: One decision per arm, each with its closing stop and its figures after it.
    """
    led = led_verdicts(bundle, verdicts, order)
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


def graded_contrast(bundle: AnalysisContextBundle, verdict: Verdict) -> bool:
    """Whether the bet read the contrast ``verdict`` states, so its evidence is graded: read off the bundle's comparison,
    never off the verdict's words."""
    return any(
        comparison.evidence_better is not None
        for family in bundle.multiple_comparisons.families
        if family.aspect == verdict.aspect
        for comparison in family.comparisons
        if comparison.contrast.variant_key == verdict.variant_key
        and comparison.name == verdict.name
        and comparison.reading == verdict.reading
    )


def led_verdicts(bundle: AnalysisContextBundle, verdicts: Sequence[Verdict], order: Sequence[str]) -> list[Verdict]:
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
    advice: Callable[[Sequence[Verdict]], str | None] | None = None,
) -> VerdictLine:
    """The verdict line, by the first rule that fires (the module docstring's order).

    :param bundle: The campaign's evidence.
    :param arm: Names an arm by its variant key, as every table of the report does.
    :param case_name: Names a case by its id, as the cases table does; its id where None. One arm's line names the
        cases it missed by it.
    :param verdicts: The aspect's typed verdicts as its page words them, whose words a comparison's line prints, so
        it reads as every other surface does; none reads the bundle's contrasts in their verdict words alone.
    :param order: The graders' measures, in the order the eval lists them (``AspectDesign.measure_order``), which a
        comparison's line leads on the first of (:func:`_lead_measure`).
    :param evidence: The figures each lead verdict rests on, said after its decision (:func:`led_sentences`); None
        says the decision alone.
    :param advice: The advice over the lead verdicts, said above the line (:attr:`VerdictLine.advice`); None, or one
        that says none, leaves the line without.
    :return: The rule and its sentences.
    """
    boards = bundle.scoreboard
    unfinished = unfinished_line(bundle, arm)
    if not boards and unfinished is None:
        return VerdictLine(rule="exploratory", text="No trial ran, so there is nothing to decide.")
    # A failed setup is no verdict either, and names what lost the trials and the remedy, so it keeps its own line.
    if boards and (setup := _setup_gate(bundle, arm)) is not None:
        return VerdictLine(rule="setup", text=setup)
    if unfinished is not None:
        return unfinished
    if len(boards) == 1:
        (board,) = boards
        if unread_arms(bundle.cells, board.variant_key):
            # Its trials are two cells' (two answering models, or two definitions): no one pass rate is the arm's.
            result = f"{arm(board.variant_key)} has no one pass rate: {_split_why(bundle, board.variant_key)}."
            return VerdictLine(rule="one_arm", text=result, lead=result)
        result = f"{_one_arm(bundle, board, arm, graded=_holds_a_check(bundle))}."
        named = case_name or (lambda case: case)
        count = pass_count(bundle, board)
        # Where it failed every case without answering, the line has said so: naming each case again says nothing.
        missed = "" if count.none_answered else _missed_cases(bundle, named)
        return VerdictLine(rule="one_arm", text=result + ungraded_sentence(count, named) + missed, lead=result)
    declared_question = bundle.question_declared
    # Every figure below is over the graded trials: an arm that lost any says so, and which, beside them.
    ungraded = ungraded_sentences(bundle, arm, case_name or (lambda case: case))
    answered = [board for board in boards if board.usable]
    if answered and all(pass_count(bundle, board).none_answered for board in answered):
        # No arm gave an answer a grader read (each runner raised on every case): there is nothing to compare, and
        # each arm's own clause says what failed, as one arm's line does.
        lead = "" if declared_question else "Exploratory: "
        failed = " ".join(f"{passed_clause(arm(board.variant_key), pass_count(bundle, board))}." for board in answered)
        return VerdictLine(rule="too_small", text=lead + failed + ungraded)
    # An arm that failed every case without answering is compared with nothing (:func:`_not_read`): no verdict on it,
    # or on any arm against it where it is the control, is said; its own sentence says what failed.
    silent = _silent(bundle)
    if silent:
        control = _control_of(bundle)
        verdicts = [
            verdict
            for verdict in verdicts
            if verdict.variant_key not in silent and (control not in silent or verdict.kind == "bar")
        ]
    if any(board.usable_cases < MIN_CASES for board in boards):
        lead = "" if declared_question else "Exploratory: "
        return VerdictLine(rule="too_small", text=lead + _too_small(bundle, arm) + ungraded)
    rule: VerdictRule = "declared" if declared_question else "exploratory"
    prefix = "" if declared_question else "Exploratory: "
    if verdicts:
        if led := led_verdicts(bundle, verdicts, order):
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
                text=prefix + text + ungraded + _guardrail_sentences(verdicts) + _not_compared_sentences(bundle, arm),
                lead=result,
                decisions=decisions,
                undecided=any(
                    verdict.outcome == "not_separated" and not graded_contrast(bundle, verdict) for verdict in led
                ),
                graded=any(verdict.outcome == "not_separated" and graded_contrast(bundle, verdict) for verdict in led),
                advice=(advice(led) or "") if advice is not None else "",
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
    read = _read_against_control(bundle)
    not_compared = _not_compared_sentences(bundle, arm)
    # Whether the line says no reading tells any arm apart from the control (:func:`_undecided`).
    undecided = True
    if said and costs_only and read:
        # A cost is never the lead on its own: what the evidence leaves undecided first, then the cost. Only where
        # some arm is read against the control: an arm never compared with it is not undecided.
        result = f"{_undecided(bundle, arm)} on every reading but cost; " + "; ".join(said)
    elif said:
        result, undecided = "; ".join(said), False
    elif declared_question and not any(family.aspect is not None for family in bundle.multiple_comparisons.families):
        result, undecided = "The declared question is untested", False
    elif read:
        result = _undecided(bundle, arm) + (
            " on any reading the question asks" if declared_question else " on any reading"
        )
    else:
        # Every arm is apart from the control: nothing was compared, so nothing is undecided either, and the line is
        # why (an arm's name is never capitalised as a sentence's opening word would be).
        result, undecided, not_compared = not_compared.strip().removesuffix("."), False, ""
    if declared_question and read:
        result = result[:1].upper() + result[1:]
    # Every arm not read against the control is said to be, and why, whatever else the line says (its own text, where
    # nothing else was compared, already is that).
    guardrails = (
        _guardrail_sentences(verdicts) if verdicts else "".join(f" {line}" for line in guardrail_lines(bundle, arm))
    ) + not_compared
    return VerdictLine(
        rule=rule,
        text=f"{prefix}{result}.{ungraded}{guardrails}",
        lead=f"{result}.",
        decisions=(result,),
        undecided=undecided,
    )


def caveat_lines(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> list[str]:
    """What qualifies the verdict, said under the scoreboard in normal weight: never bold, never first.

    :param bundle: The campaign's evidence.
    :param arm: Names an arm by its variant key.
    :return: The lines: today, that an arm is under the case floor. A "too few" line says what is withheld and why, in
        the reader's terms, never a bare count: under the floor no interval is estimated, so a rate is shown as a
        count and no gap between the arms is sized.
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


def _control_of(bundle: AnalysisContextBundle) -> str:
    """The arm the others are read against: the control where it ran, else the scoreboard's first arm."""
    return _control_key(bundle) or bundle.scoreboard[0].variant_key


def _undecided(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> str:
    """That the evidence tells no arm apart from the control, the arms by name, the control last:
    ``the evidence cannot tell v2 and v1 apart`` (:func:`~threetears.evals.analysis.style.cannot_tell_apart`). Never
    "no difference", and never "nothing separated": an undecided contrast claims none. Only the arms read against the
    control are named (:func:`_read_against_control`), and the caller asks only where one is: an arm never compared with it
    is not undecided."""
    return cannot_tell_apart([arm(key) for key in _read_against_control(bundle)], arm(_control_of(bundle)))


def _not_read(bundle: AnalysisContextBundle) -> list[str]:
    """The arms not read against the control, in the scoreboard's order
    (:func:`~threetears.evals.analysis.cells.unread_arms`, the one question every arm-level reader asks): one of the
    two is split on a coordinate of its identity, or the two share no test setup; or one of the two failed every case
    without answering (:func:`_silent`), so there is nothing of it to compare."""
    control = _control_of(bundle)
    moved = definitions_moved(bundle)
    silent = _silent(bundle)
    return [
        board.variant_key
        for board in bundle.scoreboard
        if board.variant_key != control
        and (
            control in silent
            or board.variant_key in silent
            or unread_arms(bundle.cells, control, board.variant_key, definitions_moved=moved)
        )
    ]


def _silent(bundle: AnalysisContextBundle) -> set[str]:
    """The arms whose candidate failed every graded trial without answering (``PassCount.none_answered``): no grader
    read an answer of theirs, so nothing is compared with them."""
    return {board.variant_key for board in bundle.scoreboard if pass_count(bundle, board).none_answered}


def _read_against_control(bundle: AnalysisContextBundle) -> list[str]:
    """The arms read against the control, in the scoreboard's order: every other arm but those :func:`_not_read`."""
    control = _control_of(bundle)
    apart = set(_not_read(bundle))
    return [board.variant_key for board in bundle.scoreboard if board.variant_key not in apart | {control}]


#: Why an arm split on a coordinate of its identity has no one figure, by the coordinate: its trials are two cells'.
_SPLIT_WHY: Final[dict[Coordinate, str]] = {
    "served_model": (
        "its trials were answered by more than one model, or by one no response named, so they are read per model"
    ),
    "measure_definition": (
        "its trials were read under more than one definition of a host measure, or under none recorded, so they are "
        "read per definition"
    ),
}


def _split_why(bundle: AnalysisContextBundle, key: str) -> str:
    """Why the arm ``key``, split on a coordinate of its identity, has no one figure: the first coordinate it is
    split on, in the order :data:`~threetears.evals.schema.reading_states.Coordinate` lists them."""
    on = split_arms(bundle.cells).get(key, {})
    coordinate = next(coordinate for coordinate in get_args(Coordinate) if coordinate in on)
    return _SPLIT_WHY[coordinate]


def _not_compared(bundle: AnalysisContextBundle, arm: Callable[[str], str], key: str) -> str:
    """Why the arm ``key`` is not compared with the control, as a sentence: ``next is not compared with pinned: its
    trials were read under more than one definition of a host measure, or under none recorded, so they are read per
    definition.``

    :param bundle: The aspect's evidence.
    :param arm: Names an arm by its variant key.
    :param key: An arm not read against the control (:func:`~threetears.evals.analysis.cells.unread_arms`).
    :return: The sentence, with its closing stop.
    """
    control = _control_of(bundle)
    split = split_arms(bundle.cells)
    if control in split:
        why = f"{arm(control)}'s " + _split_why(bundle, control).removeprefix("its ")
    elif key in split:
        why = _split_why(bundle, key)
    else:
        why = (
            "the two were read under different definitions of a host measure, or under none recorded, so they share "
            "no test setup"
        )
    return f"{arm(key)} is not compared with {arm(control)}: {why}."


def _not_compared_sentences(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> str:
    """`` next is not compared with pinned: ...`` — one sentence per arm not read against the control
    (:func:`_not_compared`); empty where every arm is. An arm that failed every case without answering
    (:func:`_silent`) says that once, by name, and that nothing is compared with it, in place of a sentence for each
    arm it leaves uncompared."""
    silent = _silent(bundle)
    said = "".join(
        f" {passed_clause(arm(board.variant_key), pass_count(bundle, board))}. Nothing is compared with "
        f"{arm(board.variant_key)}."
        for board in bundle.scoreboard
        if board.variant_key in silent
    )
    if _control_of(bundle) in silent:
        return said
    return said + "".join(f" {_not_compared(bundle, arm, key)}" for key in _not_read(bundle) if key not in silent)


def _one_arm(bundle: AnalysisContextBundle, board: ArmScoreboard, arm: Callable[[str], str], *, graded: bool) -> str:
    """One arm's pass count over every trial that ran (:func:`passed_clause`); where nothing graded can fail a trial
    (``graded`` False), how many it ran and that no grader says pass or fail, with no pass count, which would count
    trials nothing judged: its readings carry what was measured."""
    if not board.usable:
        return f"{arm(board.variant_key)} ran {board.ran} trials, all lost to the test setup"
    count = pass_count(bundle, board)
    if count.none_answered:
        # Whatever its graders are, none read an answer: what failed is the runner, which the clause says.
        return passed_clause(arm(board.variant_key), count)
    if not graded:
        if board.ran == board.cases:
            ran = f"{arm(board.variant_key)} ran {board.usable} of {board.ran} cases"
        else:
            ran = f"{arm(board.variant_key)} ran {board.usable} of {board.ran} trials on {board.usable_cases} cases"
        return f"{ran}; {_no_check_words(bundle)}"
    return passed_clause(arm(board.variant_key), count)


def _holds_a_check(bundle: AnalysisContextBundle) -> bool:
    """Whether anything the aspect grades can fail a trial, each reading by its declared kind
    (:func:`_declared_kind`): a fact about what was declared, never about the values a reading took.

    The readings are those its design lists (``AspectDesign.measure_order``, its graders' measures) and those the
    catalog describes, with every judged dimension. A reading whose kind this evidence does not state is no check, and
    the line says so (:func:`_no_check_words`): it is never guessed to be one. Never by the eval's kind: a measure no
    trial can fail on (a number) decides no pass."""
    readings = [*_declared_readings(bundle), *bundle.measure_catalog]
    return any(_declared_kind(bundle, name) == "check" for name in readings) or any(
        _counts(measure) for measure in bundle.judged_measures
    )


def _declared_readings(bundle: AnalysisContextBundle) -> list[str]:
    """The readings the aspect's design lists, in order (``AspectDesign.measure_order``); none where it lists none."""
    design = bundle.declared.design if bundle.declared is not None else None
    return list(design.measure_order) if design is not None else []


def _counts(measure: JudgedMeasure) -> bool:
    """Whether a judged dimension can fail a trial: a rubric dimension on the capability axis."""
    return measure.family == RUBRIC_FAMILY and measure.axis == "capability"


def _declared_kind(bundle: AnalysisContextBundle, name: str) -> Literal["check", "number", "unstated"]:
    """A reading's kind, as its declaration states it: a check where it can fail a trial, a number where it cannot.

    A judged dimension by its axis; a goal check, the classifier's accuracy and its label match by what they are; any
    other measure by its declared description (the host's descriptor the catalog carries, else the engine's core one),
    whether it reads as a rate (:func:`~threetears.evals.kernel.metrics.reads_as_rate`). ``unstated`` where this
    evidence holds no description of it: a host measure no trial scored, whose descriptor the catalog does not carry.
    """
    judged = next((measure for measure in bundle.judged_measures if measure.name == name), None)
    if judged is not None:
        return "check" if _counts(judged) else "number"
    descriptor = bundle.measure_catalog.get(name) or METRIC_DESCRIPTORS.get(name)
    if name == MATCH_MEASURE or reads_as_rate("measure", name, descriptor=descriptor):
        return "check"
    return "number" if descriptor is not None else "unstated"


def _no_check_words(bundle: AnalysisContextBundle) -> str:
    """That no grader here says pass or fail, and, where a declared reading's kind is unstated (:func:`_declared_kind`:
    it scored on no trial), that it is read as no check: ``no grader here says pass or fail; slug scored on no trial,
    so it is read as no check``."""
    unstated = [name for name in _declared_readings(bundle) if _declared_kind(bundle, name) == "unstated"]
    if not unstated:
        return "no grader here says pass or fail"
    read = "it is" if len(unstated) == 1 else "they are"
    return f"no grader here says pass or fail; {', '.join(unstated)} scored on no trial, so {read} read as no check"


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
    """Which arm passed more of its usable trials than the control, in words, and that no difference is called; where
    nothing graded can fail a trial (:func:`_holds_a_check`), how many cases each arm ran and that no grader says pass
    or fail, with no pass comparison, which would compare trials nothing judged. An arm not read against the control
    (:func:`_not_read`) is left out: its own sentence says why."""
    control = _control_of(bundle)
    boards = {board.variant_key: board for board in bundle.scoreboard}
    reference = boards[control]
    apart = set(_not_read(bundle))
    if not _holds_a_check(bundle):
        ordered = [board for board in bundle.scoreboard if board.variant_key not in apart | {control}] + [reference]
        names = [arm(board.variant_key) for board in ordered]
        apart_said = _not_compared_sentences(bundle, arm)
        if len(ordered) == 1:
            return apart_said.strip()
        # Every case that ran is the count; the trials nothing graded are said apart (:func:`ungraded_sentences`).
        if len({board.cases for board in ordered}) == 1:
            ran = f"{', '.join(names[:-1])} and {names[-1]} ran {reference.cases} cases each"
        else:
            ran = ", ".join(f"{name} ran {board.cases} cases" for name, board in zip(names, ordered, strict=True))
        return f"{ran}; {_no_check_words(bundle)}; too few cases to call a difference." + apart_said

    def share(board: ArmScoreboard) -> float:
        return board.passed / board.usable if board.usable else 0.0

    said = []
    for board in bundle.scoreboard:
        if board.variant_key == control or board.variant_key in apart:
            continue
        compared = (
            "more often than"
            if share(board) > share(reference)
            else "less often than"
            if share(board) < share(reference)
            else "as often as"
        )
        said.append(f"{arm(board.variant_key)} passed {compared} {arm(control)}")
    apart_said = _not_compared_sentences(bundle, arm)
    if not said:
        return apart_said.strip()
    return "; ".join(said) + "; too few cases to call a difference." + apart_said


def guardrail_name(bundle: AnalysisContextBundle, name: str, reading: str) -> str:
    """A guardrail as every surface names it, as its verdict heads it: a judged dimension by
    :func:`~threetears.evals.analysis.report.words.judged_heading`, a goal check by its declared name else its
    expression, any other measure by its reader name.

    :param bundle: The aspect's evidence, whose catalog and check names name it.
    :param name: The guardrail's measure key or judged dimension.
    :param reading: ``judged`` for a judged dimension; anything else reads a measure.
    :return: The name.
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
    :param arm: Names an arm by its variant key.
    :return: The sentences, the breached first; none where every guardrail held.
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
    if not setup_failed(bundle):
        return None
    boards = [board for board in bundle.scoreboard if board.ran]
    over = [board for board in boards if lost_share(board) > SETUP_GATE_SHARE]
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
            if {comparison.contrast.variant_key, comparison.control.variant_key} & _silent(bundle):
                continue  # an arm that answered nothing is compared with nothing (:func:`_not_read`)
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
    "graded_contrast",
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
    "led_verdicts",
    "loss_words",
    "measured_nothing",
    "unfinished_line",
    "lost_share",
    "passed_clause",
    "not_graded_words",
    "ungraded_sentence",
    "ungraded_sentences",
    "verdict_line",
]
