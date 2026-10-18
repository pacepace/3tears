"""The report's first screen in words: the arms line and the verdict line, both written by code, never by a memo.

**The verdict is the first rule that fires**, in this order, and it leads with the result, never a method caveat:

1. **The setup gate.** An arm lost more than a third of its trials to the test setup, or the arms' loss rates differ
   by more than a third of their trials: the run measured its test setup, not the models, so there is no verdict,
   and the line names the arm, how many it lost, the cause most of them share in words, and the remedy.
2. **One arm.** ``<eval>: n/N cases passed; k misses: …`` — how its cases came out.
3. **Too small.** An arm has fewer than :data:`MIN_CASES` usable cases: which arm passed more trials, and that no
   difference is called; the case floor itself is a caveat line under the scoreboard (:func:`caveat_lines`).
4. **A declared question.** What its contrasts and guardrails came to, in their own verdict words.
5. **Exploratory.** ``Exploratory:`` and the contrasts that separated — or ``nothing separated v2 from v1 on any
   reading``, never "no difference". A guardrail an arm is not shown safe on is a second sentence: ``Not yet shown
   safe: v2 on no invented total.``

**No floor decides the verdict.** A floor only warns: a broken one is the page's first line, above this one
(:mod:`~threetears.evals.analysis.report.decision`), and the verdict says what the contrasts came to beside it.

**A cause cannot exist without its words.** Every way a trial is lost (:data:`LOSS_CAUSE_WORDS`) and every way one
fails (:data:`FAIL_CAUSE_WORDS`) is worded here, with the remedy a reader acts on, and each table is checked against
the vocabulary it covers when this module is imported: a cause added to the kernel without words fails the import
rather than reaching a page as its raw token.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Literal, get_args

from threetears.evals.analysis.bundle.schema import (
    AnalysisContextBundle,
    ArmExclusions,
    ArmScoreboard,
    FailCause,
)
from threetears.evals.analysis.stats import SMALL_N_BAND_FLOOR
from threetears.evals.analysis.style import humanize, measure_name
from threetears.evals.analysis.report.words import LOWER_IS_BETTER, comparison_words
from threetears.evals.kernel.result_condition import CandidateFailureCause, ExclusionCause

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
        words="the test setup faulted (a simulator, a tool, the deadline or the judge's call failed)",
        remedy="read the lost trials' errors below, fix that part of the setup, then rerun every arm",
        short="setup fault",
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


def loss_words(cause: str, *, exclusions: Iterable[ArmExclusions] = (), arms: int = 2) -> CauseWords:
    """Why trials were lost to ``cause``, in words, with its remedy.

    :param cause: The loss cause.
    :ptype cause: str
    :param exclusions: The arm's cells' losses (:func:`arm_exclusions`), which carry the cap a ``truncated`` judge was
        sent.
    :ptype exclusions: Iterable[ArmExclusions]
    :param arms: How many arms the campaign ran, which the remedy names.
    :ptype arms: int
    :return: The words. A cause this build has no words for is named as itself and says so, rather than taking another
        cause's remedy.
    :rtype: CauseWords
    """
    worded = LOSS_CAUSE_WORDS.get(cause)
    if worded is None:
        return CauseWords(
            words=f"{cause} (a cause this build has no words for)", remedy="read the lost trials' errors below"
        )
    cap = cap_words(exclusions)
    sent = f"sent {cap.split(' tokens', 1)[0]}" if " tokens" in cap and cap[0].isdigit() else cap
    every = "both arms" if arms == 2 else "every arm"
    return CauseWords(
        words=worded.words.format(cap=cap), remedy=worded.remedy.format(sent=sent, every=every), short=worded.short
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
    """The verdict as the first screen prints it: which rule fired, and the sentences it wrote."""

    rule: VerdictRule
    text: str


def lost_share(board: ArmScoreboard) -> float:
    """The share of an arm's trials lost to the test setup; 0 for an arm that ran none."""
    return sum(board.lost.values()) / board.ran if board.ran else 0.0


def verdict_line(
    bundle: AnalysisContextBundle,
    arm: Callable[[str], str],
    *,
    name: str = "",
    case_name: Callable[[str], str] | None = None,
    model: Callable[[str], str] | None = None,
) -> VerdictLine:
    """The verdict line, by the first rule that fires (the module docstring's order).

    :param bundle: The campaign's evidence.
    :ptype bundle: AnalysisContextBundle
    :param arm: Names an arm by its variant key, as every table of the report does.
    :ptype arm: Callable[[str], str]
    :param name: The eval's name, which a one-arm line leads with.
    :ptype name: str
    :param case_name: Names a case by its id, as the cases table does; its id where None.
    :ptype case_name: Callable[[str], str] | None
    :param model: Names an arm by its model, where it has one; ``arm`` where None. Read by no rule today.
    :ptype model: Callable[[str], str] | None
    :return: The rule and its sentences.
    :rtype: VerdictLine
    """
    boards = bundle.scoreboard
    if not boards:
        return VerdictLine(rule="exploratory", text="No trial ran, so there is nothing to decide.")
    if (setup := _setup_gate(bundle, arm)) is not None:
        return VerdictLine(rule="setup", text=setup)
    if len(boards) == 1:
        return VerdictLine(rule="one_arm", text=_one_arm(bundle, name, case_name or (lambda case: case)))
    declared_question = bundle.question_declared
    if any(board.usable_cases < MIN_CASES for board in boards):
        lead = "" if declared_question else "Exploratory: "
        return VerdictLine(rule="too_small", text=lead + _too_small(bundle, arm))
    if declared_question and (declared := _declared(bundle, arm)):
        return VerdictLine(rule="declared", text=declared)
    leads = _separated(bundle, arm, asked=False)
    return VerdictLine(
        rule="exploratory",
        text="Exploratory: "
        + ("; ".join(leads) if leads else f"{_nothing_separated(bundle, arm)} on any reading")
        + "."
        + _guardrail_words(bundle, arm),
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


def _nothing_separated(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> str:
    """That no contrast separated, the arms by name, the control last: ``nothing separated v2 from v1``,
    ``nothing separated v2 or v3 from v1``. Never "no difference": a contrast not separated claims none."""
    key = _control_key(bundle) or bundle.scoreboard[0].variant_key
    others = [arm(board.variant_key) for board in bundle.scoreboard if board.variant_key != key]
    named = " or ".join(others) if len(others) <= 2 else f"{', '.join(others[:-1])} or {others[-1]}"
    return f"nothing separated {named} from {arm(key)}"


def _one_arm(bundle: AnalysisContextBundle, name: str, case_name: Callable[[str], str]) -> str:
    """``<eval>: n/N cases passed; k misses: …`` over the bundle's trials, case by case."""
    by_case: dict[str, list[str]] = {}
    for trial in bundle.trials:
        by_case.setdefault(trial.case_id, []).append(trial.state)
    usable = [case for case, states in by_case.items() if any(state != "lost" for state in states)]
    missed = [case for case in usable if "failed" in by_case[case]]
    text = f"{len(usable) - len(missed)}/{len(usable)} cases passed; {len(missed)} {'miss' if len(missed) == 1 else 'misses'}"
    if missed:
        shown = [case_name(case) for case in missed[:5]]
        text += ": " + ", ".join(shown) + (f", and {len(missed) - 5} more" if len(missed) > 5 else "")
    return f"{name}: {text}" if name else text


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


#: How the verdict leads a guardrail an arm is not shown safe on, by the guardrail's decision, the breached first.
_GUARDRAIL_LEADS: dict[str, str] = {"breached": "Guardrail breached", "undecided": "Not yet shown safe"}


def _guardrail_words(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> str:
    """`` Not yet shown safe: v2 on no invented total.`` — one sentence per decision naming each arm and its
    guardrails; empty when every guardrail held."""
    said = []
    for decision, lead in _GUARDRAIL_LEADS.items():
        by_arm: dict[str, list[str]] = {}
        for check in bundle.guardrails.checks:
            if check.decision == decision:
                names = by_arm.setdefault(arm(check.contrast.variant_key), [])
                if humanize("dimension", check.name) not in names:
                    names.append(humanize("dimension", check.name))
        if by_arm:
            arms = "; ".join(f"{name} on {', '.join(names)}" for name, names in sorted(by_arm.items()))
            said.append(f" {lead}: {arms}.")
    return "".join(said)


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
    parts = []
    remedies: list[str] = []
    for board in named:
        lost = sum(board.lost.values())
        cause = max(sorted(board.lost), key=lambda name: board.lost[name])
        said = loss_words(cause, exclusions=exclusions.get(board.variant_key, ()), arms=len(boards))
        most = "all" if board.lost[cause] == lost else "most"
        parts.append(f"{arm(board.variant_key)} lost {lost} of {board.ran} trials, {most} because {said.words}.")
        if said.remedy not in remedies:
            remedies.append(said.remedy)
    gap_words = "" if over else " The arms lost different shares of their trials, so they were not measured alike."
    return (
        f"No verdict: the test setup failed, not the models. {' '.join(parts)}{gap_words} Fix: {'; '.join(remedies)}."
    )


def _declared(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> str | None:
    """What the declared question's contrasts and guardrails came to, in their own words."""
    if not bundle.question_declared:
        return None
    # The first screen quotes the question's words; the verdict names it, never repeats it.
    separated = _separated(bundle, arm, asked=True)
    tested = any(family.aspect is not None for family in bundle.multiple_comparisons.families)
    if separated:
        said = "Declared question: " + "; ".join(separated)
    elif tested:
        said = f"Declared question: {_nothing_separated(bundle, arm)}"
    else:
        said = "Declared question: not tested"
    return said + "." + _guardrail_words(bundle, arm)


#: The contrast verdicts that separated, which the verdict line names.
_SEPARATED = ("improved", "regressed")


def _separated(bundle: AnalysisContextBundle, arm: Callable[[str], str], *, asked: bool) -> list[str]:
    """Each contrast that separated from the control, in words, both arms by label:
    ``v2 better than v1 on <reading>``, a lower-is-better reading's better end in brackets after it. ``asked`` reads
    only the question's family; otherwise every family."""
    said = []
    for family in bundle.multiple_comparisons.families:
        if asked and family.aspect is None:
            continue
        for comparison in family.comparisons:
            if comparison.verdict in _SEPARATED:
                heading = bundle.measure_catalog.get(comparison.name)
                reading = measure_name(comparison.name, heading.reader_name if heading is not None else None)
                if comparison.reading == "judged":
                    reading = f"{humanize('dimension', comparison.name)} (judged)"
                elif comparison.reading == "trial":
                    reading = "the pass rate"
                words = comparison_words(
                    comparison.verdict, arm(comparison.contrast.variant_key), arm(comparison.control.variant_key)
                )
                line = f"{words} on {reading}{'' if comparison.higher_is_better else LOWER_IS_BETTER}"
                if line not in said:
                    said.append(line)
    return said


__all__ = [
    "FAIL_CAUSE_WORDS",
    "LOSS_CAUSE_WORDS",
    "MIN_CASES",
    "SETUP_GATE_SHARE",
    "CauseWords",
    "caveat_lines",
    "VerdictLine",
    "VerdictRule",
    "arm_exclusions",
    "cap_words",
    "cause_short",
    "fail_words",
    "loss_words",
    "lost_share",
    "verdict_line",
]
