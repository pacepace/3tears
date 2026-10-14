"""The report's first screen in words: the basis line and the verdict line, both written by code, never by a memo.

**The verdict is the first rule that fires**, in this order:

1. **The setup gate.** An arm lost more than a third of its trials to the test setup, or the arms' loss rates differ
   by more than a third of their trials: the run measured its test setup, not the models, so there is no verdict,
   and the line names the arm, how many it lost, the cause most of them share in words, and the remedy.
2. **Too small.** An arm has fewer than :data:`MIN_CASES` cases with a usable trial: the run can show failures, not
   differences, and no rate is estimated.
3. **A declared question.** One clause per question the campaign declared before its first run, in the contrasts'
   and guardrails' own verdict words.
4. **Exploratory.** ``Leads only:`` and the contrasts that separated, in the same words — or that none did.

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
from threetears.evals.kernel.declaration import Question, preregistered_questions
from threetears.evals.kernel.result_condition import CandidateFailureCause, ExclusionCause

#: The fewest independent cases with a usable trial an arm needs before a rate on it is estimated, charted or
#: compared: below it a t interval is too wide and unstable to read (:data:`~threetears.evals.analysis.stats.SMALL_N_BAND_FLOOR`),
#: so a rate is printed as a count ("1 of 2"), never as a percentage, a difference or an interval.
MIN_CASES: Final = SMALL_N_BAND_FLOOR

#: The share of an arm's trials whose loss to the test setup, alone or as the gap between two arms, means the run
#: measured its setup rather than the models. A stand-in until a test across arms decides it.
SETUP_GATE_SHARE: Final = 1 / 3

#: Which rule decided the verdict line.
VerdictRule = Literal["setup", "too_small", "declared", "exploratory"]


@dataclass(frozen=True)
class CauseWords:
    """One cause in words: what happened, what a reader does about it, and the two or three words a table uses."""

    words: str
    remedy: str
    short: str = ""


#: Why a trial was lost to the test setup, in words, by :data:`~threetears.evals.kernel.result_condition.ExclusionCause`.
#: ``truncated``'s words and remedy name the cap the judge was sent, filled in by :func:`loss_words`.
LOSS_CAUSE_WORDS: Final[dict[str, CauseWords]] = {
    "infra": CauseWords(
        words="the test setup faulted (a simulator, a tool, the deadline or the judge's call failed)",
        remedy="read the lost trials' errors below, fix that part of the setup, then rerun every arm",
        short="test setup fault",
    ),
    "judge_cannot_tell": CauseWords(
        words="the judge said the transcript did not let it score a dimension the trial needs",
        remedy="give the judge the evidence it asked for, or narrow the dimension, then rerun every arm",
        short="judge could not tell",
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
        words="a judged dimension scored under the bar",
        remedy="read the judge's reasons below",
        short="judged under the bar",
    ),
}


def _covers(words: dict[str, CauseWords], values: Sequence[str], what: str) -> None:
    """Refuse, at import, a word table that does not cover exactly the causes it words.

    Raises:
        RuntimeError: The table and the vocabulary disagree.
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
    """The output cap a judge was sent on an arm's truncated trials, as ``max_tokens=N`` — or that it went unreported.

    Args:
        exclusions: The arm's cells' exclusions, whose ``truncated_sent_caps`` and ``truncated_cap_unreported`` say
            what the cut-off judge calls were sent, as their completions reported it.

    Returns:
        The cap in words; never filled from what the run asked for.
    """
    cells = list(exclusions)
    known = [str(cap) for cap in sorted({cap for cell in cells for cap in cell.truncated_sent_caps})]
    unreported = any(cell.truncated_cap_unreported for cell in cells)
    if not known:
        return "max_tokens not reported by the judge's client"
    said = f"max_tokens={' or '.join(known)}"
    return said + (", and unreported on some" if unreported else "")


def loss_words(cause: str, *, exclusions: Iterable[ArmExclusions] = (), arms: int = 2) -> CauseWords:
    """Why trials were lost to ``cause``, in words, with its remedy.

    Args:
        cause: The loss cause.
        exclusions: The arm's cells' losses (:func:`arm_exclusions`), which carry the cap a ``truncated`` judge was
            sent.
        arms: How many arms the campaign ran, which the remedy names.

    Returns:
        The words. A cause this build has no words for is named as itself and says so, rather than taking another
        cause's remedy.
    """
    worded = LOSS_CAUSE_WORDS.get(cause)
    if worded is None:
        return CauseWords(
            words=f"{cause} (a cause this build has no words for)", remedy="read the lost trials' errors below"
        )
    cap = cap_words(exclusions)
    sent = f"sent {cap.removeprefix('max_tokens=')}" if cap.startswith("max_tokens=") else cap
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


def basis_line(bundle: AnalysisContextBundle) -> str:
    """Line 2 of every report: whether a question was declared before the run, or that the campaign is exploratory.

    Args:
        bundle: The campaign's evidence.

    Returns:
        The line.
    """
    # The day the design froze (its first run's), which the methods state to the second.
    frozen = f" · design frozen {bundle.design_frozen_at[:10]}" if bundle.design_frozen_at else ""
    if bundle.question_declared:
        asked = "; ".join(f'"{question.text}"' for question in _declared_questions(bundle))
        return f"Question declared before the run: {asked}{frozen}"
    if post_hoc := bundle.post_hoc_questions:
        return (
            f"EXPLORATORY — question {', '.join(post_hoc)} was declared after the run began, so "
            f"nothing below is a decision.{frozen}"
        )
    return f"EXPLORATORY — no question was declared before the run, so nothing below is a decision.{frozen}"


def _declared_questions(bundle: AnalysisContextBundle) -> list[Question]:
    """The live questions the campaign declared before its first run, in declaration order.

    Read by :func:`~threetears.evals.kernel.declaration.preregistered_questions` against the instant the design froze
    (``design_frozen_at``, the first run's), the one rule the families, the reading scope and the bundle's
    ``post_hoc_questions`` were read by, so the verdict cannot count a question they treat as post hoc.
    """
    design = bundle.declared_design
    if design is None:
        return []
    return preregistered_questions(design, bundle.design_frozen_at)


def lost_share(board: ArmScoreboard) -> float:
    """The share of an arm's trials lost to the test setup; 0 for an arm that ran none."""
    return sum(board.lost.values()) / board.ran if board.ran else 0.0


def verdict_line(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> VerdictLine:
    """The verdict line, by the first rule that fires (the module docstring's order).

    Args:
        bundle: The campaign's evidence.
        arm: Names an arm by its variant key, as every table of the report does.

    Returns:
        The rule and its sentences.
    """
    boards = bundle.scoreboard
    if (setup := _setup_gate(bundle, arm)) is not None:
        return VerdictLine(rule="setup", text=setup)
    if small := [board for board in boards if board.usable_cases < MIN_CASES]:
        named = ", ".join(
            f"{arm(board.variant_key)} has {board.usable_cases} usable case{'' if board.usable_cases == 1 else 's'}"
            for board in small
        )
        return VerdictLine(
            rule="too_small",
            text=(
                f"Too few cases to estimate a rate: this run can show failures, not differences ({named}; a rate "
                f"needs {MIN_CASES})."
            ),
        )
    if bundle.question_declared and (declared := _declared(bundle, arm)):
        return VerdictLine(rule="declared", text=declared)
    leads = _separated(bundle, arm, question_id=None, every_family=True)
    return VerdictLine(
        rule="exploratory",
        text="Leads only: " + ("; ".join(leads) if leads else "no arm separated from the control") + ".",
    )


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
        parts.append(f"{arm(board.variant_key)} lost {lost} of {board.ran} trials, {most} because {said.words}")
        if said.remedy not in remedies:
            remedies.append(said.remedy)
    gap_words = "" if over else " The arms lost different shares of their trials, so they were not measured alike."
    return (
        f"NO VERDICT. The test setup failed, not the models: {'; '.join(parts)}.{gap_words} "
        f"Fix: {'; '.join(remedies)}."
    )


def _declared(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> str | None:
    """One clause per declared question: what its contrasts and guardrails came to, in their own words."""
    clauses = []
    for question in _declared_questions(bundle):
        separated = _separated(bundle, arm, question_id=question.id, every_family=False)
        tested = any(family.question_id == question.id for family in bundle.multiple_comparisons.families)
        if separated:
            clauses.append(f'"{question.text}": ' + "; ".join(separated))
        elif tested:
            clauses.append(f'"{question.text}": no arm separated from the control')
        else:
            clauses.append(f'"{question.text}": not tested')
    breached = sorted(
        {
            f"{arm(check.contrast.variant_key)} breached the guardrail on {check.name}"
            for check in bundle.guardrails.checks
            if check.decision == "breached"
        }
    )
    if not clauses:
        return None
    return ". ".join(clauses + breached) + "."


#: A separated contrast's verdict, as the verdict line words it.
_SEPARATED_WORDS: dict[str, str] = {"improved": "improved on", "regressed": "regressed on"}


def _separated(
    bundle: AnalysisContextBundle, arm: Callable[[str], str], *, question_id: str | None, every_family: bool
) -> list[str]:
    """Each contrast that separated from the control, in words: ``<arm> improved on <reading>``."""
    said = []
    for family in bundle.multiple_comparisons.families:
        if not every_family and family.question_id != question_id:
            continue
        for comparison in family.comparisons:
            if comparison.verdict in _SEPARATED_WORDS:
                heading = bundle.measure_catalog.get(comparison.name)
                reading = (heading.reader_name if heading is not None else None) or comparison.name
                if comparison.reading == "judged":
                    reading = f"{comparison.name} (judged)"
                line = f"{arm(comparison.contrast.variant_key)} {_SEPARATED_WORDS[comparison.verdict]} {reading}"
                if line not in said:
                    said.append(line)
    return said


__all__ = [
    "FAIL_CAUSE_WORDS",
    "LOSS_CAUSE_WORDS",
    "MIN_CASES",
    "SETUP_GATE_SHARE",
    "CauseWords",
    "VerdictLine",
    "VerdictRule",
    "arm_exclusions",
    "basis_line",
    "cap_words",
    "cause_short",
    "fail_words",
    "loss_words",
    "lost_share",
    "verdict_line",
]
