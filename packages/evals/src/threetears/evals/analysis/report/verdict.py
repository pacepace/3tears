"""The report's first screen in words: the arms line and the verdict line, both written by code, never by a memo.

**The verdict is the first rule that fires**, in this order, and it leads with the result, never a method caveat:

1. **The setup gate.** An arm lost more than a third of its trials to the test setup, or the arms' loss rates differ
   by more than a third of their trials: the run measured its test setup, not the models, so there is no verdict,
   and the line names the arm, how many it lost, the cause most of them share in words, and the remedy.
2. **One arm.** ``<eval>: n/N cases passed; k misses: …`` — how its cases came out.
3. **A declared decision rule.** Each clause settled or not (:mod:`~threetears.evals.analysis.report.decision`),
   and the rule's outcome in plain words: keep, switch, or unsettled with the fix and its cost. A keep names the
   counts that settled it, the arms by their models where they have one: ``Keep production: v4-flash passed 2 of 12
   trials to v4.1-flash's 10, and cost more per reply.`` The line is one sentence held to the page's numeral limit;
   the decision table directly below carries every figure and every clause's reason.
4. **Too small.** An arm has fewer than :data:`MIN_CASES` usable cases: which arm passed more trials, and that no
   difference is called; the case floor itself is a caveat line under the scoreboard (:func:`caveat_lines`).
5. **A declared question.** What its contrasts and guardrails came to, in their own verdict words.
6. **Exploratory.** ``Exploratory:`` and the contrasts that separated — or that no difference was found on any
   reading. A guardrail an arm is not shown safe on is a second sentence: ``Not yet shown safe: v2 on no invented
   total.``

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
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.report.decision import (
    RateReading,
    RuleVerdict,
    evaluate_rule,
    parse_decision_rule,
    rule_readings,
)
from threetears.evals.analysis.stats import SMALL_N_BAND_FLOOR
from threetears.evals.analysis.style import humanize, lint_prose, measure_name
from threetears.evals.kernel.declaration import Question, preregistered_questions
from threetears.evals.kernel.result_condition import CandidateFailureCause, ExclusionCause

#: The fewest independent cases with a usable trial an arm needs before a rate on it is estimated, charted or
#: compared: below it a t interval is too wide and unstable to read (:data:`~threetears.evals.analysis.stats.SMALL_N_BAND_FLOOR`),
#: so a rate is printed as a count ("1 of 2"), never as a percentage, a difference or an interval.
#:
#: The floor counts independent cases, never trials or repeats. Below it no contrast is tested, the verdict says too
#: few, and no rate (a pass rate, pass^k) is estimated for an arm; a measure's mean is still shown with its interval,
#: and a declared rule can still settle keep on what was observed, never switch. So a host's seed gate covers every
#: template whose results are compared across arms, whatever its declared reading.
MIN_CASES: Final = SMALL_N_BAND_FLOOR

#: The share of an arm's trials whose loss to the test setup, alone or as the gap between two arms, means the run
#: measured its setup rather than the models. A stand-in until a test across arms decides it.
SETUP_GATE_SHARE: Final = 1 / 3

#: Which rule decided the verdict line.
VerdictRule = Literal["setup", "one_arm", "rule", "too_small", "declared", "exploratory"]


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
        words="a judged dimension scored under the bar",
        remedy="read the judge's reasons below",
        short="below the bar",
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
    """The output cap a judge was sent on an arm's truncated trials, as ``N tokens`` — or that it went unreported.

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
        return "its max_tokens not reported by the judge's client"
    said = f"{' or '.join(known)} tokens"
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


def verdict_line(
    bundle: AnalysisContextBundle,
    arm: Callable[[str], str],
    *,
    name: str = "",
    case_name: Callable[[str], str] | None = None,
    model: Callable[[str], str] | None = None,
) -> VerdictLine:
    """The verdict line, by the first rule that fires (the module docstring's order).

    Args:
        bundle: The campaign's evidence.
        arm: Names an arm by its variant key, as every table of the report does.
        name: The eval's name, which a one-arm line leads with.
        case_name: Names a case by its id, as the cases table does; its id where None.
        model: Names an arm by its model, where it has one, for the counts a settled rule states; ``arm`` where None.

    Returns:
        The rule and its sentences.
    """
    boards = bundle.scoreboard
    if not boards:
        return VerdictLine(rule="exploratory", text="No trial ran, so there is nothing to decide.")
    if (setup := _setup_gate(bundle, arm)) is not None:
        return VerdictLine(rule="setup", text=setup)
    if len(boards) == 1:
        return VerdictLine(rule="one_arm", text=_one_arm(bundle, name, case_name or (lambda case: case)))
    if (ruled := _rule_line(bundle, arm, model or arm)) is not None:
        return VerdictLine(rule="rule", text=ruled)
    if any(board.usable_cases < MIN_CASES for board in boards):
        lead = "" if bundle.question_declared else "Exploratory: "
        return VerdictLine(rule="too_small", text=lead + _too_small(bundle, arm))
    if bundle.question_declared and (declared := _declared(bundle, arm)):
        return VerdictLine(rule="declared", text=declared)
    leads = _separated(bundle, arm, question_id=None, every_family=True)
    return VerdictLine(
        rule="exploratory",
        text="Exploratory: "
        + ("; ".join(leads) if leads else f"no difference found between {_every_arm(bundle, arm)} on any reading")
        + "."
        + _guardrail_words(bundle, arm),
    )


def caveat_lines(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> list[str]:
    """What qualifies the verdict, said under the scoreboard in normal weight: never bold, never first.

    Args:
        bundle: The campaign's evidence.
        arm: Names an arm by its variant key.

    Returns:
        The lines: today, that an arm is under the case floor. A "too few" line says what is withheld and why, in
        the reader's terms, never a bare count: under the floor no interval is estimated, so a rate is shown as a
        count and no gap between the arms is sized. Where a declared rule settled keep on every candidate, the line
        says keep follows from the counts and only how much worse the candidate is goes unestimated.
    """
    small = [board for board in bundle.scoreboard if board.usable_cases < MIN_CASES]
    if not small:
        return []
    verdicts = _rule_verdicts(bundle)
    if verdicts and all(verdict.outcome == "keep" for verdict in verdicts):
        counts = {board.usable_cases for board in bundle.scoreboard}
        if len(counts) == 1 and len(small) == len(bundle.scoreboard):
            cases = f"{small[0].usable_cases} cases per arm"
        else:
            cases = "Usable cases " + ", ".join(f"{arm(board.variant_key)} {board.usable_cases}" for board in small)
        candidate = "the candidate" if len(verdicts) == 1 else "each candidate"
        return [
            f"{cases}: keep follows from the counts above. How much worse {candidate} is goes unestimated: a gap is "
            f"estimated only from {MIN_CASES} independent cases."
        ]
    named = ", ".join(f"{arm(board.variant_key)} {board.usable_cases}" for board in small)
    return [
        f"Usable cases: {named}. Under {MIN_CASES} independent cases a rate is shown as a count, and how far apart "
        "the arms are is not estimated."
    ]


def _rule_verdicts(bundle: AnalysisContextBundle) -> list[RuleVerdict]:
    """The declared rule read on each candidate against the control; empty when no live question states one."""
    control = _control_key(bundle)
    if not bundle.question_declared or control is None:
        return []
    rules = [
        (question.id, rule) for question in _declared_questions(bundle) if (rule := parse_decision_rule(question.text))
    ]
    if not rules:
        return []
    asked, rule = rules[0]
    return [
        evaluate_rule(rule, rule_readings(bundle, control, board.variant_key, question_id=asked))
        for board in bundle.scoreboard
        if board.variant_key != control
    ]


def _control_key(bundle: AnalysisContextBundle) -> str | None:
    """The control's variant key, when it ran."""
    control = bundle.design.control_arm.variant_key if bundle.design.control_arm is not None else None
    return control if any(board.variant_key == control for board in bundle.scoreboard) else None


def _every_arm(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> str:
    """Every arm by name, the control last: ``v2 and v1``, ``v2, v3 and v1``."""
    key = _control_key(bundle) or bundle.scoreboard[0].variant_key
    names = [arm(board.variant_key) for board in bundle.scoreboard if board.variant_key != key] + [arm(key)]
    return " and ".join(names) if len(names) <= 2 else f"{', '.join(names[:-1])} and {names[-1]}"


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


def _rule_line(bundle: AnalysisContextBundle, arm: Callable[[str], str], model: Callable[[str], str]) -> str | None:
    """The declared decision rule's outcome in plain words, or None when no live question states one."""
    control = _control_key(bundle)
    verdicts = _rule_verdicts(bundle)
    if control is None or not verdicts:
        return None
    candidates = [board.variant_key for board in bundle.scoreboard if board.variant_key != control]
    return " ".join(
        rule_sentence(
            verdict,
            control=arm(control),
            candidate=arm(candidate),
            bundle=bundle,
            control_model=model(control),
            candidate_model=model(candidate),
        )
        for verdict, candidate in zip(verdicts, candidates, strict=True)
    )


def rule_sentence(
    verdict: RuleVerdict,
    *,
    control: str,
    candidate: str,
    bundle: AnalysisContextBundle,
    control_model: str | None = None,
    candidate_model: str | None = None,
) -> str:
    """A rule's outcome as the verdict line says it: the decision, then what settled it or the fix with its cost.

    A keep states the counts that settled it (``v4-flash passed 2 of 12 trials to v4.1-flash's 10, and cost more per
    reply``). Each sentence keeps its clauses only while it stays within the page's numeral and length limits
    (:func:`~threetears.evals.analysis.style.lint_prose`): the decision table below carries every clause.

    Args:
        verdict: The rule read on one candidate.
        control: The control, by name.
        candidate: The candidate, by name.
        bundle: The campaign's evidence, for the cost of more cases.
        control_model: The control by its model, which a keep's counts name it by; ``control`` where None.
        candidate_model: The candidate by its model, likewise; ``candidate`` where None.

    Returns:
        The sentence.
    """

    if verdict.outcome == "keep":
        said = _kept(verdict, control_model or control, candidate_model or candidate)
        return _within_limits(f"Keep {control}: ", said)
    if verdict.outcome == "switch":
        return f"Switch to {candidate}: every clause of the decision rule holds."
    reasons = [
        clause.reason.replace("the candidate", candidate) for clause in verdict.clauses if clause.outcome == "unsettled"
    ]
    cost = _cost_per_case(bundle)
    fix = f" Each more case costs about {cost} over every arm." if cost else ""
    return _within_limits(f"Unsettled between {control} and {candidate}: ", reasons) + fix


def _within_limits(lead: str, parts: list[str]) -> str:
    """``lead`` and as many of ``parts`` as one sentence holds within the page's limits, joined by ", and ".

    The first part is always said; each later one only while the sentence stays within the numeral and length
    limits, the decision table carrying the rest.
    """
    kept = parts[:1]
    for part in parts[1:]:
        if lint_prose(f"{lead}{', and '.join([*kept, part])}."):
            break
        kept.append(part)
    return f"{lead}{', and '.join(kept)}."


def _kept(verdict: RuleVerdict, control: str, candidate: str) -> list[str]:
    """What settled a keep, clause by clause, as counts with the candidate as their one subject."""
    said: list[str] = []
    readings = verdict.readings
    for clause in verdict.clauses:
        if clause.outcome != "keep":
            continue
        subject = "" if said else f"{candidate} "
        if clause.rate is not None:
            said.append(subject + _counts(clause.rate, control))
        elif clause.clause.kind == "cost_lower" and readings is not None:
            ours, theirs = readings.control_cost, readings.candidate_cost
            more = "more" if ours is not None and theirs is not None and theirs > ours else "no less"
            said.append(f"{subject}cost {more} per reply")
        else:
            said.append(subject + clause.reason.replace("the candidate ", ""))
    return said


def _counts(rate: RateReading, control: str) -> str:
    """A rate that settled keep, as plain counts: ``passed 2 of 12 trials to production's 10``.

    The control's total is said only where it differs from the candidate's, so the counts stay three figures.
    """
    theirs = f"{rate.candidate_passed} of {rate.candidate_n}"
    ours = (
        str(rate.control_passed) if rate.control_n == rate.candidate_n else f"{rate.control_passed} of {rate.control_n}"
    )
    if rate.name == "the pass rate":
        return f"passed {theirs} trials to {control}'s {ours}"
    return f"passed {rate.name} on {theirs} to {control}'s {ours}"


def _cost_per_case(bundle: AnalysisContextBundle) -> str | None:
    """What one more case costs over every arm, from what the run spent per case; None where unpriced."""
    spent = [board.test_cost_usd for board in bundle.scoreboard]
    cases = max((board.cases for board in bundle.scoreboard), default=0)
    if not cases or any(cost is None for cost in spent):
        return None
    return f"${format_number(sum(cost or 0.0 for cost in spent) / cases)}"


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
    """One clause per declared question: what its contrasts and guardrails came to, in their own words."""
    clauses = []
    questions = _declared_questions(bundle)
    for index, question in enumerate(questions, start=1):
        # The first screen quotes the question's words; the verdict names it, never repeats it.
        lead = "Declared question" if len(questions) == 1 else f"Question {index}"
        separated = _separated(bundle, arm, question_id=question.id, every_family=False)
        tested = any(family.question_id == question.id for family in bundle.multiple_comparisons.families)
        if separated:
            clauses.append(f"{lead}: " + "; ".join(separated))
        elif tested:
            clauses.append(f"{lead}: no difference found between {_every_arm(bundle, arm)}")
        else:
            clauses.append(f"{lead}: not tested")
    if not clauses:
        return None
    return ". ".join(clauses) + "." + _guardrail_words(bundle, arm)


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
                reading = measure_name(comparison.name, heading.reader_name if heading is not None else None)
                if comparison.reading == "judged":
                    reading = f"{humanize('dimension', comparison.name)} (judged)"
                elif comparison.reading == "trial":
                    reading = "the pass rate"
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
    "caveat_lines",
    "rule_sentence",
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
