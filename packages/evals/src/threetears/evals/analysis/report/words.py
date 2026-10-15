"""The words a report is read in — one vocabulary for the memo as written and for the page.

A confidence tier, an evidence tier, a verdict and an arm are each spelled once, here, and both readers of an
analysis — the memo the reporter eval's judge reads (:func:`~threetears.evals.analysis.reporter_kind.render_memo_as_written`)
and the :class:`~threetears.evals.analysis.report.model.Report` every door renders — take them from this module.
Two spellings of one arm on two surfaces would read as two arms.

**The first two screens speak a reader's words, not the engine's.** A trial is one case run once on one arm, and it
is passed, failed (the candidate's failure) or lost (the test setup's). The engine's own nouns for the same things —
a cell, an observation, an exclusion, the rig, the harness, the apparatus, a reserved judge axis's stored id — never
reach those screens; :data:`ENGINE_WORDS` maps each to what a reader is told instead, and
:data:`FIRST_SCREEN_BANNED` is what a test holds the rendered screens to. No arm is ever called a winner.
"""

from __future__ import annotations

import typing
from collections.abc import Callable, Sequence
from typing import Any, Literal

from threetears.evals.analysis.arms import arm_label, arm_names, multi_rig_variants, short_digest
from threetears.evals.analysis.bundle.schema import ComparisonVerdict
from threetears.evals.analysis.cells import variant_of_cell_ref
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.references import cell_index, resolve_reading, restated_dispersion
from threetears.evals.analysis.report.figures import MeasureRuler, analysis_rulers
from threetears.evals.analysis.report.model import CellValue, TableCell
from threetears.evals.kernel.analysis_measures import BarDecision
from threetears.evals.analysis.style import measure_name
from threetears.evals.kernel.campaign import ConfidenceTier, EvalAnalysis, EvidenceRow, EvidenceTier, VariantIndexEntry
from threetears.evals.kernel.metrics import goal_check_of
from threetears.evals.kernel.surface import DecisionSurface, GuardrailDecision


def _literal_values(annotation: Any) -> frozenset[str]:
    """Every value a ``Literal`` annotation admits, through an optional ``| None``."""
    if typing.get_origin(annotation) is Literal:
        return frozenset(typing.get_args(annotation))
    return frozenset(value for arg in typing.get_args(annotation) for value in _literal_values(arg))


def worded(words: dict[str, str], annotation: Any, what: str) -> dict[str, str]:
    """Refuse at import a word table that does not cover exactly the values its field admits.

    A value added to the model and not here would otherwise reach the page as its raw token — the very vocabulary
    a reader's layout exists to keep from them.

    Args:
        words: Value → its words.
        annotation: The ``Literal`` the values come from.
        what: What the values are, for the refusal.

    Returns:
        ``words``, unchanged.

    Raises:
        RuntimeError: The table and the ``Literal`` disagree.
    """
    if set(words) != _literal_values(annotation):
        raise RuntimeError(
            f"the memo's words for {what} {sorted(words)} do not match the model's {sorted(_literal_values(annotation))}"
        )
    return words


#: The engine's nouns, and what a reader of a report's first two screens is told in their place.
ENGINE_WORDS: dict[str, str] = {
    "cell": "arm",
    "observation": "trial",
    "excluded": "lost to the test setup",
    "rig": "test setup",
    "harness": "test setup",
    "apparatus": "test setup",
    "__outcome__": "outcome",
    "__transcript__": "transcript",
}

#: Words that never appear on a report's first two screens: the engine's nouns, and "winner", which no arm is.
FIRST_SCREEN_BANNED: tuple[str, ...] = (*ENGINE_WORDS, "winner")

#: A confidence tier as a reader says it.
CONFIDENCE_WORDS = worded(
    {"very_high": "very high", "high": "high", "medium": "medium", "low": "low"}, ConfidenceTier, "a confidence tier"
)

#: What a verdict stands on, as a reader says it — code's tier, read off the finding's evidence rows.
EVIDENCE_TIER_WORDS = worded(
    {
        "mechanical": "checks and measures",
        "calibrated": "judged scores from a judge that agrees with people",
        "separation": "judged scores from a judge that agrees with itself; not shown to agree with people",
        "undetermined": "judged scores whose judge's reliability has not been measured enough to say",
        "incidental": "judged scores from a judge measured as neither agreeing with people nor with itself",
        "none": "no reading it names",
    },
    EvidenceTier,
    "an evidence tier",
)


def stands_on_words(analysis: EvalAnalysis, tier: EvidenceTier) -> str:
    """What a finding stands on, as a reader says it.

    Args:
        analysis: The analysis the tier was stored in.
        tier: The finding's tier.

    Returns:
        The words.
    """
    return EVIDENCE_TIER_WORDS[tier]


#: What a contrast's test said against the control, as a reader says it: each arm by its label and the direction by
#: name, never "the control", which a reader took for a place ("improved on the control" read as "in the control").
#: Filled by :func:`comparison_words`.
COMPARISON_VERDICT_WORDS = worded(
    {
        "improved": "{arm} better than {control}{better_end}",
        "regressed": "{arm} worse than {control}{better_end}",
        "equivalent": "{arm} and {control} equivalent, within the measure's margin",
        "not_separated": "{arm} and {control} not separated",
        "untested": "untested",
    },
    ComparisonVerdict,
    "a comparison verdict",
)

#: A measure's better end, in brackets, where it is not the higher one a reader assumes.
LOWER_IS_BETTER = " (lower is better)"


def comparison_words(verdict: ComparisonVerdict, arm: str, control: str, *, higher_is_better: bool = True) -> str:
    """A contrast's verdict in words, each arm by its label: ``v2 better than v1``, ``v2 worse than v1 (lower is
    better)``, ``v2 and v1 not separated``.

    Args:
        verdict: What the contrast's test came to.
        arm: The arm read against the control, as the page names it.
        control: The control, as the page names it.
        higher_is_better: The reading's direction; a lower-is-better reading's better end is said in brackets.

    Returns:
        The words.
    """
    better_end = "" if higher_is_better else LOWER_IS_BETTER
    return COMPARISON_VERDICT_WORDS[verdict].format(arm=arm, control=control, better_end=better_end)


#: What a guardrail came to for an arm against the control, as a reader says it, each arm by its label. Filled by
#: :func:`guardrail_words`.
GUARDRAIL_DECISION_WORDS = worded(
    {
        "held": "held: {arm} shown no worse than {control} by more than the margin",
        "breached": "breached: {arm} shown worse than {control} by more than the margin",
        "undecided": "undecided: not shown held, so not known to be safe",
    },
    GuardrailDecision,
    "a guardrail decision",
)


def guardrail_words(decision: GuardrailDecision, arm: str, control: str) -> str:
    """A guardrail's decision in words, each arm by its label: ``held: v2 shown no worse than v1 by more than the
    margin``."""
    return GUARDRAIL_DECISION_WORDS[decision].format(arm=arm, control=control)


#: What a bar came to on one arm, as a reader says it.
BAR_DECISION_WORDS = worded(
    {
        "cleared": "cleared: the interval lies wholly on the good side of the bar, less its margin",
        "missed": "missed: the interval lies wholly on the bad side of the bar, less its margin",
        "undecided": "undecided: the interval straddles the bar, so not shown cleared",
        "no_interval": "no interval: fewer than two trials, so not read",
        "no_data": "no data: no trial of the measure",
    },
    BarDecision,
    "a bar decision",
)


def reading_name(name: str) -> str:
    """A judged dimension or measure key as a reader reads it: a reserved judge axis by its plain name."""
    return ENGINE_WORDS.get(name, name)


def reading_heading(surface: DecisionSurface, name: str, reading: str = "measure") -> str:
    """A reading as the page heads it, in a column, a row or a chart's title.

    A goal check by its expression, verbatim as code (:func:`~threetears.evals.analysis.style.measure_name`); how
    each trial ended (a ``trial`` reading) as the decision rule names it, "the pass rate"; any other reading by the
    surface's heading, a reserved judge axis by its plain name (:func:`reading_name`).

    Args:
        surface: The decision surface the reading is on.
        name: The measure's key, or a judged dimension's name.
        reading: ``judged`` for a judged dimension, ``trial`` for the overall pass rate; anything else reads a measure.

    Returns:
        The heading.
    """
    if reading == "trial":
        return "the pass rate"
    if reading != "judged" and goal_check_of(name) is not None:
        return measure_name(name)
    return reading_name(surface.measure_heading(name, reading))


def cell_arm_labels(surface: DecisionSurface, variant_index: Sequence[VariantIndexEntry]) -> dict[str, str]:
    """Name every cell on the surface by its arm, adding the test setup only where one arm ran under two.

    The words are :func:`~threetears.evals.analysis.arms.arm_label`'s over the index's
    :func:`~threetears.evals.analysis.arms.arm_names`, so one arm reads alike in every table of the report and in
    the memo the reporter eval's judge reads.

    Args:
        surface: The decision surface whose cells to name.
        variant_index: The variant index, which says what each cell's arm carried.

    Returns:
        ``cell_ref -> label`` for every cell.
    """
    multi = multi_rig_variants(surface.cells)
    names = arm_names(variant_index)
    return {
        ref: arm_label(cell.variant_key, names)
        + (f" (test setup {short_digest(cell.apparatus_class_id)})" if cell.variant_key in multi else "")
        for ref, cell in cell_index(surface).items()
    }


def arm_namer(analysis: EvalAnalysis) -> Callable[[str], str]:
    """Name the arm a cell reference points at, as :func:`cell_arm_labels` names it.

    A reference naming a variant the analysis's surface does not hold is named by the variant index's labeller,
    in its own words, rather than printed as the reference.

    Args:
        analysis: The analysis whose references to name.

    Returns:
        A function from a cell reference to its arm's name.
    """
    labels = cell_arm_labels(analysis.decision_surface, analysis.variant_index)
    names = arm_names(analysis.variant_index)

    def name(ref: str) -> str:
        if ref in labels:
            return labels[ref]
        variant = variant_of_cell_ref(ref)
        if variant is None:
            return "an arm this analysis cannot read"
        return arm_label(variant, names)

    return name


#: A finding's evidence table, column key → header: one layout for the memo as written and for the report.
EVIDENCE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("arm", "Arm"),
    ("measure", "Measure"),
    ("value", "Value"),
    ("cases", "Cases"),
    ("spread", "Spread"),
)


def evidence_rows(
    analysis: EvalAnalysis, evidence: Sequence[EvidenceRow], arm: Callable[[str], str]
) -> list[dict[str, TableCell]]:
    """A finding's evidence as table rows, keyed by :data:`EVIDENCE_COLUMNS` — each reading by what a reader calls it.

    A measure's value and spread are stated on its one ruler for the report
    (:func:`~threetears.evals.analysis.report.figures.analysis_rulers`): a share on 0–1 as a percent, a cost per
    reply on the scale its heading names; a judged score on its dimension's; a goal check's
    pass rate is its count (``11 of 12``), as the scoreboard counts.

    Args:
        analysis: The analysis the evidence was resolved under.
        evidence: One finding's resolved evidence rows, in the order code resolved them.
        arm: The analysis's :func:`arm_namer`.

    Returns:
        One row per evidence row, in order.
    """
    surface = analysis.decision_surface
    rulers = analysis_rulers(analysis)
    rows: list[dict[str, TableCell]] = []
    for row in evidence:
        ruler = rulers.of(row.measure_id, row.reading)
        values: dict[str, CellValue] = {
            "arm": arm(row.cell_ref),
            "measure": ruler.scaled(reading_heading(surface, row.measure_id, row.reading)),
            # Cases, never trials: the count every interval is read over.
            "cases": row.n_cases,
            "spread": _restated_spread(analysis, row, ruler),
        }
        cells = {key: TableCell(value=value) for key, value in values.items()}
        cells["value"] = TableCell(value=row.value * ruler.factor, unit=ruler.unit)
        if (passed := _passed_of(row)) is not None:
            cells["value"] = TableCell(value=passed)
        rows.append({key: cells[key] for key, _ in EVIDENCE_COLUMNS})
    return rows


def _passed_of(row: EvidenceRow) -> str | None:
    """A goal check's pass rate as the scoreboard counts it (``11 of 12``); None for any other reading.

    A check holds or not on each observation, so its rate times its observations is the count that held.
    """
    if row.reading != "measure" or goal_check_of(row.measure_id) is None or row.n <= 0:
        return None
    held = row.value * row.n
    return f"{round(held)} of {row.n}" if abs(held - round(held)) < 1e-6 else None


def _restated_spread(analysis: EvalAnalysis, row: EvidenceRow, ruler: MeasureRuler) -> str:
    """An evidence row's spread on its value's ruler: as stored where nothing is restated, else re-read off the surface."""
    if ruler.factor == 1.0:
        return row.dispersion
    surface = analysis.decision_surface
    try:
        resolved = resolve_reading(surface, row.cell_ref, row.measure_id, row.reading)
    except UnresolvableReference:
        facts = surface.measures.get(row.measure_id)
        return f"{row.dispersion} (in {facts.unit})" if facts is not None and facts.unit else row.dispersion
    return restated_dispersion(resolved, ruler.factor, unit=ruler.unit)


def positions(numbers: list[int]) -> str:
    """Finding positions as a reader counts them: from one, each named ``F<n>``."""
    return ", ".join(f"F{position + 1}" for position in numbers)


def counted(n: int, of: int, what: str) -> str:
    """``n of m <what>``: a count, never a rate."""
    return f"{n} of {of} {what}"


__all__ = [
    "BAR_DECISION_WORDS",
    "COMPARISON_VERDICT_WORDS",
    "CONFIDENCE_WORDS",
    "ENGINE_WORDS",
    "EVIDENCE_COLUMNS",
    "EVIDENCE_TIER_WORDS",
    "FIRST_SCREEN_BANNED",
    "GUARDRAIL_DECISION_WORDS",
    "LOWER_IS_BETTER",
    "arm_namer",
    "cell_arm_labels",
    "comparison_words",
    "counted",
    "evidence_rows",
    "guardrail_words",
    "positions",
    "reading_heading",
    "reading_name",
    "stands_on_words",
    "worded",
]
