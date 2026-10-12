"""The words an analysis is read in — one vocabulary for the memo as written and for the report.

A confidence tier, an evidence tier and an arm are each spelled once, here, and both readers of an
analysis — the memo the reporter eval's judge reads (:func:`~threetears.evals.analysis.reporter_kind.render_memo_as_written`)
and the :class:`~threetears.evals.analysis.report.model.Report` every surface renders — take them from this
module. Two spellings of one arm on two surfaces would read as two arms.
"""

from __future__ import annotations

import typing
from collections.abc import Callable, Sequence
from typing import Any, Literal

from threetears.evals.analysis.arms import ArmStatus, arm_label, arm_names
from threetears.evals.analysis.bundle.schema import ComparisonVerdict
from threetears.evals.analysis.cells import variant_of_cell_ref
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.references import resolve_reading, restated_dispersion
from threetears.evals.analysis.report.model import Fact, TableCell
from threetears.evals.analysis.standing import Standing
from threetears.evals.analysis.surface_table import UNITLESS, measure_rulers
from threetears.evals.analysis.viz.intent import ChartValue
from threetears.evals.analysis.viz_refs import cell_arm_labels
from threetears.evals.kernel.campaign import ConfidenceTier, EvalAnalysis, EvidenceRow, EvidenceTier
from threetears.evals.kernel.analysis_measures import BarDecision
from threetears.evals.kernel.surface import GuardrailDecision


def _literal_values(annotation: Any) -> frozenset[str]:
    """Every value a ``Literal`` annotation admits, through an optional ``| None``."""
    if typing.get_origin(annotation) is Literal:
        return frozenset(typing.get_args(annotation))
    return frozenset(value for arg in typing.get_args(annotation) for value in _literal_values(arg))


def worded(words: dict[str, str], annotation: Any, what: str) -> dict[str, str]:
    """Refuse at import a word table that does not cover exactly the values its field admits.

    A value added to the model and not here would otherwise reach the page as its raw token — the
    very vocabulary a reader's layout exists to keep from them.

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


#: The tiers a judge's measured agreement decides — the ones whose rule a stored analysis records.
_JUDGED_TIERS = frozenset({"calibrated", "separation", "undetermined", "incidental"})


def stands_on_words(analysis: EvalAnalysis, tier: EvidenceTier) -> str:
    """What a finding stands on, as a reader says it.

    Args:
        analysis: The analysis the tier was stored in.
        tier: The finding's tier.

    Returns:
        The words.
    """
    return EVIDENCE_TIER_WORDS[tier]


#: Where an arm stands, as a reader says it.
ARM_STATUS_WORDS = worded(
    {
        "winner": "winner",
        "contradicted": "contradicted: one decision adopts it and another rejects it",
        "ruled_out": "ruled out",
        "replaced_incumbent": "replaced incumbent",
        "unresolved": "unresolved",
    },
    ArmStatus,
    "an arm status",
)


#: What a contrast's test said against the control, as a reader says it.
COMPARISON_VERDICT_WORDS = worded(
    {
        "improved": "improved on the control",
        "regressed": "regressed from the control",
        "equivalent": "equivalent to the control, within the measure's margin",
        "not_separated": "not separated from the control",
        "untested": "untested",
    },
    ComparisonVerdict,
    "a comparison verdict",
)


#: What a guardrail came to for an arm against the control, as a reader says it.
GUARDRAIL_DECISION_WORDS = worded(
    {
        "held": "held: shown no worse than the control by more than the margin",
        "breached": "breached: shown worse than the control by more than the margin",
        "undecided": "undecided: not shown held, so not known to be safe",
    },
    GuardrailDecision,
    "a guardrail decision",
)


#: What a bar came to on one cell, as a reader says it.
BAR_DECISION_WORDS = worded(
    {
        "cleared": "cleared: the interval lies wholly on the good side of the bar, less its margin",
        "missed": "missed: the interval lies wholly on the bad side of the bar, less its margin",
        "undecided": "undecided: the interval straddles the bar, so not shown cleared",
        "no_interval": "no interval: fewer than two observations, so not read",
        "no_data": "no data: no observation of the measure",
    },
    BarDecision,
    "a bar decision",
)


def arm_namer(analysis: EvalAnalysis) -> Callable[[str], str]:
    """Name the arm a cell reference points at, as :func:`~threetears.evals.analysis.arms.arm_label` names it.

    With a decision surface, the words are
    :func:`~threetears.evals.analysis.viz_refs.cell_arm_labels`' — the same call the charts make, so the
    rig is named exactly where it tells two cells of one arm apart. Without one, the variant index is
    all there is to name the arm by, and the labeller is the same one; a reference naming a variant the
    index does not hold says so in that labeller's own words rather than printing the reference.

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
            return "a cell this analysis cannot read"
        return arm_label(variant, names)

    return name


#: A finding's evidence table, column key → header: one layout for the memo as written and for the report, so the
#: judge reads the table a reader sees.
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

    The measure is headed as the decision surface heads it
    (:meth:`~threetears.evals.kernel.surface.DecisionSurface.measure_heading`), never by its key unless nothing
    names it; the key stays on the analysis's evidence rows, which every reader of the record can cite.

    A measure's value and spread are stated on its one ruler for the report
    (:func:`~threetears.evals.analysis.surface_table.measure_rulers`), the value carrying its unit, as every other
    table states it: the stored row holds the figure in the measure's own unit, and a latency the surface table
    states in s read ``950`` here, unitless, in ms. A judged score has no unit and is stated as stored.

    Args:
        analysis: The analysis the evidence was resolved under.
        evidence: One finding's resolved evidence rows, in the order code resolved them.
        arm: The analysis's :func:`arm_namer`.

    Returns:
        One row per evidence row, in order.
    """
    surface = analysis.decision_surface
    rulers = measure_rulers(surface)
    rows: list[dict[str, TableCell]] = []
    for row in evidence:
        ruler = rulers.get(row.measure_id, UNITLESS) if row.reading == "measure" else UNITLESS
        values: dict[str, ChartValue] = {
            "arm": arm(row.cell_ref),
            "measure": surface.measure_heading(row.measure_id, row.reading),
            # Cases, never observations: the count every interval is read over. An analysis stored before rows
            # recorded it shows none rather than its observation count under this heading.
            "cases": row.n_cases,
            "spread": _restated_spread(analysis, row, ruler.factor),
        }
        cells = {key: TableCell(value=value) for key, value in values.items()}
        cells["value"] = TableCell(value=row.value * ruler.factor, unit=ruler.unit)
        rows.append({key: cells[key] for key, _ in EVIDENCE_COLUMNS})
    return rows


def _restated_spread(analysis: EvalAnalysis, row: EvidenceRow, factor: float) -> str:
    """An evidence row's spread on its value's ruler: as stored where nothing is restated, else re-read off the surface.

    The stored spread is words over the figures in the measure's own unit, so a restated row re-reads its reading
    off the analysis's frozen surface, which resolved it at generation. A surface that no longer resolves it (the
    resolver has since learned a refusal) keeps the stored words and names the unit they are in.
    """
    if factor == 1.0:
        return row.dispersion
    surface = analysis.decision_surface
    try:
        resolved = resolve_reading(surface, row.cell_ref, row.measure_id, row.reading)
    except UnresolvableReference:
        facts = surface.measures.get(row.measure_id)
        return f"{row.dispersion} (in {facts.unit})" if facts is not None and facts.unit else row.dispersion
    return restated_dispersion(resolved, factor)


#: What a headline or summary bullet stands on in an analysis with no findings.
STANDING_NO_FINDING = "no finding: the analysis has none"


def standing_facts(analysis: EvalAnalysis, standing: Standing) -> list[Fact]:
    """What a headline or summary bullet stands on, as the facts code states beside it.

    The same two facts a finding's title carries, read off the weakest of the findings it rests on
    (:mod:`threetears.evals.analysis.standing`): its confidence, and what its evidence is.

    Args:
        analysis: The analysis.
        standing: The sentence's standing.

    Returns:
        The facts, in reading order.
    """
    if standing.confidence is None:
        return [Fact(name="Stands on", value=STANDING_NO_FINDING)]
    facts = [Fact(name="Confidence", value=CONFIDENCE_WORDS[standing.confidence])]
    if standing.evidence_tier is not None:
        facts.append(Fact(name="Stands on", value=stands_on_words(analysis, standing.evidence_tier)))
    return facts


def positions(numbers: list[int]) -> str:
    """Finding positions as a reader counts them: from one."""
    return ", ".join(str(position + 1) for position in numbers)


__all__ = [
    "ARM_STATUS_WORDS",
    "BAR_DECISION_WORDS",
    "COMPARISON_VERDICT_WORDS",
    "CONFIDENCE_WORDS",
    "EVIDENCE_COLUMNS",
    "EVIDENCE_TIER_WORDS",
    "evidence_rows",
    "stands_on_words",
    "arm_namer",
    "STANDING_NO_FINDING",
    "positions",
    "standing_facts",
    "worded",
]
