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
from threetears.evals.analysis.bundle import ComparisonVerdict
from threetears.evals.analysis.cells import variant_of_cell_ref
from threetears.evals.analysis.viz.intent import Cell
from threetears.evals.analysis.viz_refs import cell_arm_labels
from threetears.evals.contracts.campaign import ConfidenceTier, EvalAnalysis, EvidenceRow, EvidenceTier
from threetears.evals.contracts.surface import GuardrailDecision


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
    """What a finding stands on, as a reader says it, naming the old rule for a tier stored before intervals.

    A judged tier in an analysis stored before tiers were decided on the agreement's interval
    (``judged_tier_rule`` None) was the point estimate against the bar, so it is never presented as this
    build's claim: the words say which rule decided it.

    Args:
        analysis: The analysis the tier was stored in.
        tier: The finding's tier.

    Returns:
        The words.
    """
    words = EVIDENCE_TIER_WORDS[tier]
    if analysis.judged_tier_rule is None and tier in _JUDGED_TIERS:
        return f"{words} (tier decided on the point estimate of agreement, before tiers required its interval to clear the bar)"
    return words


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
    ("n", "n"),
    ("spread", "Spread"),
)


def evidence_rows(
    analysis: EvalAnalysis, evidence: Sequence[EvidenceRow], arm: Callable[[str], str]
) -> list[dict[str, Cell]]:
    """A finding's evidence as table rows, keyed by :data:`EVIDENCE_COLUMNS` — each reading by what a reader calls it.

    The measure is headed as the decision surface heads it
    (:meth:`~threetears.evals.contracts.surface.DecisionSurface.measure_heading`), never by its key unless nothing
    names it; the key stays on the analysis's evidence rows, which every reader of the record can cite.

    Args:
        analysis: The analysis the evidence was resolved under.
        evidence: One finding's resolved evidence rows, in the order code resolved them.
        arm: The analysis's :func:`arm_namer`.

    Returns:
        One row per evidence row, in order.
    """
    surface = analysis.decision_surface
    return [
        {
            "arm": arm(row.cell_ref),
            "measure": surface.measure_heading(row.measure_id, row.reading),
            "value": row.value,
            "n": row.n,
            "spread": row.dispersion,
        }
        for row in evidence
    ]


def positions(numbers: list[int]) -> str:
    """Finding positions as a reader counts them: from one."""
    return ", ".join(str(position + 1) for position in numbers)


__all__ = [
    "ARM_STATUS_WORDS",
    "COMPARISON_VERDICT_WORDS",
    "CONFIDENCE_WORDS",
    "EVIDENCE_COLUMNS",
    "EVIDENCE_TIER_WORDS",
    "evidence_rows",
    "stands_on_words",
    "arm_namer",
    "positions",
    "worded",
]
