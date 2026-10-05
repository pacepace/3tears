"""The words an analysis is read in — one vocabulary for the memo as written and for the report.

A confidence tier, an evidence tier and an arm are each spelled once, here, and both readers of an
analysis — the memo the reporter eval's judge reads (:func:`~threetears.evals.analysis.reporter_kind.render_memo_as_written`)
and the :class:`~threetears.evals.analysis.report.model.Report` every surface renders — take them from this
module. Two spellings of one arm on two surfaces would read as two arms.
"""

from __future__ import annotations

import typing
from collections.abc import Callable
from typing import Any, Literal

from threetears.evals.analysis.arms import ArmStatus, arm_label, arm_levels, distinguishing_axes
from threetears.evals.analysis.cells import variant_of_cell_ref
from threetears.evals.analysis.viz_refs import cell_arm_labels
from threetears.evals.contracts.campaign import ConfidenceTier, EvalAnalysis, EvidenceTier


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
        "directional": "judged scores, directional until the judge's reliability is measured",
        "none": "no reading it names",
    },
    EvidenceTier,
    "an evidence tier",
)


#: Where an arm stands, as a reader says it.
ARM_STATUS_WORDS = worded(
    {
        "winner": "winner",
        "ruled_out": "ruled out",
        "replaced_incumbent": "replaced incumbent",
        "unresolved": "unresolved",
    },
    ArmStatus,
    "an arm status",
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
    index = {entry.variant_key: entry for entry in analysis.variant_index}
    distinguishing = distinguishing_axes(analysis.variant_index)

    def name(ref: str) -> str:
        if ref in labels:
            return labels[ref]
        variant = variant_of_cell_ref(ref)
        if variant is None:
            return "a cell this analysis cannot read"
        entry = index.get(variant)
        return arm_label(
            variant,
            arm_levels(entry, distinguishing),
            levels_unavailable=entry.levels_unavailable if entry else None,
            placed=entry is not None,
        )

    return name


def positions(numbers: list[int]) -> str:
    """Finding positions as a reader counts them: from one."""
    return ", ".join(str(position + 1) for position in numbers)


__all__ = [
    "ARM_STATUS_WORDS",
    "CONFIDENCE_WORDS",
    "EVIDENCE_TIER_WORDS",
    "arm_namer",
    "positions",
    "worded",
]
