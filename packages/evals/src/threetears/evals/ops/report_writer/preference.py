"""Pairwise preference between two report writer pages: the criterion, the evidence, and one case's reading.

An absolute judge reads one page, so it cannot see what that page leaves out; and the writing loop revises each page
until it passes the rubric that judge scores, so absolute scores bunch at the top. A judge reading two pages of one
case side by side sees an omission as the other page's content. This module is that question, built with no engine
change: one ordinal rubric dimension (:func:`preference_rubric`), one :class:`JudgeEvidence` holding both pages
(:func:`pair_evidence`), scored through the judge service like any dimension, once in each order. What the scores of
one case say about the candidate is :func:`case_preference`.

Everything here is pure: no storage, no model call. The pass that asks the judge and the read that decides on the
shares are their callers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from threetears.evals.ops.report_writer.rubric import shipped_rubric
from threetears.evals.ops.report_writer.writer_contract import WRITER_KIND
from threetears.evals.schema.models import JudgeEvidence, RubricDim

#: The names the pairwise pass and its read call: the criterion, the pair's evidence, and one case's reading with
#: its input and result types.
__all__ = ["CasePreference", "PairJudgment", "Slot", "case_preference", "pair_evidence", "preference_rubric"]

#: The preference dimension's name.
_PREFERRED_DIM = f"{WRITER_KIND}.preferred"

#: The description's lines before the rubric's items: what is under review, and what the score means.
_PREFERENCE_PREAMBLE = (
    "The output under review is two reports written over the same data, Report A first and Report B second, read "
    "together. The score is not a grade of either report: it says which one serves its reader better, and by how "
    "much, so read 1 and 5 by the scoring guide, never as worst and best.",
    "Weigh these items together:",
)

#: Which of the pair's two places a page sits in: ``A`` is read first, ``B`` second.
type Slot = Literal["A", "B"]

#: The lowest and highest score on the ordinal scale the dimension is judged on.
_LOW, _HIGH = 1, 5


def preference_rubric() -> RubricDim:
    """The dimension a pair is judged on: which report serves its reader better, on the editor rubric's items.

    Its description says the output under review is the two reports together and that the score is a direction
    between them, not a grade of either (the built-in prompt calls 1 worst and 5 best, which on this scale would
    read as "Report A is worst"); then the shipped rubric's items, numbered, as ``editor.md`` states them
    (:func:`~threetears.evals.ops.report_writer.rubric.shipped_rubric`). Its guide anchors 1 at Report A clearly
    better, 3 at no difference a reader would act on, and 5 at Report B clearly better. Built from the rubric's text
    alone, so the same ``editor.md`` always gives the same criterion
    (:func:`~threetears.evals.schema.models.fingerprint_criterion`) and a changed item another. Like the writer's
    absolute dimensions, it is always the shipped skill's, never an arm's, and declares the built-in judge prompt,
    so the prompt is held fixed with the criterion.

    :return: The dimension, ordinal 1-5.
    :rtype: RubricDim
    """
    return RubricDim(
        name=_PREFERRED_DIM,
        description="\n".join(
            [*_PREFERENCE_PREAMBLE, *(f"{item.n}. {item.name}. {item.definition}" for item in shipped_rubric().items)]
        ),
        scale="ordinal",
        scoring_guide={
            "1": "Report A serves the reader clearly better.",
            "3": "No difference between the two reports that a reader would act on.",
            "5": "Report B serves the reader clearly better.",
        },
        judge="builtin",
    )


def pair_evidence(a: JudgeEvidence, b: JudgeEvidence) -> JudgeEvidence:
    """What the judge reads for one order of a pair: both pages, ``a`` as Report A and ``b`` as Report B.

    Each side is the evidence its result stored, so nothing is re-rendered. The artifact is Report A's page, then
    Report B's; the case material is each page's computed facts, labelled the same way. Swapping the arguments
    swaps the labels and nothing else, so the two orders of one pair are two judged outputs of the same question.

    :param a: The stored evidence of the page placed first.
    :ptype a: JudgeEvidence
    :param b: The stored evidence of the page placed second.
    :ptype b: JudgeEvidence
    :return: The pair's evidence, with no subject: a report speaks for itself.
    :rtype: JudgeEvidence
    :raises ValueError: Either side carries a subject, which the pair has no place for.
    """
    if a.subject is not None or b.subject is not None:
        raise ValueError("a report page's evidence has no subject; a pair of pages with one would drop it unread")
    return JudgeEvidence(
        subject=None,
        case_material=f"=== Report A's facts ===\n{a.case_material}\n\n=== Report B's facts ===\n{b.case_material}",
        artifact=f"=== Report A ===\n{a.artifact}\n\n=== Report B ===\n{b.artifact}",
    )


@dataclass(frozen=True, slots=True)
class PairJudgment:
    """One judge reply on one order of a pair, from the candidate's side.

    :ivar candidate: The slot the candidate's page sat in; the control's sat in the other.
    :ivar score: The score on :func:`preference_rubric`'s 1-5 scale, or None when the judge answered it cannot tell.
    """

    candidate: Slot
    score: int | None


@dataclass(frozen=True, slots=True)
class CasePreference:
    """What one case's judgments, both orders and every repeat, say about the candidate against the control.

    A judgment's share is the part of the preference the candidate won, on 0 to 1: ``(score - 1) / 4`` when it sat in
    B, ``(5 - score) / 4`` when it sat in A. 0.5 is no preference; the control's share is one minus the candidate's.

    :ivar share: The mean of the two orders' shares, so a judge's lean to one slot cancels; one order's alone when the
        other has no score; None when no judgment scored.
    :ivar share_as_a: The candidate's mean share over the judgments where it sat in A; None when none scored.
    :ivar share_as_b: The same where it sat in B.
    :ivar scored: The judgments that gave a score.
    :ivar cannot_tell: The judgments that answered they cannot tell, in no share.
    :ivar order_consistent: Whether the two orders pick the same side (the candidate, the control, or neither);
        None unless both orders scored.
    :ivar first_slot_bias: How far the judge leans to Report A, from -0.5 to 0.5: the mean share of the page in A
        over the two orders, less 0.5, which is ``(share_as_a - share_as_b) / 2``. 0 is no lean; None unless both
        orders scored.
    """

    share: float | None
    share_as_a: float | None
    share_as_b: float | None
    scored: int
    cannot_tell: int
    order_consistent: bool | None
    first_slot_bias: float | None


def case_preference(judgments: Sequence[PairJudgment]) -> CasePreference:
    """The candidate's preference in one case, read from its judgments in both orders.

    :param judgments: Every reply on the case's pair, in either order and any number of repeats.
    :ptype judgments: Sequence[PairJudgment]
    :return: The case's reading.
    :rtype: CasePreference
    :raises ValueError: There is no judgment, a slot is neither A nor B, or a score is off the 1-5 scale.
    """
    if not judgments:
        raise ValueError("a case's preference needs at least one judgment")
    shares: dict[Slot, list[Fraction]] = {"A": [], "B": []}
    for judgment in judgments:
        if judgment.candidate not in shares:
            raise ValueError(f"preference slot {judgment.candidate!r} is neither A nor B")
        score = judgment.score
        if score is None:
            continue
        if isinstance(score, bool) or not _LOW <= score <= _HIGH:
            raise ValueError(f"preference score {score!r} is not on the 1-5 scale")
        won = _HIGH - score if judgment.candidate == "A" else score - _LOW
        shares[judgment.candidate].append(Fraction(won, _HIGH - _LOW))
    as_a, as_b = _mean(shares["A"]), _mean(shares["B"])
    by_order = [share for share in (as_a, as_b) if share is not None]
    scored = len(shares["A"]) + len(shares["B"])
    return CasePreference(
        share=float(sum(by_order) / len(by_order)) if by_order else None,
        share_as_a=None if as_a is None else float(as_a),
        share_as_b=None if as_b is None else float(as_b),
        scored=scored,
        cannot_tell=len(judgments) - scored,
        order_consistent=None if as_a is None or as_b is None else _side(as_a) == _side(as_b),
        first_slot_bias=None if as_a is None or as_b is None else float((as_a - as_b) / 2),
    )


def _mean(values: list[Fraction]) -> Fraction | None:
    """The exact mean, or None for no value."""
    return sum(values, Fraction(0)) / len(values) if values else None


def _side(share: Fraction) -> int:
    """Which side a share picks: 1 the candidate, -1 the control, 0 neither."""
    half = Fraction(1, 2)
    return 0 if share == half else (1 if share > half else -1)
