"""The promotion rule, read where a decision is read: by the promotion read on a completed pass, and by the pairwise
pass at a stop, where its rule asks whether the decision read there stands.

One rule, so the pass that stops early and the read that decides later can never part
(:mod:`~threetears.evals.ops.report_writer.writer_promotion` states it in full):

- ``promote`` when the primary reads improved on the cases a rate needs, no gated reading reads regressed, and every
  guardrail is held;
- ``keep`` when the primary reads regressed, a gated reading reads regressed, or a guardrail is breached;
- ``undecided`` otherwise. **Undecided never promotes.**

A primary is void when the judge's order consistency is under :data:`ORDER_CONSISTENCY_FLOOR`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

from threetears.evals.analysis.report.blocks import Verdict

#: What a promotion read answers.
PromotionDecision = Literal["promote", "keep", "undecided"]

#: The order consistency under which the primary reading is void (the reset design's instrument check).
ORDER_CONSISTENCY_FLOOR: Final = 0.7

#: What a void primary's reason says, the same in the decision's sentence and in the readout.
VOID: Final = f"order consistency below {ORDER_CONSISTENCY_FLOOR}"


def void_reason(order_consistency: float | None, *, consistent: int, both_orders: int) -> str | None:
    """Why the primary reading is void, when its order consistency is under the floor; else None.

    :param order_consistency: The share of the cases scored in both orders whose two orders picked the same side;
        None when no case was.
    :ptype order_consistency: float | None
    :param consistent: Those cases, counted.
    :ptype consistent: int
    :param both_orders: The cases scored in both orders.
    :ptype both_orders: int
    :return: The reason, naming the counts; None when the primary stands.
    :rtype: str | None
    """
    if order_consistency is None or order_consistency >= ORDER_CONSISTENCY_FLOOR:
        return None
    return f"{VOID} ({consistent} of {both_orders} cases scored in both orders picked the same side)"


def decided(
    primary: Verdict | None, void: str | None, verdicts: Sequence[Verdict], *, candidate: str, control: str
) -> tuple[PromotionDecision, str]:
    """The decision rule (the module docstring), and its sentence.

    :param primary: The pass's verdict on the preference; None when no pass was read.
    :ptype primary: Verdict | None
    :param void: Why the primary is void (:func:`void_reason`); None when it stands.
    :ptype void: str | None
    :param verdicts: The candidate's corrected verdicts: its gated readings and its guardrails.
    :ptype verdicts: Sequence[Verdict]
    :param candidate: The candidate arm, by its label.
    :ptype candidate: str
    :param control: The control arm, by its label.
    :ptype control: str
    :return: The decision, and the sentence that says it.
    :rtype: tuple[PromotionDecision, str]
    """
    standing = None if void is not None else primary
    regressed = [verdict for verdict in verdicts if verdict.kind == "contrast" and verdict.outcome == "regressed"]
    guardrails = [verdict for verdict in verdicts if verdict.kind == "guardrail"]
    worse = [
        *(
            [_named("the primary reading", standing)]
            if standing is not None and standing.outcome == "regressed"
            else []
        ),
        *(_named(verdict.heading, verdict) for verdict in regressed),
        *(f"guardrail {verdict.heading} (breached)" for verdict in guardrails if verdict.outcome == "breached"),
    ]
    if worse:
        return "keep", f"Keep the control: {candidate} is shown worse than {control} on {', '.join(worse)}."
    unheld = sum(verdict.outcome != "held" for verdict in guardrails)
    improved = standing is not None and standing.outcome == "improved" and standing.reason != "too_few_cases"
    if improved and not unheld:
        return "promote", (
            f"Promote {candidate}: the judge prefers its pages to {control}'s on the primary reading, no gated "
            "reading is shown worse and every guardrail is held."
        )
    if improved:
        return "undecided", (
            f"Undecided: the judge prefers {candidate}'s pages on the primary reading, but {unheld} of "
            f"{len(guardrails)} guardrails are not shown held."
        )
    detail = "" if primary is None or primary.reason_detail is None else f" ({primary.reason_detail})"
    if primary is None:
        because = "no pairwise pass was read, so the primary reading is absent"
    elif void is not None:
        because = f"{void}, so the primary reading is void"
    elif primary.reason == "too_few_cases":
        because = f"the primary reading has too few cases to decide{detail}"
    elif primary.outcome == "not_separated":
        because = f"the primary reading does not separate {candidate} from {control}"
    else:
        because = f"the primary reading is {primary.outcome.replace('_', ' ')}{detail}"
    return "undecided", f"Undecided: {because}, and only an improved primary reading promotes."


def _named(label: str, verdict: Verdict) -> str:
    """A reading the decision names, with too few cases beside it when it was read on fewer than a rate needs."""
    return f"{label} (on too few cases)" if verdict.reason == "too_few_cases" else label


#: The pairwise pass reads the rule at a stop and the promotion read reads it on the pass; the void floor and its words
#: are the promotion readout's.
__all__ = ["ORDER_CONSISTENCY_FLOOR", "PromotionDecision", "VOID", "decided", "void_reason"]
