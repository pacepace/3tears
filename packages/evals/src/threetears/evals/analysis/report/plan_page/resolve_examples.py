"""Example cards: what the failures look like, a card per example the query selects.

The cards are the examples the query selects (:func:`~threetears.evals.analysis.report_rows.query_rows`: the arms,
strata and aspect it keeps, the first ``limit``, in the input's order), so the cards are the ones the section's
headline speaks of. Each card's meta line holds the trial and the case (mono, as ids a reader can look up), the
stratum or task by its reader name, the arm as its chip, the failure class as a pill where the example names one, and
the repeat; then the case's input and what it expected, where the example carries them; then the judge's reason, or
code's line saying none was recorded; then the excerpt.
"""

from __future__ import annotations

from threetears.evals.analysis.report.charts import (
    ExampleCard,
    ExampleField,
    ExampleMetaArm,
    ExampleMetaStatus,
    ExampleMetaText,
    ExamplesChart,
)
from threetears.evals.analysis.report.plan_page.charting import ChartContext, Drawn, KindResolver, chart_block
from threetears.evals.analysis.report.plan_page.words import (
    EXPECTED_TERM,
    INPUT_TERM,
    NO_REASON,
    REPEAT_TEMPLATE,
)
from threetears.evals.analysis.report_input import Example
from threetears.evals.analysis.style import humanize

__all__ = ["RESOLVERS"]


def _place(context: ChartContext, stratum: str) -> str:
    """A stratum's or task's reader name, whichever the example's id names."""
    if any(held.id == stratum for held in context.report_input.strata):
        return context.look.reader("stratum", stratum)
    if any(held.id == stratum for held in context.report_input.tasks):
        return context.look.reader("task", stratum)
    return humanize("dimension", stratum)


def _card(context: ChartContext, example: Example) -> ExampleCard:
    meta: list[ExampleMetaText | ExampleMetaStatus | ExampleMetaArm] = []
    meta += [ExampleMetaText(text=one, mono=True) for one in (example.trial, example.case) if one]
    if example.stratum:
        meta.append(ExampleMetaText(text=_place(context, example.stratum)))
    if example.arm is not None and example.arm in context.look.arms:
        look = context.look.arm(example.arm)
        meta.append(ExampleMetaArm(tag=look.tag, label=look.name, color=look.token))
    elif example.arm is not None:
        meta.append(ExampleMetaText(text=example.arm))
    if example.outcome:
        meta.append(ExampleMetaStatus(status="bad", label=humanize("dimension", example.outcome)))
    if example.repeat is not None:
        meta.append(ExampleMetaText(text=REPEAT_TEMPLATE.format(repeat=example.repeat)))
    fields = [
        ExampleField(term=term, text=text)
        for term, text in ((INPUT_TERM, example.input), (EXPECTED_TERM, example.expected))
        if text
    ]
    return ExampleCard(
        meta=meta or None,
        fields=fields or None,
        reason=example.reason or NO_REASON,
        excerpt=example.excerpt or None,
    )


def _examples(context: ChartContext) -> Drawn:
    if context.query.source != "examples":
        raise context.refuse("example cards draw examples")
    examples = [row.item for row in context.rows() if isinstance(row.item, Example)]
    if not examples:
        raise context.refuse("it selects no example to draw")
    return Drawn(chart_block(context, ExamplesChart(cards=[_card(context, example) for example in examples])))


#: The kinds this module resolves.
RESOLVERS: dict[str, KindResolver] = {"examples": _examples}
