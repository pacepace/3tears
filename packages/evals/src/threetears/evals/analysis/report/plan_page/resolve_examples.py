"""Example cards: what the failures look like, a card per example the query selects.

The cards are the examples the query selects (:func:`~threetears.evals.analysis.report_rows.query_rows`: the arms,
strata and aspect it keeps, the first ``limit``, in the input's order), so the cards are the ones the section's
headline speaks of. Each card's meta line holds the trial and the case (mono, as ids a reader can look up), the
aspect by its reader name where the input holds more than one (so a block over two aspects' failures says which each
is), the stratum or task by its reader name, the arm as its chip, the failure class as a pill where the example names one, and
the repeat; then the case's input and what it expected, where the example carries them; then the judge's reason, or
code's line saying none was recorded; then the excerpt.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

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

__all__ = ["RESOLVERS", "readable_fields"]

#: The most fields one input or expectation gives a card: the few that say what the case was.
_FIELDS_SHOWN = 4

#: The longest a field's value reads before it is cut at a word.
_VALUE_CHARS = 80

#: The mark a cut value ends on.
_CUT = "…"


def _elided(text: str) -> str:
    """A value cut at a word where it runs past :data:`_VALUE_CHARS`, the cut marked."""
    text = " ".join(text.split())
    if len(text) <= _VALUE_CHARS:
        return text
    cut = text[:_VALUE_CHARS].rsplit(" ", 1)[0] or text[:_VALUE_CHARS]
    return cut.rstrip(" ,;:") + _CUT


def _said(value: object) -> str:
    """A JSON value as a reader says it: a scalar as it is, a list as its items, an object as its pairs."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return ", ".join(said for item in value if (said := _said(item)))
    if isinstance(value, dict):
        return "; ".join(
            f"{humanize('dimension', str(key))} {said}" for key, item in value.items() if (said := _said(item))
        )
    return str(value)


def _pairs(text: str) -> Iterator[tuple[str, object]]:
    """The key and value pairs a JSON object's text holds, in order, as far as it reads: a text cut short ends at its
    last whole pair, and a string value cut mid-way is kept as far as it goes."""
    decoder = json.JSONDecoder()
    at = text.index("{") + 1
    key: object = None
    while True:
        while at < len(text) and text[at] in " \t\r\n,":
            at += 1
        if at >= len(text) or text[at] != '"':
            return
        key = None
        try:
            key, at = decoder.raw_decode(text, at)
            while at < len(text) and text[at] in " \t\r\n":
                at += 1
            if at >= len(text) or text[at] != ":":
                return
            at += 1
            while at < len(text) and text[at] in " \t\r\n":
                at += 1
            value, at = decoder.raw_decode(text, at)
        except json.JSONDecodeError:
            # The text is an excerpt cut where the snapshot was frozen, so its last pair may be partial: the pairs read
            # so far are the card's fields, and a string value cut short is kept as far as it goes.
            if key is not None and at < len(text) and text[at] == '"':
                yield str(key), text[at + 1 :].rstrip(_CUT)
            return
        yield str(key), value


def _empty_object(text: str) -> bool:
    """Whether a text is a JSON object, read whole, with no pair in it."""
    try:
        read: object = json.loads(text)
    except json.JSONDecodeError:
        return False
    return read == {}


def readable_fields(term: str, text: str) -> list[ExampleField]:
    """What a case gave or expected, as fields a reader reads. A text that is a JSON object (a world, a payload),
    whole or cut short where the snapshot was frozen, is its key and value pairs: the first few with something to say,
    each key by its reader name and each long value cut at a word, never the JSON itself; any other text is one field,
    as it is. A text that opens on ``{`` and yields no pair (prose, ``{customer} asked for a refund``) is that one
    field too, so an input never vanishes from its card; only an object that reads whole and holds nothing (``{}``)
    gives none.

    :param term: The field's term, ``Input`` or ``Expected``, which a pair's key follows.
    :ptype term: str
    :param text: The case's input or expectation, as the example carries it.
    :ptype text: str
    :return: The fields; none for a text with nothing to say.
    :rtype: list[ExampleField]
    """
    if not text.lstrip().startswith("{"):
        return [ExampleField(term=term, text=text)] if text.strip() else []
    pairs = list(_pairs(text.lstrip()))
    if not pairs:
        return [] if _empty_object(text) else [ExampleField(term=term, text=text)]
    held = [(key, _elided(said)) for key, value in pairs if (said := _said(value).strip())]
    return [
        ExampleField(term=f"{term} · {humanize('dimension', key)}", text=said) for key, said in held[:_FIELDS_SHOWN]
    ]


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
    if example.aspect and len(context.report_input.aspects) > 1:
        meta.append(ExampleMetaText(text=context.look.reader("aspect", example.aspect)))
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
        field
        for term, text in ((INPUT_TERM, example.input), (EXPECTED_TERM, example.expected))
        if text
        for field in readable_fields(term, text)
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
