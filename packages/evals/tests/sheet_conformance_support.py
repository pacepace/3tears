"""The sheet check: a renderer of a blind read's sheet keeps the sheet's order and its slots.

A host that shows a blind read to people renders its :class:`~threetears.evals.schema.BlindSheet` (from the
``blind_read_sheet`` action) on its own page. The sample is random only if the rater meets the items in the drawn
order, and the pair test is fair only if each output sits in the slot it was drawn into. Neither shows in the
sheet's types, so this check renders a sheet built to expose each mistake and reads the text back:

- **in order, next first** — every unanswered item appears, from ``next`` on, in the sheet's order, never
  re-sorted (the sample's items are named so that sorting them by name, or by content, reorders them); an answered
  item may be left out, and is in its place where it is shown. A page that opens on an item the rater chose is a
  sample nobody drew;
- **slots as given** — an item's two outputs appear A then B as drawn, never re-sorted (the sample's B output sorts
  before its A output by text).

The check is the engine's own, in its test suite: the quick path's page and the courier example's page are proven by
it (``tests/test_blind_read_door.py``). It moves to :mod:`threetears.evals.testing` once a host's own suite calls
it, since a public name answers to a caller.

**What it cannot see.** That the page shows nothing beyond the sheet, that a skip is recorded as an answer, or that
nothing of other raters' answers reaches the page: a renderer is handed the sheet alone, and those are the page's
own behaviour. The engine's quick-path page passes it (``tests/test_blind_read_door.py``).

"""

from __future__ import annotations

from collections.abc import Callable

from threetears.evals.schema.blind_read import (
    BlindSheet,
    SheetAsk,
    SheetCriterion,
    SheetItem,
    SheetLevel,
    SheetOutput,
)

__all__ = ["SheetConformanceFailure", "check_sheet_renderer"]


class SheetConformanceFailure(AssertionError):
    """A sheet renderer broke a rule a blind read relies on; the message names the rule and what was seen.

    An :class:`AssertionError`, so a test runner reports it as a failed assertion.
    """


#: The criterion the sample asks.
_DIM = "sample.layout_kept"


def _item(word: str, name: str, *, pair: bool) -> SheetItem:
    """One sample item, its content named by ``word``; a pair's B output sorts before its A output."""
    outputs = [SheetOutput(slot="A", text=f"zulu output of the {word} case")]
    if pair:
        outputs.append(SheetOutput(slot="B", text=f"alpha output of the {word} case"))
    return SheetItem(
        item=name,
        case_material=f"the {word} case of the sample",
        outputs=outputs,
        ask=SheetAsk(verdicts=[_DIM], preference=pair),
    )


def _sample() -> BlindSheet:
    """A sheet whose drawn order is neither its items' name order nor their content order, one item answered."""
    names = ["item-m", "item-c", "item-x", "item-a"]
    return BlindSheet(
        read_id="blind-sample",
        name="sheet conformance sample",
        instructions="Answer the items in the order shown, starting at the top.",
        criteria=[
            SheetCriterion(
                dim=_DIM,
                question="The output keeps the layout.",
                scale="pass_fail",
                levels=[SheetLevel(value="pass"), SheetLevel(value="fail")],
            )
        ],
        items=[
            _item(word, name, pair=position % 2 == 0)
            for position, (word, name) in enumerate(zip(["mike", "charlie", "xray", "alpha"], names, strict=True))
        ],
        answered=["item-m"],
        next="item-c",
    )


def _at(text: str, needle: str, rule: str) -> int:
    where = text.find(needle)
    if where < 0:
        raise SheetConformanceFailure(f"{rule}: the rendered page does not show {needle!r}")
    return where


def check_sheet_renderer(render: Callable[[BlindSheet], str]) -> None:
    """Render a sample sheet with ``render`` and check its order from ``next`` on, and its slots.

    :param render: The host's renderer: a sheet in, the page's text out (its HTML, or what it prints).
    :ptype render: Callable[[BlindSheet], str]
    :raises SheetConformanceFailure: A rule is broken; the message names it.
    """
    sheet = _sample()
    text = render(sheet)
    if not isinstance(text, str):
        raise SheetConformanceFailure(f"the renderer returned {type(text).__name__}, not the page's text")
    shown = [item for item in sheet.items if item.item not in sheet.answered or item.case_material in text]
    places = [_at(text, item.case_material, "in order, next first") for item in shown]
    if places != sorted(places):
        raise SheetConformanceFailure(
            "in order, next first: the items are not shown in the sheet's order "
            f"({[shown[index].item for index in sorted(range(len(places)), key=places.__getitem__)]} shown, "
            f"{[item.item for item in shown]} drawn, {sheet.next} next)"
        )
    for item in shown:
        if len(item.outputs) != 2:
            continue
        a, b = (_at(text, output.text, "slots as given") for output in item.outputs)
        if not a < b:
            raise SheetConformanceFailure(
                f"slots as given: item {item.item}'s output B is shown before its output A; slots are never re-sorted"
            )
