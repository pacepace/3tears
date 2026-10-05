"""The seam a chart renderer sits behind, and the one conformance check every renderer passes.

**The core ships no charting library.** It decides what a chart says — a
:class:`~threetears.evals.analysis.viz.intent.ChartIntent` — and a renderer, supplied by the host or
installed as an adapter, decides how it looks. :class:`ChartRenderer` is that contract: a renderer is
constructed with its host's theme, draws an intent into its own form, and reads back out of that form
the data it actually placed. The package's own Vega-Lite renderer is one such adapter
(``threetears.evals.vega``, the ``[vega]`` extra); the core never imports it.

**Why a renderer reads its own drawing back.** The intent states its values twice — ``data``, the marks'
numbers in drawn units, and ``rows``, the values-as-drawn table a reader checks the picture against. A
renderer that dropped a mark, placed a value it was not handed, or drew a row the table does not hold
would produce a well-formed picture that disagrees with the table beside it, and nothing downstream
can see that. :func:`renderer_disagreements` is the check: it draws the intent, asks the renderer what
the drawing places, and compares that against the intent's own values. It reads the DRAWING, not the
intent the renderer was handed, so a renderer cannot pass by echoing its input.

**What is compared, and what is not.** Per identity — the field the intent's rows are named by — every
value ``data`` holds for an encoded field the values table also states must be among the values the
drawing places for that identity, as many times as the intent holds it; every identity the values table
names must be drawn; and no drawn datum may name an identity the intent does not hold. An encoded field
the table does not state — a distribution's raw samples — is the renderer's to summarise (a rug, a
density, a binned band), so it is not held to appear verbatim. A renderer may reshape its data freely (melt a
row into one record per level, draw an interval as a span and two caps, add a layout helper), because
the comparison is by value under an identity rather than by field name. Auxiliary marks with no
identity — a reference rule, a band, an anchor — are the renderer's own and are not compared. Order,
geometry and colour are a picture's properties and are each renderer's own gate.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Protocol, TypeVar

from threetears.evals.analysis.viz.intent import Cell, ChartIntent

#: A renderer's own form of a drawn chart — a Vega-Lite spec, an SVG, a component's props.
DrawingT = TypeVar("DrawingT")

#: Significant digits a drawn number is compared at: a renderer that restates a value through
#: arithmetic of its own (a unit scale, a sum) agrees with the intent to well past what a reader sees.
_SIGNIFICANT_DIGITS = 12


class ChartRenderer(Protocol[DrawingT]):
    """A chart renderer: a host's theme, bound at construction, applied to any intent.

    The theme is the renderer's own type and so is not part of this contract — a Vega-Lite theme is a
    palette and a font directory, another renderer's is whatever it draws with. What every renderer
    shares is the two operations below.
    """

    def draw(self, intent: ChartIntent) -> DrawingT:
        """Draw ``intent`` in this renderer's form.

        Args:
            intent: A decided chart intent, as :func:`~threetears.evals.analysis.viz.intent.chart_intent`
                returns it.

        Returns:
            The drawing.
        """
        ...

    def drawn_data(self, drawing: DrawingT) -> list[dict[str, Cell]]:
        """Every datum ``drawing`` places, read back out of the drawing itself.

        Each datum that belongs to a row of the chart carries that row's identity under the intent's
        identity field — a renderer that drew a shortened label resolves it back to the identity it
        stands for. A datum the renderer added for its own layout carries no identity.

        Args:
            drawing: A drawing this renderer produced.

        Returns:
            The data, as field-to-value mappings.
        """
        ...


def _comparable(value: Cell) -> tuple[str, object] | None:
    """A cell as a comparison key: numbers at :data:`_SIGNIFICANT_DIGITS`, bools apart from 0 and 1."""
    if value is None:
        return None
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int | float):
        number = float(value)
        if number == 0.0 or not math.isfinite(number):
            return ("number", number)
        return ("number", float(f"{number:.{_SIGNIFICANT_DIGITS}g}"))
    return ("text", value)


def _identity_of(datum: Mapping[str, Cell], field: str | None) -> str | None:
    """The identity a datum is drawn under, or None for one that names none (or a chart with no identity)."""
    if field is None:
        return ""
    value = datum.get(field)
    return None if value is None else str(value)


def renderer_disagreements(renderer: ChartRenderer[DrawingT], intent: ChartIntent) -> list[str]:
    """Where ``renderer``'s drawing of ``intent`` disagrees with the intent's values — empty when it agrees.

    The one conformance check for a chart renderer; see the module docstring for what it compares. A
    host bringing its own renderer runs this over the intents its reports carry.

    Args:
        renderer: The renderer under test, with whatever theme it was built with.
        intent: The intent to draw.

    Returns:
        One sentence per disagreement.
    """
    field = intent.identity.field if intent.identity is not None else None
    # The encoded fields the values table also states: what the table tells a reader the picture shows.
    # A field the table does not carry (a distribution's raw samples, which a renderer may draw as a
    # density or a binned band) is the renderer's to summarise, so it is not held to appear verbatim.
    tabled_fields = {column.key for column in intent.columns}
    encoded = [
        encoding.field for encoding in intent.encodings if encoding.field != field and encoding.field in tabled_fields
    ]

    held: dict[str, Counter[tuple[str, object]]] = {}
    for row in intent.data:
        identity = _identity_of(row, field)
        if identity is None:
            continue
        counter = held.setdefault(identity, Counter())
        counter.update(key for key in (_comparable(row[name]) for name in encoded if name in row) if key is not None)

    drawn: dict[str, Counter[tuple[str, object]]] = {}
    for datum in renderer.drawn_data(renderer.draw(intent)):
        identity = _identity_of(datum, field)
        if identity is None:
            continue
        counter = drawn.setdefault(identity, Counter())
        counter.update(key for key in (_comparable(value) for value in datum.values()) if key is not None)

    tabled = [_identity_of(row, field) for row in intent.rows] if field is not None else []
    # What the intent holds: an identity with values in `data`, or a row of the table — a withheld row
    # (no values, stated as such) is held, and a renderer saying so beside its axis is drawing it.
    known = set(held) | set(tabled)
    disagreements = [
        f"drew {identity!r}, which the intent does not hold" for identity in drawn if identity not in known
    ]
    disagreements.extend(
        f"did not draw {identity!r}, which the values table holds"
        for identity in dict.fromkeys(tabled)
        if identity is not None and identity not in drawn
    )
    for identity, values in held.items():
        missing = values - drawn.get(identity, Counter())
        if missing:
            shown = ", ".join(repr(value) for _kind, value in sorted(missing.elements(), key=repr))
            disagreements.append(f"drew {identity!r} without the intent's value(s) {shown}")
    return disagreements


def assert_renderer_conforms(renderer: ChartRenderer[DrawingT], intents: Sequence[ChartIntent]) -> None:
    """Raise unless ``renderer`` draws every one of ``intents`` in agreement with its values.

    Args:
        renderer: The renderer under test.
        intents: The intents to draw — every chart type a host's reports carry, at least.

    Raises:
        AssertionError: A drawing disagrees with its intent; the message names each disagreement.
        ValueError: ``intents`` is empty, which would pass every renderer.
    """
    if not intents:
        raise ValueError("a conformance run over no intents passes every renderer; hand it the intents to draw")
    failures = [
        f"{intent.type} ({intent.title}): {disagreement}"
        for intent in intents
        for disagreement in renderer_disagreements(renderer, intent)
    ]
    if failures:
        raise AssertionError("the renderer's drawing disagrees with the intent:\n  " + "\n  ".join(failures))


__all__ = [
    "ChartRenderer",
    "assert_renderer_conforms",
    "renderer_disagreements",
]
