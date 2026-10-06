"""The presentation rules a chart is held to, read off its intent — so they hold for every renderer.

**Written against the intent, not against any renderer's output.** A rule here is a property of what
a chart SAYS: whether its lengths start at zero, whether every quantity names its unit, whether its
intervals say what they span, whether identity rides on position rather than colour. Those are true or
false before anything is drawn, and stating them over :class:`~threetears.evals.analysis.viz.intent.ChartIntent`
is what lets a host bring its own renderer without bringing its own idea of an honest chart. How a
renderer realises them — upright text, an undashed grid, one scale across layers — is the renderer's
own gate (the Vega-Lite compiler's is :mod:`threetears.evals.vega.spec_policy`), and its
conformance test holds it to the intent it was handed.

The rules, numbered as the report standards number them:

1. **A length starts at zero.** A ``length`` or ``count`` field is measured from its axis's zero, so the
   axis declares ``zero_baseline``; a cropped axis under a length multiplies every ratio read off it.
2. **One unit per quantity, stated.** Every measured field and every drawn standard names an axis the
   intent declares, and an axis with a unit states it in the quantity a reader is told.
3. **A fixed-order palette that never refuses to draw.** A categorical scheme's domain is explicit and
   distinct — each value has one slot — and a domain wider than
   :data:`~threetears.evals.analysis.viz.payloads.SERIES_SLOTS` recycles with a warning rather than a
   refusal: the number of series is a property of the data.
5. **A named variability span.** An interval end says what the interval varies over.
7. **Identity never rides on colour.** The field a row is named by is never coloured, and a categorical
   scheme past the validated slots needs a direct label on every mark.
10. **A ranking is ordered by a measure it draws.** A ranked identity names a drawn field and its
    rows descend on it.
11. **An ordered category states its order.** An ``ordinal`` position — a build, a day, a named bin —
    is on an axis that states its order, and every position drawn is in it.
12. **A chart can be read without being seen.** Its values table has columns, every row keyed by them,
    a chart that places marks has rows, and **the table states what is drawn**: a row's value under a key
    a mark of the same identity also carries is that mark's value — the same number, the same text, or text
    spelling the number to the precision it is written at (``-31.8 s`` for -31.8; ``+72.7%`` for 0.727, a
    ``%`` reading as a percent of the value). Matched on the identity and every text field the two share (a
    timeseries' position, a sweep's levels). A column only the table carries (a delta table's arm values) is
    compared with nothing drawn, so it is the builder's to spell; everything a renderer places is tied to
    the table here, and the renderer to the marks by its conformance check.

Rules 4 (one scale across layers and panels), 6 (prose is never checked) and 8-9 (rendered text upright,
grid solid) are not here: 4 and 8-9 are properties of a rendered figure and live in each renderer's
gate, and 6 is a rule about not having a rule.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, TypeIs

from threetears.evals.analysis.viz.intent import INTERVAL_ROLES, LENGTH_ROLES, MEASURED_ROLES
from threetears.evals.analysis.viz.payloads import SERIES_SLOTS, VALIDATED_SLOTS
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.analysis.viz.intent import ChartIntent

log = get_logger(__name__)


class IntentPolicyError(ValueError):
    """A chart intent breaks a presentation rule — a data problem, never a rendering one."""


def enforce_intent(intent: ChartIntent) -> None:
    """Raise unless ``intent`` satisfies every rule.

    All are checked in one pass, so a producer fixing a chart gets the whole list.

    Args:
        intent: The chart intent.

    Raises:
        IntentPolicyError: One or more rules are broken; the message lists each.
    """
    if violations := check_intent(intent):
        raise IntentPolicyError(f"{intent.type} chart breaks the presentation rules: " + "; ".join(violations))


def check_intent(intent: ChartIntent) -> list[str]:
    """Check a chart intent against the presentation rules.

    Args:
        intent: The chart intent.

    Returns:
        One sentence per violation; empty when the intent satisfies every rule.
    """
    return [
        *_check_lengths_start_at_zero(intent),
        *_check_quantities_are_on_stated_axes(intent),
        *_check_palette(intent),
        *_check_named_variability(intent),
        *_check_identity_is_not_coloured(intent),
        *_check_ranked_by_a_drawn_measure(intent),
        *_check_ordinals_state_their_order(intent),
        *_check_readable_without_the_picture(intent),
    ]


def _check_lengths_start_at_zero(intent: ChartIntent) -> list[str]:
    """Rule 1: a length or a count is measured from an axis that starts at zero."""
    return [
        f"{encoding.field} is a {encoding.role} drawn against axis {encoding.axis!r}, which does not start at zero — "
        "a cropped baseline under a length draws a ratio the values do not contain"
        for encoding in intent.encodings
        if encoding.role in LENGTH_ROLES and (axis := intent.axis(encoding.axis)) is not None and not axis.zero_baseline
    ]


def _check_quantities_are_on_stated_axes(intent: ChartIntent) -> list[str]:
    """Rule 2: every measured field names a declared axis, and an axis with a unit states it."""
    violations: list[str] = []
    for encoding in intent.encodings:
        if encoding.role in MEASURED_ROLES and intent.axis(encoding.axis) is None:
            violations.append(
                f"{encoding.field} is a {encoding.role} measured against no declared axis ({encoding.axis!r}) — "
                "a quantity drawn without its ruler"
            )
    violations.extend(
        f"the {reference.label} is drawn across no declared axis ({reference.axis!r})"
        for reference in intent.references
        if intent.axis(reference.axis) is None
    )
    violations.extend(
        f"axis {axis.name!r} is drawn in {axis.unit} but tells the reader {axis.quantity!r}, which does not name it"
        for axis in intent.axes
        if axis.unit and f"({axis.unit})" not in axis.quantity
    )
    return violations


def _check_palette(intent: ChartIntent) -> list[str]:
    """Rule 3: each coloured value takes one fixed slot; a wide domain recycles with a warning, never a refusal."""
    violations: list[str] = []
    for colours in intent.colours:
        if not colours.domain:
            violations.append(f"the {colours.scheme} scheme on {colours.field} has no domain, so no value has a slot")
            continue
        if repeated := sorted({value for value in colours.domain if colours.domain.count(value) > 1}):
            violations.append(
                f"the {colours.scheme} scheme on {colours.field} lists {', '.join(repeated)} twice — one value, two slots"
            )
        drawn = {row.get(colours.field) for row in intent.data if row.get(colours.field) is not None}
        if unplaced := sorted(
            str(value) for value in drawn if str(value) not in colours.domain and str(value) not in colours.uncoloured
        ):
            violations.append(
                f"{colours.field} draws {', '.join(unplaced)}, which the {colours.scheme} scheme gives no slot"
            )
        if colours.scheme == "categorical" and len(colours.domain) > SERIES_SLOTS:
            # Never a refusal: the palette draws past its width by recycling, and refusing would discard a
            # legitimate chart over how many series the data has.
            log.warning(
                "chart intent %s colours %d values on %s, past the %d slots a theme supplies; slots recycle",
                intent.type,
                len(colours.domain),
                colours.field,
                SERIES_SLOTS,
            )
    return violations


def _check_named_variability(intent: ChartIntent) -> list[str]:
    """Rule 5: an interval end says what the interval varies over."""
    return [
        f"{encoding.field} is an interval end that does not say what the interval varies over"
        for encoding in intent.encodings
        if encoding.role in INTERVAL_ROLES and not encoding.varies_over.strip()
    ]


def _check_identity_is_not_coloured(intent: ChartIntent) -> list[str]:
    """Rule 7: the identity field is never coloured, and past the validated slots every mark is labelled."""
    violations: list[str] = []
    named_by = intent.identity.field if intent.identity is not None else None
    for colours in intent.colours:
        if colours.field == named_by:
            violations.append(
                f"{colours.field} names the rows and is coloured — identity rides on position, never on hue, "
                "because the palette recycles and an axis does not"
            )
        if colours.scheme == "categorical" and len(colours.domain) > VALIDATED_SLOTS and not intent.direct_labels:
            violations.append(
                f"{colours.field} takes {len(colours.domain)} categorical slots, past the {VALIDATED_SLOTS} validated "
                "ones, without a direct label on every mark — two values would draw alike with nothing to tell them apart"
            )
    return violations


def _check_ranked_by_a_drawn_measure(intent: ChartIntent) -> list[str]:
    """Rule 10: a ranking names a field the chart measures, and its rows descend on it."""
    identity = intent.identity
    if identity is None or not identity.ranked_by:
        return []
    encoding = intent.encoding(identity.ranked_by)
    if encoding is None or encoding.role not in MEASURED_ROLES:
        return [f"the rows are ranked by {identity.ranked_by!r}, which the chart does not draw as a measure"]
    values = [row.get(identity.ranked_by) for row in intent.data]
    numbers = [value for value in values if isinstance(value, int | float) and not isinstance(value, bool)]
    if len(numbers) != len(values) or numbers != sorted(numbers, reverse=True):
        return [f"the rows claim a ranking by {identity.ranked_by!r} and do not descend on it"]
    return []


def _check_ordinals_state_their_order(intent: ChartIntent) -> list[str]:
    """Rule 11: an ordinal position is on an axis that states its order, and every drawn position is in it."""
    violations: list[str] = []
    for encoding in intent.encodings:
        if encoding.role != "ordinal":
            continue
        axis = intent.axis(encoding.axis)
        if axis is None or not axis.order:
            violations.append(
                f"{encoding.field} is an ordinal position on an axis that states no order — a renderer left to sort "
                "names puts 0.10 before 0.9"
            )
            continue
        drawn = {str(value) for row in intent.data if (value := row.get(encoding.field)) is not None}
        if stray := sorted(drawn - set(axis.order)):
            violations.append(f"{encoding.field} draws {', '.join(stray)}, which its axis's stated order does not hold")
    return violations


def _check_readable_without_the_picture(intent: ChartIntent) -> list[str]:
    """Rule 12: the values table has columns, keys its rows by them, and has rows when the chart places marks."""
    violations: list[str] = []
    if not intent.columns:
        violations.append("the chart has no values table, so a surface that cannot draw it can say nothing of it")
    keys = {column.key for column in intent.columns}
    # A key whose every value is absent states nothing a column would show, so only a stated value counts.
    stated = {key for row in intent.rows for key, value in row.items() if value is not None}
    if stray := sorted(stated - keys - _UNTABLED_KEYS):
        violations.append(f"the values table's rows state {', '.join(stray)}, which no column shows")
    if intent.data and not intent.rows:
        violations.append("the chart places marks and its values table has no rows")
    violations.extend(table_disagreements(intent))
    return violations


def _is_number(value: object) -> TypeIs[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)


#: A number as a values table spells it — optionally signed, with a decimal part, perhaps a percent.
_SPELLED_NUMBER = re.compile(r"(?P<number>[-+−]?\d+(?:\.\d+)?)(?P<percent>\s*%)?")


def _spells(text: str, drawn: float) -> bool:
    """Whether ``text`` states ``drawn`` to the precision it is written at — as itself, or as a percent of it.

    A table spells a drawn number for reading (``-31.8 s`` for -31.8, ``+72.7%`` for 0.727), so a text cell
    standing for a mark's number is held to name that number: some number written in it equals the mark's
    value, or the value times 100 when a ``%`` follows it, within half a unit of the last digit written.
    """
    for match in _SPELLED_NUMBER.finditer(text):
        written = match.group("number").replace("−", "-")
        stated = float(written)
        decimals = len(written.split(".", 1)[1]) if "." in written else 0
        target = drawn * 100 if match.group("percent") else drawn
        if abs(stated - target) <= 0.5 * 10**-decimals + 1e-12:
            return True
    return False


def table_disagreements(intent: ChartIntent) -> list[str]:
    """Where the values table states a value no mark of the same identity carries — empty when it agrees.

    The half of rule 12 that ties the table a reader checks the picture against to the marks a renderer
    places; see the module docstring for what is compared and what is not.

    Args:
        intent: The chart intent.

    Returns:
        One sentence per row that disagrees with every mark it could be.
    """
    if intent.identity is None:
        return []
    field = intent.identity.field
    disagreements: list[str] = []
    # Each row against the marks of its identity: a row stating a value none of them carries.
    for row in intent.rows:
        if field not in row:
            continue
        marks = [datum for datum in intent.data if datum.get(field) == row[field] and _comparable(row, datum, field)]
        # A withheld row — stated in the table, with no mark — draws nothing the table could contradict.
        if marks and all(_differing(row, datum, field) for datum in marks):
            closest = min((_differing(row, datum, field) for datum in marks), key=len)
            stated = ", ".join(f"{key}={row[key]!r}" for key in closest)
            disagreements.append(
                f"the values table's row for {row[field]!r} states {stated}, which no mark of {row[field]!r} carries"
            )
    # Each mark against the rows that could state it: a drawn value no row states. This half also covers a
    # table that shows the identity's parts rather than the identity (a sweep's levels), whose rows carry no
    # identity to join on, and a truncated table, whose rows past the drawn ones stand for nothing drawn.
    for datum in intent.data:
        rows = [
            row
            for row in intent.rows
            if (field not in row or row[field] == datum.get(field)) and _comparable(row, datum, field)
        ]
        if rows and all(_differing(row, datum, field) for row in rows):
            closest = min((_differing(row, datum, field) for row in rows), key=len)
            drawn = ", ".join(f"{key}={datum[key]!r}" for key in closest)
            disagreements.append(
                f"the chart draws {datum.get(field)!r} at {drawn}, which no row of the values table states"
            )
    return disagreements


def _comparable(row: dict[str, object], datum: dict[str, object], field: str) -> bool:
    """Whether ``row`` and ``datum`` state anything both of them hold besides the identity."""
    return any(
        key != field and key in datum and value is not None and datum[key] is not None for key, value in row.items()
    )


def _differing(row: dict[str, object], datum: dict[str, object], field: str) -> list[str]:
    """The keys on which ``row`` states something other than ``datum`` — see :func:`table_disagreements`."""
    differ = []
    for key, value in row.items():
        if key == field or key not in datum or value is None or datum[key] is None:
            continue
        drawn = datum[key]
        if _is_number(value) and _is_number(drawn):
            if not math.isclose(value, drawn, rel_tol=1e-9, abs_tol=1e-12):
                differ.append(key)
        elif isinstance(value, str) and _is_number(drawn):
            if not _spells(value, drawn):
                differ.append(key)
        elif value != drawn:
            differ.append(key)
    return differ


#: Row keys a values table may carry without a column: the drawn short name a renderer labels a mark with,
#: kept beside the full identity the table shows.
_UNTABLED_KEYS: frozenset[str] = frozenset({"display"})


__all__ = [
    "IntentPolicyError",
    "check_intent",
    "enforce_intent",
    "table_disagreements",
]
