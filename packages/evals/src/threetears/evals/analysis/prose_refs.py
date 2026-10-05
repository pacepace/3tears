"""Figures in prose: the model writes a reference where a number goes, and code writes the number.

A number a reader sees is code's, including one inside a sentence. So the
model never types a figure into prose: it writes a **figure reference** —
``{{<cell>|<measure_id>|<reading>|<stat>}}`` — and :func:`render_prose_figures` replaces each one
with the value read off the decision surface, in the unit a chart axis would choose and spelled for a sentence.

- ``<cell>`` is the cell's short alias, its ``cell`` in ``cell_measures`` (``c1``, ``c2``, …), read back
  through :func:`~threetears.evals.analysis.references.cell_of_alias` like every other cell a writer names.
- ``<reading>`` is ``measure`` or ``judged``, stated rather than inferred, as on an evidence row.
- ``<stat>`` is one of :data:`MEASURE_STATS` for a numeric measure and :data:`JUDGED_STATS` for a
  judged dimension, which carries no distribution. A categorical measure has no point value, so it
  carries ``n`` and ``count:<category>`` — one category's count — which is the only way its counts
  reach prose, since the model types no figure. A numeric measure's zero-valued observations are
  ``n_zero``: the delivery-rate proxy the prompt's rule 2 pairs with every latency claim, which had
  no statistic until a writer asked for it the only way the grammar seemed to offer — a category
  count on a numeric measure, refused, and a paid repair.

A figure in a SENTENCE is spelled for a reader who reads it once, not for one comparing columns
(:func:`prose_number`): a goal-state check's pass rate is its count (``8 of 12``, which is what a
reader asks of a rate over a dozen tries), money carries its sign (``$0.012``), and every other
value is rounded to what a sentence can use (``0.22 rounds``, ``129 s``). Tables and charts keep
:func:`~threetears.evals.analysis.numbers.format_number`'s four figures.

A reference is STRUCTURE, not prose: it has a grammar, and what it names either resolves or is
refused. A malformed one, or one naming a cell, reading or statistic the surface does not hold, is
an :class:`~threetears.evals.analysis.errors.UnresolvableReference`, so it buys the generator's one
repair round with the resolver's account of what does exist. The words around a reference are
never read.

Rendered at generation, into the stored document, so every surface that shows the document —
the web page, the text render, the reporter eval's judge — shows the same resolved figure without
knowing references exist.

**The grammar is taught by this module, not by the store-master prompt.** The generator's prompt is
store-master: its seed fills an empty slot once, and only an operator promotion changes what a
deployment sends after that. The grammar is a code contract, so a copy of it in that prompt goes
stale on the first grammar change — the deployment keeps teaching a form this parser refuses, and
every generation there buys its paid repair round and is refused again. So :func:`reference_grammar`
writes the section that teaches it from this module's own constants, the generator appends it to
every system prompt (:func:`~threetears.evals.analysis.generator.assemble_system_prompt`), and its worked
examples are rendered by :func:`render_reference` on a small synthetic surface, so an example can
never read differently from what the renderer prints.
"""

from __future__ import annotations

import math
import re
from typing import Literal, get_args

from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.numbers import ABSENT, format_number
from threetears.evals.analysis.references import cell_aliases, cell_of_alias, require_cell, resolve_reading
from threetears.evals.analysis.viz.quantities import UNSPACED_UNITS, display_scale
from threetears.evals.contracts.analysis_measures import MeasureCollection, MeasureSummary
from threetears.evals.contracts.authored import AuthoredAnalysis
from threetears.evals.contracts.campaign import ReadingKind
from threetears.evals.contracts.metrics import goal_check_measure, goal_check_of
from threetears.evals.contracts.surface import CellFacts, DecisionSurface, MeasureFacts

#: The statistics a measure reference may name: its mean, n and zero count, and its distribution's points.
MeasureStat = Literal["mean", "n", "n_zero", "p05", "p50", "p95", "max"]
MEASURE_STATS: tuple[str, ...] = get_args(MeasureStat)

#: The readings a reference may name — a cell's two namespaces, stated rather than inferred.
READINGS: tuple[str, ...] = get_args(ReadingKind)

#: A judged dimension carries a mean and its n, and no distribution.
JUDGED_STATS: tuple[str, ...] = ("mean", "n")

#: What each reading carries; its keys are :data:`READINGS`, which the tests hold equal.
STATS_BY_READING: dict[str, tuple[str, ...]] = {"measure": MEASURE_STATS, "judged": JUDGED_STATS}

#: The statistic prefix naming one category's count on a categorical measure: ``count:<category>``.
CATEGORY_COUNT_PREFIX = "count:"

#: A reference's form, as the writer is taught it and as a malformed one is refused naming.
REFERENCE_FORM = "{{<cell>|<measure_id>|<reading>|<stat>}}"

#: One figure reference. Anything between double braces is a reference and must parse; the four
#: fields are separated by ``|``, which neither a cell reference nor a measure name contains.
FIGURE_REFERENCE = re.compile(r"\{\{([^{}]*)\}\}")

#: The magnitude at and above which a figure in a sentence is a whole number. Below it, two
#: significant figures: a sentence that says 0.2222 makes the reader do the rounding, and the
#: difference a sentence reports survives it (4.9 against 4.8; 23 s against 129 s).
PROSE_WHOLE_FROM = 10.0


def prose_number(value: float | None) -> str:
    """Spell a number for a sentence: whole from :data:`PROSE_WHOLE_FROM` up, two figures below, never an exponent.

    Args:
        value: The number, or ``None`` when there is none.

    Returns:
        Its text; :data:`~threetears.evals.analysis.numbers.ABSENT` for ``None``, NaN or an infinity.
    """
    if value is None or not math.isfinite(value):
        return ABSENT
    if value == int(value):
        return str(int(value))
    if abs(value) >= PROSE_WHOLE_FROM:
        return str(round(value))
    if abs(value) < 1e-4:
        # ``%g`` turns to an exponent here, and "$3.7e-05" is not a sentence; two figures, written out.
        return f"{value:.{1 - math.floor(math.log10(abs(value)))}f}"
    return f"{value:.2g}"


def state_in_prose(value: float, unit: str | None) -> str:
    """State one value in a sentence, in the unit a chart axis would choose for it.

    Args:
        value: The value, in the unit it was measured in.
        unit: That unit, or ``None`` when none was declared.

    Returns:
        The value's text: money as ``$0.012``, anything else joined to its unit where one is known.
    """
    factor, shown = display_scale([value], unit)
    scaled = value * factor
    if shown == "usd":
        return f"{'-' if scaled < 0 else ''}${prose_number(abs(scaled))}"
    text = prose_number(scaled)
    if not shown:
        return text
    return f"{text}{shown}" if shown in UNSPACED_UNITS else f"{text} {shown}"


def parse_reference(inner: str) -> tuple[str, str, str, str]:
    """Split one reference's inside into its four fields, or refuse it.

    Args:
        inner: The text between the braces.

    Returns:
        ``(cell, measure_id, reading, stat)``.

    Raises:
        UnresolvableReference: It is not four non-empty fields, the reading is neither kind, or the
            statistic is not one that reading carries.
    """
    fields = [field.strip() for field in inner.split("|")]
    if len(fields) != 4 or not all(fields):
        raise UnresolvableReference(f"figure reference {{{{{inner}}}}} is not `{REFERENCE_FORM}` with all four fields")
    cell, measure_id, reading, stat = fields
    allowed = STATS_BY_READING.get(reading)
    if allowed is None:
        readings = " or ".join(f"`{name}`" for name in READINGS)
        raise UnresolvableReference(f"figure reference {{{{{inner}}}}} names reading {reading!r}; it is {readings}")
    if (
        reading == "measure"
        and stat.startswith(CATEGORY_COUNT_PREFIX)
        and stat.removeprefix(CATEGORY_COUNT_PREFIX).strip()
    ):
        return cell, measure_id, reading, stat
    if stat not in allowed:
        carried = (
            f"{', '.join(allowed)}, or `{CATEGORY_COUNT_PREFIX}<category>` on a categorical one"
            if reading == "measure"
            else ", ".join(allowed)
        )
        raise UnresolvableReference(
            f"figure reference {{{{{inner}}}}} names statistic {stat!r}, which a {reading} reading does not carry; "
            f"it carries {carried}"
        )
    return cell, measure_id, reading, stat


def render_reference(surface: DecisionSurface, inner: str) -> str:
    """Resolve one reference against the surface and state its value.

    Args:
        surface: The analysis's decision surface.
        inner: The text between the braces.

    Returns:
        The value as a reader sees it: a count bare, a check's pass rate as ``<passed> of <evaluated>``,
        anything else in its display unit (:func:`state_in_prose`).

    Raises:
        UnresolvableReference: The reference is malformed, or names nothing the surface holds.
    """
    cell, measure_id, reading, stat = parse_reference(inner)
    cell_ref = cell_of_alias(surface, cell)
    if reading == "measure":
        counts = _category_counts(surface, inner, cell_ref, measure_id, stat)
        if counts is not None:
            return counts
    resolved = resolve_reading(surface, cell_ref, measure_id, "judged" if reading == "judged" else "measure")
    if stat == "n":
        return format_number(resolved.n)
    if stat == "n_zero":
        # A count of observations, not a quantity in the measure's unit: "3", never "3 items".
        zeros = _summary(surface, cell_ref, measure_id).n_zero
        if zeros is None:
            raise UnresolvableReference(
                f"figure reference {{{{{inner}}}}}: measure {measure_id!r} at {cell_ref!r} has no n_zero"
            )
        return format_number(zeros)
    if stat == "mean":
        if reading == "measure" and goal_check_of(measure_id) is not None:
            # A pass rate is passed / evaluated, so the count is exact, and it is what the reader wants.
            return f"{round(resolved.mean * resolved.n)} of {resolved.n}"
        return state_in_prose(resolved.mean, resolved.unit)
    value = getattr(_summary(surface, cell_ref, measure_id), stat)
    if value is None:
        raise UnresolvableReference(
            f"figure reference {{{{{inner}}}}}: measure {measure_id!r} at {cell_ref!r} has no {stat}"
        )
    return state_in_prose(value, resolved.unit)


def _summary(surface: DecisionSurface, cell_ref: str, measure_id: str) -> MeasureSummary:
    """The cell's summary of one measure, which :func:`resolve_reading` has already proved exists."""
    return next(m for m in require_cell(surface, cell_ref).measures.measures if m.name == measure_id)


def _category_counts(surface: DecisionSurface, inner: str, cell_ref: str, measure_id: str, stat: str) -> str | None:
    """State a categorical measure's ``n`` or one category's count; None for a numeric measure's point stat.

    Raises:
        UnresolvableReference: A category count named on a numeric measure, a category the measure
            never recorded, or a point statistic named on a categorical measure.
    """
    summary = next((m for m in require_cell(surface, cell_ref).measures.measures if m.name == measure_id), None)
    wants_count = stat.startswith(CATEGORY_COUNT_PREFIX)
    if summary is None:
        return None  # resolve_reading names what the cell does hold
    if not summary.categories:
        if wants_count:
            raise UnresolvableReference(
                f"figure reference {{{{{inner}}}}} names a category count, but {measure_id!r} at {cell_ref!r} is numeric; "
                f"it carries {', '.join(f'`{name}`' for name in MEASURE_STATS)}"
            )
        return None
    if stat == "n":
        return format_number(summary.n)
    if not wants_count:
        raise UnresolvableReference(
            f"figure reference {{{{{inner}}}}} names statistic {stat!r}, but {measure_id!r} at {cell_ref!r} is categorical "
            f"and has no point value; it carries `n` and `{CATEGORY_COUNT_PREFIX}<category>` for "
            f"{', '.join(sorted(summary.categories))}"
        )
    category = stat.removeprefix(CATEGORY_COUNT_PREFIX).strip()
    if category not in summary.categories:
        raise UnresolvableReference(
            f"figure reference {{{{{inner}}}}} names category {category!r}, which {measure_id!r} at {cell_ref!r} never "
            f"recorded; its categories are {', '.join(sorted(summary.categories))}"
        )
    return format_number(summary.categories[category])


def render_text(surface: DecisionSurface, text: str, *, where: str) -> str:
    """Replace every figure reference in one piece of prose with its value.

    Args:
        surface: The analysis's decision surface.
        text: The prose as the model wrote it.
        where: Where it sits in the document, for the refusal.

    Returns:
        The prose with each reference rendered.

    Raises:
        UnresolvableReference: A reference in it is malformed or names nothing the surface holds.
    """

    def render(match: re.Match[str]) -> str:
        try:
            return render_reference(surface, match.group(1))
        except UnresolvableReference as unresolved:
            raise UnresolvableReference(f"{where}: {unresolved}") from unresolved

    return FIGURE_REFERENCE.sub(render, text)


def render_prose_figures(document: AuthoredAnalysis, surface: DecisionSurface) -> AuthoredAnalysis:
    """Render every figure reference in the document, or refuse the first that names nothing.

    Every string in the document is walked, not a list of prose fields: a list would silently skip
    the next free-text field someone adds. An id, a cell or a vocabulary word holds no reference,
    so rendering it is a no-op.

    Args:
        document: The validated authored document.
        surface: The decision surface its references resolve against.

    Returns:
        The document with every reference replaced by its value.

    Raises:
        UnresolvableReference: A reference is malformed or names nothing the surface holds.
    """

    def walk(value: object, where: str) -> object:
        if isinstance(value, str):
            return render_text(surface, value, where=where)
        if isinstance(value, dict):
            return {key: walk(item, f"{where}.{key}" if where else key) for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item, f"{where}[{index}]") for index, item in enumerate(value)]
        return value

    return AuthoredAnalysis.model_validate(walk(document.model_dump(mode="python"), ""))


#: The heading the grammar section opens with — the name the seed prompt points the writer at.
GRAMMAR_HEADING = "FIGURE REFERENCES"

#: What each reading names, as the grammar section says it; its keys are :data:`READINGS`.
_READING_SUBJECTS: dict[str, str] = {"measure": "a measure", "judged": "a judged dimension"}

#: The worked example's measure: a latency, because the example exists to show that the unit renders.
_EXAMPLE_MEASURE = "widget_elapsed_ms"

#: The worked example's cell count; it names the LAST cell, so an alias never reads as always ``c1``.
_EXAMPLE_CELLS = 4

#: A goal-state check on the synthetic surface, whose pass rate shows the count form a rate renders in.
_EXAMPLE_CHECK = goal_check_measure("widget_saved")


def _example_surface() -> DecisionSurface:
    """The synthetic surface the grammar section's examples are rendered on — no real campaign's figures."""
    measures = MeasureCollection(
        measures=[
            MeasureSummary(
                population="scored",
                name=_EXAMPLE_MEASURE,
                attribution_scope="end_to_end",
                higher_is_better=False,
                n=12,
                mean=1200.0,
                p95=1900.0,
            ),
            MeasureSummary(
                population="scored",
                name=_EXAMPLE_CHECK,
                attribution_scope="end_to_end",
                higher_is_better=True,
                n=12,
                mean=8 / 12,
            ),
        ]
    )
    cells = [
        CellFacts(
            variant_key=f"arm{index}", apparatus_class_id="rig", run_ids=["run"], n_observations=12, measures=measures
        )
        for index in range(1, _EXAMPLE_CELLS + 1)
    ]
    return DecisionSurface(
        cells=cells, measures={_EXAMPLE_MEASURE: MeasureFacts(unit="ms"), _EXAMPLE_CHECK: MeasureFacts()}
    )


def _backticked(names: tuple[str, ...]) -> str:
    """Name alternatives as a sentence does: ``a``, ``b`` or ``c``."""
    quoted = [f"`{name}`" for name in names]
    return quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} or {quoted[-1]}"


def reference_grammar() -> str:
    """Write the system-prompt section that teaches the figure-reference grammar, from the parser's own constants.

    Every part of it is read off the code that enforces it: the readings and each one's statistics
    from :data:`STATS_BY_READING`, the category form from :data:`CATEGORY_COUNT_PREFIX`, the cell
    form from :func:`~threetears.evals.analysis.references.cell_aliases`, and both worked examples from
    the renderer on :func:`_example_surface` — so a statistic added to the parser, or a change to how
    a figure is spelled, reaches the model with the code and never waits on a prompt promotion.

    It opens by saying it supersedes any other description of the grammar, because the prompt it is
    appended to is store-master and may predate it: a deployment whose prompt still teaches an
    older form (the pre-alias ``<variant_key>:<apparatus_class_id>`` cell, a statistic list without
    ``n_zero``) would otherwise hand the writer two contracts and leave it to guess which binds.
    Refusing that prompt is not an option (a retuned prompt's divergence is disclosed, never
    refused), and disclosing it only on a failed generation reports the cost after it is paid;
    saying here which description holds is what makes the stale text harmless.

    Returns:
        The section, heading first.

    Raises:
        UnresolvableReference: An example no longer renders — the grammar changed under this
            function, and every generation would share the break, so it fails loudly here.
    """
    surface = _example_surface()
    aliases = list(cell_aliases(surface.cells))
    sentence = f"the slowest took {{{{{aliases[-1]}|{_EXAMPLE_MEASURE}|measure|p95}}}}"
    pass_rate = render_reference(surface, f"{aliases[-1]}|{_EXAMPLE_CHECK}|measure|mean")
    statistics = " and ".join(
        f"{_backticked(STATS_BY_READING[reading])} for {_READING_SUBJECTS[reading]}" for reading in READINGS
    )
    return (
        f"{GRAMMAR_HEADING}. This section is written by the code that reads your references, so it is current: where "
        "anything above describes naming a cell or writing a reference differently, follow this section.\n"
        f"Name a cell by its `cell` ({', '.join(f'`{alias}`' for alias in aliases[:2])}, …) wherever you name one — in evidence, in a chart, "
        "in a decision and in a reference. "
        f"Where your prose needs a number, write a figure reference in its place: `{REFERENCE_FORM}`, where `<reading>` is "
        f"{_backticked(READINGS)}, and `<stat>` is {statistics}. "
        f'For example, "{sentence}" reads "{render_text(surface, sentence, where="the grammar example")}": '
        "the unit is part of what renders. "
        "A reference naming a cell, measure or statistic the table does not hold is refused and sent back naming what the "
        "cell does hold. "
        "A categorical measure has no point value, so it is never an evidence row: reference a category's count in prose "
        f"with the statistic `{CATEGORY_COUNT_PREFIX}<category>` (and its total with `n`), or draw it with a `breakdown` chart. "
        f'A check\'s pass rate renders as a count ("{pass_rate}"), so never write a unit or a total after a reference.'
    )


__all__ = [
    "FIGURE_REFERENCE",
    "GRAMMAR_HEADING",
    "JUDGED_STATS",
    "MEASURE_STATS",
    "REFERENCE_FORM",
    "parse_reference",
    "prose_number",
    "reference_grammar",
    "render_prose_figures",
    "render_reference",
    "render_text",
    "state_in_prose",
]
