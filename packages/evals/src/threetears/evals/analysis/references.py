"""Resolve the model's references against the decision surface — the one place a number is filled.

The generating model names WHERE a number lives — a cell and a measure (or a judged dimension) —
and this module reads the figure off the analysis's :class:`~threetears.evals.contracts.surface.DecisionSurface`.
Every code-filled number on an analysis goes through :func:`resolve_reading`: evidence rows and
the chart compilers in :mod:`threetears.evals.analysis.viz_refs`. One
resolver rather than one per consumer, because two lookups of "the value of this measure at this
cell" are two answers to one question, and the input that separates them is the one nobody wrote a
fixture for.

A reference that names nothing the surface holds is an :class:`UnresolvableReference`, which the
generator's repair round feeds back — the message names what the cell DOES hold, so the model can
pick a real reading instead of guessing again.

**The writer names a cell by a short alias** (``c1``, ``c2``, …) that :func:`cell_aliases` mints and
:func:`cell_of_alias` reads back — one mapping for evidence rows, charts, decisions and prose alike.
The generator translates every alias to the cell's full ``cell_ref`` before resolving or storing
anything, so everything below this seam, and every stored analysis, carries the full identity.

**The point estimate is the cell's MEAN**, over its non-faulted observations — the population every
bar is adjudicated over, so a reading and a bar verdict on one cell describe the same observations.
The interval is t-based on ``n``; where ``n_cases`` is below ``n`` the observations are clustered
and the interval is narrower than the clustering supports, which :attr:`ResolvedReading.dispersion`
says in a short clause (``N obs over M cases, interval too narrow``) rather than leaving to the reader.

**One reading is a** :class:`ReadingRef` — a name and the namespace it is in, stated rather than
inferred — wherever code reads one off an authored evidence row or chart
(``viz_refs.reference_from_chart``). The authored models have already been validated strictly, so a
reading carries no policy of its own: what it names either resolves here or is refused here.

**Every interval here is computed and labelled at one level.** A measure's arrives already computed
on its summary, a judged dimension's is computed here, and both widths come from
:func:`threetears.evals.analysis.stats.ci_half_width` at :data:`threetears.evals.analysis.stats.INTERVAL_LEVEL` — the same
constant the dispersion text and every chart caption state. The level is not a second decision this
module makes; it reads the one the width was computed at.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.analysis import stats
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.numbers import format_number
from threetears.evals.contracts.analysis_measures import MeasureSummary
from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.contracts.campaign import ReadingKind
from threetears.evals.contracts.evidence_tiers import JudgedEvidenceTier
from threetears.evals.contracts.metrics import MeritAxis, classifier_label_of
from threetears.evals.contracts.surface import (
    CellFacts,
    DecisionSurface,
    JudgedDimensionFacts,
    JudgedReading,
    MeasureFacts,
)


class ResolvedReading(BaseModel):
    """One reading at one cell, as code resolved it — everything a consumer may need to draw or state it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cell_ref: str
    measure_id: str
    reading: ReadingKind
    mean: float
    n: int
    n_cases: int | None
    sem: float | None
    ci_low: float | None
    ci_high: float | None
    unit: str | None
    higher_is_better: bool | None
    merit_axis: MeritAxis | None
    dispersion: str
    #: A judged reading's evidence tier, as the surface froze it for the cell's judges; None for a measure.
    judged_tier: JudgedEvidenceTier | None


class ReadingRef(EvalBaseModel):
    """One reading: a measure, or a judged dimension, named in its namespace."""

    measure_id: str = Field(description="The measure (or judged dimension) — spelled as the surface spells it.")
    reading: ReadingKind = Field(
        description="Which of a cell's two namespaces the name is in — stated, never inferred."
    )


#: What starts a cell alias: the writer names cells ``c1``, ``c2``, … (:func:`cell_aliases`).
CELL_ALIAS_PREFIX = "c"


def cell_aliases(cells: Iterable[CellFacts]) -> dict[str, str]:
    """Mint the short alias a writer names each cell by — the one mint, read by the bundle view and the resolver.

    A cell's identity is two 64-hex digests, which tokenize poorly and which a writer has to copy
    into every reading it names; each copy costs output tokens and is a place to mistype a digest,
    and a mistype buys the paid repair round. So the writer is shown ``c1``, ``c2``, … and
    code maps each back through :func:`cell_of_alias`.

    Numbered in the order given, one alias per distinct cell. Both callers pass the same list —
    the bundle's ``cell_measures``, which the decision surface's ``cells`` is a copy of — so the
    alias the writer saw and the one the resolver reads are equal by construction. An alias is
    meaningful only against the bundle it was minted from, which is why the generator translates
    every alias to the full ``cell_ref`` before anything is stored.

    Args:
        cells: The cells, in the order to number them.

    Returns:
        ``alias -> cell_ref``, in minted order.
    """
    aliases: dict[str, str] = {}
    seen: set[str] = set()
    for cell in cells:
        ref = cell_ref(cell.variant_key, cell.apparatus_class_id)
        if ref not in seen:
            seen.add(ref)
            aliases[f"{CELL_ALIAS_PREFIX}{len(aliases) + 1}"] = ref
    return aliases


def cell_of_alias(surface: DecisionSurface, alias: str) -> str:
    """Resolve the alias a writer named a cell by to the cell's ``cell_ref``, or refuse it.

    The one place an alias is read: evidence rows, charts, decisions and figure references in prose
    all come through here.

    Args:
        surface: The analysis's decision surface.
        alias: The cell as the writer named it.

    Returns:
        The cell's full ``cell_ref``.

    Raises:
        UnresolvableReference: No cell carries that alias.
    """
    aliases = cell_aliases(surface.cells)
    ref = aliases.get(alias)
    if ref is None:
        known = ", ".join(aliases) or "(none — the surface holds no cell)"
        raise UnresolvableReference(
            f"reference names cell {alias!r}, which is no cell's alias; a cell is named by its `cell` in "
            f"`cell_measures` — the cells are: {known}"
        )
    return ref


def in_writer_terms(text: str, surface: DecisionSurface) -> str:
    """Name every cell in a refusal the way the writer names it — by alias, never by its full ``cell_ref``.

    Everything below the alias seam speaks full refs, so a refusal raised there (a measure a real cell
    does not carry, a statistic it lacks) quotes the digest pair. That text is sent back on the repair
    round, and a writer shown the handle it was never taught to use can copy it — which is refused. So
    the generator passes a refusal through here where it hands it to the writer — the repair round and
    the provenance of the analysis that round produced — and nowhere else: the tally and the attempt
    record keep the full refs, since an alias is meaningful only against the bundle it was minted from.

    Args:
        text: The refusal as the resolver worded it.
        surface: The surface the aliases were minted from.

    Returns:
        The text with each full ``cell_ref`` replaced by its alias.
    """
    for alias, ref in cell_aliases(surface.cells).items():
        text = text.replace(ref, alias)
    return text


def cell_index(surface: DecisionSurface) -> dict[str, CellFacts]:
    """Key a surface's cells by the ``cell_ref`` a reference carries.

    Args:
        surface: The analysis's decision surface.

    Returns:
        ``cell_ref -> CellFacts`` for every cell.
    """
    return {cell_ref(cell.variant_key, cell.apparatus_class_id): cell for cell in surface.cells}


def require_cell(surface: DecisionSurface, ref: str) -> CellFacts:
    """Return the cell a reference names, or refuse naming the cells that exist.

    Args:
        surface: The analysis's decision surface.
        ref: The reference's cell (a ``cell_ref``).

    Returns:
        The cell's facts.

    Raises:
        UnresolvableReference: No cell carries that ref.
    """
    cells = cell_index(surface)
    cell = cells.get(ref)
    if cell is None:
        known = ", ".join(sorted(cells)) or "(none — the surface holds no cell)"
        raise UnresolvableReference(
            f"reference names cell {ref!r}, which the decision surface does not hold — the cells are: {known}"
        )
    return cell


def resolve_reading(
    surface: DecisionSurface, ref: str, measure_id: str, reading: ReadingKind = "measure"
) -> ResolvedReading:
    """Read one numeric reading off the surface.

    Args:
        surface: The analysis's decision surface.
        ref: The cell the reference names (a ``cell_ref``).
        measure_id: The measure (or judged dimension) it names.
        reading: Which of the cell's two namespaces to read — already a :data:`ReadingKind`, because
            every model-authored kind is validated against the authored models before it gets here.

    Returns:
        The resolved reading.

    Raises:
        UnresolvableReference: The cell, or the reading in it, does not exist; the measure is
            categorical (it has no point estimate); or it has no mean (nothing was scored).
        ValueError: ``reading`` is neither kind — a caller that bypassed the authored models,
            which no repair round can correct.
        RuntimeError: The surface holds a reading with no facts entry for its name — see
            :func:`_measure_facts`.
    """
    cell = require_cell(surface, ref)
    if reading == "judged":
        return _resolve_judged(surface, cell, ref, measure_id)
    if reading == "measure":
        return _resolve_measure(surface, cell, ref, measure_id)
    raise ValueError(
        f"reading {reading!r} is neither 'measure' nor 'judged' — validate model output against the authored models"
    )


def _resolve_measure(surface: DecisionSurface, cell: CellFacts, ref: str, measure_id: str) -> ResolvedReading:
    """Resolve a ``measure`` reading.

    Raises:
        UnresolvableReference: See :func:`resolve_reading`.
    """
    summary = _find_measure(cell, measure_id)
    if summary is None:
        judged_hint = (
            f" `{measure_id}` IS a judged dimension in this cell — set reading to 'judged'."
            if any(j.dimension == measure_id for j in cell.judged)
            else ""
        )
        raise UnresolvableReference(
            f"reference names measure {measure_id!r} at cell {ref!r}, which measured no such measure.{judged_hint} "
            f"Its measures are: {', '.join(m.name for m in cell.measures.measures) or '(none)'}"
        )
    if summary.categories:
        raise UnresolvableReference(
            f"reference names measure {measure_id!r} at cell {ref!r} as a point estimate, but it is categorical and "
            f"has none — its evidence is its category counts, which only a `breakdown` chart draws; in prose, "
            f"reference one category's count as `{{{{<cell>|{measure_id}|measure|count:<category>}}}}`"
        )
    if summary.texts:
        raise UnresolvableReference(
            f"reference names measure {measure_id!r} at cell {ref!r} as a point estimate, but it is text — its "
            "observations are evidence to quote, listed whole, and no number stands for what was said"
        )
    # A boolean measure's point estimate is its rate, bounded by the Wilson interval its summary carries.
    point = summary.rate if summary.rate is not None else summary.mean
    if point is None:
        raise UnresolvableReference(f"reference names measure {measure_id!r} at cell {ref!r}, which has no mean (n=0)")
    facts = _measure_facts(surface, measure_id)
    return ResolvedReading(
        cell_ref=ref,
        measure_id=measure_id,
        reading="measure",
        mean=point,
        n=summary.n,
        n_cases=summary.n_independent or None,
        sem=summary.sem,
        ci_low=summary.ci_low,
        ci_high=summary.ci_high,
        unit=facts.unit,
        higher_is_better=summary.higher_is_better,
        merit_axis=facts.merit_axis,
        dispersion=(
            # F1 is a function of one confusion matrix: there is no spread to estimate at any n, so "unestimable at
            # n=…" would imply more data could supply one.
            "none by construction: one value computed from the cell's confusion counts"
            if (label := classifier_label_of(measure_id)) is not None and label[0] == "f1"
            else _dispersion(summary.sem, summary.ci_low, summary.ci_high, summary.n, summary.n_independent or None)
        ),
        judged_tier=None,
    )


def _resolve_judged(surface: DecisionSurface, cell: CellFacts, ref: str, dimension: str) -> ResolvedReading:
    """Resolve a ``judged`` reading, computing its interval the way a measure's is computed.

    Raises:
        UnresolvableReference: See :func:`resolve_reading`.
    """
    judged = _find_judged(cell, dimension)
    if judged is None:
        measure_hint = (
            f" `{dimension}` IS a measure in this cell — set reading to 'measure'."
            if _find_measure(cell, dimension) is not None
            else ""
        )
        raise UnresolvableReference(
            f"reference names judged dimension {dimension!r} at cell {ref!r}, which scored no such dimension."
            f"{measure_hint} Its judged dimensions are: {', '.join(j.dimension for j in cell.judged) or '(none)'}"
        )
    if judged.mean is None:
        raise UnresolvableReference(
            f"reference names judged dimension {dimension!r} at cell {ref!r}, which has no scores"
        )
    # The same width function the bundle computes every measure's interval with, so a judged
    # interval and a measure's drawn side by side are at one level by construction.
    half = None if judged.sem is None else stats.ci_half_width(judged.sem, judged.n)
    ci_low, ci_high = (None, None) if half is None else (judged.mean - half, judged.mean + half)
    facts = _dimension_facts(surface, dimension)
    return ResolvedReading(
        cell_ref=ref,
        measure_id=dimension,
        reading="judged",
        mean=judged.mean,
        n=judged.n,
        n_cases=judged.n_independent or None,
        sem=judged.sem,
        ci_low=ci_low,
        ci_high=ci_high,
        unit=None,
        higher_is_better=facts.higher_is_better,
        merit_axis="quality",
        dispersion=_dispersion(judged.sem, ci_low, ci_high, judged.n, judged.n_independent or None),
        judged_tier=judged.evidence_tier,
    )


def _measure_facts(surface: DecisionSurface, measure_id: str) -> MeasureFacts:
    """The surface's facts for a measure a cell carries — which the surface guarantees exist.

    A missing entry is a surface built inconsistently, never something the model wrote: the unit,
    axis and direction a figure is stated with are unknown, and guessing them would call an
    unknown-polarity measure one thing or another. So it is a hard error rather than a repairable
    refusal — a repair round would regenerate against the same surface and meet the same gap.

    Raises:
        RuntimeError: The surface holds no entry for ``measure_id``.
    """
    facts = surface.measures.get(measure_id)
    if facts is None:
        raise RuntimeError(
            f"the decision surface carries measure {measure_id!r} in a cell and no `measures` entry for it — every "
            "measure a cell carries has one, so this surface was built inconsistently"
        )
    return facts


def _dimension_facts(surface: DecisionSurface, dimension: str) -> JudgedDimensionFacts:
    """The surface's facts for a judged dimension a cell scored — which the surface guarantees exist.

    Raises:
        RuntimeError: The surface holds no entry for ``dimension``; see :func:`_measure_facts`.
    """
    facts = surface.dimensions.get(dimension)
    if facts is None:
        raise RuntimeError(
            f"the decision surface carries judged dimension {dimension!r} in a cell and no `dimensions` entry for it — "
            "every dimension a cell scored has one, so this surface was built inconsistently"
        )
    return facts


def _find_measure(cell: CellFacts, name: str) -> MeasureSummary | None:
    return next((m for m in cell.measures.measures if m.name == name), None)


def _find_judged(cell: CellFacts, dimension: str) -> JudgedReading | None:
    return next((j for j in cell.judged if j.dimension == dimension), None)


def _dispersion(sem: float | None, ci_low: float | None, ci_high: float | None, n: int, n_cases: int | None) -> str:
    """State a reading's spread in words, including when there is none to state.

    Returns:
        The dispersion text an evidence row carries.
    """
    if ci_low is None or ci_high is None:
        return f"unestimable at n={n}"
    # A rate carries its Wilson interval and no standard error, and its interval is its spread: stated, never
    # reported as unestimable because the sem it does not have is absent.
    text = f"{stats.INTERVAL_LEVEL:.0%} CI [{format_number(ci_low)}, {format_number(ci_high)}]"
    if sem is not None:
        text = f"sem {format_number(sem)}; {text}"
    if n_cases is not None and n_cases < n:
        # Short on purpose: it rides on every reading of a clustered cell, in every table a reader scans.
        text += f"; {n} obs over {n_cases} cases, interval too narrow"
    return text


__all__ = [
    "CELL_ALIAS_PREFIX",
    "ReadingRef",
    "ResolvedReading",
    "cell_aliases",
    "cell_index",
    "cell_of_alias",
    "in_writer_terms",
    "require_cell",
    "resolve_reading",
]
