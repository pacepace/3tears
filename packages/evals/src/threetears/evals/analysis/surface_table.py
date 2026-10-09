"""The decision surface's table — laid out once, here, and served to every surface that shows it.

:mod:`threetears.evals.contracts.surface` freezes the FACTS: one entry per cell, the bars adjudicated
against them, and what each measure is. How those facts become a table — the row order, which
measures stand in as the cost and latency columns, the one unit each column is stated in, which
verdict sits under which bar, the replication sentence, the run notes — is a layout, and a layout
decided twice is two layouts. It was: the MCP render and the browser each derived it, and the
browser needed two hand-kept mirrors of server code to do so. So it is derived here, laid into the
analysis's :class:`~threetears.evals.analysis.report.Report` (:func:`~threetears.evals.analysis.service.analysis_report`),
and every surface renders what it is served — the pattern :mod:`threetears.evals.analysis.arms` set
for the arm table.

**Persisted: the facts. Derived: the table.** Computed on every read and never written back, so a
stored analysis carries no layout a later reader would have to un-decide.

**What a surface still does itself: style the row and mark the control.** A row's name is served
as ``label``, computed by :func:`~threetears.evals.analysis.arms.arm_label` over the analysis's
:func:`~threetears.evals.analysis.arms.arm_names` — the labeller the arm
table's rows are served through — so the page and the MCP render print one spelling rather than
each re-spelling the arm, and a surface whose two tables named one arm two ways would tell its
reader they were two arms.

**The words are served for the same reason.** The provenance sentence above the table and the word
under each bar's verdict are fields of the table (``provenance``, ``verdict_word``). If each render
kept its own copy, a rewording in one would leave the two surfaces saying different things about the
same numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Self

from pydantic import Field, computed_field, model_validator

from threetears.evals.analysis.arms import (
    ArmLevel,
    arm_label,
    arm_names,
    distinguishing_axes,
    multi_rig_variants,
    naming_levels,
    short_digest,
)
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.viz.quantities import display_scale
from threetears.evals.contracts.analysis_measures import BarAdjudication, BarVerdict, MeasureSummary
from threetears.evals.contracts.campaign import EvalAnalysis, VariantIndexEntry
from threetears.evals.contracts.metrics import MeritAxis
from threetears.evals.contracts.base import EvalDocumentModel
from threetears.evals.contracts.surface import CellFacts, DecisionSurface, all_failed_sentence

#: Said in place of the table when the surface froze no cell.
NO_CELLS = "No cells — the campaign measured nothing this analysis could freeze."

#: The merit axes a surface stands in as columns, in column order. Read off each measure's FROZEN
#: axis rather than a list of names, so a campaign measuring cost under a name no one anticipated
#: still gets its column, and a registry edit after generation cannot move one.
_SURFACE_AXES: tuple[MeritAxis, ...] = ("cost", "latency")

#: Where a table stands. Two states rather than an optional table, because "the surface froze no
#: cell" is a fact with its own sentence, and is not an empty table.
SurfaceState = Literal["no_cells", "measured"]

#: A bar's verdict on one cell, as a word — never a colour alone. ``no_data`` is the verdict the
#: server wrote for a cell that carried no observation of the bar's measure, which is not a miss.
SurfaceVerdict = Literal["clears", "misses", "no_data"]

_VERDICT_OF: dict[bool | None, SurfaceVerdict] = {True: "clears", False: "misses", None: "no_data"}

#: Each verdict as the word a reader acts on — the one spelling every surface prints. Served on the
#: value as ``verdict_word`` so no render keeps its own copy of the vocabulary.
VERDICT_WORDS: dict[SurfaceVerdict, str] = {"clears": "clears", "misses": "misses", "no_data": "no data"}

#: Said under a cost or latency column — and under a bar that read nothing — for a cell whose every result the
#: candidate failed. Not a blank: a blank reads as "not measured", and this arm was measured — every result
#: failed, which its rates and bars count. Never a number: a failed call's round trip and its empty spend are
#: what read as a fast, free arm.
NO_SUCCESSFUL_RESULTS = "no successful results"

#: The sentence stated above the table: where its numbers came from. Served as ``provenance`` on every
#: table that states numbers, so the surfaces showing it print one sentence rather than each keeping a
#: copy.
SURFACE_PROVENANCE = "Computed from the campaign's cells by code — no number here was written by the model."


class SurfaceColumn(EvalDocumentModel):
    """One numeric column of the table — an adjudicated bar, or a cost or latency measure.

    The unit is chosen ONCE for the column, over every quantity it states, and every value under it
    is already restated in that unit: a column reading 900 ms above 1.2 s is two rulers read as one.
    """

    kind: Literal["bar", "merit"] = Field(
        description=(
            "`bar` — an adjudicated bar, one verdict per cell. `merit` — a cost or latency measure's mean per cell, "
            "with its spread and sample."
        )
    )
    measure_id: str = Field(min_length=1, description="The measure the column is read on.")
    unit: str = Field(
        default="",
        description=(
            "The unit every number in the column is stated in, after restatement. Empty when the measure "
            "declares none — restating an unknown unit would be inventing one."
        ),
    )
    header: str = Field(min_length=1, description="The column's header text, spelled once for every surface.")
    direction: Literal["higher_is_better", "lower_is_better"] | None = Field(
        default=None, description="Which way clearing runs. Set on a bar column, None on a merit column."
    )
    source: Literal["declared", "registered"] | None = Field(
        default=None, description="Whose bar it is. Set on a bar column, None on a merit column."
    )
    threshold: float | None = Field(
        default=None,
        description="The bar's threshold, ALREADY restated in `unit`. Set on a bar column, None on a merit column.",
    )
    axis: MeritAxis | None = Field(
        default=None, description="The measure's frozen merit axis. Set on a merit column, None on a bar column."
    )

    @model_validator(mode="after")
    def _facts_match_kind(self) -> Self:
        """Refuse a column carrying the other kind's facts, or missing its own.

        Raises:
            ValueError: If a bar column lacks a direction, source or threshold or carries an axis,
                or a merit column lacks an axis or carries any bar fact.
        """
        bar_facts = (self.direction, self.source, self.threshold)
        if self.kind == "bar" and (any(fact is None for fact in bar_facts) or self.axis is not None):
            raise ValueError(f"bar column {self.measure_id!r} needs direction, source and threshold, and no axis")
        if self.kind == "merit" and (self.axis is None or any(fact is not None for fact in bar_facts)):
            raise ValueError(f"merit column {self.measure_id!r} needs an axis and no bar facts")
        return self


class SurfaceValue(EvalDocumentModel):
    """One cell's number in one column, already in the column's unit."""

    value: float | None = Field(
        default=None,
        description=(
            "The number, in the column's unit. None under a bar whose verdict read no observation, and under a cost "
            "or latency column for a cell whose every result failed — whose `text` then says so."
        ),
    )
    sem: float | None = Field(default=None, description="Its standard error, in the column's unit. None below n=2.")
    n: int = Field(ge=0, description="Observations behind the value.")
    verdict: SurfaceVerdict | None = Field(
        default=None, description="The bar's verdict on this cell. Set under a bar column, None under a merit column."
    )
    text: str = Field(
        min_length=1,
        description=(
            "The value as a reader sees it, without the verdict word: `value ± sem (n=N)` under every column, bar or "
            "cost or latency, the `±` only where a sem exists — or `no successful results` where the cell's every "
            "result failed and there is no value. Spelled once here so the surfaces cannot disagree on a number."
        ),
    )

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description=(
            "The verdict as the word a reader acts on — `clears`, `misses` or `no data` — never a colour alone. "
            "Set exactly when `verdict` is, so every surface prints one word for one verdict."
        )
    )
    @property
    def verdict_word(self) -> str | None:
        """The verdict's word, from :data:`VERDICT_WORDS` — None under a merit column."""
        return None if self.verdict is None else VERDICT_WORDS[self.verdict]


class SurfaceRunNote(EvalDocumentModel):
    """One member run of a cell that ran short of its design, or did not complete."""

    kind: Literal["short", "incomplete"] = Field(description="Which note — the word a reader meets first.")
    run_id: str = Field(min_length=1, description="The run the note is about.")
    text: str = Field(
        min_length=1,
        description=(
            "What happened: the bundle's sentence for a short run, `status <status>` for an incomplete one — a "
            "status is all an incomplete run's record holds, so it is labelled a status rather than dressed as a reason."
        ),
    )


class SurfaceRow(EvalDocumentModel):
    """One measured cell — one arm under one rig — laid out against the table's columns."""

    variant_key: str = Field(min_length=1, description="The arm's variant — its key into the analysis's variant index.")
    apparatus_class_id: str = Field(
        min_length=1, description="The rig it was measured under — the other half of its cell."
    )
    is_control: bool = Field(default=False, description="Whether this is the campaign's declared control.")
    placed: bool = Field(
        description=(
            "Whether the analysis's variant index holds this variant. False means nothing says what the arm ran, "
            "and the row must read as unplaced — a label borrowed from a neighbour would put these numbers "
            "under levels the arm never ran."
        )
    )
    label: str = Field(
        min_length=1,
        description=(
            "What this row is called — `arm_label` over the analysis's `arm_names` and this row's `rig`, the arm "
            "table's labeller, so one arm reads alike in both tables and no two rows read alike. Without the "
            "control marker, which each surface spells for itself."
        ),
    )
    levels: list[ArmLevel] = Field(
        default_factory=list,
        description=(
            "What the arm is NAMED by, axis by axis, exactly as the arm table's row for it carries them (that "
            "row's `settings` holds everything it ran). Empty when unplaced, or when the index cannot describe "
            "the arm (`levels_unavailable`)."
        ),
    )
    levels_unavailable: str | None = Field(
        default=None, description="Why the index cannot describe this arm, when it says so. As on the arm table's row."
    )
    rig: str | None = Field(
        default=None,
        description=(
            "The rig's id, cut to what tells it apart, when this variant was measured under more than one rig — "
            "one stack under two rigs is two rows with one name. None when the variant has one cell."
        ),
    )
    replication: str = Field(
        min_length=1,
        description=(
            "How many observations the cell pooled and how they were drawn — cases lead, since they are the draws — "
            "then how many the harness faulted and how many the candidate failed, when any."
        ),
    )
    all_failed: bool = Field(
        default=False,
        description=(
            "Whether the candidate failed every result of the cell the harness did not fault. Its cost and latency "
            "columns then read `no successful results`, never a number, and the table's `all_failed_disclosure` "
            "names it."
        ),
    )
    flags: list[Literal["short", "incomplete"]] = Field(
        default_factory=list,
        description="Which run notes the cell carries, in that order — the row's at-a-glance column.",
    )
    run_notes: list[SurfaceRunNote] = Field(
        default_factory=list, description="Every note behind those flags: short runs by run id, then incomplete ones."
    )
    values: list[SurfaceValue | None] = Field(
        default_factory=list,
        description=(
            "One entry per column, in column order. None where the cell has nothing to state there: a cost or "
            "latency measure it did not carry, or a bar the server wrote no verdict on for it. A cell whose every "
            "result failed states `no successful results` under a cost or latency column instead of nothing."
        ),
    )


class SurfaceUnadjudicatedBar(EvalDocumentModel):
    """A bar no cell could be read against — listed below the table, never drawn as a column of misses."""

    measure_id: str = Field(min_length=1, description="What the bar names.")
    source: Literal["declared", "registered"] = Field(description="Whose bar it is.")
    state: Literal["names_no_stored_measure", "not_numeric"] = Field(description="Why no verdict exists, as a state.")
    reason: str = Field(
        min_length=1,
        description="Why no verdict exists, in words — the adjudication's reason, or the state's name when it gave none.",
    )


class SurfaceTable(EvalDocumentModel):
    """The decision surface laid out — derived on every read, never stored."""

    state: SurfaceState = Field(
        description=(
            "`no_cells` — the surface froze no cell; `disclosure` stands in for the table, and any unadjudicated "
            "bar is still listed. `measured` — the table."
        )
    )
    disclosure: str | None = Field(
        default=None,
        description="The sentence shown in place of the table. Set exactly when `state` is not `measured`.",
    )
    columns: list[SurfaceColumn] = Field(
        default_factory=list,
        description="The numeric columns, in order: every adjudicated bar, then cost, then latency measures by name.",
    )
    rows: list[SurfaceRow] = Field(
        default_factory=list,
        description="One per cell: the control's cells first, then every other by (variant_key, apparatus_class_id).",
    )
    unadjudicated_bars: list[SurfaceUnadjudicatedBar] = Field(
        default_factory=list, description="Every bar with no verdict, in the surface's bar order."
    )
    all_failed_disclosure: str | None = Field(
        default=None,
        description=(
            "The sentence stated beside the table when some row's every result failed — that there is no cost or "
            "latency to read there, and the failures count against the arm — naming those rows' arms unless every "
            "row failed. Set exactly when some row is `all_failed`."
        ),
    )

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description="The sentence stated above the table: its numbers were computed by code, none written by the model."
    )
    @property
    def provenance(self) -> str:
        """:data:`SURFACE_PROVENANCE` — every analysis carries the surface it was generated from."""
        return SURFACE_PROVENANCE

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Refuse a table whose state and contents disagree, or whose rows do not fit its columns.

        Raises:
            ValueError: On a disclosure present exactly when it should not be, rows under a state that
                has none (or none under ``measured``), an all-failed disclosure without an all-failed row (or
                an all-failed row without one),
                a row whose values do not line up one-to-one with the columns, or a verdict under
                a column that is not a bar (or a bar value without one).
        """
        if (self.disclosure is None) != (self.state == "measured"):
            raise ValueError(f"a {self.state} surface table carries a disclosure exactly when it is not `measured`")
        if (self.state == "measured") != bool(self.rows):
            raise ValueError(f"a {self.state} surface table carries rows exactly when it is `measured`")
        if (self.all_failed_disclosure is None) == any(row.all_failed for row in self.rows):
            raise ValueError("a surface table carries an all-failed disclosure exactly when some row is all_failed")
        for row in self.rows:
            where = f"row {short_digest(row.variant_key)}@{short_digest(row.apparatus_class_id)}"
            if len(row.values) != len(self.columns):
                raise ValueError(f"{where} states {len(row.values)} value(s) under {len(self.columns)} column(s)")
            for column, value in zip(self.columns, row.values, strict=True):
                if value is not None and (value.verdict is not None) != (column.kind == "bar"):
                    raise ValueError(
                        f"{where} carries a verdict exactly where its column is not a bar ({column.measure_id!r})"
                    )
        return self


def _flag_sources(cell: CellFacts) -> tuple[tuple[Literal["short", "incomplete"], dict[str, str]], ...]:
    """Each flag a cell can carry, beside the runs that would raise it.

    Args:
        cell: The cell.

    Returns:
        ``(flag, runs)`` per flag, in the order the row lists them.
    """
    return (("short", cell.short_runs), ("incomplete", cell.incomplete_runs))


def _replication(cell: CellFacts) -> str:
    """How many observations a cell pooled, and how they were drawn.

    The case count leads the repeats because cases are the independent draws: 12 observations over
    4 cases × 3 is a sample of four, and a reader shown only the 12 overstates it threefold.

    Args:
        cell: The cell.

    Returns:
        ``{n} obs over {cases} cases × {repeats}``, with the faulted count beside it when any, and the
        failed count after that — the observations a cost or latency column did not read, and the ones
        every rate and bar counted against the arm.
    """
    if cell.n_cases is None:
        text = f"{cell.n_observations} obs, cases unrecorded"
    else:
        low, high = cell.repeats_per_case_min, cell.repeats_per_case_max
        if low is None or high is None:
            repeats = ""
        elif low == high:
            repeats = f" × {low}"
        else:
            repeats = f" × {low}–{high}"
        text = f"{cell.n_observations} obs over {cell.n_cases} cases{repeats}"
    if cell.n_infra_excluded:
        text += f", {cell.n_infra_excluded} faulted, excluded"
    if cell.n_candidate_failed:
        text += f", {cell.n_candidate_failed} failed by the candidate"
    return text


def _run_notes(cell: CellFacts) -> list[SurfaceRunNote]:
    """The sentences behind a cell's flags — short runs by id, then incomplete runs by id."""
    return [
        *(SurfaceRunNote(kind="short", run_id=run, text=sentence) for run, sentence in sorted(cell.short_runs.items())),
        *(
            SurfaceRunNote(kind="incomplete", run_id=run, text=f"status {status}")
            for run, status in sorted(cell.incomplete_runs.items())
        ),
    ]


def _verdict_of(bar: BarAdjudication, cell: CellFacts) -> BarVerdict | None:
    """The bar's verdict on this cell, or None when the adjudication carries none for it."""
    return next(
        (
            v
            for v in bar.verdicts
            if (v.variant_key, v.apparatus_class_id) == (cell.variant_key, cell.apparatus_class_id)
        ),
        None,
    )


def _summary_of(cell: CellFacts, name: str) -> MeasureSummary | None:
    """The cell's summary of one measure, or None when the cell does not carry it."""
    return next((m for m in cell.measures.measures if m.name == name), None)


def _scaled(value: float | None, factor: float) -> float | None:
    """A value restated by a column's factor — absence stays absence, never a zero."""
    return None if value is None else value * factor


def _value_text(value: float | None, sem: float | None, n: int) -> str:
    """How every column spells one cell's number: ``value ± sem (n=N)``, the ``±`` only when a spread exists.

    One spelling for a bar and a cost or latency column alike, so a mean never appears without the
    spread and sample a reader needs to weigh it, whichever kind of column it sits in.
    """
    spread = f" ± {format_number(sem)}" if sem is not None else ""
    return f"{format_number(value)}{spread} (n={n})"


def _bar_column(bar: BarAdjudication, surface: DecisionSurface, cells: list[CellFacts]) -> tuple[SurfaceColumn, float]:
    """One adjudicated bar's column, and the factor its values are restated by.

    The ruler is chosen over the threshold and every verdict value the column states, never per
    value — the threshold is a quantity in the same column, and a header in seconds over cells in
    milliseconds is the two-ruler defect moved into the header.
    """
    facts = surface.measures.get(bar.measure_id)
    values = [bar.threshold] + [
        v.value for cell in cells if (v := _verdict_of(bar, cell)) is not None and v.value is not None
    ]
    factor, unit = display_scale(values, facts.unit if facts else None)
    threshold = bar.threshold * factor
    sign = "≥" if bar.direction == "higher_is_better" else "≤"
    registered = " (registered)" if bar.source == "registered" else ""
    header = f"{bar.measure_id} {sign} {f'{format_number(threshold)} {unit}'.rstrip()}{registered}"
    column = SurfaceColumn(
        kind="bar",
        measure_id=bar.measure_id,
        unit=unit,
        header=header,
        direction=bar.direction,
        source=bar.source,
        threshold=threshold,
    )
    return column, factor


def _bar_value(bar: BarAdjudication, cell: CellFacts, factor: float) -> SurfaceValue | None:
    """One cell's verdict under one bar, restated — None when the server wrote no verdict for it.

    None rather than ``no_data``: the server writes one verdict per cell on an adjudicated bar, so a
    missing one is a join that did not meet, and ``no_data`` is the server's own word for a cell
    that carried no observation — a claim this is not known to be.
    """
    verdict = _verdict_of(bar, cell)
    if verdict is None:
        return None
    value, sem = _scaled(verdict.value, factor), _scaled(verdict.sem, factor)
    # A bar that read nothing on a cell whose every result failed read nothing because nothing was delivered —
    # a latency bar's measure leaves the failures out — and says that rather than a blank number.
    text = NO_SUCCESSFUL_RESULTS if value is None and cell.all_failed else _value_text(value, sem, verdict.n)
    return SurfaceValue(value=value, sem=sem, n=verdict.n, verdict=_VERDICT_OF[verdict.cleared], text=text)


def _merit_columns(surface: DecisionSurface, cells: list[CellFacts]) -> list[tuple[SurfaceColumn, float]]:
    """The cost and latency columns — by FROZEN axis, only for a measure some cell carries."""
    present = {m.name for cell in cells for m in cell.measures.measures}
    columns = []
    for axis in _SURFACE_AXES:
        for name, facts in sorted(surface.measures.items()):
            if facts.merit_axis != axis or name not in present:
                continue
            means = [s.mean for cell in cells if (s := _summary_of(cell, name)) is not None and s.mean is not None]
            factor, unit = display_scale(means, facts.unit)
            header = f"{name} ({unit})" if unit else name
            columns.append((SurfaceColumn(kind="merit", measure_id=name, unit=unit, header=header, axis=axis), factor))
    return columns


def _merit_value(cell: CellFacts, name: str, factor: float) -> SurfaceValue | None:
    """One cell's mean of a cost or latency measure, with its spread and sample, restated — None where the cell lacks it.

    The mean, its sem and its n are read off ONE summary, so the spread and sample a reader is shown
    are the ones behind the mean beside them.

    **A cell whose every result failed states that, never a blank.** Its cost and latency are read over
    the results the candidate delivered, of which it has none, so it carries no summary to state — and a
    blank beside its neighbours' figures reads as "not measured", when the arm was measured and failed.
    """
    summary = _summary_of(cell, name)
    if summary is None or summary.mean is None:
        return SurfaceValue(n=0, text=NO_SUCCESSFUL_RESULTS) if cell.all_failed else None
    mean, sem = summary.mean * factor, _scaled(summary.sem, factor)
    return SurfaceValue(value=mean, sem=sem, n=summary.n, text=_value_text(mean, sem, summary.n))


def build_surface_table(analysis: EvalAnalysis) -> SurfaceTable:
    """Lay out one stored analysis's decision surface.

    Args:
        analysis: The analysis to read. Nothing is written back to it.

    Returns:
        The table, as :func:`surface_table_of` lays out the analysis's frozen surface and variant index.
    """
    return surface_table_of(analysis.decision_surface, analysis.variant_index)


def surface_table_of(surface: DecisionSurface, variant_index: Sequence[VariantIndexEntry]) -> SurfaceTable:
    """Lay out a decision surface — a stored analysis's, or one frozen from a campaign's evidence alone.

    Args:
        surface: The surface.
        variant_index: The variant index that names its arms.

    Returns:
        The table.
    """

    control = surface.control_variant_key
    cells = sorted(surface.cells, key=lambda c: (c.variant_key != control, c.variant_key, c.apparatus_class_id))
    unadjudicated = [
        SurfaceUnadjudicatedBar(
            measure_id=bar.measure_id, source=bar.source, state=bar.state, reason=bar.reason or bar.state
        )
        for bar in surface.bars
        if bar.state != "adjudicated"
    ]
    if not cells:
        return SurfaceTable(state="no_cells", disclosure=NO_CELLS, unadjudicated_bars=unadjudicated)

    adjudicated = [bar for bar in surface.bars if bar.state == "adjudicated"]
    bar_columns = [_bar_column(bar, surface, cells) for bar in adjudicated]
    merit_columns = _merit_columns(surface, cells)

    index = {entry.variant_key: entry for entry in variant_index}
    distinguishing = distinguishing_axes(variant_index)
    names = arm_names(variant_index)
    multi_rig = multi_rig_variants(cells)

    rows = []
    for cell in cells:
        entry = index.get(cell.variant_key)
        notes = _run_notes(cell)
        rig = short_digest(cell.apparatus_class_id) if cell.variant_key in multi_rig else None
        rows.append(
            SurfaceRow(
                variant_key=cell.variant_key,
                apparatus_class_id=cell.apparatus_class_id,
                is_control=cell.variant_key == control,
                placed=entry is not None,
                # The arm table's own functions, so the two tables name one arm alike.
                label=arm_label(cell.variant_key, names, rig=rig),
                levels=naming_levels(entry, distinguishing),
                levels_unavailable=entry.levels_unavailable if entry else None,
                rig=rig,
                replication=_replication(cell),
                all_failed=cell.all_failed,
                flags=[flag for flag, runs in _flag_sources(cell) if runs],
                run_notes=notes,
                values=[
                    *(_bar_value(bar, cell, factor) for bar, (_, factor) in zip(adjudicated, bar_columns, strict=True)),
                    *(_merit_value(cell, column.measure_id, factor) for column, factor in merit_columns),
                ],
            )
        )
    return SurfaceTable(
        state="measured",
        columns=[column for column, _ in (*bar_columns, *merit_columns)],
        rows=rows,
        unadjudicated_bars=unadjudicated,
        all_failed_disclosure=_all_failed_disclosure(rows),
    )


def _all_failed_disclosure(rows: list[SurfaceRow]) -> str | None:
    """The sentence stated beside the table when some row's every result failed, naming those arms when not all.

    Args:
        rows: The table's rows.

    Returns:
        :func:`~threetears.evals.contracts.surface.all_failed_sentence`, followed by the failed rows' labels
        when only some failed; None when none did.
    """
    failed = [row.label for row in rows if row.all_failed]
    if not failed:
        return None
    sentence = all_failed_sentence(len(failed), len(rows))
    return sentence if len(failed) == len(rows) else f"{sentence} Every result failed in: {'; '.join(failed)}."


__all__ = [
    "NO_CELLS",
    "NO_SUCCESSFUL_RESULTS",
    "SURFACE_PROVENANCE",
    "VERDICT_WORDS",
    "SurfaceColumn",
    "SurfaceRow",
    "SurfaceRunNote",
    "SurfaceState",
    "SurfaceTable",
    "SurfaceUnadjudicatedBar",
    "SurfaceValue",
    "SurfaceVerdict",
    "build_surface_table",
    "surface_table_of",
]
