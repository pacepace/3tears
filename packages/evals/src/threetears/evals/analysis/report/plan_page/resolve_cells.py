"""The charts of cells: a heatmap of a matrix, a difference heatmap of two arms, and confusion matrices.

**heatmap** shades one figure over a matrix, the cases under each value:

- *cells*: the target matrix, its row axis's levels by its column axis's, a matrix per arm (and per measure, where
  the query names several), in panels by arm where there are several or the plan asks for them; the floor is the
  stored line, else the line of the one criterion read in every cell on the measure;
- *strata*, *tasks*: the slices by the arms, a matrix per measure (panels by measure where there are several); a
  task chart's floor is the one criterion read in every task. Where no arm holds a figure in every slice, the
  matrix keeps the slices every arm holding the measure has one in.

An arm (or a slice) lacking a figure anywhere in its matrix is left out, and a caption names it; cells below the floor are
outlined by the kit. The count format follows the largest count: thousands where one reaches ten thousand.

**diff_heatmap** shades one arm less another over the same matrix (the target matrix's cells by one measure, else
the slices by each measure): the query's second arm less its first where it names two or more, else its one arm less
the control, else the recommended arm (or the page's first candidate) less the control. Each cell prints the compared
arm's own figure under the difference (``detail``), and a caption says which arm is less which. A rate's difference
is in points, grayed within the kit's half a point; another unit's is shaded to its largest difference.

**confusion** draws one matrix per label axis the query keeps and per arm (the query's, else every arm the axis holds),
in panels where there is more than one: titled by the arm, with its chip, where there are several arms, else by the
axis. A table stated in row percents is drawn as it is; one in counts is normalized by the kit. The labels are the
axis's level names where the input's factors name them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from threetears.evals.analysis.report.charts import (
    ChartColumn,
    ChartReference,
    ConfusionChart,
    ConfusionPanel,
    DiffHeatmapCell,
    DiffHeatmapChart,
    DiffHeatmapRow,
    HeatmapCell,
    HeatmapChart,
    HeatmapPanel,
    HeatmapRow,
)
from threetears.evals.analysis.report.plan_page.charting import (
    ChartContext,
    CriterionWhere,
    Drawn,
    KindResolver,
    chart_block,
    criterion_line,
    figure_unit,
    measure_words,
    missing_note,
    reference_line,
    slice_name,
    stat_value,
)
from threetears.evals.analysis.report.plan_page.look import Quantity, band_domain, figure_text, quantity
from threetears.evals.analysis.report_input import Cell, ConfusionTable, Stat, Stratum, Task
from threetears.evals.analysis.style import humanize

__all__ = ["RESOLVERS"]

#: The count from which a matrix's counts print in thousands.
_THOUSANDS_FROM = 10_000

#: A cell's tooltip line under its place: its value and the cases under it.
_CASES_TEMPLATE = "{value} · n = {n}"
#: A cell's tooltip line where its value is below the floor.
_BELOW_TEMPLATE = "Below the {line}"
#: The caption naming which arm a difference heatmap draws less which, and in what unit.
_DIFFERENCE_NOTE = "Each cell is {compared} less {base}, {unit}; the small figure is {compared}'s own."
#: What a rate's difference is stated in, in the caption.
_POINTS_WORDS = "in points"
#: What any other difference is stated in, in the caption.
_UNIT_WORDS = "in its own unit"
#: The caption under confusion matrices.
_CONFUSION_NOTE = "Rows are the true label and columns the predicted one; each row is shared out to 100%."

#: The fraction of a non-rate difference heatmap's largest difference that is grayed as no difference.
_NEUTRAL_SHARE = 1 / 16


@dataclass(frozen=True)
class _Matrix:
    """One matrix as drawn: its title (an arm's or a measure's), its arm (for a chip), its rows, columns and cells."""

    title: str
    arm: str
    subtitle: str | None
    columns: list[str]
    rows: list[tuple[str, list[tuple[float, float | None, list[str]]]]]
    reference: ChartReference | None


def _matrix_axes(context: ChartContext, cells: Sequence[Cell]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """The target matrix's row and column levels a set of cells covers, each as ``(id, name)``, in the axes' order."""
    matrix = context.report_input.target_matrix
    if matrix is None:
        raise context.refuse("its cells lie on no target matrix the input declares")
    rows = {cell.row for cell in cells}
    columns = {cell.column for cell in cells}
    return (
        [(level.id, level.name or level.id) for level in matrix.rows.levels if level.id in rows],
        [(level.id, level.name or level.id) for level in matrix.columns.levels if level.id in columns],
    )


def _cell_keys(context: ChartContext, held: Sequence[dict[str, dict[str, Stat]]]) -> list[str]:
    """The measures a matrix chart reads: the ones its query names, else every one its cells or slices hold."""
    if context.query.measures:
        return list(context.query.measures)
    return list(dict.fromkeys(key for per_arm in held for figures in per_arm.values() for key in figures))


def _value(context: ChartContext, key: str, stat: Stat | None) -> float | None:
    read = stat_value(context.report_input, key, stat) if stat is not None else None
    return read[0] if read is not None else None


def _arm_name(context: ChartContext, arm: str) -> str:
    return context.look.arm(arm).name if arm in context.look.arms else arm


def _cell_tip(place: str, value: float, n: float | None, drawn: Quantity, line: ChartReference | None) -> list[str]:
    said = figure_text(value, drawn)
    tip = [place, _CASES_TEMPLATE.format(value=said, n=f"{n:,.0f}") if n is not None else said]
    if line is not None and line.kind == "floor" and value < line.value:
        tip.append(_BELOW_TEMPLATE.format(line=line.label or line.kind))
    return tip


def _line(context: ChartContext, key: str, drawn: Quantity, where: CriterionWhere | None) -> ChartReference | None:
    stored = reference_line(context, drawn)
    if stored is not None or where is None:
        return stored
    return criterion_line(context, key, drawn, where=where)


# --- heatmap ------------------------------------------------------------------------------------------------------


def _cell_matrices(context: ChartContext) -> tuple[list[_Matrix], Quantity, list[str], list[float]]:
    cells = [row.item for row in context.rows() if isinstance(row.item, Cell)]
    if not cells:
        raise context.refuse("it selects no cell")
    row_levels, column_levels = _matrix_axes(context, cells)
    by_place = {(cell.row, cell.column): cell for cell in cells}
    keys = _cell_keys(context, [cell.per_arm for cell in cells])
    held_arms = dict.fromkeys(arm for cell in cells for arm in cell.per_arm)
    arms = context.arm_order(context.query.arms or held_arms)
    values = [
        value
        for cell in cells
        for arm in arms
        for key in keys
        if (value := _value(context, key, cell.per_arm.get(arm, {}).get(key))) is not None
    ]
    drawn = quantity(figure_unit(context.report_input, keys[0]) if keys else "", values)
    matrices: list[_Matrix] = []
    missing: list[str] = []
    counts: list[float] = []
    for key in keys:
        line = _line(context, key, drawn, "cells")
        for arm in arms:
            rows: list[tuple[str, list[tuple[float, float | None, list[str]]]]] = []
            whole = True
            for row_id, row_name in row_levels:
                drawn_row: list[tuple[float, float | None, list[str]]] = []
                for column_id, column_name in column_levels:
                    cell = by_place.get((row_id, column_id))
                    value = _value(context, key, cell.per_arm.get(arm, {}).get(key)) if cell is not None else None
                    if cell is None or value is None:
                        whole = False
                        continue
                    shown = value * drawn.factor
                    n = float(cell.n_cases) if cell.n_cases is not None else None
                    drawn_row.append((shown, n, _cell_tip(f"{row_name} · {column_name}", shown, n, drawn, line)))
                rows.append((row_name, drawn_row))
            if not whole:
                missing.append(f"{_arm_name(context, arm)}, {measure_words(context.report_input, key)}")
                continue
            counts += [n for _, cells_drawn in rows for _, n, _ in cells_drawn if n is not None]
            matrices.append(
                _Matrix(
                    title=_arm_name(context, arm),
                    arm=arm,
                    subtitle=measure_words(context.report_input, key) if len(keys) > 1 else None,
                    columns=[name for _, name in column_levels],
                    rows=rows,
                    reference=line,
                )
            )
    return matrices, drawn, missing, counts


def _slice_matrices(context: ChartContext) -> tuple[list[_Matrix], Quantity, list[str], list[float]]:
    """strata or tasks: the slices by the arms, a matrix per measure."""
    slices = [row.item for row in context.rows() if isinstance(row.item, Stratum | Task)]
    keys = _cell_keys(context, [item.per_arm for item in slices])
    allowed = list(context.query.arms) or list(context.look.arms)
    arms_held = [arm for arm in context.arm_order(allowed) if any(arm in item.per_arm for item in slices)]
    values = [
        value
        for item in slices
        for arm in arms_held
        for key in keys
        if (value := _value(context, key, item.per_arm.get(arm, {}).get(key))) is not None
    ]
    drawn = quantity(figure_unit(context.report_input, keys[0]) if keys else "", values)
    where: CriterionWhere | None = "tasks" if context.query.source == "tasks" else None
    matrices: list[_Matrix] = []
    missing: list[str] = []
    counts: list[float] = []
    for key in keys:
        line = _line(context, key, drawn, where)

        def held(arm: str, item: Stratum | Task, key: str = key) -> bool:
            return _value(context, key, item.per_arm.get(arm, {}).get(key)) is not None

        kept = list(slices)
        whole = [arm for arm in arms_held if all(held(arm, item) for item in kept)]
        if not whole:
            # No arm holds a figure in every slice: keep the slices every arm holding the measure has one in.
            holders = [arm for arm in arms_held if any(held(arm, item) for item in slices)]
            kept = [item for item in slices if all(held(arm, item) for arm in holders)]
            whole = holders if kept else []
        name = measure_words(context.report_input, key)
        missing += [f"{_arm_name(context, arm)}, {name}" for arm in arms_held if arm not in whole]
        missing += [f"{slice_name(context, item)}, {name}" for item in slices if whole and item not in kept]
        if not whole:
            continue
        rows: list[tuple[str, list[tuple[float, float | None, list[str]]]]] = []
        for item in kept:
            name = slice_name(context, item)
            n = float(item.n_cases) if item.n_cases is not None else None
            counts += [n] if n is not None else []
            row = []
            for arm in whole:
                value = _value(context, key, item.per_arm[arm][key])
                assert value is not None
                shown = value * drawn.factor
                row.append((shown, n, _cell_tip(f"{name} · {_arm_name(context, arm)}", shown, n, drawn, line)))
            rows.append((name, row))
        matrices.append(
            _Matrix(
                title=measure_words(context.report_input, key),
                arm="",
                subtitle=None,
                columns=[_arm_name(context, arm) for arm in whole],
                rows=rows,
                reference=line,
            )
        )
    return matrices, drawn, missing, counts


def _heat_rows(matrix: _Matrix) -> list[HeatmapRow]:
    return [
        HeatmapRow(label=label, cells=[HeatmapCell(value=value, n=n, tip=tip) for value, n, tip in cells])
        for label, cells in matrix.rows
    ]


def _heatmap(context: ChartContext) -> Drawn:
    source = context.query.source
    if source == "cells":
        matrices, drawn, missing, counts = _cell_matrices(context)
    elif source in ("strata", "tasks"):
        matrices, drawn, missing, counts = _slice_matrices(context)
    else:
        raise context.refuse("a heatmap draws cells, strata or tasks")
    if not matrices:
        raise context.refuse("no arm holds a figure in every place of its matrix")
    ends = [value for matrix in matrices for _, cells in matrix.rows for value, _, _ in cells]
    ends += [matrix.reference.value for matrix in matrices if matrix.reference is not None]
    domain = context.spec.domain or band_domain(ends, drawn)
    n_fmt = ("k:1" if max(counts) >= _THOUSANDS_FROM else "int") if counts else None
    panelled = len(matrices) > 1 or (context.query.small_multiples_by == "arm" and bool(matrices[0].arm))
    first = matrices[0]
    shared_line = first.reference if all(matrix.reference == first.reference for matrix in matrices) else None
    if panelled:
        data = HeatmapChart(
            domain=(domain[0], domain[1]),
            fmt=drawn.fmt,
            n_fmt=n_fmt,
            reference=shared_line,
            columns=[ChartColumn(label=label) for label in first.columns],
            panels=[
                HeatmapPanel(
                    title=matrix.title,
                    subtitle=matrix.subtitle,
                    tag=context.look.arm(matrix.arm).tag if matrix.arm in context.look.arms else None,
                    color=context.look.arm(matrix.arm).token if matrix.arm in context.look.arms else None,
                    columns=[ChartColumn(label=label) for label in matrix.columns],
                    rows=_heat_rows(matrix),
                    reference=matrix.reference if shared_line is None else None,
                )
                for matrix in matrices
            ],
        )
    else:
        data = HeatmapChart(
            columns=[ChartColumn(label=label) for label in first.columns],
            rows=_heat_rows(first),
            domain=(domain[0], domain[1]),
            fmt=drawn.fmt,
            n_fmt=n_fmt,
            reference=first.reference,
        )
    return Drawn(chart_block(context, data), missing_note(missing))


# --- diff_heatmap -------------------------------------------------------------------------------------------------


def _compared(context: ChartContext) -> tuple[str, str]:
    """The two arms a difference heatmap draws, ``(base, compared)``: the compared one less the base."""
    named = list(context.query.arms)
    if len(named) >= 2:
        return named[0], named[1]
    control = context.look.control
    if control is None:
        raise context.refuse("it names fewer than two arms, and the input names no control to draw an arm less")
    if named:
        return control.id, named[0]
    candidates = [arm for arm in context.look.order if arm != control.id]
    pick = context.look.recommended if context.look.recommended in candidates else None
    if pick is None and not candidates:
        raise context.refuse("the input holds no candidate to draw less the control")
    return control.id, pick or candidates[0]


def _difference(
    context: ChartContext, key: str, base: Stat | None, compared: Stat | None
) -> tuple[float, float] | None:
    ours, theirs = _value(context, key, compared), _value(context, key, base)
    return (ours, theirs) if ours is not None and theirs is not None else None


def _diff_heatmap(context: ChartContext) -> Drawn:
    base, compared = _compared(context)
    source = context.query.source
    report_input = context.report_input
    places: list[tuple[str, list[tuple[str, Stat | None, Stat | None, str]]]] = []
    if source == "cells":
        cells = [row.item for row in context.rows() if isinstance(row.item, Cell)]
        if not cells:
            raise context.refuse("it selects no cell")
        keys = _cell_keys(context, [cell.per_arm for cell in cells])
        if len(keys) != 1:
            raise context.refuse(f"a difference heatmap of cells reads one measure, and it reads {keys}")
        (key,) = keys
        row_levels, column_levels = _matrix_axes(context, cells)
        by_place = {(cell.row, cell.column): cell for cell in cells}
        columns = [name for _, name in column_levels]
        for row_id, row_name in row_levels:
            line = []
            for column_id, column_name in column_levels:
                cell = by_place.get((row_id, column_id))
                held = cell.per_arm if cell is not None else {}
                line.append((key, held.get(base, {}).get(key), held.get(compared, {}).get(key), column_name))
            places.append((row_name, line))
    elif source in ("strata", "tasks"):
        slices = [row.item for row in context.rows() if isinstance(row.item, Stratum | Task)]
        keys = _cell_keys(context, [item.per_arm for item in slices])
        columns = [measure_words(report_input, key) for key in keys]
        for item in slices:
            places.append(
                (
                    slice_name(context, item),
                    [
                        (key, item.per_arm.get(base, {}).get(key), item.per_arm.get(compared, {}).get(key), column)
                        for key, column in zip(keys, columns, strict=True)
                    ],
                )
            )
    else:
        raise context.refuse("a difference heatmap draws cells, strata or tasks")
    units = {figure_unit(report_input, key) for _, line in places for key, *_ in line}
    if len(units) != 1:
        raise context.refuse(f"its measures are in more than one unit ({sorted(units)})")
    unit = units.pop()
    level = quantity(unit)
    change = quantity(unit, change=True)
    base_name, compared_name = _arm_name(context, base), _arm_name(context, compared)
    rows: list[DiffHeatmapRow] = []
    for row_name, line in places:
        drawn_cells = []
        for key, theirs, ours, column in line:
            pair = _difference(context, key, theirs, ours)
            if pair is None:
                raise context.refuse(f"{base_name} or {compared_name} holds no figure at {row_name} · {column}")
            mine, other = pair[0] * level.factor, pair[1] * level.factor
            drawn_cells.append(
                DiffHeatmapCell(
                    value=mine - other,
                    detail=mine,
                    tip=[
                        f"{row_name} · {column}",
                        f"{base_name} {figure_text(other, level)}",
                        f"{compared_name} {figure_text(mine, level)}",
                        figure_text(mine - other, change),
                    ],
                )
            )
        rows.append(DiffHeatmapRow(label=row_name, cells=drawn_cells))
    largest = max((abs(cell.value) for row in rows for cell in row.cells), default=0.0)
    rate = level.rate
    data = DiffHeatmapChart(
        columns=[ChartColumn(label=label) for label in columns],
        rows=rows,
        fmt="+num" if rate else change.fmt,
        detail_fmt=level.fmt,
        neutral=None if rate or not largest else largest * _NEUTRAL_SHARE,
        saturate=None if rate or not largest else largest,
    )
    note = _DIFFERENCE_NOTE.format(compared=compared_name, base=base_name, unit=_POINTS_WORDS if rate else _UNIT_WORDS)
    return Drawn(chart_block(context, data), (note,))


# --- confusion ----------------------------------------------------------------------------------------------------


def _label_names(context: ChartContext, axis: str, labels: Sequence[str]) -> list[str]:
    """An axis's labels by the names the input's factors (or its target matrix's axes) give its levels."""
    factors = list(context.report_input.factors)
    matrix = context.report_input.target_matrix
    if matrix is not None:
        factors += [matrix.rows, matrix.columns]
    factor = next((held for held in factors if held.id == axis), None)
    names = {level.id: level.name for level in factor.levels if level.name} if factor is not None else {}
    return [names.get(label, label) for label in labels]


def _axis_name(context: ChartContext, axis: str) -> str:
    factors = list(context.report_input.factors)
    matrix = context.report_input.target_matrix
    if matrix is not None:
        factors += [matrix.rows, matrix.columns]
    factor = next((held for held in factors if held.id == axis), None)
    return (factor.name if factor is not None else None) or humanize("dimension", axis)


def _confusion(context: ChartContext) -> Drawn:
    tables = [(row.key, row.item) for row in context.rows() if isinstance(row.item, ConfusionTable)]
    if not tables:
        raise context.refuse("it selects no confusion table")
    held = dict.fromkeys(arm for _, table in tables for arm in table.per_arm)
    arms = context.arm_order(context.query.arms or held)
    drawn: list[tuple[str, str, ConfusionTable, list[list[float]]]] = []
    missing: list[str] = []
    for axis, table in tables:
        for arm in arms:
            matrix = table.per_arm.get(arm)
            if matrix is None:
                missing.append(f"{_arm_name(context, arm)}, {_axis_name(context, axis)}")
                continue
            drawn.append((axis, arm, table, [[float(value) for value in row] for row in matrix]))
    if not drawn:
        raise context.refuse("no arm it reads holds a confusion table")
    several_arms = len({arm for _, arm, _, _ in drawn}) > 1
    several_axes = len({axis for axis, _, _, _ in drawn}) > 1
    notes = (_CONFUSION_NOTE, *missing_note(missing))
    if len(drawn) == 1:
        axis, _, table, matrix = drawn[0]
        data = ConfusionChart(
            labels=_label_names(context, axis, table.labels),
            matrix=matrix,
            normalized=table.unit == "row_percent",
        )
        return Drawn(chart_block(context, data), notes)
    panels = []
    for axis, arm, table, matrix in drawn:
        look = context.look.arm(arm) if several_arms and arm in context.look.arms else None
        panels.append(
            ConfusionPanel(
                title=look.name if look is not None else _axis_name(context, axis),
                subtitle=_axis_name(context, axis) if look is not None and several_axes else None,
                tag=look.tag if look is not None else None,
                color=look.token if look is not None else None,
                labels=_label_names(context, axis, table.labels),
                matrix=matrix,
                normalized=table.unit == "row_percent",
            )
        )
    return Drawn(chart_block(context, ConfusionChart(panels=panels)), notes)


#: The kinds this module resolves.
RESOLVERS: dict[str, KindResolver] = {"heatmap": _heatmap, "diff_heatmap": _diff_heatmap, "confusion": _confusion}
