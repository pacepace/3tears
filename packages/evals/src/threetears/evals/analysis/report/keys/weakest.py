"""The weakest margin: the reading nearest its line, as a ``weakest.*`` key reads it."""

from __future__ import annotations

from dataclasses import replace

from threetears.evals.analysis.report.keys.figure import Figure, _listed
from threetears.evals.analysis.report.keys.names import reader_name
from threetears.evals.analysis.report.keys.reads import _floor_rate, _gap_figure, _scale_of, pass_rate_measure
from threetears.evals.analysis.report.keys.vocabulary import _level, _number, _refuse, _text, joint_level
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_margins import Margin, undecided_arms, weakest_margin
from threetears.evals.analysis.stats import INTERVAL_LEVEL


# --- the weakest margin --------------------------------------------------------------------------------------------

#: What every weakest-margin key begins with.
WEAKEST_PREFIX = "weakest."
_WEAKEST_FIELDS = ("margin", "arm", "criterion", "value", "threshold", "interval")


def _cell_name(report_input: ReportInput, cell_id: str) -> str:
    cell = report_input.cells[cell_id]
    matrix = report_input.target_matrix
    if matrix is None:
        return cell_id
    row = next((level.name or level.id for level in matrix.rows.levels if level.id == cell.row), cell.row)
    column = next((level.name or level.id for level in matrix.columns.levels if level.id == cell.column), cell.column)
    return f"{row} × {column}"


def _margin_name(margin: Margin, report_input: ReportInput) -> str:
    """What a margin is read against, in reader words: a criterion by its reader name and where its figure sits, or an
    aspect's margin or floor by the aspect's reader name.

    :param margin: One margin (:mod:`~threetears.evals.analysis.report_margins`).
    :ptype margin: Margin
    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The words ``{weakest.criterion}`` prints.
    :rtype: str
    """
    if margin.kind == "criterion":
        name = reader_name("criterion", margin.source, report_input)
        if margin.task is not None:
            return f"{name} in {reader_name('task', margin.task, report_input)}"
        if margin.cell is not None:
            return f"{name} in {_cell_name(report_input, margin.cell)}"
        return name
    aspect = reader_name("aspect", margin.aspect or margin.source, report_input)
    return f"the margin on {aspect}" if margin.kind == "margin" else f"the floor on {aspect}"


#: In a figure's words (:attr:`Figure.said_between`), where the size of the family a joint level holds across is
#: printed, as a count (:attr:`Figure.joint`).
_COMPARISONS_SLOT = "{comparisons}"


def _weakest_interval(report_input: ReportInput, margin: Margin, path: str) -> Figure:
    """Which interval a weakest margin's worst case is read on, in the package's own words, which state what the
    interval guarantees: the 95% interval adjusted for its family's comparisons where the worst case is read on the
    arm's multiplicity-corrected pass-rate contrast (``the 95% interval, adjusted for 16 comparisons``), the joint level
    and the family's size each a number the page's forms print; the nominal 95% interval where nothing is adjusted (a
    reading's and a floor's, ``the 95% interval``); a criterion's figure is read as measured, on no interval. The
    per-interval level a correction sets (``99.69%``) is no reading text's: Method and the glossary state it.

    :param report_input: The report input, whose contrast families say what a corrected interval guarantees together.
    :ptype report_input: ReportInput
    :param margin: The weakest margin.
    :ptype margin: Margin
    :param path: The key.
    :ptype path: str
    :return: What ``{weakest.interval}`` prints.
    :rtype: Figure
    """
    what = "the level of the interval the weakest margin is read on"
    if margin.kind == "criterion":
        return _text(path, "the figure as measured, on no interval", "which interval the weakest margin is read on")
    if margin.corrected:
        # The family that set its level, as the margin's reading found it: the first on its aspect, or on every aspect.
        family = next(
            (held for held in report_input.contrast_families if held.aspect in (margin.aspect, None) and held.level),
            None,
        )
        joint = joint_level(family)
        if joint is None:
            # The family states no joint guarantee to say, and the per-interval level is no reading text's.
            return _text(path, "the multiplicity-corrected interval", "which interval the weakest margin is read on")
        read = _level(path, joint[0], what)
        return replace(
            read, said_between=("the ", f" interval, adjusted for {_COMPARISONS_SLOT} comparisons"), joint=joint
        )
    return replace(_level(path, INTERVAL_LEVEL, what), said_between=("the ", " interval"))


def _weakest_rate(report_input: ReportInput, found: Margin) -> tuple[str, str] | None:
    """The rate a weakest margin read on an interval is a difference of: a floor's own rate (:func:`_floor_rate`), or
    the aspect's pass rate, which a reading's margin reads."""
    if found.kind == "floor":
        read = next(
            held
            for held in report_input.floors
            if (held.aspect, held.floor, held.arm) == (found.aspect, found.source, found.arm)
        )
        return _floor_rate(report_input, read)
    return pass_rate_measure(report_input)


def _weakest(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    found = weakest_margin(report_input)
    if found is None:
        undecided = undecided_arms(report_input)
        why = (
            f"; undecided: {_listed(undecided)}. A gate has no figure for them somewhere and none they have fails, so "
            "no figure of theirs could change the call alone (`count.unknown.<criterion>.<arm>` counts the places)"
            if undecided
            else ""
        )
        raise _refuse(
            path,
            "names the weakest margin, and the input reads no figure against a criterion, margin or floor that could "
            f"change the call alone{why}",
        )
    field = values[0]
    if field == "arm":
        return Figure(path=path, kind="arm", value=found.arm)
    if field == "criterion":
        return _text(path, _margin_name(found, report_input), "what the weakest margin is read against")
    if field == "interval":
        return _weakest_interval(report_input, found, path)
    if found.kind != "criterion":
        if field == "margin":
            # A gap is a change in points (#1127 item 4) and a distance from the line, which prints no ``+``.
            read = replace(_number(path, found.gap, "the weakest margin", "points", change=True), distance=True)
        elif field == "value":
            read = _number(path, found.value, "the weakest value", "points", change=True)
        else:
            # The line is declared (minus the margin), which prints as declared: it is no change.
            read = _number(path, found.line, "the weakest threshold", "points")
        return replace(read, set_of=_weakest_rate(report_input, found))
    scale, unit = _scale_of(report_input, found.measure or "", (found.line,))
    if field == "value":
        # A criterion's figure is a level of the measure it reads, as its own key's is (:func:`_criterion_figure_field`).
        read = _number(path, found.value, "the weakest margin's figure", scale, unit=unit)
        return replace(read, measure=found.measure or "")
    if field == "threshold":
        return _number(path, found.line, "the weakest margin's threshold", scale, unit=unit)
    # The margin is the criterion's gap, so it reads as ``criteria.<id>.<arm>.gap`` does, at its measure's changes' set.
    read = _gap_figure(report_input, path, found.measure or "", found.line, found.gap, "the weakest margin")
    return replace(read, measure=found.measure or "")


__all__ = [
    "WEAKEST_PREFIX",
]
