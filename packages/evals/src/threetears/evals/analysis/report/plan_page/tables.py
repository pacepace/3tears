"""The planned page's two tables of code's own: the options table and the arms key.

**The options table** (GENERATOR §2.6): a row per option in the plan's order, the recommended (or leaned) one set
apart; its first columns name the option, as its arm's chip, and why it is shown; then one column per family code
filled (``Options.columns``), each cell from the input, by the package's number rule. The families are the plan
checks' vocabulary (:data:`~threetears.evals.analysis.report_plan_checks.OPTION_COLUMN_FAMILIES`), each with its one
cell renderer, and a family the vocabulary holds that no renderer fills stops this module loading:

===============  ===================================================================================================
family           cell
===============  ===================================================================================================
pass_rate        the pass rate, and its passes over its usable trials (``765 of 897``) where the scoreboard holds them
reading.change   the asked reading's change in points, and its interval
reading.verdict  a pill by GENERATOR §3's colour rule (worse is bad, better good, not separated neutral), the margin
                 verdict in its words; the control's reads ``Control``
judged:<dim>     the arm's figure on the judged dimension
gate             the arm's gate status (:func:`~threetears.evals.analysis.report_gates.gate_check`) as a pill, and
                 its first binding reason
measure:<key>    the arm's figure on the measure: its mean, else the statistic a criterion on it reads
cost_per_1k      dollars per 1,000 of what the cost counts, and the uplift against the control's
===============  ===================================================================================================

A cell the input holds no figure for is a dash. A column outside these families is refused: it is no column the
page can fill. A pill's cell says its status in words (``Passes``, ``Worse, within margin``), so Markdown and a
screen reader, which lose the colour, lose nothing.

**The arms key** (GENERATOR §2.8, §6.1): each arm's full configuration once: its chip, its role, its settings as the
campaign declared them, its served model, the share of its output spent reasoning, and its usable trials, each column
present where some arm holds it.
"""

from __future__ import annotations

from collections.abc import Callable

from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.blocks import TableBlock, TableCell, TableColumn
from threetears.evals.analysis.report.charts import ChartStatus
from threetears.evals.analysis.report.cost_unit import per_count
from threetears.evals.analysis.report.figure_keys import Figure, cost_basis, format_figure, resolve
from threetears.evals.analysis.report.plan_page.charting import figure_stat, measure_words, stat_value
from threetears.evals.analysis.report.plan_page.look import GATE_STATUS, ArmLook, PageLook
from threetears.evals.analysis.report.plan_page.model import PLANNED_SECTION, PageArmsKey, PageOptions
from threetears.evals.analysis.report.plan_page.words import (
    ARM_HEADERS,
    ARM_ROLE_WORDS,
    ARMS_EYEBROW,
    ARMS_TAKEAWAY,
    COST_HEADER,
    GATE_WORDS,
    OPTION_HEADERS,
    OPTIONS_TITLE,
    PASSES_TEMPLATE,
    ROLE_WORDS,
    reading_words,
)
from threetears.evals.analysis.report_gates import gate_check
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_plan_checks import OPTION_COLUMN_FAMILIES
from threetears.evals.analysis.report_rows import query_aspect
from threetears.evals.kernel.report_plan import ReportPlan

__all__ = ["arms_key", "options_table"]

#: The roles whose row the options table sets apart: the recommendation, or the lean.
_SET_APART = ("recommended", "lean")


#: Each reading direction's pill (GENERATOR §3: by the direction of the change, never by the margin).
_READING_STATUS: dict[str, ChartStatus] = {"worse": "bad", "better": "good", "": "neutral"}

#: An options cell: the family's column, the arm, the input and the look in; the cell out.
_CellReader = Callable[[str, str, ReportInput, PageLook], TableCell]


def _figure(key: str, report_input: ReportInput) -> Figure | None:
    """The figure a key names, or None where the input holds none for it (a dash in its cell)."""
    try:
        return resolve(key, report_input)
    except UnresolvableReference:
        # NOSILENT: every key here is built by code from an arm id and a column family, never typed by the planner, so
        # one that does not resolve means the input holds no such figure for that arm: the cell prints a dash, which
        # is what the table says for an absent figure.
        return None


def _said(figure: Figure | None, filters: tuple[str, ...] = ()) -> str | None:
    """A figure as its cell says it, by the package's number rule; None (a dash) for none."""
    return format_figure(figure, filters) if figure is not None else None


def _pass_rate(column: str, arm: str, report_input: ReportInput, look: PageLook) -> TableCell:
    figure = _figure(f"pass_rate.{arm}", report_input)
    row = report_input.scoreboard.get(arm)
    passes = (
        PASSES_TEMPLATE.format(passed=row.passed, usable=row.usable)
        if figure is not None and row is not None and row.passed is not None and row.usable
        else None
    )
    return TableCell(value=_said(figure), flag=passes)


def _reading_change(column: str, arm: str, report_input: ReportInput, look: PageLook) -> TableCell:
    figure = _figure(f"reading.{arm}.change", report_input)
    interval = _figure(f"reading.{arm}.interval", report_input) if figure is not None else None
    return TableCell(value=_said(figure), flag=_said(interval))


def _reading_verdict(column: str, arm: str, report_input: ReportInput, look: PageLook) -> TableCell:
    if arm in look.arms and look.arm(arm).control:
        return TableCell(value=ARM_ROLE_WORDS["control"], status="neutral")
    aspect = query_aspect("", report_input)
    reading = next((one for one in aspect.readings if one.arm == arm), None) if aspect is not None else None
    if reading is None or reading.within_interval is None:
        return TableCell(value=None)
    low, high = (float(end) for end in reading.within_interval)
    direction = "worse" if high < 0 else "better" if low > 0 else ""
    margin = aspect.margin if aspect is not None else None
    within = -low < float(margin) if margin is not None else None
    return TableCell(value=reading_words(direction, within), status=_READING_STATUS[direction])


def _judged(column: str, arm: str, report_input: ReportInput, look: PageLook) -> TableCell:
    dimension = column.removeprefix("judged:")
    return TableCell(value=_said(_figure(f"judged.{dimension}.{arm}", report_input)))


def _gate(column: str, arm: str, report_input: ReportInput, look: PageLook) -> TableCell:
    checked = gate_check(report_input, arm)
    reasons = (*checked.fails, *checked.borders)
    return TableCell(
        value=GATE_WORDS[checked.status], status=GATE_STATUS[checked.status], flag=reasons[0] if reasons else None
    )


def _measure(column: str, arm: str, report_input: ReportInput, look: PageLook) -> TableCell:
    key = column.removeprefix("measure:")
    stat = figure_stat(report_input, key, arm)
    read = stat_value(report_input, key, stat) if stat is not None else None
    if read is None:
        return TableCell(value=None)
    return TableCell(value=_said(_figure(f"measure.{key}.{arm}.{read[1]}", report_input)))


def _cost(column: str, arm: str, report_input: ReportInput, look: PageLook) -> TableCell:
    figure = _figure(f"cost_per_1k.{arm}", report_input)
    control = look.control
    uplift = (
        _figure(f"cost.uplift.{arm}", report_input)
        if figure is not None and control is not None and arm != control.id
        else None
    )
    flag = None
    if uplift is not None and isinstance(uplift.value, float):
        flag = ("+" if uplift.value > 0 else "") + format_figure(uplift)
    return TableCell(value=_said(figure, ("usd",)), flag=flag)


#: Each column family's cell reader, keyed by the family exactly as
#: :data:`~threetears.evals.analysis.report_plan_checks.OPTION_COLUMN_FAMILIES` spells it (``judged:<dim>``,
#: ``measure:<key>`` for the keyed ones).
_RENDERERS: dict[str, _CellReader] = {
    "pass_rate": _pass_rate,
    "reading.change": _reading_change,
    "reading.verdict": _reading_verdict,
    "judged:<dim>": _judged,
    "gate": _gate,
    "measure:<key>": _measure,
    "cost_per_1k": _cost,
}

if _unrendered := [family for family in OPTION_COLUMN_FAMILIES if family not in _RENDERERS]:
    # The vocabulary is the plan checks'; a family they accept that this table cannot fill would refuse a stored plan.
    raise RuntimeError(f"the options table has no cell renderer for the column families {_unrendered}")

#: Each keyed family by its prefix (``judged`` → ``judged:<dim>``).
_KEYED = {family.partition(":")[0]: family for family in OPTION_COLUMN_FAMILIES if ":" in family}


def _family(column: str) -> str:
    """The family a column is of: itself for a family written alone, its keyed family for ``judged:<dim>`` and
    ``measure:<key>``; a column of no family is returned as it is, which no renderer holds."""
    prefix, keyed, _ = column.partition(":")
    return _KEYED.get(prefix, column) if keyed else column


def _header(column: str, report_input: ReportInput) -> str:
    family = _family(column)
    if family == "judged:<dim>":
        return measure_words(report_input, column.removeprefix("judged:"), kind="judged")
    if family == "measure:<key>":
        return measure_words(report_input, column.removeprefix("measure:"))
    if family == "cost_per_1k":
        return COST_HEADER.format(per=per_count(cost_basis(report_input).noun))
    return OPTION_HEADERS[family]


def _chip(arm: ArmLook) -> TableCell:
    """An arm's cell, drawn as its chip: its tag on its colour, then its name."""
    return TableCell(value=arm.name, tag=arm.tag, color=arm.token)


def _option_cell(option: str, look: PageLook) -> TableCell:
    """An option's cell: its arm's chip, or a modeled option's id."""
    return _chip(look.arm(option)) if option in look.arms else TableCell(value=option)


def options_table(plan: ReportPlan, report_input: ReportInput, look: PageLook) -> PageOptions:
    """The options table (this module's account).

    :param plan: The stored plan.
    :ptype plan: ReportPlan
    :param report_input: The report input it was written against.
    :ptype report_input: ReportInput
    :param look: The page's look.
    :ptype look: PageLook
    :return: The table and the row it sets apart.
    :rtype: PageOptions
    :raises UnresolvableReference: A column is no family the page fills.
    """
    if stray := [column for column in plan.options.columns if _family(column) not in _RENDERERS]:
        raise UnresolvableReference(
            f"the options table's columns {stray} are no column family the page fills; it fills "
            f"{list(OPTION_COLUMN_FAMILIES)}"
        )
    columns = [
        TableColumn(key="option", header=OPTION_HEADERS["option"]),
        TableColumn(key="role", header=OPTION_HEADERS["role"]),
        *(TableColumn(key=column, header=_header(column, report_input)) for column in plan.options.columns),
    ]
    rows = []
    for row in plan.options.rows:
        cells = {"option": _option_cell(row.option, look), "role": TableCell(value=ROLE_WORDS[row.role])}
        for column in plan.options.columns:
            cells[column] = (
                _RENDERERS[_family(column)](column, row.option, report_input, look)
                if row.option in look.arms
                else TableCell(value=None)
            )
        rows.append(cells)
    highlighted = next((index for index, row in enumerate(plan.options.rows) if row.role in _SET_APART), None)
    table = TableBlock(
        section=PLANNED_SECTION,
        name="options",
        title=OPTIONS_TITLE,
        columns=columns,
        rows=rows,
        order="as the plan lists the options, the recommended first",
        total_rows=len(rows),
    )
    return PageOptions(table=table, highlighted=highlighted)


def _settings(report_input: ReportInput, arm_id: str) -> str:
    arm = next(arm for arm in report_input.arms if arm.id == arm_id)
    return " · ".join(
        str(value) if setting == "model" else f"{setting} {value}" for setting, value in arm.settings.items()
    )


def arms_key(report_input: ReportInput, look: PageLook) -> PageArmsKey:
    """The arms key (this module's account).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param look: The page's look.
    :ptype look: PageLook
    :return: The key, under code's eyebrow and takeaway.
    :rtype: PageArmsKey
    """
    arms = report_input.arms
    present = {
        "settings": any(arm.settings for arm in arms),
        "served_model": any(arm.served_model for arm in arms),
        "reasoning": any(arm.reasoning_ratio is not None for arm in arms),
        "usable": any(
            (row := report_input.scoreboard.get(arm.id)) is not None and row.usable is not None and row.ran
            for arm in arms
        ),
    }
    keys = ["arm", "role", *(key for key, shown in present.items() if shown)]
    columns = [TableColumn(key=key, header=ARM_HEADERS[key]) for key in keys]
    rows = []
    for arm in arms:
        board = report_input.scoreboard.get(arm.id)
        reasoning = _figure(f"reasoning_ratio.{arm.id}", report_input) if arm.reasoning_ratio is not None else None
        cells: dict[str, TableCell] = {
            "arm": _chip(look.arm(arm.id)),
            "role": TableCell(value=ARM_ROLE_WORDS[arm.role]),
            "settings": TableCell(value=_settings(report_input, arm.id) or None),
            "served_model": TableCell(value=arm.served_model),
            "reasoning": TableCell(value=_said(reasoning)),
            "usable": TableCell(
                value=PASSES_TEMPLATE.format(passed=board.usable, usable=board.ran)
                if board is not None and board.usable is not None and board.ran
                else None
            ),
        }
        rows.append({key: cells[key] for key in keys})
    table = TableBlock(
        section=PLANNED_SECTION,
        name="arms_key",
        title=ARMS_EYEBROW,
        columns=columns,
        rows=rows,
        order="in the campaign's arm order",
        total_rows=len(rows),
    )
    return PageArmsKey(eyebrow=ARMS_EYEBROW, takeaway=ARMS_TAKEAWAY, table=table)
