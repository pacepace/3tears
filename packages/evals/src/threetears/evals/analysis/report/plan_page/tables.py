"""The planned page's two tables of code's own: the options table and the arms key.

**The options table** (GENERATOR §2.6): a row per option in the plan's order, the recommended (or leaned) one set
apart; its first columns name the option, as its arm's chip, and why it is shown, then what it runs (its settings as
the campaign declared them, and its served model where the settings do not name it), so the page says what each
option is wherever some option declares either; then one column per family code filled (``Options.columns``), each
cell from the input, by the package's number rule. The families are the plan
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
cost_per_1k      dollars per 1,000 of what the cost counts, the uplift against the control's, and how many of the
                 arm's calls went unanswered beside it, where any did (``calls.unanswered.<arm>``); an arm with no
                 cost says none beside its dash
===============  ===================================================================================================

A cell the input holds no figure for is a dash. A column outside these families is refused: it is no column the
page can fill. A pill's cell says its status in words (``Passes``, ``Worse, within margin``), so Markdown and a
screen reader, which lose the colour, lose nothing.

**The arms table** (:func:`arms_table`), a report document's options table: every arm in the input's order, whatever
the document says, as its chip, its role (the arms the document recommends marked ``recommended`` beside it), what
it runs (as the options table says it, where some arm declares it), its quality and its cost side by side:

- its pass rate, with its passes over its usable trials, where some arm's scoreboard holds one;
- its cost per run (per one of what the cost counts), or per 1,000 where the column's
  :func:`~threetears.evals.kernel.numbers.money_form` goes per 1,000 (#993): one form for the whole column, so the
  header says which and every cell is written in it. Beside it, the uplift against the control's, the priced share of
  the calls the cost counts (``all 897 calls priced``), and the arm's unanswered calls. A partly priced arm states
  ``n of m calls priced`` in place of a cost, which would be a smaller total read as whole; an arm none of whose calls
  were priced reads ``unpriced``, never ``$0``;
- what priced it: its price sources, the providers that served its model calls, where some arm's pricing names one,
  and the tools its cost spent on as tool spend, never as a provider of the model
  (:class:`~threetears.evals.analysis.report_input.Pricing`).

It reads no plan: which arms and columns it shows is the input's, and only the mark is the document's.

**The arms key** (GENERATOR §2.8, §6.1): each arm's full configuration once: its chip, its role, its settings as the
campaign declared them, its served model, the share of its output spent reasoning, and its usable trials, each column
present where some arm holds it.

**More than one aspect.** The input's scoreboard holds the lead aspect's trials alone, so where more than one aspect
counts usable trials, each table lays out one column group per aspect, in the input's aspect order, each head naming
its aspect (``Speed pass rate``): the arms table each aspect's pass rate, its cost and its usable trials; the options
table each aspect's pass rate and cost where the plan places the family (every other family reads the lead aspect, as
the plan's checks do); the arms key each aspect's usable trials. Each aspect's figures are its keys'
(``aspects.<aspect>.scoreboard.<arm>.*``, ``aspects.<aspect>.calls.unanswered.<arm>``), and its cost the one cost
rule read on it (:func:`~threetears.evals.analysis.report.figure_keys.aspect_costs_per_thousand`), every aspect's cost
in one form. An aspect's cost carries no uplift against the control's: that ratio's key reads the lead aspect alone.
An input of one aspect, or whose other aspects count no trials of their own, is laid out as above.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping

from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.blocks import TableBlock, TableCell, TableColumn
from threetears.evals.analysis.report.charts import ChartStatus
from threetears.evals.analysis.report.cost_unit import (
    PRICED_SHARE_TEMPLATE,
    REPLIES,
    MoneyRuler,
    money_ruler,
    per_one,
    priced_in_part,
)
from threetears.evals.analysis.report.figure_keys import (
    Figure,
    aspect_costs_per_thousand,
    cost_basis,
    cost_ruler,
    format_figure,
    held_figure,
)
from threetears.evals.analysis.report.plan_page.charting import figure_stat, measure_words, stat_value
from threetears.evals.analysis.report.plan_page.look import GATE_STATUS, ArmLook, PageLook, figure_text, quantity
from threetears.evals.analysis.report.plan_page.model import PLANNED_SECTION, PageArmsKey, PageOptions
from threetears.evals.analysis.report.plan_page.words import (
    ALL_PRICED_TEMPLATE,
    ARM_HEADERS,
    ARM_ROLE_WORDS,
    ARMS_EYEBROW,
    ASPECT_HEADER_TEMPLATE,
    ARMS_TABLE_ORDER,
    ARMS_TAKEAWAY,
    COST_HEADER,
    GATE_WORDS,
    OPTION_HEADERS,
    OPTIONS_TITLE,
    PASSES_TEMPLATE,
    PRICED_HEADER,
    RECOMMENDED_MARK,
    ROLE_WORDS,
    SERVED_BY_TEMPLATE,
    TOOL_SPEND_TEMPLATE,
    UPLIFT_NOT_COMPARABLE,
    reading_words,
)
from threetears.evals.analysis.report_gates import gate_check
from threetears.evals.analysis.report_input import Aspect, Pricing, ReportInput
from threetears.evals.analysis.report_margins import aspect_board
from threetears.evals.analysis.report_plan_checks import OPTION_COLUMN_FAMILIES
from threetears.evals.analysis.report_rows import query_aspect
from threetears.evals.analysis.style import lever_reader_name
from threetears.evals.kernel.numbers import UNPRICED
from threetears.evals.kernel.report_plan import ReportPlan

__all__ = ["arms_key", "arms_table", "options_table", "pass_rate_cell"]

#: The roles whose row the options table sets apart: the recommendation, or the lean.
_SET_APART = ("recommended", "lean")


#: Each reading direction's pill (GENERATOR §3: by the direction of the change, never by the margin).
_READING_STATUS: dict[str, ChartStatus] = {"worse": "bad", "better": "good", "": "neutral"}

#: An options cell: the family's column, the arm, the input and the look in; the cell out.
_CellReader = Callable[[str, str, ReportInput, PageLook], TableCell]


def _said(figure: Figure | None, filters: tuple[str, ...] = ()) -> str | None:
    """A figure as its cell says it, by the package's number rule; None (a dash) for none."""
    return format_figure(figure, filters) if figure is not None else None


def _pass_rate(column: str, arm: str, report_input: ReportInput, look: PageLook) -> TableCell:
    return pass_rate_cell(report_input, arm)


def _reading_change(column: str, arm: str, report_input: ReportInput, look: PageLook) -> TableCell:
    figure = held_figure(f"reading.{arm}.change", report_input)
    interval = held_figure(f"reading.{arm}.interval", report_input) if figure is not None else None
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
    return TableCell(value=_said(held_figure(f"judged.{dimension}.{arm}", report_input)))


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
    return TableCell(value=_said(held_figure(f"measure.{key}.{arm}.{read[1]}", report_input)))


def _cost(column: str, arm: str, report_input: ReportInput, look: PageLook) -> TableCell:
    figure = held_figure(f"cost_per_1k.{arm}", report_input)
    control = look.control
    uplift = (
        held_figure(f"cost.uplift.{arm}", report_input)
        if figure is not None and control is not None and arm != control.id
        else None
    )
    marks: list[str] = []
    if uplift is not None and isinstance(uplift.value, float):
        marks.append(("+" if uplift.value > 0 else "") + format_figure(uplift))
    # Beside the cost, never in it: the cost counts the answered calls, so an arm whose calls went unanswered says how
    # many, where any did.
    if figure is not None:
        # A cost over part of the arm's calls is a smaller total, never stated bare.
        share = priced_in_part(report_input, arm)
        marks += [share] if share is not None else []
        marks += _unanswered_marks(arm, report_input)
    # Per the count the column's head names, by the one owner of money's scale (figure_keys.cost_ruler). An arm the cost
    # rule states no cost for says it is unpriced, never a blank.
    said = (
        cost_ruler(report_input).spell(figure.value)
        if figure is not None and isinstance(figure.value, float)
        else UNPRICED
    )
    return TableCell(value=said, flag="; ".join(marks) or None)


def _unanswered_marks(arm: str, report_input: ReportInput, where: str = "") -> list[str]:
    """How many of the arm's calls went unanswered, where any did: retried and lost together, else each kind a role
    counted alone (the two together are not counted there). ``where`` is the keys' aspect prefix
    (``aspects.<aspect>.``), empty for the lead aspect's."""
    together = held_figure(f"{where}calls.unanswered.{arm}", report_input)
    if together is not None:
        return [_calls(together, "unanswered")] if together.value else []
    kinds = ((kind, held_figure(f"{where}calls.unanswered.{arm}.{kind}", report_input)) for kind in ("lost", "retried"))
    return [_calls(held, f"{kind} unanswered") for kind, held in kinds if held is not None and held.value]


def _calls(figure: Figure, what: str) -> str:
    """A count of calls in words: ``1 call unanswered``, ``3 calls unanswered``."""
    return f"{format_figure(figure)} {'call' if figure.value == 1 else 'calls'} {what}"


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
        return COST_HEADER.format(per=cost_ruler(report_input).words(cost_basis(report_input).noun))
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
    described = any(_runs(report_input, row.option) for row in plan.options.rows)
    aspects = _laid_out_aspects(report_input)
    # With more than one aspect, the pass rate and the cost are each aspect's, a column per aspect where the plan
    # places the family; every other family reads the lead aspect as the plan's checks do.
    grouped = {
        column: _aspect_groups(report_input, look, aspects, (column,), per=1000)
        for column in plan.options.columns
        if aspects and column in (_PASS_RATE, _COST)
    }
    columns = [
        TableColumn(key="option", header=OPTION_HEADERS["option"]),
        TableColumn(key="role", header=OPTION_HEADERS["role"]),
        *([TableColumn(key=_RUNS, header=ARM_HEADERS["settings"])] if described else []),
    ]
    for column in plan.options.columns:
        columns += (
            grouped[column][0] if column in grouped else [TableColumn(key=column, header=_header(column, report_input))]
        )
    rows = []
    for row in plan.options.rows:
        cells = {"option": _option_cell(row.option, look), "role": TableCell(value=ROLE_WORDS[row.role])}
        if described:
            cells[_RUNS] = TableCell(value=_runs(report_input, row.option) or None)
        for column in plan.options.columns:
            if column in grouped:
                group_columns, by_arm = grouped[column]
                held = by_arm.get(row.option, {})
                cells |= {
                    held_column.key: held.get(held_column.key, TableCell(value=None)) for held_column in group_columns
                }
                continue
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


#: The options table's column saying what an option runs.
_RUNS = "runs"


def _runs(report_input: ReportInput, option: str) -> str:
    """What an option runs, in a few words: its settings as the campaign declared them, then its served model where
    they do not name it; empty for an option that is no arm, or declares neither."""
    arm = next((arm for arm in report_input.arms if arm.id == option), None)
    if arm is None:
        return ""
    settings = _settings(report_input, arm.id)
    served = arm.served_model
    if served and served not in settings:
        return f"{settings} · {served}" if settings else served
    return settings


def arms_table(report_input: ReportInput, look: PageLook, recommends: Collection[str]) -> TableBlock:
    """The arms table, a report document's options table (this module's account).

    :param report_input: The report input the document was written from.
    :ptype report_input: ReportInput
    :param look: The page's look.
    :ptype look: PageLook
    :param recommends: The arms the document recommends, by id; empty for no call.
    :ptype recommends: Collection[str]
    :return: The table, a row per arm in the input's order.
    :rtype: TableBlock
    """
    arms = [arm.id for arm in report_input.arms]
    described = any(_runs(report_input, arm) for arm in arms)
    aspects = _laid_out_aspects(report_input)
    quality = {arm: pass_rate_cell(report_input, arm) for arm in arms}
    rated = not aspects and any(cell.value is not None for cell in quality.values())
    header, cells = _arm_costs(report_input, look)
    costed = not aspects and any(cell.value is not None for cell in cells.values())
    cells = {arm: _unknown_cost(cell) for arm, cell in cells.items()} if costed else cells
    grouped, by_aspect = _aspect_groups(report_input, look, aspects, (_PASS_RATE, _COST, "usable"), per=1)
    priced = {arm: _priced_by(_pricing(report_input, arm)) for arm in arms}
    sourced = any(cell.value is not None or cell.flag is not None for cell in priced.values())
    columns = [
        TableColumn(key="option", header=OPTION_HEADERS["option"]),
        TableColumn(key="role", header=OPTION_HEADERS["role"]),
        *([TableColumn(key=_RUNS, header=ARM_HEADERS["settings"])] if described else []),
        *([TableColumn(key=_PASS_RATE, header=OPTION_HEADERS[_PASS_RATE])] if rated else []),
        *([TableColumn(key=_COST, header=header)] if costed else []),
        *grouped,
        *([TableColumn(key=_PRICED, header=PRICED_HEADER)] if sourced else []),
    ]
    rows = []
    for arm in report_input.arms:
        row = {
            "option": _option_cell(arm.id, look),
            "role": TableCell(value=ARM_ROLE_WORDS[arm.role], flag=RECOMMENDED_MARK if arm.id in recommends else None),
        }
        if described:
            row[_RUNS] = TableCell(value=_runs(report_input, arm.id) or None)
        if rated:
            row[_PASS_RATE] = quality[arm.id]
        if costed:
            row[_COST] = cells[arm.id]
        row |= by_aspect[arm.id]
        if sourced:
            row[_PRICED] = priced[arm.id]
        rows.append(row)
    return TableBlock(
        section=PLANNED_SECTION,
        name="options",
        title=OPTIONS_TITLE,
        columns=columns,
        rows=rows,
        order=ARMS_TABLE_ORDER,
        total_rows=len(rows),
    )


#: The arms table's cost column: the options table's cost family.
_COST = "cost_per_1k"

#: The arms table's quality column, the options table's pass-rate family, and its column of what priced each cost.
_PASS_RATE = "pass_rate"
_PRICED = "priced_by"


def _pricing(report_input: ReportInput, arm: str, aspect: str | None = None) -> Pricing | None:
    """How an arm's cost was priced on one aspect (the lead's where None), as its scoreboard row says; None where the
    input does not say."""
    row = aspect_board(report_input, aspect).get(arm)
    return row.pricing if row is not None else None


def _arm_costs(report_input: ReportInput, look: PageLook) -> tuple[str, dict[str, TableCell]]:
    """The arms table's cost column: its header and each arm's cell, every cost in one form (:func:`_costs`)."""
    header, costs, ruler = _costs(report_input, [None], per=1)
    return header, {arm.id: _cost_cell(arm.id, report_input, look, costs, ruler) for arm in report_input.arms}


def _costs(
    report_input: ReportInput, aspects: Collection[str | None], *, per: int
) -> tuple[str, dict[tuple[str | None, str], float], MoneyRuler]:
    """Each arm's cost per ``per`` of what the cost counts on each aspect asked (None for the lead's, the one-aspect
    column's), by aspect and arm, with the form every cost asked shares and the head that says it (#993).

    An arm's cost is its cost per 1,000 of what the cost counts on the aspect
    (:func:`~threetears.evals.analysis.report.figure_keys.aspect_costs_per_thousand`) scaled to ``per``. The form is
    chosen from every cost stated (:func:`~threetears.evals.kernel.numbers.money_form`), never one cell's, so the
    aspects' costs read side by side: per ``per`` in dollars, or, where some cost is under a cent there, per 1,000
    times as many (and again), which the header says, so the cells are written without the mark. A partly priced arm
    is left out of the choice and states its priced share in place of a cost.
    """
    noun = cost_basis(report_input).noun
    costs: dict[tuple[str | None, str], float] = {}
    for aspect in aspects:
        board = aspect_board(report_input, aspect)
        held = aspect_costs_per_thousand(report_input, aspect)
        for arm in (arm.id for arm in report_input.arms):
            row = board.get(arm)
            if priced_in_part(report_input, arm, aspect) is not None:
                continue
            if arm in held:
                costs[aspect, arm] = held[arm] * per / 1000
            elif noun == REPLIES and row is not None and row.reply_cost_usd is not None:
                # The cost source the page reads may lack an arm another source holds: the scoreboard's cost per reply
                # is the same cost over the same count, so an arm it states is not left blank beside arms that carry a
                # dollar figure.
                costs[aspect, arm] = row.reply_cost_usd * per
    # The one owner of money's scale states the column: per ``per``, or per 1,000 (1,000,000) times as many where a
    # cost is under a cent there.
    ruler = money_ruler(costs.values(), per=per, start=per)
    return COST_HEADER.format(per=per_one(noun) if ruler.counted == 1 else ruler.words(noun)), costs, ruler


def _unknown_cost(cell: TableCell) -> TableCell:
    """A cost cell that states nothing (the arm's cost is not known) says so: ``unpriced``, never a blank."""
    return TableCell(value=UNPRICED, flag=cell.flag) if cell.value is None else cell


def _cost_cell(
    arm: str,
    report_input: ReportInput,
    look: PageLook,
    costs: Mapping[tuple[str | None, str], float],
    ruler: MoneyRuler,
    aspect: str | None = None,
) -> TableCell:
    """One arm's cost cell on one aspect (the lead's where None) in the column's form: its cost, then the uplift (on
    the one-aspect column alone, whose ratio's key reads the lead aspect), the priced share and the unanswered calls.
    A partly priced arm states its share in place of a cost, and an arm none of whose calls were priced reads
    ``unpriced``."""
    pricing = _pricing(report_input, arm, aspect)
    share = _priced_share(pricing)
    unanswered = _unanswered_marks(arm, report_input, "" if aspect is None else f"aspects.{aspect}.")
    if pricing is not None and pricing.priced_calls == 0 and pricing.calls:
        return TableCell(value=UNPRICED, flag="; ".join([share or "", *unanswered]).strip("; ") or None)
    if priced_in_part(report_input, arm, aspect) is not None:
        return TableCell(value=share, flag="; ".join(unanswered) or None)
    if (aspect, arm) not in costs:
        return TableCell(value=None)
    marks: list[str] = []
    control = look.control
    if aspect is None and control is not None and arm != control.id:
        # An uplift on a control whose cost counts only part of its calls is a ratio to a smaller total: none is read.
        partial = priced_in_part(report_input, control.id)
        uplift = held_figure(f"cost.uplift.{arm}", report_input) if partial is None else None
        if partial is not None:
            marks.append(UPLIFT_NOT_COMPARABLE.format(share=partial))
        elif uplift is not None and isinstance(uplift.value, float):
            marks.append(("+" if uplift.value > 0 else "") + format_figure(uplift))
    if share is not None:
        marks.append(share)
    marks += unanswered
    return TableCell(value=ruler.spell(costs[aspect, arm]), flag="; ".join(marks) or None)


def _priced_share(pricing: Pricing | None) -> str | None:
    """How many of the calls an arm's cost counts were priced, in words; None where they were not counted."""
    if pricing is None or pricing.calls is None or pricing.priced_calls is None or not pricing.calls:
        return None
    if pricing.priced_calls == pricing.calls:
        return ALL_PRICED_TEMPLATE.format(calls=pricing.calls)
    return PRICED_SHARE_TEMPLATE.format(priced=pricing.priced_calls, calls=pricing.calls)


def _priced_by(pricing: Pricing | None) -> TableCell:
    """What priced an arm's cost: its price sources, and beside them who served its model calls, then the tools it
    spent on as tool spend (never as the model's providers); empty where unsaid."""
    if pricing is None:
        return TableCell(value=None)
    said = [
        template.format(providers=", ".join(providers))
        for template, providers in (
            (SERVED_BY_TEMPLATE, pricing.providers),
            (TOOL_SPEND_TEMPLATE, pricing.tool_providers),
        )
        if providers
    ]
    return TableCell(value=", ".join(pricing.sources) or None, flag="; ".join(said) or None)


# --- more than one aspect ----------------------------------------------------------------------------------------
#
# The by-arm maps hold the lead aspect's figures alone, so a table of an input with several aspects lays out each
# aspect's figures in a column group of its own. Each is read by the reader the one-aspect column reads, given the
# aspect: its scoreboard (``aspect_board``, which ``aspects.<aspect>.scoreboard.*`` reads), its unanswered-call keys
# (``aspects.<aspect>.calls.unanswered.*``) and the one cost rule read on it.


def _laid_out_aspects(report_input: ReportInput) -> list[Aspect]:
    """The aspects a table lays out a column group each, in the input's order: every aspect some arm's trials count
    usable on, where more than one is; none otherwise, and the table is laid out for the one aspect it reads."""
    held = [
        aspect
        for aspect in report_input.aspects
        if any(row.usable for row in aspect_board(report_input, aspect.id).values())
    ]
    return held if len(held) > 1 else []


def _aspect_key(family: str, aspect: Aspect) -> str:
    """The key of one aspect's column of a family (``pass_rate@speed``)."""
    return f"{family}@{aspect.id}"


def _aspect_header(aspect: Aspect, header: str) -> str:
    """One aspect's column head: its name, then what the column holds (``Speed pass rate``)."""
    name = aspect.name or aspect.id
    return ASPECT_HEADER_TEMPLATE.format(aspect=name[:1].upper() + name[1:], header=header[:1].lower() + header[1:])


def pass_rate_cell(report_input: ReportInput, arm: str, aspect: str | None = None) -> TableCell:
    """An arm's pass rate on one aspect, as its scoreboard counts it, its passes over its usable trials beside it; on
    the lead aspect, an arm its scoreboard counts no passes for reads the ``pass_rate.<arm>`` figure, which falls back
    to the input's ``pass_rate`` measure where no arm's scoreboard counts passes. Every table's pass-rate cell is this
    one, so the options, arms and full-results tables state one rate for an arm.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param arm: The arm, by id.
    :ptype arm: str
    :param aspect: The aspect, by id (:func:`~threetears.evals.analysis.report_margins.aspect_board`); None for the
        lead aspect's, the input's own scoreboard.
    :ptype aspect: str | None
    :return: The cell; a dash where neither the scoreboard nor, on the lead aspect, the figure states a rate.
    :rtype: TableCell
    """
    row = aspect_board(report_input, aspect).get(arm)
    if row is None or row.passed is None or not row.usable:
        return TableCell(value=_said(held_figure(f"pass_rate.{arm}", report_input)) if aspect is None else None)
    drawn = quantity("share")
    return TableCell(
        value=figure_text(row.passed / row.usable * drawn.factor, drawn),
        flag=PASSES_TEMPLATE.format(passed=row.passed, usable=row.usable),
    )


def _trials(report_input: ReportInput, arm: str, aspect: str | None = None) -> TableCell:
    """An arm's usable trials on one aspect (the lead's where None), over those it ran there."""
    row = aspect_board(report_input, aspect).get(arm)
    if row is None or row.usable is None or not row.ran:
        return TableCell(value=None)
    return TableCell(value=PASSES_TEMPLATE.format(passed=row.usable, usable=row.ran))


def _aspect_groups(
    report_input: ReportInput, look: PageLook, aspects: list[Aspect], families: Collection[str], *, per: int
) -> tuple[list[TableColumn], dict[str, dict[str, TableCell]]]:
    """Each laid-out aspect's column group, in the input's aspect order, of the families asked (``pass_rate``,
    ``cost_per_1k``, ``usable``, in that order within a group), and each arm's cells, by arm; a column no arm fills is
    left out, and a cost column that some arm fills says ``unpriced`` for an arm it states no cost for."""
    if not aspects:
        return [], {arm.id: {} for arm in report_input.arms}
    cost_header, costs, ruler = _costs(report_input, [aspect.id for aspect in aspects], per=per)
    readers: dict[str, tuple[str, Callable[[Aspect, str], TableCell]]] = {
        _PASS_RATE: (OPTION_HEADERS[_PASS_RATE], lambda aspect, arm: pass_rate_cell(report_input, arm, aspect.id)),
        _COST: (cost_header, lambda aspect, arm: _cost_cell(arm, report_input, look, costs, ruler, aspect.id)),
        "usable": (ARM_HEADERS["usable"], lambda aspect, arm: _trials(report_input, arm, aspect.id)),
    }
    columns: list[TableColumn] = []
    cells: dict[str, dict[str, TableCell]] = {arm.id: {} for arm in report_input.arms}
    for aspect in aspects:
        for family, (header, read) in readers.items():
            if family not in families:
                continue
            column = {arm.id: read(aspect, arm.id) for arm in report_input.arms}
            if not any(cell.value is not None for cell in column.values()):
                continue
            if family == _COST:
                column = {arm: _unknown_cost(cell) for arm, cell in column.items()}
            key = _aspect_key(family, aspect)
            columns.append(TableColumn(key=key, header=_aspect_header(aspect, header)))
            for arm, cell in column.items():
                cells[arm][key] = cell
    return columns, cells


def _setting_name(report_input: ReportInput, setting: str) -> str:
    """A setting as a reader is shown it (:func:`~threetears.evals.analysis.style.lever_reader_name`, the one source
    a computed arm name reads too): the name the input declares for it (the factor its id keys, ``Reasoning budget``),
    else the package's reader name for the id (``max output tokens``, never ``max_output_tokens``)."""
    declared = {factor.id: factor.name for factor in report_input.factors if factor.name}
    return lever_reader_name(setting, declared)


def _settings(report_input: ReportInput, arm_id: str) -> str:
    """An arm's configuration, once: its model as written, then each other setting by its reader name and its value
    as written (``Meridian-4.1 · max output tokens 32k · reasoning medium``)."""
    arm = next(arm for arm in report_input.arms if arm.id == arm_id)
    return " · ".join(
        str(value) if setting == "model" else f"{_setting_name(report_input, setting)} {value}"
        for setting, value in arm.settings.items()
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
    aspects = _laid_out_aspects(report_input)
    present = {
        "settings": any(arm.settings for arm in arms),
        "served_model": any(arm.served_model for arm in arms),
        "reasoning": any(arm.reasoning_ratio is not None for arm in arms),
        "usable": not aspects and any(_trials(report_input, arm.id).value is not None for arm in arms),
    }
    keys = ["arm", "role", *(key for key, shown in present.items() if shown)]
    # With more than one aspect, each aspect's usable trials, a column per aspect.
    grouped, by_aspect = _aspect_groups(report_input, look, aspects, ("usable",), per=1)
    columns = [*(TableColumn(key=key, header=ARM_HEADERS[key]) for key in keys), *grouped]
    rows = []
    for arm in arms:
        reasoning = held_figure(f"reasoning_ratio.{arm.id}", report_input) if arm.reasoning_ratio is not None else None
        cells: dict[str, TableCell] = {
            "arm": _chip(look.arm(arm.id)),
            "role": TableCell(value=ARM_ROLE_WORDS[arm.role]),
            "settings": TableCell(value=_settings(report_input, arm.id) or None),
            "served_model": TableCell(value=arm.served_model),
            "reasoning": TableCell(value=_said(reasoning)),
            "usable": _trials(report_input, arm.id),
        }
        rows.append({**{key: cells[key] for key in keys}, **by_aspect[arm.id]})
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
