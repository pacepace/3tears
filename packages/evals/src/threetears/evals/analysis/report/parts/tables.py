"""The tables of code's own a report document's page lays out: the arms table and the arms key.

**The arms table** (:func:`arms_table`), a report document's options table: every arm in the input's order, whatever
the document says, as its chip, its role (the arms the document recommends marked ``recommended`` beside it), what
it runs (its settings as the campaign declared them, and its served model where the settings do not name it, where
some arm declares either), its quality and its cost side by side:

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
its aspect (``Speed pass rate``): the arms table each aspect's pass rate, its cost and its usable trials; the arms key
each aspect's usable trials. Each aspect's figures are its keys'
(``aspects.<aspect>.scoreboard.<arm>.*``, ``aspects.<aspect>.calls.unanswered.<arm>``), and its cost the one cost
rule read on it (:func:`~threetears.evals.analysis.report.figure_keys.aspect_costs_per_thousand`), every aspect's cost
in one form. An aspect's cost carries no uplift against the control's: that ratio's key reads the lead aspect alone.
An input of one aspect, or whose other aspects count no trials of their own, is laid out as above.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping

from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME
from threetears.evals.analysis.report.blocks import TableBlock, TableCell, TableColumn
from threetears.evals.analysis.report.cost_unit import (
    PRICED_SHARE_TEMPLATE,
    MoneyRuler,
    money_ruler,
    per_one,
    priced_in_part,
)
from threetears.evals.analysis.report.figure_keys import (
    Figure,
    aspect_costs_per_thousand,
    cost_basis,
    cost_measure,
    format_figure,
    held_figure,
)
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.parts.look import ArmLook, PageLook
from threetears.evals.analysis.report.parts.model import WRITER_SECTION, PageArmsKey
from threetears.evals.analysis.report.words import given_name
from threetears.evals.analysis.report.parts.words import (
    ALL_PRICED_TEMPLATE,
    ARM_HEADERS,
    ARM_ROLE_WORDS,
    ARMS_EYEBROW,
    ASPECT_HEADER_TEMPLATE,
    ARMS_TABLE_ORDER,
    ARMS_TAKEAWAY,
    COST_HEADER,
    OPTION_HEADERS,
    OPTIONS_TITLE,
    PASSES_TEMPLATE,
    PRICED_HEADER,
    RECOMMENDED_MARK,
    SERVED_BY_IN_PART_TEMPLATE,
    SERVED_BY_TEMPLATE,
    TOOL_SPEND_TEMPLATE,
    UPLIFT_NOT_COMPARABLE,
)
from threetears.evals.analysis.report_input import Aspect, Pricing, ReportInput
from threetears.evals.analysis.report_margins import aspect_board
from threetears.evals.analysis.style import lever_reader_name
from threetears.evals.kernel.numbers import UNPRICED

__all__ = ["arms_key", "arms_table", "pass_rate_cell"]


def _said(figure: Figure | None, forms: PageForms, filters: tuple[str, ...] = ()) -> str | None:
    """A figure as its cell says it, by the package's number rule in the page's forms; None (a dash) for none."""
    return format_figure(figure, filters, forms=forms) if figure is not None else None


def _unanswered_marks(arm: str, report_input: ReportInput, forms: PageForms, where: str = "") -> list[str]:
    """How many of the arm's calls went unanswered, where any did: retried and lost together, else each kind a role
    counted alone (the two together are not counted there). ``where`` is the keys' aspect prefix
    (``aspects.<aspect>.``), empty for the lead aspect's."""
    together = held_figure(f"{where}calls.unanswered.{arm}", report_input)
    if together is not None:
        return [_calls(together, "unanswered", forms)] if together.value else []
    kinds = ((kind, held_figure(f"{where}calls.unanswered.{arm}.{kind}", report_input)) for kind in ("lost", "retried"))
    return [_calls(held, f"{kind} unanswered", forms) for kind, held in kinds if held is not None and held.value]


def _calls(figure: Figure, what: str, forms: PageForms) -> str:
    """A count of calls in words: ``1 call unanswered``, ``3 calls unanswered``."""
    return f"{format_figure(figure, forms=forms)} {'call' if figure.value == 1 else 'calls'} {what}"


def _chip(arm: ArmLook) -> TableCell:
    """An arm's cell, drawn as its chip: its tag on its colour, then its name."""
    return TableCell(value=arm.name, tag=arm.tag, color=arm.token)


def _option_cell(option: str, look: PageLook) -> TableCell:
    """An option's cell: its arm's chip, or a modeled option's id."""
    return _chip(look.arm(option)) if option in look.arms else TableCell(value=option)


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


def arms_table(
    report_input: ReportInput, look: PageLook, recommends: Collection[str], *, forms: PageForms
) -> TableBlock:
    """The arms table, a report document's options table (this module's account).

    :param report_input: The report input the document was written from.
    :ptype report_input: ReportInput
    :param look: The page's look.
    :ptype look: PageLook
    :param recommends: The arms the document recommends, by id; empty for no call.
    :ptype recommends: Collection[str]
    :param forms: The page's forms (:meth:`~threetears.evals.analysis.report.forms.PageForms.of` over the same
        input), which state each arm's pass rate.
    :ptype forms: PageForms
    :return: The table, a row per arm in the input's order.
    :rtype: TableBlock
    :raises ValueError: The forms were built from another input.
    """
    forms = forms.checked(report_input)
    arms = [arm.id for arm in report_input.arms]
    described = any(_runs(report_input, arm) for arm in arms)
    aspects = _laid_out_aspects(report_input)
    quality = {arm: pass_rate_cell(report_input, arm, forms=forms) for arm in arms}
    rated = not aspects and any(cell.value is not None for cell in quality.values())
    header, cells = _arm_costs(report_input, look, forms)
    costed = not aspects and any(cell.value is not None for cell in cells.values())
    cells = {arm: _unknown_cost(cell) for arm, cell in cells.items()} if costed else cells
    grouped, by_aspect = _aspect_groups(
        report_input, look, aspects, (_PASS_RATE, _COST, "usable"), per=1, forms=forms
    )
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
        section=WRITER_SECTION,
        name="options",
        title=OPTIONS_TITLE,
        columns=columns,
        # Each rate prints its measure's decided text, at its measure's one precision on the whole page, recorded as
        # it prints.
        rows=forms.as_printed(rows),
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


def _arm_costs(report_input: ReportInput, look: PageLook, forms: PageForms) -> tuple[str, dict[str, TableCell]]:
    """The arms table's cost column: its header and each arm's cell, every cost in one form (:func:`_costs`)."""
    header, costs, ruler = _costs(report_input, [None], per=1)
    return header, {arm.id: _cost_cell(arm.id, report_input, look, costs, ruler, forms) for arm in report_input.arms}


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
    costs: dict[tuple[str | None, str], float] = {}
    for aspect in aspects:
        held = aspect_costs_per_thousand(report_input, aspect)
        for arm in (arm.id for arm in report_input.arms):
            if priced_in_part(report_input, arm, aspect) is not None:
                continue
            if arm in held:
                costs[aspect, arm] = held[arm] * per / 1000
    # The one owner of money's scale states the column: per ``per``, or per 1,000 (1,000,000) times as many where a
    # cost is under a cent there.
    ruler = money_ruler(
        costs.values(), per=per, start=per, reconciles=report_input.meta.volume_per_month is not None
    )
    noun = cost_basis(report_input).noun
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
    forms: PageForms,
    aspect: str | None = None,
) -> TableCell:
    """One arm's cost cell on one aspect (the lead's where None) in the column's form: its cost, then the uplift (on
    the one-aspect column alone, whose ratio's key reads the lead aspect), the priced share and the unanswered calls.
    A partly priced arm states its share in place of a cost, and an arm none of whose calls were priced reads
    ``unpriced``."""
    pricing = _pricing(report_input, arm, aspect)
    share = _priced_share(pricing)
    unanswered = _unanswered_marks(arm, report_input, forms, "" if aspect is None else f"aspects.{aspect}.")
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
            marks.append(("+" if uplift.value > 0 else "") + format_figure(uplift, forms=forms))
    if share is not None:
        marks.append(share)
    marks += unanswered
    # The cost is a figure of the measure the page reads it from, where that is a measure (not the scoreboard), as its
    # key's is (``cost_per_1k.<arm>``).
    held = cost_measure(report_input)
    cost = forms.money_on(ruler, costs[aspect, arm], measure=held[0] if held else "")
    return TableCell(value=cost.text, flag="; ".join(marks) or None)


def _priced_share(pricing: Pricing | None) -> str | None:
    """How many of the calls an arm's cost counts were priced, in words; None where they were not counted."""
    if pricing is None or pricing.calls is None or pricing.priced_calls is None or not pricing.calls:
        return None
    if pricing.priced_calls == pricing.calls:
        return ALL_PRICED_TEMPLATE.format(calls=pricing.calls)
    return PRICED_SHARE_TEMPLATE.format(priced=pricing.priced_calls, calls=pricing.calls)


def _priced_by(pricing: Pricing | None) -> TableCell:
    """What priced an arm's cost: its price sources, and beside them who served its model calls (and that the rest
    were not recorded where only some replies named one, never guessed), then the tools it spent on as tool spend
    (never as the model's providers); empty where unsaid.

    Where no reply named a provider the cell names none, as it did before any was recorded: every page built over
    calls recorded before then reads byte for byte as it did, and a judge comparing such pages is asked what it was
    asked when its replies were recorded."""
    if pricing is None:
        return TableCell(value=None)
    served = ", ".join(pricing.providers)
    said: list[str] = []
    if served:
        template = SERVED_BY_IN_PART_TEMPLATE if pricing.providers_in_part else SERVED_BY_TEMPLATE
        said.append(template.format(providers=served))
    if pricing.tool_providers:
        said.append(TOOL_SPEND_TEMPLATE.format(providers=", ".join(pricing.tool_providers)))
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
    name = given_name(aspect.id, aspect.name)
    return ASPECT_HEADER_TEMPLATE.format(aspect=name[:1].upper() + name[1:], header=header[:1].lower() + header[1:])


#: The measure an arm's pass rate falls back to where no scoreboard counts its passes (``pass_rate.<arm>``).
_FALLBACK_PASS_RATE = "pass_rate"


def pass_rate_cell(report_input: ReportInput, arm: str, aspect: str | None = None, *, forms: PageForms) -> TableCell:
    """An arm's pass rate on one aspect, as its scoreboard counts it, its passes over its usable trials beside it; on
    the lead aspect, an arm its scoreboard counts no passes for reads the ``pass_rate.<arm>`` figure, which falls back
    to the input's ``pass_rate`` measure where no arm's scoreboard counts passes. Every table's pass-rate cell is this
    one, so the options, arms and full-results tables state one rate for an arm, in the pass rate's form on the page.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param arm: The arm, by id.
    :ptype arm: str
    :param aspect: The aspect, by id (:func:`~threetears.evals.analysis.report_margins.aspect_board`); None for the
        lead aspect's, the input's own scoreboard.
    :ptype aspect: str | None
    :param forms: The page's forms, which hold the pass rate's wherever a scoreboard counts passes.
    :ptype forms: PageForms
    :return: The cell; a dash where neither the scoreboard nor, on the lead aspect, the figure states a rate.
    :rtype: TableCell
    :raises ValueError: The forms were built from another input.
    """
    forms = forms.checked(report_input)
    row = aspect_board(report_input, aspect).get(arm)
    if row is None or row.passed is None or not row.usable:
        figure = held_figure(f"pass_rate.{arm}", report_input) if aspect is None else None
        cell = TableCell(value=_said(figure, forms) if aspect is None else None)
        # Where no scoreboard counts this arm's passes, the figure is the input's own pass rate measure's, which the
        # column rules record the cell under.
        return cell if figure is None else forms.of_measure(cell, measure=_FALLBACK_PASS_RATE, kind="measure")
    cell = TableCell(
        value=forms.level(TRIAL_PASS_NAME, row.passed / row.usable, kind="trial").text,
        flag=PASSES_TEMPLATE.format(passed=row.passed, usable=row.usable),
    )
    # A figure of the pass rate, which the column rules record the cell under.
    return forms.of_measure(cell, measure=TRIAL_PASS_NAME, kind="trial")


def _trials(report_input: ReportInput, arm: str, aspect: str | None = None) -> TableCell:
    """An arm's usable trials on one aspect (the lead's where None), over those it ran there."""
    row = aspect_board(report_input, aspect).get(arm)
    if row is None or row.usable is None or not row.ran:
        return TableCell(value=None)
    return TableCell(value=PASSES_TEMPLATE.format(passed=row.usable, usable=row.ran))


def _aspect_groups(
    report_input: ReportInput,
    look: PageLook,
    aspects: list[Aspect],
    families: Collection[str],
    *,
    per: int,
    forms: PageForms,
) -> tuple[list[TableColumn], dict[str, dict[str, TableCell]]]:
    """Each laid-out aspect's column group, in the input's aspect order, of the families asked (``pass_rate``,
    ``cost_per_1k``, ``usable``, in that order within a group), and each arm's cells, by arm; a column no arm fills is
    left out, and a cost column that some arm fills says ``unpriced`` for an arm it states no cost for. A pass rate and
    a cost's marks are stated in the page's ``forms``.
    """
    if not aspects:
        return [], {arm.id: {} for arm in report_input.arms}
    cost_header, costs, ruler = _costs(report_input, [aspect.id for aspect in aspects], per=per)
    readers: dict[str, tuple[str, Callable[[Aspect, str], TableCell]]] = {
        _PASS_RATE: (OPTION_HEADERS[_PASS_RATE], lambda aspect, arm: _aspect_pass_rate(report_input, arm, aspect, forms)),
        _COST: (cost_header, lambda aspect, arm: _cost_cell(arm, report_input, look, costs, ruler, forms, aspect.id)),
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


def _aspect_pass_rate(report_input: ReportInput, arm: str, aspect: Aspect, forms: PageForms) -> TableCell:
    """One aspect's pass-rate cell (:func:`pass_rate_cell`), in the page's forms."""
    return pass_rate_cell(report_input, arm, aspect.id, forms=forms)


def _setting_name(report_input: ReportInput, setting: str) -> str:
    """A setting as a reader is shown it (:func:`~threetears.evals.analysis.style.lever_reader_name`, the one source
    a computed arm name reads too): the name the input stores for its lever (``lever_names``) or for the factor its id
    keys (``Reasoning budget``), else its id read as words where nothing declares it."""
    declared = report_input.lever_names | {factor.id: factor.name for factor in report_input.factors if factor.name}
    return lever_reader_name(setting, declared)


def _settings(report_input: ReportInput, arm_id: str) -> str:
    """An arm's configuration, once: its model as written, then each other setting by its reader name and its value
    as written (``Meridian-4.1 · max output tokens 32k · reasoning medium``)."""
    arm = next(arm for arm in report_input.arms if arm.id == arm_id)
    return " · ".join(
        str(value) if setting == "model" else f"{_setting_name(report_input, setting)} {value}"
        for setting, value in arm.settings.items()
    )


def arms_key(report_input: ReportInput, look: PageLook, *, forms: PageForms) -> PageArmsKey:
    """The arms key (this module's account).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param look: The page's look.
    :ptype look: PageLook
    :param forms: The page's forms, which state each arm's reasoning share.
    :ptype forms: PageForms
    :return: The key, under code's eyebrow and takeaway.
    :rtype: PageArmsKey
    :raises ValueError: The forms were built from another input.
    """
    forms = forms.checked(report_input)
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
    grouped, by_aspect = _aspect_groups(report_input, look, aspects, ("usable",), per=1, forms=forms)
    columns = [*(TableColumn(key=key, header=ARM_HEADERS[key]) for key in keys), *grouped]
    rows = []
    for arm in arms:
        reasoning = held_figure(f"reasoning_ratio.{arm.id}", report_input) if arm.reasoning_ratio is not None else None
        cells: dict[str, TableCell] = {
            "arm": _chip(look.arm(arm.id)),
            "role": TableCell(value=ARM_ROLE_WORDS[arm.role]),
            "settings": TableCell(value=_settings(report_input, arm.id) or None),
            "served_model": TableCell(value=arm.served_model),
            "reasoning": TableCell(value=_said(reasoning, forms)),
            "usable": _trials(report_input, arm.id),
        }
        rows.append({**{key: cells[key] for key in keys}, **by_aspect[arm.id]})
    table = TableBlock(
        section=WRITER_SECTION,
        name="arms_key",
        title=ARMS_EYEBROW,
        columns=columns,
        rows=rows,
        order="in the campaign's arm order",
        total_rows=len(rows),
    )
    return PageArmsKey(eyebrow=ARMS_EYEBROW, takeaway=ARMS_TAKEAWAY, table=table)
