"""What every part of an aspect's page reads: the control and the candidates, the floors read on each, each arm's,
cell's and case's name, the cost ruler, the chart ledger, a reading's cell and its interval's words, the strata as
printed, and the methods' fold.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Protocol

from threetears.evals.analysis.arms import arm_names, multi_rig_variants, short_digest
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.quantities import PERCENT, is_money
from threetears.evals.analysis.references import resolve_reading
from threetears.evals.analysis.floors import FloorVerdict, floor_readings, read_floor
from threetears.evals.analysis.report.figures import ContrastPlaces, MeasureRuler, reply_cost_ruler
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.model import (
    CellValue,
    DisclosureBlock,
    DisclosureSource,
    TableBlock,
    TableCell,
    TableColumn,
    reading_anchor,
    table_row,
)
from threetears.evals.analysis.report.unit_once import unit_spaced
from threetears.evals.analysis.report.verdict import MIN_CASES
from threetears.evals.analysis.report.words import stratum_name, stratum_names
from threetears.evals.analysis.style import ArmName, arm_display_name, arm_name, humanize, page_date
from threetears.evals.kernel.campaign import ReadingKind, VariantIndexEntry
from threetears.evals.kernel.numbers import ABSENT, MoneyForm
from threetears.evals.kernel.surface import DecisionSurface


def _candidates(bundle: AnalysisContextBundle) -> list[str]:
    """Every arm on the scoreboard but the control, by variant key, in the scoreboard's order; empty with no control."""
    control = _control_key(bundle)
    if control is None:
        return []
    return [board.variant_key for board in bundle.scoreboard if board.variant_key != control]


def _floor_verdicts(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> dict[str, tuple[FloorVerdict, ...]]:
    """The aspect's floors read on each candidate against the control, each arm by ``arm``'s name, by the candidate's
    variant key; empty where the aspect declares no floor or no control ran."""
    declared = bundle.declared
    control = _control_key(bundle)
    if declared is None or not declared.floors or control is None:
        return {}
    read: dict[str, tuple[FloorVerdict, ...]] = {}
    for key in _candidates(bundle):
        readings = floor_readings(bundle, control, key, check_names=bundle.named_checks())
        read[key] = tuple(
            read_floor(floor, readings, control=arm(control), candidate=arm(key)) for floor in declared.floors
        )
    return read


def _control_key(bundle: AnalysisContextBundle) -> str | None:
    """The control's variant key, when it ran."""
    control = bundle.design.control_arm.variant_key if bundle.design.control_arm is not None else None
    return control if any(board.variant_key == control for board in bundle.scoreboard) else None


def _arm_names(
    bundle: AnalysisContextBundle,
    index: Sequence[VariantIndexEntry] | None = None,
    *,
    labels: Mapping[str, str] | None = None,
) -> dict[str, ArmName]:
    """Every arm's name, given label first; an arm whose label another shares is told apart by its digest.

    ``index`` names the arms of another variant index (a stored analysis's own) by the same rule; the bundle's when
    None. ``labels`` are a campaign's own labels for its arms, by variant key, which replace their variants' labels.
    """
    displays = arm_names(bundle.variant_index if index is None else index)
    control = _control_key(bundle)
    names = {
        key: arm_name(display, control=key == control, declared=bundle.lever_names) for key, display in displays.items()
    }
    # A name spelled from the arm's levers is printed by the one display-name rule, so every surface reads it alike.
    names = {
        key: replace(name, label=arm_display_name(name.label, name.levers, bundle.lever_names))
        for key, name in names.items()
    }
    if labels:
        # A campaign's own labels, where its page is being built: the label is the name, the model stays beside it.
        names = {
            key: replace(name, label=labels[key], given=True, levers=()) if key in labels else name
            for key, name in names.items()
        }
    taken = [name.label for name in names.values()]
    return {
        key: name
        if taken.count(name.label) == 1
        else ArmName(label=f"{name.label} ({short_digest(key)})", model=name.model, control=name.control)
        for key, name in names.items()
    }


def _namer(names: dict[str, ArmName]) -> Callable[[str], str]:
    """Name an arm by its variant key as every table of the report names it: by its label, never as a digest."""

    def name(variant_key: str) -> str:
        if variant_key in names:
            return names[variant_key].label
        return f"unplaced arm {short_digest(variant_key)}"

    return name


def _date(stamp: str) -> str:
    """An ISO-8601 instant as the page dates it (``14 Oct 2026``, :func:`~threetears.evals.analysis.style.page_date`)."""
    return page_date(stamp)


def _cost_ruler(bundle: AnalysisContextBundle) -> MeasureRuler:
    """The ruler every cost per reply on the page is stated on, chosen over the scoreboard's arms."""
    return reply_cost_ruler(board.reply_cost_usd for board in bundle.scoreboard)


def _per_reply(bundle: AnalysisContextBundle | None) -> str:
    """The words a sentence says a cost per reply is per: ``per reply``, or ``per 1,000 replies`` on that ruler."""
    per = _cost_ruler(bundle).per if bundle is not None else ""
    return per or "per reply"


class _ChartLedger:
    """Which intervals were drawn in the tables, and why each left out was: the one sentence in the methods.

    An interval is drawn beside a figure only when its arm has :data:`MIN_CASES` usable cases and one was computed;
    the figure stands alone otherwise, no stub. Every interval left out is counted here, so the methods say once, in
    one sentence, what was not drawn and why.
    """

    def __init__(self, bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> None:
        self._arm = arm
        self._usable = {board.variant_key: board.usable_cases for board in bundle.scoreboard}
        self._small: dict[str, int] = {}
        self._no_interval: list[str] = []

    def drawable(self, variant_key: str, n_cases: int | None, interval: tuple[float, float] | None, what: str) -> bool:
        """Whether a row of ``variant_key``'s may be drawn; one that may not is counted for the sentence."""
        usable = min(self._usable.get(variant_key, 0), n_cases if n_cases is not None else 0)
        if usable < MIN_CASES:
            self._small[variant_key] = min(self._small.get(variant_key, usable), usable)
            return False
        if interval is None:
            self.left_out(what)
            return False
        return True

    def left_out(self, what: str) -> None:
        """Count a reading no row of which could be drawn, for want of an interval."""
        if what not in self._no_interval:
            self._no_interval.append(what)

    def sentence(self) -> str | None:
        """Every interval not drawn, in one sentence; None when nothing was left out, or no arm could have one drawn
        at all (the caveat under the scoreboard says so). Read after the contrasts are laid out."""
        if all(usable < MIN_CASES for usable in self._usable.values()):
            return None
        parts = []
        if self._small:
            named = "; ".join(
                f"{self._arm(key)}, {usable} usable case{'' if usable == 1 else 's'}"
                for key, usable in sorted(self._small.items(), key=lambda item: self._arm(item[0]))
            )
            parts.append(
                f"No interval drawn for {named}: an interval is estimated only from {MIN_CASES} independent cases."
            )
        if self._no_interval:
            parts.append(f"No interval drawn, none computed: {', '.join(self._no_interval)}.")
        return " ".join(parts) or None


def _readings_table(name: str, title: str, rows: list[dict[str, CellValue]]) -> TableBlock:
    """Readings the page lists rather than strings into a sentence, one row each, each named once."""
    columns = [TableColumn(key="reading", header="Reading", term="reading")]
    if any("declare" in row for row in rows):
        columns.append(TableColumn(key="declare", header="Declaring one", term="margin"))
    return TableBlock(
        section="not_tested",
        name=name,
        title=title,
        columns=columns,
        rows=[table_row({column.key: row.get(column.key) for column in columns}) for row in rows],
        order="as the bundle lists them",
        total_rows=len(rows),
    )


def _note(text: str, source: DisclosureSource = "comparisons") -> DisclosureBlock:
    """Engine text for the methods' closed ``Disclosures`` fold."""
    return DisclosureBlock(section="methods", source=source, text=text, fold="Disclosures")


class _OneCell(Protocol):
    """Anything naming one cell: an arm's variant under one rig."""

    @property
    def variant_key(self) -> str: ...

    @property
    def apparatus_class_id(self) -> str: ...


def _at(
    cell: TableCell, surface: DecisionSurface, at: _OneCell, name: str, reading: str, figure: float | None
) -> TableCell:
    """``cell``, printing ``figure`` for ``name`` read at ``at``'s cell, carrying the id a finding citing that reading
    links to (:func:`~threetears.evals.analysis.report.model.reading_anchor`) where ``figure`` is the cell's own: the
    mean a finding's evidence resolves to (:func:`~threetears.evals.analysis.references.resolve_reading`). A figure
    read over other cases (a paired test's mean over the cases both sides ran, a guardrail's mean of per-case means)
    is another number, and a link from the finding's figure never lands on another number; such a cell is handed back
    as it was.

    :func:`_traced` keeps the first such cell on the page.
    """
    kind: ReadingKind | None = "measure" if reading == "measure" else "judged" if reading == "judged" else None
    if figure is None or cell.value is None or kind is None:
        return cell
    ref = cell_ref(at.variant_key, at.apparatus_class_id)
    try:
        own = resolve_reading(surface, ref, name, kind).mean
    except UnresolvableReference:
        # NOSILENT: the cell holds no figure of its own for this reading, so a finding's figure has nothing to match
        return cell
    if not math.isclose(figure, own, rel_tol=1e-9, abs_tol=1e-12):
        return cell
    return cell.model_copy(update={"anchor": reading_anchor(ref, name, reading)})


def _interval_on_delta(
    interval: tuple[float, float] | None,
    ruler: MeasureRuler,
    forms: PageForms,
    level: float | None = None,
    *,
    form: MoneyForm | None = None,
    places: ContrastPlaces | None = None,
    said_level: str | None = None,
    measure: str,
    kind: str,
) -> str | None:
    """An interval on a delta, on the measure's ruler in the unit a change is stated in
    (:meth:`~threetears.evals.analysis.report.forms.PageForms.ruled_change_interval`): money's ends in the form its
    change reads in (``form``); any other unit's at its reading's one precision for changes (``places``), both ends
    alike (``[-6.9, 11.0] pts``), and never past what a change can be (a score's from -4 to 4 on a 1–5 scale).

    The level it holds at is said once, in the table's header, so a cell carries it only where a caller asks — a
    program's row (:func:`contrast_rows`), which has no header to carry it.
    """
    if interval is None:
        return None
    text = forms.ruled_change_interval(
        ruler, interval[0], interval[1], form=form, places=places, measure=measure, kind=kind
    ).text
    if said_level is not None:
        return f"{text} at {said_level}"
    return text if level is None else f"{text} at {_level_text(level, forms)}"


def _level_text(level: float, forms: PageForms) -> str:
    """An interval's level as a reader reads it: ``99%``."""
    return forms.in_unit(100 * level, PERCENT, role="confidence").text


def _cell_namer(surface: DecisionSurface, names: dict[str, ArmName]) -> Callable[[str, str], str]:
    """Name a cell by its two coordinates as every table does: the arm's label, its test setup where it ran under two."""
    multi = multi_rig_variants(surface.cells)
    namer = _namer(names)

    def name(variant_key: str, apparatus_class_id: str) -> str:
        rig = f" (test setup {short_digest(apparatus_class_id)})" if variant_key in multi else ""
        return namer(variant_key) + rig

    return name


#: The column a stratum's cases are listed under when they declare no stratum, in an arm where others do.
NO_STRATUM = "(no stratum)"


def _printed_strata(bundle: AnalysisContextBundle) -> dict[str, str]:
    """Each stratum the aspect's cells and trials declare, by id, to the name its page prints for it
    (:func:`~threetears.evals.analysis.report.words.stratum_names`: its declared name, else its id humanized, else, where
    two would print alike, its id)."""
    ids = [held.stratum for cell in bundle.cell_measures for held in cell.strata if held.stratum is not None]
    ids += [trial.stratum for trial in bundle.trials if trial.stratum]
    return stratum_names(sorted(set(ids)), bundle.stratum_names)


def _stratum_word(name: str | None, printed: Mapping[str, str]) -> str:
    """A stratum as a column header, a case's row and a disclosure print it: its printed name
    (:func:`_printed_strata`), else :data:`NO_STRATUM`. A program reads a stratum by its id
    (:meth:`~threetears.evals.quick.result.Result.strata`)."""
    return NO_STRATUM if name is None else printed.get(name) or stratum_name(name)


def _interval(
    low: float | None,
    high: float | None,
    ruler: MeasureRuler,
    forms: PageForms,
    places: ContrastPlaces | None = None,
    *,
    measure: str,
    kind: str,
) -> str:
    """A level's interval in its one form, `` (low to high)`` on its ruler, after the figure it follows, which carries
    the unit (:func:`_unit_on`), so the interval writes none but money's ``$`` on each end
    (:meth:`~threetears.evals.analysis.report.forms.PageForms.ruled_interval`); nothing when either end is
    unestimated."""
    if low is None or high is None:
        return ""
    printed = forms.ruled_interval(ruler, low, high, places, measure=measure, kind=kind)
    return f" ({printed.text if is_money(ruler.unit) else printed.digits})"


def _unit_on(text: str, ruler: MeasureRuler) -> str:
    """A level's text with its ruler's unit on the figure (``2.42 s``), as every figure followed by its interval
    writes it; a percent and money carry theirs already, a bare figure none, an absent one none."""
    if not ruler.unit or ruler.unit == PERCENT or is_money(ruler.unit) or text == ABSENT:
        return text
    return unit_spaced(text, ruler.unit)


def _case_numbers(bundle: AnalysisContextBundle) -> dict[str, int]:
    """Each case's number, as the cases table orders it: by label, from 1."""
    labels = {trial.case_id: trial.case_label for trial in bundle.trials}
    return {case: index for index, case in enumerate(sorted(labels, key=lambda case: (labels[case], case)), start=1)}


def _case_namer(bundle: AnalysisContextBundle) -> Callable[[str], str]:
    """Name a case as every table does: by the name its author gave it where its label reads as one
    (``double-charge``), else ``case N`` — never its first input."""
    labels = {trial.case_id: trial.case_label for trial in bundle.trials}
    numbers = _case_numbers(bundle)

    def name(case: str) -> str:
        if case not in numbers:
            return humanize("case", case) or "a case"
        return humanize("case", labels[case]) or f"case {numbers[case]}"

    return name


def _listed(ids: Sequence[str]) -> str:
    """Ids, as a disclosure lists them."""
    return ", ".join(ids)


__all__ = [
    "NO_STRATUM",
]
