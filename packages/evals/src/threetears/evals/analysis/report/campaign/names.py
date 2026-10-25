"""What a campaign's page calls an aspect: the one place a name is chosen, so every part names an aspect alike.

Every sentence and table cell the page writes about an aspect takes its name from here: :func:`aspect_name` for one
aspect's evidence, and :class:`AspectNames` for the parts that hold only an aspect's label. Identifiers (a floor's
key, a fold's id, a link) keep the label. An aspect is called by the short name its campaign declares
(``Aspect.short_name``), else by its label, whole. A name is declared, never derived: the fallback is always the full
label, never a shortening of it. Where a short name stands in for a label, :func:`legend` gives the pairs a page
prints once, so a reader can read each short name back to its label.

A measure is named here too, by :func:`measure_name_of`: a goal check by the display name its template declares
(``EvalTemplate.goal_check_names``, carried on the bundle beside the catalog and never in it, so a name moves no
reading or fingerprint), else by its whole expression; every other measure as
:func:`~threetears.evals.analysis.style.measure_name` names it. Where a name stands in for a check, the page prints the
pair once, in its Design & provenance (:func:`check_legend_blocks`), so a reader can read each name back to the
expression it stands for.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.report.model import (
    ReportBlock,
    TableBlock,
    TableCell,
    TableColumn,
    fold_anchor,
    fold_ids,
)
from threetears.evals.analysis.style import code, measure_name
from threetears.evals.kernel.declaration import printed_aspect_name

__all__ = ["AspectNames", "aspect_name", "check_legend_blocks", "legend", "measure_name_of"]


def aspect_name(bundle: AnalysisContextBundle) -> str:
    """What the page calls the aspect ``bundle`` reads: its declared short name, else its label, as declared.

    :param bundle: The aspect's evidence.
    :ptype bundle: AnalysisContextBundle
    :return: The aspect's name on the page.
    :rtype: str
    """
    return printed_aspect_name(bundle.aspect, bundle.short_name)


@dataclass(frozen=True)
class AspectNames:
    """What the page calls each of a campaign's aspects, by its label (:func:`aspect_name` of each one's evidence),
    and each aspect's goal checks, for a part that holds only an aspect's label.

    :ivar by_label: Each aspect's name, by its label, in the campaign's order.
    :ivar checks: Each aspect's declared check display names (``AnalysisContextBundle.goal_check_names``: a check's
        expression to its name), by the aspect's label.
    :ivar folds: The id of the section around each aspect's fold on the campaign's page (:func:`~threetears.evals.analysis.report.model.fold_anchor`), by the aspect's label: where a cell naming the
        aspect links.
    """

    by_label: Mapping[str, str]
    checks: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    folds: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def of(cls, bundles: Iterable[AnalysisContextBundle]) -> AspectNames:
        """The names of the aspects ``bundles`` read.

        :param bundles: Each aspect's evidence, in the campaign's order.
        :ptype bundles: Iterable[AnalysisContextBundle]
        :return: Their names.
        :rtype: AspectNames
        """
        held = list(bundles)
        labels = [bundle.aspect for bundle in held]
        return cls(
            {bundle.aspect: aspect_name(bundle) for bundle in held},
            {bundle.aspect: dict(bundle.goal_check_names) for bundle in held},
            {label: fold_anchor(slug) for label, slug in zip(labels, fold_ids(labels), strict=True)},
        )

    def __call__(self, label: str) -> str:
        """What the page calls the aspect labelled ``label``; a label no evidence was given for is printed whole.

        :param label: The aspect's label.
        :ptype label: str
        :return: Its name on the page.
        :rtype: str
        """
        return self.by_label.get(label, label)

    def cell(self, label: str) -> TableCell:
        """A table cell naming the aspect labelled ``label``, linked to its fold, so a reader goes from a row to the
        aspect's own page; unlinked for an aspect the page has no fold for.

        :param label: The aspect's label.
        :ptype label: str
        :return: The cell.
        :rtype: TableCell
        """
        return TableCell(value=self(label), ref=self.folds.get(label))

    def check_names(self, label: str) -> Mapping[str, str]:
        """The display names the aspect labelled ``label`` declares for its goal checks, by check expression: what a
        part naming a check on that aspect passes on (``floor_statement``'s ``check_names``); none for an aspect no
        evidence was given for, whose checks print whole.

        :param label: The aspect's label.
        :ptype label: str
        :return: Each named check's display name, by its expression.
        :rtype: Mapping[str, str]
        """
        return self.checks.get(label, {})


def legend(names: AspectNames) -> list[tuple[str, str]]:
    """The pairs one legend row prints: each aspect a short name stands in for, as ``(short name, label)``.

    In the campaign's order. An aspect printed by its label needs no entry, so a campaign declaring no short name has
    an empty legend, and a page prints none.

    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :return: ``(name, label)`` for each aspect whose name is not its label.
    :rtype: list[tuple[str, str]]
    """
    return [(name, label) for label, name in names.by_label.items() if name != label]


def _legend_rows(names: AspectNames) -> list[tuple[str, str, list[str]]]:
    """The rows the check legend prints: each declared check name with the expression it stands for and the aspects,
    by label, that declare it so, as ``(name, expression, labels)``.

    One row per name and expression, in the campaign's order and then each aspect's declared order: a check two
    aspects name alike is one row holding both, and a check two aspects name apart is a row per name. A check no
    aspect names is printed whole wherever it is named, so it needs no row, and a campaign naming no check has an
    empty legend.

    :param names: What the page calls each aspect, with each one's declared check names.
    :ptype names: AspectNames
    :return: ``(name, expression, labels)`` for each declared name.
    :rtype: list[tuple[str, str, list[str]]]
    """
    rows: dict[tuple[str, str], list[str]] = {}
    for label in names.by_label:
        for check, name in names.check_names(label).items():
            rows.setdefault((name, check), []).append(label)
    return [(name, check, labels) for (name, check), labels in rows.items()]


def check_legend_blocks(names: AspectNames) -> list[ReportBlock]:
    """A campaign's check legend, in its Design & provenance: each declared check name beside the expression it
    stands for, once, as code; with the aspects declaring it only where some row holds on fewer than every aspect.
    Nothing where no check is named.

    :param names: What the page calls each aspect, with each one's declared check names.
    :ptype names: AspectNames
    :return: The legend's table, or nothing.
    :rtype: list[ReportBlock]
    """
    rows = _legend_rows(names)
    if not rows:
        return []
    every = list(names.by_label)
    apart = any(labels != every for _, _, labels in rows)
    columns = [TableColumn(key="name", header="Check"), TableColumn(key="expression", header="Expression")]
    if apart:
        columns.append(TableColumn(key="aspects", header="Aspects", term="aspect"))
    cells: list[dict[str, TableCell]] = []
    for name, check, labels in rows:
        row = {"name": TableCell(value=name), "expression": TableCell(value=code(check))}
        if apart:
            # One aspect's cell links its fold; a list of several links none, since a cell links one place.
            row["aspects"] = TableCell(
                value=", ".join(names(label) for label in labels),
                ref=names.folds.get(labels[0]) if len(labels) == 1 else None,
            )
        cells.append(row)
    return [
        TableBlock(
            section="provenance",
            name="check_names",
            title="Checks",
            columns=columns,
            rows=cells,
            order="as the aspects declare them, in the campaign's order",
            total_rows=len(cells),
        )
    ]


def measure_name_of(bundle: AnalysisContextBundle, measure: str) -> str:
    """What the page calls ``measure`` on the aspect ``bundle`` reads.

    A goal check's measure by the display name the aspect's template declares for it, else by its whole expression,
    as code; any other measure by its catalog reader name, else its key humanized
    (:func:`~threetears.evals.analysis.style.measure_name`).

    :param bundle: The aspect's evidence.
    :ptype bundle: AnalysisContextBundle
    :param measure: The measure's key, as the bundle's readings carry it.
    :ptype measure: str
    :return: Its name on the page.
    :rtype: str
    """
    described = bundle.measure_catalog.get(measure)
    return measure_name(
        measure, described.reader_name if described is not None else None, declared=bundle.check_name(measure)
    )
