"""What a campaign's page calls an aspect: the one place a name is chosen, so every part names an aspect alike.

Every sentence and table cell the page writes about an aspect takes its name from here: :func:`aspect_name` for one
aspect's evidence, and :class:`AspectNames` for the parts that hold only an aspect's label. Identifiers (a floor's
key, a fold's id, a link) keep the label. An aspect is called by the short name its campaign declares
(``Aspect.short_name``), else by its label, whole. A name is declared, never derived: the fallback is always the full
label, never a shortening of it. Where a short name stands in for a label, :func:`legend` gives the pairs a page
prints once, so a reader can read each short name back to its label.

A measure is named here too, by :func:`measure_name_of`: a goal check by the display name its template declares
(``EvalTemplate.goal_check_names``, carried on the bundle beside the catalog and never in it, so a name moves no
reading or fingerprint), else ``Goal check``, numbered where the aspect reads several
(``AnalysisContextBundle.named_checks``); every other measure as :func:`~threetears.evals.analysis.style.measure_name`
names it. No header or lead prints a check's expression: the page's glossary pairs each check's name with it, once
(``CampaignReport.glossary``), so a reader can read each name back to the expression it stands for.
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

__all__ = ["AspectNames", "aspect_name", "legend", "measure_name_of"]


def aspect_name(bundle: AnalysisContextBundle, *, campaign: str | None = None) -> str:
    """What the page calls the aspect ``bundle`` reads: its declared short name, else its label, as declared; a label
    the engine gave (``classify``, ``score``) gives way to the campaign's name where the page has it
    (:func:`~threetears.evals.kernel.declaration.printed_aspect_name`).

    :param bundle: The aspect's evidence.
    :ptype bundle: AnalysisContextBundle
    :param campaign: The campaign's name; None keeps every label.
    :ptype campaign: str | None
    :return: The aspect's name on the page.
    :rtype: str
    """
    return printed_aspect_name(bundle.aspect, bundle.short_name, campaign=campaign)


@dataclass(frozen=True)
class AspectNames:
    """What the page calls each of a campaign's aspects, by its label (:func:`aspect_name` of each one's evidence),
    and each aspect's goal checks, for a part that holds only an aspect's label.

    :ivar by_label: Each aspect's name, by its label, in the campaign's order.
    :ivar checks: Each aspect's check names (``AnalysisContextBundle.named_checks``: a check's expression to its
        declared name, else ``Goal check``), by the aspect's label.
    :ivar folds: The id of the section around each aspect's fold on the campaign's page (:func:`~threetears.evals.analysis.report.model.fold_anchor`), by the aspect's label: where a cell naming the
        aspect links.
    """

    by_label: Mapping[str, str]
    checks: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    folds: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def of(cls, bundles: Iterable[AnalysisContextBundle], *, campaign: str | None = None) -> AspectNames:
        """The names of the aspects ``bundles`` read.

        :param bundles: Each aspect's evidence, in the campaign's order.
        :ptype bundles: Iterable[AnalysisContextBundle]
        :param campaign: The campaign's name, which an engine-given label gives way to (:func:`aspect_name`); None
            keeps every label.
        :ptype campaign: str | None
        :return: Their names.
        :rtype: AspectNames
        """
        held = list(bundles)
        labels = [bundle.aspect for bundle in held]
        return cls(
            {bundle.aspect: aspect_name(bundle, campaign=campaign) for bundle in held},
            {bundle.aspect: bundle.named_checks() for bundle in held},
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
        """The names the aspect labelled ``label`` gives its goal checks, by check expression
        (``AnalysisContextBundle.named_checks``): what a part naming a check on that aspect passes on
        (``floor_statement``'s ``check_names``); none for an aspect no evidence was given for.

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


def measure_name_of(bundle: AnalysisContextBundle, measure: str) -> str:
    """What the page calls ``measure`` on the aspect ``bundle`` reads.

    A goal check's measure by the display name the aspect's template declares for it, else ``Goal check``, numbered
    where the aspect reads several (``AnalysisContextBundle.check_name``); any other measure by its catalog reader
    name, else its key humanized
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
