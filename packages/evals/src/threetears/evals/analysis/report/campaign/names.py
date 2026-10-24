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
:func:`~threetears.evals.analysis.style.measure_name` names it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.style import measure_name
from threetears.evals.kernel.declaration import printed_aspect_name

__all__ = ["AspectNames", "aspect_name", "legend", "measure_name_of"]


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
    """What the page calls each of a campaign's aspects, by its label (:func:`aspect_name` of each one's evidence).

    :ivar by_label: Each aspect's name, by its label, in the campaign's order.
    """

    by_label: Mapping[str, str]

    @classmethod
    def of(cls, bundles: Iterable[AnalysisContextBundle]) -> AspectNames:
        """The names of the aspects ``bundles`` read.

        :param bundles: Each aspect's evidence, in the campaign's order.
        :ptype bundles: Iterable[AnalysisContextBundle]
        :return: Their names.
        :rtype: AspectNames
        """
        return cls({bundle.aspect: aspect_name(bundle) for bundle in bundles})

    def __call__(self, label: str) -> str:
        """What the page calls the aspect labelled ``label``; a label no evidence was given for is printed whole.

        :param label: The aspect's label.
        :ptype label: str
        :return: Its name on the page.
        :rtype: str
        """
        return self.by_label.get(label, label)


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
