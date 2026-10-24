"""What a campaign's page calls an aspect: the one place a name is chosen, so every part names an aspect alike.

Every sentence and table cell the page writes about an aspect takes its name from here: :func:`aspect_name` for one
aspect's evidence, and :class:`AspectNames` for the parts that hold only an aspect's label. Identifiers (a floor's
key, a fold's id, a link) keep the label. Today an aspect is called by its label, as declared. A name is declared,
never derived: the fallback is always the full label, never a shortening of it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle

__all__ = ["AspectNames", "aspect_name"]


def aspect_name(bundle: AnalysisContextBundle) -> str:
    """What the page calls the aspect ``bundle`` reads: its label, as declared.

    :param bundle: The aspect's evidence.
    :ptype bundle: AnalysisContextBundle
    :return: The aspect's name on the page.
    :rtype: str
    """
    return bundle.aspect


@dataclass(frozen=True)
class AspectNames:
    """What the page calls each of a campaign's aspects, by its label (:func:`aspect_name` of each one's evidence).

    :ivar by_label: Each aspect's name, by its label.
    """

    by_label: Mapping[str, str]

    @classmethod
    def of(cls, bundles: Iterable[AnalysisContextBundle]) -> AspectNames:
        """The names of the aspects ``bundles`` read.

        :param bundles: Each aspect's evidence.
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
