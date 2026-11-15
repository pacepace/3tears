"""An interval's level, an interval's words and a count, as every surface prints them, and which of a figure key's
figures a table states.
"""

from __future__ import annotations

import re

from threetears.evals.analysis.report.keys.figure import Figure
from threetears.evals.analysis.report.keys.vocabulary import joint_level
from threetears.evals.analysis.report.parts.look import Quantity, figure_text
from threetears.evals.analysis.report.number_forms.policy import policy_level
from threetears.evals.analysis.report.parts.words import FAMILY_LEVEL_TEMPLATE
from threetears.evals.analysis.report.unit_once import interval_beside
from threetears.evals.analysis.report_input import ContrastFamily
from threetears.evals.kernel.numbers import spell_count


# --- an interval's level, an interval's words and a count, as every surface prints them -------------------------------


def interval_level(level: str, family: ContrastFamily | None) -> str:
    """An interval's level as every heading prints it, a chart's head and a table's column alike: where ``family``
    corrects its comparisons together at this level, the joint guarantee they hold at, in plain words (``95% (adjusted
    for 16 comparisons)``, which a chart's head prints ``95% intervals (adjusted for 16 comparisons)``), so a reader
    meets the guarantee they act on; any other level as the one rule prints a level as text
    (:func:`~threetears.evals.analysis.report.number_forms.policy.policy_level`: ``95%``, an uncorrected 98.33% as
    ``98.3%``), so a level reads alike in a head, a cell and the prose (owner ruling, 2026-11-14). A level the input
    states in no percent's digits reads as stated. The per-interval level is Method's and the glossary's.

    :param level: The level as the input states it (``99.69%``).
    :ptype level: str
    :param family: The contrast family the interval belongs to, or None.
    :ptype family: ContrastFamily | None
    :return: The level's words.
    :rtype: str
    """
    stated = level.strip()
    joint = joint_level(family) if family is not None and family.level == level else None
    if joint is None:
        return level_text(stated)
    return FAMILY_LEVEL_TEMPLATE.format(joint=policy_level(joint[0] * 100), size=joint[1])


#: A level as an input states it in percent: its digits, then ``%``.
_STATED_PERCENT = re.compile(r"(\d+(?:\.\d+)?)%")


def level_text(stated: str) -> str:
    """A level the input states as text (``98.33%``), as the one rule prints a level
    (:func:`~threetears.evals.analysis.report.number_forms.policy.policy_level`: ``98.3%``); one the input states in no
    percent's digits as stated.

    :param stated: The level as the input states it.
    :ptype stated: str
    :return: Its text.
    :rtype: str
    """
    found = _STATED_PERCENT.fullmatch(stated.strip())
    return policy_level(float(found[1])) if found is not None else stated.strip()


def _interval_words(low: float, high: float, drawn: Quantity) -> str:
    """An interval in a tooltip: its two ends by the package's number rule, joined as prose joins them.

    :param low: Its low end, in display units.
    :ptype low: float
    :param high: Its high end, in display units.
    :ptype high: float
    :param drawn: The quantity drawn.
    :ptype drawn: Quantity
    :return: The words (``-10.8 to -3.4``).
    :rtype: str
    """
    return interval_beside(figure_text(low, drawn), figure_text(high, drawn))


def _count_text(count: float) -> str:
    """A count as a chart's words print it: the figure keys' whole-number spelling, thousands separated
    (``1,234``), so a count reads alike on a chart and in the prose beside it.

    :param count: The count.
    :ptype count: float
    :return: The text.
    :rtype: str
    """
    return spell_count(round(count))


#: How every family spells a level the tables print and its interval's two ends.
_LEVEL_FIGURES = frozenset({"mean", "low", "high"})

#: How the criteria families spell the level of the measure a criterion reads, and how they and the weakest margin
#: spell its gap from the threshold.
_CRITERION_LEVEL = "value"
_CRITERION_GAPS = frozenset({"gap", "margin"})

#: A plain end's digits: a sign, the digits and a decimal part, never a bound or grouped thousands.
_PLAIN_END = re.compile(r"[+\-]?\d+(?:\.\d+)?")


def _tabled(figure: Figure) -> bool:
    """Whether a key's figure is one the page's tables print of a measure or a judged score, so the key reads it at the
    table's precision. Decided by what the figure is, not by how its family spells it: a change of any family, its
    interval and that interval's ends (each resolver marks them :attr:`~threetears.evals.analysis.report.figure_keys.
    Figure.change`, whether its family spells the change ``delta``, ``difference``, ``effect``, ``change`` or
    ``vs_control``); any interval; and a level, which every family but a criterion's spells ``mean``, with its
    interval's two ends, or a key naming no statistic, which reads its family's own figure; a criterion's figure is the
    level of the measure it reads (``Figure.measure``), which its family spells ``value``, and its ``gap`` (the weakest
    margin's ``margin``) a difference on that measure. No other statistic is the table's figure: an adjusted p of
    0.0031 is never a score to two decimals, nor a half-width or a standard error a change."""
    statistic = figure.statistic
    if figure.change or figure.kind == "interval" or statistic is None:
        return True
    return statistic in _LEVEL_FIGURES or (bool(figure.measure) and statistic in (_CRITERION_LEVEL, *_CRITERION_GAPS))


__all__ = [
    "level_text",
]
