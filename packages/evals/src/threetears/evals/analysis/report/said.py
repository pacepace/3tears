"""A floor's sentence with its figures held as numbers: words, and each figure as its value and what it is.

A floor is read, and a campaign's rows are read, before any page exists, so the sentences that say why cannot be
spelled where they are read: the form a figure prints in is its page's (one form per measure,
:class:`~threetears.evals.analysis.report.forms.PageForms`). So the reading says its sentence as a :class:`Said`: the
words, and each figure in its place as a :class:`SaidFigure` (its value, its role, and the rate it is of), and the page
spells it (:meth:`~threetears.evals.analysis.report.forms.PageForms.said`).

The campaign's readings freeze their reasons as text, which the campaign's fingerprint reads, so those are spelled once
as they were frozen (:meth:`Said.stored`), and no page decides them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

from threetears.evals.kernel.numbers import POINT_PLACES, format_number

__all__ = ["Said", "SaidFigure"]


@dataclass(frozen=True)
class SaidFigure:
    """One figure a sentence states, as a number: its value, its role, and the rate it is of.

    :ivar value: The figure, in its role's unit.
    :ivar role: What it is. ``margin``: a floor's or an aspect's margin, in points, as declared. ``points``: how far a
        rate's drop reaches, in points, unsigned (``at most 8``, ``at least 12.3``). ``change``: a rate's drop in
        points, positive for a drop and negative for a gain, said in words (``a drop of 92.7``, ``a gain of 35.5``,
        ``no change``). ``level``: an interval's level, as a share (``95%``). ``count``: a count of passes, trials or
        cases, whole.
    :ivar measure: For ``points`` and ``change``, the key of the measure the rate is compared on; None for the trials'
        own pass rate. Unread for ``margin`` and ``level``, which are of no measure.
    :ivar bound: For ``points``, the bound the sentence's words state the figure as: ``lower`` (``at least``), which a
        page rounds down at its places, ``upper`` (``at most``, ``as much as``), which it rounds up, so a bound never
        reads past itself; None for a figure stated as itself, rounded to the nearest.
    :ivar margin: For a bound, the margin the sentence compares it with (``within 2.5 points (it is at most 2.4)``),
        which its printed figure never crosses, as it never crosses zero; None for none.
    """

    value: float
    role: Literal["margin", "points", "change", "level", "count"]
    measure: str | None = None
    bound: Literal["lower", "upper"] | None = None
    margin: float | None = None


@dataclass(frozen=True)
class Said:
    """A sentence as words and figures, in order, its figures spelled by whoever prints it (:meth:`words`).

    A sentence joins another, or a string, with ``+`` on either side. It is never formatted into a string as it is:
    its figures have no text until a page's forms give them one.

    :ivar parts: The words and figures, in order; adjacent words are held joined.
    """

    parts: tuple[str | SaidFigure, ...] = ()

    @classmethod
    def of(cls, *parts: str | SaidFigure | Said) -> Said:
        """A sentence of words, figures and other sentences, in order.

        :param parts: Each part.
        :ptype parts: str | SaidFigure | Said
        :return: The sentence.
        :rtype: Said
        """
        held: list[str | SaidFigure] = []
        for part in parts:
            for piece in part.parts if isinstance(part, Said) else (part,):
                if isinstance(piece, str) and held and isinstance(held[-1], str):
                    held[-1] += piece
                elif not isinstance(piece, str) or piece:
                    held.append(piece)
        return cls(tuple(held))

    def __add__(self, other: str | Said) -> Said:
        return Said.of(self, other)

    def __radd__(self, other: str) -> Said:
        return Said.of(other, self)

    def __bool__(self) -> bool:
        return bool(self.parts)

    def __format__(self, spec: str) -> str:
        raise TypeError(_UNSPELLED)

    def __str__(self) -> str:
        raise TypeError(_UNSPELLED)

    def words(self, spell: Callable[[SaidFigure], str]) -> str:
        """The sentence in words, each figure spelled by ``spell``: a ``change`` read as a drop or a gain of its size,
        spelled as ``points``, and ``no change`` where that size reads as zero.

        :param spell: A figure's text, for every role but ``change``.
        :ptype spell: Callable[[SaidFigure], str]
        :return: The sentence.
        :rtype: str
        """
        said: list[str] = []
        for part in self.parts:
            if isinstance(part, str):
                said.append(part)
            elif part.role == "change":
                size = spell(replace(part, value=abs(part.value), role="points", bound=None, margin=None))
                said.append(
                    "no change"
                    if not size.strip("0.")
                    else f"a drop of {size}"
                    if part.value > 0
                    else f"a gain of {size}"
                )
            else:
                said.append(spell(part))
        return "".join(said)

    def mentions(self, phrase: str) -> bool:
        """Whether the sentence's words hold ``phrase``, within one run of words.

        :param phrase: The words.
        :ptype phrase: str
        :return: Whether they are there.
        :rtype: bool
        """
        return any(isinstance(part, str) and phrase in part for part in self.parts)

    def stored(self) -> str:
        """The sentence as a campaign's readings freeze it, which the campaign's fingerprint reads: a margin as declared
        to four significant figures, points to a decimal, a level in percent to the hundredth that tells two apart, a
        count's digits ungrouped.

        :return: The sentence.
        :rtype: str
        """
        return self.words(_stored_figure)


#: Why a sentence held as numbers is never turned into a string as it is: its figures have no text yet, and the
#: dataclass's own text (``Said(parts=...)``) would reach a page.
_UNSPELLED = "a sentence's figures are spelled by a page's forms (PageForms.said), never turned into text as held"


def _stored_figure(figure: SaidFigure) -> str:
    """One figure as a campaign's readings froze it, which the campaign's fingerprint reads, so it never moves."""
    if figure.role == "count":
        return str(int(figure.value))
    if figure.role == "margin":
        return format_number(figure.value)
    if figure.role == "level":
        return format_number(round(100 * figure.value, 2)) + "%"
    return format_number(round(figure.value, POINT_PLACES))
