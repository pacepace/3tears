"""Finding a quoted stretch in the text it quotes: the verbatim check every claim excerpt and evidence quote passes.

A model asked to copy a stretch of text rarely returns its exact characters. It straightens curly quotes, turns a dash
into a hyphen, drops a soft hyphen or a zero-width character, changes case and closes up a run of spaces. None of that
changes what was said, so :func:`verbatim_span` folds both sides the same way before it looks, and maps a match back to
offsets in the original. What a caller stores is the original's own slice at those offsets, never the model's string.
A changed word never matches, and neither does an ellipsis: an elided quote is not verbatim (:func:`is_elided`).

**The fold** (:data:`VERBATIM_FOLD`): Unicode NFKC and case folding; typographic quotes to ``'`` and ``"`` and
typographic dashes and the minus sign to ``-``; the soft hyphen and the zero-width characters removed; every
whitespace run (a no-break space included, which NFKC makes a space) as one space. A base character and the combining
marks after it fold as one unit, so a decomposed accent matches a composed one, and a match never starts or ends
inside a unit: ``ss`` finds ``ß`` whole, never half of it. Every step that checks a quote with this helper names
:data:`VERBATIM_FOLD` in its identity, so a changed fold is a changed definition that never pools with the old one.
"""

from __future__ import annotations

import unicodedata
from typing import Final

#: What the fold is called in the identity of every step that checks a quote with it. Raise it with any change to
#: :func:`verbatim_span`'s fold.
VERBATIM_FOLD: Final = "verbatim-fold-v1"

#: Characters that fold to nothing: the soft hyphen and the zero-width space, non-joiner, joiner, word joiner and
#: byte-order mark.
_REMOVED: Final = frozenset("­​‌‍⁠﻿")

#: Typographic quotes and dashes, as the ASCII a model writes in their place.
_TO_ASCII: Final = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "‹": "'",
        "›": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "«": '"',
        "»": '"',
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
    }
)

#: An ellipsis once folded: NFKC writes U+2026 as three full stops.
_ELLIPSIS: Final = "..."


def _units(text: str) -> list[tuple[int, int]]:
    """Each fold unit of ``text`` as its ``(start, end)`` offsets: a character and the combining marks after it."""
    units: list[tuple[int, int]] = []
    for index, character in enumerate(text):
        if units and unicodedata.combining(character):
            units[-1] = (units[-1][0], index + 1)
        else:
            units.append((index, index + 1))
    return units


def _fold(text: str) -> tuple[str, list[tuple[int, int]]]:
    """``text`` folded, and for each folded character the offsets of the unit of ``text`` it came from.

    :param text: The text to fold.
    :return: The folded text, and one ``(start, end)`` per folded character.
    """
    folded: list[str] = []
    origins: list[tuple[int, int]] = []
    for start, end in _units(text):
        for character in unicodedata.normalize("NFKC", text[start:end]).casefold().translate(_TO_ASCII):
            if character in _REMOVED:
                continue
            if character.isspace():
                if folded and folded[-1] == " ":
                    origins[-1] = (origins[-1][0], end)
                    continue
                character = " "
            folded.append(character)
            origins.append((start, end))
    return "".join(folded), origins


def is_elided(needle: str) -> bool:
    """Whether ``needle`` holds an ellipsis (``…``, or three full stops), which no verbatim stretch may.

    :param needle: The stretch a model quoted.
    :return: True when it holds one.
    """
    return _ELLIPSIS in unicodedata.normalize("NFKC", needle)


def verbatim_span(needle: str, haystack: str) -> tuple[int, int] | None:
    """Where ``needle`` stands in ``haystack``, both folded alike (:data:`VERBATIM_FOLD`), as offsets in the original.

    ``haystack[start:end]`` is the original's own stretch, curly quotes, soft hyphens and all, so a caller stores that
    slice rather than ``needle``. The first match is the one returned.

    :param needle: The stretch a model quoted. Its ends' whitespace is ignored.
    :param haystack: The text it says it quotes.
    :return: ``(start, end)`` in ``haystack``, or None when the folded needle is not in the folded haystack, is blank,
        holds an ellipsis (:func:`is_elided`), or matches only from or to the middle of a fold unit.
    """
    if is_elided(needle):
        return None
    sought = _fold(needle)[0].strip()
    if not sought:
        return None
    folded, origins = _fold(haystack)
    at = folded.find(sought)
    while at != -1:
        last = at + len(sought) - 1
        starts_a_unit = at == 0 or origins[at - 1] != origins[at]
        ends_a_unit = last == len(folded) - 1 or origins[last + 1] != origins[last]
        if starts_a_unit and ends_a_unit:
            return origins[at][0], origins[last][1]
        at = folded.find(sought, at + 1)
    return None


__all__ = ["VERBATIM_FOLD", "is_elided", "verbatim_span"]
