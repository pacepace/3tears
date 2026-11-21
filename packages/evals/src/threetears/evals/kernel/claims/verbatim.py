"""Finding a quoted stretch in the text it quotes: the verbatim check every claim excerpt and evidence quote passes.

A model asked to copy a stretch of text rarely returns its exact characters. It straightens curly quotes, turns a dash
into a hyphen, drops a soft hyphen or a zero-width character, changes case and closes up a run of spaces. None of that
changes what was said, so :func:`verbatim_span` folds both sides the same way before it looks, and maps a match back to
offsets in the original. What a caller stores is the original's own slice at those offsets, never the model's string.
A changed word never matches. Neither does an elided quote: the folded needle must stand unbroken in the folded text,
so an ellipsis matches only where the text itself holds one.

**The fold** (:data:`VERBATIM_FOLD`): Unicode NFKC and case folding; typographic quotes to ``'`` and ``"`` and
typographic dashes and the minus sign to ``-``; the soft hyphen and the zero-width characters removed; every
whitespace run (a no-break space included, which NFKC makes a space) as one space. A base character and the combining
marks after it fold as one unit, so a decomposed accent matches a composed one, and a match never starts or ends
inside a unit: ``ss`` finds ``ß`` whole, never half of it.

**Its identity is the fold itself.** :attr:`VerbatimFold.identity` names the revision and a digest of the two tables
and of the Unicode database version NFKC and case folding read (``unicodedata.unidata_version``). Every step that checks
a quote with this fold names that identity, so an edited table or a Python whose Unicode data differs is a changed
definition that never pools with the old one, with no label to remember to raise.

Engine-internal: the claims steps import it, and nothing here is exported.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Final

from threetears.evals.schema.hashing import canonical_digest


@dataclass(frozen=True)
class VerbatimFold:
    """How two texts are folded alike before a quote is looked for, and what that fold is called.

    :ivar revision: The rule's revision, raised with any change to how :meth:`span` uses the tables.
    :ivar to_ascii: Each typographic character, and the ASCII a model writes in its place.
    :ivar removed: The characters that fold to nothing.
    """

    revision: str
    to_ascii: Mapping[str, str]
    removed: frozenset[str]

    @property
    def identity(self) -> str:
        """``<revision>:<sha12>``: the revision, and a digest of the revision, both tables and the Unicode database
        version, so a change to any of them is a changed identity."""
        digest = canonical_digest(
            {
                "revision": self.revision,
                "to_ascii": sorted(self.to_ascii.items()),
                "removed": sorted(self.removed),
                "unicode": unicodedata.unidata_version,
            }
        )
        return f"{self.revision}:{digest[:12]}"

    @cached_property
    def _table(self) -> dict[int, str]:
        return str.maketrans(dict(self.to_ascii))

    def fold(self, text: str) -> tuple[str, list[tuple[int, int]]]:
        """``text`` folded, and for each folded character the offsets of the unit of ``text`` it came from.

        :param text: The text to fold.
        :return: The folded text, and one ``(start, end)`` per folded character.
        """
        folded: list[str] = []
        origins: list[tuple[int, int]] = []
        for start, end in _units(text):
            for character in unicodedata.normalize("NFKC", text[start:end]).casefold().translate(self._table):
                if character in self.removed:
                    continue
                if character.isspace():
                    if folded and folded[-1] == " ":
                        origins[-1] = (origins[-1][0], end)
                        continue
                    character = " "
                folded.append(character)
                origins.append((start, end))
        return "".join(folded), origins

    def span(self, needle: str, haystack: str) -> tuple[int, int] | None:
        """Where ``needle`` stands in ``haystack``, both folded alike, as offsets in the original.

        ``haystack[start:end]`` is the original's own stretch, curly quotes, soft hyphens and all. The first match is
        the one returned, so a needle the text holds twice is placed at its first occurrence.

        :param needle: The stretch a model quoted. Its ends' whitespace is ignored.
        :param haystack: The text it says it quotes.
        :return: ``(start, end)`` in ``haystack``, or None when the folded needle is blank, is not in the folded
            haystack, or matches only from or to the middle of a fold unit.
        """
        sought = self.fold(needle)[0].strip()
        if not sought:
            return None
        folded, origins = self.fold(haystack)
        at = folded.find(sought)
        while at != -1:
            last = at + len(sought) - 1
            starts_a_unit = at == 0 or origins[at - 1] != origins[at]
            ends_a_unit = last == len(folded) - 1 or origins[last + 1] != origins[last]
            if starts_a_unit and ends_a_unit:
                return origins[at][0], origins[last][1]
            at = folded.find(sought, at + 1)
        return None


def _units(text: str) -> list[tuple[int, int]]:
    """Each fold unit of ``text`` as its ``(start, end)`` offsets: a character and the combining marks after it."""
    units: list[tuple[int, int]] = []
    for index, character in enumerate(text):
        if units and unicodedata.combining(character):
            units[-1] = (units[-1][0], index + 1)
        else:
            units.append((index, index + 1))
    return units


#: The fold every claims step checks a quote with. Revision 2 dropped revision 1's refusal of any needle holding an
#: ellipsis, which refused only an ellipsis the text really holds.
VERBATIM_FOLD: Final = VerbatimFold(
    revision="verbatim-fold-v2",
    to_ascii={
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
    },
    # The soft hyphen and the zero-width space, non-joiner, joiner, word joiner and byte-order mark.
    removed=frozenset("­​‌‍⁠﻿"),
)


def verbatim_span(needle: str, haystack: str) -> tuple[int, int] | None:
    """Where ``needle`` stands in ``haystack`` under :data:`VERBATIM_FOLD` (:meth:`VerbatimFold.span`).

    :param needle: The stretch a model quoted.
    :param haystack: The text it says it quotes.
    :return: ``(start, end)`` in ``haystack``, or None.
    """
    return VERBATIM_FOLD.span(needle, haystack)


__all__: list[str] = []
