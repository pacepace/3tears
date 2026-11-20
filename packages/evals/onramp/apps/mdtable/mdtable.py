"""Markdown tables for the staff handbook, with their columns lined up so the source reads as well as the page.

``format_table(header, rows, align=None)`` writes a GitHub-flavoured Markdown table:

- every cell is padded with spaces to its column's width, the width of the column's widest cell (at least 3), with one
  space between a cell and the pipes either side;
- ``align`` gives each column's alignment, ``"l"``, ``"r"`` or ``"c"``; the rule under the header is dashes the
  column's width, with a colon at the left (``:--``), the right (``--:``) or both ends (``:-:``) to match, and a cell
  is padded on the right, the left, or both (the odd space going on the right). A column with no alignment given is
  padded as a left-aligned one, and its rule is dashes alone;
- a cell's width is the number of characters a reader sees: an accented letter counts once whether it arrives as one
  character (``é``) or as a letter followed by a combining accent (``e`` and U+0301), as text copied out of some apps
  does;
- whitespace inside a cell is collapsed to single spaces, invisible format characters (zero-width spaces, byte-order
  marks) are removed, and a ``|`` in a cell is escaped as ``\\|`` (which counts as two characters);
- a row shorter than the header is filled with empty cells; a longer one is refused.

Cells are written as they arrive otherwise: no character in them is changed.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

MIN_WIDTH = 3


class TableError(ValueError):
    """A table that cannot be written: a row with more cells than the header, or an unknown alignment."""


def clean(cell: object) -> str:
    """A cell's text as the table writes it: whitespace collapsed, format characters gone, pipes escaped."""
    text = "" if cell is None else str(cell)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return " ".join(text.split()).replace("|", "\\|")


def display_width(text: str) -> int:
    """How many characters a reader sees in ``text``."""
    return len(text)


def pad(text: str, width: int, align: str) -> str:
    """``text`` padded with spaces to ``width`` seen characters."""
    gap = max(width - display_width(text), 0)
    if align == "r":
        return " " * gap + text
    if align == "c":
        return " " * (gap // 2) + text + " " * (gap - gap // 2)
    return text + " " * gap


def rule(width: int, align: str) -> str:
    """The dashes under the header for one column."""
    if align == "r":
        return "-" * (width - 1) + ":"
    if align == "c":
        return ":" + "-" * (width - 2) + ":"
    if align == "l":
        return ":" + "-" * (width - 1)
    return "-" * width


def format_table(header: Sequence[object], rows: Sequence[Sequence[object]], align: Sequence[str] | None = None) -> str:
    """The table as Markdown lines joined by newlines, with no trailing newline."""
    names = [clean(cell) for cell in header]
    body = []
    for row in rows:
        if len(row) > len(names):
            raise TableError(f"a row has {len(row)} cells and the header {len(names)}")
        body.append([clean(cell) for cell in row] + [""] * (len(names) - len(row)))
    aligns = list(align) if align is not None else []
    if any(a not in ("l", "r", "c") for a in aligns):
        raise TableError(f"alignment must be l, r or c: {aligns}")
    marks = aligns + [""] * (len(names) - len(aligns))
    widths = [max([MIN_WIDTH, *(display_width(r[i]) for r in [names, *body])]) for i in range(len(names))]

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(pad(c, w, m or "l") for c, w, m in zip(cells, widths, marks, strict=True)) + " |"

    rules = "| " + " | ".join(rule(w, m) for w, m in zip(widths, marks, strict=True)) + " |"
    return "\n".join([line(names), rules, *(line(r) for r in body)])
