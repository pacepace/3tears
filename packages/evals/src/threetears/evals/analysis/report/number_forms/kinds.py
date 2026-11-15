"""What a page's forms are keyed by: how a measure's figures read, a measure's kind, what a figure is to its
measure (:data:`Role`), and which page a set of forms is for.
"""

from __future__ import annotations

from typing import Literal


#: How a measure's figures read: a rate in percent (its changes in points), a judged score on the 1–5 rubric, money,
#: or any other measured unit (a count per trial, a time).
_ReadsAs = Literal["rate", "score", "money", "measured"]

#: Which family a measure's figures come from, as a contrast names it: ``measure``, ``judged``, or ``trial`` (how each
#: trial ended, the pass rate); None where a contrast names none, which every decision reads as a measure's own. Free
#: text, as the report input stores it.
_MeasureKind = str | None

#: What a figure is to its measure, or which fixed form a figure that is no measurement takes.
Role = Literal[
    "level",
    "change",
    "interval",
    "count",
    "p",
    "effect",
    "confidence",
    "declared",
    "share",
    "multiple",
    "money_total",
    "lone",
    "reading",
    "echoed",
    "hits",
    "money",
    "check",
    "preference",
    "pass_k",
    "half_width",
    "column",
]

#: Which page a set of forms is for: a written page (a report input), a campaign's code page, an aspect's own page
#: (code's page over one aspect's bundle), or an off-page surface (a command line's lines, a lens's text, the rung-5
#: page's side-by-side figures), which states no page's measure.
_PageKind = Literal["written", "code", "aspect", "off"]


__all__ = [
    "Role",
]
