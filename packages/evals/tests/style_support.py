"""The style rules applied to a whole rendered page: test support, since nothing in the package lints a page.

The package applies :func:`~threetears.evals.analysis.style.lint_prose` to nothing it writes at run time; its tests hold
the text code writes to the rules by linting the rendered ``report.md`` through here. :func:`absence_violations` holds a
page's first screen to saying what the page is, never what it is not (the presentation review's A6).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.report.campaign.names import measure_name_of
from threetears.evals.analysis.report.model import ReportBlock
from threetears.evals.analysis.report.verdict import guardrail_name
from threetears.evals.analysis.style import StyleViolation, lint_prose

__all__ = ["Absence", "absence_violations", "lint_page", "reading_names", "unlinked"]

#: A line quoting something whole, as the declared question does under the title: data, not prose.
_QUOTED_LINE = re.compile(r'^(?:[A-Z][a-z]+(?: [a-z]+)*: )?".+"$')

#: A header's closing unit, which names what its column's figures are in (``(%)``, ``($/1,000 replies)``): a name
#: with its unit is still the name.
_HEADER_UNIT = re.compile(r"\s*\([^()]*\)$")


def lint_page(markdown: str, *, question: str | None = None, names: Collection[str] = ()) -> list[StyleViolation]:
    """Every rule a rendered ``report.md`` breaks, outside the data it quotes.

    Headings and table headers are held to the header rule; prose lines to every rule. Table body cells, fenced
    blocks, folded trial details, the methods' disclosures fold and a line quoting something whole (the declared
    question under the title) are data and quoted text, not prose.

    :param markdown: The page.
    :ptype markdown: str
    :param question: The declared question, which only its own quoted line may carry.
    :ptype question: str | None
    :param names: The arms' names, which head a column as a name, not a header's words.
    :ptype names: Collection[str]
    :return: The violations, in page order.
    :rtype: list[StyleViolation]
    """
    found: list[StyleViolation] = []
    fenced = False
    folded = 0
    header_row = False
    titled = False
    for line in markdown.splitlines():
        if line.startswith("```"):
            fenced = not fenced
            continue
        folded += line.count("<details") - line.count("</details>")
        if fenced or folded > 0 or line.startswith("</details>") or not line.strip():
            header_row = False
            continue
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split(" | ")]
            if not header_row and not line.startswith("|---"):
                bare = {_HEADER_UNIT.sub("", name) for name in names}
                headers = [cell for cell in cells if _HEADER_UNIT.sub("", cell.removeprefix("▸ ")) not in bare]
                found += [v for cell in headers for v in lint_prose(cell, header=True) if v.rule == "header"]
            header_row = True
            continue
        header_row = False
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            # The page's own title, its first level-one heading, is a name: held to no header length. Any other heading is.
            title = not titled and line.startswith("# ")
            titled = titled or title
            found += lint_prose(heading, header=not title)
            continue
        if _QUOTED_LINE.match(line):
            continue
        found += lint_prose(line.lstrip("-> *").strip(), question=question)
    return found


#: The words a line states an absence by: what a page does not hold, rather than what it does.
_ABSENCE = re.compile(r"\b(?:No|Nothing|nothing|not|never|none)\b(?:\s+[\w-]+)?")

#: The first-screen sections and sources a line may state an absence in: what warns (a campaign's ``warnings``; an
#: aspect page's broken floor, missed bar and guardrail lines and its case-floor caveat), where an absence is the
#: warning, and the question, which is the author's words, quoted.
_ABSENCE_SECTIONS = frozenset({"warnings"})
_ABSENCE_SOURCES = frozenset({"floor", "bar", "guardrails", "caveat", "basis"})

#: The id prefix of a report-wide caveat line, which states an absence where that absence changes the reading.
_CAVEAT_ID = "caveat-"

#: Code a reader copies, not prose.
_CODE_SPAN = re.compile(r"`[^`]*`")


@dataclass(frozen=True)
class Absence:
    """One first-screen line that states an absence: the line, and the absence's words (``not separated``)."""

    text: str
    detail: str


def reading_names(bundles: Iterable[AnalysisContextBundle]) -> list[str]:
    """Every measure and guardrail name a page over ``bundles`` prints, as it prints them: names, which
    :func:`absence_violations` masks as it masks code (a guardrail named ``No invented total score`` states no
    absence)."""
    names: set[str] = set()
    for bundle in bundles:
        names |= {measure_name_of(bundle, key) for key in bundle.measure_catalog}
        names |= {guardrail_name(bundle, check.name, check.reading) for check in bundle.guardrails.checks}
    return sorted(names, key=len, reverse=True)


def absence_violations(blocks: Iterable[ReportBlock], *, names: Iterable[str] = ()) -> list[Absence]:
    """Every line of a page's first screen that opens or turns on an absence (``No``, ``Nothing``, ``not``,
    ``never``, ``none``): the page says what it is, never what it is not. What warns and the report-wide caveats are
    exempt, since there the absence is the warning or changes the reading, and so is the question quoted; a block
    with no words (a table, a chart) holds none.

    :param blocks: The page's first screen, in order (its blocks in the first-screen sections).
    :ptype blocks: Iterable[ReportBlock]
    :param names: The page's measure and guardrail names (:func:`reading_names`), masked before matching.
    :ptype names: Iterable[str]
    :return: One per absence found; empty when the first screen says only what the page is.
    :rtype: list[Absence]
    """
    found: list[Absence] = []
    masks = sorted(set(names), key=len, reverse=True)
    for block in blocks:
        text = getattr(block, "text", None)
        if not isinstance(text, str) or block.section in _ABSENCE_SECTIONS:
            continue
        if getattr(block, "source", None) in _ABSENCE_SOURCES:
            continue
        if str(getattr(block, "anchor", None) or "").startswith(_CAVEAT_ID):
            continue
        masked = _CODE_SPAN.sub("", text)
        for name in masks:
            masked = masked.replace(name, "")
        found += [Absence(text, match.group(0)) for match in _ABSENCE.finditer(masked)]
    return found


#: A link the page puts on a glossary term where its prose first uses it: a bare ``<a>`` holding the term's words.
_FIRST_USE_LINK = re.compile(r'<a href="#[^"]*" title="[^"]*">([^<]*)</a>')


def unlinked(page: str) -> str:
    """A rendered HTML page with each glossary term's first-use link unwrapped to its words, so a test reads a line's
    words as the page prints them whatever terms it first uses.

    :param page: The page's HTML.
    :ptype page: str
    :return: The page, those links unwrapped.
    :rtype: str
    """
    return _FIRST_USE_LINK.sub(r"\1", page)
