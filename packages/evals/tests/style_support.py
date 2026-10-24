"""The style rules applied to a whole rendered page: test support, since nothing in the package lints a page.

The package applies :func:`~threetears.evals.analysis.style.lint_prose` to nothing it writes at run time; its tests hold
the text code writes to the rules by linting the rendered ``report.md`` through here.
"""

from __future__ import annotations

import re
from collections.abc import Collection

from threetears.evals.analysis.style import StyleViolation, lint_prose

__all__ = ["lint_page"]

#: A line quoting something whole, as the declared question does under the title: data, not prose.
_QUOTED_LINE = re.compile(r'^(?:[A-Z][a-z]+(?: [a-z]+)*: )?".+"$')


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
                headers = [cell for cell in cells if cell.removeprefix("▸ ") not in names]
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
