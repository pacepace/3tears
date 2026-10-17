"""Text a model reads, kept under the heading it belongs to.

A tool's result is often a list: a heading per item (a path, a title, a URL)
and the item's own text under it. Written as ``f"      {excerpt}"``, only the
first line of a many-line text gets the indent; its other lines land at the
margin, where nothing says which heading they belong to. An agent read one
note's lines as the note above it and gave that note the other's size and tag
(metallm acceptance run, 2026-10-16).

So a text that can hold line breaks goes through :func:`under`, which indents
every line, or :func:`one_line`, which makes it one. It lives here, in the
leaf every package depends on, so a package held to the leaf floor (search,
SR-L7) uses it as any other does. The enforcement check
:func:`threetears.enforcement.text_blocks.unbroken_text` finds a free text
interpolated after an indent that went through neither.
"""

from __future__ import annotations

import textwrap

__all__ = ["one_line", "under"]


def under(text: str, indent: str, *, first: str | None = None) -> list[str]:
    """``text`` as lines that all sit under one heading: each indented, none blank.

    The text's own indentation is kept relative to its least-indented line, so a
    list or a code block inside it keeps its shape. Blank lines are dropped: a
    blank line is where a reader expects the next item.

    :param text: the text, of any number of lines
    :ptype text: str
    :param indent: what every line starts with
    :ptype indent: str
    :param first: what the first line starts with in place of ``indent``: a list marker (``"  - "``),
        or the item's heading (``"- [chunk:<id>] "``)
    :ptype first: str | None
    :return: the lines; for a text with nothing in it, ``first`` alone (trimmed) when given, so
        the heading and the id it carries are still shown, else ``[]``
    :rtype: list[str]
    """
    lines = [line.rstrip() for line in textwrap.dedent(text.expandtabs(4)).splitlines() if line.strip()]
    if not lines:
        return [first.rstrip()] if first is not None else []
    return [((first if first is not None and i == 0 else indent) + line) for i, line in enumerate(lines)]


def one_line(text: str) -> str:
    """``text`` on one line: its lines joined by one space, each trimmed, blank ones dropped.

    For a text read as a single line of a list: a title, a name, a short reason.
    Spacing inside a line is kept: a listing that sets a path two spaces from its
    facts is read by splitting there.

    :param text: the text
    :ptype text: str
    :return: it, on one line
    :rtype: str
    """
    return " ".join(line.strip() for line in text.splitlines() if line.strip())
