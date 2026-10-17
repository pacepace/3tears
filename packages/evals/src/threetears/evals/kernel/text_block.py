"""Free text kept under the item it belongs to, every line of it.

A list the engine prints -- a run's summary, an action's help, the memo a judge reads -- puts an item's
text after an indent or a list marker. Written as ``f"  error: {error}"``, only the first line of a
many-line text gets that place; its other lines land at the margin, where they read as the next item.
So a text that can hold line breaks goes through :func:`under`, which indents every line, or
:func:`one_line`, which makes it one; the repo's ``tests/enforcement/test_text_sits_under_its_heading.py``
finds one that went through neither.

The same two functions as :mod:`threetears.observe.text_block`, with the same behaviour: the engine
depends on no other 3tears package, so it keeps its own copy rather than importing that one. Internal:
no public root exports either name.
"""

from __future__ import annotations

import textwrap


def under(text: str, indent: str, *, first: str | None = None) -> list[str]:
    """``text`` as lines that all sit under one heading: each indented, none blank.

    As :func:`threetears.observe.text_block.under`. The text's own indentation is kept relative to its
    least-indented line, so a list inside it keeps its shape. Blank lines are dropped: a blank line is
    where a reader expects the next item.

    :param text: The text, of any number of lines.
    :ptype text: str
    :param indent: What every line starts with.
    :ptype indent: str
    :param first: What the first line starts with in place of ``indent``: the item's marker and heading
        (``"- kind: "``).
    :ptype first: str | None
    :return: The lines; for a text with nothing in it, ``first`` alone (trimmed) when given, so the item is
        still shown, else ``[]``.
    :rtype: list[str]
    """
    lines = [line.rstrip() for line in textwrap.dedent(text.expandtabs(4)).splitlines() if line.strip()]
    if not lines:
        return [first.rstrip()] if first is not None else []
    return [((first if first is not None and i == 0 else indent) + line) for i, line in enumerate(lines)]


def one_line(text: str) -> str:
    """``text`` on one line: its lines joined by one space, each trimmed, blank ones dropped.

    As :func:`threetears.observe.text_block.one_line`. For a text read as a single line of a list, a
    heading or a bold lead: a title, a summary, an error. Spacing inside a line is kept.

    :param text: The text.
    :ptype text: str
    :return: It, on one line.
    :rtype: str
    """
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


__all__ = ["one_line", "under"]
