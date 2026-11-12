"""The pieces of a report's Markdown every writer shares: a table, a chart as the table of its values, and the
escapes that keep someone else's words from becoming structure.

One owner for each, so a campaign's page and a report document's page print a table or a chart alike,
line for line. Nothing quoted is structure: a line of someone else's words opens no block of its own (:func:`inert`),
a quoted string is a code span or a fence no run of backticks inside it can close (:func:`code_span`,
:func:`fenced`), and a table cell is one line with its pipes escaped.

**Nothing quoted is markup either.** A Markdown host that renders raw HTML turns ``<script>`` or ``<img onerror=...>``
anywhere in a line into an element, not only where a line begins. :func:`escaped` is the one place a field's text is
made safe: its ``<`` and the ``&`` that would begin an entity become entities, so a reader sees the same text and no
renderer can read it as a tag. Every writer of Markdown prints a field through it (a heading, a bold lead-in, a
``<summary>``, a table cell), exactly once: text already escaped is escaped again as an entity of its own, and it reads
as the entity's source. The package's own markup, a ``<details>`` or an anchor, is written around the escaped text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from threetears.evals.analysis.report.blocks import ChartBlock, TableBlock, TableCell
from threetears.evals.analysis.report.words import table_label


def code_span(text: str) -> str:
    """One line of quoted text as a code span no run of backticks inside it can close: nothing in it is structure, a
    tag or an emphasis. Padded with a space where it begins or ends with a backtick, which the span then strips."""
    text = one_line(text)
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    ticks = "`" * (longest + 1)
    pad = " " if text[:1] == "`" or text[-1:] == "`" or (text[:1] == " " and text[-1:] == " ") else ""
    return f"{ticks}{pad}{text}{pad}{ticks}"


def fenced(text: str) -> list[str]:
    """Text as a fenced block whose fence no run of backticks inside it can close, so nothing in it is structure."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return [fence, *text.splitlines(), fence]


#: What a renderer could read as markup: a ``<`` that opens a tag, a closing tag, a comment, a declaration, a processing
#: instruction or an autolink (it is followed by a letter, ``/``, ``!`` or ``?``, or ends the text, where the words
#: after it could be the package's), or an ``&`` that begins an entity (``&lt;``, ``&#60;``, ``&#x3c;``). The
#: backslashes just before one belong to the match, since Markdown would let them escape the entity written in its
#: place. A ``<`` before a space or a digit (``p < 0.001``) is text to every renderer and stays as it is.
_MARKUP = re.compile(r"(\\*)(<(?=[A-Za-z/!?]|$)|&(?=#[0-9]+;|#[xX][0-9a-fA-F]+;|[A-Za-z][A-Za-z0-9]*;))")


def escaped(text: str, *, attribute: bool = False) -> str:
    """Someone else's words with nothing in them a Markdown renderer could read as markup: every ``<`` that opens a tag
    or the like and every ``&`` that begins an entity an entity itself. A reader sees the same text; a tag, a comment, an
    autolink or an entity the words carry is text, wherever in the line it sits. A backslash just before one is an
    entity too, since Markdown would let it escape the entity's own ``&``.

    :param text: The words, as written.
    :ptype text: str
    :param attribute: Whether the words sit in a double-quoted attribute value, whose quotes are escaped as well.
    :ptype attribute: bool
    :return: The words, escaped. Escape a string once: the result escaped again reads as its own source.
    :rtype: str
    """
    text = _MARKUP.sub(lambda m: "&#92;" * len(m[1]) + ("&lt;" if m[2] == "<" else "&amp;"), text)
    return text.replace('"', "&quot;") if attribute else text


#: A line's opening that would make it a block of its own rather than a paragraph's text: a heading, a quote, a
#: list item or a code fence — or a whole line that is a thematic break. Its first character is escaped. A table and a
#: setext heading each need a second line, which a paragraph laid out on one never has. An HTML block opens on a
#: ``<``, which :func:`escaped` has already turned into an entity.
_BLOCK_OPENING = re.compile(r"#|>|[-+*](?:\s|$)|`{3}|~{3}|([-*_])(?:\s*\1){2,}\s*$")

#: An ordered list item's opening (``1.`` or ``1)``), whose delimiter is escaped instead.
_ORDERED_OPENING = re.compile(r"(\d{1,9})([.)])(?=\s|$)")


def inert(line: str) -> str:
    """One line of someone else's words that opens no block of its own and carries no markup (:func:`escaped`): its
    leading marker escaped."""
    line = escaped(line.lstrip())
    if _BLOCK_OPENING.match(line):
        return "\\" + line
    if match := _ORDERED_OPENING.match(line):
        return f"{match.group(1)}\\{match.group(2)}{line[match.end() :]}"
    return line


def paragraphs(text: str) -> list[str]:
    """An author's words laid out as paragraphs on one line each and bullet lists, nothing else: a heading, a quote or
    a fence they carry reads as the same text, escaped."""
    out: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue
        if out:
            out.append("")
        if all(line[:2] in ("- ", "* ") for line in lines):
            out += [f"- {inert(line[2:])}" for line in lines]  # one line: each is a line of paragraph.splitlines()
        else:
            out.append(inert(" ".join(lines)))
    return out


def table_lines(block: TableBlock) -> list[str]:
    """A table block's rows, or a line saying it has none."""
    if not block.rows:
        return ["_(no rows)_"]
    headers = [column.header for column in block.columns]
    lines = markdown_table(headers, [[_words(row[column.key]) for column in block.columns] for row in block.rows])
    if block.total_rows > len(block.rows):
        lines += ["", f"_{len(block.rows)} of {block.total_rows} rows shown._"]
    return lines


def _words(cell: TableCell) -> str:
    """A cell as Markdown prints it: its text as a table prints words (:func:`~threetears.evals.analysis.report.words.
    table_label`), after its arm's tag where it names an arm (``A · Spark``). A status is the text's own word; Markdown
    has no colour to add."""
    text = table_label(cell.text)
    return text if cell.tag is None else f"{cell.tag} · {text}"


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    """A GitHub-flavoured Markdown table: one line per row, each cell on one line with its pipes escaped and its text
    escaped as markup (:func:`escaped`).

    :param headers: The column headers.
    :ptype headers: Sequence[str]
    :param rows: Each row's cells as text, in header order.
    :ptype rows: Iterable[Sequence[str]]
    :return: The table's lines.
    :rtype: list[str]
    """
    return pipe_table([escaped(header) for header in headers], [[escaped(text) for text in row] for row in rows])


def pipe_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    """A table laid out in pipes for a reader that is not a Markdown renderer (the editor reads a page as this text):
    one line per row, each cell on one line with its pipes escaped, its words as given. Markdown that a renderer reads
    goes through :func:`markdown_table`, which escapes each cell as markup first.

    :param headers: The column headers.
    :ptype headers: Sequence[str]
    :param rows: Each row's cells as text, in header order.
    :ptype rows: Iterable[Sequence[str]]
    :return: The table's lines.
    :rtype: list[str]
    """
    lines = ["| " + " | ".join(_cell(header) for header in headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(_cell(text) for text in row) + " |" for row in rows]
    return lines


def _cell(text: str) -> str:
    """Text a table cell can hold: one line, its pipes escaped."""
    return one_line(text).replace("|", "\\|")


def one_line(text: str) -> str:
    """Text a heading, a list item or a cell carries, on one line — a newline would end either mid-sentence."""
    return " ".join(text.split("\n")).strip()


def chart_lines(block: ChartBlock) -> list[str]:
    """A chart as Markdown: its title, then the table of every value it draws, so a reader of Markdown can check
    every number the picture would have shown.

    :param block: The chart.
    :ptype block: ChartBlock
    :return: Its lines.
    :rtype: list[str]
    """
    headers, rows = block.values_table()
    return [f"**{escaped(one_line(block.title))}**", "", *markdown_table(headers, rows)]


__all__ = [
    "chart_lines",
    "code_span",
    "escaped",
    "fenced",
    "inert",
    "markdown_table",
    "one_line",
    "paragraphs",
    "pipe_table",
    "table_lines",
]
