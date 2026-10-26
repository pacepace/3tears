"""The pieces of a report's Markdown every writer shares: a table, a chart as the table of its values, and the
escapes that keep someone else's words from becoming structure.

One owner for each, so a campaign's memo page and a page built from a report plan print a table or a chart alike,
line for line. Nothing quoted is structure: a line of someone else's words opens no block of its own (:func:`inert`),
a quoted string is a code span or a fence no run of backticks inside it can close (:func:`code_span`,
:func:`fenced`), and a table cell is one line with its pipes escaped.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from threetears.evals.analysis.report.blocks import ChartBlock, TableBlock


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


#: A line's opening that would make it a block of its own rather than a paragraph's text: a heading, a quote, a
#: list item, a code fence or an HTML block — or a whole line that is a thematic break. Its first character is
#: escaped. A table and a setext heading each need a second line, which a paragraph laid out on one never has.
_BLOCK_OPENING = re.compile(r"#|>|[-+*](?:\s|$)|`{3}|~{3}|<|([-*_])(?:\s*\1){2,}\s*$")

#: An ordered list item's opening (``1.`` or ``1)``), whose delimiter is escaped instead.
_ORDERED_OPENING = re.compile(r"(\d{1,9})([.)])(?=\s|$)")


def inert(line: str) -> str:
    """One line of someone else's words that opens no block of its own: its leading marker escaped."""
    line = line.lstrip()
    if _BLOCK_OPENING.match(line):
        return "\\" + line
    if match := _ORDERED_OPENING.match(line):
        return f"{match.group(1)}\\{match.group(2)}{line[match.end() :]}"
    return line


def paragraphs(text: str) -> list[str]:
    """An author's words as the page lays them out: paragraphs on one line each and bullet lists, nothing else.

    The page (:func:`~threetears.evals.analysis.report.html_parts.markdown_html`) renders only those, with bold and code inside them; a heading, a quote or a
    fence an author's words carry reads there as the text it is, and here as the same text, escaped.
    """
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
    lines = markdown_table(headers, [[row[column.key].text for column in block.columns] for row in block.rows])
    if block.total_rows > len(block.rows):
        lines += ["", f"_{len(block.rows)} of {block.total_rows} rows shown._"]
    return lines


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    """A GitHub-flavoured Markdown table: one line per row, each cell on one line with its pipes escaped.

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
    return [f"**{block.title}**", "", *markdown_table(headers, rows)]


__all__ = [
    "chart_lines",
    "code_span",
    "fenced",
    "inert",
    "markdown_table",
    "one_line",
    "paragraphs",
    "table_lines",
]
