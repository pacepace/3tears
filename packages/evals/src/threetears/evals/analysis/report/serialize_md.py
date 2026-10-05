"""The report as Markdown — the form an agent reads and a memo is pasted as.

Every block is rendered, chart blocks included: a chart is its title, the author's caption, its values
as drawn as a table, and its disclosures, so an agent reading Markdown can check every claim a picture
would have made. Sections follow :data:`~threetears.evals.analysis.report.model.SECTION_TITLES`; a
section with no blocks is left out. Deterministic: one report renders to one string.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from threetears.evals.analysis.report.model import (
    SECTION_TITLES,
    ChartBlock,
    DisclosureBlock,
    Fact,
    Report,
    ReportBlock,
    TableBlock,
    TextBlock,
)
from threetears.evals.analysis.report.model import finding_number, report_byline, report_title
from threetears.evals.analysis.report.words import positions
from threetears.evals.analysis.viz.intent import Cell
from threetears.evals.analysis.viz.quantities import render_cell


def report_markdown(report: Report) -> str:
    """Render a report as Markdown.

    Args:
        report: The report.

    Returns:
        The Markdown text, ending in a newline.
    """
    lines = [f"# {_one_line(report_title(report))}", "", f"_{_one_line(report_byline(report))}_"]
    for section, title in SECTION_TITLES.items():
        blocks = [block for block in report.blocks if block.section == section]
        if not blocks:
            continue
        lines += ["", f"## {title}"]
        for block in blocks:
            lines += ["", *_block(block)]
    return "\n".join(lines) + "\n"


def _block(block: ReportBlock) -> list[str]:
    """One block's lines."""
    if isinstance(block, TextBlock):
        return _text(block)
    if isinstance(block, TableBlock):
        return [f"**{block.title}** ({block.order})", "", *_table(block)]
    if isinstance(block, ChartBlock):
        return _chart(block)
    return _disclosure(block)


def _text(block: TextBlock) -> list[str]:
    """An author's words, laid out by what they were written as."""
    body = block.body.strip()
    rests = f" Rests on finding {positions(block.rests_on)}." if block.rests_on else ""
    facts = _facts(block.facts)
    if block.role == "finding_title":
        heading = f"### {finding_number(block)}. {_one_line(body)}"
        return [heading, "", facts] if facts else [heading]
    if block.role in ("answer", "decision", "next_step"):
        lead = f"**{_one_line(body)}**" if block.role != "answer" else _one_line(body)
        return [f"- {lead}" + (f" — {facts}." if facts else "") + rests]
    if block.role in ("revisit_when", "next_step_why"):
        label = "Revisit when" if block.role == "revisit_when" else "Why"
        return [f"  - {label}: {_one_line(body)}"]
    if block.role == "caveat":
        return [f"> **Caveat** ({facts}): {_one_line(body)}"]
    if block.role == "carried_forward":
        return [f"Carried forward: {_one_line(body)}"]
    return [body + rests]


def _chart(block: ChartBlock) -> list[str]:
    """A chart as Markdown: its title, the author's caption, the values as drawn, and what it must disclose."""
    if block.intent is None:
        return [f"_A {block.viz_type} chart is stored for this finding and cannot be drawn: {block.error}_"]
    intent = block.intent
    lines = [f"**Chart: {intent.title}** ({intent.type})"]
    if intent.caption.strip():
        lines += ["", _one_line(intent.caption)]
    lines += [
        "",
        *_rows(
            [column.header for column in intent.columns],
            [[row.get(column.key) for column in intent.columns] for row in intent.rows],
        ),
    ]
    notes = [line for line in (intent.footnote, *intent.disclosures) if line]
    if notes:
        lines += ["", *(f"- {_one_line(note)}" for note in notes)]
    return lines


def _disclosure(block: DisclosureBlock) -> list[str]:
    """Something code had to say, set apart from the author's words."""
    return [f"> {_one_line(block.text)}"]


def _table(block: TableBlock) -> list[str]:
    """A table block's rows, or a line saying it has none."""
    if not block.rows:
        return ["_(no rows)_"]
    lines = _rows(
        [column.header for column in block.columns],
        [[row.get(column.key) for column in block.columns] for row in block.rows],
    )
    if block.total_rows > len(block.rows):
        lines += ["", f"_{len(block.rows)} of {block.total_rows} rows shown._"]
    return lines


def _rows(headers: Sequence[str], rows: Iterable[Sequence[Cell]]) -> list[str]:
    """A GitHub-flavoured Markdown table."""
    lines = ["| " + " | ".join(_cell(header) for header in headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(_cell(render_cell(value)) for value in row) + " |" for row in rows]
    return lines


def _facts(facts: Sequence[Fact]) -> str:
    """Facts code states beside an author's words, on one line."""
    return " · ".join(f"{fact.name}: {fact.value}" for fact in facts)


def _cell(text: str) -> str:
    """Text a table cell can hold: one line, its pipes escaped."""
    return _one_line(text).replace("|", "\\|")


def _one_line(text: str) -> str:
    """Text a heading, a list item or a cell carries, on one line — a newline would end either mid-sentence."""
    return " ".join(text.split("\n")).strip()


__all__ = [
    "report_markdown",
]
