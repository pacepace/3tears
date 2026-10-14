"""The report as Markdown — the form an agent reads and a pull request quotes, with the page's values and order.

Every block the page draws is printed here, in the page's order: the title on line 1, the basis on line 2, the
verdict, the scoreboard; a chart is its title and the table of the values it draws, so a reader of Markdown can
check every number a picture would have shown, and a chart the page leaves out (too few cases, no interval) is left
out here too, by the same block list. Deterministic: one report renders to one string.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from threetears.evals.analysis.report.model import (
    FIRST_SCREEN,
    SECTION_TITLES,
    ChartBlock,
    DisclosureBlock,
    Fact,
    Report,
    ReportBlock,
    TableBlock,
    TextBlock,
    TrialBlock,
)


def report_markdown(report: Report) -> str:
    """Render a report as Markdown.

    Args:
        report: The report.

    Returns:
        The Markdown text, ending in a newline.
    """
    lines = [f"# {_one_line(report.title)}"]
    head = [block for block in report.blocks if block.section in FIRST_SCREEN]
    for index, block in enumerate(head):
        # Line 2 is the basis, directly under the title: every other block is set off by a blank line.
        lines += [*([] if index == 0 else [""]), *_block(block)]
    for section in SECTION_TITLES:
        if section in FIRST_SCREEN:
            continue
        blocks = [block for block in report.blocks if block.section == section]
        if not blocks:
            continue
        lines += ["", f"## {SECTION_TITLES[section]}"]
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
        headers, rows = block.values_table()
        return [f"**Chart: {block.title}**", "", *markdown_table(headers, rows)]
    if isinstance(block, TrialBlock):
        return _trial(block)
    return _disclosure(block)


def _text(block: TextBlock) -> list[str]:
    """An author's words, laid out by what they were written as."""
    body = block.body.strip()
    rests = f" Rests on {', '.join(f'F{number}' for number in block.rests_on_numbers)}." if block.rests_on else ""
    facts = _facts(block.facts)
    if block.role == "finding_title":
        heading = f"### F{block.finding_number}. {_one_line(body)}"
        return [heading, "", facts] if facts else [heading]
    if block.role in ("answer", "recommendation", "next_step"):
        lead = {"answer": "", "recommendation": "**Recommendation:** ", "next_step": "**Next:** "}[block.role]
        return [f"- {lead}{_one_line(body)}" + (f" — {facts}." if facts else "") + rests]
    if block.role in ("revisit_when", "next_step_why"):
        label = "Revisit when" if block.role == "revisit_when" else "Why"
        return [f"  - {label}: {_one_line(body)}"]
    if block.role == "caveat":
        return [f"> **Caveat** ({facts}): {_one_line(body)}"]
    if block.role == "carried_forward":
        return [f"Carried forward: {_one_line(body)}"]
    return [body + rests]


def _disclosure(block: DisclosureBlock) -> list[str]:
    """Something code had to say: the first screen's lines as lines, the verdict in bold, every other one quoted."""
    if block.source == "verdict":
        return [f"**{_one_line(block.text)}**"]
    if block.section in FIRST_SCREEN:
        return [_one_line(block.text)]
    return [f"> {_one_line(block.text)}"]


def _trial(block: TrialBlock) -> list[str]:
    """One trial's drill-down, as a heading and its fields."""
    state = block.state if block.cause_words is None else f"{block.state}: {block.cause_words}"
    lines = [f"### {block.trial_id} — {_one_line(block.case_label)} · {block.arm} · repeat {block.repeat} · {state}"]
    if block.first_input:
        lines += ["", "What the case asked:", "", *_fenced(block.first_input)]
    if block.excerpt:
        lines += ["", "What the judge saw (its last part):", "", *_fenced(block.excerpt)]
    if block.failed_checks:
        lines += ["", "Failed checks:", *(f"- {check.name}: `{check.expression}`" for check in block.failed_checks)]
    if block.judge:
        lines += ["", "What the judge said:"]
        for entry in block.judge:
            score = "could not tell" if entry.score is None else f"{entry.score} ({entry.scale})"
            asked = f" Asked: {_one_line(entry.criterion)}." if entry.criterion else ""
            lines.append(f"- **{entry.dim}**: {score}.{asked} {_one_line(entry.reason)}".rstrip())
    if block.error_words:
        lines += ["", "Recorded error:", "", *_fenced(block.error_words)]
    lines += ["", f"Stored result: `{block.result_id}` (read it with result_get)."]
    return lines


def _fenced(text: str) -> list[str]:
    """Text as a fenced block whose fence no run of backticks inside it can close."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return [fence, *text.splitlines(), fence]


def _table(block: TableBlock) -> list[str]:
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

    Args:
        headers: The column headers.
        rows: Each row's cells as text, in header order.

    Returns:
        The table's lines.
    """
    lines = ["| " + " | ".join(_cell(header) for header in headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(_cell(text) for text in row) + " |" for row in rows]
    return lines


def _facts(facts: Sequence[Fact]) -> str:
    """Facts code states beside an author's words, on one line."""
    return " · ".join(f"{fact.name}: {fact.value}" if fact.name != "See" else f"See {fact.value}" for fact in facts)


def _cell(text: str) -> str:
    """Text a table cell can hold: one line, its pipes escaped."""
    return _one_line(text).replace("|", "\\|")


def _one_line(text: str) -> str:
    """Text a heading, a list item or a cell carries, on one line — a newline would end either mid-sentence."""
    return " ".join(text.split("\n")).strip()


__all__ = [
    "markdown_table",
    "report_markdown",
]
