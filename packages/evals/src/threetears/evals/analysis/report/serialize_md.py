"""The report as Markdown — the form an agent reads and a pull request quotes, with the page's values and order.

Every block the page draws is printed here, in the page's order: the title on line 1, the arms on line 2, the verdict,
the scoreboard; a chart is its title and the table of the values it draws, so a reader of Markdown can check every
number a picture would have shown, and a chart the page leaves out is left out here too, by the same block list. An
outcome bar whose every column is one of the scoreboard's prints nothing here: its table would repeat the scoreboard. What
the page folds (a trial's record, a finding's evidence, the engine's disclosures) is a ``<details>`` here as well,
open where the page shows it open. Deterministic: one report renders to one string.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from threetears.evals.analysis.report.model import (
    FIRST_SCREEN,
    SECTION_TITLES,
    ChartBlock,
    DisclosureBlock,
    Report,
    ReportBlock,
    TableBlock,
    TextBlock,
    TrialBlock,
)
from threetears.evals.analysis.report.verdict import cause_short
from threetears.evals.analysis.style import GLOSSARY

#: How an answer's resolution leads it.
RESOLUTION_WORDS = {"answered": "Answered", "partial": "Partly answered", "unanswerable": "Unanswerable"}


def report_markdown(report: Report) -> str:
    """Render a report as Markdown.

    Args:
        report: The report.

    Returns:
        The Markdown text, ending in a newline.
    """
    lines = [f"# {_one_line(report.title)}"]
    head = [block for block in report.blocks if block.section in FIRST_SCREEN and not _repeats_scoreboard(block)]
    for index, block in enumerate(head):
        # Line 2 is the arms, directly under the title: every other block is set off by a blank line.
        lines += [*([] if index == 0 else [""]), *_block(block)]
    for section in SECTION_TITLES:
        if section in FIRST_SCREEN or section == "glossary":
            continue
        blocks = [block for block in report.blocks if block.section == section and not _repeats_scoreboard(block)]
        if not blocks:
            continue
        lines += ["", f"## {SECTION_TITLES[section]}"]
        lines += _folded(blocks)
    if report.terms:
        lines += ["", f"## {SECTION_TITLES['glossary']}", ""]
        lines += [f"- **{term}**: {GLOSSARY[term]}" for term in report.terms]
    return "\n".join(lines) + "\n"


#: The failure causes the scoreboard gives a column of their own, as an outcome bar labels them.
_SCOREBOARD_FAILURES = frozenset(f"failed: {cause_short(cause)}" for cause in ("check", "judged"))


def _repeats_scoreboard(block: ReportBlock) -> bool:
    """Whether a block is an outcome bar whose table would only repeat the scoreboard's numbers.

    The scoreboard has a column for passed, failed by check and failed by judge, and one each for every other failure
    cause and every loss cause together; a bar with at most one of each of those last two maps onto it column for
    column. A bar that splits them further says something the scoreboard does not, and is printed.
    """
    if not isinstance(block, ChartBlock) or block.chart != "outcome_bar" or block.section != "scoreboard":
        return False
    segments = [segment for bar in block.bars for segment in bar.segments]
    other = {s.label for s in segments if s.state == "failed" and s.label not in _SCOREBOARD_FAILURES}
    lost = {s.label for s in segments if s.state == "lost"}
    return len(other) <= 1 and len(lost) <= 1


def _folded(blocks: Sequence[ReportBlock]) -> list[str]:
    """A section's blocks, each consecutive run naming one fold inside one ``<details>``."""
    lines: list[str] = []
    open_fold: str | None = None
    for block in blocks:
        if block.fold != open_fold:
            if open_fold is not None:
                lines += ["", "</details>"]
            if block.fold is not None:
                lines += ["", f"<details><summary>{block.fold}</summary>"]
            open_fold = block.fold
        lines += ["", *_block(block)]
    if open_fold is not None:
        lines += ["", "</details>"]
    return lines


def _block(block: ReportBlock) -> list[str]:
    """One block's lines."""
    if isinstance(block, TextBlock):
        return _text(block)
    if isinstance(block, TableBlock):
        if block.details:
            return [f"<details><summary>{block.title}</summary>", "", *_table(block), "", "</details>"]
        return [f"**{block.title}**", "", *_table(block)]
    if isinstance(block, ChartBlock):
        headers, rows = block.values_table()
        return [f"**{block.title}**", "", *markdown_table(headers, rows)]
    if isinstance(block, TrialBlock):
        return _trial(block)
    return _disclosure(block)


def _text(block: TextBlock) -> list[str]:
    """An author's words, laid out by what they were written as."""
    body = _one_line(block.body.strip())
    facts = {fact.name: fact.value for fact in block.facts}
    if block.role == "finding_title":
        # A finding leads with its headline, unnumbered: nothing on the page cites it by number.
        return [f"**{body}**"]
    if block.role == "answer":
        lead = RESOLUTION_WORDS.get(facts.get("Resolution", ""), "")
        return [f"**{lead}.** {body}" if lead else body]
    if block.role == "recommendation":
        disposition = facts.get("Disposition", "")
        return [f"**{disposition[:1].upper()}{disposition[1:]}:** {body}" if disposition else body]
    if block.role == "next_step":
        return [f"**Next:** {body}"]
    if block.role == "revisit_when":
        return [f"Revisit when: {body}"]
    if block.role == "caveat":
        return [f"Caveat ({facts.get('Kind', 'unkinded')}): {body}"]
    if block.role == "carried_forward":
        return [f"Carried forward: {body}"]
    return [block.body.strip()]


def _disclosure(block: DisclosureBlock) -> list[str]:
    """Something code had to say: the verdict in bold, every other line plain."""
    if block.source == "verdict":
        return [f"**{_one_line(block.text)}**"]
    return [_one_line(block.text)]


def trial_summary(block: TrialBlock) -> str:
    """A trial's one-line summary: ``T4 · case 1 · candidate · failed`` — never the start of what it asked."""
    return f"{block.trial_id} · {block.case_label} · {block.arm} · {block.state}"


def _trial(block: TrialBlock) -> list[str]:
    """One trial's record, open when the block says so: what it asked, what the judge saw and said, what failed,
    and where it is stored."""
    opened = " open" if block.open else ""
    lines = [f'<details id="{block.trial_id}"{opened}><summary>{trial_summary(block)}</summary>', ""]
    if block.cause_words:
        lines += [f"Why: {_one_line(block.cause_words)}", ""]
    if block.first_input:
        lines += ["What the case asked:", "", *_fenced(block.first_input), ""]
    if block.excerpt:
        lines += ["What the judge saw (its last part):", "", *_fenced(block.excerpt), ""]
    if block.failed_checks:
        # A name that only restates the expression ("Pass rate of <expression>") adds nothing beside it.
        lines += [
            "Failed checks:",
            *(
                f"- `{check.expression}`" if check.expression in check.name else f"- {check.name}: `{check.expression}`"
                for check in block.failed_checks
            ),
            "",
        ]
    if block.judge:
        lines += ["What the judge said:"]
        for entry in block.judge:
            score = "could not tell" if entry.score is None else f"{entry.score} ({entry.scale})"
            # A criterion is written as a sentence of its own, so one that already ends is not ended twice.
            asked = f" Asked: {_one_line(entry.criterion).rstrip('.')}." if entry.criterion else ""
            lines.append(f"- **{entry.dim}**: {score}.{asked} {_one_line(entry.reason)}".rstrip())
        lines.append("")
    if block.error_words:
        lines += ["Recorded error:", "", *_fenced(block.error_words), ""]
    lines += [f"Stored result: `{block.result_id}` (read it with result_get).", "", "</details>"]
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


def _cell(text: str) -> str:
    """Text a table cell can hold: one line, its pipes escaped."""
    return _one_line(text).replace("|", "\\|")


def _one_line(text: str) -> str:
    """Text a heading, a list item or a cell carries, on one line — a newline would end either mid-sentence."""
    return " ".join(text.split("\n")).strip()


__all__ = [
    "RESOLUTION_WORDS",
    "markdown_table",
    "report_markdown",
    "trial_summary",
]
