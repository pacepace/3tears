"""The report as Markdown — the form an agent reads and a pull request quotes, with the page's values and order.

Every block the page draws is printed here, in the page's order: the title on line 1, then the declared question
quoted whole where there is one and under it the subhead (the arms), else the arms on line 2; the verdict; what
warns (a broken floor's line, in bold) and each report-wide caveat once; the scoreboard; a chart is its title and the table of the values
it draws, so a reader of Markdown can check every
number a picture would have shown, and a chart the page leaves out is left out here too, by the same block list. An
outcome bar whose every column is one of the scoreboard's prints nothing here: its table would repeat the scoreboard. What
the page folds (a trial's record, a finding's evidence, the engine's disclosures) is a ``<details>`` here as well,
open where the page shows it open. Deterministic: one report renders to one string.

**Nothing quoted is structure.** What a trial carried (what the case asked, what the judge saw, a recorded error) is a
fenced block whose fence is longer than any run of backticks inside it, so a transcript's own ``## heading`` or
code fence is text, never a section of this document's outline. Every other string a trial's record quotes (why it
failed, a failed check's name and expression, the judge's criterion and reason) takes the same escape chosen by its
content: a fence for text over several lines, else a code span whose backticks no run inside it can close, so a
``</details>`` the judge wrote never closes the fold it sits in. The author's words are laid out as the page lays
them out — paragraphs and bullet lists, bold and code — with any other block a line would open (a heading, a quote,
a fence, raw HTML) escaped. A trial's summary line is escaped as the HTML it sits in. The trials are the page's:
a record for each failed or lost trial, the passed ones one table of their stored results.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from typing import TYPE_CHECKING

from threetears.evals.analysis.report.blocks import (
    FIRST_SCREEN,
    SECTION_TITLES,
    ChartBlock,
    DisclosureBlock,
    ReportBlock,
    TableBlock,
    TextBlock,
    TrialBlock,
)
from threetears.evals.analysis.report.md_parts import (
    chart_lines,
    code_span,
    fenced,
    inert,
    one_line,
    paragraphs,
    table_lines,
)
from threetears.evals.analysis.report.plan_page.markdown import planned_page_markdown
from threetears.evals.analysis.report.verdict import cause_short
from threetears.evals.analysis.style import GLOSSARY

if TYPE_CHECKING:
    # The documents import this module to write themselves, so their types are named for the checker alone.
    from threetears.evals.analysis.report.model import CampaignReport, Report

#: How an answer's resolution leads it.
RESOLUTION_WORDS = {"answered": "Answered", "partial": "Partly answered", "unanswerable": "Unanswerable"}

#: How the writer's summary of an aspect leads each of its three parts, on a campaign's page.
SUMMARY_LEADS: dict[str, str] = {"what_changed": "What changed:", "how_sure": "How sure:", "means": "What it means:"}


def report_markdown(report: Report) -> str:
    """Render a report as Markdown.

    :param report: The report.
    :ptype report: Report
    :return: The Markdown text, ending in a newline.
    :rtype: str
    """
    # The title first, then the first screen in order: what warns is under the verdict, never above the title.
    lines = [f"# {one_line(report.title)}"]
    head = [block for block in report.blocks if block.section in FIRST_SCREEN and not _repeats_scoreboard(block)]
    for index, block in enumerate(head):
        # Line 2 (the arms, or a declared question) is directly under the title: every other block is set off by a
        # blank line.
        lines += [*([] if index == 0 else [""]), *_block(block)]
    lines += _section_lines(report, depth=2)
    lines += _glossary_lines(report.terms)
    return "\n".join(lines) + "\n"


def campaign_report_markdown(report: CampaignReport) -> str:
    """Render a campaign's page as Markdown, in the page's order: the title, then the first screen (the question, the
    answer, what warns and each report-wide caveat), the sections the page reads before its folds (the findings, the
    next steps, Design & provenance), then each aspect under its own heading with what code read there, its writer's
    summary and its own page in one ``<details>`` (its title, question, broken floors and verdict line left out, and
    any disclosure the campaign states once for every aspect; its headings a level down, its trials' ids prefixed with
    the aspect's), then how the page was made, then the one glossary.

    A planned analysis's page (``plan_page`` set) is the page its report plan draws
    (:func:`~threetears.evals.analysis.report.plan_page.markdown.planned_page_markdown`), with each aspect's fold and
    the glossary as its appendix after Method.

    :param report: The campaign's page.
    :ptype report: CampaignReport
    :return: The Markdown text, ending in a newline.
    :rtype: str
    """
    if report.plan_page is not None:
        return planned_page_markdown(report.plan_page, appendix=[*_fold_lines(report), *_glossary_lines(report.terms)])
    # The title first, then the decision: the question, the answer, then what warns, as consequences.
    lines = [f"# {one_line(report.title)}"]
    head = [block for block in report.blocks if block.section in FIRST_SCREEN]
    for index, block in enumerate(head):
        lines += [*([] if index == 0 else [""]), *_block(block)]
    # What the page reads and the writer says sit before the folds; how the page was made (the methods) after them.
    before, after = report.headed_sections
    titles = report.section_titles
    for section in before:
        lines += ["", f"## {titles[section]}", *_folded([b for b in report.blocks if b.section == section])]
    lines += _fold_lines(report)
    for section in after:
        lines += ["", f"## {titles[section]}", *_folded([b for b in report.blocks if b.section == section])]
    lines += _glossary_lines(report.terms)
    return "\n".join(lines) + "\n"


def _fold_lines(report: CampaignReport) -> list[str]:
    """Each aspect under its own heading with what code read there, then its page in one ``<details>``, each opening
    on a blank line."""
    lines: list[str] = []
    for fold, slug in zip(report.aspects, report.fold_ids, strict=True):
        prefix = f"a-{slug}--"
        lines += ["", f"## {inert(one_line(fold.aspect))}", "", f"{inert(one_line(fold.shows))}.", ""]
        lines += [f"<details><summary>{html.escape(one_line(fold.aspect), quote=False)} in full</summary>"]
        lines += _folded(fold.summary, prefix=prefix)
        own = [block for block in fold.head if not _repeats_scoreboard(block)]
        for block in own:
            lines += ["", *_block(block, prefix=prefix)]
        lines += _section_lines(fold.report, depth=3, prefix=prefix)
        lines += ["", "</details>"]
    return lines


def _section_lines(report: Report, *, depth: int, prefix: str = "") -> list[str]:
    """Each section after the first screen under a heading of ``depth``, its trials' ids prefixed with ``prefix``."""
    titles = report.section_titles
    lines: list[str] = []
    for section in titles:
        if section in FIRST_SCREEN or section == "glossary":
            continue
        blocks = [block for block in report.blocks if block.section == section and not _repeats_scoreboard(block)]
        if not blocks:
            continue
        lines += ["", f"{'#' * depth} {titles[section]}"]
        lines += _folded(blocks, prefix=prefix)
    return lines


def _glossary_lines(terms: Sequence[str]) -> list[str]:
    """The glossary defining ``terms``; nothing when the page links none."""
    if not terms:
        return []
    return ["", f"## {SECTION_TITLES['glossary']}", "", *(f"- **{term}**: {GLOSSARY[term]}" for term in terms)]


#: The sources of what warns, set apart under the title: a missed bar, a broken floor, a guardrail an arm is not
#: shown safe on.
_WARNING_SOURCES = frozenset({"bar", "floor", "guardrails"})


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


def _folded(blocks: Sequence[ReportBlock], *, prefix: str = "") -> list[str]:
    """A section's blocks, each consecutive run naming one fold inside one ``<details>``; a trial's id prefixed with
    ``prefix``."""
    lines: list[str] = []
    open_fold: str | None = None
    for block in blocks:
        if block.fold != open_fold:
            if open_fold is not None:
                lines += ["", "</details>"]
            if block.fold is not None:
                lines += ["", f"<details><summary>{block.fold}</summary>"]
            open_fold = block.fold
        lines += ["", *_block(block, prefix=prefix)]
    if open_fold is not None:
        lines += ["", "</details>"]
    return lines


def _block(block: ReportBlock, *, prefix: str = "") -> list[str]:
    """One block's lines; a trial's id prefixed with ``prefix``."""
    if isinstance(block, TextBlock):
        return _text(block)
    if isinstance(block, TableBlock):
        if block.details:
            return [f"<details><summary>{block.title}</summary>", "", *table_lines(block), "", "</details>"]
        # A table in a fold of its own name is titled once, by the fold's summary.
        if block.fold == block.title:
            return table_lines(block)
        return [f"**{block.title}**", "", *table_lines(block)]
    if isinstance(block, ChartBlock):
        return chart_lines(block)
    if isinstance(block, TrialBlock):
        return _trial(block, prefix=prefix)
    return _disclosure(block)


def _text(block: TextBlock) -> list[str]:
    """An author's words, laid out by what they were written as."""
    body = one_line(block.body.strip())
    facts = {fact.name: fact.value for fact in block.facts}
    if block.role == "finding_title":
        # A finding leads with its headline, unnumbered: nothing on the page cites it by number.
        return [f"**{body}**"]
    if block.role == "answer":
        lead = RESOLUTION_WORDS.get(facts.get("Resolution", ""), "")
        return [f"**{lead}.** {body}" if lead else inert(body)]
    if block.role == "recommendation":
        disposition = facts.get("Disposition", "")
        sure = f" ({facts['Confidence']} confidence)" if "Confidence" in facts else ""
        return [f"**{disposition[:1].upper()}{disposition[1:]}:** {body}{sure}" if disposition else inert(body + sure)]
    if block.role == "override":
        over = (
            f"The memo recommends {facts['Arm']} over {facts.get('Over', 'it')}" if "Arm" in facts else "The memo's why"
        )
        return [f"**{over}:** {body}"]
    if block.role in SUMMARY_LEADS:
        return [f"**{SUMMARY_LEADS[block.role]}** {body}"]
    if block.role == "reason":
        return [f"- {inert(body)}"]  # one line: body is one_line(...) above
    if block.role == "next_step":
        return [f"**Next:** {body}"]
    if block.role == "caveat":
        return [f"Caveat ({facts.get('Kind', 'unkinded')}): {body}"]
    if block.role == "carried_forward":
        return [f"Carried forward: {body}"]
    return paragraphs(block.body)


def _disclosure(block: DisclosureBlock) -> list[str]:
    """Something code had to say: the verdict and what warns in bold, every other line plain."""
    if block.source == "verdict" or (block.section in ("header", "warnings") and block.source in _WARNING_SOURCES):
        return [f"**{one_line(block.text)}**"]
    return [inert(one_line(block.text))]


def trial_summary(block: TrialBlock) -> str:
    """A trial's one-line summary: ``T4 · case 1 · candidate · failed`` — never the start of what it asked."""
    return f"{block.trial_id} · {block.case_label} · {block.arm} · {block.state}"


def _trial(block: TrialBlock, *, prefix: str = "") -> list[str]:
    """One trial's record, open when the block says so: what it asked, what the judge saw and said, what failed,
    and where it is stored."""
    opened = " open" if block.open else ""
    summary = html.escape(one_line(trial_summary(block)), quote=False)
    lines = [f'<details id="{html.escape(prefix + block.trial_id)}"{opened}><summary>{summary}</summary>', ""]
    if block.cause_words:
        lines += [*_quoted("Why:", block.cause_words), ""]
    if block.first_input:
        lines += ["What the case asked:", "", *fenced(block.first_input), ""]
    if block.excerpt:
        lines += ["What the judge saw (its last part):", "", *fenced(block.excerpt), ""]
    if block.failed_checks:
        # A name that only restates the expression ("Pass rate of <expression>") adds nothing beside it.
        lines += [
            "Failed checks:",
            *(
                f"- {code_span(check.expression)}"
                if check.expression in check.name
                else f"- {code_span(check.name)}: {code_span(check.expression)}"
                for check in block.failed_checks
            ),
            "",
        ]
    if block.judge:
        lines += ["What the judge said:"]
        for entry in block.judge:
            score = "could not tell" if entry.score is None else f"{entry.score} ({entry.scale})"
            asked = f" Asked: {code_span(entry.criterion)}." if entry.criterion else ""
            said = f" {code_span(entry.reason)}" if entry.reason.strip() else ""
            lines.append(f"- {code_span(entry.dim)}: {score}.{asked}{said}")  # one line: asked and said are code spans
        lines.append("")
    if block.error_words:
        lines += ["Recorded error:", "", *fenced(block.error_words), ""]
    lines += [f"Stored result: `{block.result_id}` (read it with result_get).", "", "</details>"]
    return lines


def _quoted(label: str, text: str) -> list[str]:
    """``label`` and a string the trial carried, escaped by its content: fenced under it when it runs over several
    lines (:func:`fenced`), else a code span beside it (:func:`code_span`)."""
    if "\n" in text.strip():
        return [label, "", *fenced(text)]
    return [f"{label} {code_span(text)}"]


__all__ = [
    "RESOLUTION_WORDS",
    "SUMMARY_LEADS",
    "campaign_report_markdown",
    "report_markdown",
    "trial_summary",
]
