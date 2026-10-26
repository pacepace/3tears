"""A planned page as a document: every part a campaign page built from its report plan draws, in GENERATOR §2 order.

:func:`~threetears.evals.analysis.report.plan_page.build.plan_page` builds one; the writers
(:func:`~threetears.evals.analysis.report.plan_page.html.planned_page_html`,
:func:`~threetears.evals.analysis.report.plan_page.markdown.planned_page_markdown`) lay it out. Nothing here reads
the input or the plan: every figure is already written out, every chart already resolved.

**Who wrote what.** A field the planner wrote, its figure keys written out, is typed model prose; a line code wrote
(the confidence's words and checked lines, slot titles, the warning each answer answers, a chart's notes, the
masthead's frame) is plain text.

**Charts and tables are the report's blocks.** A chart is a :class:`~threetears.evals.analysis.report.blocks.ChartBlock`
carrying the kit's data, a table a :class:`~threetears.evals.analysis.report.blocks.TableBlock`, so each prints as its
values table, carries the kit's JSON and is laid out by the writers every page shares. A block's ``section`` is
required of it; a planned page sets it to :data:`PLANNED_SECTION` and its writers ignore it, since the page's own
order is fixed by this model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from threetears.evals.analysis.report.blocks import ChartBlock, ReportSection, TableBlock
from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.schema.prose import ModelProse

__all__ = [
    "PLANNED_SECTION",
    "PageAnswer",
    "PageArmsKey",
    "PageChart",
    "PageClause",
    "PageConfidence",
    "PageExplore",
    "PageFlag",
    "PageMasthead",
    "PageOptions",
    "PageSection",
    "PageSlot",
    "PageTerm",
    "PlannedPage",
]

#: The ``section`` every block on a planned page carries: required of a block, ignored by the planned page's writers.
PLANNED_SECTION: ReportSection = "memo"


class PageMasthead(EvalDocumentModel):
    """The page's one-line masthead (GENERATOR §2.1), its figure keys written out."""

    kind: str = Field(description="What the campaign is (`Change · control vs candidates`).")
    system: str = Field(description="The system under test.")
    counts: str = Field(description="Arms, aspects or tasks, cases × repeats; empty for none.")
    run_date: str = Field(description="The run date; empty where the input states none.")
    synthetic: bool = Field(description="Whether the data is made up for an example.")


class PageFlag(EvalDocumentModel):
    """The amber line under the lede for the report-wide disclosures no confidence clause carries (GENERATOR §2.4)."""

    title: ModelProse = Field(description="The flag in one line.")
    text: ModelProse = Field(description="What it weakens and what survives.")
    discloses: list[str] = Field(description="Each disclosure it carries, by caveat id (or `no_rule`), as code filled.")


class PageClause(EvalDocumentModel):
    """One open caveat the decision's confidence is conditional on, and the planner's clause for it."""

    caveat: str = Field(description="The caveat's id.")
    text: ModelProse = Field(description="What it is, and what would make it material.")


class PageConfidence(EvalDocumentModel):
    """How sure the decision is (GENERATOR §2.5): the form in code's words, the planner's clauses, then code's checked
    line for each caveat it proved immaterial."""

    form: Literal["high", "high_if_immaterial", "low"] = Field(description="The plan's confidence form.")
    words: str = Field(description="The form, in GENERATOR §2.5's words.")
    clauses: list[PageClause] = Field(description="Each open caveat with its clause, in the plan's order.")
    checked: list[str] = Field(description="Code's line for each caveat it proved immaterial, in the input's order.")


class PageSlot(EvalDocumentModel):
    """One decision slot: its key, its title in code's words, the planner's text, and whether it is the one set
    apart (the recommendation, or the lean)."""

    key: str = Field(description="The slot's key, as the plan names it.")
    title: str = Field(description="Its title, by mode and kind (GENERATOR §2.5).")
    text: ModelProse = Field(description="The planner's text, its keys written out.")
    highlighted: bool = Field(description="Whether it is the recommendation (decide) or the lean (tradeoff).")


class PageAnswer(EvalDocumentModel):
    """Why the recommendation stands over one warning code raised on its arm."""

    warning: str = Field(description="The warning, in code's words.")
    text: ModelProse = Field(description="The planner's answer.")


class PageOptions(EvalDocumentModel):
    """The options table (GENERATOR §2.6) and the row it sets apart."""

    table: TableBlock = Field(description="The table: a row per option, a column per family code filled.")
    highlighted: int | None = Field(description="The position of the recommended (or leaned) row; None for none.")


class PageChart(EvalDocumentModel):
    """One chart as a page draws it: the planner's title, the resolved chart, and the lines code adds under it."""

    title: ModelProse = Field(description="The planner's title, its keys written out; empty where a headline says it.")
    chart: ChartBlock | TableBlock = Field(description="The chart with every value it draws, or a table.")
    notes: list[str] = Field(description="Code's captions: a comparison not in the evidence, what bold means.")


class PageArmsKey(EvalDocumentModel):
    """The arms key (GENERATOR §2.8): each arm's full configuration once, under code's eyebrow and takeaway."""

    eyebrow: str = Field(description="Its eyebrow, in code's words.")
    takeaway: str = Field(description="Its one line, in code's words.")
    table: TableBlock = Field(description="A row per arm.")


class PageSection(EvalDocumentModel):
    """One evidence section (GENERATOR §2.9): the finding it states and the chart that shows it."""

    id: str = Field(description="The section's id, as the plan names it.")
    eyebrow: ModelProse = Field(description="The question it answers, in a few words.")
    headline: ModelProse = Field(description="The finding.")
    takeaway: ModelProse = Field(description="How to read the chart; empty for none.")
    chart: PageChart = Field(description="The chart.")
    caption: ModelProse = Field(description="The line under the chart; empty for none.")


class PageExplore(EvalDocumentModel):
    """The explore part under its divider (GENERATOR §2.10), as a static table where the answer flips."""

    title: ModelProse = Field(description="The part's title.")
    flips: TableBlock = Field(description="Each candidate's flip point and whether it holds.")
    holds: str = Field(description="Whether the call holds at every setting, in code's words.")


class PageTerm(EvalDocumentModel):
    """One Method term and its definition."""

    term: ModelProse
    text: ModelProse


class PlannedPage(EvalDocumentModel):
    """A campaign page built from its report plan, in GENERATOR §2 order: every part, already resolved.

    A part the plan drops is absent: no flag, no explore, no arms key, an empty summary, an unlisted full-results
    table.
    """

    title: str = Field(description="The campaign's name, the page's title.")
    masthead: PageMasthead
    headline: ModelProse = Field(description="The decision as an instruction with its reason, or the trade.")
    lede: ModelProse
    flag: PageFlag | None = Field(description="The flag, or None where nothing needs disclosing.")
    confidence: PageConfidence
    decision: list[PageSlot] = Field(description="The four decision slots, in the mode's order.")
    answers: list[PageAnswer] = Field(description="One per warning code raised on the recommended arm.")
    options: PageOptions
    summary: list[PageChart] = Field(description="The summary charts beside the options table.")
    arms_key: PageArmsKey | None = Field(description="The arms key, or None where the plan shows none.")
    sections: list[PageSection] = Field(description="The evidence sections, in reading order.")
    explore: PageExplore | None = Field(default=None, description="The explore part, or None.")
    full_results: list[TableBlock] = Field(default_factory=list, description="The full-results tables, in order.")
    method: list[PageTerm]
    made: str = Field(default="", description="How the page was made, in one line; empty for none.")
